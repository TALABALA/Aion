"""
Hot Module Replacement Implementation

Provides Webpack/Vite-style hot module replacement for plugins
with state preservation through dispose/accept callbacks.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent

import structlog

logger = structlog.get_logger(__name__)


class HMREventType(str, Enum):
    """Types of HMR events."""

    # Module events
    MODULE_CHANGED = "module_changed"
    MODULE_ADDED = "module_added"
    MODULE_REMOVED = "module_removed"

    # HMR lifecycle
    BEFORE_DISPOSE = "before_dispose"
    AFTER_DISPOSE = "after_dispose"
    BEFORE_ACCEPT = "before_accept"
    AFTER_ACCEPT = "after_accept"

    # Errors
    HMR_ERROR = "hmr_error"
    FULL_RELOAD_REQUIRED = "full_reload_required"


@dataclass
class HMREvent:
    """An HMR event."""

    type: HMREventType
    module_id: str
    timestamp: float = field(default_factory=time.time)
    data: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class HMRConfig:
    """Configuration for hot module replacement."""

    enabled: bool = True

    # File watching
    watch_patterns: list[str] = field(default_factory=lambda: ["*.py"])
    ignore_patterns: list[str] = field(default_factory=lambda: [
        "__pycache__", "*.pyc", ".git"
    ])

    # Debouncing
    debounce_ms: int = 100

    # State transfer
    state_transfer_timeout: float = 5.0

    # Recovery
    max_hmr_failures: int = 3
    fallback_to_full_reload: bool = True


@dataclass
class ModuleState:
    """
    State container for a module during HMR.

    Plugins implement dispose() to save state and accept() to restore it.
    """

    module_id: str
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    # Transfer metadata
    from_version: Optional[str] = None
    to_version: Optional[str] = None

    def get(self, key: str, default: Any = None) -> Any:
        """Get state value."""
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set state value."""
        self.data[key] = value

    def update(self, data: dict[str, Any]) -> None:
        """Update state with multiple values."""
        self.data.update(data)

    def clear(self) -> None:
        """Clear all state."""
        self.data.clear()


class StateTransfer:
    """
    Manages state transfer during HMR.

    Coordinates dispose/accept lifecycle for clean state migration.
    """

    def __init__(self, timeout: float = 5.0):
        self.timeout = timeout
        self._pending_states: dict[str, ModuleState] = {}
        self._dispose_handlers: dict[str, Callable[[ModuleState], None]] = {}
        self._accept_handlers: dict[str, Callable[[ModuleState], None]] = {}
        self._decline_handlers: dict[str, Callable[[], None]] = {}

    def register_dispose(
        self,
        module_id: str,
        handler: Callable[[ModuleState], None],
    ) -> None:
        """
        Register a dispose handler for state cleanup.

        The handler receives a ModuleState to store any data that
        should survive the reload.

        Example:
            def dispose(state: ModuleState):
                state.set("cache", self.cache)
                state.set("connections", self.connection_count)
            hmr.register_dispose("my-plugin", dispose)
        """
        self._dispose_handlers[module_id] = handler

    def register_accept(
        self,
        module_id: str,
        handler: Callable[[ModuleState], None],
    ) -> None:
        """
        Register an accept handler for state restoration.

        The handler receives the ModuleState from dispose() to
        restore preserved data.

        Example:
            def accept(state: ModuleState):
                self.cache = state.get("cache", {})
                self.connection_count = state.get("connections", 0)
            hmr.register_accept("my-plugin", accept)
        """
        self._accept_handlers[module_id] = handler

    def register_decline(
        self,
        module_id: str,
        handler: Callable[[], None],
    ) -> None:
        """
        Register a decline handler that prevents HMR.

        If a module declines HMR, a full reload is required.
        """
        self._decline_handlers[module_id] = handler

    async def perform_dispose(self, module_id: str, version: str) -> Optional[ModuleState]:
        """
        Perform dispose phase of HMR.

        Returns the state to transfer, or None if no handler.
        """
        handler = self._dispose_handlers.get(module_id)
        if not handler:
            return None

        state = ModuleState(
            module_id=module_id,
            from_version=version,
        )

        try:
            # Run dispose handler
            if asyncio.iscoroutinefunction(handler):
                await asyncio.wait_for(handler(state), timeout=self.timeout)
            else:
                handler(state)

            self._pending_states[module_id] = state
            return state

        except asyncio.TimeoutError:
            logger.error(
                "Dispose handler timed out",
                module_id=module_id,
            )
            return None
        except Exception as e:
            logger.error(
                "Dispose handler failed",
                module_id=module_id,
                error=str(e),
            )
            return None

    async def perform_accept(self, module_id: str, version: str) -> bool:
        """
        Perform accept phase of HMR.

        Returns True if state was successfully restored.
        """
        handler = self._accept_handlers.get(module_id)
        state = self._pending_states.pop(module_id, None)

        if not handler:
            return True  # No handler, nothing to restore

        if not state:
            logger.warning(
                "No state to restore",
                module_id=module_id,
            )
            return True

        state.to_version = version

        try:
            if asyncio.iscoroutinefunction(handler):
                await asyncio.wait_for(handler(state), timeout=self.timeout)
            else:
                handler(state)

            return True

        except asyncio.TimeoutError:
            logger.error(
                "Accept handler timed out",
                module_id=module_id,
            )
            return False
        except Exception as e:
            logger.error(
                "Accept handler failed",
                module_id=module_id,
                error=str(e),
            )
            return False

    def should_decline(self, module_id: str) -> bool:
        """Check if module has declined HMR."""
        return module_id in self._decline_handlers

    def clear_handlers(self, module_id: str) -> None:
        """Clear all handlers for a module."""
        self._dispose_handlers.pop(module_id, None)
        self._accept_handlers.pop(module_id, None)
        self._decline_handlers.pop(module_id, None)
        self._pending_states.pop(module_id, None)


class HotModuleReplacer:
    """
    Hot Module Replacement engine for plugins.

    Watches for file changes and performs hot reloads with
    state preservation.
    """

    def __init__(
        self,
        plugin_id: str,
        plugin_path: Path,
        config: Optional[HMRConfig] = None,
    ):
        self.plugin_id = plugin_id
        self.plugin_path = Path(plugin_path)
        self.config = config or HMRConfig()

        self.state_transfer = StateTransfer(
            timeout=self.config.state_transfer_timeout,
        )

        self._observer: Optional[Observer] = None
        self._module_hashes: dict[str, str] = {}
        self._failure_count = 0
        self._event_handlers: list[Callable[[HMREvent], None]] = []
        self._debounce_task: Optional[asyncio.Task] = None
        self._pending_changes: Set[str] = set()
        self._lock = asyncio.Lock()

        # Loaded module references
        self._loaded_modules: dict[str, Any] = {}

    def on_event(self, handler: Callable[[HMREvent], None]) -> None:
        """Register an event handler."""
        self._event_handlers.append(handler)

    def _emit_event(self, event: HMREvent) -> None:
        """Emit an HMR event to all handlers."""
        for handler in self._event_handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Event handler error: {e}")

    async def start(self) -> None:
        """Start watching for changes."""
        if not self.config.enabled:
            return

        # Calculate initial hashes
        self._update_hashes()

        # Start file watcher
        self._observer = Observer()
        handler = _FileChangeHandler(self)
        self._observer.schedule(
            handler,
            str(self.plugin_path),
            recursive=True,
        )
        self._observer.start()

        logger.info(
            "HMR started",
            plugin_id=self.plugin_id,
            path=str(self.plugin_path),
        )

    async def stop(self) -> None:
        """Stop watching for changes."""
        if self._observer:
            self._observer.stop()
            self._observer.join()
            self._observer = None

        if self._debounce_task:
            self._debounce_task.cancel()
            try:
                await self._debounce_task
            except asyncio.CancelledError:
                pass

        logger.info("HMR stopped", plugin_id=self.plugin_id)

    def _update_hashes(self) -> None:
        """Update file hashes for change detection."""
        for pattern in self.config.watch_patterns:
            for file_path in self.plugin_path.rglob(pattern):
                if self._should_ignore(file_path):
                    continue
                self._module_hashes[str(file_path)] = self._hash_file(file_path)

    def _hash_file(self, path: Path) -> str:
        """Calculate hash of a file."""
        try:
            return hashlib.md5(path.read_bytes()).hexdigest()
        except Exception:
            return ""

    def _should_ignore(self, path: Path) -> bool:
        """Check if path should be ignored."""
        path_str = str(path)
        for pattern in self.config.ignore_patterns:
            if pattern in path_str:
                return True
        return False

    async def _on_file_changed(self, path: str) -> None:
        """Handle file change with debouncing."""
        async with self._lock:
            self._pending_changes.add(path)

        # Debounce changes
        if self._debounce_task:
            self._debounce_task.cancel()

        self._debounce_task = asyncio.create_task(
            self._process_changes_after_debounce()
        )

    async def _process_changes_after_debounce(self) -> None:
        """Process changes after debounce period."""
        await asyncio.sleep(self.config.debounce_ms / 1000)

        async with self._lock:
            changes = self._pending_changes.copy()
            self._pending_changes.clear()

        for path in changes:
            await self._handle_change(path)

    async def _handle_change(self, path: str) -> None:
        """Handle a single file change."""
        file_path = Path(path)

        # Check if hash actually changed
        new_hash = self._hash_file(file_path)
        old_hash = self._module_hashes.get(path, "")

        if new_hash == old_hash:
            return  # No actual change

        self._module_hashes[path] = new_hash

        # Determine module ID from path
        module_id = self._path_to_module_id(file_path)

        self._emit_event(HMREvent(
            type=HMREventType.MODULE_CHANGED,
            module_id=module_id,
            data={"path": path, "hash": new_hash},
        ))

        # Check if module declined HMR
        if self.state_transfer.should_decline(module_id):
            self._emit_event(HMREvent(
                type=HMREventType.FULL_RELOAD_REQUIRED,
                module_id=module_id,
                data={"reason": "Module declined HMR"},
            ))
            return

        # Perform HMR
        try:
            await self._perform_hmr(module_id, path)
            self._failure_count = 0
        except Exception as e:
            self._failure_count += 1
            self._emit_event(HMREvent(
                type=HMREventType.HMR_ERROR,
                module_id=module_id,
                error=str(e),
            ))

            if (self.config.fallback_to_full_reload and
                    self._failure_count >= self.config.max_hmr_failures):
                self._emit_event(HMREvent(
                    type=HMREventType.FULL_RELOAD_REQUIRED,
                    module_id=module_id,
                    data={"reason": f"Too many HMR failures ({self._failure_count})"},
                ))

    async def _perform_hmr(self, module_id: str, path: str) -> None:
        """Perform hot module replacement."""
        # Get current version
        old_version = self._module_hashes.get(path, "unknown")

        # Phase 1: Dispose
        self._emit_event(HMREvent(
            type=HMREventType.BEFORE_DISPOSE,
            module_id=module_id,
        ))

        state = await self.state_transfer.perform_dispose(module_id, old_version)

        self._emit_event(HMREvent(
            type=HMREventType.AFTER_DISPOSE,
            module_id=module_id,
            data={"state_preserved": state is not None},
        ))

        # Phase 2: Reload module
        python_module = self._loaded_modules.get(module_id)
        if python_module:
            try:
                importlib.reload(python_module)
            except Exception as e:
                raise RuntimeError(f"Module reload failed: {e}")

        # Phase 3: Accept
        new_version = self._hash_file(Path(path))

        self._emit_event(HMREvent(
            type=HMREventType.BEFORE_ACCEPT,
            module_id=module_id,
        ))

        success = await self.state_transfer.perform_accept(module_id, new_version)

        self._emit_event(HMREvent(
            type=HMREventType.AFTER_ACCEPT,
            module_id=module_id,
            data={"success": success},
        ))

        if not success:
            raise RuntimeError("State restoration failed")

        logger.info(
            "HMR completed",
            plugin_id=self.plugin_id,
            module_id=module_id,
        )

    def _path_to_module_id(self, path: Path) -> str:
        """Convert file path to module ID."""
        try:
            rel_path = path.relative_to(self.plugin_path)
            # Convert path to module notation
            module_parts = list(rel_path.with_suffix("").parts)
            return ".".join(module_parts)
        except ValueError:
            return str(path)

    def register_module(self, module_id: str, module: Any) -> None:
        """Register a loaded module for HMR tracking."""
        self._loaded_modules[module_id] = module

    def unregister_module(self, module_id: str) -> None:
        """Unregister a module from HMR tracking."""
        self._loaded_modules.pop(module_id, None)
        self.state_transfer.clear_handlers(module_id)

    def get_stats(self) -> dict[str, Any]:
        """Get HMR statistics."""
        return {
            "plugin_id": self.plugin_id,
            "enabled": self.config.enabled,
            "watching": self._observer is not None and self._observer.is_alive(),
            "tracked_files": len(self._module_hashes),
            "tracked_modules": len(self._loaded_modules),
            "failure_count": self._failure_count,
        }


class _FileChangeHandler(FileSystemEventHandler):
    """Watchdog handler for file changes."""

    def __init__(self, hmr: HotModuleReplacer):
        self.hmr = hmr
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def on_modified(self, event):
        if isinstance(event, FileModifiedEvent) and not event.is_directory:
            self._schedule_change(event.src_path)

    def _schedule_change(self, path: str) -> None:
        """Schedule change handling on the event loop."""
        try:
            if self._loop is None:
                self._loop = asyncio.get_event_loop()
            asyncio.run_coroutine_threadsafe(
                self.hmr._on_file_changed(path),
                self._loop,
            )
        except RuntimeError:
            # No event loop available
            pass


class HMRManager:
    """
    Manages HMR for multiple plugins.

    Provides centralized hot reload management.
    """

    def __init__(self, config: Optional[HMRConfig] = None):
        self.config = config or HMRConfig()
        self._replacers: dict[str, HotModuleReplacer] = {}

    async def register_plugin(
        self,
        plugin_id: str,
        plugin_path: Path,
        config: Optional[HMRConfig] = None,
    ) -> HotModuleReplacer:
        """Register a plugin for HMR."""
        hmr = HotModuleReplacer(
            plugin_id=plugin_id,
            plugin_path=plugin_path,
            config=config or self.config,
        )
        self._replacers[plugin_id] = hmr
        await hmr.start()
        return hmr

    async def unregister_plugin(self, plugin_id: str) -> None:
        """Unregister a plugin from HMR."""
        hmr = self._replacers.pop(plugin_id, None)
        if hmr:
            await hmr.stop()

    def get_hmr(self, plugin_id: str) -> Optional[HotModuleReplacer]:
        """Get HMR instance for a plugin."""
        return self._replacers.get(plugin_id)

    async def shutdown(self) -> None:
        """Stop all HMR instances."""
        for hmr in self._replacers.values():
            await hmr.stop()
        self._replacers.clear()

    def get_stats(self) -> dict[str, Any]:
        """Get statistics for all HMR instances."""
        return {
            "total_plugins": len(self._replacers),
            "per_plugin": {
                pid: hmr.get_stats()
                for pid, hmr in self._replacers.items()
            },
        }
