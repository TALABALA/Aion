"""
AION Plugin Manager

Central coordinator for all plugin operations.
"""

from __future__ import annotations

import asyncio
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Type, TYPE_CHECKING

import structlog

from aion.plugins.types import (
    PluginEvent,
    PluginEventData,
    PluginInfo,
    PluginManifest,
    PluginState,
    PluginType,
    PluginPermissions,
    ValidationResult,
)
from aion.plugins.registry import PluginRegistry
from aion.plugins.loader import PluginLoader, PluginLoadError
from aion.plugins.lifecycle import LifecycleManager
from aion.plugins.dependencies.resolver import DependencyResolver
from aion.plugins.hooks.system import HookSystem, BUILTIN_HOOKS
from aion.plugins.hooks.events import PluginEventEmitter
from aion.plugins.sandbox.runtime import SandboxRuntime
from aion.plugins.sandbox.permissions import PermissionChecker
from aion.plugins.validation import PluginValidator
from aion.plugins.interfaces.base import BasePlugin

if TYPE_CHECKING:
    from aion.core.kernel import AIONKernel

logger = structlog.get_logger(__name__)


class PluginManager:
    """
    Central plugin management system.

    Features:
    - Plugin discovery and loading
    - Lifecycle management (load/unload/activate/suspend)
    - Dependency resolution
    - Hook system for extensibility
    - Sandboxed execution
    - Hot reload support
    - Event emission
    - Configuration management
    """

    def __init__(
        self,
        kernel: Optional["AIONKernel"] = None,
        plugin_dirs: Optional[List[Path]] = None,
        enable_sandbox: bool = True,
        auto_discover: bool = True,
    ):
        self.kernel = kernel
        self.plugin_dirs = plugin_dirs or [
            Path("./plugins"),
            Path.home() / ".aion" / "plugins",
        ]
        self.enable_sandbox = enable_sandbox
        self.auto_discover = auto_discover

        # Core components
        self.registry = PluginRegistry()
        self.loader = PluginLoader(self.plugin_dirs)
        self.resolver = DependencyResolver()
        self.hooks = HookSystem()
        self.events = PluginEventEmitter()
        self.validator = PluginValidator()
        self.lifecycle = LifecycleManager(self)

        # Sandbox
        self.sandbox: Optional[SandboxRuntime] = None
        if enable_sandbox:
            self.sandbox = SandboxRuntime()

        # Event handlers
        self._event_handlers: Dict[PluginEvent, List[Callable]] = {
            event: [] for event in PluginEvent
        }

        # State
        self._initialized = False
        self._initializing = False
        self._lock = asyncio.Lock()

    # === Initialization ===

    async def initialize(self) -> None:
        """Initialize the plugin manager."""
        if self._initialized or self._initializing:
            return

        async with self._lock:
            if self._initialized:
                return

            self._initializing = True

            try:
                logger.info("Initializing Plugin Manager")

                # Initialize sandbox
                if self.sandbox:
                    await self.sandbox.initialize()

                # Create plugin directories
                for plugin_dir in self.plugin_dirs:
                    plugin_dir.mkdir(parents=True, exist_ok=True)

                # Discover plugins
                if self.auto_discover:
                    await self.discover()

                # Load enabled plugins
                await self._load_enabled_plugins()

                self._initialized = True
                logger.info(
                    f"Plugin Manager initialized with {self.registry.count()} plugins"
                )

                # Emit startup hook
                await self.hooks.dispatch("system.startup")

            finally:
                self._initializing = False

    async def shutdown(self) -> None:
        """Shutdown the plugin manager."""
        if not self._initialized:
            return

        logger.info("Shutting down Plugin Manager")

        # Emit shutdown hook
        await self.hooks.dispatch("system.shutdown")

        # Unload all plugins in reverse dependency order
        await self.lifecycle.shutdown_all()

        # Shutdown sandbox
        if self.sandbox:
            await self.sandbox.shutdown()

        # Cleanup loader
        self.loader.cleanup()

        self._initialized = False
        logger.info("Plugin Manager shutdown complete")

    # === Discovery ===

    async def discover(self, path: Optional[Path] = None) -> List[PluginManifest]:
        """
        Discover available plugins.

        Args:
            path: Optional specific path to scan

        Returns:
            List of discovered manifests
        """
        discovered = []

        if path:
            manifest = self.loader.discover_single(path)
            if manifest:
                discovered.append(manifest)
        else:
            discovered = self.loader.discover()

        # Register discovered plugins
        for manifest in discovered:
            if not self.registry.exists(manifest.id):
                plugin_info = PluginInfo(
                    manifest=manifest,
                    state=PluginState.DISCOVERED,
                    path=self.loader.get_plugin_path(manifest.id),
                    source="local",
                )
                self.registry.register(plugin_info)

                await self._emit_event(PluginEvent.DISCOVERED, manifest.id)
                logger.debug(f"Discovered plugin: {manifest.id}")

        logger.info(f"Discovered {len(discovered)} plugins")
        return discovered

    # === Loading ===

    async def load(
        self,
        plugin_id: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Load a plugin.

        Args:
            plugin_id: Plugin identifier
            config: Plugin configuration

        Returns:
            True if loaded successfully
        """
        plugin_info = self.registry.get(plugin_id)
        if not plugin_info:
            logger.error(f"Plugin not found: {plugin_id}")
            return False

        if plugin_info.state in (PluginState.LOADED, PluginState.ACTIVE):
            logger.warning(f"Plugin already loaded: {plugin_id}")
            return True

        try:
            await self._emit_event(PluginEvent.LOADING, plugin_id)

            # Validate
            validation = self.validator.validate(plugin_info.manifest)
            if not validation.valid:
                logger.error(f"Plugin validation failed: {validation.errors}")
                plugin_info.state = PluginState.ERROR
                plugin_info.last_error = "; ".join(validation.errors)
                return False

            # Check dependencies
            available = {p.id: p for p in self.registry.get_all()}
            satisfied, missing = self.resolver.check_dependencies(plugin_info, available)

            if not satisfied:
                logger.error(f"Missing dependencies for {plugin_id}: {missing}")
                plugin_info.state = PluginState.ERROR
                plugin_info.last_error = f"Missing dependencies: {missing}"
                return False

            # Load the plugin class
            start_time = datetime.now()

            try:
                plugin_class = self.loader.load_plugin(plugin_info.manifest)
            except PluginLoadError as e:
                logger.error(f"Failed to load plugin {plugin_id}: {e}")
                plugin_info.state = PluginState.ERROR
                plugin_info.last_error = str(e)
                await self._emit_event(PluginEvent.LOAD_FAILED, plugin_id, error=str(e))
                return False

            # Instantiate
            instance = plugin_class()

            # Wrap with sandbox if enabled
            if self.sandbox and plugin_info.manifest.permissions.level.value != "full":
                instance = self.sandbox.wrap_plugin(
                    instance,
                    plugin_info.manifest.permissions,
                )

            # Initialize
            merged_config = {
                **plugin_info.manifest.default_config,
                **(config or {}),
            }

            try:
                await instance.initialize(self.kernel, merged_config)
            except Exception as e:
                logger.error(f"Plugin initialization failed: {e}")
                plugin_info.state = PluginState.ERROR
                plugin_info.last_error = str(e)
                await self._emit_event(PluginEvent.INIT_FAILED, plugin_id, error=str(e))
                return False

            # Update plugin info
            plugin_info.instance = instance
            plugin_info.config = merged_config
            plugin_info.state = PluginState.LOADED
            plugin_info.loaded_at = datetime.now()
            plugin_info.load_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            plugin_info.clear_error()

            # Register hooks from plugin
            await self._register_plugin_hooks(plugin_info)

            await self._emit_event(PluginEvent.LOADED, plugin_id)
            logger.info(
                f"Loaded plugin: {plugin_id} ({plugin_info.load_time_ms:.2f}ms)"
            )

            return True

        except Exception as e:
            logger.error(f"Error loading plugin {plugin_id}: {e}")
            plugin_info.state = PluginState.ERROR
            plugin_info.last_error = str(e)
            plugin_info.error_trace = traceback.format_exc()
            await self._emit_event(PluginEvent.ERROR, plugin_id, error=str(e))
            return False

    async def unload(self, plugin_id: str) -> bool:
        """
        Unload a plugin.

        Args:
            plugin_id: Plugin identifier

        Returns:
            True if unloaded successfully
        """
        plugin_info = self.registry.get(plugin_id)
        if not plugin_info:
            return False

        if plugin_info.state == PluginState.UNLOADED:
            return True

        try:
            await self._emit_event(PluginEvent.UNLOADING, plugin_id)

            # Check dependents
            dependents = self.registry.get_dependents(plugin_id)
            if dependents:
                logger.warning(f"Unloading {plugin_id} will affect: {dependents}")
                # Unload dependents first
                for dep_id in dependents:
                    await self.unload(dep_id)

            # Deactivate if active
            if plugin_info.state == PluginState.ACTIVE:
                await self.suspend(plugin_id)

            # Shutdown instance
            if plugin_info.instance:
                try:
                    await plugin_info.instance.shutdown()
                except Exception as e:
                    logger.error(f"Error shutting down plugin {plugin_id}: {e}")

            # Unregister hooks
            self.hooks.unregister_plugin(plugin_id)

            # Update state
            plugin_info.instance = None
            plugin_info.state = PluginState.UNLOADED

            await self._emit_event(PluginEvent.UNLOADED, plugin_id)
            logger.info(f"Unloaded plugin: {plugin_id}")

            return True

        except Exception as e:
            logger.error(f"Error unloading plugin {plugin_id}: {e}")
            plugin_info.state = PluginState.ERROR
            plugin_info.last_error = str(e)
            return False

    async def reload(self, plugin_id: str) -> bool:
        """
        Hot-reload a plugin.

        Args:
            plugin_id: Plugin identifier

        Returns:
            True if reloaded successfully
        """
        plugin_info = self.registry.get(plugin_id)
        if not plugin_info:
            return False

        await self._emit_event(PluginEvent.RELOADING, plugin_id)

        # Save config
        config = plugin_info.config.copy()
        was_active = plugin_info.state == PluginState.ACTIVE

        # Unload
        if not await self.unload(plugin_id):
            await self._emit_event(PluginEvent.RELOAD_FAILED, plugin_id)
            return False

        # Clear loader cache
        self.loader.clear_cache(plugin_id)

        # Reload
        if not await self.load(plugin_id, config):
            await self._emit_event(PluginEvent.RELOAD_FAILED, plugin_id)
            return False

        # Reactivate if was active
        if was_active:
            await self.activate(plugin_id)

        plugin_info = self.registry.get(plugin_id)
        if plugin_info:
            plugin_info.reload_count += 1
            plugin_info.last_reloaded_at = datetime.now()

        await self._emit_event(PluginEvent.RELOADED, plugin_id)
        logger.info(f"Reloaded plugin: {plugin_id}")

        return True

    # === Activation ===

    async def activate(self, plugin_id: str) -> bool:
        """
        Activate a loaded plugin.

        Args:
            plugin_id: Plugin identifier

        Returns:
            True if activated
        """
        plugin_info = self.registry.get(plugin_id)
        if not plugin_info:
            return False

        if plugin_info.state == PluginState.ACTIVE:
            return True

        if plugin_info.state != PluginState.LOADED:
            # Try to load first
            if not await self.load(plugin_id):
                return False

        try:
            await self._emit_event(PluginEvent.ACTIVATING, plugin_id)

            if plugin_info.instance:
                await plugin_info.instance.on_activate()

            plugin_info.state = PluginState.ACTIVE
            plugin_info.activated_at = datetime.now()

            await self._emit_event(PluginEvent.ACTIVATED, plugin_id)
            logger.info(f"Activated plugin: {plugin_id}")

            return True

        except Exception as e:
            logger.error(f"Error activating plugin {plugin_id}: {e}")
            plugin_info.state = PluginState.ERROR
            plugin_info.last_error = str(e)
            await self._emit_event(PluginEvent.ACTIVATION_FAILED, plugin_id, error=str(e))
            return False

    async def suspend(self, plugin_id: str) -> bool:
        """
        Suspend an active plugin.

        Args:
            plugin_id: Plugin identifier

        Returns:
            True if suspended
        """
        plugin_info = self.registry.get(plugin_id)
        if not plugin_info:
            return False

        if plugin_info.state != PluginState.ACTIVE:
            return False

        try:
            if plugin_info.instance:
                await plugin_info.instance.on_suspend()

            plugin_info.state = PluginState.SUSPENDED

            await self._emit_event(PluginEvent.SUSPENDED, plugin_id)
            logger.info(f"Suspended plugin: {plugin_id}")

            return True

        except Exception as e:
            logger.error(f"Error suspending plugin {plugin_id}: {e}")
            return False

    async def resume(self, plugin_id: str) -> bool:
        """Resume a suspended plugin."""
        plugin_info = self.registry.get(plugin_id)
        if not plugin_info:
            return False

        if plugin_info.state != PluginState.SUSPENDED:
            return False

        try:
            if plugin_info.instance:
                await plugin_info.instance.on_resume()

            plugin_info.state = PluginState.ACTIVE

            await self._emit_event(PluginEvent.RESUMED, plugin_id)
            return True

        except Exception as e:
            logger.error(f"Error resuming plugin {plugin_id}: {e}")
            return False

    # === Configuration ===

    async def configure(
        self,
        plugin_id: str,
        config: Dict[str, Any],
    ) -> bool:
        """
        Update plugin configuration.

        Args:
            plugin_id: Plugin identifier
            config: New configuration

        Returns:
            True if configured successfully
        """
        plugin_info = self.registry.get(plugin_id)
        if not plugin_info or not plugin_info.instance:
            return False

        try:
            await plugin_info.instance.configure(config)
            plugin_info.config = config

            await self._emit_event(
                PluginEvent.CONFIG_CHANGED,
                plugin_id,
                data={"config": config},
            )
            return True

        except Exception as e:
            logger.error(f"Error configuring plugin {plugin_id}: {e}")
            return False

    # === Queries ===

    def get_plugin(self, plugin_id: str) -> Optional[PluginInfo]:
        """Get plugin info."""
        return self.registry.get(plugin_id)

    def get_plugins_by_type(self, plugin_type: PluginType) -> List[PluginInfo]:
        """Get all plugins of a type."""
        return self.registry.get_by_type(plugin_type)

    def get_active_plugins(self) -> List[PluginInfo]:
        """Get all active plugins."""
        return self.registry.get_active()

    def list_plugins(self) -> List[Dict[str, Any]]:
        """List all plugins as dictionaries."""
        return [p.to_dict() for p in self.registry.get_all()]

    def search_plugins(
        self,
        query: Optional[str] = None,
        plugin_type: Optional[PluginType] = None,
        tags: Optional[List[str]] = None,
        active_only: bool = False,
    ) -> List[PluginInfo]:
        """Search plugins with filters."""
        return self.registry.search(
            query=query,
            plugin_type=plugin_type,
            tags=tags,
            active_only=active_only,
        )

    # === Tool Integration ===

    def get_plugin_tools(self) -> List:
        """Get all tools from active tool plugins."""
        from aion.plugins.interfaces.tool import ToolPlugin

        tools = []
        for plugin_info in self.get_plugins_by_type(PluginType.TOOL):
            if plugin_info.state == PluginState.ACTIVE:
                instance = plugin_info.instance
                # Handle sandboxed wrapper
                if hasattr(instance, 'unwrapped'):
                    instance = instance.unwrapped
                if isinstance(instance, ToolPlugin):
                    tools.extend(instance.get_tools())
        return tools

    # === Agent Integration ===

    def get_agent_types(self) -> List[Dict[str, Any]]:
        """Get all agent types from active agent plugins."""
        from aion.plugins.interfaces.agent import AgentPlugin

        agent_types = []
        for plugin_info in self.get_plugins_by_type(PluginType.AGENT):
            if plugin_info.state == PluginState.ACTIVE:
                instance = plugin_info.instance
                if hasattr(instance, 'unwrapped'):
                    instance = instance.unwrapped
                if isinstance(instance, AgentPlugin):
                    for agent_type in instance.get_agent_types():
                        agent_types.append(agent_type.to_dict())
        return agent_types

    # === Internal Methods ===

    async def _load_enabled_plugins(self) -> None:
        """Load all enabled plugins on startup."""
        # Resolve load order
        all_plugins = self.registry.get_all()
        result = self.resolver.resolve(all_plugins)

        if not result.success:
            logger.warning(f"Dependency resolution issues: {result.to_dict()}")

        # Load in order
        for plugin_id in result.load_order:
            plugin_info = self.registry.get(plugin_id)
            if plugin_info and self._should_auto_load(plugin_info):
                if await self.load(plugin_id):
                    await self.activate(plugin_id)

    def _should_auto_load(self, plugin_info: PluginInfo) -> bool:
        """Check if plugin should auto-load."""
        # Check manifest auto_enable flag
        if plugin_info.manifest.auto_enable:
            return True

        # Check if builtin
        if plugin_info.source == "builtin":
            return True

        return False

    async def _register_plugin_hooks(self, plugin_info: PluginInfo) -> None:
        """Register hooks defined by plugin."""
        instance = plugin_info.instance
        if hasattr(instance, 'unwrapped'):
            instance = instance.unwrapped

        # Register hooks from manifest
        for hook_name in plugin_info.manifest.hooks:
            handler_name = f"hook_{hook_name.replace('.', '_')}"
            if hasattr(instance, handler_name):
                handler = getattr(instance, handler_name)
                self.hooks.register(hook_name, plugin_info.id, handler)

        # Auto-discover hook methods
        for name in dir(instance):
            if name.startswith("hook_"):
                handler = getattr(instance, name)
                if callable(handler):
                    hook_name = name[5:].replace("_", ".")
                    if not self.hooks.get_handlers(hook_name):
                        self.hooks.register(hook_name, plugin_info.id, handler)

    async def _emit_event(
        self,
        event: PluginEvent,
        plugin_id: str,
        data: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        """Emit a plugin event."""
        await self.events.emit(event, plugin_id, data, error)

        # Call registered handlers
        for handler in self._event_handlers.get(event, []):
            try:
                result = handler(PluginEventData(
                    event=event,
                    plugin_id=plugin_id,
                    data=data or {},
                    error=error,
                ))
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"Error in event handler: {e}")

    def on_event(self, event: PluginEvent, handler: Callable) -> None:
        """Register event handler."""
        self._event_handlers[event].append(handler)

    def off_event(self, event: PluginEvent, handler: Callable) -> None:
        """Unregister event handler."""
        if handler in self._event_handlers.get(event, []):
            self._event_handlers[event].remove(handler)

    # === Health & Stats ===

    async def check_health(self, plugin_id: str) -> Dict[str, Any]:
        """Check plugin health."""
        plugin_info = self.registry.get(plugin_id)
        if not plugin_info:
            return {"healthy": False, "error": "Plugin not found"}

        if not plugin_info.instance:
            return {
                "healthy": plugin_info.state not in (PluginState.ERROR, PluginState.QUARANTINED),
                "state": plugin_info.state.value,
            }

        try:
            return await plugin_info.instance.health_check()
        except Exception as e:
            return {"healthy": False, "error": str(e)}

    def get_stats(self) -> Dict[str, Any]:
        """Get plugin system statistics."""
        return {
            "total_plugins": self.registry.count(),
            "active_plugins": len(self.registry.get_active()),
            "by_type": {
                t.value: len(self.registry.get_by_type(t))
                for t in PluginType
                if self.registry.get_by_type(t)
            },
            "by_state": {
                s.value: len(self.registry.get_by_state(s))
                for s in PluginState
                if self.registry.get_by_state(s)
            },
            "hooks": {
                "defined": len(self.hooks.list_hooks()),
                "active": len(self.hooks.list_active_hooks()),
            },
            "sandbox_enabled": self.enable_sandbox,
            "initialized": self._initialized,
        }
