"""
AION Plugin Lifecycle Management

Manages the lifecycle of plugins from discovery to shutdown.
"""

from __future__ import annotations

import asyncio
import traceback
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set, TYPE_CHECKING

import structlog

from aion.plugins.types import (
    PluginEvent,
    PluginEventData,
    PluginInfo,
    PluginState,
    ValidationResult,
)

if TYPE_CHECKING:
    from aion.plugins.manager import PluginManager

logger = structlog.get_logger(__name__)


# Valid state transitions
STATE_TRANSITIONS: Dict[PluginState, Set[PluginState]] = {
    PluginState.DISCOVERED: {PluginState.VALIDATING, PluginState.DISABLED},
    PluginState.VALIDATING: {PluginState.VALIDATED, PluginState.ERROR},
    PluginState.VALIDATED: {PluginState.LOADING, PluginState.DISABLED},
    PluginState.LOADING: {PluginState.LOADED, PluginState.ERROR},
    PluginState.LOADED: {PluginState.INITIALIZING, PluginState.UNLOADING, PluginState.ERROR},
    PluginState.INITIALIZING: {PluginState.INITIALIZED, PluginState.ERROR},
    PluginState.INITIALIZED: {PluginState.ACTIVATING, PluginState.UNLOADING},
    PluginState.ACTIVATING: {PluginState.ACTIVE, PluginState.ERROR},
    PluginState.ACTIVE: {PluginState.SUSPENDING, PluginState.STOPPING, PluginState.ERROR, PluginState.QUARANTINED},
    PluginState.SUSPENDING: {PluginState.SUSPENDED, PluginState.ERROR},
    PluginState.SUSPENDED: {PluginState.ACTIVATING, PluginState.STOPPING},
    PluginState.STOPPING: {PluginState.STOPPED, PluginState.ERROR},
    PluginState.STOPPED: {PluginState.UNLOADING, PluginState.LOADING},
    PluginState.ERROR: {PluginState.LOADING, PluginState.UNLOADING, PluginState.QUARANTINED},
    PluginState.UNLOADING: {PluginState.UNLOADED, PluginState.ERROR},
    PluginState.UNLOADED: {PluginState.LOADING, PluginState.DISCOVERED},
    PluginState.DISABLED: {PluginState.DISCOVERED, PluginState.VALIDATING},
    PluginState.QUARANTINED: {PluginState.UNLOADING, PluginState.LOADING},
}


class LifecycleError(Exception):
    """Error during lifecycle transition."""

    def __init__(
        self,
        plugin_id: str,
        from_state: PluginState,
        to_state: PluginState,
        message: str,
    ):
        self.plugin_id = plugin_id
        self.from_state = from_state
        self.to_state = to_state
        super().__init__(
            f"Lifecycle error for {plugin_id}: "
            f"Cannot transition from {from_state.value} to {to_state.value}: {message}"
        )


class LifecycleManager:
    """
    Manages plugin lifecycle state transitions.

    Features:
    - State machine with valid transitions
    - Event emission on state changes
    - Health monitoring
    - Automatic recovery
    - Graceful degradation
    """

    def __init__(self, manager: "PluginManager"):
        self.manager = manager
        self._locks: Dict[str, asyncio.Lock] = {}
        self._recovery_attempts: Dict[str, int] = {}
        self._max_recovery_attempts = 3

        # Lifecycle callbacks
        self._pre_transition: List[Callable] = []
        self._post_transition: List[Callable] = []

    def _get_lock(self, plugin_id: str) -> asyncio.Lock:
        """Get or create lock for plugin."""
        if plugin_id not in self._locks:
            self._locks[plugin_id] = asyncio.Lock()
        return self._locks[plugin_id]

    def can_transition(
        self,
        current_state: PluginState,
        target_state: PluginState,
    ) -> bool:
        """Check if state transition is valid."""
        valid_targets = STATE_TRANSITIONS.get(current_state, set())
        return target_state in valid_targets

    async def transition(
        self,
        plugin_id: str,
        target_state: PluginState,
        reason: str = "",
    ) -> bool:
        """
        Transition plugin to a new state.

        Args:
            plugin_id: Plugin identifier
            target_state: Target state
            reason: Reason for transition

        Returns:
            True if transition successful

        Raises:
            LifecycleError: If transition is invalid
        """
        lock = self._get_lock(plugin_id)

        async with lock:
            plugin_info = self.manager.registry.get(plugin_id)
            if not plugin_info:
                raise LifecycleError(
                    plugin_id,
                    PluginState.DISCOVERED,
                    target_state,
                    "Plugin not found",
                )

            current_state = plugin_info.state

            if current_state == target_state:
                return True

            if not self.can_transition(current_state, target_state):
                raise LifecycleError(
                    plugin_id,
                    current_state,
                    target_state,
                    f"Invalid transition",
                )

            # Execute pre-transition callbacks
            for callback in self._pre_transition:
                try:
                    await callback(plugin_id, current_state, target_state)
                except Exception as e:
                    logger.error(f"Pre-transition callback error: {e}")

            # Perform transition actions
            try:
                await self._execute_transition(plugin_info, target_state)
            except Exception as e:
                logger.error(f"Transition error for {plugin_id}: {e}")
                plugin_info.record_error(str(e), traceback.format_exc())
                self.manager.registry.set_state(plugin_id, PluginState.ERROR)
                await self._emit_event(plugin_id, PluginEvent.ERROR, error=str(e))
                return False

            # Update state
            old_state = plugin_info.state
            self.manager.registry.set_state(plugin_id, target_state)

            # Execute post-transition callbacks
            for callback in self._post_transition:
                try:
                    await callback(plugin_id, old_state, target_state)
                except Exception as e:
                    logger.error(f"Post-transition callback error: {e}")

            # Emit appropriate event
            await self._emit_state_event(plugin_id, target_state, reason)

            logger.info(
                f"Plugin {plugin_id} transitioned: {old_state.value} -> {target_state.value}"
            )

            return True

    async def _execute_transition(
        self,
        plugin_info: PluginInfo,
        target_state: PluginState,
    ) -> None:
        """Execute transition-specific actions."""
        instance = plugin_info.instance

        if target_state == PluginState.ACTIVE:
            if instance:
                await instance.on_activate()
            plugin_info.activated_at = datetime.now()

        elif target_state == PluginState.SUSPENDED:
            if instance:
                await instance.on_suspend()

        elif target_state == PluginState.STOPPED:
            if instance:
                await instance.shutdown()

        elif target_state == PluginState.UNLOADED:
            if instance:
                await instance.shutdown()
            plugin_info.instance = None
            plugin_info.module = None

    async def _emit_state_event(
        self,
        plugin_id: str,
        state: PluginState,
        reason: str = "",
    ) -> None:
        """Emit event for state change."""
        event_map = {
            PluginState.VALIDATED: PluginEvent.VALIDATION_PASSED,
            PluginState.LOADED: PluginEvent.LOADED,
            PluginState.INITIALIZED: PluginEvent.INITIALIZED,
            PluginState.ACTIVE: PluginEvent.ACTIVATED,
            PluginState.SUSPENDED: PluginEvent.SUSPENDED,
            PluginState.UNLOADED: PluginEvent.UNLOADED,
            PluginState.ERROR: PluginEvent.ERROR,
            PluginState.QUARANTINED: PluginEvent.QUARANTINED,
        }

        event = event_map.get(state)
        if event:
            await self._emit_event(plugin_id, event, data={"reason": reason})

    async def _emit_event(
        self,
        plugin_id: str,
        event: PluginEvent,
        data: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        """Emit a lifecycle event."""
        if hasattr(self.manager, '_emit_event'):
            await self.manager._emit_event(plugin_id, event, data, error)

    # === Lifecycle Operations ===

    async def load(self, plugin_id: str) -> bool:
        """Load a plugin through proper lifecycle."""
        plugin_info = self.manager.registry.get(plugin_id)
        if not plugin_info:
            return False

        # Transition through states
        transitions = []

        if plugin_info.state == PluginState.DISCOVERED:
            transitions = [
                PluginState.VALIDATING,
                PluginState.VALIDATED,
                PluginState.LOADING,
                PluginState.LOADED,
            ]
        elif plugin_info.state == PluginState.VALIDATED:
            transitions = [PluginState.LOADING, PluginState.LOADED]
        elif plugin_info.state in (PluginState.UNLOADED, PluginState.STOPPED):
            transitions = [PluginState.LOADING, PluginState.LOADED]

        for target_state in transitions:
            if not await self.transition(plugin_id, target_state):
                return False

        return True

    async def initialize(self, plugin_id: str, config: Dict[str, Any]) -> bool:
        """Initialize a loaded plugin."""
        plugin_info = self.manager.registry.get(plugin_id)
        if not plugin_info:
            return False

        if plugin_info.state != PluginState.LOADED:
            return False

        # Transition to initializing
        if not await self.transition(plugin_id, PluginState.INITIALIZING):
            return False

        try:
            # Actually initialize the plugin
            instance = plugin_info.instance
            if instance:
                await instance.initialize(self.manager.kernel, config)
                instance._mark_initialized()
                plugin_info.config = config
                plugin_info.initialized_at = datetime.now()

            # Transition to initialized
            return await self.transition(plugin_id, PluginState.INITIALIZED)

        except Exception as e:
            logger.error(f"Initialization error for {plugin_id}: {e}")
            plugin_info.record_error(str(e), traceback.format_exc())
            await self.transition(plugin_id, PluginState.ERROR)
            return False

    async def activate(self, plugin_id: str) -> bool:
        """Activate an initialized plugin."""
        plugin_info = self.manager.registry.get(plugin_id)
        if not plugin_info:
            return False

        if plugin_info.state == PluginState.INITIALIZED:
            if not await self.transition(plugin_id, PluginState.ACTIVATING):
                return False

        if plugin_info.state == PluginState.ACTIVATING:
            return await self.transition(plugin_id, PluginState.ACTIVE)

        if plugin_info.state == PluginState.SUSPENDED:
            return await self.transition(plugin_id, PluginState.ACTIVATING)

        return False

    async def suspend(self, plugin_id: str) -> bool:
        """Suspend an active plugin."""
        plugin_info = self.manager.registry.get(plugin_id)
        if not plugin_info:
            return False

        if plugin_info.state != PluginState.ACTIVE:
            return False

        if not await self.transition(plugin_id, PluginState.SUSPENDING):
            return False

        return await self.transition(plugin_id, PluginState.SUSPENDED)

    async def resume(self, plugin_id: str) -> bool:
        """Resume a suspended plugin."""
        plugin_info = self.manager.registry.get(plugin_id)
        if not plugin_info:
            return False

        if plugin_info.state != PluginState.SUSPENDED:
            return False

        return await self.activate(plugin_id)

    async def stop(self, plugin_id: str) -> bool:
        """Stop a plugin."""
        plugin_info = self.manager.registry.get(plugin_id)
        if not plugin_info:
            return False

        if plugin_info.state in (PluginState.ACTIVE, PluginState.SUSPENDED):
            if not await self.transition(plugin_id, PluginState.STOPPING):
                return False
            return await self.transition(plugin_id, PluginState.STOPPED)

        return False

    async def unload(self, plugin_id: str) -> bool:
        """Unload a plugin."""
        plugin_info = self.manager.registry.get(plugin_id)
        if not plugin_info:
            return False

        # Stop first if needed
        if plugin_info.state in (PluginState.ACTIVE, PluginState.SUSPENDED):
            await self.stop(plugin_id)

        if plugin_info.state in (PluginState.STOPPED, PluginState.LOADED, PluginState.ERROR):
            if not await self.transition(plugin_id, PluginState.UNLOADING):
                return False
            return await self.transition(plugin_id, PluginState.UNLOADED)

        return False

    # === Health & Recovery ===

    async def check_health(self, plugin_id: str) -> Dict[str, Any]:
        """Check plugin health."""
        plugin_info = self.manager.registry.get(plugin_id)
        if not plugin_info:
            return {"healthy": False, "error": "Plugin not found"}

        if plugin_info.state != PluginState.ACTIVE:
            return {
                "healthy": plugin_info.state not in (PluginState.ERROR, PluginState.QUARANTINED),
                "state": plugin_info.state.value,
            }

        if plugin_info.instance:
            try:
                health = await plugin_info.instance.health_check()
                return health
            except Exception as e:
                return {"healthy": False, "error": str(e)}

        return {"healthy": False, "error": "No instance"}

    async def recover(self, plugin_id: str) -> bool:
        """Attempt to recover a failed plugin."""
        plugin_info = self.manager.registry.get(plugin_id)
        if not plugin_info:
            return False

        if plugin_info.state not in (PluginState.ERROR, PluginState.QUARANTINED):
            return True  # Nothing to recover

        attempts = self._recovery_attempts.get(plugin_id, 0)
        if attempts >= self._max_recovery_attempts:
            logger.warning(
                f"Max recovery attempts reached for {plugin_id}, quarantining"
            )
            await self.transition(plugin_id, PluginState.QUARANTINED)
            return False

        self._recovery_attempts[plugin_id] = attempts + 1

        logger.info(f"Attempting recovery for {plugin_id} (attempt {attempts + 1})")

        try:
            # Try to reload
            await self.unload(plugin_id)
            await asyncio.sleep(1.0)  # Brief pause

            # Reload
            if await self.load(plugin_id):
                config = plugin_info.config
                if await self.initialize(plugin_id, config):
                    if await self.activate(plugin_id):
                        # Recovery successful
                        self._recovery_attempts[plugin_id] = 0
                        plugin_info.clear_error()
                        await self._emit_event(plugin_id, PluginEvent.RECOVERED)
                        return True

        except Exception as e:
            logger.error(f"Recovery failed for {plugin_id}: {e}")

        return False

    async def quarantine(self, plugin_id: str, reason: str = "") -> bool:
        """Quarantine a problematic plugin."""
        return await self.transition(plugin_id, PluginState.QUARANTINED, reason)

    # === Callbacks ===

    def on_pre_transition(self, callback: Callable) -> None:
        """Register pre-transition callback."""
        self._pre_transition.append(callback)

    def on_post_transition(self, callback: Callable) -> None:
        """Register post-transition callback."""
        self._post_transition.append(callback)

    # === Batch Operations ===

    async def activate_all(self, plugin_ids: List[str]) -> Dict[str, bool]:
        """Activate multiple plugins."""
        results = {}
        for plugin_id in plugin_ids:
            results[plugin_id] = await self.activate(plugin_id)
        return results

    async def suspend_all(self, plugin_ids: Optional[List[str]] = None) -> Dict[str, bool]:
        """Suspend multiple plugins."""
        if plugin_ids is None:
            plugin_ids = [p.id for p in self.manager.registry.get_active()]

        results = {}
        for plugin_id in plugin_ids:
            results[plugin_id] = await self.suspend(plugin_id)
        return results

    async def shutdown_all(self) -> Dict[str, bool]:
        """Shutdown all plugins gracefully."""
        results = {}

        # Get all active plugins
        active = self.manager.registry.get_active()

        # Shutdown in reverse dependency order
        for plugin_info in reversed(active):
            results[plugin_info.id] = await self.unload(plugin_info.id)

        return results
