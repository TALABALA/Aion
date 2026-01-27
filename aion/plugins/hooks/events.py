"""
AION Plugin Event Emitter

Centralized event emission for plugin lifecycle events.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set
from collections import defaultdict

import structlog

from aion.plugins.types import PluginEvent, PluginEventData

logger = structlog.get_logger(__name__)


class PluginEventEmitter:
    """
    Event emitter for plugin lifecycle events.

    Features:
    - Async event emission
    - Multiple listeners per event
    - Event history tracking
    - Event filtering
    - Wildcard subscriptions
    """

    def __init__(self, max_history: int = 1000):
        self._listeners: Dict[PluginEvent, List[Callable]] = defaultdict(list)
        self._wildcard_listeners: List[Callable] = []
        self._plugin_listeners: Dict[str, Set[PluginEvent]] = defaultdict(set)
        self._history: List[PluginEventData] = []
        self._max_history = max_history
        self._paused = False

    # === Subscription ===

    def on(
        self,
        event: PluginEvent,
        callback: Callable[[PluginEventData], Any],
        plugin_id: Optional[str] = None,
    ) -> None:
        """
        Subscribe to an event.

        Args:
            event: Event to subscribe to
            callback: Callback function
            plugin_id: Optional plugin that owns this listener
        """
        self._listeners[event].append(callback)

        if plugin_id:
            self._plugin_listeners[plugin_id].add(event)

        logger.debug(f"Subscribed to event: {event.value}")

    def on_all(self, callback: Callable[[PluginEventData], Any]) -> None:
        """Subscribe to all events."""
        self._wildcard_listeners.append(callback)

    def off(
        self,
        event: PluginEvent,
        callback: Callable[[PluginEventData], Any],
    ) -> bool:
        """
        Unsubscribe from an event.

        Args:
            event: Event to unsubscribe from
            callback: Callback to remove

        Returns:
            True if callback was removed
        """
        if event in self._listeners and callback in self._listeners[event]:
            self._listeners[event].remove(callback)
            return True
        return False

    def off_all(self, callback: Callable[[PluginEventData], Any]) -> bool:
        """Unsubscribe from all events."""
        if callback in self._wildcard_listeners:
            self._wildcard_listeners.remove(callback)
            return True
        return False

    def unsubscribe_plugin(self, plugin_id: str) -> int:
        """
        Remove all listeners for a plugin.

        Args:
            plugin_id: Plugin identifier

        Returns:
            Number of listeners removed
        """
        events = self._plugin_listeners.pop(plugin_id, set())
        count = 0

        for event in events:
            # Note: This removes ALL listeners for these events
            # In production, we'd track callbacks per plugin
            count += len(self._listeners.get(event, []))
            self._listeners[event] = []

        return count

    # === Emission ===

    async def emit(
        self,
        event: PluginEvent,
        plugin_id: str,
        data: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        trace: Optional[str] = None,
        source: str = "system",
    ) -> None:
        """
        Emit an event.

        Args:
            event: Event type
            plugin_id: Plugin the event is about
            data: Event data
            error: Error message if applicable
            trace: Stack trace if applicable
            source: Event source
        """
        if self._paused:
            return

        event_data = PluginEventData(
            event=event,
            plugin_id=plugin_id,
            timestamp=datetime.now(),
            data=data or {},
            error=error,
            trace=trace,
            source=source,
        )

        # Add to history
        self._add_to_history(event_data)

        # Call specific listeners
        for callback in self._listeners.get(event, []):
            await self._call_listener(callback, event_data)

        # Call wildcard listeners
        for callback in self._wildcard_listeners:
            await self._call_listener(callback, event_data)

    async def _call_listener(
        self,
        callback: Callable,
        event_data: PluginEventData,
    ) -> None:
        """Call a listener safely."""
        try:
            result = callback(event_data)
            if asyncio.iscoroutine(result):
                await result
        except Exception as e:
            logger.error(f"Error in event listener: {e}")

    def emit_sync(
        self,
        event: PluginEvent,
        plugin_id: str,
        data: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        """
        Emit an event synchronously (schedules async emission).

        Useful when called from sync code.
        """
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.emit(event, plugin_id, data, error))
        except RuntimeError:
            # No running loop, create one for sync emission
            asyncio.run(self.emit(event, plugin_id, data, error))

    # === History ===

    def _add_to_history(self, event_data: PluginEventData) -> None:
        """Add event to history."""
        self._history.append(event_data)

        # Trim history if needed
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

    def get_history(
        self,
        plugin_id: Optional[str] = None,
        event: Optional[PluginEvent] = None,
        limit: int = 100,
    ) -> List[PluginEventData]:
        """
        Get event history.

        Args:
            plugin_id: Filter by plugin
            event: Filter by event type
            limit: Maximum events to return

        Returns:
            List of events (newest first)
        """
        events = self._history.copy()

        if plugin_id:
            events = [e for e in events if e.plugin_id == plugin_id]

        if event:
            events = [e for e in events if e.event == event]

        return events[-limit:][::-1]

    def clear_history(self, plugin_id: Optional[str] = None) -> int:
        """
        Clear event history.

        Args:
            plugin_id: Only clear events for this plugin

        Returns:
            Number of events cleared
        """
        if plugin_id:
            original_count = len(self._history)
            self._history = [e for e in self._history if e.plugin_id != plugin_id]
            return original_count - len(self._history)
        else:
            count = len(self._history)
            self._history = []
            return count

    # === Control ===

    def pause(self) -> None:
        """Pause event emission."""
        self._paused = True

    def resume(self) -> None:
        """Resume event emission."""
        self._paused = False

    @property
    def is_paused(self) -> bool:
        """Check if emission is paused."""
        return self._paused

    # === Stats ===

    def get_stats(self) -> Dict[str, Any]:
        """Get emitter statistics."""
        event_counts: Dict[str, int] = defaultdict(int)
        for event_data in self._history:
            event_counts[event_data.event.value] += 1

        return {
            "total_events": len(self._history),
            "listeners": {
                event.value: len(callbacks)
                for event, callbacks in self._listeners.items()
                if callbacks
            },
            "wildcard_listeners": len(self._wildcard_listeners),
            "event_counts": dict(event_counts),
            "paused": self._paused,
        }

    # === Convenience Methods ===

    async def emit_discovered(self, plugin_id: str, **data) -> None:
        """Emit plugin discovered event."""
        await self.emit(PluginEvent.DISCOVERED, plugin_id, data)

    async def emit_loaded(self, plugin_id: str, **data) -> None:
        """Emit plugin loaded event."""
        await self.emit(PluginEvent.LOADED, plugin_id, data)

    async def emit_activated(self, plugin_id: str, **data) -> None:
        """Emit plugin activated event."""
        await self.emit(PluginEvent.ACTIVATED, plugin_id, data)

    async def emit_error(
        self,
        plugin_id: str,
        error: str,
        trace: Optional[str] = None,
        **data,
    ) -> None:
        """Emit plugin error event."""
        await self.emit(PluginEvent.ERROR, plugin_id, data, error, trace)

    async def emit_unloaded(self, plugin_id: str, **data) -> None:
        """Emit plugin unloaded event."""
        await self.emit(PluginEvent.UNLOADED, plugin_id, data)


# === Global Event Emitter ===

_global_emitter: Optional[PluginEventEmitter] = None


def get_event_emitter() -> PluginEventEmitter:
    """Get the global event emitter."""
    global _global_emitter
    if _global_emitter is None:
        _global_emitter = PluginEventEmitter()
    return _global_emitter


def set_event_emitter(emitter: PluginEventEmitter) -> None:
    """Set the global event emitter."""
    global _global_emitter
    _global_emitter = emitter
