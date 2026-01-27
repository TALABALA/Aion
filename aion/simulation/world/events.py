"""AION Simulation Events - Causal event graph and event bus.

Provides:
- EventBus: Typed, priority-ordered event dispatch with async handlers.
- CausalGraph: DAG tracking event causality for root-cause analysis,
  counterfactual reasoning, and deterministic replay ordering.
"""

from __future__ import annotations

import asyncio
import heapq
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set, Tuple

import structlog

from aion.simulation.types import EventType, SimulationEvent

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# EventBus
# ---------------------------------------------------------------------------


@dataclass(order=True)
class _HandlerEntry:
    """Priority-ordered handler wrapper."""

    priority: int
    handler: Any = field(compare=False)
    filter_fn: Any = field(compare=False, default=None)


class EventBus:
    """Typed, priority-ordered async event bus.

    Features:
    - Handlers registered per EventType with integer priority (lower = first).
    - Optional filter predicate per handler.
    - Wildcard handlers that receive all events.
    - Async and sync handler support.
    - Dead-letter tracking for events with no handlers.
    """

    def __init__(self) -> None:
        self._handlers: Dict[EventType, List[_HandlerEntry]] = defaultdict(list)
        self._wildcard_handlers: List[_HandlerEntry] = []
        self._dead_letters: List[SimulationEvent] = []
        self._event_counts: Dict[EventType, int] = defaultdict(int)

    # -- Registration --

    def on(
        self,
        event_type: EventType,
        handler: Callable,
        priority: int = 100,
        filter_fn: Optional[Callable[[SimulationEvent], bool]] = None,
    ) -> None:
        """Register a handler for a specific event type."""
        entry = _HandlerEntry(priority=priority, handler=handler, filter_fn=filter_fn)
        heap = self._handlers[event_type]
        heapq.heappush(heap, entry)

    def on_any(
        self,
        handler: Callable,
        priority: int = 100,
        filter_fn: Optional[Callable[[SimulationEvent], bool]] = None,
    ) -> None:
        """Register a wildcard handler that receives all events."""
        entry = _HandlerEntry(priority=priority, handler=handler, filter_fn=filter_fn)
        heapq.heappush(self._wildcard_handlers, entry)

    def off(self, event_type: EventType, handler: Callable) -> None:
        """Remove a handler."""
        self._handlers[event_type] = [
            h for h in self._handlers[event_type] if h.handler is not handler
        ]
        heapq.heapify(self._handlers[event_type])

    # -- Dispatch --

    async def emit(
        self,
        event: SimulationEvent,
    ) -> List[SimulationEvent]:
        """Emit an event, calling all matching handlers.

        Returns:
            List of resultant events produced by handlers.
        """
        self._event_counts[event.type] += 1
        result_events: List[SimulationEvent] = []

        # Merge typed handlers + wildcard handlers, sorted by priority
        typed = list(self._handlers.get(event.type, []))
        all_handlers = sorted(typed + list(self._wildcard_handlers))

        if not all_handlers:
            self._dead_letters.append(event)
            return result_events

        for entry in all_handlers:
            if entry.filter_fn and not entry.filter_fn(event):
                continue
            try:
                result = entry.handler(event)
                if asyncio.iscoroutine(result):
                    result = await result
                if result is not None:
                    if isinstance(result, list):
                        result_events.extend(result)
                    elif isinstance(result, SimulationEvent):
                        result_events.append(result)
            except Exception as exc:
                logger.error(
                    "event_handler_error",
                    event_type=event.type.value,
                    action=event.action,
                    error=str(exc),
                )
                event.success = False
                event.error = str(exc)

        return result_events

    # -- Introspection --

    @property
    def dead_letters(self) -> List[SimulationEvent]:
        return list(self._dead_letters)

    @property
    def event_counts(self) -> Dict[EventType, int]:
        return dict(self._event_counts)

    def clear(self) -> None:
        self._handlers.clear()
        self._wildcard_handlers.clear()
        self._dead_letters.clear()
        self._event_counts.clear()


# ---------------------------------------------------------------------------
# CausalGraph
# ---------------------------------------------------------------------------


class CausalGraph:
    """DAG tracking causal relationships between simulation events.

    Supports:
    - Adding events with parent causation links.
    - Root-cause analysis (trace back to origin).
    - Impact analysis (trace forward to all effects).
    - Causal chain extraction.
    - Critical path detection.
    """

    def __init__(self) -> None:
        self._nodes: Dict[str, SimulationEvent] = {}
        self._children: Dict[str, List[str]] = defaultdict(list)
        self._parents: Dict[str, Optional[str]] = {}
        self._roots: Set[str] = set()

    def add_event(
        self,
        event: SimulationEvent,
        caused_by: Optional[str] = None,
    ) -> None:
        """Add an event to the causal graph."""
        self._nodes[event.id] = event
        self._parents[event.id] = caused_by

        if caused_by and caused_by in self._nodes:
            self._children[caused_by].append(event.id)
            parent = self._nodes[caused_by]
            if event.id not in parent.causes:
                parent.causes.append(event.id)
            event.caused_by = caused_by
            event.causal_depth = self._nodes[caused_by].causal_depth + 1
        else:
            self._roots.add(event.id)

    def get_event(self, event_id: str) -> Optional[SimulationEvent]:
        return self._nodes.get(event_id)

    # -- Analysis --

    def root_cause(self, event_id: str) -> Optional[SimulationEvent]:
        """Trace back to the root cause of an event."""
        current = event_id
        visited: Set[str] = set()
        while current and current not in visited:
            visited.add(current)
            parent = self._parents.get(current)
            if parent is None:
                return self._nodes.get(current)
            current = parent
        return self._nodes.get(current) if current else None

    def causal_chain(self, event_id: str) -> List[SimulationEvent]:
        """Get the full causal chain from root to event."""
        chain: List[str] = []
        current: Optional[str] = event_id
        visited: Set[str] = set()
        while current and current not in visited:
            visited.add(current)
            chain.append(current)
            current = self._parents.get(current)
        chain.reverse()
        return [self._nodes[eid] for eid in chain if eid in self._nodes]

    def impact_set(self, event_id: str) -> List[SimulationEvent]:
        """Get all events caused (directly or transitively) by an event."""
        result: List[SimulationEvent] = []
        queue = list(self._children.get(event_id, []))
        visited: Set[str] = set()
        while queue:
            eid = queue.pop(0)
            if eid in visited:
                continue
            visited.add(eid)
            if eid in self._nodes:
                result.append(self._nodes[eid])
            queue.extend(self._children.get(eid, []))
        return result

    def critical_path(self) -> List[SimulationEvent]:
        """Find the longest causal chain (critical path)."""
        if not self._roots:
            return []

        longest: List[str] = []

        def _dfs(node_id: str, path: List[str]) -> None:
            nonlocal longest
            path.append(node_id)
            children = self._children.get(node_id, [])
            if not children:
                if len(path) > len(longest):
                    longest = list(path)
            else:
                for child in children:
                    _dfs(child, path)
            path.pop()

        for root in self._roots:
            _dfs(root, [])

        return [self._nodes[eid] for eid in longest if eid in self._nodes]

    def depth_stats(self) -> Dict[str, Any]:
        """Return statistics about causal depth."""
        if not self._nodes:
            return {"max_depth": 0, "avg_depth": 0.0, "total_events": 0, "root_count": 0}
        depths = [e.causal_depth for e in self._nodes.values()]
        return {
            "max_depth": max(depths),
            "avg_depth": sum(depths) / len(depths),
            "total_events": len(self._nodes),
            "root_count": len(self._roots),
        }

    @property
    def roots(self) -> List[SimulationEvent]:
        return [self._nodes[r] for r in self._roots if r in self._nodes]

    @property
    def size(self) -> int:
        return len(self._nodes)

    def clear(self) -> None:
        self._nodes.clear()
        self._children.clear()
        self._parents.clear()
        self._roots.clear()
