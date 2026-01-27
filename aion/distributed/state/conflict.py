"""
AION Distributed Conflict Resolver

Resolves conflicts arising from concurrent writes in the distributed state.
Supports multiple resolution strategies: Last-Write-Wins, Vector Clock
causality analysis, CRDT merge operations, and custom application hooks.
"""

from __future__ import annotations

import copy
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Deque, Dict, List, Optional, Protocol, Tuple

import structlog

from aion.distributed.types import (
    ConflictResolution,
    ReplicationEvent,
    VectorClock,
)

logger = structlog.get_logger(__name__)


# ------------------------------------------------------------------
# CRDT types for merge-based resolution
# ------------------------------------------------------------------


@dataclass
class GCounter:
    """
    Grow-only counter CRDT.

    Each node maintains its own increment count.  The merged value is
    the element-wise max across all nodes.
    """

    counts: Dict[str, int] = field(default_factory=dict)

    def increment(self, node_id: str, amount: int = 1) -> None:
        self.counts[node_id] = self.counts.get(node_id, 0) + amount

    def value(self) -> int:
        return sum(self.counts.values())

    def merge(self, other: GCounter) -> GCounter:
        merged = GCounter()
        all_keys = set(self.counts.keys()) | set(other.counts.keys())
        for key in all_keys:
            merged.counts[key] = max(
                self.counts.get(key, 0), other.counts.get(key, 0)
            )
        return merged

    def to_dict(self) -> Dict[str, int]:
        return dict(self.counts)


@dataclass
class PNCounter:
    """
    Positive-Negative counter CRDT.

    Composed of two G-Counters: one for increments and one for decrements.
    The net value is increments minus decrements.
    """

    increments: GCounter = field(default_factory=GCounter)
    decrements: GCounter = field(default_factory=GCounter)

    def increment(self, node_id: str, amount: int = 1) -> None:
        self.increments.increment(node_id, amount)

    def decrement(self, node_id: str, amount: int = 1) -> None:
        self.decrements.increment(node_id, amount)

    def value(self) -> int:
        return self.increments.value() - self.decrements.value()

    def merge(self, other: PNCounter) -> PNCounter:
        merged = PNCounter()
        merged.increments = self.increments.merge(other.increments)
        merged.decrements = self.decrements.merge(other.decrements)
        return merged


# ------------------------------------------------------------------
# Conflict audit entry
# ------------------------------------------------------------------


@dataclass
class ConflictRecord:
    """Audit log entry for a resolved conflict."""

    conflict_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    key: str = ""
    strategy_used: str = ""
    event_count: int = 0
    winning_event_id: str = ""
    losing_event_ids: List[str] = field(default_factory=list)
    resolved_at: datetime = field(default_factory=datetime.now)
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "conflict_id": self.conflict_id,
            "key": self.key,
            "strategy_used": self.strategy_used,
            "event_count": self.event_count,
            "winning_event_id": self.winning_event_id,
            "losing_event_ids": self.losing_event_ids,
            "resolved_at": self.resolved_at.isoformat(),
            "details": self.details,
        }


# ------------------------------------------------------------------
# Custom hook protocol
# ------------------------------------------------------------------


class ConflictHook(Protocol):
    """Protocol for custom conflict resolution hooks."""

    def __call__(
        self, events: List[ReplicationEvent]
    ) -> ReplicationEvent: ...


# ------------------------------------------------------------------
# ConflictResolver
# ------------------------------------------------------------------


class ConflictResolver:
    """
    Resolves conflicts in distributed state.

    Provides multiple built-in strategies and supports pluggable custom
    hooks for application-specific conflict semantics.

    Strategies:
    - LAST_WRITE_WINS: select the event with the latest timestamp
    - VECTOR_CLOCK: use causal ordering to determine a winner or
      detect true concurrency
    - CRDT_MERGE: merge values using conflict-free replicated data
      type operations (G-Counter, PN-Counter)
    - CUSTOM: delegate to a user-supplied hook

    All resolved conflicts are recorded in an audit trail for
    operational visibility and debugging.
    """

    def __init__(
        self,
        strategy: ConflictResolution = ConflictResolution.LAST_WRITE_WINS,
        *,
        custom_hook: Optional[ConflictHook] = None,
        max_audit_records: int = 10000,
    ) -> None:
        self._strategy = strategy
        self._custom_hook = custom_hook
        self._max_audit_records = max_audit_records

        # Audit trail — uses a bounded deque so records are naturally
        # evicted in FIFO order without rewriting the collection.
        # This makes the audit log effectively append-only with bounded
        # memory: old records fall off the tail when maxlen is exceeded.
        self._audit_log: Deque[ConflictRecord] = deque(maxlen=max_audit_records)

        # Per-key conflict counters for monitoring
        self._conflict_counts: Dict[str, int] = defaultdict(int)
        self._total_conflicts: int = 0
        self._total_resolved: int = 0

        logger.info(
            "conflict_resolver.init",
            strategy=strategy.value,
            has_custom_hook=custom_hook is not None,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve(self, events: List[ReplicationEvent]) -> ReplicationEvent:
        """
        Resolve a list of conflicting events, returning the winning event.

        Uses the configured strategy to determine which event should prevail.
        Falls back to LWW if a strategy is unable to produce a clear winner.
        """
        if not events:
            raise ValueError("Cannot resolve empty event list")

        if len(events) == 1:
            return events[0]

        self._total_conflicts += 1

        key = events[0].key
        self._conflict_counts[key] += 1

        logger.debug(
            "conflict_resolver.resolve",
            key=key,
            strategy=self._strategy.value,
            event_count=len(events),
        )

        if self._strategy == ConflictResolution.LAST_WRITE_WINS:
            winner = self.resolve_lww(events)
        elif self._strategy == ConflictResolution.VECTOR_CLOCK:
            winner = self.resolve_vector_clock(events)
        elif self._strategy == ConflictResolution.CRDT_MERGE:
            winner = self.resolve_crdt(events)
        elif self._strategy == ConflictResolution.CUSTOM:
            winner = self._resolve_custom(events)
        else:
            logger.warning(
                "conflict_resolver.unknown_strategy",
                strategy=self._strategy.value,
            )
            winner = self.resolve_lww(events)

        self._total_resolved += 1
        self._record_resolution(events, winner, self._strategy.value)
        return winner

    def resolve_lww(self, events: List[ReplicationEvent]) -> ReplicationEvent:
        """
        Last-Write-Wins resolution.

        Selects the event with the most recent timestamp. Ties are broken
        by event_id lexicographic order for determinism.
        """
        if not events:
            raise ValueError("Cannot resolve empty event list")

        winner = max(
            events,
            key=lambda e: (e.timestamp, e.event_id),
        )
        logger.debug(
            "conflict_resolver.lww",
            winner_id=winner.event_id,
            winner_ts=winner.timestamp.isoformat(),
        )
        return winner

    def resolve_vector_clock(
        self, events: List[ReplicationEvent]
    ) -> ReplicationEvent:
        """
        Vector clock based causality resolution.

        If one event causally dominates all others it is the winner.
        If events are truly concurrent (no causal ordering) this falls
        back to LWW among the concurrent subset.
        """
        if not events:
            raise ValueError("Cannot resolve empty event list")

        if len(events) == 1:
            return events[0]

        # Try to find a single dominator
        for candidate in events:
            dominates_all = True
            for other in events:
                if other.event_id == candidate.event_id:
                    continue
                if not candidate.vector_clock.dominates(other.vector_clock):
                    dominates_all = False
                    break
            if dominates_all:
                logger.debug(
                    "conflict_resolver.vector_clock.dominator_found",
                    winner_id=candidate.event_id,
                )
                return candidate

        # No single dominator: find concurrent events
        concurrent_events = self._find_concurrent_set(events)
        logger.debug(
            "conflict_resolver.vector_clock.concurrent",
            concurrent_count=len(concurrent_events),
        )
        # Fall back to LWW among concurrent events
        return self.resolve_lww(concurrent_events)

    def resolve_crdt(
        self, events: List[ReplicationEvent]
    ) -> ReplicationEvent:
        """
        CRDT merge resolution.

        Attempts to interpret the event values as CRDT payloads and merge
        them. Supports GCounter and PNCounter. If the value is not a
        recognised CRDT format it falls back to LWW.
        """
        if not events:
            raise ValueError("Cannot resolve empty event list")

        if len(events) == 1:
            return events[0]

        # Attempt G-Counter merge
        g_counter_result = self._try_gcounter_merge(events)
        if g_counter_result is not None:
            return g_counter_result

        # Attempt PN-Counter merge
        pn_counter_result = self._try_pncounter_merge(events)
        if pn_counter_result is not None:
            return pn_counter_result

        # Not a recognised CRDT payload, fall back to LWW
        logger.debug("conflict_resolver.crdt.fallback_to_lww")
        return self.resolve_lww(events)

    def detect_conflicts(self, events: List[ReplicationEvent]) -> bool:
        """
        Detect whether a set of events for the same key are in conflict.

        Conflict exists when two or more events have concurrent vector
        clocks (neither dominates the other).
        """
        if len(events) < 2:
            return False

        for i, a in enumerate(events):
            for b in events[i + 1 :]:
                if a.vector_clock.is_concurrent(b.vector_clock):
                    logger.debug(
                        "conflict_resolver.conflict_detected",
                        event_a=a.event_id,
                        event_b=b.event_id,
                        key=a.key,
                    )
                    return True
        return False

    # ------------------------------------------------------------------
    # Custom hook
    # ------------------------------------------------------------------

    def set_custom_hook(self, hook: ConflictHook) -> None:
        """Register a custom conflict resolution hook."""
        self._custom_hook = hook
        logger.info("conflict_resolver.custom_hook_set")

    def _resolve_custom(
        self, events: List[ReplicationEvent]
    ) -> ReplicationEvent:
        """Delegate to the registered custom hook, falling back to LWW."""
        if self._custom_hook is None:
            logger.warning("conflict_resolver.no_custom_hook_registered")
            return self.resolve_lww(events)
        try:
            result = self._custom_hook(events)
            logger.debug(
                "conflict_resolver.custom_hook.resolved",
                winner_id=result.event_id,
            )
            return result
        except Exception as exc:
            logger.error(
                "conflict_resolver.custom_hook.error",
                error=str(exc),
            )
            return self.resolve_lww(events)

    # ------------------------------------------------------------------
    # CRDT merge helpers
    # ------------------------------------------------------------------

    def _try_gcounter_merge(
        self, events: List[ReplicationEvent]
    ) -> Optional[ReplicationEvent]:
        """Attempt to merge events as G-Counter CRDT payloads."""
        counters: List[GCounter] = []
        for event in events:
            val = event.value
            if isinstance(val, dict) and val.get("_crdt_type") == "g_counter":
                gc = GCounter(counts=dict(val.get("counts", {})))
                counters.append(gc)
            else:
                return None  # Not all events are G-Counter payloads

        merged = counters[0]
        for other in counters[1:]:
            merged = merged.merge(other)

        # Build result event based on the latest event
        result = copy.deepcopy(events[-1])
        result.event_id = str(uuid.uuid4())
        result.value = {
            "_crdt_type": "g_counter",
            "counts": merged.to_dict(),
            "_merge_provenance": {
                "source_event_ids": [e.event_id for e in events],
                "source_nodes": [e.source_node for e in events],
                "merged_at": datetime.now().isoformat(),
            },
        }
        result.timestamp = datetime.now()

        # Merge all vector clocks
        merged_clock = VectorClock()
        for event in events:
            merged_clock.merge(event.vector_clock)
        result.vector_clock = merged_clock

        logger.debug(
            "conflict_resolver.gcounter_merge",
            merged_value=merged.value(),
            source_events=len(events),
        )
        return result

    def _try_pncounter_merge(
        self, events: List[ReplicationEvent]
    ) -> Optional[ReplicationEvent]:
        """Attempt to merge events as PN-Counter CRDT payloads."""
        counters: List[PNCounter] = []
        for event in events:
            val = event.value
            if isinstance(val, dict) and val.get("_crdt_type") == "pn_counter":
                pn = PNCounter(
                    increments=GCounter(counts=dict(val.get("increments", {}))),
                    decrements=GCounter(counts=dict(val.get("decrements", {}))),
                )
                counters.append(pn)
            else:
                return None

        merged = counters[0]
        for other in counters[1:]:
            merged = merged.merge(other)

        result = copy.deepcopy(events[-1])
        result.event_id = str(uuid.uuid4())
        result.value = {
            "_crdt_type": "pn_counter",
            "increments": merged.increments.to_dict(),
            "decrements": merged.decrements.to_dict(),
            "_merge_provenance": {
                "source_event_ids": [e.event_id for e in events],
                "source_nodes": [e.source_node for e in events],
                "merged_at": datetime.now().isoformat(),
            },
        }
        result.timestamp = datetime.now()

        merged_clock = VectorClock()
        for event in events:
            merged_clock.merge(event.vector_clock)
        result.vector_clock = merged_clock

        logger.debug(
            "conflict_resolver.pncounter_merge",
            merged_value=merged.value(),
            source_events=len(events),
        )
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_concurrent_set(
        self, events: List[ReplicationEvent]
    ) -> List[ReplicationEvent]:
        """
        Find the maximal set of events that are mutually concurrent.

        Removes any event that is causally dominated by another.
        """
        result: List[ReplicationEvent] = []
        for candidate in events:
            dominated = False
            for other in events:
                if other.event_id == candidate.event_id:
                    continue
                if other.vector_clock.dominates(candidate.vector_clock):
                    dominated = True
                    break
            if not dominated:
                result.append(candidate)
        return result if result else events

    def _record_resolution(
        self,
        events: List[ReplicationEvent],
        winner: ReplicationEvent,
        strategy: str,
    ) -> None:
        """Record a conflict resolution in the audit log.

        The deque's ``maxlen`` handles eviction automatically — the oldest
        record is dropped when the capacity is exceeded, so this method
        only needs to append.
        """
        record = ConflictRecord(
            key=winner.key,
            strategy_used=strategy,
            event_count=len(events),
            winning_event_id=winner.event_id,
            losing_event_ids=[
                e.event_id for e in events if e.event_id != winner.event_id
            ],
            details={
                "winning_timestamp": winner.timestamp.isoformat(),
                "winning_source": winner.source_node,
            },
        )
        self._audit_log.append(record)

        logger.debug(
            "conflict_resolver.recorded",
            conflict_id=record.conflict_id,
            key=record.key,
        )

    # ------------------------------------------------------------------
    # Monitoring
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Return conflict resolution statistics."""
        return {
            "strategy": self._strategy.value,
            "total_conflicts": self._total_conflicts,
            "total_resolved": self._total_resolved,
            "audit_log_size": len(self._audit_log),
            "top_conflict_keys": dict(
                sorted(
                    self._conflict_counts.items(),
                    key=lambda kv: kv[1],
                    reverse=True,
                )[:20]
            ),
            "has_custom_hook": self._custom_hook is not None,
        }

    def get_audit_log(
        self,
        limit: int = 100,
        key: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve recent conflict resolution audit records.

        Optionally filter by key.
        """
        records: List[ConflictRecord] = list(self._audit_log)
        if key:
            records = [r for r in records if r.key == key]
        return [r.to_dict() for r in records[-limit:]]
