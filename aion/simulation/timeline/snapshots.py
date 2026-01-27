"""AION Snapshot Store - Efficient state snapshot storage with deduplication.

Provides:
- SnapshotStore: Manages snapshots with content-addressable deduplication,
  LRU eviction, and compression-friendly serialization.
"""

from __future__ import annotations

import copy
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Set

import structlog

from aion.simulation.types import TimelineSnapshot, WorldState

logger = structlog.get_logger(__name__)


class SnapshotStore:
    """Manages simulation state snapshots with deduplication.

    Features:
    - Content-addressable storage (identical states stored once).
    - LRU eviction when capacity exceeded.
    - Ordered access by tick.
    - Tag-based retrieval.
    - Diff-aware: can skip storing identical consecutive states.
    """

    def __init__(self, max_snapshots: int = 10_000) -> None:
        self._snapshots: OrderedDict[str, TimelineSnapshot] = OrderedDict()
        self._by_tick: Dict[int, List[str]] = {}
        self._by_fingerprint: Dict[str, str] = {}  # fingerprint -> snapshot_id
        self._max_snapshots = max_snapshots

    def store(self, snapshot: TimelineSnapshot, deduplicate: bool = True) -> str:
        """Store a snapshot, optionally deduplicating identical states.

        Returns:
            The snapshot ID (may be an existing ID if deduplicated).
        """
        fp = snapshot.compute_fingerprint()

        if deduplicate and fp in self._by_fingerprint:
            existing_id = self._by_fingerprint[fp]
            logger.debug("snapshot_deduplicated", fingerprint=fp)
            return existing_id

        # Evict if over capacity
        while len(self._snapshots) >= self._max_snapshots:
            evicted_id, _ = self._snapshots.popitem(last=False)
            self._remove_from_indices(evicted_id)

        self._snapshots[snapshot.id] = snapshot
        self._by_fingerprint[fp] = snapshot.id

        if snapshot.tick not in self._by_tick:
            self._by_tick[snapshot.tick] = []
        self._by_tick[snapshot.tick].append(snapshot.id)

        return snapshot.id

    def get(self, snapshot_id: str) -> Optional[TimelineSnapshot]:
        snapshot = self._snapshots.get(snapshot_id)
        if snapshot is not None:
            # Move to end (LRU)
            self._snapshots.move_to_end(snapshot_id)
        return snapshot

    def get_at_tick(self, tick: int) -> List[TimelineSnapshot]:
        ids = self._by_tick.get(tick, [])
        return [self._snapshots[sid] for sid in ids if sid in self._snapshots]

    def get_nearest(self, tick: int, direction: str = "before") -> Optional[TimelineSnapshot]:
        """Get the nearest snapshot to a tick.

        Args:
            tick: Target tick.
            direction: 'before' for latest snapshot <= tick,
                       'after' for earliest snapshot >= tick.
        """
        all_ticks = sorted(self._by_tick.keys())
        if not all_ticks:
            return None

        if direction == "before":
            candidates = [t for t in all_ticks if t <= tick]
            if not candidates:
                return None
            target_tick = candidates[-1]
        else:
            candidates = [t for t in all_ticks if t >= tick]
            if not candidates:
                return None
            target_tick = candidates[0]

        ids = self._by_tick.get(target_tick, [])
        if not ids:
            return None
        return self._snapshots.get(ids[-1])

    def get_range(self, start_tick: int, end_tick: int) -> List[TimelineSnapshot]:
        result: List[TimelineSnapshot] = []
        for tick in sorted(self._by_tick.keys()):
            if tick < start_tick:
                continue
            if tick > end_tick:
                break
            for sid in self._by_tick[tick]:
                if sid in self._snapshots:
                    result.append(self._snapshots[sid])
        return result

    def remove(self, snapshot_id: str) -> bool:
        if snapshot_id not in self._snapshots:
            return False
        del self._snapshots[snapshot_id]
        self._remove_from_indices(snapshot_id)
        return True

    @property
    def count(self) -> int:
        return len(self._snapshots)

    @property
    def tick_range(self) -> tuple:
        if not self._by_tick:
            return (0, 0)
        ticks = list(self._by_tick.keys())
        return (min(ticks), max(ticks))

    def all_ids(self) -> List[str]:
        return list(self._snapshots.keys())

    def clear(self) -> None:
        self._snapshots.clear()
        self._by_tick.clear()
        self._by_fingerprint.clear()

    def _remove_from_indices(self, snapshot_id: str) -> None:
        # Remove from tick index
        for tick, ids in list(self._by_tick.items()):
            if snapshot_id in ids:
                ids.remove(snapshot_id)
                if not ids:
                    del self._by_tick[tick]
        # Remove from fingerprint index
        self._by_fingerprint = {
            fp: sid for fp, sid in self._by_fingerprint.items() if sid != snapshot_id
        }
