"""
AION Replicated Log

Manages the append-only replicated log for the Raft consensus protocol.
Provides indexed access, range queries, truncation, and log compaction
with snapshot support.

The log is the central data structure in Raft: all state changes are
first appended to the log, replicated to a majority of nodes, and
then applied to the state machine.

Features:
- In-memory storage with O(1) index lookup (adjusted for compaction offset)
- Asyncio lock for thread-safety in concurrent access
- Log compaction via snapshot (discards entries up to snapshot index)
- Optional write-ahead log (WAL) hook for durability
- Conflict detection helpers for AppendEntries processing
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import structlog

from aion.distributed.types import RaftLogEntry, SnapshotMetadata

logger = structlog.get_logger(__name__)


class ReplicatedLog:
    """
    Append-only replicated log with snapshot-based compaction.

    Entries are stored in memory with a logical index that starts at 0
    and grows monotonically. After compaction, a snapshot offset tracks
    the last compacted index so that physical array indices map correctly
    to logical Raft indices.

    Thread-safety is provided via an asyncio.Lock, ensuring safe
    concurrent access from election timers, heartbeat loops, and
    RPC handlers.
    """

    def __init__(self) -> None:
        self._log = logger.bind(component="replicated_log")

        # In-memory log storage
        self._entries: List[RaftLogEntry] = []

        # Compaction state: entries[0] corresponds to logical index (snapshot_index + 1)
        self._snapshot_index: int = -1
        self._snapshot_term: int = 0

        # Concurrency
        self._lock = asyncio.Lock()

        # Optional WAL callback: called with (entry, "append"|"truncate") for durability
        self._wal_callback: Optional[Callable[[RaftLogEntry, str], Any]] = None

        # Metrics
        self._total_appends: int = 0
        self._total_truncations: int = 0
        self._total_compactions: int = 0

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def snapshot_index(self) -> int:
        """The index of the last compacted snapshot."""
        return self._snapshot_index

    @property
    def snapshot_term(self) -> int:
        """The term of the last compacted snapshot."""
        return self._snapshot_term

    @property
    def length(self) -> int:
        """Number of entries currently held in memory (post-compaction)."""
        return len(self._entries)

    # -------------------------------------------------------------------------
    # Index Helpers
    # -------------------------------------------------------------------------

    def _physical_index(self, logical_index: int) -> int:
        """Convert a logical Raft index to a physical array index."""
        return logical_index - (self._snapshot_index + 1)

    def _logical_index(self, physical_index: int) -> int:
        """Convert a physical array index to a logical Raft index."""
        return physical_index + (self._snapshot_index + 1)

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def append(self, entry: RaftLogEntry) -> int:
        """
        Append an entry to the log.

        The entry's index is set automatically based on the current
        last log index. Returns the index assigned to the entry.

        Args:
            entry: The log entry to append.

        Returns:
            The logical index assigned to the appended entry.
        """
        async with self._lock:
            expected_index = self.get_last_index() + 1
            entry.index = expected_index
            self._entries.append(entry)
            self._total_appends += 1

            if self._wal_callback is not None:
                self._wal_callback(entry, "append")

            self._log.debug(
                "entry_appended",
                index=entry.index,
                term=entry.term,
                command=entry.command,
            )
            return entry.index

    async def append_entries(self, entries: List[RaftLogEntry]) -> int:
        """
        Append multiple entries atomically.

        Args:
            entries: Ordered list of entries to append.

        Returns:
            The index of the last appended entry, or current last index
            if the list was empty.
        """
        if not entries:
            return self.get_last_index()

        async with self._lock:
            for entry in entries:
                expected_index = self.get_last_index() + 1
                entry.index = expected_index
                self._entries.append(entry)
                self._total_appends += 1

                if self._wal_callback is not None:
                    self._wal_callback(entry, "append")

            last = self._entries[-1]
            self._log.debug(
                "entries_appended",
                count=len(entries),
                first_index=entries[0].index,
                last_index=last.index,
            )
            return last.index

    def get(self, index: int) -> Optional[RaftLogEntry]:
        """
        Get a log entry by its logical index.

        Args:
            index: The logical Raft index.

        Returns:
            The entry at that index, or None if not found or compacted.
        """
        if index <= self._snapshot_index:
            return None  # Compacted away

        phys = self._physical_index(index)
        if 0 <= phys < len(self._entries):
            return self._entries[phys]
        return None

    def get_range(self, start: int, end: int) -> List[RaftLogEntry]:
        """
        Get entries in the logical index range [start, end) (exclusive end).

        Args:
            start: Start logical index (inclusive).
            end: End logical index (exclusive).

        Returns:
            List of entries in the range. Entries before the snapshot
            offset are excluded.
        """
        effective_start = max(start, self._snapshot_index + 1)
        if effective_start >= end:
            return []

        phys_start = self._physical_index(effective_start)
        phys_end = self._physical_index(end)

        phys_start = max(0, phys_start)
        phys_end = min(len(self._entries), phys_end)

        if phys_start >= phys_end:
            return []

        return list(self._entries[phys_start:phys_end])

    def get_entries_since(self, index: int) -> List[RaftLogEntry]:
        """
        Get all entries from the given logical index onwards (inclusive).

        Args:
            index: Starting logical index (inclusive).

        Returns:
            All entries from index to the end of the log.
        """
        if index > self.get_last_index():
            return []
        return self.get_range(index, self.get_last_index() + 1)

    async def truncate_after(self, index: int) -> int:
        """
        Remove all entries after the given logical index.

        Used during AppendEntries conflict resolution: when a follower
        discovers a conflict, it truncates its log from the conflict
        point and accepts the leader's entries.

        Args:
            index: The last index to keep. All entries with index > this
                   value are removed.

        Returns:
            Number of entries removed.
        """
        async with self._lock:
            if index < self._snapshot_index:
                self._log.warning(
                    "truncate_below_snapshot",
                    index=index,
                    snapshot_index=self._snapshot_index,
                )
                return 0

            phys = self._physical_index(index)
            keep_count = phys + 1

            if keep_count < 0:
                keep_count = 0

            removed = len(self._entries) - keep_count
            if removed > 0:
                removed_entries = self._entries[keep_count:]
                self._entries = self._entries[:keep_count]
                self._total_truncations += 1

                if self._wal_callback is not None:
                    for entry in removed_entries:
                        self._wal_callback(entry, "truncate")

                self._log.info(
                    "log_truncated",
                    after_index=index,
                    removed_count=removed,
                    new_last_index=self.get_last_index(),
                )
            return removed

    def get_last_index(self) -> int:
        """
        Get the logical index of the last entry.

        Returns the snapshot index if the log is empty (all entries
        have been compacted).
        """
        if self._entries:
            return self._entries[-1].index
        return self._snapshot_index

    def get_last_term(self) -> int:
        """
        Get the term of the last entry.

        Returns the snapshot term if the log is empty.
        """
        if self._entries:
            return self._entries[-1].term
        return self._snapshot_term

    def get_term_at(self, index: int) -> int:
        """
        Get the term of the entry at a given logical index.

        Returns 0 if the index is invalid, or the snapshot term
        if the index matches the snapshot boundary.
        """
        if index == self._snapshot_index:
            return self._snapshot_term
        entry = self.get(index)
        return entry.term if entry else 0

    async def compact(self, snapshot_index: int) -> int:
        """
        Compact the log up to and including the given index.

        All entries at or before snapshot_index are discarded from
        memory. The snapshot_index and snapshot_term are updated so
        that subsequent index calculations remain correct.

        This must be called after a snapshot has been successfully
        taken and persisted.

        Args:
            snapshot_index: The last index included in the snapshot.

        Returns:
            Number of entries removed.
        """
        async with self._lock:
            if snapshot_index <= self._snapshot_index:
                self._log.debug(
                    "compact_no_op",
                    snapshot_index=snapshot_index,
                    current_snapshot_index=self._snapshot_index,
                )
                return 0

            # Find the entry to get the term
            entry = self.get(snapshot_index)
            if entry is None:
                self._log.warning(
                    "compact_index_not_found",
                    snapshot_index=snapshot_index,
                    last_index=self.get_last_index(),
                )
                return 0

            snapshot_term = entry.term

            # Compute how many entries to discard
            phys_cutoff = self._physical_index(snapshot_index) + 1
            removed = min(phys_cutoff, len(self._entries))

            if removed > 0:
                self._entries = self._entries[phys_cutoff:]

            self._snapshot_index = snapshot_index
            self._snapshot_term = snapshot_term
            self._total_compactions += 1

            self._log.info(
                "log_compacted",
                snapshot_index=snapshot_index,
                snapshot_term=snapshot_term,
                entries_removed=removed,
                entries_remaining=len(self._entries),
            )
            return removed

    async def reset_to_snapshot(
        self, snapshot_index: int, snapshot_term: int
    ) -> None:
        """
        Reset the log to a snapshot state, discarding all entries.

        Used when a follower installs a snapshot from the leader
        because it has fallen too far behind.

        Args:
            snapshot_index: The last index included in the snapshot.
            snapshot_term: The term of the last snapshot entry.
        """
        async with self._lock:
            self._entries.clear()
            self._snapshot_index = snapshot_index
            self._snapshot_term = snapshot_term

            self._log.info(
                "log_reset_to_snapshot",
                snapshot_index=snapshot_index,
                snapshot_term=snapshot_term,
            )

    def has_entry_at(self, index: int, term: int) -> bool:
        """
        Check if the log contains an entry at the given index with the given term.

        Used for AppendEntries consistency check (log matching property).

        Args:
            index: Logical index.
            term: Expected term.

        Returns:
            True if the entry exists and has the matching term.
        """
        if index == -1:
            # Empty log always matches
            return True
        if index == self._snapshot_index:
            return self._snapshot_term == term
        entry = self.get(index)
        if entry is None:
            return False
        return entry.term == term

    def find_conflict_index(self, term: int) -> int:
        """
        Find the first index of entries with the given term.

        Used for the conflict optimization in AppendEntries responses:
        when a conflict is detected, the follower reports the first
        index of the conflicting term so the leader can skip back
        to the right point.

        Args:
            term: The conflicting term to search for.

        Returns:
            The first logical index of an entry with this term,
            or the snapshot index + 1 if not found.
        """
        for entry in self._entries:
            if entry.term == term:
                return entry.index
        return self._snapshot_index + 1

    # -------------------------------------------------------------------------
    # WAL Support
    # -------------------------------------------------------------------------

    def set_wal_callback(
        self, callback: Optional[Callable[[RaftLogEntry, str], Any]]
    ) -> None:
        """
        Set an optional write-ahead log callback.

        The callback is invoked synchronously for each append or truncate
        operation, receiving the entry and the operation type.

        Args:
            callback: A callable(entry, operation) or None to disable.
        """
        self._wal_callback = callback
        self._log.info("wal_callback_configured", enabled=callback is not None)

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Get log statistics."""
        return {
            "entries_in_memory": len(self._entries),
            "snapshot_index": self._snapshot_index,
            "snapshot_term": self._snapshot_term,
            "last_index": self.get_last_index(),
            "last_term": self.get_last_term(),
            "total_appends": self._total_appends,
            "total_truncations": self._total_truncations,
            "total_compactions": self._total_compactions,
        }
