"""
AION Distributed State Management

Provides state synchronization, replication, conflict resolution,
and snapshot management for the AION distributed computing system.

Modules:
- sync: Anti-entropy state synchronization across cluster nodes
- replication: Consistency-aware state change replication
- conflict: Multi-strategy conflict resolution (LWW, Vector Clock, CRDT)
- snapshot: Compressed, checksummed state snapshots for log compaction
"""

from __future__ import annotations

from aion.distributed.state.conflict import (
    ConflictRecord,
    ConflictResolver,
    GCounter,
    PNCounter,
)
from aion.distributed.state.replication import (
    AckStatus,
    PendingAck,
    ReplicationStats,
    StateReplicator,
)
from aion.distributed.state.snapshot import (
    SnapshotError,
    SnapshotManager,
)
from aion.distributed.state.sync import (
    StateSynchronizer,
    SyncDirection,
    SyncProgress,
    SyncStatus,
)

__all__ = [
    # sync
    "StateSynchronizer",
    "SyncDirection",
    "SyncProgress",
    "SyncStatus",
    # replication
    "StateReplicator",
    "AckStatus",
    "PendingAck",
    "ReplicationStats",
    # conflict
    "ConflictResolver",
    "ConflictRecord",
    "GCounter",
    "PNCounter",
    # snapshot
    "SnapshotManager",
    "SnapshotError",
]
