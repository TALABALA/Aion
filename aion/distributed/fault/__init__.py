"""
AION Fault Tolerance System

Provides failure detection, automatic recovery, failover handling,
and data replication to ensure cluster resilience.

Components:
- :class:`FailureDetector` -- Phi accrual failure detector with SWIM-style
  indirect probing for adaptive, low-false-positive failure detection.
- :class:`RecoveryManager` -- Orchestrates structured recovery plans with
  task reassignment, data re-replication, and progress tracking.
- :class:`FailoverHandler` -- Handles automatic leader failover, network
  partition detection, split-brain resolution, and fencing.
- :class:`DataReplicator` -- Maintains replication factor across shards
  with background repair and anti-entropy Merkle tree synchronisation.
"""

from __future__ import annotations

from aion.distributed.fault.detector import FailureDetector
from aion.distributed.fault.failover import FailoverHandler
from aion.distributed.fault.recovery import RecoveryManager
from aion.distributed.fault.replication import DataReplicator

__all__ = [
    "FailureDetector",
    "FailoverHandler",
    "RecoveryManager",
    "DataReplicator",
]
