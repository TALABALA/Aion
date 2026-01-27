"""
AION Consensus Module

Production-grade Raft consensus protocol implementation for the
AION distributed computing system. Provides leader election, log
replication, and deterministic state machine application with full
safety guarantees.

Components:
- RaftConsensus: Main Raft protocol coordinator with election and replication
- LeaderElection: Election helper with PreVote protocol extension
- ReplicatedLog: Indexed append-only log with compaction support
- ConsensusStateMachine: Deterministic state machine for committed entries
"""

from aion.distributed.consensus.raft import RaftConsensus
from aion.distributed.consensus.leader import LeaderElection
from aion.distributed.consensus.log import ReplicatedLog
from aion.distributed.consensus.state_machine import ConsensusStateMachine

__all__ = [
    "RaftConsensus",
    "LeaderElection",
    "ReplicatedLog",
    "ConsensusStateMachine",
]
