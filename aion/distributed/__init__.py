"""
AION Distributed Computing System

Enterprise-grade distributed infrastructure for running AION as a
coordinated cluster of nodes. Provides:

- **Cluster Management**: Node discovery, lifecycle, topology awareness
- **Raft Consensus**: Leader election with PreVote extension, log replication
- **Distributed Tasks**: Priority queue with dependency DAG, dead letter queue
- **State Synchronization**: Anti-entropy sync, Merkle tree comparison
- **Memory Sharding**: Consistent hashing, distributed FAISS vector search
- **Load Balancing**: Adaptive strategies, power-of-two-choices, circuit breakers
- **Fault Tolerance**: Phi accrual failure detection, automatic failover
- **Distributed Training**: Gradient sync, experience sharing, ring all-reduce

Architecture:
    Raft consensus ensures a single leader coordinates all cluster operations.
    Nodes discover each other via configurable methods (static, DNS, multicast).
    Tasks are distributed based on node capability, load, and locality.
    State is replicated with configurable consistency levels (ONE to ALL).
    Memory is sharded via consistent hashing with automatic rebalancing.
"""

from aion.distributed.types import (
    # Node types
    NodeRole,
    NodeStatus,
    NodeCapability,
    NodeInfo,
    # Cluster types
    ClusterState,
    ClusterMetrics,
    # Task types
    TaskStatus,
    TaskPriority,
    TaskType,
    DistributedTask,
    # Consistency
    ConsistencyLevel,
    ReplicationMode,
    ConflictResolution,
    ShardingStrategy,
    # Raft types
    RaftState,
    RaftLogEntry,
    RaftMessageType,
    VoteRequest,
    VoteResponse,
    AppendEntriesRequest,
    AppendEntriesResponse,
    InstallSnapshotRequest,
    InstallSnapshotResponse,
    # Utility types
    VectorClock,
    ShardInfo,
    HeartbeatMessage,
    HealthReport,
    SnapshotMetadata,
    ReplicationEvent,
    ConfigChange,
)

from aion.distributed.config import (
    DistributedConfig,
    NetworkConfig,
    RaftConfig,
    DiscoveryConfig,
    HealthConfig,
    TaskQueueConfig,
    ReplicationConfig,
    ShardingConfig,
    LoadBalancingConfig,
    FaultToleranceConfig,
    TrainingConfig,
    get_default_config,
    get_development_config,
    get_production_config,
)

__all__ = [
    # Node types
    "NodeRole",
    "NodeStatus",
    "NodeCapability",
    "NodeInfo",
    # Cluster types
    "ClusterState",
    "ClusterMetrics",
    # Task types
    "TaskStatus",
    "TaskPriority",
    "TaskType",
    "DistributedTask",
    # Consistency
    "ConsistencyLevel",
    "ReplicationMode",
    "ConflictResolution",
    "ShardingStrategy",
    # Raft types
    "RaftState",
    "RaftLogEntry",
    "RaftMessageType",
    "VoteRequest",
    "VoteResponse",
    "AppendEntriesRequest",
    "AppendEntriesResponse",
    "InstallSnapshotRequest",
    "InstallSnapshotResponse",
    # Utility types
    "VectorClock",
    "ShardInfo",
    "HeartbeatMessage",
    "HealthReport",
    "SnapshotMetadata",
    "ReplicationEvent",
    "ConfigChange",
    # Configuration
    "DistributedConfig",
    "NetworkConfig",
    "RaftConfig",
    "DiscoveryConfig",
    "HealthConfig",
    "TaskQueueConfig",
    "ReplicationConfig",
    "ShardingConfig",
    "LoadBalancingConfig",
    "FaultToleranceConfig",
    "TrainingConfig",
    "get_default_config",
    "get_development_config",
    "get_production_config",
]
