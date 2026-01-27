"""
AION Distributed Computing Types

Production-grade type definitions for the distributed computing infrastructure.
Implements SOTA patterns including:
- Comprehensive node lifecycle management
- Multi-level consistency guarantees
- Advanced sharding strategies with locality awareness
- Full Raft consensus protocol types
- Conflict-free replicated data type (CRDT) support
- Vector clock based causality tracking
"""

from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, IntEnum
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple


# =============================================================================
# Node Types
# =============================================================================


class NodeRole(str, Enum):
    """Role of a node in the cluster."""
    LEADER = "leader"
    FOLLOWER = "follower"
    CANDIDATE = "candidate"
    OBSERVER = "observer"       # Read-only node, does not vote
    LEARNER = "learner"         # Catching up node, does not vote yet
    WITNESS = "witness"         # Lightweight node for quorum only


class NodeStatus(str, Enum):
    """Status of a node."""
    STARTING = "starting"
    JOINING = "joining"         # Joining the cluster
    SYNCING = "syncing"         # Synchronizing state
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    SUSPECTED = "suspected"     # Suspected failure (phi accrual)
    DRAINING = "draining"       # Preparing to leave gracefully
    LEAVING = "leaving"         # Actively leaving
    OFFLINE = "offline"
    PARTITIONED = "partitioned"  # Network partition detected


class NodeCapability(str, Enum):
    """Node capability flags."""
    COMPUTE = "compute"
    MEMORY = "memory"
    STORAGE = "storage"
    GPU = "gpu"
    TOOLS = "tools"
    AGENTS = "agents"
    PLANNING = "planning"
    TRAINING = "training"
    INFERENCE = "inference"
    VECTOR_SEARCH = "vector_search"


# =============================================================================
# Task Types
# =============================================================================


class TaskStatus(str, Enum):
    """Status of a distributed task."""
    PENDING = "pending"
    QUEUED = "queued"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"
    TIMEOUT = "timeout"
    DEAD_LETTER = "dead_letter"  # Exhausted retries


class TaskPriority(IntEnum):
    """Task priority levels (lower = higher priority)."""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4


class TaskType(str, Enum):
    """Built-in task types."""
    TOOL_EXECUTION = "tool_execution"
    MEMORY_OPERATION = "memory_operation"
    AGENT_OPERATION = "agent_operation"
    PLANNING_OPERATION = "planning_operation"
    TRAINING_STEP = "training_step"
    INFERENCE = "inference"
    STATE_SYNC = "state_sync"
    HEALTH_CHECK = "health_check"
    REBALANCE = "rebalance"
    SNAPSHOT = "snapshot"


# =============================================================================
# Consistency & Replication
# =============================================================================


class ConsistencyLevel(str, Enum):
    """Consistency levels for distributed operations."""
    ONE = "one"              # Ack from one node (fastest)
    LOCAL_QUORUM = "local_quorum"  # Majority in local datacenter
    QUORUM = "quorum"        # Ack from majority
    ALL = "all"              # Ack from all replicas (strongest)
    LOCAL = "local"          # Local node only (no replication)
    EVENTUAL = "eventual"    # Fire-and-forget with background sync


class ReplicationMode(str, Enum):
    """Data replication modes."""
    SYNC = "sync"            # Synchronous replication
    ASYNC = "async"          # Asynchronous replication
    SEMI_SYNC = "semi_sync"  # At least one sync replica


class ConflictResolution(str, Enum):
    """Conflict resolution strategies."""
    LAST_WRITE_WINS = "lww"          # Timestamp-based
    VECTOR_CLOCK = "vector_clock"     # Causality-based
    CRDT_MERGE = "crdt_merge"         # CRDT merge function
    CUSTOM = "custom"                 # Application-defined


class ShardingStrategy(str, Enum):
    """Strategies for data sharding."""
    CONSISTENT_HASH = "consistent_hash"  # Consistent hashing (default)
    RANGE = "range"                      # Range-based partitioning
    ROUND_ROBIN = "round_robin"          # Simple round-robin
    LOCALITY = "locality"                # Data locality aware
    DIRECTORY = "directory"              # Lookup-table based
    COMPOSITE = "composite"              # Multi-key strategy


# =============================================================================
# Consensus Types
# =============================================================================


class RaftMessageType(str, Enum):
    """Types of Raft protocol messages."""
    VOTE_REQUEST = "vote_request"
    VOTE_RESPONSE = "vote_response"
    APPEND_ENTRIES = "append_entries"
    APPEND_ENTRIES_RESPONSE = "append_entries_response"
    INSTALL_SNAPSHOT = "install_snapshot"
    INSTALL_SNAPSHOT_RESPONSE = "install_snapshot_response"
    PRE_VOTE_REQUEST = "pre_vote_request"
    PRE_VOTE_RESPONSE = "pre_vote_response"


# =============================================================================
# Vector Clock for Causality Tracking
# =============================================================================


@dataclass
class VectorClock:
    """
    Vector clock for causal ordering of events.

    Each node maintains a logical clock counter.
    Enables partial ordering of distributed events.
    """
    clocks: Dict[str, int] = field(default_factory=dict)

    def increment(self, node_id: str) -> None:
        """Increment this node's clock."""
        self.clocks[node_id] = self.clocks.get(node_id, 0) + 1

    def merge(self, other: "VectorClock") -> None:
        """Merge with another vector clock (element-wise max)."""
        for node_id, clock in other.clocks.items():
            self.clocks[node_id] = max(self.clocks.get(node_id, 0), clock)

    def is_concurrent(self, other: "VectorClock") -> bool:
        """Check if two vector clocks are concurrent (incomparable)."""
        has_greater = False
        has_lesser = False
        all_keys = set(self.clocks.keys()) | set(other.clocks.keys())
        for key in all_keys:
            a = self.clocks.get(key, 0)
            b = other.clocks.get(key, 0)
            if a > b:
                has_greater = True
            elif b > a:
                has_lesser = True
        return has_greater and has_lesser

    def dominates(self, other: "VectorClock") -> bool:
        """Check if this clock strictly dominates (happens-after) another."""
        at_least_one_greater = False
        for key in set(self.clocks.keys()) | set(other.clocks.keys()):
            a = self.clocks.get(key, 0)
            b = other.clocks.get(key, 0)
            if a < b:
                return False
            if a > b:
                at_least_one_greater = True
        return at_least_one_greater

    def copy(self) -> "VectorClock":
        return VectorClock(clocks=dict(self.clocks))

    def to_dict(self) -> Dict[str, int]:
        return dict(self.clocks)


# =============================================================================
# Node Info
# =============================================================================


@dataclass
class NodeInfo:
    """Comprehensive information about a cluster node."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    host: str = "localhost"
    port: int = 5000
    grpc_port: int = 5001
    metrics_port: int = 9090

    # Role and status
    role: NodeRole = NodeRole.FOLLOWER
    status: NodeStatus = NodeStatus.STARTING

    # Capabilities
    capabilities: Set[str] = field(default_factory=lambda: {
        NodeCapability.COMPUTE.value,
        NodeCapability.MEMORY.value,
        NodeCapability.TOOLS.value,
    })
    max_concurrent_tasks: int = 10

    # Resources
    cpu_cores: int = 1
    memory_mb: int = 1024
    gpu_count: int = 0
    disk_gb: int = 0

    # Current load
    current_tasks: int = 0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    network_rx_bytes: int = 0
    network_tx_bytes: int = 0

    # Topology
    region: str = ""
    zone: str = ""
    rack: str = ""
    labels: Dict[str, str] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)

    # Timestamps
    started_at: datetime = field(default_factory=datetime.now)
    last_heartbeat: datetime = field(default_factory=datetime.now)
    joined_at: Optional[datetime] = None

    # Version info
    version: str = "1.0.0"
    protocol_version: int = 1

    # Weights for load balancing
    weight: float = 1.0

    @property
    def address(self) -> str:
        return f"{self.host}:{self.port}"

    @property
    def grpc_address(self) -> str:
        return f"{self.host}:{self.grpc_port}"

    @property
    def load_score(self) -> float:
        """Calculate load score (0.0 - 1.0, lower is better)."""
        if self.max_concurrent_tasks == 0:
            return 1.0
        task_load = self.current_tasks / self.max_concurrent_tasks
        resource_load = (self.cpu_usage + self.memory_usage) / 2.0
        return min(1.0, (task_load * 0.6 + resource_load * 0.4))

    @property
    def available_capacity(self) -> int:
        """Number of additional tasks this node can accept."""
        return max(0, self.max_concurrent_tasks - self.current_tasks)

    def is_available(self) -> bool:
        """Check if node can accept new tasks."""
        return (
            self.status == NodeStatus.HEALTHY
            and self.current_tasks < self.max_concurrent_tasks
        )

    def is_voter(self) -> bool:
        """Check if this node participates in consensus voting."""
        return self.role in (NodeRole.LEADER, NodeRole.FOLLOWER, NodeRole.CANDIDATE)

    def heartbeat_age_seconds(self) -> float:
        """Seconds since last heartbeat."""
        return (datetime.now() - self.last_heartbeat).total_seconds()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "address": self.address,
            "grpc_address": self.grpc_address,
            "role": self.role.value,
            "status": self.status.value,
            "load_score": round(self.load_score, 4),
            "current_tasks": self.current_tasks,
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "capabilities": sorted(self.capabilities),
            "cpu_cores": self.cpu_cores,
            "memory_mb": self.memory_mb,
            "gpu_count": self.gpu_count,
            "cpu_usage": round(self.cpu_usage, 4),
            "memory_usage": round(self.memory_usage, 4),
            "region": self.region,
            "zone": self.zone,
            "version": self.version,
            "started_at": self.started_at.isoformat(),
            "last_heartbeat": self.last_heartbeat.isoformat(),
            "weight": self.weight,
        }


# =============================================================================
# Cluster State
# =============================================================================


@dataclass
class ClusterState:
    """Current state of the cluster."""
    cluster_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "aion-cluster"

    # Nodes
    nodes: Dict[str, NodeInfo] = field(default_factory=dict)
    leader_id: Optional[str] = None

    # Consensus
    term: int = 0
    commit_index: int = 0

    # Configuration
    min_nodes: int = 1
    replication_factor: int = 3
    read_consistency: ConsistencyLevel = ConsistencyLevel.QUORUM
    write_consistency: ConsistencyLevel = ConsistencyLevel.QUORUM

    # Epoch (incremented on topology changes)
    epoch: int = 0
    config_version: int = 0

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    @property
    def voter_nodes(self) -> List[NodeInfo]:
        """Nodes that participate in consensus."""
        return [n for n in self.nodes.values() if n.is_voter()]

    @property
    def healthy_nodes(self) -> List[NodeInfo]:
        return [n for n in self.nodes.values() if n.status == NodeStatus.HEALTHY]

    @property
    def quorum_size(self) -> int:
        """Required quorum (majority of voters)."""
        voters = len(self.voter_nodes)
        return voters // 2 + 1

    @property
    def has_quorum(self) -> bool:
        healthy_voters = [
            n for n in self.nodes.values()
            if n.status == NodeStatus.HEALTHY and n.is_voter()
        ]
        return len(healthy_voters) >= self.quorum_size

    def get_leader(self) -> Optional[NodeInfo]:
        if self.leader_id:
            return self.nodes.get(self.leader_id)
        return None

    def get_node(self, node_id: str) -> Optional[NodeInfo]:
        return self.nodes.get(node_id)

    def increment_epoch(self) -> None:
        self.epoch += 1
        self.updated_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "name": self.name,
            "leader_id": self.leader_id,
            "term": self.term,
            "epoch": self.epoch,
            "node_count": len(self.nodes),
            "healthy_count": len(self.healthy_nodes),
            "has_quorum": self.has_quorum,
            "replication_factor": self.replication_factor,
        }


# =============================================================================
# Distributed Task
# =============================================================================


@dataclass
class DistributedTask:
    """A task to be executed in the cluster."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    task_type: str = ""

    # Payload
    payload: Dict[str, Any] = field(default_factory=dict)

    # Scheduling
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING

    # Assignment
    assigned_node: Optional[str] = None
    required_capabilities: Set[str] = field(default_factory=set)
    preferred_nodes: List[str] = field(default_factory=list)
    excluded_nodes: Set[str] = field(default_factory=set)

    # Execution
    max_retries: int = 3
    retry_count: int = 0
    timeout_seconds: int = 300
    idempotency_key: Optional[str] = None

    # Dependencies
    depends_on: List[str] = field(default_factory=list)
    parent_task_id: Optional[str] = None

    # Results
    result: Optional[Any] = None
    error: Optional[str] = None
    error_traceback: Optional[str] = None

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    deadline: Optional[datetime] = None

    # Tracking
    source_node: str = ""
    correlation_id: Optional[str] = None
    attempt_history: List[Dict[str, Any]] = field(default_factory=list)

    # Routing
    routing_key: Optional[str] = None
    queue_name: str = "default"

    @property
    def is_terminal(self) -> bool:
        """Check if task is in a terminal state."""
        return self.status in (
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
            TaskStatus.DEAD_LETTER,
        )

    @property
    def execution_time_ms(self) -> Optional[float]:
        """Get execution time in milliseconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds() * 1000
        return None

    @property
    def is_expired(self) -> bool:
        """Check if task has exceeded its deadline."""
        if self.deadline:
            return datetime.now() > self.deadline
        return False

    def can_retry(self) -> bool:
        return self.retry_count < self.max_retries

    def record_attempt(self, node_id: str, error: Optional[str] = None) -> None:
        self.attempt_history.append({
            "attempt": self.retry_count,
            "node_id": node_id,
            "timestamp": datetime.now().isoformat(),
            "error": error,
        })

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "task_type": self.task_type,
            "priority": self.priority.value,
            "status": self.status.value,
            "assigned_node": self.assigned_node,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "source_node": self.source_node,
            "queue_name": self.queue_name,
            "created_at": self.created_at.isoformat(),
            "execution_time_ms": self.execution_time_ms,
            "is_terminal": self.is_terminal,
        }


# =============================================================================
# Shard Info
# =============================================================================


@dataclass
class ShardInfo:
    """Information about a data shard."""
    shard_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""

    # Location
    primary_node: str = ""
    replica_nodes: List[str] = field(default_factory=list)

    # Range (for range-based sharding)
    start_key: Optional[str] = None
    end_key: Optional[str] = None

    # Hash range (for consistent hashing)
    hash_start: int = 0
    hash_end: int = 0

    # Statistics
    item_count: int = 0
    size_bytes: int = 0

    # Status
    status: str = "active"
    version: int = 0
    last_sync: datetime = field(default_factory=datetime.now)
    last_compaction: Optional[datetime] = None

    def is_responsible_for(self, key_hash: int) -> bool:
        """Check if this shard is responsible for a given hash."""
        if self.hash_start <= self.hash_end:
            return self.hash_start <= key_hash <= self.hash_end
        # Wraps around
        return key_hash >= self.hash_start or key_hash <= self.hash_end

    def to_dict(self) -> Dict[str, Any]:
        return {
            "shard_id": self.shard_id,
            "name": self.name,
            "primary_node": self.primary_node,
            "replica_nodes": self.replica_nodes,
            "item_count": self.item_count,
            "size_bytes": self.size_bytes,
            "status": self.status,
            "version": self.version,
        }


# =============================================================================
# Raft Protocol Types
# =============================================================================


@dataclass
class RaftLogEntry:
    """Entry in the Raft consensus log."""
    index: int = 0
    term: int = 0
    command: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    # Configuration change entries
    is_config_change: bool = False
    config_data: Optional[Dict[str, Any]] = None

    # Noop entries (for leader establishment)
    is_noop: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "term": self.term,
            "command": self.command,
            "timestamp": self.timestamp.isoformat(),
            "is_config_change": self.is_config_change,
        }


@dataclass
class RaftState:
    """Raft consensus state for a node."""
    # Persistent state (on all servers)
    current_term: int = 0
    voted_for: Optional[str] = None
    log: List[RaftLogEntry] = field(default_factory=list)

    # Volatile state (on all servers)
    commit_index: int = -1
    last_applied: int = -1

    # Volatile state (on leaders)
    next_index: Dict[str, int] = field(default_factory=dict)
    match_index: Dict[str, int] = field(default_factory=dict)

    # Snapshot state
    last_snapshot_index: int = -1
    last_snapshot_term: int = 0

    @property
    def last_log_index(self) -> int:
        if self.log:
            return self.log[-1].index
        return self.last_snapshot_index

    @property
    def last_log_term(self) -> int:
        if self.log:
            return self.log[-1].term
        return self.last_snapshot_term

    def get_entry(self, index: int) -> Optional[RaftLogEntry]:
        """Get log entry at a given index."""
        offset = self.last_snapshot_index + 1
        adjusted = index - offset
        if 0 <= adjusted < len(self.log):
            return self.log[adjusted]
        return None

    def get_term_at(self, index: int) -> int:
        """Get term at a given index."""
        if index == self.last_snapshot_index:
            return self.last_snapshot_term
        entry = self.get_entry(index)
        return entry.term if entry else 0


# =============================================================================
# Raft RPC Messages
# =============================================================================


@dataclass
class VoteRequest:
    """Raft RequestVote RPC."""
    term: int
    candidate_id: str
    last_log_index: int
    last_log_term: int
    # Pre-vote extension (prevents term inflation during partitions)
    is_pre_vote: bool = False


@dataclass
class VoteResponse:
    """Raft RequestVote response."""
    term: int
    vote_granted: bool
    voter_id: str


@dataclass
class AppendEntriesRequest:
    """Raft AppendEntries RPC (also used as heartbeat)."""
    term: int
    leader_id: str
    prev_log_index: int
    prev_log_term: int
    entries: List[RaftLogEntry] = field(default_factory=list)
    leader_commit: int = -1


@dataclass
class AppendEntriesResponse:
    """Raft AppendEntries response."""
    term: int
    success: bool
    match_index: int = -1
    node_id: str = ""
    # Conflict optimization (skip to conflicting term start)
    conflict_term: Optional[int] = None
    conflict_index: Optional[int] = None


@dataclass
class InstallSnapshotRequest:
    """Raft InstallSnapshot RPC for log compaction."""
    term: int
    leader_id: str
    last_included_index: int
    last_included_term: int
    offset: int
    data: bytes = b""
    done: bool = False


@dataclass
class InstallSnapshotResponse:
    """Raft InstallSnapshot response."""
    term: int
    node_id: str


# =============================================================================
# Heartbeat and Health Messages
# =============================================================================


@dataclass
class HeartbeatMessage:
    """Heartbeat message between nodes."""
    node_id: str
    term: int
    timestamp: datetime = field(default_factory=datetime.now)
    load_score: float = 0.0
    status: NodeStatus = NodeStatus.HEALTHY
    current_tasks: int = 0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    epoch: int = 0


@dataclass
class HealthReport:
    """Comprehensive health report from a node."""
    node_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    status: NodeStatus = NodeStatus.HEALTHY

    # Resource metrics
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    disk_usage: float = 0.0
    network_latency_ms: float = 0.0
    goroutine_count: int = 0

    # Task metrics
    tasks_running: int = 0
    tasks_completed: int = 0
    tasks_failed: int = 0
    task_queue_depth: int = 0

    # Errors
    recent_errors: List[str] = field(default_factory=list)
    error_rate: float = 0.0

    def is_healthy(self) -> bool:
        return (
            self.cpu_usage < 0.95
            and self.memory_usage < 0.95
            and self.error_rate < 0.1
        )


# =============================================================================
# Snapshot Types
# =============================================================================


@dataclass
class SnapshotMetadata:
    """Metadata about a state snapshot."""
    snapshot_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    last_included_index: int = 0
    last_included_term: int = 0
    node_count: int = 0
    size_bytes: int = 0
    checksum: str = ""
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "last_included_index": self.last_included_index,
            "last_included_term": self.last_included_term,
            "size_bytes": self.size_bytes,
            "checksum": self.checksum,
            "created_at": self.created_at.isoformat(),
        }


# =============================================================================
# Replication Types
# =============================================================================


@dataclass
class ReplicationEvent:
    """Event to be replicated across nodes."""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_node: str = ""
    event_type: str = ""
    key: str = ""
    value: Any = None
    vector_clock: VectorClock = field(default_factory=VectorClock)
    timestamp: datetime = field(default_factory=datetime.now)
    consistency: ConsistencyLevel = ConsistencyLevel.QUORUM

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "source_node": self.source_node,
            "event_type": self.event_type,
            "key": self.key,
            "vector_clock": self.vector_clock.to_dict(),
            "timestamp": self.timestamp.isoformat(),
        }


# =============================================================================
# Cluster Configuration Change
# =============================================================================


@dataclass
class ConfigChange:
    """Cluster configuration change (joint consensus)."""
    change_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    change_type: str = ""  # "add_node", "remove_node", "update_config"
    node_id: Optional[str] = None
    node_info: Optional[NodeInfo] = None
    config_key: Optional[str] = None
    config_value: Optional[Any] = None
    timestamp: datetime = field(default_factory=datetime.now)


# =============================================================================
# Metrics Types
# =============================================================================


@dataclass
class ClusterMetrics:
    """Aggregated cluster metrics."""
    timestamp: datetime = field(default_factory=datetime.now)

    # Cluster health
    total_nodes: int = 0
    healthy_nodes: int = 0
    has_quorum: bool = False

    # Load
    avg_cpu_usage: float = 0.0
    avg_memory_usage: float = 0.0
    total_tasks_running: int = 0
    total_tasks_queued: int = 0

    # Throughput
    tasks_per_second: float = 0.0
    bytes_replicated: int = 0

    # Consensus
    current_term: int = 0
    leader_id: Optional[str] = None
    commit_lag: int = 0

    # Latency
    avg_rpc_latency_ms: float = 0.0
    p99_rpc_latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "total_nodes": self.total_nodes,
            "healthy_nodes": self.healthy_nodes,
            "has_quorum": self.has_quorum,
            "avg_cpu_usage": round(self.avg_cpu_usage, 4),
            "avg_memory_usage": round(self.avg_memory_usage, 4),
            "total_tasks_running": self.total_tasks_running,
            "tasks_per_second": round(self.tasks_per_second, 2),
            "current_term": self.current_term,
            "leader_id": self.leader_id,
            "avg_rpc_latency_ms": round(self.avg_rpc_latency_ms, 2),
        }
