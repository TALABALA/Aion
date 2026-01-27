"""
AION Distributed Computing Configuration

Comprehensive configuration for all distributed subsystems using
Pydantic for validation and environment variable support.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set

from pydantic import BaseModel, Field, field_validator


class NetworkConfig(BaseModel):
    """Network configuration."""
    host: str = Field(default="0.0.0.0", description="Bind address")
    port: int = Field(default=5000, ge=1024, le=65535, description="HTTP port")
    grpc_port: int = Field(default=5001, ge=1024, le=65535, description="gRPC port")
    metrics_port: int = Field(default=9090, ge=1024, le=65535, description="Metrics port")
    advertise_host: Optional[str] = Field(default=None, description="External hostname")
    max_connections: int = Field(default=1000, ge=1)
    connection_timeout_seconds: float = Field(default=5.0, gt=0)
    request_timeout_seconds: float = Field(default=30.0, gt=0)
    keepalive_interval_seconds: float = Field(default=10.0, gt=0)
    tls_enabled: bool = Field(default=False)
    tls_cert_path: Optional[str] = None
    tls_key_path: Optional[str] = None
    tls_ca_path: Optional[str] = None


class RaftConfig(BaseModel):
    """Raft consensus configuration."""
    election_timeout_min_ms: int = Field(default=150, ge=50, description="Min election timeout")
    election_timeout_max_ms: int = Field(default=300, ge=100, description="Max election timeout")
    heartbeat_interval_ms: int = Field(default=50, ge=10, description="Leader heartbeat interval")
    max_log_entries_per_request: int = Field(default=100, ge=1)
    snapshot_threshold: int = Field(default=10000, ge=100, description="Entries before snapshot")
    snapshot_trailing_logs: int = Field(default=1000, ge=0)
    pre_vote_enabled: bool = Field(default=True, description="Enable PreVote protocol")
    leader_lease_timeout_ms: int = Field(default=500, ge=100)
    max_append_entries_batch: int = Field(default=64, ge=1)

    @field_validator("election_timeout_max_ms")
    @classmethod
    def validate_election_timeout(cls, v: int, info) -> int:
        min_val = info.data.get("election_timeout_min_ms", 150)
        if v <= min_val:
            raise ValueError("election_timeout_max_ms must be greater than min")
        return v


class DiscoveryConfig(BaseModel):
    """Node discovery configuration."""
    method: Literal["static", "dns", "multicast", "kubernetes", "consul"] = "static"
    static_nodes: List[str] = Field(default_factory=list, description="Static seed nodes")
    dns_name: Optional[str] = None
    dns_poll_interval_seconds: int = Field(default=30, ge=5)
    multicast_group: str = "239.1.2.3"
    multicast_port: int = 5002
    multicast_ttl: int = 1
    kubernetes_namespace: str = "default"
    kubernetes_service: str = "aion"
    kubernetes_label_selector: str = "app=aion"
    consul_address: str = "localhost:8500"
    consul_service: str = "aion"


class HealthConfig(BaseModel):
    """Health checking configuration."""
    check_interval_seconds: float = Field(default=5.0, gt=0)
    heartbeat_interval_seconds: float = Field(default=1.0, gt=0)
    failure_timeout_seconds: float = Field(default=15.0, gt=0)
    suspect_timeout_seconds: float = Field(default=10.0, gt=0)
    max_missed_heartbeats: int = Field(default=5, ge=1)
    phi_accrual_threshold: float = Field(default=8.0, gt=0, description="Phi accrual failure detector threshold")
    phi_accrual_window_size: int = Field(default=100, ge=10)
    deregister_after_seconds: float = Field(default=60.0, gt=0)


class TaskQueueConfig(BaseModel):
    """Distributed task queue configuration."""
    max_queue_size: int = Field(default=10000, ge=1)
    max_concurrent_per_node: int = Field(default=10, ge=1)
    default_timeout_seconds: int = Field(default=300, ge=1)
    default_max_retries: int = Field(default=3, ge=0)
    retry_backoff_base_seconds: float = Field(default=1.0, ge=0.1)
    retry_backoff_max_seconds: float = Field(default=60.0, ge=1.0)
    dead_letter_enabled: bool = True
    dead_letter_max_size: int = Field(default=1000, ge=0)
    task_ttl_seconds: int = Field(default=3600, ge=60)
    enable_priority_queue: bool = True
    enable_task_deduplication: bool = True
    deduplication_window_seconds: int = Field(default=300, ge=1)


class ReplicationConfig(BaseModel):
    """Data replication configuration."""
    factor: int = Field(default=3, ge=1, description="Replication factor")
    mode: Literal["sync", "async", "semi_sync"] = "semi_sync"
    min_sync_replicas: int = Field(default=1, ge=0)
    read_consistency: Literal["one", "quorum", "all", "local"] = "quorum"
    write_consistency: Literal["one", "quorum", "all"] = "quorum"
    conflict_resolution: Literal["lww", "vector_clock", "crdt_merge"] = "lww"
    sync_interval_seconds: float = Field(default=5.0, gt=0)
    compaction_interval_seconds: float = Field(default=3600.0, gt=0)


class ShardingConfig(BaseModel):
    """Memory sharding configuration."""
    strategy: Literal["consistent_hash", "range", "round_robin", "locality"] = "consistent_hash"
    num_virtual_nodes: int = Field(default=150, ge=10, description="Virtual nodes per physical node")
    rebalance_threshold: float = Field(default=0.2, ge=0.01, le=1.0)
    auto_rebalance: bool = True
    rebalance_cooldown_seconds: float = Field(default=60.0, ge=0)
    max_shard_size_bytes: int = Field(default=1_073_741_824, ge=1_048_576)  # 1GB default


class LoadBalancingConfig(BaseModel):
    """Load balancing configuration."""
    strategy: Literal[
        "round_robin", "least_connections", "weighted",
        "capability_aware", "locality_aware", "adaptive",
    ] = "adaptive"
    locality_preference: bool = True
    health_weight: float = Field(default=0.3, ge=0, le=1.0)
    load_weight: float = Field(default=0.4, ge=0, le=1.0)
    latency_weight: float = Field(default=0.3, ge=0, le=1.0)
    sticky_session_enabled: bool = False
    sticky_session_ttl_seconds: int = Field(default=300, ge=1)
    circuit_breaker_enabled: bool = True
    circuit_breaker_threshold: int = Field(default=5, ge=1)
    circuit_breaker_timeout_seconds: float = Field(default=30.0, ge=1.0)


class FaultToleranceConfig(BaseModel):
    """Fault tolerance configuration."""
    enable_auto_recovery: bool = True
    recovery_grace_period_seconds: float = Field(default=30.0, ge=0)
    failover_timeout_seconds: float = Field(default=10.0, ge=1.0)
    max_failover_attempts: int = Field(default=3, ge=1)
    enable_split_brain_detection: bool = True
    partition_healing_enabled: bool = True
    data_repair_enabled: bool = True
    repair_interval_seconds: float = Field(default=300.0, ge=10.0)


class TrainingConfig(BaseModel):
    """Distributed training configuration."""
    sync_interval_seconds: float = Field(default=60.0, gt=0)
    gradient_compression: bool = True
    gradient_compression_ratio: float = Field(default=0.1, gt=0, le=1.0)
    experience_share_ratio: float = Field(default=0.1, gt=0, le=1.0)
    parameter_sync_mode: Literal["sync", "async", "periodic"] = "periodic"
    all_reduce_enabled: bool = True
    gradient_clip_norm: float = Field(default=1.0, gt=0)
    learning_rate_warmup_steps: int = Field(default=100, ge=0)


class DistributedConfig(BaseModel):
    """
    Master configuration for the AION Distributed Computing System.

    All sub-configurations are accessible via dot notation.
    """
    # Cluster identity
    cluster_name: str = Field(default="aion-cluster", min_length=1)
    node_name: str = Field(default="aion-node", min_length=1)
    node_id: Optional[str] = Field(default=None, description="Override node ID")

    # Enable/disable
    enabled: bool = True
    min_nodes: int = Field(default=1, ge=1)

    # Sub-configurations
    network: NetworkConfig = Field(default_factory=NetworkConfig)
    raft: RaftConfig = Field(default_factory=RaftConfig)
    discovery: DiscoveryConfig = Field(default_factory=DiscoveryConfig)
    health: HealthConfig = Field(default_factory=HealthConfig)
    task_queue: TaskQueueConfig = Field(default_factory=TaskQueueConfig)
    replication: ReplicationConfig = Field(default_factory=ReplicationConfig)
    sharding: ShardingConfig = Field(default_factory=ShardingConfig)
    load_balancing: LoadBalancingConfig = Field(default_factory=LoadBalancingConfig)
    fault_tolerance: FaultToleranceConfig = Field(default_factory=FaultToleranceConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)

    # Data paths
    data_dir: str = Field(default="data/distributed")
    wal_dir: str = Field(default="data/distributed/wal")
    snapshot_dir: str = Field(default="data/distributed/snapshots")

    # Capabilities
    capabilities: List[str] = Field(
        default_factory=lambda: ["compute", "memory", "tools"],
    )

    # Resource limits
    max_concurrent_tasks: int = Field(default=10, ge=1)
    max_memory_mb: int = Field(default=4096, ge=256)

    def to_flat_dict(self) -> Dict[str, Any]:
        """Flatten config for cluster manager consumption."""
        return {
            "cluster_name": self.cluster_name,
            "node_name": self.node_name,
            "host": self.network.host,
            "port": self.network.port,
            "grpc_port": self.network.grpc_port,
            "capabilities": self.capabilities,
            "max_tasks": self.max_concurrent_tasks,
            "min_nodes": self.min_nodes,
            "replication_factor": self.replication.factor,
            "heartbeat_interval": self.health.heartbeat_interval_seconds,
            "health_check_interval": self.health.check_interval_seconds,
            "health_timeout": self.health.failure_timeout_seconds,
        }


def get_default_config() -> DistributedConfig:
    """Get default distributed configuration."""
    return DistributedConfig()


def get_development_config() -> DistributedConfig:
    """Get development/single-node configuration."""
    return DistributedConfig(
        cluster_name="aion-dev",
        node_name="dev-node",
        min_nodes=1,
        raft=RaftConfig(
            election_timeout_min_ms=500,
            election_timeout_max_ms=1000,
            heartbeat_interval_ms=100,
        ),
        health=HealthConfig(
            check_interval_seconds=10.0,
            failure_timeout_seconds=30.0,
        ),
        replication=ReplicationConfig(factor=1),
    )


def get_production_config() -> DistributedConfig:
    """Get production cluster configuration."""
    return DistributedConfig(
        min_nodes=3,
        raft=RaftConfig(
            pre_vote_enabled=True,
            snapshot_threshold=50000,
        ),
        replication=ReplicationConfig(
            factor=3,
            mode="semi_sync",
            min_sync_replicas=1,
        ),
        fault_tolerance=FaultToleranceConfig(
            enable_auto_recovery=True,
            enable_split_brain_detection=True,
        ),
        load_balancing=LoadBalancingConfig(
            strategy="adaptive",
            circuit_breaker_enabled=True,
        ),
    )
