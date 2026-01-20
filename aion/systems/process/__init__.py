"""
AION Process & Agent Manager System

Enterprise-grade, state-of-the-art OS process management layer for AION that enables:
- Long-running AI agent management with full lifecycle control
- Background task execution with worker pools
- Scheduled job processing (cron, interval, one-time, DAG workflows)
- Inter-process communication via distributed event bus
- Resource limit enforcement and fair-share scheduling
- Process lifecycle supervision with restart policies
- Distributed cluster support with leader election
- Fault tolerance (circuit breaker, bulkhead, saga patterns)
- Comprehensive observability (OpenTelemetry, Prometheus)
- Security (capability-based access, sandboxing, audit logging)
- Process pooling for pre-warmed agents

Example usage:
    ```python
    from aion.systems.process import (
        # Core components
        ProcessSupervisor,
        EventBus,
        TaskScheduler,
        WorkerPool,

        # Configuration
        AgentConfig,
        ProcessPriority,
        RestartPolicy,

        # Agents
        BaseAgent,

        # Distributed
        ClusterCoordinator,
        DistributedEventBus,

        # DAG Workflows
        DAGScheduler,
        DAGDefinition,
        DAGTask,

        # Resilience
        CircuitBreaker,
        Bulkhead,
        SagaOrchestrator,

        # Observability
        Tracer,
        MetricsRegistry,
        AIONMetrics,

        # Security
        SecurityManager,
        Capability,
        CapabilityManager,

        # Pooling
        ProcessPool,
        PoolConfig,
    )

    # Initialize core components
    async with EventBus() as event_bus:
        async with ProcessSupervisor(event_bus) as supervisor:
            # Define a custom agent
            class MyAgent(BaseAgent):
                async def run(self):
                    while not self._shutdown_requested:
                        await self._paused.wait()
                        await self.sleep(1)

            # Register and spawn
            supervisor.register_agent_class("my_agent", MyAgent)
            agent_id = await supervisor.spawn_agent(AgentConfig(
                name="my_agent_1",
                agent_class="my_agent",
                priority=ProcessPriority.NORMAL,
                restart_policy=RestartPolicy.ON_FAILURE,
            ))
    ```
"""

# Core models
from aion.systems.process.models import (
    # State enums
    ProcessState,
    ProcessPriority,
    ProcessType,
    RestartPolicy,
    MessageType,
    SignalType,
    # Resource management
    ResourceLimits,
    ResourceUsage,
    # Process information
    ProcessInfo,
    AgentConfig,
    TaskDefinition,
    # Communication
    Event,
    AgentMessage,
    # Checkpointing
    ProcessCheckpoint,
    # Statistics
    SupervisorStats,
    WorkerInfo,
)

# Event bus
from aion.systems.process.event_bus import (
    EventBus,
    TypedEventBus,
    Subscription,
)

# Agent base class
from aion.systems.process.agent_base import (
    BaseAgent,
    AgentContext,
)

# Process supervisor
from aion.systems.process.supervisor import ProcessSupervisor

# Task scheduler
from aion.systems.process.scheduler import (
    TaskScheduler,
    CronParser,
)

# Worker pool
from aion.systems.process.worker_pool import (
    WorkerPool,
    Worker,
    QueuedTask,
    TaskStatus,
)

# Built-in agents
from aion.systems.process.builtin_agents import (
    HealthMonitorAgent,
    GarbageCollectorAgent,
    MetricsCollectorAgent,
    WatchdogAgent,
    LogAggregatorAgent,
)

# Persistence
from aion.systems.process.persistence import (
    ProcessStore,
    SQLiteProcessStore,
    InMemoryProcessStore,
    create_store,
)

# Resource management
from aion.systems.process.resources import (
    ResourceManager,
    ResourcePressure,
    ResourceQuota,
    TokenBucket,
    SlidingWindowRateLimiter,
    ResourceThrottler,
)

# === NEW SOTA FEATURES ===

# Cluster / Distributed support
from aion.systems.process.cluster import (
    ClusterCoordinator,
    NodeInfo,
    NodeState,
    LeaderState,
    ClusterMessage,
    ProcessMigration,
    ConsistentHashRing,
    ClusterTransport,
    UDPClusterTransport,
    TCPClusterTransport,
)

# Distributed Event Bus
from aion.systems.process.distributed_bus import (
    DistributedEventBus,
    DistributedMessage,
    DistributedBusBackend,
    InMemoryDistributedBackend,
    RedisDistributedBackend,
    ConsumerGroup,
    TopicConfig,
    DeliveryGuarantee,
    PartitionStrategy,
)

# DAG-based Task Scheduler
from aion.systems.process.dag_scheduler import (
    DAGScheduler,
    DAGDefinition,
    DAGTask,
    DAGRun,
    TaskStatus as DAGTaskStatus,
    TriggerRule,
    FairShareGroup,
    create_dag,
    create_task,
)

# Observability (OpenTelemetry + Prometheus)
from aion.systems.process.observability import (
    # Tracing
    Tracer,
    Span,
    SpanContext,
    SpanKind,
    SpanStatus,
    SpanExporter,
    ConsoleSpanExporter,
    OTLPSpanExporter,
    # Metrics
    MetricsRegistry,
    Counter,
    Gauge,
    Histogram,
    Summary,
    AIONMetrics,
    # Globals
    get_tracer,
    get_metrics,
    set_tracer,
    set_metrics,
)

# Resilience patterns
from aion.systems.process.resilience import (
    # Circuit Breaker
    CircuitBreaker,
    CircuitState,
    CircuitBreakerConfig,
    CircuitBreakerMetrics,
    CircuitOpenError,
    # Bulkhead
    Bulkhead,
    BulkheadConfig,
    BulkheadFullError,
    # Retry
    Retry,
    RetryConfig,
    BackoffStrategy,
    # Timeout
    Timeout,
    # Saga
    SagaOrchestrator,
    SagaDefinition,
    SagaStep,
    SagaStepStatus,
    SagaFailedError,
    # Combined
    resilient,
    ResilienceManager,
    get_resilience_manager,
)

# Security
from aion.systems.process.security import (
    # Capabilities
    Capability,
    CapabilityToken,
    CapabilityManager,
    require_capability,
    # Sandboxing
    Sandbox,
    SandboxConfig,
    SandboxManager,
    # Audit logging
    AuditLogger,
    AuditEvent,
    AuditEventType,
    AuditSink,
    ConsoleAuditSink,
    FileAuditSink,
    # Security Manager
    SecurityManager,
    get_security_manager,
    initialize_security,
)

# Process Pooling
from aion.systems.process.pool import (
    ProcessPool,
    PoolConfig,
    PoolStats,
    PooledProcess,
    PooledProcessState,
    ProcessPoolManager,
    PoolExhaustedError,
    OptimizedCronParser,
    get_pool_manager,
)

# === TRUE SOTA FEATURES ===

# Raft Consensus
from aion.systems.process.consensus import (
    RaftNode,
    RaftState,
    RaftLog,
    LogEntry,
    Snapshot,
    AppendEntriesRequest,
    AppendEntriesResponse,
    RequestVoteRequest,
    RequestVoteResponse,
    NotLeaderError,
    # Vector Clocks
    VectorClock,
    # SWIM Protocol
    SWIMProtocol,
    SWIMMember,
    SWIMState,
)

# Advanced Patterns
from aion.systems.process.advanced_patterns import (
    # Adaptive Circuit Breaker
    AdaptiveCircuitBreaker,
    CircuitBreakerOpenError,
    # Hedge Requests
    HedgeRequest,
    # Tail-Based Sampling
    TailBasedSampler,
    SamplingDecision,
    TraceData,
    # Dynamic DAGs
    DynamicDAG,
    DynamicTask,
    BackfillOperation,
)

__all__ = [
    # === Core ===
    # State enums
    "ProcessState",
    "ProcessPriority",
    "ProcessType",
    "RestartPolicy",
    "MessageType",
    "SignalType",
    # Resource management
    "ResourceLimits",
    "ResourceUsage",
    "ResourceManager",
    "ResourcePressure",
    "ResourceQuota",
    "TokenBucket",
    "SlidingWindowRateLimiter",
    "ResourceThrottler",
    # Process information
    "ProcessInfo",
    "AgentConfig",
    "TaskDefinition",
    # Communication
    "Event",
    "AgentMessage",
    "EventBus",
    "TypedEventBus",
    "Subscription",
    # Checkpointing
    "ProcessCheckpoint",
    # Statistics
    "SupervisorStats",
    "WorkerInfo",
    # Core classes
    "BaseAgent",
    "AgentContext",
    "ProcessSupervisor",
    "TaskScheduler",
    "CronParser",
    "WorkerPool",
    "Worker",
    "QueuedTask",
    "TaskStatus",
    # Built-in agents
    "HealthMonitorAgent",
    "GarbageCollectorAgent",
    "MetricsCollectorAgent",
    "WatchdogAgent",
    "LogAggregatorAgent",
    # Persistence
    "ProcessStore",
    "SQLiteProcessStore",
    "InMemoryProcessStore",
    "create_store",

    # === SOTA Features ===

    # Cluster / Distributed
    "ClusterCoordinator",
    "NodeInfo",
    "NodeState",
    "LeaderState",
    "ClusterMessage",
    "ProcessMigration",
    "ConsistentHashRing",
    "ClusterTransport",
    "UDPClusterTransport",
    "TCPClusterTransport",

    # Distributed Event Bus
    "DistributedEventBus",
    "DistributedMessage",
    "DistributedBusBackend",
    "InMemoryDistributedBackend",
    "RedisDistributedBackend",
    "ConsumerGroup",
    "TopicConfig",
    "DeliveryGuarantee",
    "PartitionStrategy",

    # DAG Scheduler
    "DAGScheduler",
    "DAGDefinition",
    "DAGTask",
    "DAGRun",
    "DAGTaskStatus",
    "TriggerRule",
    "FairShareGroup",
    "create_dag",
    "create_task",

    # Observability
    "Tracer",
    "Span",
    "SpanContext",
    "SpanKind",
    "SpanStatus",
    "SpanExporter",
    "ConsoleSpanExporter",
    "OTLPSpanExporter",
    "MetricsRegistry",
    "Counter",
    "Gauge",
    "Histogram",
    "Summary",
    "AIONMetrics",
    "get_tracer",
    "get_metrics",
    "set_tracer",
    "set_metrics",

    # Resilience
    "CircuitBreaker",
    "CircuitState",
    "CircuitBreakerConfig",
    "CircuitBreakerMetrics",
    "CircuitOpenError",
    "Bulkhead",
    "BulkheadConfig",
    "BulkheadFullError",
    "Retry",
    "RetryConfig",
    "BackoffStrategy",
    "Timeout",
    "SagaOrchestrator",
    "SagaDefinition",
    "SagaStep",
    "SagaStepStatus",
    "SagaFailedError",
    "resilient",
    "ResilienceManager",
    "get_resilience_manager",

    # Security
    "Capability",
    "CapabilityToken",
    "CapabilityManager",
    "require_capability",
    "Sandbox",
    "SandboxConfig",
    "SandboxManager",
    "AuditLogger",
    "AuditEvent",
    "AuditEventType",
    "AuditSink",
    "ConsoleAuditSink",
    "FileAuditSink",
    "SecurityManager",
    "get_security_manager",
    "initialize_security",

    # Process Pooling
    "ProcessPool",
    "PoolConfig",
    "PoolStats",
    "PooledProcess",
    "PooledProcessState",
    "ProcessPoolManager",
    "PoolExhaustedError",
    "OptimizedCronParser",
    "get_pool_manager",

    # === TRUE SOTA ===

    # Raft Consensus
    "RaftNode",
    "RaftState",
    "RaftLog",
    "LogEntry",
    "Snapshot",
    "AppendEntriesRequest",
    "AppendEntriesResponse",
    "RequestVoteRequest",
    "RequestVoteResponse",
    "NotLeaderError",
    "VectorClock",
    "SWIMProtocol",
    "SWIMMember",
    "SWIMState",

    # Advanced Patterns
    "AdaptiveCircuitBreaker",
    "CircuitBreakerOpenError",
    "HedgeRequest",
    "TailBasedSampler",
    "SamplingDecision",
    "TraceData",
    "DynamicDAG",
    "DynamicTask",
    "BackfillOperation",
]

__version__ = "2.1.0"
