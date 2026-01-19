"""
AION Process Manager - Core Data Models

State-of-the-art process and agent management data structures with:
- Type-safe dataclasses with full serialization support
- Comprehensive process state machine
- Resource tracking and limits
- Event-driven communication primitives
- Hierarchical process relationships
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Optional, Union, TypeVar, Generic
import hashlib
import json


class ProcessState(Enum):
    """
    Process state machine states.

    State transitions:
    CREATED -> STARTING -> RUNNING -> STOPPING -> STOPPED
                      |         |              |
                      v         v              v
                   FAILED   PAUSED ------> TERMINATED
                      |         |
                      v         v
                  (restart) RUNNING
    """
    CREATED = "created"       # Process created but not started
    STARTING = "starting"     # Process initializing
    RUNNING = "running"       # Process actively executing
    PAUSED = "paused"         # Process temporarily suspended
    STOPPING = "stopping"     # Graceful shutdown in progress
    STOPPED = "stopped"       # Clean termination
    FAILED = "failed"         # Terminated due to error
    TERMINATED = "terminated" # Force killed

    def is_active(self) -> bool:
        """Check if process is in an active state."""
        return self in (ProcessState.RUNNING, ProcessState.PAUSED)

    def is_terminal(self) -> bool:
        """Check if process is in a terminal state."""
        return self in (ProcessState.STOPPED, ProcessState.FAILED, ProcessState.TERMINATED)

    def can_transition_to(self, target: "ProcessState") -> bool:
        """Validate state transition."""
        valid_transitions = {
            ProcessState.CREATED: {ProcessState.STARTING, ProcessState.TERMINATED},
            ProcessState.STARTING: {ProcessState.RUNNING, ProcessState.FAILED, ProcessState.TERMINATED},
            ProcessState.RUNNING: {ProcessState.PAUSED, ProcessState.STOPPING, ProcessState.FAILED, ProcessState.TERMINATED},
            ProcessState.PAUSED: {ProcessState.RUNNING, ProcessState.STOPPING, ProcessState.TERMINATED},
            ProcessState.STOPPING: {ProcessState.STOPPED, ProcessState.FAILED, ProcessState.TERMINATED},
            ProcessState.STOPPED: set(),  # Terminal
            ProcessState.FAILED: {ProcessState.STARTING},  # Can restart
            ProcessState.TERMINATED: set(),  # Terminal
        }
        return target in valid_transitions.get(self, set())


class ProcessPriority(Enum):
    """
    Process priority levels for scheduling and resource allocation.
    Lower value = higher priority.
    """
    CRITICAL = 0    # System-critical, never preempt or kill
    HIGH = 1        # Important user-facing tasks
    NORMAL = 2      # Default priority
    LOW = 3         # Background tasks
    IDLE = 4        # Only run when system is idle

    def __lt__(self, other: "ProcessPriority") -> bool:
        return self.value < other.value

    def __le__(self, other: "ProcessPriority") -> bool:
        return self.value <= other.value


class ProcessType(Enum):
    """Types of processes managed by the supervisor."""
    AGENT = "agent"           # Long-running AI agent
    TASK = "task"             # One-shot task
    WORKER = "worker"         # Worker pool member
    SYSTEM = "system"         # System-level process
    SCHEDULED = "scheduled"   # Scheduled/cron job
    CHILD = "child"           # Child of another process


class RestartPolicy(Enum):
    """Restart behavior on process failure."""
    NEVER = "never"                          # Don't restart
    ON_FAILURE = "on_failure"                # Restart only on non-zero exit
    ALWAYS = "always"                        # Always restart (unless explicitly stopped)
    EXPONENTIAL_BACKOFF = "exponential_backoff"  # Restart with increasing delays

    def should_restart(self, exit_code: Optional[int], explicit_stop: bool) -> bool:
        """Determine if restart should occur."""
        if explicit_stop:
            return False
        if self == RestartPolicy.NEVER:
            return False
        if self == RestartPolicy.ON_FAILURE:
            return exit_code != 0
        return True  # ALWAYS or EXPONENTIAL_BACKOFF


class MessageType(Enum):
    """Types of inter-agent messages."""
    TEXT = "text"               # Plain text message
    COMMAND = "command"         # Command to execute
    QUERY = "query"             # Query requiring response
    RESPONSE = "response"       # Response to a query
    ERROR = "error"             # Error notification
    EVENT = "event"             # Event notification
    HEARTBEAT = "heartbeat"     # Health check
    SIGNAL = "signal"           # Control signal
    DATA = "data"               # Data transfer


class SignalType(Enum):
    """Process control signals."""
    PAUSE = "pause"
    RESUME = "resume"
    STOP = "stop"
    KILL = "kill"
    RELOAD = "reload"
    CHECKPOINT = "checkpoint"
    CUSTOM = "custom"


@dataclass
class ResourceLimits:
    """
    Resource limits for a process.
    None values indicate no limit.
    """
    max_memory_mb: Optional[int] = None
    max_cpu_percent: Optional[float] = None
    max_tokens_per_minute: Optional[int] = None
    max_tokens_total: Optional[int] = None
    max_runtime_seconds: Optional[int] = None
    max_concurrent_tools: int = 5
    max_child_processes: int = 3
    max_queue_size: int = 1000
    max_events_per_second: float = 100.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_memory_mb": self.max_memory_mb,
            "max_cpu_percent": self.max_cpu_percent,
            "max_tokens_per_minute": self.max_tokens_per_minute,
            "max_tokens_total": self.max_tokens_total,
            "max_runtime_seconds": self.max_runtime_seconds,
            "max_concurrent_tools": self.max_concurrent_tools,
            "max_child_processes": self.max_child_processes,
            "max_queue_size": self.max_queue_size,
            "max_events_per_second": self.max_events_per_second,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResourceLimits":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def merge_with(self, other: "ResourceLimits") -> "ResourceLimits":
        """Merge with another limits object, taking the more restrictive value."""
        return ResourceLimits(
            max_memory_mb=min(filter(None, [self.max_memory_mb, other.max_memory_mb]), default=None),
            max_cpu_percent=min(filter(None, [self.max_cpu_percent, other.max_cpu_percent]), default=None),
            max_tokens_per_minute=min(filter(None, [self.max_tokens_per_minute, other.max_tokens_per_minute]), default=None),
            max_tokens_total=min(filter(None, [self.max_tokens_total, other.max_tokens_total]), default=None),
            max_runtime_seconds=min(filter(None, [self.max_runtime_seconds, other.max_runtime_seconds]), default=None),
            max_concurrent_tools=min(self.max_concurrent_tools, other.max_concurrent_tools),
            max_child_processes=min(self.max_child_processes, other.max_child_processes),
            max_queue_size=min(self.max_queue_size, other.max_queue_size),
            max_events_per_second=min(self.max_events_per_second, other.max_events_per_second),
        )


@dataclass
class ResourceUsage:
    """Current resource usage of a process with comprehensive tracking."""
    memory_mb: float = 0.0
    cpu_percent: float = 0.0
    tokens_used: int = 0
    tokens_per_minute: float = 0.0
    runtime_seconds: float = 0.0
    tool_calls: int = 0
    active_children: int = 0
    queue_size: int = 0
    events_emitted: int = 0
    events_received: int = 0
    errors_count: int = 0
    last_activity: Optional[datetime] = None

    def exceeds_limits(self, limits: ResourceLimits) -> tuple[bool, Optional[str]]:
        """Check if usage exceeds limits. Returns (exceeded, reason)."""
        checks = [
            (limits.max_memory_mb and self.memory_mb > limits.max_memory_mb,
             f"Memory limit exceeded: {self.memory_mb:.1f}MB > {limits.max_memory_mb}MB"),
            (limits.max_cpu_percent and self.cpu_percent > limits.max_cpu_percent,
             f"CPU limit exceeded: {self.cpu_percent:.1f}% > {limits.max_cpu_percent}%"),
            (limits.max_tokens_total and self.tokens_used > limits.max_tokens_total,
             f"Token limit exceeded: {self.tokens_used} > {limits.max_tokens_total}"),
            (limits.max_tokens_per_minute and self.tokens_per_minute > limits.max_tokens_per_minute,
             f"Token rate limit exceeded: {self.tokens_per_minute:.1f}/min > {limits.max_tokens_per_minute}/min"),
            (limits.max_runtime_seconds and self.runtime_seconds > limits.max_runtime_seconds,
             f"Runtime limit exceeded: {self.runtime_seconds:.1f}s > {limits.max_runtime_seconds}s"),
            (self.active_children > limits.max_child_processes,
             f"Child process limit exceeded: {self.active_children} > {limits.max_child_processes}"),
            (self.queue_size > limits.max_queue_size,
             f"Queue size limit exceeded: {self.queue_size} > {limits.max_queue_size}"),
        ]

        for exceeded, reason in checks:
            if exceeded:
                return True, reason
        return False, None

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_mb": self.memory_mb,
            "cpu_percent": self.cpu_percent,
            "tokens_used": self.tokens_used,
            "tokens_per_minute": self.tokens_per_minute,
            "runtime_seconds": self.runtime_seconds,
            "tool_calls": self.tool_calls,
            "active_children": self.active_children,
            "queue_size": self.queue_size,
            "events_emitted": self.events_emitted,
            "events_received": self.events_received,
            "errors_count": self.errors_count,
            "last_activity": self.last_activity.isoformat() if self.last_activity else None,
        }


@dataclass
class ProcessInfo:
    """
    Comprehensive information about a managed process.
    This is the central data structure for process tracking.
    """
    id: str
    name: str
    type: ProcessType
    state: ProcessState
    priority: ProcessPriority

    # Lifecycle timestamps
    created_at: datetime
    started_at: Optional[datetime] = None
    stopped_at: Optional[datetime] = None
    last_heartbeat: Optional[datetime] = None

    # Parent/child relationships for process hierarchy
    parent_id: Optional[str] = None
    children_ids: list[str] = field(default_factory=list)

    # Restart configuration
    restart_policy: RestartPolicy = RestartPolicy.ON_FAILURE
    restart_count: int = 0
    max_restarts: int = 5
    restart_delay_seconds: float = 1.0
    last_restart_at: Optional[datetime] = None

    # Resources
    limits: ResourceLimits = field(default_factory=ResourceLimits)
    usage: ResourceUsage = field(default_factory=ResourceUsage)

    # Agent-specific fields
    agent_class: Optional[str] = None
    agent_config_hash: Optional[str] = None

    # Execution state
    error: Optional[str] = None
    error_traceback: Optional[str] = None
    exit_code: Optional[int] = None

    # Communication
    input_channels: list[str] = field(default_factory=list)
    output_channels: list[str] = field(default_factory=list)

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type.value,
            "state": self.state.value,
            "priority": self.priority.name,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "stopped_at": self.stopped_at.isoformat() if self.stopped_at else None,
            "last_heartbeat": self.last_heartbeat.isoformat() if self.last_heartbeat else None,
            "parent_id": self.parent_id,
            "children_ids": self.children_ids.copy(),
            "restart_policy": self.restart_policy.value,
            "restart_count": self.restart_count,
            "max_restarts": self.max_restarts,
            "agent_class": self.agent_class,
            "limits": self.limits.to_dict(),
            "usage": self.usage.to_dict(),
            "error": self.error,
            "exit_code": self.exit_code,
            "input_channels": self.input_channels.copy(),
            "output_channels": self.output_channels.copy(),
            "metadata": self.metadata.copy(),
            "tags": self.tags.copy(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProcessInfo":
        """Reconstruct ProcessInfo from dictionary."""
        data = data.copy()
        data["type"] = ProcessType(data["type"])
        data["state"] = ProcessState(data["state"])
        data["priority"] = ProcessPriority[data["priority"]]
        data["restart_policy"] = RestartPolicy(data["restart_policy"])
        data["created_at"] = datetime.fromisoformat(data["created_at"])

        for dt_field in ["started_at", "stopped_at", "last_heartbeat", "last_restart_at"]:
            if data.get(dt_field):
                data[dt_field] = datetime.fromisoformat(data[dt_field])

        if "limits" in data:
            data["limits"] = ResourceLimits.from_dict(data["limits"])
        if "usage" in data:
            usage_data = data["usage"]
            if usage_data.get("last_activity"):
                usage_data["last_activity"] = datetime.fromisoformat(usage_data["last_activity"])
            data["usage"] = ResourceUsage(**usage_data)

        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def update_state(self, new_state: ProcessState) -> bool:
        """Update state with validation."""
        if self.state.can_transition_to(new_state):
            self.state = new_state
            return True
        return False


@dataclass
class AgentConfig:
    """
    Configuration for spawning an AI agent process.
    Comprehensive settings for behavior, resources, and communication.
    """
    name: str
    agent_class: str  # Fully qualified class name or registered name

    # Behavior configuration
    system_prompt: Optional[str] = None
    tools: list[str] = field(default_factory=list)  # Tool names to enable
    memory_enabled: bool = True
    planning_enabled: bool = True
    reasoning_enabled: bool = True

    # Goals and instructions
    initial_goal: Optional[str] = None
    instructions: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)

    # Resources
    priority: ProcessPriority = ProcessPriority.NORMAL
    limits: ResourceLimits = field(default_factory=ResourceLimits)
    restart_policy: RestartPolicy = RestartPolicy.ON_FAILURE
    max_restarts: int = 5
    restart_delay_seconds: float = 1.0

    # Communication channels
    input_channels: list[str] = field(default_factory=list)
    output_channels: list[str] = field(default_factory=list)
    broadcast_channels: list[str] = field(default_factory=list)

    # LLM configuration
    model: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 4096

    # Behavior flags
    auto_checkpoint: bool = True
    checkpoint_interval_seconds: int = 300
    idle_timeout_seconds: Optional[int] = None

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "agent_class": self.agent_class,
            "system_prompt": self.system_prompt,
            "tools": self.tools.copy(),
            "memory_enabled": self.memory_enabled,
            "planning_enabled": self.planning_enabled,
            "reasoning_enabled": self.reasoning_enabled,
            "initial_goal": self.initial_goal,
            "instructions": self.instructions.copy(),
            "constraints": self.constraints.copy(),
            "priority": self.priority.name,
            "limits": self.limits.to_dict(),
            "restart_policy": self.restart_policy.value,
            "max_restarts": self.max_restarts,
            "restart_delay_seconds": self.restart_delay_seconds,
            "input_channels": self.input_channels.copy(),
            "output_channels": self.output_channels.copy(),
            "broadcast_channels": self.broadcast_channels.copy(),
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "auto_checkpoint": self.auto_checkpoint,
            "checkpoint_interval_seconds": self.checkpoint_interval_seconds,
            "idle_timeout_seconds": self.idle_timeout_seconds,
            "metadata": self.metadata.copy(),
            "tags": self.tags.copy(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentConfig":
        data = data.copy()
        if "priority" in data:
            data["priority"] = ProcessPriority[data["priority"]]
        if "restart_policy" in data:
            data["restart_policy"] = RestartPolicy(data["restart_policy"])
        if "limits" in data:
            data["limits"] = ResourceLimits.from_dict(data["limits"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def get_config_hash(self) -> str:
        """Generate a hash of the configuration for change detection."""
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]


@dataclass
class TaskDefinition:
    """
    Definition of a schedulable task.
    Supports one-time, interval, and cron-based scheduling.
    """
    id: str
    name: str

    # What to execute
    handler: str  # Function path or handler name
    params: dict[str, Any] = field(default_factory=dict)

    # Scheduling
    schedule_type: str = "once"  # "once", "interval", "cron"
    run_at: Optional[datetime] = None  # For "once"
    interval_seconds: Optional[int] = None  # For "interval"
    cron_expression: Optional[str] = None  # For "cron" (e.g., "0 9 * * *")
    timezone: str = "UTC"

    # Execution configuration
    timeout_seconds: int = 300
    max_retries: int = 3
    retry_delay_seconds: int = 60
    retry_backoff_multiplier: float = 2.0

    # State
    enabled: bool = True
    last_run: Optional[datetime] = None
    last_run_duration: Optional[float] = None
    last_run_success: Optional[bool] = None
    last_error: Optional[str] = None
    next_run: Optional[datetime] = None
    run_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    consecutive_failures: int = 0

    # Resources
    priority: ProcessPriority = ProcessPriority.NORMAL
    limits: ResourceLimits = field(default_factory=ResourceLimits)

    # Options
    coalesce: bool = True  # Skip missed runs if behind
    max_instances: int = 1  # Max concurrent executions
    jitter_seconds: int = 0  # Random delay up to this value

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "handler": self.handler,
            "params": self.params.copy(),
            "schedule_type": self.schedule_type,
            "run_at": self.run_at.isoformat() if self.run_at else None,
            "interval_seconds": self.interval_seconds,
            "cron_expression": self.cron_expression,
            "timezone": self.timezone,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "enabled": self.enabled,
            "last_run": self.last_run.isoformat() if self.last_run else None,
            "last_run_duration": self.last_run_duration,
            "last_run_success": self.last_run_success,
            "last_error": self.last_error,
            "next_run": self.next_run.isoformat() if self.next_run else None,
            "run_count": self.run_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "consecutive_failures": self.consecutive_failures,
            "priority": self.priority.name,
            "coalesce": self.coalesce,
            "max_instances": self.max_instances,
            "metadata": self.metadata.copy(),
            "tags": self.tags.copy(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskDefinition":
        data = data.copy()
        for dt_field in ["run_at", "last_run", "next_run"]:
            if data.get(dt_field):
                data[dt_field] = datetime.fromisoformat(data[dt_field])
        if "priority" in data:
            data["priority"] = ProcessPriority[data["priority"]]
        if "limits" in data:
            data["limits"] = ResourceLimits.from_dict(data["limits"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class Event:
    """
    An event in the event bus for inter-process communication.
    Supports correlation for request/response patterns.
    """
    id: str
    type: str  # Event type/channel (e.g., "process.started", "agent.message")
    source: str  # Process ID that emitted the event
    payload: dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)

    # Correlation for request/response
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None  # Channel to send responses

    # Delivery options
    ttl_seconds: Optional[int] = None  # Time-to-live
    priority: int = 0  # Higher = more important
    persistent: bool = False  # Should be persisted

    # Tracing
    trace_id: Optional[str] = None
    parent_event_id: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "source": self.source,
            "payload": self.payload.copy(),
            "timestamp": self.timestamp.isoformat(),
            "correlation_id": self.correlation_id,
            "reply_to": self.reply_to,
            "ttl_seconds": self.ttl_seconds,
            "priority": self.priority,
            "persistent": self.persistent,
            "trace_id": self.trace_id,
            "parent_event_id": self.parent_event_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Event":
        data = data.copy()
        if "timestamp" in data:
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def is_expired(self) -> bool:
        """Check if event has expired based on TTL."""
        if self.ttl_seconds is None:
            return False
        return (datetime.now() - self.timestamp).total_seconds() > self.ttl_seconds

    def create_response(
        self,
        source: str,
        payload: dict[str, Any],
        event_type: Optional[str] = None,
    ) -> "Event":
        """Create a response event to this event."""
        return Event(
            id=str(uuid.uuid4()),
            type=event_type or f"{self.type}.response",
            source=source,
            payload=payload,
            correlation_id=self.correlation_id or self.id,
            parent_event_id=self.id,
            trace_id=self.trace_id,
        )


@dataclass
class AgentMessage:
    """
    A message between agents for direct communication.
    Supports various message types and response tracking.
    """
    id: str
    from_agent: str
    to_agent: str  # Or "broadcast" for all agents
    content: str
    message_type: MessageType = MessageType.TEXT
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    # Response handling
    requires_response: bool = False
    response_timeout: float = 30.0
    correlation_id: Optional[str] = None
    in_reply_to: Optional[str] = None

    # Priority and ordering
    priority: int = 0
    sequence_number: Optional[int] = None

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "from_agent": self.from_agent,
            "to_agent": self.to_agent,
            "content": self.content,
            "message_type": self.message_type.value,
            "payload": self.payload.copy(),
            "timestamp": self.timestamp.isoformat(),
            "requires_response": self.requires_response,
            "response_timeout": self.response_timeout,
            "correlation_id": self.correlation_id,
            "in_reply_to": self.in_reply_to,
            "priority": self.priority,
            "sequence_number": self.sequence_number,
            "metadata": self.metadata.copy(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentMessage":
        data = data.copy()
        if "message_type" in data:
            data["message_type"] = MessageType(data["message_type"])
        if "timestamp" in data:
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def create_response(
        self,
        from_agent: str,
        content: str,
        payload: Optional[dict[str, Any]] = None,
    ) -> "AgentMessage":
        """Create a response message to this message."""
        return AgentMessage(
            id=str(uuid.uuid4()),
            from_agent=from_agent,
            to_agent=self.from_agent,
            content=content,
            message_type=MessageType.RESPONSE,
            payload=payload or {},
            correlation_id=self.correlation_id or self.id,
            in_reply_to=self.id,
        )


@dataclass
class ProcessCheckpoint:
    """
    Checkpoint for process state persistence and recovery.
    Enables resume after restart or crash.
    """
    id: str
    process_id: str
    timestamp: datetime
    state: ProcessState

    # Serialized state
    internal_state: dict[str, Any] = field(default_factory=dict)
    message_queue_snapshot: list[dict] = field(default_factory=list)

    # Context
    resource_usage: ResourceUsage = field(default_factory=ResourceUsage)
    restart_count: int = 0

    # Metadata
    reason: str = "scheduled"  # scheduled, manual, pre_shutdown, error
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "process_id": self.process_id,
            "timestamp": self.timestamp.isoformat(),
            "state": self.state.value,
            "internal_state": self.internal_state.copy(),
            "message_queue_snapshot": self.message_queue_snapshot.copy(),
            "resource_usage": self.resource_usage.to_dict(),
            "restart_count": self.restart_count,
            "reason": self.reason,
            "metadata": self.metadata.copy(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProcessCheckpoint":
        data = data.copy()
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        data["state"] = ProcessState(data["state"])
        if "resource_usage" in data:
            data["resource_usage"] = ResourceUsage(**data["resource_usage"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class SupervisorStats:
    """Statistics for the process supervisor."""
    processes_spawned: int = 0
    processes_terminated: int = 0
    processes_failed: int = 0
    restarts_performed: int = 0
    resource_violations: int = 0
    signals_sent: int = 0
    checkpoints_created: int = 0

    # Current state
    total_processes: int = 0
    running_processes: int = 0
    paused_processes: int = 0
    failed_processes: int = 0

    # Resource totals
    total_memory_mb: float = 0.0
    total_tokens_used: int = 0

    # Timing
    uptime_seconds: float = 0.0
    last_health_check: Optional[datetime] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "processes_spawned": self.processes_spawned,
            "processes_terminated": self.processes_terminated,
            "processes_failed": self.processes_failed,
            "restarts_performed": self.restarts_performed,
            "resource_violations": self.resource_violations,
            "signals_sent": self.signals_sent,
            "checkpoints_created": self.checkpoints_created,
            "total_processes": self.total_processes,
            "running_processes": self.running_processes,
            "paused_processes": self.paused_processes,
            "failed_processes": self.failed_processes,
            "total_memory_mb": self.total_memory_mb,
            "total_tokens_used": self.total_tokens_used,
            "uptime_seconds": self.uptime_seconds,
            "last_health_check": self.last_health_check.isoformat() if self.last_health_check else None,
        }


@dataclass
class WorkerInfo:
    """Information about a worker in the worker pool."""
    id: str
    state: ProcessState
    current_task_id: Optional[str] = None
    tasks_completed: int = 0
    tasks_failed: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    last_task_at: Optional[datetime] = None
    total_runtime_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "state": self.state.value,
            "current_task_id": self.current_task_id,
            "tasks_completed": self.tasks_completed,
            "tasks_failed": self.tasks_failed,
            "created_at": self.created_at.isoformat(),
            "last_task_at": self.last_task_at.isoformat() if self.last_task_at else None,
            "total_runtime_seconds": self.total_runtime_seconds,
        }
