"""
AION Process & Agent Manager System

State-of-the-art OS process management layer for AION that enables:
- Long-running AI agent management
- Background task execution
- Scheduled job processing
- Inter-process communication
- Resource limit enforcement
- Process lifecycle supervision

Example usage:
    ```python
    from aion.systems.process import (
        ProcessSupervisor,
        EventBus,
        TaskScheduler,
        WorkerPool,
        AgentConfig,
        BaseAgent,
        ProcessPriority,
        RestartPolicy,
    )

    # Initialize core components
    event_bus = EventBus()
    await event_bus.initialize()

    supervisor = ProcessSupervisor(event_bus)
    await supervisor.initialize()

    scheduler = TaskScheduler(supervisor, event_bus)
    await scheduler.initialize()

    # Define a custom agent
    class MyAgent(BaseAgent):
        async def run(self):
            while not self._shutdown_requested:
                await self._paused.wait()
                # Agent work here
                await self.sleep(1)

    # Register and spawn
    supervisor.register_agent_class("my_agent", MyAgent)

    agent_id = await supervisor.spawn_agent(AgentConfig(
        name="my_agent_1",
        agent_class="my_agent",
        priority=ProcessPriority.NORMAL,
        restart_policy=RestartPolicy.ON_FAILURE,
    ))

    # Schedule tasks
    await scheduler.schedule_cron(
        name="daily_cleanup",
        handler="cleanup_handler",
        cron_expression="0 3 * * *",
    )
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

__all__ = [
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
]

__version__ = "1.0.0"
