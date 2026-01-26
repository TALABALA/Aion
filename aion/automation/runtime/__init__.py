"""
AION Workflow Runtime - True SOTA Workflow Orchestration

Complete Temporal/Cadence-style workflow engine with:
- Deterministic replay with side-effect recording
- Activity heartbeats with progress reporting
- Workflow queries for state inspection
- Signals for external events
- Continue-as-new for history management
- Child workflow orchestration
- Search attributes for indexing
- Timer and schedule management
"""

from aion.automation.runtime.deterministic import (
    # Core types
    Command,
    CommandType,
    HistoryEvent,
    HistoryEventType,
    WorkflowExecution,
    WorkflowExecutionStatus,
    # Workflow history and replay
    WorkflowHistory,
    ReplayContext,
    ReplayMode,
    ReplayResult,
    # Workflow context (user-facing API)
    WorkflowContext,
    # Activity execution
    ActivityContext,
    ActivityResult,
    ActivityOptions,
    # Workers
    WorkflowWorker,
    ActivityWorker,
    # Search attributes
    SearchAttributeIndex,
    SearchAttributeType,
    # Decorators
    workflow,
    workflow_run,
    workflow_signal,
    workflow_query,
    activity,
)

from aion.automation.runtime.service import (
    # Activity heartbeats
    ActivityHeartbeat,
    ActivityHeartbeatManager,
    # Workflow state persistence
    WorkflowStateStore,
    # Signal and query handling
    SignalManager,
    PendingSignal,
    QueryManager,
    # Task queues
    TaskQueue,
    TaskType,
    Task,
    # Complete workflow service
    WorkflowService,
    WorkflowServiceConfig,
    # Client API
    WorkflowClient,
    WorkflowHandle,
    # Activity worker with heartbeats
    ActivityWorkerService,
)

__all__ = [
    # Core types
    "Command",
    "CommandType",
    "HistoryEvent",
    "HistoryEventType",
    "WorkflowExecution",
    "WorkflowExecutionStatus",
    # Workflow history and replay
    "WorkflowHistory",
    "ReplayContext",
    "ReplayMode",
    "ReplayResult",
    # Workflow context
    "WorkflowContext",
    # Activity execution
    "ActivityContext",
    "ActivityResult",
    "ActivityOptions",
    # Workers
    "WorkflowWorker",
    "ActivityWorker",
    # Search attributes
    "SearchAttributeIndex",
    "SearchAttributeType",
    # Decorators
    "workflow",
    "workflow_run",
    "workflow_signal",
    "workflow_query",
    "activity",
    # Activity heartbeats
    "ActivityHeartbeat",
    "ActivityHeartbeatManager",
    # Workflow state persistence
    "WorkflowStateStore",
    # Signal and query handling
    "SignalManager",
    "PendingSignal",
    "QueryManager",
    # Task queues
    "TaskQueue",
    "TaskType",
    "Task",
    # Complete workflow service
    "WorkflowService",
    "WorkflowServiceConfig",
    # Client API
    "WorkflowClient",
    "WorkflowHandle",
    # Activity worker service
    "ActivityWorkerService",
]
