"""
AION Workflow Automation System

Event-driven automation engine that orchestrates AION capabilities
through triggers, conditions, and actions.

Core Features:
- Multiple trigger types (schedule, webhook, event, data change, manual)
- Conditional execution with expression evaluation
- Rich action types (tool, agent, goal, webhook, notification, etc.)
- Human-in-the-loop approvals
- Sub-workflow composition
- Built-in workflow templates

SOTA Features:
- Event-sourced execution with replay capability
- Distributed task queue with Redis/RabbitMQ backends
- OpenTelemetry integration for tracing and metrics
- Visual workflow builder with React DAG editor
- Saga pattern with compensation transactions
"""

from aion.automation.types import (
    # Enums
    WorkflowStatus,
    ExecutionStatus,
    TriggerType,
    ActionType,
    ConditionOperator,
    ApprovalStatus,
    # Triggers
    TriggerConfig,
    Trigger,
    # Conditions
    Condition,
    # Actions
    ActionConfig,
    WorkflowStep,
    # Workflow
    Workflow,
    # Execution
    StepResult,
    WorkflowExecution,
    # Approvals
    ApprovalRequest,
    # Templates
    WorkflowTemplate,
)
from aion.automation.engine import WorkflowEngine
from aion.automation.registry import WorkflowRegistry
from aion.automation.triggers.manager import TriggerManager
from aion.automation.actions.executor import ActionExecutor
from aion.automation.conditions.evaluator import ConditionEvaluator
from aion.automation.approval.manager import ApprovalManager
from aion.automation.execution.context import ExecutionContext
from aion.automation.templates.builtin import get_builtin_templates

# SOTA imports (conditional)
try:
    from aion.automation.engine_enhanced import EnhancedWorkflowEngine, create_enhanced_engine
    ENHANCED_ENGINE_AVAILABLE = True
except ImportError:
    ENHANCED_ENGINE_AVAILABLE = False

try:
    from aion.automation.execution.event_store import EventStore, EventType, WorkflowReplayer
    EVENT_SOURCING_AVAILABLE = True
except ImportError:
    EVENT_SOURCING_AVAILABLE = False

try:
    from aion.automation.distributed import TaskQueue, Worker, WorkerPool, DistributedScheduler
    DISTRIBUTED_AVAILABLE = True
except ImportError:
    DISTRIBUTED_AVAILABLE = False

try:
    from aion.automation.observability import TelemetryProvider, WorkflowTracer, WorkflowMetrics
    OBSERVABILITY_AVAILABLE = True
except ImportError:
    OBSERVABILITY_AVAILABLE = False

try:
    from aion.automation.saga import SagaOrchestrator, SagaDefinition, CompensationManager
    SAGA_AVAILABLE = True
except ImportError:
    SAGA_AVAILABLE = False

try:
    from aion.automation.visual import WorkflowValidator, WorkflowExporter
    VISUAL_AVAILABLE = True
except ImportError:
    VISUAL_AVAILABLE = False

__all__ = [
    # Enums
    "WorkflowStatus",
    "ExecutionStatus",
    "TriggerType",
    "ActionType",
    "ConditionOperator",
    "ApprovalStatus",
    # Core types
    "TriggerConfig",
    "Trigger",
    "Condition",
    "ActionConfig",
    "WorkflowStep",
    "Workflow",
    "StepResult",
    "WorkflowExecution",
    "ApprovalRequest",
    "WorkflowTemplate",
    # Components
    "WorkflowEngine",
    "WorkflowRegistry",
    "TriggerManager",
    "ActionExecutor",
    "ConditionEvaluator",
    "ApprovalManager",
    "ExecutionContext",
    # Functions
    "get_builtin_templates",
    # SOTA Feature Flags
    "ENHANCED_ENGINE_AVAILABLE",
    "EVENT_SOURCING_AVAILABLE",
    "DISTRIBUTED_AVAILABLE",
    "OBSERVABILITY_AVAILABLE",
    "SAGA_AVAILABLE",
    "VISUAL_AVAILABLE",
]

# Conditionally export SOTA components
if ENHANCED_ENGINE_AVAILABLE:
    __all__.extend(["EnhancedWorkflowEngine", "create_enhanced_engine"])

if EVENT_SOURCING_AVAILABLE:
    __all__.extend(["EventStore", "EventType", "WorkflowReplayer"])

if DISTRIBUTED_AVAILABLE:
    __all__.extend(["TaskQueue", "Worker", "WorkerPool", "DistributedScheduler"])

if OBSERVABILITY_AVAILABLE:
    __all__.extend(["TelemetryProvider", "WorkflowTracer", "WorkflowMetrics"])

if SAGA_AVAILABLE:
    __all__.extend(["SagaOrchestrator", "SagaDefinition", "CompensationManager"])

if VISUAL_AVAILABLE:
    __all__.extend(["WorkflowValidator", "WorkflowExporter"])
