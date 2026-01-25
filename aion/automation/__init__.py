"""
AION Workflow Automation System

Event-driven automation engine that orchestrates AION capabilities
through triggers, conditions, and actions.

Features:
- Multiple trigger types (schedule, webhook, event, data change, manual)
- Conditional execution with expression evaluation
- Rich action types (tool, agent, goal, webhook, notification, etc.)
- Human-in-the-loop approvals
- Sub-workflow composition
- Built-in workflow templates
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
]
