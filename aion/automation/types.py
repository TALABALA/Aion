"""
AION Workflow Automation Types

Core dataclasses for workflow definitions and execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Callable
import uuid


# === Enums ===


class WorkflowStatus(str, Enum):
    """Status of a workflow definition."""
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    ARCHIVED = "archived"


class ExecutionStatus(str, Enum):
    """Status of a workflow execution."""
    PENDING = "pending"
    RUNNING = "running"
    WAITING = "waiting"      # Waiting for approval/input
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"


class TriggerType(str, Enum):
    """Types of workflow triggers."""
    SCHEDULE = "schedule"        # Cron-based
    WEBHOOK = "webhook"          # HTTP webhook
    EVENT = "event"              # Internal event
    DATA_CHANGE = "data_change"  # Data mutation
    MANUAL = "manual"            # User-initiated
    WORKFLOW = "workflow"        # Sub-workflow completion


class ActionType(str, Enum):
    """Types of workflow actions."""
    TOOL = "tool"                # Execute a tool
    AGENT = "agent"              # Spawn/manage agent
    GOAL = "goal"                # Create/manage goal
    WEBHOOK = "webhook"          # Call external URL
    NOTIFICATION = "notification"  # Send notification
    DATA = "data"                # Data operation
    WORKFLOW = "workflow"        # Invoke sub-workflow
    APPROVAL = "approval"        # Human approval gate
    CONDITION = "condition"      # Conditional branch
    LOOP = "loop"                # Loop construct
    DELAY = "delay"              # Wait/delay
    TRANSFORM = "transform"      # Data transformation
    LLM = "llm"                  # LLM completion
    SCRIPT = "script"            # Execute script
    PARALLEL = "parallel"        # Parallel execution


class ConditionOperator(str, Enum):
    """Condition operators for comparisons."""
    EQUALS = "eq"
    NOT_EQUALS = "ne"
    GREATER_THAN = "gt"
    LESS_THAN = "lt"
    GREATER_EQUAL = "gte"
    LESS_EQUAL = "lte"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    MATCHES = "matches"          # Regex
    IS_NULL = "is_null"
    IS_NOT_NULL = "is_not_null"
    IS_EMPTY = "is_empty"
    IS_NOT_EMPTY = "is_not_empty"
    IN = "in"
    NOT_IN = "not_in"
    IS_TRUE = "is_true"
    IS_FALSE = "is_false"


class ApprovalStatus(str, Enum):
    """Status of an approval request."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class DataOperation(str, Enum):
    """Data operation types."""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    UPDATE = "update"
    QUERY = "query"


class AgentOperation(str, Enum):
    """Agent operation types."""
    SPAWN = "spawn"
    TERMINATE = "terminate"
    MESSAGE = "message"
    WAIT = "wait"
    STATUS = "status"


class GoalOperation(str, Enum):
    """Goal operation types."""
    CREATE = "create"
    PAUSE = "pause"
    RESUME = "resume"
    ABANDON = "abandon"
    UPDATE = "update"
    STATUS = "status"


# === Trigger Configuration ===


@dataclass
class TriggerConfig:
    """Configuration for a workflow trigger."""
    trigger_type: TriggerType = TriggerType.MANUAL

    # Schedule trigger
    cron_expression: Optional[str] = None
    timezone: str = "UTC"

    # Webhook trigger
    webhook_path: Optional[str] = None
    webhook_secret: Optional[str] = None
    webhook_methods: List[str] = field(default_factory=lambda: ["POST"])

    # Event trigger
    event_type: Optional[str] = None
    event_source: Optional[str] = None
    event_filter: Optional[Dict[str, Any]] = None

    # Data change trigger
    data_source: Optional[str] = None  # memory, knowledge, state
    data_operation: Optional[str] = None  # create, update, delete
    data_filter: Optional[Dict[str, Any]] = None

    # Rate limiting
    min_interval_seconds: float = 0.0  # Minimum time between triggers
    max_triggers_per_hour: Optional[int] = None

    # Common
    enabled: bool = True
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.trigger_type.value,
            "cron": self.cron_expression,
            "timezone": self.timezone,
            "webhook_path": self.webhook_path,
            "event_type": self.event_type,
            "event_source": self.event_source,
            "data_source": self.data_source,
            "enabled": self.enabled,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TriggerConfig":
        """Create from dictionary."""
        trigger_type = TriggerType(data.get("type", "manual"))
        return cls(
            trigger_type=trigger_type,
            cron_expression=data.get("cron"),
            timezone=data.get("timezone", "UTC"),
            webhook_path=data.get("webhook_path"),
            webhook_secret=data.get("webhook_secret"),
            event_type=data.get("event_type"),
            event_source=data.get("event_source"),
            event_filter=data.get("event_filter"),
            data_source=data.get("data_source"),
            data_operation=data.get("data_operation"),
            data_filter=data.get("data_filter"),
            enabled=data.get("enabled", True),
            description=data.get("description", ""),
        )


@dataclass
class Trigger:
    """A workflow trigger instance."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    workflow_id: str = ""
    config: TriggerConfig = field(default_factory=TriggerConfig)

    # State
    last_triggered_at: Optional[datetime] = None
    next_trigger_at: Optional[datetime] = None
    trigger_count: int = 0
    consecutive_failures: int = 0

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "workflow_id": self.workflow_id,
            "config": self.config.to_dict(),
            "last_triggered_at": self.last_triggered_at.isoformat() if self.last_triggered_at else None,
            "next_trigger_at": self.next_trigger_at.isoformat() if self.next_trigger_at else None,
            "trigger_count": self.trigger_count,
            "created_at": self.created_at.isoformat(),
        }


# === Conditions ===


@dataclass
class Condition:
    """A condition for branching or filtering."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Left side (value or expression)
    left: str = ""  # Expression like "{{ trigger.data.status }}"

    # Operator
    operator: ConditionOperator = ConditionOperator.EQUALS

    # Right side (value or expression)
    right: Any = None

    # Logical combination
    and_conditions: List["Condition"] = field(default_factory=list)
    or_conditions: List["Condition"] = field(default_factory=list)

    # Negation
    negate: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "id": self.id,
            "left": self.left,
            "operator": self.operator.value,
            "right": self.right,
            "negate": self.negate,
        }
        if self.and_conditions:
            result["and"] = [c.to_dict() for c in self.and_conditions]
        if self.or_conditions:
            result["or"] = [c.to_dict() for c in self.or_conditions]
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Condition":
        """Create from dictionary."""
        condition = cls(
            id=data.get("id", str(uuid.uuid4())),
            left=data.get("left", ""),
            operator=ConditionOperator(data.get("operator", "eq")),
            right=data.get("right"),
            negate=data.get("negate", False),
        )
        if "and" in data:
            condition.and_conditions = [cls.from_dict(c) for c in data["and"]]
        if "or" in data:
            condition.or_conditions = [cls.from_dict(c) for c in data["or"]]
        return condition


# === Actions ===


@dataclass
class ActionConfig:
    """Configuration for a workflow action."""
    action_type: ActionType = ActionType.TOOL

    # Tool action
    tool_name: Optional[str] = None
    tool_params: Optional[Dict[str, Any]] = None

    # Agent action
    agent_operation: Optional[str] = None  # spawn, terminate, message
    agent_role: Optional[str] = None
    agent_config: Optional[Dict[str, Any]] = None
    agent_id: Optional[str] = None  # For operations on existing agent

    # Goal action
    goal_operation: Optional[str] = None  # create, pause, resume, abandon
    goal_title: Optional[str] = None
    goal_description: Optional[str] = None
    goal_config: Optional[Dict[str, Any]] = None
    goal_id: Optional[str] = None  # For operations on existing goal

    # Webhook action
    webhook_url: Optional[str] = None
    webhook_method: str = "POST"
    webhook_headers: Optional[Dict[str, str]] = None
    webhook_body: Optional[Dict[str, Any]] = None
    webhook_timeout: float = 30.0

    # Notification action
    notification_channel: Optional[str] = None  # email, slack, webhook, console
    notification_message: Optional[str] = None
    notification_title: Optional[str] = None
    notification_recipients: Optional[List[str]] = None
    notification_metadata: Optional[Dict[str, Any]] = None

    # Data action
    data_operation: Optional[str] = None  # read, write, delete, query
    data_source: Optional[str] = None  # memory, knowledge, state, context
    data_key: Optional[str] = None
    data_value: Optional[Any] = None
    data_query: Optional[Dict[str, Any]] = None

    # Workflow action (sub-workflow)
    sub_workflow_id: Optional[str] = None
    sub_workflow_inputs: Optional[Dict[str, Any]] = None
    wait_for_completion: bool = True

    # Approval action
    approval_message: Optional[str] = None
    approval_title: Optional[str] = None
    approvers: Optional[List[str]] = None
    approval_timeout_hours: float = 24.0
    auto_approve: bool = False
    require_all_approvers: bool = False

    # Delay action
    delay_seconds: Optional[float] = None
    delay_until: Optional[str] = None  # Expression for datetime

    # Transform action
    transform_expression: Optional[str] = None
    transform_output_key: Optional[str] = None

    # LLM action
    llm_prompt: Optional[str] = None
    llm_model: Optional[str] = None
    llm_system_prompt: Optional[str] = None
    llm_temperature: float = 0.7
    llm_max_tokens: Optional[int] = None

    # Script action
    script_language: str = "python"
    script_code: Optional[str] = None
    script_timeout: float = 60.0

    # Parallel action
    parallel_steps: Optional[List[str]] = None  # Step IDs to run in parallel
    parallel_wait_all: bool = True

    # Common
    timeout_seconds: float = 300.0
    retry_count: int = 0
    retry_delay_seconds: float = 5.0
    retry_backoff_multiplier: float = 2.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "type": self.action_type.value,
            "timeout": self.timeout_seconds,
            "retry_count": self.retry_count,
        }

        # Add type-specific fields
        if self.action_type == ActionType.TOOL:
            result["tool_name"] = self.tool_name
            result["tool_params"] = self.tool_params
        elif self.action_type == ActionType.WEBHOOK:
            result["url"] = self.webhook_url
            result["method"] = self.webhook_method
            result["headers"] = self.webhook_headers
            result["body"] = self.webhook_body
        elif self.action_type == ActionType.AGENT:
            result["operation"] = self.agent_operation
            result["role"] = self.agent_role
            result["config"] = self.agent_config
        elif self.action_type == ActionType.GOAL:
            result["operation"] = self.goal_operation
            result["title"] = self.goal_title
            result["config"] = self.goal_config
        elif self.action_type == ActionType.LLM:
            result["prompt"] = self.llm_prompt
            result["model"] = self.llm_model
            result["system_prompt"] = self.llm_system_prompt
        elif self.action_type == ActionType.NOTIFICATION:
            result["channel"] = self.notification_channel
            result["message"] = self.notification_message
            result["recipients"] = self.notification_recipients
        elif self.action_type == ActionType.APPROVAL:
            result["message"] = self.approval_message
            result["approvers"] = self.approvers
            result["timeout_hours"] = self.approval_timeout_hours
        elif self.action_type == ActionType.DELAY:
            result["delay_seconds"] = self.delay_seconds
            result["delay_until"] = self.delay_until
        elif self.action_type == ActionType.DATA:
            result["operation"] = self.data_operation
            result["source"] = self.data_source
            result["key"] = self.data_key
            result["value"] = self.data_value
        elif self.action_type == ActionType.WORKFLOW:
            result["sub_workflow_id"] = self.sub_workflow_id
            result["inputs"] = self.sub_workflow_inputs
            result["wait"] = self.wait_for_completion
        elif self.action_type == ActionType.TRANSFORM:
            result["expression"] = self.transform_expression
            result["output_key"] = self.transform_output_key
        elif self.action_type == ActionType.SCRIPT:
            result["language"] = self.script_language
            result["code"] = self.script_code
        elif self.action_type == ActionType.PARALLEL:
            result["steps"] = self.parallel_steps
            result["wait_all"] = self.parallel_wait_all

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ActionConfig":
        """Create from dictionary."""
        action_type = ActionType(data.get("type", "tool"))
        return cls(
            action_type=action_type,
            tool_name=data.get("tool_name"),
            tool_params=data.get("tool_params"),
            agent_operation=data.get("agent_operation") or data.get("operation"),
            agent_role=data.get("agent_role") or data.get("role"),
            agent_config=data.get("agent_config") or data.get("config"),
            goal_operation=data.get("goal_operation") or data.get("operation"),
            goal_title=data.get("goal_title") or data.get("title"),
            goal_config=data.get("goal_config"),
            webhook_url=data.get("webhook_url") or data.get("url"),
            webhook_method=data.get("webhook_method") or data.get("method", "POST"),
            webhook_headers=data.get("webhook_headers") or data.get("headers"),
            webhook_body=data.get("webhook_body") or data.get("body"),
            notification_channel=data.get("notification_channel") or data.get("channel"),
            notification_message=data.get("notification_message") or data.get("message"),
            notification_recipients=data.get("notification_recipients") or data.get("recipients"),
            data_operation=data.get("data_operation"),
            data_source=data.get("data_source") or data.get("source"),
            data_key=data.get("data_key") or data.get("key"),
            data_value=data.get("data_value") or data.get("value"),
            sub_workflow_id=data.get("sub_workflow_id"),
            sub_workflow_inputs=data.get("sub_workflow_inputs") or data.get("inputs"),
            wait_for_completion=data.get("wait_for_completion", data.get("wait", True)),
            approval_message=data.get("approval_message") or data.get("message"),
            approvers=data.get("approvers"),
            approval_timeout_hours=data.get("approval_timeout_hours", data.get("timeout_hours", 24.0)),
            delay_seconds=data.get("delay_seconds"),
            delay_until=data.get("delay_until"),
            transform_expression=data.get("transform_expression") or data.get("expression"),
            transform_output_key=data.get("transform_output_key") or data.get("output_key"),
            llm_prompt=data.get("llm_prompt") or data.get("prompt"),
            llm_model=data.get("llm_model") or data.get("model"),
            llm_system_prompt=data.get("llm_system_prompt") or data.get("system_prompt"),
            script_language=data.get("script_language", "python"),
            script_code=data.get("script_code") or data.get("code"),
            parallel_steps=data.get("parallel_steps") or data.get("steps"),
            timeout_seconds=data.get("timeout_seconds", data.get("timeout", 300.0)),
            retry_count=data.get("retry_count", 0),
            retry_delay_seconds=data.get("retry_delay_seconds", 5.0),
        )


@dataclass
class WorkflowStep:
    """A step in a workflow."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""

    # Action
    action: ActionConfig = field(default_factory=ActionConfig)

    # Flow control
    condition: Optional[Condition] = None  # Execute only if condition met

    # Branching
    on_success: Optional[str] = None  # Step ID to go to on success
    on_failure: Optional[str] = None  # Step ID to go to on failure

    # Loop
    loop_over: Optional[str] = None  # Expression returning list
    loop_variable: str = "item"
    loop_index_variable: str = "index"
    max_iterations: int = 1000  # Safety limit

    # Parallel branches (for condition action type)
    branches: Optional[List["WorkflowBranch"]] = None

    # Error handling
    continue_on_error: bool = False
    error_handler_step: Optional[str] = None
    max_retries: int = 0

    # Output
    output_variable: Optional[str] = None  # Store output in this variable

    # Metadata
    position: Dict[str, float] = field(default_factory=dict)  # For visual editor
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "action": self.action.to_dict(),
            "on_success": self.on_success,
            "on_failure": self.on_failure,
            "continue_on_error": self.continue_on_error,
        }
        if self.condition:
            result["condition"] = self.condition.to_dict()
        if self.loop_over:
            result["loop_over"] = self.loop_over
            result["loop_variable"] = self.loop_variable
        if self.output_variable:
            result["output_variable"] = self.output_variable
        if self.branches:
            result["branches"] = [b.to_dict() for b in self.branches]
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowStep":
        """Create from dictionary."""
        step = cls(
            id=data.get("id", str(uuid.uuid4())),
            name=data.get("name", ""),
            description=data.get("description", ""),
            action=ActionConfig.from_dict(data.get("action", {})),
            on_success=data.get("on_success"),
            on_failure=data.get("on_failure"),
            loop_over=data.get("loop_over"),
            loop_variable=data.get("loop_variable", "item"),
            continue_on_error=data.get("continue_on_error", False),
            error_handler_step=data.get("error_handler_step"),
            output_variable=data.get("output_variable"),
            position=data.get("position", {}),
            tags=data.get("tags", []),
        )
        if "condition" in data:
            step.condition = Condition.from_dict(data["condition"])
        if "branches" in data:
            step.branches = [WorkflowBranch.from_dict(b) for b in data["branches"]]
        return step


@dataclass
class WorkflowBranch:
    """A conditional branch in a workflow."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    condition: Optional[Condition] = None  # None = default branch
    target_step_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "condition": self.condition.to_dict() if self.condition else None,
            "target_step_id": self.target_step_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowBranch":
        """Create from dictionary."""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            name=data.get("name", ""),
            condition=Condition.from_dict(data["condition"]) if data.get("condition") else None,
            target_step_id=data.get("target_step_id", ""),
        )


# === Workflow Definition ===


@dataclass
class Workflow:
    """A workflow definition."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Identity
    name: str = ""
    description: str = ""
    version: int = 1

    # Owner
    owner_id: str = ""
    tenant_id: Optional[str] = None

    # Triggers
    triggers: List[TriggerConfig] = field(default_factory=list)

    # Steps
    steps: List[WorkflowStep] = field(default_factory=list)
    entry_step_id: Optional[str] = None  # First step to execute

    # Inputs/Outputs
    input_schema: Dict[str, Any] = field(default_factory=dict)  # JSON Schema
    output_schema: Dict[str, Any] = field(default_factory=dict)
    default_inputs: Dict[str, Any] = field(default_factory=dict)

    # Settings
    timeout_seconds: float = 3600.0  # 1 hour default
    max_concurrent_executions: int = 10
    max_retries: int = 0
    retry_delay_seconds: float = 60.0

    # Status
    status: WorkflowStatus = WorkflowStatus.DRAFT

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    # Metadata
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Statistics
    execution_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    avg_duration_ms: float = 0.0

    def get_step(self, step_id: str) -> Optional[WorkflowStep]:
        """Get a step by ID."""
        for step in self.steps:
            if step.id == step_id:
                return step
        return None

    def get_entry_step(self) -> Optional[WorkflowStep]:
        """Get the entry step."""
        if self.entry_step_id:
            return self.get_step(self.entry_step_id)
        if self.steps:
            return self.steps[0]
        return None

    def validate(self) -> List[str]:
        """Validate workflow definition. Returns list of errors."""
        errors = []

        if not self.name:
            errors.append("Workflow name is required")

        if not self.steps:
            errors.append("Workflow must have at least one step")

        # Check entry step exists
        if self.entry_step_id:
            if not self.get_step(self.entry_step_id):
                errors.append(f"Entry step not found: {self.entry_step_id}")

        # Check step references
        step_ids = {s.id for s in self.steps}
        for step in self.steps:
            if step.on_success and step.on_success not in step_ids:
                errors.append(f"Step {step.id} references invalid on_success: {step.on_success}")
            if step.on_failure and step.on_failure not in step_ids:
                errors.append(f"Step {step.id} references invalid on_failure: {step.on_failure}")
            if step.error_handler_step and step.error_handler_step not in step_ids:
                errors.append(f"Step {step.id} references invalid error_handler: {step.error_handler_step}")

        return errors

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "owner_id": self.owner_id,
            "tenant_id": self.tenant_id,
            "status": self.status.value,
            "triggers": [t.to_dict() for t in self.triggers],
            "steps": [s.to_dict() for s in self.steps],
            "entry_step_id": self.entry_step_id,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "default_inputs": self.default_inputs,
            "timeout_seconds": self.timeout_seconds,
            "max_concurrent_executions": self.max_concurrent_executions,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "tags": self.tags,
            "metadata": self.metadata,
            "execution_count": self.execution_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Workflow":
        """Create from dictionary."""
        workflow = cls(
            id=data.get("id", str(uuid.uuid4())),
            name=data.get("name", ""),
            description=data.get("description", ""),
            version=data.get("version", 1),
            owner_id=data.get("owner_id", ""),
            tenant_id=data.get("tenant_id"),
            status=WorkflowStatus(data.get("status", "draft")),
            entry_step_id=data.get("entry_step_id"),
            input_schema=data.get("input_schema", {}),
            output_schema=data.get("output_schema", {}),
            default_inputs=data.get("default_inputs", {}),
            timeout_seconds=data.get("timeout_seconds", 3600.0),
            max_concurrent_executions=data.get("max_concurrent_executions", 10),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
        )

        # Parse triggers
        for trigger_data in data.get("triggers", []):
            workflow.triggers.append(TriggerConfig.from_dict(trigger_data))

        # Parse steps
        for step_data in data.get("steps", []):
            workflow.steps.append(WorkflowStep.from_dict(step_data))

        # Parse timestamps
        if "created_at" in data:
            if isinstance(data["created_at"], str):
                workflow.created_at = datetime.fromisoformat(data["created_at"])
            else:
                workflow.created_at = data["created_at"]
        if "updated_at" in data:
            if isinstance(data["updated_at"], str):
                workflow.updated_at = datetime.fromisoformat(data["updated_at"])
            else:
                workflow.updated_at = data["updated_at"]

        return workflow


# === Execution Types ===


@dataclass
class StepResult:
    """Result of executing a workflow step."""
    step_id: str = ""
    step_name: str = ""
    status: ExecutionStatus = ExecutionStatus.COMPLETED

    # Output
    output: Any = None
    error: Optional[str] = None
    error_type: Optional[str] = None

    # Timing
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    duration_ms: float = 0.0

    # Retries
    attempt: int = 1
    max_attempts: int = 1

    # Loop info
    loop_index: Optional[int] = None
    loop_item: Any = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def complete(self, output: Any = None) -> None:
        """Mark step as completed."""
        self.status = ExecutionStatus.COMPLETED
        self.output = output
        self.completed_at = datetime.now()
        self.duration_ms = (self.completed_at - self.started_at).total_seconds() * 1000

    def fail(self, error: str, error_type: str = None) -> None:
        """Mark step as failed."""
        self.status = ExecutionStatus.FAILED
        self.error = error
        self.error_type = error_type
        self.completed_at = datetime.now()
        self.duration_ms = (self.completed_at - self.started_at).total_seconds() * 1000

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "step_id": self.step_id,
            "step_name": self.step_name,
            "status": self.status.value,
            "output": self.output,
            "error": self.error,
            "error_type": self.error_type,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_ms": self.duration_ms,
            "attempt": self.attempt,
            "max_attempts": self.max_attempts,
            "loop_index": self.loop_index,
        }


@dataclass
class WorkflowExecution:
    """An instance of workflow execution."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    workflow_id: str = ""
    workflow_name: str = ""
    workflow_version: int = 1

    # Trigger info
    trigger_id: Optional[str] = None
    trigger_type: Optional[TriggerType] = None
    trigger_data: Dict[str, Any] = field(default_factory=dict)

    # Inputs
    inputs: Dict[str, Any] = field(default_factory=dict)

    # State
    status: ExecutionStatus = ExecutionStatus.PENDING
    current_step_id: Optional[str] = None
    current_step_name: Optional[str] = None

    # Results
    step_results: Dict[str, StepResult] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    error_step_id: Optional[str] = None

    # Context (shared state during execution)
    context: Dict[str, Any] = field(default_factory=dict)

    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_ms: float = 0.0

    # Retry info
    attempt: int = 1
    max_attempts: int = 1

    # Owner
    initiated_by: Optional[str] = None  # User ID or "system"
    tenant_id: Optional[str] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    def start(self) -> None:
        """Mark execution as started."""
        self.status = ExecutionStatus.RUNNING
        self.started_at = datetime.now()

    def complete(self, outputs: Dict[str, Any] = None) -> None:
        """Mark execution as completed."""
        self.status = ExecutionStatus.COMPLETED
        self.outputs = outputs or {}
        self.completed_at = datetime.now()
        if self.started_at:
            self.duration_ms = (self.completed_at - self.started_at).total_seconds() * 1000
        self.current_step_id = None
        self.current_step_name = None

    def fail(self, error: str, step_id: str = None) -> None:
        """Mark execution as failed."""
        self.status = ExecutionStatus.FAILED
        self.error = error
        self.error_step_id = step_id
        self.completed_at = datetime.now()
        if self.started_at:
            self.duration_ms = (self.completed_at - self.started_at).total_seconds() * 1000

    def cancel(self) -> None:
        """Mark execution as cancelled."""
        self.status = ExecutionStatus.CANCELLED
        self.completed_at = datetime.now()
        if self.started_at:
            self.duration_ms = (self.completed_at - self.started_at).total_seconds() * 1000

    def is_terminal(self) -> bool:
        """Check if execution is in a terminal state."""
        return self.status in (
            ExecutionStatus.COMPLETED,
            ExecutionStatus.FAILED,
            ExecutionStatus.CANCELLED,
            ExecutionStatus.TIMED_OUT,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "workflow_id": self.workflow_id,
            "workflow_name": self.workflow_name,
            "workflow_version": self.workflow_version,
            "trigger_type": self.trigger_type.value if self.trigger_type else None,
            "trigger_data": self.trigger_data,
            "inputs": self.inputs,
            "status": self.status.value,
            "current_step_id": self.current_step_id,
            "current_step_name": self.current_step_name,
            "step_results": {k: v.to_dict() for k, v in self.step_results.items()},
            "outputs": self.outputs,
            "error": self.error,
            "error_step_id": self.error_step_id,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_ms": self.duration_ms,
            "attempt": self.attempt,
            "initiated_by": self.initiated_by,
            "tenant_id": self.tenant_id,
            "metadata": self.metadata,
            "tags": self.tags,
        }


# === Approval Types ===


@dataclass
class ApprovalRequest:
    """A human approval request."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Context
    execution_id: str = ""
    step_id: str = ""
    workflow_id: str = ""
    workflow_name: str = ""

    # Request
    title: str = ""
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    # Approvers
    approvers: List[str] = field(default_factory=list)
    requires_all: bool = False  # Require all approvers

    # Status
    status: ApprovalStatus = ApprovalStatus.PENDING

    # Responses
    responses: List["ApprovalResponse"] = field(default_factory=list)

    # Response (final decision)
    responded_by: Optional[str] = None
    response_message: Optional[str] = None

    # Timing
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    responded_at: Optional[datetime] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """Check if request is expired."""
        if self.expires_at and datetime.now() > self.expires_at:
            return True
        return False

    def approve(self, approver: str, message: str = "") -> None:
        """Approve the request."""
        self.responses.append(ApprovalResponse(
            approver=approver,
            decision="approved",
            message=message,
        ))

        # Check if we have enough approvals
        approvals = [r for r in self.responses if r.decision == "approved"]
        if self.requires_all:
            if len(approvals) >= len(self.approvers):
                self.status = ApprovalStatus.APPROVED
                self.responded_by = approver
                self.response_message = message
                self.responded_at = datetime.now()
        else:
            self.status = ApprovalStatus.APPROVED
            self.responded_by = approver
            self.response_message = message
            self.responded_at = datetime.now()

    def reject(self, approver: str, message: str = "") -> None:
        """Reject the request."""
        self.responses.append(ApprovalResponse(
            approver=approver,
            decision="rejected",
            message=message,
        ))
        self.status = ApprovalStatus.REJECTED
        self.responded_by = approver
        self.response_message = message
        self.responded_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "execution_id": self.execution_id,
            "step_id": self.step_id,
            "workflow_id": self.workflow_id,
            "workflow_name": self.workflow_name,
            "title": self.title,
            "message": self.message,
            "details": self.details,
            "approvers": self.approvers,
            "requires_all": self.requires_all,
            "status": self.status.value,
            "responses": [r.to_dict() for r in self.responses],
            "responded_by": self.responded_by,
            "response_message": self.response_message,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "responded_at": self.responded_at.isoformat() if self.responded_at else None,
        }


@dataclass
class ApprovalResponse:
    """A single approval response."""
    approver: str = ""
    decision: str = ""  # approved, rejected
    message: str = ""
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "approver": self.approver,
            "decision": self.decision,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
        }


# === Template Types ===


@dataclass
class WorkflowTemplate:
    """A reusable workflow template."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    category: str = ""

    # Template workflow
    workflow: Workflow = field(default_factory=Workflow)

    # Parameters (customizable values)
    parameters: Dict[str, Any] = field(default_factory=dict)
    parameter_schema: Dict[str, Any] = field(default_factory=dict)  # JSON Schema

    # Metadata
    author: str = ""
    version: str = "1.0.0"
    tags: List[str] = field(default_factory=list)
    icon: str = ""

    # Statistics
    usage_count: int = 0
    rating: float = 0.0

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def instantiate(self, name: str, parameters: Dict[str, Any] = None) -> Workflow:
        """Create a workflow instance from this template."""
        import copy

        # Deep copy the workflow
        workflow = copy.deepcopy(self.workflow)
        workflow.id = str(uuid.uuid4())
        workflow.name = name
        workflow.version = 1
        workflow.status = WorkflowStatus.DRAFT
        workflow.created_at = datetime.now()
        workflow.updated_at = datetime.now()
        workflow.metadata["template_id"] = self.id
        workflow.metadata["template_name"] = self.name

        # Apply parameters (would need proper implementation)
        if parameters:
            workflow.metadata["template_parameters"] = parameters

        return workflow

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "workflow": self.workflow.to_dict(),
            "parameters": self.parameters,
            "parameter_schema": self.parameter_schema,
            "author": self.author,
            "version": self.version,
            "tags": self.tags,
            "icon": self.icon,
            "usage_count": self.usage_count,
            "rating": self.rating,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


# === Event Types ===


@dataclass
class WorkflowEvent:
    """An event in the workflow system."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str = ""
    source: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    # Optional context
    workflow_id: Optional[str] = None
    execution_id: Optional[str] = None
    step_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "event_type": self.event_type,
            "source": self.source,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "workflow_id": self.workflow_id,
            "execution_id": self.execution_id,
            "step_id": self.step_id,
        }
