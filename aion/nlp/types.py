"""
AION Natural Language Programming - Core Type System

SOTA type definitions using algebraic data types, protocol-based polymorphism,
and rich semantic modeling for the intent-to-system compilation pipeline.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, FrozenSet, List, Optional, Protocol, Set, Tuple, Union


def _utcnow() -> datetime:
    """Return timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


# =============================================================================
# Enumerations
# =============================================================================


class IntentType(str, Enum):
    """Types of user intents with semantic grouping."""

    # Creation intents
    CREATE_TOOL = "create_tool"
    CREATE_WORKFLOW = "create_workflow"
    CREATE_AGENT = "create_agent"
    CREATE_API = "create_api"
    CREATE_INTEGRATION = "create_integration"
    CREATE_FUNCTION = "create_function"

    # Modification intents
    MODIFY_EXISTING = "modify_existing"
    EXTEND = "extend"
    COMPOSE = "compose"

    # Lifecycle intents
    DELETE = "delete"
    DEPLOY = "deploy"
    ROLLBACK = "rollback"
    PAUSE = "pause"
    RESUME = "resume"

    # Query intents
    QUERY = "query"
    EXPLAIN = "explain"
    DEBUG = "debug"
    TEST = "test"
    LIST = "list"
    STATUS = "status"

    @property
    def is_creation(self) -> bool:
        return self.value.startswith("create_")

    @property
    def is_modification(self) -> bool:
        return self in (self.MODIFY_EXISTING, self.EXTEND, self.COMPOSE)

    @property
    def is_destructive(self) -> bool:
        return self in (self.DELETE, self.ROLLBACK)

    @property
    def requires_synthesis(self) -> bool:
        return self.is_creation or self.is_modification


class EntityType(str, Enum):
    """Types of entities extracted from user requests."""

    # Naming
    TOOL_NAME = "tool_name"
    WORKFLOW_NAME = "workflow_name"
    AGENT_NAME = "agent_name"
    API_NAME = "api_name"
    SYSTEM_REFERENCE = "system_reference"

    # Data flow
    DATA_SOURCE = "data_source"
    DATA_TARGET = "data_target"
    DATA_FORMAT = "data_format"
    DATA_FIELD = "data_field"

    # Control flow
    TRIGGER = "trigger"
    CONDITION = "condition"
    ACTION = "action"
    SCHEDULE = "schedule"

    # Parameters
    PARAMETER = "parameter"
    PARAMETER_TYPE = "parameter_type"
    DEFAULT_VALUE = "default_value"
    CONSTRAINT = "constraint"

    # Integration
    API_ENDPOINT = "api_endpoint"
    API_METHOD = "api_method"
    AUTH_TYPE = "auth_type"
    SERVICE_NAME = "service_name"

    # Behavioral
    ERROR_HANDLING = "error_handling"
    RETRY_POLICY = "retry_policy"
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"

    # Descriptive
    DESCRIPTION = "description"
    EXAMPLE = "example"
    FORMAT_SPEC = "format_spec"


class SpecificationType(str, Enum):
    """Types of specifications the system can generate."""

    TOOL = "tool"
    WORKFLOW = "workflow"
    AGENT = "agent"
    API = "api"
    INTEGRATION = "integration"
    FUNCTION = "function"
    COMPOSITE = "composite"


class ValidationStatus(str, Enum):
    """Validation pipeline status."""

    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"


class DeploymentStatus(str, Enum):
    """Deployment lifecycle status."""

    DRAFT = "draft"
    VALIDATING = "validating"
    VALIDATED = "validated"
    DEPLOYING = "deploying"
    ACTIVE = "active"
    PAUSED = "paused"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    DEPRECATED = "deprecated"


class Complexity(str, Enum):
    """Task complexity classification."""

    TRIVIAL = "trivial"
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    EXPERT = "expert"


class SafetyLevel(str, Enum):
    """Safety classification for generated code."""

    SAFE = "safe"
    LOW_RISK = "low_risk"
    MEDIUM_RISK = "medium_risk"
    HIGH_RISK = "high_risk"
    DANGEROUS = "dangerous"


# =============================================================================
# Core Data Types
# =============================================================================


@dataclass(frozen=True)
class Entity:
    """An extracted entity from user input (immutable for safety)."""

    type: EntityType
    value: str
    confidence: float = 1.0
    span_start: int = 0
    span_end: int = 0
    metadata: Tuple[Tuple[str, Any], ...] = ()

    @property
    def span(self) -> Tuple[int, int]:
        return (self.span_start, self.span_end)

    def meta(self, key: str, default: Any = None) -> Any:
        for k, v in self.metadata:
            if k == key:
                return v
        return default

    def with_confidence(self, confidence: float) -> Entity:
        return Entity(
            type=self.type,
            value=self.value,
            confidence=confidence,
            span_start=self.span_start,
            span_end=self.span_end,
            metadata=self.metadata,
        )


@dataclass
class Intent:
    """Parsed user intent with full semantic analysis."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: IntentType = IntentType.CREATE_TOOL
    confidence: float = 0.0
    complexity: Complexity = Complexity.SIMPLE

    # Extracted information
    entities: List[Entity] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)

    # Original input
    raw_input: str = ""
    normalized_input: str = ""

    # Multi-intent support (e.g., "create a tool and a workflow")
    sub_intents: List[Intent] = field(default_factory=list)

    # Clarification
    needs_clarification: bool = False
    clarification_questions: List[str] = field(default_factory=list)
    ambiguity_score: float = 0.0

    # Context
    context_references: List[str] = field(default_factory=list)
    depends_on: List[str] = field(default_factory=list)

    # Ensemble predictions (SOTA: keep all classifier votes)
    predictions: List[Dict[str, Any]] = field(default_factory=list)

    def get_entity(self, entity_type: EntityType) -> Optional[Entity]:
        for entity in self.entities:
            if entity.type == entity_type:
                return entity
        return None

    def get_entities(self, entity_type: EntityType) -> List[Entity]:
        return [e for e in self.entities if e.type == entity_type]

    def get_all_entity_values(self, entity_type: EntityType) -> List[str]:
        return [e.value for e in self.entities if e.type == entity_type]

    @property
    def name(self) -> Optional[str]:
        """Get the primary name from entities or parameters."""
        name_types = [
            EntityType.TOOL_NAME,
            EntityType.WORKFLOW_NAME,
            EntityType.AGENT_NAME,
            EntityType.API_NAME,
        ]
        for nt in name_types:
            entity = self.get_entity(nt)
            if entity:
                return entity.value
        return self.parameters.get("name")

    @property
    def description(self) -> Optional[str]:
        entity = self.get_entity(EntityType.DESCRIPTION)
        if entity:
            return entity.value
        return self.parameters.get("description")

    @property
    def is_compound(self) -> bool:
        return len(self.sub_intents) > 0

    @property
    def fingerprint(self) -> str:
        """Content-based hash for deduplication."""
        content = f"{self.type.value}:{self.raw_input}:{sorted([(e.type.value, e.value) for e in self.entities])}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


# =============================================================================
# Specification Types
# =============================================================================


@dataclass
class ParameterSpec:
    """Specification for a single parameter."""

    name: str
    type: str = "string"
    description: str = ""
    required: bool = True
    default: Any = None
    constraints: Dict[str, Any] = field(default_factory=dict)
    examples: List[Any] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "name": self.name,
            "type": self.type,
            "description": self.description,
            "required": self.required,
        }
        if self.default is not None:
            d["default"] = self.default
        if self.constraints:
            d["constraints"] = self.constraints
        return d


@dataclass
class ToolSpecification:
    """Specification for a tool."""

    name: str
    description: str

    # Parameters
    parameters: List[ParameterSpec] = field(default_factory=list)

    # Return type
    return_type: str = "Any"
    return_description: str = ""

    # Implementation hints
    implementation_notes: str = ""
    dependencies: List[str] = field(default_factory=list)
    estimated_complexity: Complexity = Complexity.SIMPLE

    # API integration
    api_endpoint: Optional[str] = None
    api_method: str = "GET"
    api_headers: Dict[str, str] = field(default_factory=dict)

    # Authentication
    auth_required: bool = False
    auth_type: Optional[str] = None

    # Rate limiting
    rate_limit: Optional[float] = None

    # Error handling
    retry_on_failure: bool = True
    max_retries: int = 3
    timeout_seconds: float = 30.0

    # Idempotency
    idempotent: bool = False
    cache_ttl: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": [p.to_dict() for p in self.parameters],
            "required": [p.name for p in self.parameters if p.required],
            "return_type": self.return_type,
            "return_description": self.return_description,
            "api_endpoint": self.api_endpoint,
            "api_method": self.api_method,
            "auth_required": self.auth_required,
            "retry_on_failure": self.retry_on_failure,
            "idempotent": self.idempotent,
        }


@dataclass
class WorkflowStep:
    """A single step in a workflow."""

    id: str
    name: str
    action: str
    params: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)
    condition: Optional[str] = None
    on_error: str = "stop"
    timeout_seconds: float = 60.0
    retry_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "action": self.action,
            "params": self.params,
            "depends_on": self.depends_on,
            "condition": self.condition,
            "on_error": self.on_error,
        }


@dataclass
class WorkflowSpecification:
    """Specification for a workflow with DAG-based execution."""

    name: str
    description: str

    # Trigger
    trigger_type: str = "manual"
    trigger_config: Dict[str, Any] = field(default_factory=dict)

    # Steps (DAG)
    steps: List[WorkflowStep] = field(default_factory=list)

    # Conditions and branching
    conditions: List[Dict[str, Any]] = field(default_factory=list)

    # Error handling
    on_error: str = "stop"
    max_retries: int = 3
    retry_backoff: str = "exponential"

    # Inputs/Outputs
    inputs: List[ParameterSpec] = field(default_factory=list)
    outputs: List[Dict[str, Any]] = field(default_factory=list)

    # Concurrency
    max_parallel_steps: int = 5

    # Timeout
    timeout_seconds: float = 300.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "trigger_type": self.trigger_type,
            "trigger_config": self.trigger_config,
            "steps": [s.to_dict() for s in self.steps],
            "on_error": self.on_error,
            "max_retries": self.max_retries,
            "inputs": [i.to_dict() for i in self.inputs],
            "max_parallel_steps": self.max_parallel_steps,
        }


@dataclass
class AgentSpecification:
    """Specification for an autonomous agent."""

    name: str
    description: str

    # Personality and behavior
    system_prompt: str = ""
    personality_traits: List[str] = field(default_factory=list)
    communication_style: str = "professional"

    # Capabilities
    allowed_tools: List[str] = field(default_factory=list)
    allowed_actions: List[str] = field(default_factory=list)
    required_capabilities: List[str] = field(default_factory=list)

    # Goals (hierarchical)
    primary_goal: str = ""
    sub_goals: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)

    # Constraints
    constraints: List[str] = field(default_factory=list)
    forbidden_actions: List[str] = field(default_factory=list)

    # Resource limits
    max_iterations: int = 100
    timeout_seconds: float = 300.0
    max_tokens_per_turn: int = 4096
    max_tool_calls: int = 50

    # Memory
    memory_enabled: bool = True
    memory_types: List[str] = field(default_factory=lambda: ["episodic", "semantic"])
    context_window_strategy: str = "sliding"

    # Collaboration
    can_delegate: bool = False
    can_collaborate: bool = False
    delegation_targets: List[str] = field(default_factory=list)

    # Learning
    learn_from_feedback: bool = True
    adaptation_rate: float = 0.1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "system_prompt": self.system_prompt,
            "personality_traits": self.personality_traits,
            "allowed_tools": self.allowed_tools,
            "primary_goal": self.primary_goal,
            "sub_goals": self.sub_goals,
            "constraints": self.constraints,
            "max_iterations": self.max_iterations,
            "memory_enabled": self.memory_enabled,
        }


@dataclass
class APIEndpointSpec:
    """Specification for a single API endpoint."""

    path: str
    method: str = "GET"
    description: str = ""
    parameters: List[ParameterSpec] = field(default_factory=list)
    request_body: Optional[Dict[str, Any]] = None
    response_schema: Optional[Dict[str, Any]] = None
    status_codes: Dict[int, str] = field(default_factory=lambda: {200: "Success"})
    auth_required: bool = False
    rate_limit: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "method": self.method,
            "description": self.description,
            "parameters": [p.to_dict() for p in self.parameters],
            "request_body": self.request_body,
            "response_schema": self.response_schema,
            "auth_required": self.auth_required,
        }


@dataclass
class APISpecification:
    """Specification for a complete API."""

    name: str
    description: str
    base_path: str = "/api"
    version: str = "v1"

    # Endpoints
    endpoints: List[APIEndpointSpec] = field(default_factory=list)

    # Models
    models: List[Dict[str, Any]] = field(default_factory=list)

    # Auth
    auth_type: Optional[str] = None
    auth_config: Dict[str, Any] = field(default_factory=dict)

    # Rate limiting
    rate_limit: Optional[int] = None

    # CORS
    cors_origins: List[str] = field(default_factory=lambda: ["*"])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "base_path": self.base_path,
            "version": self.version,
            "endpoints": [e.to_dict() for e in self.endpoints],
            "models": self.models,
            "auth_type": self.auth_type,
        }


@dataclass
class IntegrationSpecification:
    """Specification for a system integration."""

    name: str
    description: str

    # Source and target
    source_system: str = ""
    source_config: Dict[str, Any] = field(default_factory=dict)
    target_system: str = ""
    target_config: Dict[str, Any] = field(default_factory=dict)

    # Data mapping
    field_mapping: List[Dict[str, Any]] = field(default_factory=list)
    transform_rules: List[Dict[str, Any]] = field(default_factory=list)

    # Sync configuration
    sync_mode: str = "incremental"
    sync_direction: str = "one_way"
    sync_schedule: Optional[str] = None

    # Error handling
    on_conflict: str = "skip"
    on_error: str = "retry"
    max_retries: int = 3

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "source_system": self.source_system,
            "target_system": self.target_system,
            "field_mapping": self.field_mapping,
            "sync_mode": self.sync_mode,
            "sync_direction": self.sync_direction,
        }


# Union type for all specifications
Specification = Union[
    ToolSpecification,
    WorkflowSpecification,
    AgentSpecification,
    APISpecification,
    IntegrationSpecification,
]


# =============================================================================
# Generated Code Types
# =============================================================================


@dataclass
class CodeArtifact:
    """A single generated code artifact."""

    filename: str
    code: str
    language: str = "python"
    artifact_type: str = "source"
    dependencies: List[str] = field(default_factory=list)

    @property
    def line_count(self) -> int:
        return len(self.code.strip().splitlines())


@dataclass
class GeneratedCode:
    """Complete generated code bundle with all artifacts."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Primary code
    language: str = "python"
    code: str = ""

    # Metadata
    filename: str = ""
    spec_type: SpecificationType = SpecificationType.TOOL

    # Dependencies
    imports: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)

    # Tests
    test_code: str = ""

    # Documentation
    docstring: str = ""

    # Additional artifacts (e.g., configs, migrations, schemas)
    artifacts: List[CodeArtifact] = field(default_factory=list)

    # Generation metadata
    generation_model: str = ""
    generation_temperature: float = 0.0
    generation_timestamp: datetime = field(default_factory=_utcnow)
    iteration: int = 1

    @property
    def all_code(self) -> str:
        parts = [self.code]
        for artifact in self.artifacts:
            parts.append(f"\n# --- {artifact.filename} ---\n{artifact.code}")
        return "\n".join(parts)

    @property
    def line_count(self) -> int:
        """Count lines of primary code."""
        return len(self.code.strip().splitlines()) if self.code.strip() else 0

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.code.encode()).hexdigest()[:16]


# =============================================================================
# Validation Types
# =============================================================================


@dataclass
class ValidationIssue:
    """A single validation issue."""

    severity: str  # "error", "warning", "info"
    message: str
    line: Optional[int] = None
    column: Optional[int] = None
    rule: str = ""
    suggestion: str = ""

    @property
    def is_error(self) -> bool:
        return self.severity == "error"


@dataclass
class ValidationResult:
    """Result of validation pipeline."""

    status: ValidationStatus = ValidationStatus.PENDING

    # Issues
    issues: List[ValidationIssue] = field(default_factory=list)

    # Test results
    tests_passed: int = 0
    tests_failed: int = 0
    tests_skipped: int = 0
    test_details: List[Dict[str, Any]] = field(default_factory=list)

    # Safety analysis
    safety_level: SafetyLevel = SafetyLevel.SAFE
    safety_score: float = 1.0
    safety_concerns: List[str] = field(default_factory=list)

    # Performance
    validation_time_ms: float = 0.0

    @property
    def errors(self) -> List[str]:
        return [i.message for i in self.issues if i.is_error]

    @property
    def warnings(self) -> List[str]:
        return [i.message for i in self.issues if i.severity == "warning"]

    @property
    def suggestions(self) -> List[str]:
        return [i.suggestion for i in self.issues if i.suggestion]

    @property
    def is_valid(self) -> bool:
        return self.status == ValidationStatus.PASSED and not self.errors

    @property
    def is_safe(self) -> bool:
        return self.safety_level in (SafetyLevel.SAFE, SafetyLevel.LOW_RISK)

    def add_error(self, message: str, **kwargs: Any) -> None:
        self.issues.append(ValidationIssue(severity="error", message=message, **kwargs))

    def add_warning(self, message: str, **kwargs: Any) -> None:
        self.issues.append(ValidationIssue(severity="warning", message=message, **kwargs))

    def merge(self, other: ValidationResult) -> None:
        """Merge another validation result into this one."""
        self.issues.extend(other.issues)
        self.tests_passed += other.tests_passed
        self.tests_failed += other.tests_failed
        self.tests_skipped += other.tests_skipped
        self.test_details.extend(other.test_details)
        self.safety_concerns.extend(other.safety_concerns)
        self.safety_score = min(self.safety_score, other.safety_score)
        # Recalculate safety level from merged score
        if self.safety_score >= 0.9:
            self.safety_level = SafetyLevel.SAFE
        elif self.safety_score >= 0.7:
            self.safety_level = SafetyLevel.LOW_RISK
        elif self.safety_score >= 0.5:
            self.safety_level = SafetyLevel.MEDIUM_RISK
        elif self.safety_score >= 0.25:
            self.safety_level = SafetyLevel.HIGH_RISK
        else:
            self.safety_level = SafetyLevel.DANGEROUS
        # Downgrade status if needed
        severity_order = [
            ValidationStatus.PASSED,
            ValidationStatus.WARNING,
            ValidationStatus.FAILED,
        ]
        if other.status in severity_order and self.status in severity_order:
            if severity_order.index(other.status) > severity_order.index(self.status):
                self.status = other.status


# =============================================================================
# Deployment Types
# =============================================================================


@dataclass
class DeploymentRecord:
    """Record of a specific deployment version."""

    version: int
    code_fingerprint: str
    deployed_at: datetime = field(default_factory=_utcnow)
    deployed_by: str = ""
    change_summary: str = ""
    rollback_safe: bool = True


@dataclass
class DeployedSystem:
    """A deployed system with full lifecycle tracking."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""

    # Type and spec
    system_type: SpecificationType = SpecificationType.TOOL
    specification: Optional[Any] = None

    # Code
    generated_code: GeneratedCode = field(default_factory=GeneratedCode)
    validation_result: Optional[ValidationResult] = None

    # Status
    status: DeploymentStatus = DeploymentStatus.DRAFT
    version: int = 1

    # Version history
    deployment_history: List[DeploymentRecord] = field(default_factory=list)

    # Timestamps
    created_at: datetime = field(default_factory=_utcnow)
    updated_at: datetime = field(default_factory=_utcnow)
    created_by: str = ""

    # Metrics
    invocation_count: int = 0
    error_count: int = 0
    success_count: int = 0
    avg_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    _latency_reservoir: List[float] = field(default_factory=list, repr=False)

    # Tags
    tags: List[str] = field(default_factory=list)

    # Max reservoir size for p99 calculation
    _RESERVOIR_SIZE: int = field(default=1000, repr=False, init=False)

    @property
    def error_rate(self) -> float:
        total = self.invocation_count
        if total == 0:
            return 0.0
        return self.error_count / total

    @property
    def is_active(self) -> bool:
        return self.status == DeploymentStatus.ACTIVE

    def record_invocation(self, latency_ms: float, success: bool) -> None:
        """Record an invocation for metrics tracking."""
        self.invocation_count += 1
        if success:
            self.success_count += 1
        else:
            self.error_count += 1

        # Exponential moving average for latency
        alpha = 0.1
        self.avg_latency_ms = alpha * latency_ms + (1 - alpha) * self.avg_latency_ms

        # Reservoir sampling for true p99 calculation
        self._latency_reservoir.append(latency_ms)
        if len(self._latency_reservoir) > self._RESERVOIR_SIZE:
            self._latency_reservoir = self._latency_reservoir[-self._RESERVOIR_SIZE:]
        if self._latency_reservoir:
            sorted_latencies = sorted(self._latency_reservoir)
            p99_index = min(len(sorted_latencies) - 1, int(len(sorted_latencies) * 0.99))
            self.p99_latency_ms = sorted_latencies[p99_index]

        self.updated_at = _utcnow()


# =============================================================================
# Session Types
# =============================================================================


@dataclass
class ConversationMessage:
    """A single message in a programming session."""

    role: str  # "user", "assistant", "system"
    content: str
    timestamp: datetime = field(default_factory=_utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class ProgrammingSession:
    """A natural language programming session with full state tracking."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""

    # Current work
    current_intent: Optional[Intent] = None
    current_spec: Optional[Any] = None
    current_code: Optional[GeneratedCode] = None
    current_validation: Optional[ValidationResult] = None

    # History
    messages: List[ConversationMessage] = field(default_factory=list)
    intent_history: List[Intent] = field(default_factory=list)
    iterations: int = 0

    # Context
    context: Dict[str, Any] = field(default_factory=dict)
    referenced_systems: List[str] = field(default_factory=list)
    active_suggestions: List[str] = field(default_factory=list)

    # Session state
    state: str = "active"  # active, paused, completed, abandoned

    # Timestamps
    started_at: datetime = field(default_factory=_utcnow)
    last_activity: datetime = field(default_factory=_utcnow)

    def add_message(self, role: str, content: str, **metadata: Any) -> None:
        self.messages.append(ConversationMessage(
            role=role,
            content=content,
            metadata=metadata,
        ))
        self.last_activity = _utcnow()

    def get_context_window(self, max_messages: int = 20) -> List[ConversationMessage]:
        """Get recent messages for context."""
        return self.messages[-max_messages:]

    @property
    def duration_seconds(self) -> float:
        return (self.last_activity - self.started_at).total_seconds()


# =============================================================================
# Protocol Definitions (for dependency injection)
# =============================================================================


class SynthesizerProtocol(Protocol):
    """Protocol that all synthesizers must implement."""

    async def synthesize(self, spec: Any) -> GeneratedCode: ...


class ValidatorProtocol(Protocol):
    """Protocol that all validators must implement."""

    async def validate(self, code: GeneratedCode) -> ValidationResult: ...


class DeployerProtocol(Protocol):
    """Protocol that all deployers must implement."""

    async def deploy(
        self, code: GeneratedCode, spec: Any, user_id: str
    ) -> DeployedSystem: ...
