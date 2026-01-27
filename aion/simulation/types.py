"""AION Simulation Environment Types.

Comprehensive type system for the simulation framework including:
- Entity Component System (ECS) types
- Causal event graph types
- Copy-on-write state types
- Statistical evaluation types
- Deterministic execution primitives
"""

from __future__ import annotations

import copy
import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    Union,
)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class SimulationStatus(str, Enum):
    """Status of a simulation."""

    CREATED = "created"
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class EntityType(str, Enum):
    """Types of entities in the simulation."""

    AGENT = "agent"
    USER = "user"
    SYSTEM = "system"
    RESOURCE = "resource"
    MESSAGE = "message"
    TASK = "task"
    EVENT = "event"
    ENVIRONMENT = "environment"
    SENSOR = "sensor"
    ACTUATOR = "actuator"


class EventType(str, Enum):
    """Types of simulation events."""

    AGENT_ACTION = "agent_action"
    USER_INPUT = "user_input"
    SYSTEM_EVENT = "system_event"
    STATE_CHANGE = "state_change"
    TIME_TICK = "time_tick"
    TRIGGER = "trigger"
    ERROR = "error"
    CONSTRAINT_VIOLATION = "constraint_violation"
    RULE_FIRED = "rule_fired"
    BRANCH_POINT = "branch_point"
    CHECKPOINT = "checkpoint"


class TimeMode(str, Enum):
    """Time progression modes."""

    REAL_TIME = "real_time"
    ACCELERATED = "accelerated"
    STEP = "step"
    EVENT_DRIVEN = "event_driven"
    DETERMINISTIC = "deterministic"


class ScenarioType(str, Enum):
    """Types of scenarios."""

    SIMPLE = "simple"
    SEQUENTIAL = "sequential"
    BRANCHING = "branching"
    ADVERSARIAL = "adversarial"
    STRESS = "stress"
    RANDOM = "random"
    REGRESSION = "regression"
    DIFFERENTIAL = "differential"


class ComponentType(str, Enum):
    """ECS component types."""

    TRANSFORM = "transform"
    IDENTITY = "identity"
    BEHAVIOR = "behavior"
    STATE = "state"
    RELATIONSHIP = "relationship"
    CONSTRAINT = "constraint"
    OBSERVABLE = "observable"
    RESOURCE_POOL = "resource_pool"


class ConstraintType(str, Enum):
    """Types of constraints in the simulation."""

    INVARIANT = "invariant"
    PRECONDITION = "precondition"
    POSTCONDITION = "postcondition"
    RESOURCE_LIMIT = "resource_limit"
    TEMPORAL = "temporal"
    CAUSAL = "causal"
    MUTUAL_EXCLUSION = "mutual_exclusion"


class FuzzStrategy(str, Enum):
    """Fuzzing strategies for adversarial generation."""

    RANDOM = "random"
    MUTATION = "mutation"
    GRAMMAR = "grammar"
    COVERAGE_GUIDED = "coverage_guided"
    EVOLUTIONARY = "evolutionary"


# ---------------------------------------------------------------------------
# Component Protocol (ECS)
# ---------------------------------------------------------------------------


class Component(Protocol):
    """Protocol for ECS components."""

    component_type: ComponentType

    def clone(self) -> "Component": ...

    def to_dict(self) -> Dict[str, Any]: ...


# ---------------------------------------------------------------------------
# Core Entity (ECS)
# ---------------------------------------------------------------------------


@dataclass
class Entity:
    """An entity in the simulation world using ECS pattern.

    Entities are lightweight identifiers with attached components.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: EntityType = EntityType.RESOURCE
    name: str = ""

    # ECS Components
    components: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Legacy property access (backed by components)
    properties: Dict[str, Any] = field(default_factory=dict)
    state: Dict[str, Any] = field(default_factory=dict)

    # Relationships (graph edges)
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    relationships: Dict[str, List[str]] = field(default_factory=dict)

    # Behavior references
    behaviors: List[str] = field(default_factory=list)

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    tags: Set[str] = field(default_factory=set)
    version: int = 0

    def get_property(self, key: str, default: Any = None) -> Any:
        return self.properties.get(key, default)

    def set_property(self, key: str, value: Any) -> None:
        self.properties[key] = value
        self.version += 1
        self.updated_at = datetime.utcnow()

    def add_component(self, name: str, data: Dict[str, Any]) -> None:
        """Attach a component to this entity."""
        self.components[name] = data
        self.version += 1
        self.updated_at = datetime.utcnow()

    def get_component(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a component by name."""
        return self.components.get(name)

    def has_component(self, name: str) -> bool:
        return name in self.components

    def remove_component(self, name: str) -> Optional[Dict[str, Any]]:
        self.version += 1
        return self.components.pop(name, None)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "name": self.name,
            "components": self.components,
            "properties": self.properties,
            "state": self.state,
            "version": self.version,
            "tags": list(self.tags),
        }

    def fingerprint(self) -> str:
        """Content-addressable fingerprint for change detection."""
        content = json.dumps(
            {"properties": self.properties, "state": self.state, "components": self.components},
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(content.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Simulation Event (Causal DAG node)
# ---------------------------------------------------------------------------


@dataclass
class SimulationEvent:
    """An event in the simulation, forming a node in the causal DAG."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: EventType = EventType.SYSTEM_EVENT

    # Event data
    source_id: Optional[str] = None
    target_id: Optional[str] = None
    action: str = ""
    data: Dict[str, Any] = field(default_factory=dict)

    # Timing
    timestamp: datetime = field(default_factory=datetime.utcnow)
    simulation_time: float = 0.0
    tick: int = 0

    # Result
    result: Optional[Any] = None
    success: bool = True
    error: Optional[str] = None

    # Causal DAG edges
    caused_by: Optional[str] = None  # Parent event ID
    causes: List[str] = field(default_factory=list)  # Child event IDs
    causal_depth: int = 0  # Depth in causal chain

    # Determinism
    sequence_number: int = 0  # Global ordering for deterministic replay

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "action": self.action,
            "data": self.data,
            "simulation_time": self.simulation_time,
            "tick": self.tick,
            "success": self.success,
            "error": self.error,
            "caused_by": self.caused_by,
            "causes": self.causes,
            "sequence_number": self.sequence_number,
        }


# ---------------------------------------------------------------------------
# World State (Copy-on-Write friendly)
# ---------------------------------------------------------------------------


@dataclass
class WorldState:
    """Complete state of the simulation world.

    Designed for efficient copy-on-write snapshots via structural sharing.
    Entity dictionaries are shallow-copied on snapshot; individual entities
    are deep-copied only when mutated (copy-on-write semantics).
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Time
    simulation_time: float = 0.0
    real_time: datetime = field(default_factory=datetime.utcnow)
    tick: int = 0

    # Entities (keyed by ID)
    entities: Dict[str, Entity] = field(default_factory=dict)

    # Global state
    global_state: Dict[str, Any] = field(default_factory=dict)

    # Events
    pending_events: List[SimulationEvent] = field(default_factory=list)
    event_history: List[SimulationEvent] = field(default_factory=list)

    # Metrics
    metrics: Dict[str, float] = field(default_factory=dict)

    # Determinism
    rng_state: Optional[Any] = None  # Captured RNG state for replay
    event_sequence: int = 0  # Monotonic event counter

    # Change tracking
    _dirty_entities: Set[str] = field(default_factory=set)

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        return self.entities.get(entity_id)

    def add_entity(self, entity: Entity) -> None:
        self.entities[entity.id] = entity
        self._dirty_entities.add(entity.id)

    def remove_entity(self, entity_id: str) -> Optional[Entity]:
        self._dirty_entities.discard(entity_id)
        return self.entities.pop(entity_id, None)

    def mutate_entity(self, entity_id: str) -> Optional[Entity]:
        """Get entity for mutation, performing COW copy if needed."""
        entity = self.entities.get(entity_id)
        if entity is None:
            return None
        if entity_id not in self._dirty_entities:
            entity = copy.deepcopy(entity)
            self.entities[entity_id] = entity
            self._dirty_entities.add(entity_id)
        return entity

    def next_event_sequence(self) -> int:
        """Get next monotonic event sequence number."""
        seq = self.event_sequence
        self.event_sequence += 1
        return seq

    def clone(self) -> "WorldState":
        """Create a COW-friendly snapshot.

        Shallow-copies the entity dict; entities are only deep-copied
        when mutated via ``mutate_entity``.
        """
        return WorldState(
            id=str(uuid.uuid4()),
            simulation_time=self.simulation_time,
            real_time=self.real_time,
            tick=self.tick,
            entities=dict(self.entities),  # shallow copy
            global_state=copy.deepcopy(self.global_state),
            pending_events=[],
            event_history=list(self.event_history),  # shallow copy of list
            metrics=dict(self.metrics),
            rng_state=copy.deepcopy(self.rng_state),
            event_sequence=self.event_sequence,
            _dirty_entities=set(),  # fresh dirty set
        )

    def fingerprint(self) -> str:
        """Content fingerprint of the world state."""
        entity_fps = sorted(
            (eid, e.fingerprint()) for eid, e in self.entities.items()
        )
        content = json.dumps(
            {
                "tick": self.tick,
                "entities": entity_fps,
                "global_state": self.global_state,
                "metrics": self.metrics,
            },
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(content.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Scenario
# ---------------------------------------------------------------------------


@dataclass
class Scenario:
    """A simulation scenario."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    type: ScenarioType = ScenarioType.SIMPLE

    # Initial setup
    initial_state: Dict[str, Any] = field(default_factory=dict)
    initial_entities: List[Dict[str, Any]] = field(default_factory=list)

    # Events to inject
    scripted_events: List[Dict[str, Any]] = field(default_factory=list)

    # User simulation
    simulated_users: List[Dict[str, Any]] = field(default_factory=list)

    # Goals and success criteria
    goals: List[Dict[str, Any]] = field(default_factory=list)
    success_criteria: List[Dict[str, Any]] = field(default_factory=list)
    failure_criteria: List[Dict[str, Any]] = field(default_factory=list)

    # Constraints
    max_steps: int = 1000
    max_time: float = 3600.0

    # Configuration
    config: Dict[str, Any] = field(default_factory=dict)

    # Tags for categorization
    tags: Set[str] = field(default_factory=set)

    # Difficulty & coverage metadata
    difficulty: float = 0.5  # 0.0 = trivial, 1.0 = extreme
    coverage_domains: Set[str] = field(default_factory=set)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "type": self.type.value,
            "initial_entities": self.initial_entities,
            "scripted_events": self.scripted_events,
            "goals": self.goals,
            "max_steps": self.max_steps,
            "difficulty": self.difficulty,
            "tags": list(self.tags),
        }


# ---------------------------------------------------------------------------
# Simulation Config
# ---------------------------------------------------------------------------


@dataclass
class SimulationConfig:
    """Configuration for a simulation run."""

    # Time settings
    time_mode: TimeMode = TimeMode.STEP
    time_scale: float = 1.0
    tick_duration: float = 1.0

    # Limits
    max_ticks: int = 10_000
    max_events: int = 100_000
    timeout_seconds: float = 300.0

    # Resource limits per agent
    max_memory_mb: int = 512
    max_cpu_percent: float = 50.0
    max_actions_per_agent: int = 10_000

    # Determinism
    seed: Optional[int] = None
    deterministic: bool = True

    # Recording
    record_all_events: bool = True
    record_state_snapshots: bool = True
    snapshot_interval: int = 100
    record_causal_graph: bool = True

    # Debugging
    verbose: bool = False
    breakpoints: List[Dict[str, Any]] = field(default_factory=list)

    # Parallel execution
    parallel_scenarios: int = 1
    scenario_timeout: float = 60.0


# ---------------------------------------------------------------------------
# Agent in Simulation
# ---------------------------------------------------------------------------


@dataclass
class AgentInSimulation:
    """An agent being tested in simulation."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str = ""

    # Configuration
    config: Dict[str, Any] = field(default_factory=dict)

    # Mocked resources
    mocked_tools: Dict[str, Any] = field(default_factory=dict)
    mocked_memory: Dict[str, Any] = field(default_factory=dict)

    # Tracking
    actions_taken: List[Dict[str, Any]] = field(default_factory=list)
    messages_sent: List[Dict[str, Any]] = field(default_factory=list)
    decisions_made: List[Dict[str, Any]] = field(default_factory=list)

    # Metrics
    total_actions: int = 0
    successful_actions: int = 0
    failed_actions: int = 0
    total_tokens: int = 0

    # Resource tracking
    peak_memory_mb: float = 0.0
    total_time_ms: float = 0.0


# ---------------------------------------------------------------------------
# Timeline Snapshot
# ---------------------------------------------------------------------------


@dataclass
class TimelineSnapshot:
    """A snapshot of simulation state at a point in time."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # When
    tick: int = 0
    simulation_time: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)

    # State (COW clone)
    world_state: WorldState = field(default_factory=WorldState)

    # Branching
    parent_snapshot_id: Optional[str] = None
    branch_name: Optional[str] = None

    # Metadata
    description: str = ""
    tags: Set[str] = field(default_factory=set)

    # Content fingerprint for deduplication
    state_fingerprint: str = ""

    def compute_fingerprint(self) -> str:
        self.state_fingerprint = self.world_state.fingerprint()
        return self.state_fingerprint


# ---------------------------------------------------------------------------
# Simulation Result
# ---------------------------------------------------------------------------


@dataclass
class SimulationResult:
    """Result of a simulation run."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    simulation_id: str = ""
    scenario_id: str = ""

    # Status
    status: SimulationStatus = SimulationStatus.COMPLETED

    # Timing
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    total_ticks: int = 0
    total_simulation_time: float = 0.0
    total_real_time: float = 0.0

    # Results
    goals_achieved: List[str] = field(default_factory=list)
    goals_failed: List[str] = field(default_factory=list)
    success: bool = False

    # Metrics
    metrics: Dict[str, float] = field(default_factory=dict)

    # Errors
    errors: List[str] = field(default_factory=list)

    # Data
    final_state: Optional[WorldState] = None
    event_count: int = 0
    snapshot_count: int = 0

    # Causal graph summary
    causal_depth_max: int = 0
    causal_chains: int = 0

    # Resource usage
    peak_memory_mb: float = 0.0
    agent_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Evaluation Types
# ---------------------------------------------------------------------------


@dataclass
class EvaluationMetric:
    """A metric for evaluating simulation results."""

    name: str
    description: str = ""

    # Calculation
    calculator: Optional[Callable] = None

    # Thresholds
    target: Optional[float] = None
    min_threshold: Optional[float] = None
    max_threshold: Optional[float] = None

    # Aggregation
    aggregation: str = "mean"  # mean, sum, min, max, last, p50, p95, p99

    # Weighting
    weight: float = 1.0

    # Statistical
    confidence_level: float = 0.95


@dataclass
class Assertion:
    """An assertion to check during/after simulation."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""

    # Condition
    condition: str = ""
    condition_fn: Optional[Callable] = None  # Compiled condition for performance

    # When to check
    check_at: str = "end"  # start, end, always, tick:N, event:TYPE

    # Severity
    severity: str = "error"  # error, warning, info

    # Result
    passed: Optional[bool] = None
    message: str = ""
    failure_context: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Constraint (for physics/rules)
# ---------------------------------------------------------------------------


@dataclass
class Constraint:
    """A constraint in the simulation world."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    type: ConstraintType = ConstraintType.INVARIANT

    # Condition that must hold
    condition: Optional[Callable[[WorldState], bool]] = None
    condition_expr: str = ""

    # What to do on violation
    on_violation: str = "error"  # error, warn, rollback, correct
    correction: Optional[Callable[[WorldState], None]] = None

    # Scope
    entity_filter: Optional[Callable[[Entity], bool]] = None
    priority: int = 0

    # Tracking
    violations: int = 0
    last_violation_tick: int = -1
