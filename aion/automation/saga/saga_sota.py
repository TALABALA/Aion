"""
AION Saga Pattern - True SOTA Implementation

Production-grade saga orchestration with:
- Persistent saga state (survives crashes)
- Automatic crash recovery
- Distributed coordination with locking
- Idempotent step execution
- Parallel step execution with DAG support
- Timeout handling at saga and step level
- Comprehensive compensation strategies
- Saga versioning for upgrades
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import (
    Any,
    Callable,
    Coroutine,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

import structlog

logger = structlog.get_logger(__name__)


# =============================================================================
# Core Types
# =============================================================================


class SagaStatus(str, Enum):
    """Saga execution states."""
    CREATED = "created"
    RUNNING = "running"
    COMPENSATING = "compensating"
    COMPLETED = "completed"
    COMPENSATED = "compensated"
    FAILED = "failed"
    PARTIALLY_COMPENSATED = "partially_compensated"
    TIMED_OUT = "timed_out"


class StepStatus(str, Enum):
    """Step execution states."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    COMPENSATING = "compensating"
    COMPENSATED = "compensated"
    COMPENSATION_FAILED = "compensation_failed"


class CompensationStrategy(str, Enum):
    """How to handle compensation failures."""
    RETRY = "retry"  # Retry compensation
    SKIP = "skip"  # Skip and continue
    FAIL = "fail"  # Fail entire saga
    MANUAL = "manual"  # Require manual intervention


@dataclass
class StepDefinition:
    """
    Definition of a saga step.

    Supports:
    - Async action execution
    - Compensation with retry
    - Timeout configuration
    - Dependencies on other steps
    """
    id: str
    name: str

    # Action configuration
    action_type: str  # Handler type to invoke
    action_params: Dict[str, Any] = field(default_factory=dict)

    # Compensation configuration
    compensation_type: Optional[str] = None
    compensation_params: Dict[str, Any] = field(default_factory=dict)
    compensation_strategy: CompensationStrategy = CompensationStrategy.RETRY
    compensation_max_retries: int = 3

    # Timeout
    timeout_seconds: Optional[float] = None

    # Dependencies (step IDs that must complete first)
    depends_on: List[str] = field(default_factory=list)

    # Retry configuration
    max_retries: int = 3
    retry_delay: float = 1.0
    retry_backoff: float = 2.0

    # Idempotency key template (uses step context)
    idempotency_key_template: Optional[str] = None

    # Whether step can run in parallel with others
    parallel: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "action_type": self.action_type,
            "action_params": self.action_params,
            "compensation_type": self.compensation_type,
            "compensation_params": self.compensation_params,
            "compensation_strategy": self.compensation_strategy.value,
            "compensation_max_retries": self.compensation_max_retries,
            "timeout_seconds": self.timeout_seconds,
            "depends_on": self.depends_on,
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay,
            "retry_backoff": self.retry_backoff,
            "idempotency_key_template": self.idempotency_key_template,
            "parallel": self.parallel,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StepDefinition":
        return cls(
            id=data["id"],
            name=data["name"],
            action_type=data["action_type"],
            action_params=data.get("action_params", {}),
            compensation_type=data.get("compensation_type"),
            compensation_params=data.get("compensation_params", {}),
            compensation_strategy=CompensationStrategy(
                data.get("compensation_strategy", "retry")
            ),
            compensation_max_retries=data.get("compensation_max_retries", 3),
            timeout_seconds=data.get("timeout_seconds"),
            depends_on=data.get("depends_on", []),
            max_retries=data.get("max_retries", 3),
            retry_delay=data.get("retry_delay", 1.0),
            retry_backoff=data.get("retry_backoff", 2.0),
            idempotency_key_template=data.get("idempotency_key_template"),
            parallel=data.get("parallel", True),
        )


@dataclass
class SagaDefinition:
    """
    Definition of a saga (distributed transaction).

    Supports:
    - Versioned schemas for upgrade
    - DAG-based step execution
    - Global timeout
    - Metadata for tracking
    """
    id: str
    name: str
    version: str = "1.0.0"
    steps: List[StepDefinition] = field(default_factory=list)
    description: str = ""

    # Global timeout for entire saga
    timeout_seconds: Optional[float] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def add_step(
        self,
        id: str,
        name: str,
        action_type: str,
        **kwargs,
    ) -> "SagaDefinition":
        """Add a step (fluent API)."""
        step = StepDefinition(
            id=id,
            name=name,
            action_type=action_type,
            **kwargs,
        )
        self.steps.append(step)
        return self

    def get_step(self, step_id: str) -> Optional[StepDefinition]:
        """Get step by ID."""
        for step in self.steps:
            if step.id == step_id:
                return step
        return None

    def get_execution_order(self) -> List[List[str]]:
        """
        Get step execution order as levels (for parallel execution).

        Returns list of lists, where each inner list contains steps
        that can run in parallel.
        """
        # Build dependency graph
        remaining = {s.id for s in self.steps}
        completed: Set[str] = set()
        levels: List[List[str]] = []

        while remaining:
            # Find steps with all dependencies satisfied
            ready = []
            for step_id in remaining:
                step = self.get_step(step_id)
                if all(dep in completed for dep in step.depends_on):
                    ready.append(step_id)

            if not ready:
                raise ValueError("Circular dependency detected in saga steps")

            # Group parallel steps
            parallel_steps = []
            sequential_steps = []
            for step_id in ready:
                step = self.get_step(step_id)
                if step.parallel:
                    parallel_steps.append(step_id)
                else:
                    sequential_steps.append(step_id)

            # Add parallel steps as one level
            if parallel_steps:
                levels.append(parallel_steps)

            # Add sequential steps as individual levels
            for step_id in sequential_steps:
                levels.append([step_id])

            # Update tracking
            for step_id in ready:
                remaining.remove(step_id)
                completed.add(step_id)

        return levels

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "steps": [s.to_dict() for s in self.steps],
            "description": self.description,
            "timeout_seconds": self.timeout_seconds,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SagaDefinition":
        return cls(
            id=data["id"],
            name=data["name"],
            version=data.get("version", "1.0.0"),
            steps=[StepDefinition.from_dict(s) for s in data.get("steps", [])],
            description=data.get("description", ""),
            timeout_seconds=data.get("timeout_seconds"),
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(),
        )


@dataclass
class StepState:
    """Runtime state of a saga step."""
    step_id: str
    status: StepStatus = StepStatus.PENDING

    # Execution tracking
    attempts: int = 0
    compensation_attempts: int = 0

    # Results
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    compensation_error: Optional[str] = None

    # Timestamps
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    compensated_at: Optional[datetime] = None

    # Idempotency
    idempotency_key: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "status": self.status.value,
            "attempts": self.attempts,
            "compensation_attempts": self.compensation_attempts,
            "result": self.result,
            "error": self.error,
            "compensation_error": self.compensation_error,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "compensated_at": self.compensated_at.isoformat() if self.compensated_at else None,
            "idempotency_key": self.idempotency_key,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StepState":
        return cls(
            step_id=data["step_id"],
            status=StepStatus(data.get("status", "pending")),
            attempts=data.get("attempts", 0),
            compensation_attempts=data.get("compensation_attempts", 0),
            result=data.get("result"),
            error=data.get("error"),
            compensation_error=data.get("compensation_error"),
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            compensated_at=datetime.fromisoformat(data["compensated_at"]) if data.get("compensated_at") else None,
            idempotency_key=data.get("idempotency_key"),
        )


@dataclass
class SagaState:
    """
    Persistent saga execution state.

    This is the core state that survives crashes and enables recovery.
    """
    # Identity
    saga_id: str
    execution_id: str
    definition_version: str

    # Status
    status: SagaStatus = SagaStatus.CREATED

    # Context passed between steps
    context: Dict[str, Any] = field(default_factory=dict)

    # Step states
    step_states: Dict[str, StepState] = field(default_factory=dict)

    # Tracking
    completed_steps: List[str] = field(default_factory=list)
    failed_step: Optional[str] = None
    compensated_steps: List[str] = field(default_factory=list)

    # Error info
    error: Optional[str] = None

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    deadline: Optional[datetime] = None

    # Lock info (for distributed coordination)
    locked_by: Optional[str] = None
    lock_expires_at: Optional[datetime] = None

    # Version for optimistic concurrency
    version: int = 0

    def is_locked(self) -> bool:
        """Check if saga is locked by another worker."""
        if not self.locked_by:
            return False
        if self.lock_expires_at and self.lock_expires_at < datetime.now():
            return False
        return True

    def is_timed_out(self) -> bool:
        """Check if saga has exceeded deadline."""
        if self.deadline and datetime.now() > self.deadline:
            return True
        return False

    def get_step_state(self, step_id: str) -> StepState:
        """Get or create step state."""
        if step_id not in self.step_states:
            self.step_states[step_id] = StepState(step_id=step_id)
        return self.step_states[step_id]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "saga_id": self.saga_id,
            "execution_id": self.execution_id,
            "definition_version": self.definition_version,
            "status": self.status.value,
            "context": self.context,
            "step_states": {k: v.to_dict() for k, v in self.step_states.items()},
            "completed_steps": self.completed_steps,
            "failed_step": self.failed_step,
            "compensated_steps": self.compensated_steps,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "locked_by": self.locked_by,
            "lock_expires_at": self.lock_expires_at.isoformat() if self.lock_expires_at else None,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SagaState":
        state = cls(
            saga_id=data["saga_id"],
            execution_id=data["execution_id"],
            definition_version=data["definition_version"],
            status=SagaStatus(data.get("status", "created")),
            context=data.get("context", {}),
            completed_steps=data.get("completed_steps", []),
            failed_step=data.get("failed_step"),
            compensated_steps=data.get("compensated_steps", []),
            error=data.get("error"),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(),
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            deadline=datetime.fromisoformat(data["deadline"]) if data.get("deadline") else None,
            locked_by=data.get("locked_by"),
            lock_expires_at=datetime.fromisoformat(data["lock_expires_at"]) if data.get("lock_expires_at") else None,
            version=data.get("version", 0),
        )

        for step_id, step_data in data.get("step_states", {}).items():
            state.step_states[step_id] = StepState.from_dict(step_data)

        return state


# =============================================================================
# State Store Interface
# =============================================================================


class SagaStateStore(ABC):
    """Abstract base class for saga state persistence."""

    @abstractmethod
    async def save(self, state: SagaState) -> None:
        """Save saga state with optimistic concurrency."""
        pass

    @abstractmethod
    async def get(self, execution_id: str) -> Optional[SagaState]:
        """Get saga state by execution ID."""
        pass

    @abstractmethod
    async def try_lock(
        self,
        execution_id: str,
        worker_id: str,
        lock_duration: float = 60.0,
    ) -> bool:
        """Try to acquire lock on saga."""
        pass

    @abstractmethod
    async def unlock(self, execution_id: str, worker_id: str) -> bool:
        """Release lock on saga."""
        pass

    @abstractmethod
    async def get_recoverable(self, limit: int = 100) -> List[SagaState]:
        """Get sagas that need recovery (running but unlocked)."""
        pass

    @abstractmethod
    async def get_timed_out(self, limit: int = 100) -> List[SagaState]:
        """Get sagas that have exceeded deadline."""
        pass


class RedisSagaStateStore(SagaStateStore):
    """Redis-based saga state store with distributed locking."""

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        prefix: str = "aion:saga:",
    ):
        self.redis_url = redis_url
        self.prefix = prefix
        self._client = None

    async def _get_client(self):
        if self._client is None:
            try:
                import redis.asyncio as redis
                self._client = redis.from_url(self.redis_url, decode_responses=True)
            except ImportError:
                raise ImportError("redis package required")
        return self._client

    def _state_key(self, execution_id: str) -> str:
        return f"{self.prefix}state:{execution_id}"

    def _lock_key(self, execution_id: str) -> str:
        return f"{self.prefix}lock:{execution_id}"

    def _index_key(self, status: str) -> str:
        return f"{self.prefix}idx:{status}"

    async def save(self, state: SagaState) -> None:
        client = await self._get_client()

        state_key = self._state_key(state.execution_id)

        # Optimistic concurrency check
        current_version = await client.hget(state_key, "version")
        if current_version and int(current_version) != state.version:
            raise ConcurrencyError(
                f"Saga {state.execution_id} was modified (expected {state.version}, got {current_version})"
            )

        state.version += 1
        state_json = json.dumps(state.to_dict())

        async with client.pipeline(transaction=True) as pipe:
            pipe.set(state_key, state_json)
            pipe.hset(state_key, "version", state.version)

            # Update status index
            for status in SagaStatus:
                pipe.srem(self._index_key(status.value), state.execution_id)
            pipe.sadd(self._index_key(state.status.value), state.execution_id)

            await pipe.execute()

    async def get(self, execution_id: str) -> Optional[SagaState]:
        client = await self._get_client()

        state_json = await client.get(self._state_key(execution_id))
        if not state_json:
            return None

        return SagaState.from_dict(json.loads(state_json))

    async def try_lock(
        self,
        execution_id: str,
        worker_id: str,
        lock_duration: float = 60.0,
    ) -> bool:
        client = await self._get_client()
        lock_key = self._lock_key(execution_id)

        # Use SET NX EX for atomic lock acquisition
        acquired = await client.set(
            lock_key,
            worker_id,
            nx=True,
            ex=int(lock_duration),
        )

        if acquired:
            # Update state with lock info
            state = await self.get(execution_id)
            if state:
                state.locked_by = worker_id
                state.lock_expires_at = datetime.now() + timedelta(seconds=lock_duration)
                await self.save(state)

        return bool(acquired)

    async def unlock(self, execution_id: str, worker_id: str) -> bool:
        client = await self._get_client()
        lock_key = self._lock_key(execution_id)

        # Only unlock if we own the lock
        current_owner = await client.get(lock_key)
        if current_owner != worker_id:
            return False

        await client.delete(lock_key)

        # Update state
        state = await self.get(execution_id)
        if state:
            state.locked_by = None
            state.lock_expires_at = None
            await self.save(state)

        return True

    async def get_recoverable(self, limit: int = 100) -> List[SagaState]:
        client = await self._get_client()

        # Get running sagas
        running_ids = await client.smembers(self._index_key(SagaStatus.RUNNING.value))

        recoverable = []
        for exec_id in list(running_ids)[:limit]:
            lock_key = self._lock_key(exec_id)

            # Check if unlocked
            if not await client.exists(lock_key):
                state = await self.get(exec_id)
                if state:
                    recoverable.append(state)

        return recoverable

    async def get_timed_out(self, limit: int = 100) -> List[SagaState]:
        client = await self._get_client()

        running_ids = await client.smembers(self._index_key(SagaStatus.RUNNING.value))

        timed_out = []
        now = datetime.now()

        for exec_id in list(running_ids)[:limit]:
            state = await self.get(exec_id)
            if state and state.deadline and state.deadline < now:
                timed_out.append(state)

        return timed_out


class ConcurrencyError(Exception):
    """Raised when optimistic concurrency check fails."""
    pass


# =============================================================================
# Step Handlers
# =============================================================================


StepHandler = Callable[[StepDefinition, SagaState], Coroutine[Any, Any, Dict[str, Any]]]
CompensationHandler = Callable[[StepDefinition, StepState, SagaState], Coroutine[Any, Any, None]]


class StepHandlerRegistry:
    """Registry for step action and compensation handlers."""

    def __init__(self):
        self._action_handlers: Dict[str, StepHandler] = {}
        self._compensation_handlers: Dict[str, CompensationHandler] = {}

    def register_action(self, action_type: str, handler: StepHandler) -> None:
        """Register an action handler."""
        self._action_handlers[action_type] = handler
        logger.debug(f"Registered action handler: {action_type}")

    def register_compensation(self, comp_type: str, handler: CompensationHandler) -> None:
        """Register a compensation handler."""
        self._compensation_handlers[comp_type] = handler
        logger.debug(f"Registered compensation handler: {comp_type}")

    def get_action_handler(self, action_type: str) -> Optional[StepHandler]:
        return self._action_handlers.get(action_type)

    def get_compensation_handler(self, comp_type: str) -> Optional[CompensationHandler]:
        return self._compensation_handlers.get(comp_type)


# =============================================================================
# SOTA Saga Orchestrator
# =============================================================================


class SagaOrchestratorSOTA:
    """
    Production-grade saga orchestrator.

    Features:
    - Persistent state (survives crashes)
    - Automatic crash recovery
    - Distributed coordination with locking
    - Idempotent step execution
    - Parallel step execution
    - Timeout handling
    - Comprehensive compensation
    """

    def __init__(
        self,
        state_store: SagaStateStore,
        handler_registry: Optional[StepHandlerRegistry] = None,
        worker_id: Optional[str] = None,
        recovery_interval: float = 30.0,
        lock_duration: float = 60.0,
    ):
        self.state_store = state_store
        self.handler_registry = handler_registry or StepHandlerRegistry()
        self.worker_id = worker_id or f"saga-worker-{uuid.uuid4().hex[:8]}"
        self.recovery_interval = recovery_interval
        self.lock_duration = lock_duration

        # Saga definitions
        self._definitions: Dict[str, SagaDefinition] = {}

        # Background tasks
        self._recovery_task: Optional[asyncio.Task] = None
        self._timeout_task: Optional[asyncio.Task] = None

        self._initialized = False
        self._shutdown_event = asyncio.Event()

    async def initialize(self) -> None:
        """Initialize the orchestrator."""
        if self._initialized:
            return

        # Start recovery loop
        self._recovery_task = asyncio.create_task(self._recovery_loop())
        self._timeout_task = asyncio.create_task(self._timeout_loop())

        self._initialized = True
        logger.info("SOTA Saga Orchestrator initialized", worker_id=self.worker_id)

    async def shutdown(self) -> None:
        """Shutdown the orchestrator."""
        self._shutdown_event.set()

        if self._recovery_task:
            self._recovery_task.cancel()
            try:
                await self._recovery_task
            except asyncio.CancelledError:
                pass

        if self._timeout_task:
            self._timeout_task.cancel()
            try:
                await self._timeout_task
            except asyncio.CancelledError:
                pass

        self._initialized = False
        logger.info("SOTA Saga Orchestrator shutdown")

    def register_saga(self, definition: SagaDefinition) -> None:
        """Register a saga definition."""
        self._definitions[definition.id] = definition
        logger.info(f"Registered saga: {definition.id} v{definition.version}")

    def register_action(self, action_type: str, handler: StepHandler) -> None:
        """Register an action handler."""
        self.handler_registry.register_action(action_type, handler)

    def register_compensation(self, comp_type: str, handler: CompensationHandler) -> None:
        """Register a compensation handler."""
        self.handler_registry.register_compensation(comp_type, handler)

    async def execute(
        self,
        saga_id: str,
        context: Dict[str, Any],
        execution_id: Optional[str] = None,
    ) -> SagaState:
        """
        Execute a saga.

        Returns final saga state after completion or failure.
        """
        definition = self._definitions.get(saga_id)
        if not definition:
            raise ValueError(f"Saga not found: {saga_id}")

        # Create execution state
        exec_id = execution_id or str(uuid.uuid4())

        state = SagaState(
            saga_id=saga_id,
            execution_id=exec_id,
            definition_version=definition.version,
            context=context,
        )

        if definition.timeout_seconds:
            state.deadline = datetime.now() + timedelta(seconds=definition.timeout_seconds)

        # Save initial state
        await self.state_store.save(state)

        # Try to acquire lock
        if not await self.state_store.try_lock(exec_id, self.worker_id, self.lock_duration):
            raise RuntimeError(f"Failed to acquire lock for saga {exec_id}")

        try:
            return await self._run_saga(definition, state)
        finally:
            await self.state_store.unlock(exec_id, self.worker_id)

    async def _run_saga(
        self,
        definition: SagaDefinition,
        state: SagaState,
    ) -> SagaState:
        """Run saga execution (main logic)."""
        state.status = SagaStatus.RUNNING
        state.started_at = datetime.now()
        await self.state_store.save(state)

        try:
            # Get execution order (levels of parallelizable steps)
            levels = definition.get_execution_order()

            for level in levels:
                # Check timeout
                if state.is_timed_out():
                    raise TimeoutError("Saga execution timed out")

                # Execute steps in this level
                if len(level) == 1:
                    await self._execute_step(definition, state, level[0])
                else:
                    await self._execute_parallel_steps(definition, state, level)

                # Save state after each level
                await self.state_store.save(state)

            # All steps completed
            state.status = SagaStatus.COMPLETED
            state.completed_at = datetime.now()
            await self.state_store.save(state)

            logger.info(
                "Saga completed",
                saga_id=definition.id,
                execution_id=state.execution_id,
            )

            return state

        except Exception as e:
            logger.error(
                "Saga failed, starting compensation",
                saga_id=definition.id,
                execution_id=state.execution_id,
                error=str(e),
            )

            state.error = str(e)
            state.status = SagaStatus.COMPENSATING
            await self.state_store.save(state)

            # Run compensation
            await self._compensate(definition, state)

            return state

    async def _execute_step(
        self,
        definition: SagaDefinition,
        state: SagaState,
        step_id: str,
    ) -> None:
        """Execute a single step with retry."""
        step_def = definition.get_step(step_id)
        if not step_def:
            raise ValueError(f"Step not found: {step_id}")

        step_state = state.get_step_state(step_id)

        # Check if already completed (idempotency)
        if step_state.status == StepStatus.COMPLETED:
            logger.debug(f"Step {step_id} already completed, skipping")
            return

        # Get handler
        handler = self.handler_registry.get_action_handler(step_def.action_type)
        if not handler:
            raise ValueError(f"No handler for action type: {step_def.action_type}")

        # Generate idempotency key
        if step_def.idempotency_key_template:
            try:
                idem_key = step_def.idempotency_key_template.format(**state.context)
            except KeyError:
                idem_key = None
        else:
            idem_key = f"{state.execution_id}:{step_id}"

        step_state.idempotency_key = idem_key

        # Execute with retry
        last_error = None

        for attempt in range(step_def.max_retries + 1):
            step_state.attempts = attempt + 1
            step_state.status = StepStatus.RUNNING
            step_state.started_at = datetime.now()

            try:
                # Execute with timeout
                if step_def.timeout_seconds:
                    result = await asyncio.wait_for(
                        handler(step_def, state),
                        timeout=step_def.timeout_seconds,
                    )
                else:
                    result = await handler(step_def, state)

                # Success
                step_state.status = StepStatus.COMPLETED
                step_state.completed_at = datetime.now()
                step_state.result = result

                # Update context with result
                if result:
                    state.context[f"step_{step_id}"] = result

                state.completed_steps.append(step_id)

                logger.debug(
                    "Step completed",
                    step_id=step_id,
                    attempts=step_state.attempts,
                )

                return

            except asyncio.TimeoutError:
                last_error = f"Step timed out after {step_def.timeout_seconds}s"
                logger.warning(f"Step {step_id} timeout, attempt {attempt + 1}")

            except Exception as e:
                last_error = str(e)
                logger.warning(
                    f"Step {step_id} failed, attempt {attempt + 1}: {e}"
                )

            # Retry delay
            if attempt < step_def.max_retries:
                delay = step_def.retry_delay * (step_def.retry_backoff ** attempt)
                await asyncio.sleep(delay)

        # All retries failed
        step_state.status = StepStatus.FAILED
        step_state.error = last_error
        state.failed_step = step_id

        raise RuntimeError(f"Step {step_id} failed after {step_def.max_retries + 1} attempts: {last_error}")

    async def _execute_parallel_steps(
        self,
        definition: SagaDefinition,
        state: SagaState,
        step_ids: List[str],
    ) -> None:
        """Execute steps in parallel."""
        tasks = []

        for step_id in step_ids:
            task = asyncio.create_task(
                self._execute_step(definition, state, step_id)
            )
            tasks.append((step_id, task))

        # Wait for all to complete
        results = await asyncio.gather(
            *[t[1] for t in tasks],
            return_exceptions=True,
        )

        # Check for failures
        errors = []
        for (step_id, _), result in zip(tasks, results):
            if isinstance(result, Exception):
                errors.append((step_id, result))

        if errors:
            # First error becomes the main error
            step_id, error = errors[0]
            state.failed_step = step_id
            raise error

    async def _compensate(
        self,
        definition: SagaDefinition,
        state: SagaState,
    ) -> None:
        """Run compensation for completed steps in reverse order."""
        # Get completed steps in reverse order
        to_compensate = list(reversed(state.completed_steps))

        compensation_errors = []

        for step_id in to_compensate:
            if step_id in state.compensated_steps:
                continue

            step_def = definition.get_step(step_id)
            if not step_def or not step_def.compensation_type:
                continue

            step_state = state.get_step_state(step_id)
            step_state.status = StepStatus.COMPENSATING

            try:
                await self._execute_compensation(
                    definition, state, step_def, step_state
                )

                step_state.status = StepStatus.COMPENSATED
                step_state.compensated_at = datetime.now()
                state.compensated_steps.append(step_id)

                logger.info(f"Step {step_id} compensated")

            except Exception as e:
                logger.error(f"Compensation failed for step {step_id}: {e}")
                step_state.status = StepStatus.COMPENSATION_FAILED
                step_state.compensation_error = str(e)
                compensation_errors.append((step_id, str(e)))

                # Handle based on strategy
                if step_def.compensation_strategy == CompensationStrategy.FAIL:
                    break
                elif step_def.compensation_strategy == CompensationStrategy.MANUAL:
                    # Mark for manual intervention
                    break

            # Save after each compensation
            await self.state_store.save(state)

        # Set final status
        if compensation_errors:
            state.status = SagaStatus.PARTIALLY_COMPENSATED
        else:
            state.status = SagaStatus.COMPENSATED

        state.completed_at = datetime.now()
        await self.state_store.save(state)

    async def _execute_compensation(
        self,
        definition: SagaDefinition,
        state: SagaState,
        step_def: StepDefinition,
        step_state: StepState,
    ) -> None:
        """Execute compensation for a step with retry."""
        handler = self.handler_registry.get_compensation_handler(step_def.compensation_type)
        if not handler:
            raise ValueError(f"No compensation handler for: {step_def.compensation_type}")

        last_error = None

        for attempt in range(step_def.compensation_max_retries + 1):
            step_state.compensation_attempts = attempt + 1

            try:
                await handler(step_def, step_state, state)
                return

            except Exception as e:
                last_error = str(e)
                logger.warning(
                    f"Compensation for {step_def.id} failed, attempt {attempt + 1}: {e}"
                )

            if attempt < step_def.compensation_max_retries:
                await asyncio.sleep(1.0 * (2 ** attempt))

        raise RuntimeError(
            f"Compensation for {step_def.id} failed after "
            f"{step_def.compensation_max_retries + 1} attempts: {last_error}"
        )

    async def _recovery_loop(self) -> None:
        """Background loop to recover stalled sagas."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(self.recovery_interval)

                # Get recoverable sagas
                sagas = await self.state_store.get_recoverable(limit=10)

                for state in sagas:
                    if not await self.state_store.try_lock(
                        state.execution_id,
                        self.worker_id,
                        self.lock_duration,
                    ):
                        continue

                    try:
                        definition = self._definitions.get(state.saga_id)
                        if not definition:
                            logger.warning(f"Cannot recover saga, definition not found: {state.saga_id}")
                            continue

                        logger.info(
                            "Recovering saga",
                            execution_id=state.execution_id,
                            status=state.status.value,
                        )

                        await self._run_saga(definition, state)

                    except Exception as e:
                        logger.error(
                            "Recovery failed",
                            execution_id=state.execution_id,
                            error=str(e),
                        )
                    finally:
                        await self.state_store.unlock(state.execution_id, self.worker_id)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Recovery loop error", error=str(e))

    async def _timeout_loop(self) -> None:
        """Background loop to handle timed out sagas."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(self.recovery_interval)

                # Get timed out sagas
                sagas = await self.state_store.get_timed_out(limit=10)

                for state in sagas:
                    if not await self.state_store.try_lock(
                        state.execution_id,
                        self.worker_id,
                        self.lock_duration,
                    ):
                        continue

                    try:
                        definition = self._definitions.get(state.saga_id)
                        if not definition:
                            continue

                        logger.warning(
                            "Saga timed out, compensating",
                            execution_id=state.execution_id,
                        )

                        state.status = SagaStatus.TIMED_OUT
                        state.error = "Saga execution timed out"
                        await self._compensate(definition, state)

                    except Exception as e:
                        logger.error(
                            "Timeout handling failed",
                            execution_id=state.execution_id,
                            error=str(e),
                        )
                    finally:
                        await self.state_store.unlock(state.execution_id, self.worker_id)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Timeout loop error", error=str(e))

    async def get_execution(self, execution_id: str) -> Optional[SagaState]:
        """Get saga execution state."""
        return await self.state_store.get(execution_id)

    def get_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics."""
        return {
            "worker_id": self.worker_id,
            "registered_sagas": len(self._definitions),
            "initialized": self._initialized,
        }


# =============================================================================
# Builder for Fluent API
# =============================================================================


class SagaBuilder:
    """Fluent builder for creating saga definitions."""

    def __init__(self, name: str):
        self._definition = SagaDefinition(
            id=str(uuid.uuid4()),
            name=name,
        )
        self._current_step: Optional[str] = None

    def id(self, saga_id: str) -> "SagaBuilder":
        self._definition.id = saga_id
        return self

    def version(self, version: str) -> "SagaBuilder":
        self._definition.version = version
        return self

    def description(self, desc: str) -> "SagaBuilder":
        self._definition.description = desc
        return self

    def timeout(self, seconds: float) -> "SagaBuilder":
        self._definition.timeout_seconds = seconds
        return self

    def step(
        self,
        id: str,
        name: str,
        action_type: str,
        **kwargs,
    ) -> "SagaBuilder":
        """Add a step."""
        step = StepDefinition(
            id=id,
            name=name,
            action_type=action_type,
            **kwargs,
        )
        self._definition.steps.append(step)
        self._current_step = id
        return self

    def with_compensation(
        self,
        comp_type: str,
        strategy: CompensationStrategy = CompensationStrategy.RETRY,
        max_retries: int = 3,
        **params,
    ) -> "SagaBuilder":
        """Add compensation to current step."""
        if not self._current_step:
            raise ValueError("No current step")

        step = self._definition.get_step(self._current_step)
        if step:
            step.compensation_type = comp_type
            step.compensation_params = params
            step.compensation_strategy = strategy
            step.compensation_max_retries = max_retries

        return self

    def depends_on(self, *step_ids: str) -> "SagaBuilder":
        """Set dependencies for current step."""
        if not self._current_step:
            raise ValueError("No current step")

        step = self._definition.get_step(self._current_step)
        if step:
            step.depends_on = list(step_ids)

        return self

    def sequential(self) -> "SagaBuilder":
        """Mark current step as sequential (not parallel)."""
        if not self._current_step:
            raise ValueError("No current step")

        step = self._definition.get_step(self._current_step)
        if step:
            step.parallel = False

        return self

    def build(self) -> SagaDefinition:
        return self._definition


def saga(name: str) -> SagaBuilder:
    """Create a new saga with fluent API."""
    return SagaBuilder(name)


# =============================================================================
# Factory Functions
# =============================================================================


async def create_redis_saga_orchestrator(
    redis_url: str = "redis://localhost:6379",
    **kwargs,
) -> SagaOrchestratorSOTA:
    """Create saga orchestrator with Redis state store."""
    state_store = RedisSagaStateStore(redis_url=redis_url)
    orchestrator = SagaOrchestratorSOTA(state_store=state_store, **kwargs)
    await orchestrator.initialize()
    return orchestrator
