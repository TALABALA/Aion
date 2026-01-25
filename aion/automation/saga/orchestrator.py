"""
AION Saga Orchestrator

Orchestration-based saga pattern implementation.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

import structlog

logger = structlog.get_logger(__name__)


class SagaStatus(str, Enum):
    """Status of a saga execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    COMPENSATING = "compensating"
    COMPENSATED = "compensated"
    FAILED = "failed"
    PARTIALLY_COMPENSATED = "partially_compensated"


@dataclass
class SagaStep:
    """
    A step in a saga.

    Each step has:
    - A forward action (the main operation)
    - A compensation action (undo operation)
    """
    id: str
    name: str

    # Forward action
    action: Callable[..., Any]
    action_args: tuple = field(default_factory=tuple)
    action_kwargs: Dict[str, Any] = field(default_factory=dict)

    # Compensation action (rollback)
    compensation: Optional[Callable[..., Any]] = None
    compensation_args: tuple = field(default_factory=tuple)
    compensation_kwargs: Dict[str, Any] = field(default_factory=dict)

    # Configuration
    timeout_seconds: Optional[int] = None
    retry_count: int = 0
    retry_delay: float = 1.0

    # State
    status: str = "pending"
    result: Any = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    compensated_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status,
            "result": str(self.result) if self.result else None,
            "error": self.error,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "compensated_at": self.compensated_at.isoformat() if self.compensated_at else None,
        }


@dataclass
class SagaDefinition:
    """
    Definition of a saga (distributed transaction).

    A saga consists of a sequence of steps, each with
    a forward action and a compensating action.
    """
    id: str
    name: str
    steps: List[SagaStep] = field(default_factory=list)
    description: str = ""

    # Global settings
    timeout_seconds: Optional[int] = None
    parallel: bool = False  # Execute steps in parallel where possible

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def add_step(
        self,
        name: str,
        action: Callable,
        compensation: Optional[Callable] = None,
        action_args: tuple = (),
        action_kwargs: Optional[Dict] = None,
        compensation_args: tuple = (),
        compensation_kwargs: Optional[Dict] = None,
        timeout_seconds: Optional[int] = None,
        retry_count: int = 0,
    ) -> "SagaDefinition":
        """Add a step to the saga (fluent API)."""
        step = SagaStep(
            id=f"step_{len(self.steps)}",
            name=name,
            action=action,
            action_args=action_args,
            action_kwargs=action_kwargs or {},
            compensation=compensation,
            compensation_args=compensation_args,
            compensation_kwargs=compensation_kwargs or {},
            timeout_seconds=timeout_seconds,
            retry_count=retry_count,
        )
        self.steps.append(step)
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "steps": [s.to_dict() for s in self.steps],
            "timeout_seconds": self.timeout_seconds,
            "parallel": self.parallel,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class SagaState:
    """
    Runtime state of a saga execution.
    """
    saga_id: str
    execution_id: str
    status: SagaStatus = SagaStatus.PENDING

    # Execution tracking
    current_step_index: int = 0
    completed_steps: List[str] = field(default_factory=list)
    failed_step: Optional[str] = None
    compensated_steps: List[str] = field(default_factory=list)

    # Context passed between steps
    context: Dict[str, Any] = field(default_factory=dict)

    # Results from each step
    step_results: Dict[str, Any] = field(default_factory=dict)

    # Error information
    error: Optional[str] = None
    error_step: Optional[str] = None

    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "saga_id": self.saga_id,
            "execution_id": self.execution_id,
            "status": self.status.value,
            "current_step_index": self.current_step_index,
            "completed_steps": self.completed_steps,
            "failed_step": self.failed_step,
            "compensated_steps": self.compensated_steps,
            "context": self.context,
            "step_results": {k: str(v)[:200] for k, v in self.step_results.items()},
            "error": self.error,
            "error_step": self.error_step,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


class SagaOrchestrator:
    """
    Orchestrates saga execution with compensation support.

    Features:
    - Sequential and parallel step execution
    - Automatic compensation on failure
    - Idempotency support
    - State persistence
    - Retry with backoff
    """

    def __init__(
        self,
        event_store: Optional["EventStore"] = None,
        max_concurrent_sagas: int = 100,
    ):
        self.event_store = event_store
        self._semaphore = asyncio.Semaphore(max_concurrent_sagas)

        # Registered saga definitions
        self._definitions: Dict[str, SagaDefinition] = {}

        # Active saga executions
        self._executions: Dict[str, SagaState] = {}

        # Callbacks
        self._on_step_completed: List[Callable] = []
        self._on_step_failed: List[Callable] = []
        self._on_saga_completed: List[Callable] = []
        self._on_compensation_started: List[Callable] = []

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the saga orchestrator."""
        if self._initialized:
            return

        self._initialized = True
        logger.info("Saga orchestrator initialized")

    async def shutdown(self) -> None:
        """Shutdown the saga orchestrator."""
        self._initialized = False
        logger.info("Saga orchestrator shutdown")

    def register_saga(self, definition: SagaDefinition) -> None:
        """Register a saga definition."""
        self._definitions[definition.id] = definition
        logger.info(f"Registered saga: {definition.id}")

    async def execute(
        self,
        saga_id: str,
        context: Optional[Dict[str, Any]] = None,
        execution_id: Optional[str] = None,
    ) -> SagaState:
        """
        Execute a saga.

        Args:
            saga_id: ID of the registered saga definition
            context: Initial context to pass to steps
            execution_id: Optional execution ID for idempotency

        Returns:
            Final saga state
        """
        definition = self._definitions.get(saga_id)
        if not definition:
            raise ValueError(f"Saga not found: {saga_id}")

        # Create execution state
        state = SagaState(
            saga_id=saga_id,
            execution_id=execution_id or str(uuid.uuid4()),
            context=context or {},
        )

        self._executions[state.execution_id] = state

        async with self._semaphore:
            try:
                state.status = SagaStatus.RUNNING
                state.started_at = datetime.now()

                await self._record_event(state, "saga.started", {
                    "saga_id": saga_id,
                    "context": context,
                })

                if definition.parallel:
                    await self._execute_parallel(definition, state)
                else:
                    await self._execute_sequential(definition, state)

                state.status = SagaStatus.COMPLETED
                state.completed_at = datetime.now()

                await self._record_event(state, "saga.completed", {
                    "results": state.step_results,
                })

                for callback in self._on_saga_completed:
                    try:
                        await callback(state)
                    except Exception as e:
                        logger.error(f"Saga completion callback failed: {e}")

                logger.info(
                    "Saga completed",
                    saga_id=saga_id,
                    execution_id=state.execution_id,
                )

            except Exception as e:
                state.error = str(e)
                state.status = SagaStatus.COMPENSATING

                await self._record_event(state, "saga.failed", {
                    "error": str(e),
                    "step": state.error_step,
                })

                logger.error(
                    "Saga failed, starting compensation",
                    saga_id=saga_id,
                    execution_id=state.execution_id,
                    error=str(e),
                )

                # Compensate
                await self._compensate(definition, state)

        return state

    async def _execute_sequential(
        self,
        definition: SagaDefinition,
        state: SagaState,
    ) -> None:
        """Execute saga steps sequentially."""
        for i, step in enumerate(definition.steps):
            state.current_step_index = i

            try:
                result = await self._execute_step(step, state)
                state.step_results[step.id] = result
                state.completed_steps.append(step.id)

                # Allow step to modify context for subsequent steps
                if isinstance(result, dict):
                    state.context.update(result)

                for callback in self._on_step_completed:
                    try:
                        await callback(step, result, state)
                    except Exception:
                        pass

            except Exception as e:
                state.error_step = step.id
                state.failed_step = step.id

                for callback in self._on_step_failed:
                    try:
                        await callback(step, e, state)
                    except Exception:
                        pass

                raise

    async def _execute_parallel(
        self,
        definition: SagaDefinition,
        state: SagaState,
    ) -> None:
        """Execute saga steps in parallel where possible."""
        # For now, execute all steps in parallel
        # A more sophisticated implementation would analyze dependencies
        tasks = []

        for step in definition.steps:
            task = asyncio.create_task(self._execute_step(step, state))
            tasks.append((step, task))

        results = await asyncio.gather(
            *[t[1] for t in tasks],
            return_exceptions=True,
        )

        failed = False
        for (step, _), result in zip(tasks, results):
            if isinstance(result, Exception):
                state.error_step = step.id
                state.failed_step = step.id
                state.error = str(result)
                failed = True
            else:
                state.step_results[step.id] = result
                state.completed_steps.append(step.id)

        if failed:
            raise Exception(f"Saga step failed: {state.error}")

    async def _execute_step(
        self,
        step: SagaStep,
        state: SagaState,
    ) -> Any:
        """Execute a single saga step with retries."""
        step.status = "running"
        step.started_at = datetime.now()

        await self._record_event(state, "step.started", {
            "step_id": step.id,
            "step_name": step.name,
        })

        last_error = None

        for attempt in range(step.retry_count + 1):
            try:
                # Prepare arguments with context
                kwargs = {**step.action_kwargs, "context": state.context}

                if step.timeout_seconds:
                    result = await asyncio.wait_for(
                        step.action(*step.action_args, **kwargs),
                        timeout=step.timeout_seconds,
                    )
                else:
                    if asyncio.iscoroutinefunction(step.action):
                        result = await step.action(*step.action_args, **kwargs)
                    else:
                        result = step.action(*step.action_args, **kwargs)

                step.status = "completed"
                step.result = result
                step.completed_at = datetime.now()

                await self._record_event(state, "step.completed", {
                    "step_id": step.id,
                    "attempt": attempt + 1,
                })

                return result

            except Exception as e:
                last_error = e
                logger.warning(
                    f"Step {step.id} failed, attempt {attempt + 1}/{step.retry_count + 1}",
                    error=str(e),
                )

                if attempt < step.retry_count:
                    await asyncio.sleep(step.retry_delay * (attempt + 1))

        step.status = "failed"
        step.error = str(last_error)

        await self._record_event(state, "step.failed", {
            "step_id": step.id,
            "error": str(last_error),
        })

        raise last_error

    async def _compensate(
        self,
        definition: SagaDefinition,
        state: SagaState,
    ) -> None:
        """Run compensation for completed steps in reverse order."""
        for callback in self._on_compensation_started:
            try:
                await callback(state)
            except Exception:
                pass

        await self._record_event(state, "compensation.started", {
            "completed_steps": state.completed_steps,
        })

        # Get completed steps in reverse order
        steps_to_compensate = []
        for step_id in reversed(state.completed_steps):
            step = next((s for s in definition.steps if s.id == step_id), None)
            if step and step.compensation:
                steps_to_compensate.append(step)

        compensation_errors = []

        for step in steps_to_compensate:
            try:
                logger.info(f"Compensating step: {step.id}")

                # Pass the original result and context to compensation
                kwargs = {
                    **step.compensation_kwargs,
                    "context": state.context,
                    "original_result": state.step_results.get(step.id),
                }

                if asyncio.iscoroutinefunction(step.compensation):
                    await step.compensation(*step.compensation_args, **kwargs)
                else:
                    step.compensation(*step.compensation_args, **kwargs)

                step.compensated_at = datetime.now()
                state.compensated_steps.append(step.id)

                await self._record_event(state, "step.compensated", {
                    "step_id": step.id,
                })

                logger.info(f"Step {step.id} compensated successfully")

            except Exception as e:
                compensation_errors.append((step.id, str(e)))
                logger.error(f"Compensation failed for step {step.id}: {e}")

                await self._record_event(state, "compensation.failed", {
                    "step_id": step.id,
                    "error": str(e),
                })

        if compensation_errors:
            state.status = SagaStatus.PARTIALLY_COMPENSATED
            logger.error(
                "Saga compensation partially failed",
                errors=compensation_errors,
            )
        else:
            state.status = SagaStatus.COMPENSATED
            logger.info("Saga fully compensated")

        state.completed_at = datetime.now()

        await self._record_event(state, "compensation.completed", {
            "status": state.status.value,
            "compensated_steps": state.compensated_steps,
        })

    async def _record_event(
        self,
        state: SagaState,
        event_type: str,
        data: Dict[str, Any],
    ) -> None:
        """Record an event to the event store."""
        if self.event_store:
            try:
                from aion.automation.execution.event_store import EventType
                await self.event_store.append(
                    state.execution_id,
                    EventType.STATE_CHANGED,  # Use generic event type
                    {
                        "saga_event": event_type,
                        **data,
                    },
                )
            except Exception as e:
                logger.warning(f"Failed to record saga event: {e}")

    def get_execution(self, execution_id: str) -> Optional[SagaState]:
        """Get saga execution state."""
        return self._executions.get(execution_id)

    def on_step_completed(self, callback: Callable) -> None:
        """Register callback for step completion."""
        self._on_step_completed.append(callback)

    def on_step_failed(self, callback: Callable) -> None:
        """Register callback for step failure."""
        self._on_step_failed.append(callback)

    def on_saga_completed(self, callback: Callable) -> None:
        """Register callback for saga completion."""
        self._on_saga_completed.append(callback)

    def on_compensation_started(self, callback: Callable) -> None:
        """Register callback for compensation start."""
        self._on_compensation_started.append(callback)

    def get_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics."""
        executions = list(self._executions.values())
        return {
            "registered_sagas": len(self._definitions),
            "total_executions": len(executions),
            "by_status": {
                status.value: len([e for e in executions if e.status == status])
                for status in SagaStatus
            },
        }


# Builder for creating sagas with fluent API
class SagaBuilder:
    """Fluent builder for creating saga definitions."""

    def __init__(self, name: str):
        self.definition = SagaDefinition(
            id=str(uuid.uuid4()),
            name=name,
        )

    def with_description(self, description: str) -> "SagaBuilder":
        self.definition.description = description
        return self

    def with_timeout(self, seconds: int) -> "SagaBuilder":
        self.definition.timeout_seconds = seconds
        return self

    def parallel(self) -> "SagaBuilder":
        self.definition.parallel = True
        return self

    def step(
        self,
        name: str,
        action: Callable,
        compensation: Optional[Callable] = None,
        **kwargs,
    ) -> "SagaBuilder":
        self.definition.add_step(
            name=name,
            action=action,
            compensation=compensation,
            **kwargs,
        )
        return self

    def build(self) -> SagaDefinition:
        return self.definition


def create_saga(name: str) -> SagaBuilder:
    """Create a new saga with fluent API."""
    return SagaBuilder(name)
