"""
AION Plan Executor

High-level executor for complex plan execution with:
- State machine management
- Error recovery
- Progress tracking
- Event emission
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Optional

import structlog

from aion.systems.planning.graph import (
    PlanningGraph,
    ExecutionPlan,
    PlanNode,
    NodeStatus,
    NodeType,
)

logger = structlog.get_logger(__name__)


class ExecutorState(Enum):
    """State of the executor."""
    IDLE = auto()
    RUNNING = auto()
    PAUSED = auto()
    STOPPING = auto()
    STOPPED = auto()
    ERROR = auto()


@dataclass
class ExecutionProgress:
    """Progress tracking for plan execution."""
    plan_id: str
    total_nodes: int
    completed_nodes: int
    failed_nodes: int
    current_nodes: list[str]
    elapsed_ms: float
    estimated_remaining_ms: float

    @property
    def progress_percent(self) -> float:
        """Get completion percentage."""
        if self.total_nodes == 0:
            return 100.0
        return (self.completed_nodes / self.total_nodes) * 100


@dataclass
class ExecutionEvent:
    """An event during plan execution."""
    type: str
    timestamp: datetime
    plan_id: str
    node_id: Optional[str]
    data: dict[str, Any] = field(default_factory=dict)


class PlanExecutor:
    """
    High-level plan executor with advanced features.

    Features:
    - Pause/resume execution
    - Real-time progress tracking
    - Event streaming
    - Error recovery strategies
    """

    def __init__(
        self,
        planning_graph: PlanningGraph,
        max_concurrent: int = 10,
    ):
        self.graph = planning_graph
        self.max_concurrent = max_concurrent

        self._state = ExecutorState.IDLE
        self._current_plan_id: Optional[str] = None
        self._pause_event = asyncio.Event()
        self._pause_event.set()  # Not paused initially
        self._stop_requested = False

        # Event subscribers
        self._event_handlers: list[Callable[[ExecutionEvent], None]] = []

        # Execution tracking
        self._start_time: Optional[datetime] = None
        self._node_durations: dict[str, float] = {}

    @property
    def state(self) -> ExecutorState:
        """Get current executor state."""
        return self._state

    def subscribe(self, handler: Callable[[ExecutionEvent], None]) -> None:
        """Subscribe to execution events."""
        self._event_handlers.append(handler)

    def unsubscribe(self, handler: Callable[[ExecutionEvent], None]) -> None:
        """Unsubscribe from execution events."""
        if handler in self._event_handlers:
            self._event_handlers.remove(handler)

    def _emit_event(
        self,
        event_type: str,
        node_id: Optional[str] = None,
        data: Optional[dict[str, Any]] = None,
    ) -> None:
        """Emit an execution event."""
        if not self._current_plan_id:
            return

        event = ExecutionEvent(
            type=event_type,
            timestamp=datetime.now(),
            plan_id=self._current_plan_id,
            node_id=node_id,
            data=data or {},
        )

        for handler in self._event_handlers:
            try:
                handler(event)
            except Exception as e:
                logger.warning("Event handler error", error=str(e))

    async def execute(
        self,
        plan_id: str,
        context: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Execute a plan with full feature set.

        Args:
            plan_id: Plan to execute
            context: Execution context

        Returns:
            Execution results
        """
        if self._state not in (ExecutorState.IDLE, ExecutorState.STOPPED):
            raise RuntimeError(f"Executor not ready, current state: {self._state}")

        plan = self.graph.get_plan(plan_id)
        if not plan:
            raise KeyError(f"Plan not found: {plan_id}")

        self._state = ExecutorState.RUNNING
        self._current_plan_id = plan_id
        self._start_time = datetime.now()
        self._stop_requested = False

        self._emit_event("execution_started")

        try:
            result = await self.graph.execute_plan(
                plan_id,
                context=context,
                on_node_start=self._on_node_start,
                on_node_complete=self._on_node_complete,
            )

            self._state = ExecutorState.STOPPED
            self._emit_event("execution_completed", data=result)

            return result

        except Exception as e:
            self._state = ExecutorState.ERROR
            self._emit_event("execution_failed", data={"error": str(e)})
            raise

        finally:
            self._current_plan_id = None

    async def _on_node_start(self, node: PlanNode) -> None:
        """Called when a node starts execution."""
        # Check for pause
        await self._pause_event.wait()

        # Check for stop
        if self._stop_requested:
            raise asyncio.CancelledError("Execution stopped by user")

        self._emit_event("node_started", node_id=node.id)

    async def _on_node_complete(self, node: PlanNode) -> None:
        """Called when a node completes execution."""
        duration = node.duration_ms()
        self._node_durations[node.id] = duration

        self._emit_event(
            "node_completed",
            node_id=node.id,
            data={
                "status": node.status.name,
                "duration_ms": duration,
                "result_preview": str(node.result)[:200] if node.result else None,
            },
        )

    def pause(self) -> None:
        """Pause execution."""
        if self._state == ExecutorState.RUNNING:
            self._pause_event.clear()
            self._state = ExecutorState.PAUSED
            self._emit_event("execution_paused")
            logger.info("Execution paused")

    def resume(self) -> None:
        """Resume execution."""
        if self._state == ExecutorState.PAUSED:
            self._pause_event.set()
            self._state = ExecutorState.RUNNING
            self._emit_event("execution_resumed")
            logger.info("Execution resumed")

    def stop(self) -> None:
        """Request execution stop."""
        if self._state in (ExecutorState.RUNNING, ExecutorState.PAUSED):
            self._stop_requested = True
            self._pause_event.set()  # Unblock if paused
            self._state = ExecutorState.STOPPING
            self._emit_event("execution_stopping")
            logger.info("Execution stop requested")

    def get_progress(self) -> Optional[ExecutionProgress]:
        """Get current execution progress."""
        if not self._current_plan_id:
            return None

        plan = self.graph.get_plan(self._current_plan_id)
        if not plan:
            return None

        # Count nodes by status
        completed = sum(
            1 for n in plan.nodes.values()
            if n.status == NodeStatus.COMPLETED
        )
        failed = sum(
            1 for n in plan.nodes.values()
            if n.status == NodeStatus.FAILED
        )
        running = [
            n.id for n in plan.nodes.values()
            if n.status == NodeStatus.RUNNING
        ]

        # Calculate timing
        elapsed_ms = 0.0
        if self._start_time:
            elapsed_ms = (datetime.now() - self._start_time).total_seconds() * 1000

        # Estimate remaining
        avg_duration = (
            sum(self._node_durations.values()) / len(self._node_durations)
            if self._node_durations else 1000.0
        )
        remaining_nodes = len(plan.nodes) - completed - failed
        estimated_remaining_ms = remaining_nodes * avg_duration

        return ExecutionProgress(
            plan_id=self._current_plan_id,
            total_nodes=len(plan.nodes),
            completed_nodes=completed,
            failed_nodes=failed,
            current_nodes=running,
            elapsed_ms=elapsed_ms,
            estimated_remaining_ms=estimated_remaining_ms,
        )


class RecoveryStrategy(Enum):
    """Error recovery strategies."""
    RETRY = auto()
    SKIP = auto()
    ABORT = auto()
    ROLLBACK = auto()


@dataclass
class RecoveryPolicy:
    """Policy for handling errors during execution."""
    default_strategy: RecoveryStrategy = RecoveryStrategy.RETRY
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    node_strategies: dict[str, RecoveryStrategy] = field(default_factory=dict)

    def get_strategy(self, node_id: str) -> RecoveryStrategy:
        """Get recovery strategy for a node."""
        return self.node_strategies.get(node_id, self.default_strategy)


class ResilientExecutor(PlanExecutor):
    """
    Plan executor with advanced error recovery.

    Extends PlanExecutor with:
    - Configurable recovery policies
    - Automatic retry with backoff
    - Checkpoint-based rollback
    """

    def __init__(
        self,
        planning_graph: PlanningGraph,
        recovery_policy: Optional[RecoveryPolicy] = None,
        **kwargs,
    ):
        super().__init__(planning_graph, **kwargs)
        self.policy = recovery_policy or RecoveryPolicy()

    async def execute_with_recovery(
        self,
        plan_id: str,
        context: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Execute a plan with automatic error recovery.

        Args:
            plan_id: Plan to execute
            context: Execution context

        Returns:
            Execution results
        """
        attempt = 0
        last_error = None

        while attempt < self.policy.max_retries:
            try:
                return await self.execute(plan_id, context)

            except Exception as e:
                last_error = e
                attempt += 1

                strategy = self.policy.default_strategy

                if strategy == RecoveryStrategy.ABORT:
                    raise

                elif strategy == RecoveryStrategy.SKIP:
                    logger.warning("Skipping failed execution", error=str(e))
                    break

                elif strategy == RecoveryStrategy.ROLLBACK:
                    plan = self.graph.get_plan(plan_id)
                    if plan and plan.checkpoints:
                        last_checkpoint = plan.checkpoints[-1]
                        self.graph.rollback_to_checkpoint(plan_id, last_checkpoint.id)
                        logger.info("Rolled back to checkpoint", checkpoint=last_checkpoint.id)

                elif strategy == RecoveryStrategy.RETRY:
                    delay = self.policy.retry_delay_seconds * (2 ** (attempt - 1))
                    logger.warning(
                        "Retrying execution",
                        attempt=attempt,
                        delay=delay,
                        error=str(e),
                    )
                    await asyncio.sleep(delay)

        if last_error:
            raise last_error

        return {"status": "skipped"}
