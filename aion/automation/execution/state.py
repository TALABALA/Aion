"""
AION Execution State Manager

State management for workflow executions.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
import uuid

import structlog

from aion.automation.types import (
    WorkflowExecution,
    ExecutionStatus,
    StepResult,
)

logger = structlog.get_logger(__name__)


class ExecutionStateManager:
    """
    Manages execution state for workflows.

    Features:
    - Checkpoint/restore for fault tolerance
    - State persistence
    - State transitions
    - Event emission
    """

    def __init__(
        self,
        persistence_path: Optional[Path] = None,
        checkpoint_interval: int = 10,  # Checkpoint every N steps
    ):
        self.persistence_path = persistence_path
        self.checkpoint_interval = checkpoint_interval

        # Active states
        self._states: Dict[str, ExecutionState] = {}

        # Event handlers
        self._state_handlers: List[Callable] = []

        self._initialized = False
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize the state manager."""
        if self._initialized:
            return

        # Load any persisted states
        if self.persistence_path:
            await self._load_checkpoints()

        self._initialized = True
        logger.info("Execution state manager initialized")

    async def shutdown(self) -> None:
        """Shutdown the state manager."""
        # Checkpoint all active states
        for state in self._states.values():
            await self._checkpoint(state)

        self._initialized = False

    # === State Management ===

    async def create_state(
        self,
        execution: WorkflowExecution,
    ) -> "ExecutionState":
        """Create state for an execution."""
        async with self._lock:
            state = ExecutionState(
                execution_id=execution.id,
                workflow_id=execution.workflow_id,
                context_data={},
            )
            self._states[execution.id] = state

            logger.debug("state_created", execution_id=execution.id)
            return state

    async def get_state(
        self,
        execution_id: str,
    ) -> Optional["ExecutionState"]:
        """Get state for an execution."""
        return self._states.get(execution_id)

    async def update_state(
        self,
        execution_id: str,
        step_id: str,
        status: ExecutionStatus,
        result: Optional[StepResult] = None,
        context_updates: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update execution state."""
        async with self._lock:
            state = self._states.get(execution_id)
            if not state:
                return

            # Update state
            state.current_step_id = step_id
            state.status = status
            state.steps_completed += 1
            state.updated_at = datetime.now()

            if result:
                state.step_results[step_id] = result

            if context_updates:
                state.context_data.update(context_updates)

            # Fire handlers
            await self._fire_handlers(state)

            # Checkpoint if needed
            if state.steps_completed % self.checkpoint_interval == 0:
                await self._checkpoint(state)

    async def complete_state(
        self,
        execution_id: str,
        outputs: Dict[str, Any] = None,
    ) -> None:
        """Mark state as completed."""
        async with self._lock:
            state = self._states.get(execution_id)
            if not state:
                return

            state.status = ExecutionStatus.COMPLETED
            state.outputs = outputs or {}
            state.completed_at = datetime.now()
            state.updated_at = datetime.now()

            # Final checkpoint
            await self._checkpoint(state)

            # Clean up after checkpoint
            if execution_id in self._states:
                del self._states[execution_id]

    async def fail_state(
        self,
        execution_id: str,
        error: str,
        step_id: str = None,
    ) -> None:
        """Mark state as failed."""
        async with self._lock:
            state = self._states.get(execution_id)
            if not state:
                return

            state.status = ExecutionStatus.FAILED
            state.error = error
            state.error_step_id = step_id
            state.completed_at = datetime.now()
            state.updated_at = datetime.now()

            # Final checkpoint
            await self._checkpoint(state)

    async def delete_state(
        self,
        execution_id: str,
    ) -> bool:
        """Delete state for an execution."""
        async with self._lock:
            if execution_id in self._states:
                del self._states[execution_id]

                # Delete checkpoint
                if self.persistence_path:
                    checkpoint_file = self.persistence_path / f"{execution_id}.json"
                    if checkpoint_file.exists():
                        checkpoint_file.unlink()

                return True
            return False

    # === Checkpoint/Restore ===

    async def _checkpoint(self, state: "ExecutionState") -> None:
        """Checkpoint state to disk."""
        if not self.persistence_path:
            return

        try:
            self.persistence_path.mkdir(parents=True, exist_ok=True)
            checkpoint_file = self.persistence_path / f"{state.execution_id}.json"

            with open(checkpoint_file, "w") as f:
                json.dump(state.to_dict(), f, indent=2, default=str)

            logger.debug("state_checkpointed", execution_id=state.execution_id)

        except Exception as e:
            logger.error("checkpoint_error", execution_id=state.execution_id, error=str(e))

    async def _load_checkpoints(self) -> None:
        """Load checkpointed states from disk."""
        if not self.persistence_path or not self.persistence_path.exists():
            return

        try:
            for checkpoint_file in self.persistence_path.glob("*.json"):
                try:
                    with open(checkpoint_file, "r") as f:
                        data = json.load(f)
                    state = ExecutionState.from_dict(data)
                    self._states[state.execution_id] = state
                except Exception as e:
                    logger.error("load_checkpoint_error", file=str(checkpoint_file), error=str(e))

            logger.info("checkpoints_loaded", count=len(self._states))

        except Exception as e:
            logger.error("load_checkpoints_error", error=str(e))

    async def restore_execution(
        self,
        execution_id: str,
    ) -> Optional["ExecutionState"]:
        """Restore execution from checkpoint."""
        state = self._states.get(execution_id)
        if state:
            logger.info("execution_restored", execution_id=execution_id)
            return state
        return None

    # === Event Handlers ===

    def on_state_change(self, handler: Callable) -> None:
        """Register state change handler."""
        self._state_handlers.append(handler)

    async def _fire_handlers(self, state: "ExecutionState") -> None:
        """Fire state change handlers."""
        for handler in self._state_handlers:
            try:
                result = handler(state)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error("handler_error", error=str(e))

    # === Statistics ===

    def get_stats(self) -> Dict[str, Any]:
        """Get state manager statistics."""
        status_counts = {}
        for state in self._states.values():
            status = state.status.value
            status_counts[status] = status_counts.get(status, 0) + 1

        return {
            "active_states": len(self._states),
            "by_status": status_counts,
        }


@dataclass
class ExecutionState:
    """State for a single execution."""
    execution_id: str
    workflow_id: str

    # Current state
    status: ExecutionStatus = ExecutionStatus.PENDING
    current_step_id: Optional[str] = None

    # Progress
    steps_completed: int = 0
    step_results: Dict[str, StepResult] = None

    # Context
    context_data: Dict[str, Any] = None

    # Outputs
    outputs: Dict[str, Any] = None

    # Error info
    error: Optional[str] = None
    error_step_id: Optional[str] = None

    # Timestamps
    created_at: datetime = None
    updated_at: datetime = None
    completed_at: Optional[datetime] = None

    def __post_init__(self):
        if self.step_results is None:
            self.step_results = {}
        if self.context_data is None:
            self.context_data = {}
        if self.outputs is None:
            self.outputs = {}
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "execution_id": self.execution_id,
            "workflow_id": self.workflow_id,
            "status": self.status.value,
            "current_step_id": self.current_step_id,
            "steps_completed": self.steps_completed,
            "step_results": {k: v.to_dict() for k, v in self.step_results.items()},
            "context_data": self.context_data,
            "outputs": self.outputs,
            "error": self.error,
            "error_step_id": self.error_step_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionState":
        """Create from dictionary."""
        state = cls(
            execution_id=data["execution_id"],
            workflow_id=data["workflow_id"],
            status=ExecutionStatus(data.get("status", "pending")),
            current_step_id=data.get("current_step_id"),
            steps_completed=data.get("steps_completed", 0),
            context_data=data.get("context_data", {}),
            outputs=data.get("outputs", {}),
            error=data.get("error"),
            error_step_id=data.get("error_step_id"),
        )

        # Parse timestamps
        if data.get("created_at"):
            state.created_at = datetime.fromisoformat(data["created_at"])
        if data.get("updated_at"):
            state.updated_at = datetime.fromisoformat(data["updated_at"])
        if data.get("completed_at"):
            state.completed_at = datetime.fromisoformat(data["completed_at"])

        return state


# Import dataclass here to avoid circular import
from dataclasses import dataclass, field
