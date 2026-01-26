"""
AION Compensation Manager

Manages compensation actions for saga transactions.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import structlog

logger = structlog.get_logger(__name__)


class CompensationStatus(str, Enum):
    """Status of a compensation action."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class CompensationAction:
    """
    A compensation action to undo a completed operation.
    """
    id: str
    name: str
    step_id: str

    # The compensation function
    action: Callable[..., Any]
    action_args: tuple = field(default_factory=tuple)
    action_kwargs: Dict[str, Any] = field(default_factory=dict)

    # The original operation's result (for context)
    original_result: Any = None
    original_context: Dict[str, Any] = field(default_factory=dict)

    # Configuration
    timeout_seconds: Optional[int] = None
    retry_count: int = 3
    retry_delay: float = 1.0
    is_idempotent: bool = True  # Safe to retry

    # State
    status: CompensationStatus = CompensationStatus.PENDING
    error: Optional[str] = None
    attempts: int = 0

    # Timing
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "step_id": self.step_id,
            "status": self.status.value,
            "error": self.error,
            "attempts": self.attempts,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


@dataclass
class CompensationResult:
    """Result of executing a compensation action."""
    action_id: str
    success: bool
    error: Optional[str] = None
    attempts: int = 1
    duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_id": self.action_id,
            "success": self.success,
            "error": self.error,
            "attempts": self.attempts,
            "duration_ms": self.duration_ms,
        }


class CompensationManager:
    """
    Manages compensation actions for distributed transactions.

    Features:
    - Queued compensation execution
    - Retry with exponential backoff
    - Idempotency support
    - Dead letter handling
    - Persistence support
    """

    def __init__(
        self,
        max_concurrent: int = 10,
        default_timeout: int = 60,
        dead_letter_threshold: int = 3,
    ):
        self.max_concurrent = max_concurrent
        self.default_timeout = default_timeout
        self.dead_letter_threshold = dead_letter_threshold

        self._semaphore = asyncio.Semaphore(max_concurrent)

        # Pending compensations by execution
        self._pending: Dict[str, List[CompensationAction]] = {}

        # Completed compensations
        self._completed: Dict[str, List[CompensationResult]] = {}

        # Dead letter queue (failed after max retries)
        self._dead_letters: List[CompensationAction] = []

        # Callbacks
        self._on_compensation_completed: List[Callable] = []
        self._on_compensation_failed: List[Callable] = []

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the compensation manager."""
        if self._initialized:
            return

        self._initialized = True
        logger.info("Compensation manager initialized")

    async def shutdown(self) -> None:
        """Shutdown the compensation manager."""
        self._initialized = False
        logger.info("Compensation manager shutdown")

    def register_compensation(
        self,
        execution_id: str,
        step_id: str,
        name: str,
        action: Callable,
        original_result: Any = None,
        original_context: Optional[Dict] = None,
        action_args: tuple = (),
        action_kwargs: Optional[Dict] = None,
        timeout_seconds: Optional[int] = None,
        retry_count: int = 3,
        is_idempotent: bool = True,
    ) -> CompensationAction:
        """
        Register a compensation action for later execution.

        Args:
            execution_id: The saga/workflow execution ID
            step_id: The step this compensates
            name: Human-readable name
            action: The compensation function
            original_result: Result from the original operation
            original_context: Context from the original operation
            action_args: Positional arguments for action
            action_kwargs: Keyword arguments for action
            timeout_seconds: Timeout for the compensation
            retry_count: Number of retries
            is_idempotent: Whether safe to retry

        Returns:
            CompensationAction
        """
        compensation = CompensationAction(
            id=str(uuid.uuid4()),
            name=name,
            step_id=step_id,
            action=action,
            action_args=action_args,
            action_kwargs=action_kwargs or {},
            original_result=original_result,
            original_context=original_context or {},
            timeout_seconds=timeout_seconds or self.default_timeout,
            retry_count=retry_count,
            is_idempotent=is_idempotent,
        )

        if execution_id not in self._pending:
            self._pending[execution_id] = []

        self._pending[execution_id].append(compensation)

        logger.debug(
            "Registered compensation",
            execution_id=execution_id,
            step_id=step_id,
            compensation_id=compensation.id,
        )

        return compensation

    async def execute_all(
        self,
        execution_id: str,
        reverse_order: bool = True,
    ) -> List[CompensationResult]:
        """
        Execute all pending compensations for an execution.

        Args:
            execution_id: The execution to compensate
            reverse_order: Whether to execute in reverse order (LIFO)

        Returns:
            List of compensation results
        """
        compensations = self._pending.get(execution_id, [])
        if not compensations:
            return []

        if reverse_order:
            compensations = list(reversed(compensations))

        results = []
        for compensation in compensations:
            result = await self.execute_compensation(compensation)
            results.append(result)

            # Stop if non-idempotent compensation fails
            if not result.success and not compensation.is_idempotent:
                logger.error(
                    "Non-idempotent compensation failed, stopping",
                    compensation_id=compensation.id,
                )
                break

        # Move to completed
        if execution_id in self._pending:
            del self._pending[execution_id]

        if execution_id not in self._completed:
            self._completed[execution_id] = []
        self._completed[execution_id].extend(results)

        return results

    async def execute_compensation(
        self,
        compensation: CompensationAction,
    ) -> CompensationResult:
        """
        Execute a single compensation action.

        Args:
            compensation: The compensation to execute

        Returns:
            CompensationResult
        """
        async with self._semaphore:
            import time
            start_time = time.time()

            compensation.status = CompensationStatus.RUNNING
            compensation.started_at = datetime.now()

            last_error = None

            for attempt in range(compensation.retry_count + 1):
                compensation.attempts = attempt + 1

                try:
                    # Prepare kwargs with context
                    kwargs = {
                        **compensation.action_kwargs,
                        "original_result": compensation.original_result,
                        "context": compensation.original_context,
                    }

                    if compensation.timeout_seconds:
                        if asyncio.iscoroutinefunction(compensation.action):
                            await asyncio.wait_for(
                                compensation.action(*compensation.action_args, **kwargs),
                                timeout=compensation.timeout_seconds,
                            )
                        else:
                            await asyncio.wait_for(
                                asyncio.to_thread(
                                    compensation.action,
                                    *compensation.action_args,
                                    **kwargs,
                                ),
                                timeout=compensation.timeout_seconds,
                            )
                    else:
                        if asyncio.iscoroutinefunction(compensation.action):
                            await compensation.action(*compensation.action_args, **kwargs)
                        else:
                            compensation.action(*compensation.action_args, **kwargs)

                    # Success
                    compensation.status = CompensationStatus.COMPLETED
                    compensation.completed_at = datetime.now()

                    duration_ms = (time.time() - start_time) * 1000

                    result = CompensationResult(
                        action_id=compensation.id,
                        success=True,
                        attempts=attempt + 1,
                        duration_ms=duration_ms,
                    )

                    for callback in self._on_compensation_completed:
                        try:
                            await callback(compensation, result)
                        except Exception:
                            pass

                    logger.info(
                        "Compensation completed",
                        compensation_id=compensation.id,
                        attempts=attempt + 1,
                        duration_ms=duration_ms,
                    )

                    return result

                except Exception as e:
                    last_error = e
                    logger.warning(
                        f"Compensation attempt {attempt + 1} failed",
                        compensation_id=compensation.id,
                        error=str(e),
                    )

                    if attempt < compensation.retry_count:
                        # Exponential backoff
                        delay = compensation.retry_delay * (2 ** attempt)
                        await asyncio.sleep(delay)

            # All retries exhausted
            compensation.status = CompensationStatus.FAILED
            compensation.error = str(last_error)
            compensation.completed_at = datetime.now()

            duration_ms = (time.time() - start_time) * 1000

            result = CompensationResult(
                action_id=compensation.id,
                success=False,
                error=str(last_error),
                attempts=compensation.retry_count + 1,
                duration_ms=duration_ms,
            )

            # Add to dead letter queue
            if compensation.attempts >= self.dead_letter_threshold:
                self._dead_letters.append(compensation)
                logger.error(
                    "Compensation moved to dead letter queue",
                    compensation_id=compensation.id,
                )

            for callback in self._on_compensation_failed:
                try:
                    await callback(compensation, result)
                except Exception:
                    pass

            return result

    def get_pending(self, execution_id: str) -> List[CompensationAction]:
        """Get pending compensations for an execution."""
        return self._pending.get(execution_id, [])

    def get_completed(self, execution_id: str) -> List[CompensationResult]:
        """Get completed compensations for an execution."""
        return self._completed.get(execution_id, [])

    def get_dead_letters(self) -> List[CompensationAction]:
        """Get all dead letter compensations."""
        return self._dead_letters.copy()

    async def retry_dead_letter(self, compensation_id: str) -> Optional[CompensationResult]:
        """Retry a compensation from the dead letter queue."""
        compensation = next(
            (c for c in self._dead_letters if c.id == compensation_id),
            None,
        )

        if not compensation:
            return None

        # Reset status
        compensation.status = CompensationStatus.PENDING
        compensation.attempts = 0
        compensation.error = None

        # Remove from dead letters
        self._dead_letters = [c for c in self._dead_letters if c.id != compensation_id]

        # Execute
        return await self.execute_compensation(compensation)

    def on_compensation_completed(self, callback: Callable) -> None:
        """Register callback for successful compensation."""
        self._on_compensation_completed.append(callback)

    def on_compensation_failed(self, callback: Callable) -> None:
        """Register callback for failed compensation."""
        self._on_compensation_failed.append(callback)

    def get_stats(self) -> Dict[str, Any]:
        """Get compensation manager statistics."""
        all_pending = sum(len(v) for v in self._pending.values())
        all_completed = sum(len(v) for v in self._completed.values())
        successful = sum(
            len([r for r in results if r.success])
            for results in self._completed.values()
        )

        return {
            "pending_count": all_pending,
            "completed_count": all_completed,
            "successful_count": successful,
            "failed_count": all_completed - successful,
            "dead_letter_count": len(self._dead_letters),
            "executions_with_pending": len(self._pending),
        }


# Decorator for creating compensatable operations
def compensatable(
    compensation: Callable,
    name: Optional[str] = None,
    is_idempotent: bool = True,
):
    """
    Decorator to mark a function as compensatable.

    Usage:
        @compensatable(undo_create_user)
        async def create_user(data):
            ...

    The compensation function receives the original result
    as an argument.
    """
    def decorator(func):
        func._compensation = compensation
        func._compensation_name = name or f"compensate_{func.__name__}"
        func._is_idempotent = is_idempotent
        return func
    return decorator


class CompensationContext:
    """
    Context manager for automatic compensation registration.

    Usage:
        async with CompensationContext(manager, execution_id) as ctx:
            result = await create_order(data)
            ctx.register(
                "Create Order",
                cancel_order,
                original_result=result,
            )

            result2 = await reserve_inventory(data)
            ctx.register(
                "Reserve Inventory",
                release_inventory,
                original_result=result2,
            )

            # If anything fails, compensations run automatically
    """

    def __init__(
        self,
        manager: CompensationManager,
        execution_id: str,
    ):
        self.manager = manager
        self.execution_id = execution_id
        self._step_counter = 0
        self._should_compensate = False

    def register(
        self,
        name: str,
        compensation: Callable,
        original_result: Any = None,
        **kwargs,
    ) -> CompensationAction:
        """Register a compensation action."""
        self._step_counter += 1
        return self.manager.register_compensation(
            execution_id=self.execution_id,
            step_id=f"step_{self._step_counter}",
            name=name,
            action=compensation,
            original_result=original_result,
            **kwargs,
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            # Exception occurred, run compensations
            logger.info(
                "Exception occurred, running compensations",
                execution_id=self.execution_id,
                error=str(exc_val),
            )
            await self.manager.execute_all(self.execution_id)

        return False  # Don't suppress the exception
