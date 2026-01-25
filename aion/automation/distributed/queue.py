"""
AION Distributed Task Queue

Core task queue implementation with pluggable backends.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Union

import structlog

logger = structlog.get_logger(__name__)


class TaskStatus(str, Enum):
    """Task execution status."""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"
    TIMEOUT = "timeout"


class TaskPriority(int, Enum):
    """Task priority levels."""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4


@dataclass
class Task:
    """
    A distributed task to be executed.

    Tasks are the unit of work distributed across workers.
    """
    id: str
    name: str
    payload: Dict[str, Any]
    priority: TaskPriority = TaskPriority.NORMAL

    # Execution tracking
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Retry configuration
    max_retries: int = 3
    retry_count: int = 0
    retry_delay_seconds: int = 5
    retry_backoff_multiplier: float = 2.0

    # Timeout
    timeout_seconds: Optional[int] = None

    # Routing
    queue_name: str = "default"
    routing_key: Optional[str] = None
    worker_affinity: Optional[str] = None  # Prefer specific worker

    # Dependencies
    depends_on: List[str] = field(default_factory=list)
    parent_task_id: Optional[str] = None

    # Metadata
    correlation_id: Optional[str] = None
    execution_id: Optional[str] = None  # Workflow execution ID
    step_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Result
    result: Optional[Any] = None
    error: Optional[str] = None
    error_traceback: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize task to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "payload": self.payload,
            "priority": self.priority.value,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "max_retries": self.max_retries,
            "retry_count": self.retry_count,
            "retry_delay_seconds": self.retry_delay_seconds,
            "retry_backoff_multiplier": self.retry_backoff_multiplier,
            "timeout_seconds": self.timeout_seconds,
            "queue_name": self.queue_name,
            "routing_key": self.routing_key,
            "worker_affinity": self.worker_affinity,
            "depends_on": self.depends_on,
            "parent_task_id": self.parent_task_id,
            "correlation_id": self.correlation_id,
            "execution_id": self.execution_id,
            "step_id": self.step_id,
            "metadata": self.metadata,
            "result": self.result,
            "error": self.error,
            "error_traceback": self.error_traceback,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Task":
        """Deserialize task from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            payload=data["payload"],
            priority=TaskPriority(data.get("priority", TaskPriority.NORMAL.value)),
            status=TaskStatus(data.get("status", TaskStatus.PENDING.value)),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(),
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            max_retries=data.get("max_retries", 3),
            retry_count=data.get("retry_count", 0),
            retry_delay_seconds=data.get("retry_delay_seconds", 5),
            retry_backoff_multiplier=data.get("retry_backoff_multiplier", 2.0),
            timeout_seconds=data.get("timeout_seconds"),
            queue_name=data.get("queue_name", "default"),
            routing_key=data.get("routing_key"),
            worker_affinity=data.get("worker_affinity"),
            depends_on=data.get("depends_on", []),
            parent_task_id=data.get("parent_task_id"),
            correlation_id=data.get("correlation_id"),
            execution_id=data.get("execution_id"),
            step_id=data.get("step_id"),
            metadata=data.get("metadata", {}),
            result=data.get("result"),
            error=data.get("error"),
            error_traceback=data.get("error_traceback"),
        )

    def should_retry(self) -> bool:
        """Check if task should be retried."""
        return (
            self.status == TaskStatus.FAILED and
            self.retry_count < self.max_retries
        )

    def get_retry_delay(self) -> float:
        """Get delay before next retry with exponential backoff."""
        return self.retry_delay_seconds * (
            self.retry_backoff_multiplier ** self.retry_count
        )


@dataclass
class TaskResult:
    """Result of task execution."""
    task_id: str
    status: TaskStatus
    result: Optional[Any] = None
    error: Optional[str] = None
    error_traceback: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    worker_id: Optional[str] = None
    execution_time_ms: float = 0.0

    @property
    def success(self) -> bool:
        return self.status == TaskStatus.COMPLETED


class TaskQueue:
    """
    Distributed task queue with pluggable backends.

    Features:
    - Priority-based queuing
    - Task dependencies
    - Retries with exponential backoff
    - Dead letter queue
    - Task deduplication
    - Delayed execution
    """

    def __init__(
        self,
        backend: "QueueBackend",
        dead_letter_queue: Optional["QueueBackend"] = None,
        max_dead_letters: int = 1000,
    ):
        from aion.automation.distributed.backends import QueueBackend
        self.backend = backend
        self.dead_letter_queue = dead_letter_queue
        self.max_dead_letters = max_dead_letters

        # Task handlers
        self._handlers: Dict[str, Callable] = {}

        # Active tasks
        self._active_tasks: Dict[str, Task] = {}

        # Callbacks
        self._on_task_started: List[Callable] = []
        self._on_task_completed: List[Callable] = []
        self._on_task_failed: List[Callable] = []

        # Deduplication
        self._recent_task_ids: Set[str] = set()
        self._dedup_window_seconds: int = 300

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the task queue."""
        if self._initialized:
            return

        await self.backend.initialize()
        if self.dead_letter_queue:
            await self.dead_letter_queue.initialize()

        self._initialized = True
        logger.info("Task queue initialized")

    async def shutdown(self) -> None:
        """Shutdown the task queue."""
        await self.backend.shutdown()
        if self.dead_letter_queue:
            await self.dead_letter_queue.shutdown()

        self._initialized = False
        logger.info("Task queue shutdown")

    def register_handler(
        self,
        task_name: str,
        handler: Callable,
    ) -> None:
        """Register a handler for a task type."""
        self._handlers[task_name] = handler
        logger.debug(f"Registered handler for task: {task_name}")

    async def enqueue(
        self,
        name: str,
        payload: Dict[str, Any],
        priority: TaskPriority = TaskPriority.NORMAL,
        queue_name: str = "default",
        delay_seconds: Optional[int] = None,
        timeout_seconds: Optional[int] = None,
        max_retries: int = 3,
        depends_on: Optional[List[str]] = None,
        correlation_id: Optional[str] = None,
        execution_id: Optional[str] = None,
        step_id: Optional[str] = None,
        dedup_key: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Task:
        """
        Enqueue a task for execution.

        Args:
            name: Task name (must have registered handler)
            payload: Task payload
            priority: Execution priority
            queue_name: Target queue
            delay_seconds: Delay before execution
            timeout_seconds: Task timeout
            max_retries: Maximum retry attempts
            depends_on: Task IDs this task depends on
            correlation_id: Correlation ID for tracking
            execution_id: Associated workflow execution
            step_id: Associated workflow step
            dedup_key: Key for deduplication
            metadata: Additional metadata

        Returns:
            Created task
        """
        # Deduplication check
        if dedup_key and dedup_key in self._recent_task_ids:
            raise ValueError(f"Duplicate task with key: {dedup_key}")

        task = Task(
            id=str(uuid.uuid4()),
            name=name,
            payload=payload,
            priority=priority,
            queue_name=queue_name,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            depends_on=depends_on or [],
            correlation_id=correlation_id,
            execution_id=execution_id,
            step_id=step_id,
            metadata=metadata or {},
        )

        # Handle delayed execution
        if delay_seconds:
            await self.backend.enqueue_delayed(task, delay_seconds)
        else:
            await self.backend.enqueue(task)

        # Track for deduplication
        if dedup_key:
            self._recent_task_ids.add(dedup_key)
            # Schedule cleanup
            asyncio.create_task(
                self._cleanup_dedup_key(dedup_key, self._dedup_window_seconds)
            )

        logger.info(
            "Task enqueued",
            task_id=task.id,
            name=name,
            queue=queue_name,
            priority=priority.name,
        )

        return task

    async def _cleanup_dedup_key(self, key: str, delay: int) -> None:
        """Remove dedup key after window expires."""
        await asyncio.sleep(delay)
        self._recent_task_ids.discard(key)

    async def dequeue(
        self,
        queue_name: str = "default",
        timeout_seconds: int = 30,
    ) -> Optional[Task]:
        """
        Dequeue a task for execution.

        Uses blocking pop with timeout for efficiency.
        """
        return await self.backend.dequeue(queue_name, timeout_seconds)

    async def complete(
        self,
        task_id: str,
        result: Any = None,
        worker_id: Optional[str] = None,
    ) -> TaskResult:
        """Mark a task as completed."""
        task = await self.backend.get_task(task_id)
        if not task:
            raise ValueError(f"Task not found: {task_id}")

        task.status = TaskStatus.COMPLETED
        task.completed_at = datetime.now()
        task.result = result

        await self.backend.update_task(task)

        result_obj = TaskResult(
            task_id=task_id,
            status=TaskStatus.COMPLETED,
            result=result,
            started_at=task.started_at,
            completed_at=task.completed_at,
            worker_id=worker_id,
            execution_time_ms=(
                (task.completed_at - task.started_at).total_seconds() * 1000
                if task.started_at and task.completed_at else 0
            ),
        )

        # Notify callbacks
        for callback in self._on_task_completed:
            try:
                await callback(task, result_obj)
            except Exception as e:
                logger.error("Task completion callback failed", error=str(e))

        logger.info(
            "Task completed",
            task_id=task_id,
            execution_time_ms=result_obj.execution_time_ms,
        )

        return result_obj

    async def fail(
        self,
        task_id: str,
        error: str,
        error_traceback: Optional[str] = None,
        worker_id: Optional[str] = None,
    ) -> TaskResult:
        """Mark a task as failed, potentially retrying."""
        task = await self.backend.get_task(task_id)
        if not task:
            raise ValueError(f"Task not found: {task_id}")

        task.error = error
        task.error_traceback = error_traceback

        if task.should_retry():
            # Schedule retry
            task.status = TaskStatus.RETRYING
            task.retry_count += 1
            delay = task.get_retry_delay()

            await self.backend.update_task(task)
            await self.backend.enqueue_delayed(task, int(delay))

            logger.warning(
                "Task failed, scheduling retry",
                task_id=task_id,
                retry_count=task.retry_count,
                delay_seconds=delay,
                error=error,
            )
        else:
            # Move to dead letter queue
            task.status = TaskStatus.FAILED
            task.completed_at = datetime.now()
            await self.backend.update_task(task)

            if self.dead_letter_queue:
                await self.dead_letter_queue.enqueue(task)

            logger.error(
                "Task failed permanently",
                task_id=task_id,
                retries=task.retry_count,
                error=error,
            )

        result_obj = TaskResult(
            task_id=task_id,
            status=task.status,
            error=error,
            error_traceback=error_traceback,
            started_at=task.started_at,
            completed_at=task.completed_at,
            worker_id=worker_id,
        )

        # Notify callbacks
        for callback in self._on_task_failed:
            try:
                await callback(task, result_obj)
            except Exception as e:
                logger.error("Task failure callback failed", error=str(e))

        return result_obj

    async def cancel(self, task_id: str, reason: str = "") -> bool:
        """Cancel a pending or running task."""
        task = await self.backend.get_task(task_id)
        if not task:
            return False

        if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
            return False

        task.status = TaskStatus.CANCELLED
        task.completed_at = datetime.now()
        task.error = f"Cancelled: {reason}"

        await self.backend.update_task(task)
        await self.backend.remove_from_queue(task.queue_name, task_id)

        logger.info("Task cancelled", task_id=task_id, reason=reason)
        return True

    async def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID."""
        return await self.backend.get_task(task_id)

    async def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """Get task status."""
        task = await self.backend.get_task(task_id)
        return task.status if task else None

    async def wait_for_task(
        self,
        task_id: str,
        timeout_seconds: int = 300,
        poll_interval: float = 0.5,
    ) -> TaskResult:
        """Wait for a task to complete."""
        start_time = time.time()

        while True:
            task = await self.backend.get_task(task_id)
            if not task:
                raise ValueError(f"Task not found: {task_id}")

            if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                return TaskResult(
                    task_id=task_id,
                    status=task.status,
                    result=task.result,
                    error=task.error,
                    error_traceback=task.error_traceback,
                    started_at=task.started_at,
                    completed_at=task.completed_at,
                )

            if time.time() - start_time > timeout_seconds:
                raise TimeoutError(f"Timeout waiting for task: {task_id}")

            await asyncio.sleep(poll_interval)

    async def get_queue_stats(self, queue_name: str = "default") -> Dict[str, Any]:
        """Get queue statistics."""
        return await self.backend.get_queue_stats(queue_name)

    async def get_pending_tasks(
        self,
        queue_name: str = "default",
        limit: int = 100,
    ) -> List[Task]:
        """Get pending tasks in queue."""
        return await self.backend.get_pending_tasks(queue_name, limit)

    async def retry_dead_letters(
        self,
        queue_name: str = "default",
        limit: int = 10,
    ) -> int:
        """Retry tasks from dead letter queue."""
        if not self.dead_letter_queue:
            return 0

        count = 0
        for _ in range(limit):
            task = await self.dead_letter_queue.dequeue(queue_name, timeout_seconds=1)
            if not task:
                break

            # Reset retry count and re-enqueue
            task.retry_count = 0
            task.status = TaskStatus.PENDING
            task.error = None
            task.error_traceback = None

            await self.backend.enqueue(task)
            count += 1

        logger.info(f"Retried {count} dead letter tasks")
        return count

    def on_task_started(self, callback: Callable) -> None:
        """Register callback for task started events."""
        self._on_task_started.append(callback)

    def on_task_completed(self, callback: Callable) -> None:
        """Register callback for task completed events."""
        self._on_task_completed.append(callback)

    def on_task_failed(self, callback: Callable) -> None:
        """Register callback for task failed events."""
        self._on_task_failed.append(callback)


class TaskBatch:
    """Batch of tasks for atomic operations."""

    def __init__(self, queue: TaskQueue):
        self.queue = queue
        self.tasks: List[Task] = []
        self._committed = False

    async def add(
        self,
        name: str,
        payload: Dict[str, Any],
        **kwargs,
    ) -> str:
        """Add task to batch."""
        if self._committed:
            raise RuntimeError("Batch already committed")

        task = Task(
            id=str(uuid.uuid4()),
            name=name,
            payload=payload,
            **kwargs,
        )
        self.tasks.append(task)
        return task.id

    async def commit(self) -> List[Task]:
        """Commit all tasks atomically."""
        if self._committed:
            raise RuntimeError("Batch already committed")

        # Enqueue all tasks
        for task in self.tasks:
            await self.queue.backend.enqueue(task)

        self._committed = True
        logger.info(f"Committed batch of {len(self.tasks)} tasks")
        return self.tasks

    async def rollback(self) -> None:
        """Rollback uncommitted batch."""
        if self._committed:
            raise RuntimeError("Cannot rollback committed batch")

        self.tasks.clear()
        logger.info("Batch rolled back")
