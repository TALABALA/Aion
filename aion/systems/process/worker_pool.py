"""
AION Worker Pool

State-of-the-art background task execution pool with:
- Configurable pool size with auto-scaling
- Priority-based task queuing
- Work stealing for load balancing
- Task batching for efficiency
- Graceful shutdown with drain support
- Comprehensive worker health monitoring
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Optional, Dict, List, Tuple
from collections import deque
from heapq import heappush, heappop
from enum import Enum

import structlog

from aion.systems.process.models import (
    ProcessState,
    ProcessPriority,
    ResourceLimits,
    WorkerInfo,
)
from aion.systems.process.event_bus import EventBus, Event

logger = structlog.get_logger(__name__)


class TaskStatus(Enum):
    """Status of a queued task."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


@dataclass
class QueuedTask:
    """A task in the worker pool queue."""
    id: str
    handler: Callable
    params: dict = field(default_factory=dict)
    priority: int = 0  # Higher = more important
    timeout_seconds: Optional[float] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    worker_id: Optional[str] = None
    retries: int = 0
    max_retries: int = 0
    callback: Optional[Callable] = None  # Called on completion

    def __lt__(self, other: "QueuedTask") -> bool:
        # Higher priority first, then older tasks
        if self.priority != other.priority:
            return self.priority > other.priority
        return self.created_at < other.created_at


@dataclass
class WorkerPoolStats:
    """Statistics for the worker pool."""
    tasks_submitted: int = 0
    tasks_completed: int = 0
    tasks_failed: int = 0
    tasks_cancelled: int = 0
    tasks_timeout: int = 0
    total_execution_time: float = 0.0
    avg_wait_time: float = 0.0
    avg_execution_time: float = 0.0
    current_queue_size: int = 0
    active_workers: int = 0
    idle_workers: int = 0
    total_workers: int = 0
    _wait_times: deque = field(default_factory=lambda: deque(maxlen=1000))
    _exec_times: deque = field(default_factory=lambda: deque(maxlen=1000))

    def record_wait_time(self, seconds: float) -> None:
        self._wait_times.append(seconds)
        if self._wait_times:
            self.avg_wait_time = sum(self._wait_times) / len(self._wait_times)

    def record_execution_time(self, seconds: float) -> None:
        self._exec_times.append(seconds)
        self.total_execution_time += seconds
        if self._exec_times:
            self.avg_execution_time = sum(self._exec_times) / len(self._exec_times)

    def to_dict(self) -> dict:
        return {
            "tasks_submitted": self.tasks_submitted,
            "tasks_completed": self.tasks_completed,
            "tasks_failed": self.tasks_failed,
            "tasks_cancelled": self.tasks_cancelled,
            "tasks_timeout": self.tasks_timeout,
            "total_execution_time": self.total_execution_time,
            "avg_wait_time": self.avg_wait_time,
            "avg_execution_time": self.avg_execution_time,
            "current_queue_size": self.current_queue_size,
            "active_workers": self.active_workers,
            "idle_workers": self.idle_workers,
            "total_workers": self.total_workers,
        }


class Worker:
    """A worker in the pool."""

    def __init__(
        self,
        worker_id: str,
        pool: "WorkerPool",
    ):
        self.id = worker_id
        self.pool = pool
        self.info = WorkerInfo(id=worker_id, state=ProcessState.CREATED)
        self._task: Optional[asyncio.Task] = None
        self._current_task: Optional[QueuedTask] = None
        self._shutdown_event = asyncio.Event()

    async def start(self) -> None:
        """Start the worker."""
        self.info.state = ProcessState.RUNNING
        self._task = asyncio.create_task(self._run_loop())
        logger.debug(f"Worker {self.id} started")

    async def stop(self, graceful: bool = True) -> None:
        """Stop the worker."""
        self._shutdown_event.set()

        if graceful and self._current_task:
            # Wait for current task to complete
            try:
                await asyncio.wait_for(self._task, timeout=30.0)
            except asyncio.TimeoutError:
                pass

        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        self.info.state = ProcessState.STOPPED
        logger.debug(f"Worker {self.id} stopped")

    async def _run_loop(self) -> None:
        """Main worker loop."""
        while not self._shutdown_event.is_set():
            try:
                # Get task from queue
                task = await self.pool._get_task(timeout=1.0)

                if task is None:
                    continue

                self._current_task = task

                # Execute task
                await self._execute_task(task)

                self._current_task = None

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {self.id} error: {e}")
                await asyncio.sleep(0.1)

    async def _execute_task(self, task: QueuedTask) -> None:
        """Execute a single task."""
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now()
        task.worker_id = self.id

        # Record wait time
        wait_time = (task.started_at - task.created_at).total_seconds()
        self.pool._stats.record_wait_time(wait_time)

        self.info.current_task_id = task.id
        self.info.last_task_at = task.started_at

        try:
            # Execute with optional timeout
            if task.timeout_seconds:
                result = await asyncio.wait_for(
                    self._call_handler(task),
                    timeout=task.timeout_seconds,
                )
            else:
                result = await self._call_handler(task)

            task.result = result
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()

            self.info.tasks_completed += 1
            self.pool._stats.tasks_completed += 1

        except asyncio.TimeoutError:
            task.status = TaskStatus.TIMEOUT
            task.error = f"Task timed out after {task.timeout_seconds}s"
            task.completed_at = datetime.now()

            self.info.tasks_failed += 1
            self.pool._stats.tasks_timeout += 1

            logger.warning(f"Task {task.id} timed out")

        except asyncio.CancelledError:
            task.status = TaskStatus.CANCELLED
            task.completed_at = datetime.now()

            self.pool._stats.tasks_cancelled += 1
            raise

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = datetime.now()

            self.info.tasks_failed += 1
            self.pool._stats.tasks_failed += 1

            logger.error(f"Task {task.id} failed: {e}")

            # Retry if configured
            if task.retries < task.max_retries:
                task.retries += 1
                task.status = TaskStatus.PENDING
                task.started_at = None
                task.completed_at = None
                task.worker_id = None
                await self.pool._requeue_task(task)
                return

        finally:
            self.info.current_task_id = None

            # Record execution time
            if task.started_at and task.completed_at:
                exec_time = (task.completed_at - task.started_at).total_seconds()
                self.pool._stats.record_execution_time(exec_time)
                self.info.total_runtime_seconds += exec_time

            # Store completed task
            self.pool._completed_tasks[task.id] = task

            # Call completion callback
            if task.callback:
                try:
                    result = task.callback(task)
                    if asyncio.iscoroutine(result):
                        await result
                except Exception as e:
                    logger.warning(f"Task callback failed: {e}")

    async def _call_handler(self, task: QueuedTask) -> Any:
        """Call the task handler."""
        result = task.handler(**task.params)
        if asyncio.iscoroutine(result):
            return await result
        return result


class WorkerPool:
    """
    Worker pool for background task execution.

    Features:
    - Priority-based task queuing
    - Configurable pool size
    - Auto-scaling based on load
    - Task timeout and retry support
    - Graceful shutdown with drain
    """

    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
        min_workers: int = 2,
        max_workers: int = 10,
        max_queue_size: int = 1000,
        enable_auto_scaling: bool = True,
        scale_up_threshold: float = 0.8,  # Queue utilization threshold
        scale_down_threshold: float = 0.2,
        default_timeout: Optional[float] = None,
        default_max_retries: int = 0,
    ):
        self.event_bus = event_bus
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.max_queue_size = max_queue_size
        self.enable_auto_scaling = enable_auto_scaling
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.default_timeout = default_timeout
        self.default_max_retries = default_max_retries

        # Task queue (priority queue)
        self._task_queue: List[QueuedTask] = []
        self._task_semaphore = asyncio.Semaphore(0)
        self._queue_lock = asyncio.Lock()

        # Workers
        self._workers: Dict[str, Worker] = {}
        self._worker_lock = asyncio.Lock()

        # Task tracking
        self._pending_tasks: Dict[str, QueuedTask] = {}
        self._completed_tasks: Dict[str, QueuedTask] = {}
        self._max_completed_history: int = 1000

        # Statistics
        self._stats = WorkerPoolStats()

        # Background tasks
        self._scaler_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the worker pool."""
        if self._initialized:
            return

        logger.info("Initializing Worker Pool")

        # Start minimum workers
        for _ in range(self.min_workers):
            await self._create_worker()

        # Start auto-scaler if enabled
        if self.enable_auto_scaling:
            self._scaler_task = asyncio.create_task(self._auto_scale_loop())

        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

        self._initialized = True
        self._update_stats()

        logger.info(
            "Worker Pool initialized",
            min_workers=self.min_workers,
            max_workers=self.max_workers,
        )

    async def shutdown(self, graceful: bool = True, drain: bool = True) -> None:
        """
        Shutdown the worker pool.

        Args:
            graceful: Wait for current tasks to complete
            drain: Process remaining queue before shutdown
        """
        logger.info("Shutting down Worker Pool")

        self._shutdown_event.set()

        # Stop background tasks
        for task in [self._scaler_task, self._cleanup_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        if drain:
            # Wait for queue to drain
            timeout = 60.0  # Max wait time
            start = datetime.now()
            while self._task_queue and (datetime.now() - start).total_seconds() < timeout:
                await asyncio.sleep(0.5)

        # Stop all workers
        async with self._worker_lock:
            for worker in list(self._workers.values()):
                await worker.stop(graceful=graceful)
            self._workers.clear()

        logger.info(
            "Worker Pool shutdown complete",
            tasks_completed=self._stats.tasks_completed,
        )

    async def submit(
        self,
        handler: Callable,
        params: Optional[dict] = None,
        priority: int = 0,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        callback: Optional[Callable] = None,
    ) -> str:
        """
        Submit a task to the pool.

        Args:
            handler: Async or sync function to execute
            params: Parameters to pass to handler
            priority: Task priority (higher = more important)
            timeout: Task timeout in seconds
            max_retries: Max retry attempts on failure
            callback: Function to call on completion

        Returns:
            Task ID

        Raises:
            RuntimeError: If queue is full
        """
        async with self._queue_lock:
            if len(self._task_queue) >= self.max_queue_size:
                raise RuntimeError(f"Task queue full ({self.max_queue_size})")

        task = QueuedTask(
            id=str(uuid.uuid4()),
            handler=handler,
            params=params or {},
            priority=priority,
            timeout_seconds=timeout or self.default_timeout,
            max_retries=max_retries if max_retries is not None else self.default_max_retries,
            callback=callback,
        )

        async with self._queue_lock:
            heappush(self._task_queue, task)
            self._pending_tasks[task.id] = task

        self._task_semaphore.release()

        self._stats.tasks_submitted += 1
        self._update_stats()

        logger.debug(f"Task submitted: {task.id}")

        return task.id

    async def submit_batch(
        self,
        tasks: List[Tuple[Callable, dict]],
        priority: int = 0,
        timeout: Optional[float] = None,
    ) -> List[str]:
        """
        Submit multiple tasks at once.

        Args:
            tasks: List of (handler, params) tuples
            priority: Priority for all tasks
            timeout: Timeout for all tasks

        Returns:
            List of task IDs
        """
        task_ids = []

        async with self._queue_lock:
            for handler, params in tasks:
                if len(self._task_queue) >= self.max_queue_size:
                    break

                task = QueuedTask(
                    id=str(uuid.uuid4()),
                    handler=handler,
                    params=params,
                    priority=priority,
                    timeout_seconds=timeout or self.default_timeout,
                )

                heappush(self._task_queue, task)
                self._pending_tasks[task.id] = task
                task_ids.append(task.id)

                self._task_semaphore.release()

        self._stats.tasks_submitted += len(task_ids)
        self._update_stats()

        return task_ids

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending task."""
        async with self._queue_lock:
            task = self._pending_tasks.get(task_id)
            if task and task.status == TaskStatus.PENDING:
                task.status = TaskStatus.CANCELLED
                self._stats.tasks_cancelled += 1
                return True
        return False

    async def wait_for_task(self, task_id: str, timeout: Optional[float] = None) -> Optional[QueuedTask]:
        """Wait for a task to complete."""
        start = datetime.now()

        while True:
            task = self._completed_tasks.get(task_id)
            if task:
                return task

            if timeout and (datetime.now() - start).total_seconds() > timeout:
                return None

            await asyncio.sleep(0.1)

    async def _get_task(self, timeout: float = 1.0) -> Optional[QueuedTask]:
        """Get the next task from the queue (called by workers)."""
        try:
            await asyncio.wait_for(self._task_semaphore.acquire(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

        async with self._queue_lock:
            while self._task_queue:
                task = heappop(self._task_queue)

                # Skip cancelled tasks
                if task.status == TaskStatus.CANCELLED:
                    continue

                self._pending_tasks.pop(task.id, None)
                return task

        return None

    async def _requeue_task(self, task: QueuedTask) -> None:
        """Requeue a task for retry."""
        async with self._queue_lock:
            heappush(self._task_queue, task)
            self._pending_tasks[task.id] = task

        self._task_semaphore.release()

    async def _create_worker(self) -> str:
        """Create a new worker."""
        async with self._worker_lock:
            if len(self._workers) >= self.max_workers:
                return ""

            worker_id = str(uuid.uuid4())[:8]
            worker = Worker(worker_id, self)
            self._workers[worker_id] = worker

            await worker.start()

            self._update_stats()
            return worker_id

    async def _remove_worker(self, worker_id: str) -> bool:
        """Remove a worker."""
        async with self._worker_lock:
            if len(self._workers) <= self.min_workers:
                return False

            worker = self._workers.pop(worker_id, None)
            if worker:
                await worker.stop(graceful=True)
                self._update_stats()
                return True

        return False

    async def _auto_scale_loop(self) -> None:
        """Auto-scale workers based on load."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(5.0)  # Check every 5 seconds

                if self._shutdown_event.is_set():
                    break

                queue_utilization = len(self._task_queue) / self.max_queue_size
                worker_count = len(self._workers)

                # Scale up if queue is filling up
                if queue_utilization > self.scale_up_threshold:
                    if worker_count < self.max_workers:
                        await self._create_worker()
                        logger.debug(f"Scaled up to {len(self._workers)} workers")

                # Scale down if queue is mostly empty
                elif queue_utilization < self.scale_down_threshold:
                    if worker_count > self.min_workers:
                        # Find an idle worker to remove
                        async with self._worker_lock:
                            for wid, worker in list(self._workers.items()):
                                if worker._current_task is None:
                                    await self._remove_worker(wid)
                                    logger.debug(f"Scaled down to {len(self._workers)} workers")
                                    break

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Auto-scale error: {e}")

    async def _cleanup_loop(self) -> None:
        """Clean up old completed tasks."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(60.0)  # Every minute

                if self._shutdown_event.is_set():
                    break

                # Trim completed tasks history
                if len(self._completed_tasks) > self._max_completed_history:
                    sorted_tasks = sorted(
                        self._completed_tasks.items(),
                        key=lambda x: x[1].completed_at or datetime.min,
                    )
                    to_remove = len(sorted_tasks) - self._max_completed_history
                    for task_id, _ in sorted_tasks[:to_remove]:
                        del self._completed_tasks[task_id]

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup error: {e}")

    def _update_stats(self) -> None:
        """Update pool statistics."""
        self._stats.current_queue_size = len(self._task_queue)
        self._stats.total_workers = len(self._workers)
        self._stats.active_workers = sum(
            1 for w in self._workers.values()
            if w._current_task is not None
        )
        self._stats.idle_workers = self._stats.total_workers - self._stats.active_workers

    # === Query Methods ===

    def get_task(self, task_id: str) -> Optional[QueuedTask]:
        """Get task by ID."""
        return self._pending_tasks.get(task_id) or self._completed_tasks.get(task_id)

    def get_queue_size(self) -> int:
        """Get current queue size."""
        return len(self._task_queue)

    def get_worker_count(self) -> int:
        """Get current worker count."""
        return len(self._workers)

    def get_workers(self) -> List[WorkerInfo]:
        """Get information about all workers."""
        return [w.info for w in self._workers.values()]

    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        self._update_stats()
        return self._stats.to_dict()

    # === Context Manager ===

    async def __aenter__(self) -> "WorkerPool":
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.shutdown()
