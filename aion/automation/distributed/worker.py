"""
AION Distributed Workers

Worker processes that execute tasks from the distributed queue.
"""

from __future__ import annotations

import asyncio
import os
import signal
import socket
import time
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

import structlog

from aion.automation.distributed.queue import (
    Task,
    TaskQueue,
    TaskResult,
    TaskStatus,
)

logger = structlog.get_logger(__name__)


class WorkerStatus(str, Enum):
    """Worker status."""
    STARTING = "starting"
    IDLE = "idle"
    BUSY = "busy"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"


@dataclass
class WorkerStats:
    """Worker statistics."""
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_execution_time_ms: float = 0.0
    last_task_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    uptime_seconds: float = 0.0

    def update_uptime(self) -> None:
        if self.started_at:
            self.uptime_seconds = (datetime.now() - self.started_at).total_seconds()


class Worker:
    """
    A worker that processes tasks from the queue.

    Features:
    - Concurrent task execution
    - Graceful shutdown
    - Health monitoring
    - Task timeout handling
    - Heartbeat reporting
    """

    def __init__(
        self,
        queue: TaskQueue,
        worker_id: Optional[str] = None,
        queue_names: Optional[List[str]] = None,
        concurrency: int = 1,
        heartbeat_interval: float = 30.0,
    ):
        self.queue = queue
        self.worker_id = worker_id or f"worker-{socket.gethostname()}-{uuid.uuid4().hex[:8]}"
        self.queue_names = queue_names or ["default"]
        self.concurrency = concurrency
        self.heartbeat_interval = heartbeat_interval

        # State
        self.status = WorkerStatus.STOPPED
        self.stats = WorkerStats()
        self.current_tasks: Dict[str, Task] = {}

        # Task handlers
        self._handlers: Dict[str, Callable] = {}

        # Control
        self._shutdown_event = asyncio.Event()
        self._pause_event = asyncio.Event()
        self._pause_event.set()  # Not paused initially

        # Background tasks
        self._worker_tasks: List[asyncio.Task] = []
        self._heartbeat_task: Optional[asyncio.Task] = None

    def register_handler(
        self,
        task_name: str,
        handler: Callable,
    ) -> None:
        """Register a handler for a task type."""
        self._handlers[task_name] = handler
        logger.debug(f"Worker {self.worker_id} registered handler: {task_name}")

    async def start(self) -> None:
        """Start the worker."""
        if self.status != WorkerStatus.STOPPED:
            raise RuntimeError(f"Worker already in state: {self.status}")

        self.status = WorkerStatus.STARTING
        self.stats.started_at = datetime.now()
        self._shutdown_event.clear()

        logger.info(
            "Starting worker",
            worker_id=self.worker_id,
            queues=self.queue_names,
            concurrency=self.concurrency,
        )

        # Start worker coroutines
        for i in range(self.concurrency):
            task = asyncio.create_task(
                self._worker_loop(i),
                name=f"{self.worker_id}-loop-{i}",
            )
            self._worker_tasks.append(task)

        # Start heartbeat
        self._heartbeat_task = asyncio.create_task(
            self._heartbeat_loop(),
            name=f"{self.worker_id}-heartbeat",
        )

        self.status = WorkerStatus.IDLE
        logger.info("Worker started", worker_id=self.worker_id)

    async def stop(self, timeout: float = 30.0) -> None:
        """Stop the worker gracefully."""
        if self.status == WorkerStatus.STOPPED:
            return

        self.status = WorkerStatus.STOPPING
        self._shutdown_event.set()

        logger.info(
            "Stopping worker",
            worker_id=self.worker_id,
            active_tasks=len(self.current_tasks),
        )

        # Wait for tasks to complete
        if self._worker_tasks:
            done, pending = await asyncio.wait(
                self._worker_tasks,
                timeout=timeout,
            )

            # Cancel remaining tasks
            for task in pending:
                task.cancel()

        # Stop heartbeat
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        self.status = WorkerStatus.STOPPED
        logger.info("Worker stopped", worker_id=self.worker_id)

    def pause(self) -> None:
        """Pause the worker (finish current tasks but don't take new ones)."""
        self._pause_event.clear()
        self.status = WorkerStatus.PAUSED
        logger.info("Worker paused", worker_id=self.worker_id)

    def resume(self) -> None:
        """Resume a paused worker."""
        self._pause_event.set()
        self.status = WorkerStatus.IDLE
        logger.info("Worker resumed", worker_id=self.worker_id)

    async def _worker_loop(self, worker_index: int) -> None:
        """Main worker loop."""
        while not self._shutdown_event.is_set():
            try:
                # Wait if paused
                await self._pause_event.wait()

                if self._shutdown_event.is_set():
                    break

                # Try to get a task
                task = None
                for queue_name in self.queue_names:
                    task = await self.queue.dequeue(queue_name, timeout_seconds=5)
                    if task:
                        break

                if task:
                    self.status = WorkerStatus.BUSY
                    await self._execute_task(task)
                    self.status = WorkerStatus.IDLE if self._pause_event.is_set() else WorkerStatus.PAUSED

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    "Worker loop error",
                    worker_id=self.worker_id,
                    index=worker_index,
                    error=str(e),
                )
                await asyncio.sleep(1)

    async def _execute_task(self, task: Task) -> None:
        """Execute a single task."""
        self.current_tasks[task.id] = task
        start_time = time.time()

        logger.info(
            "Executing task",
            worker_id=self.worker_id,
            task_id=task.id,
            task_name=task.name,
        )

        try:
            # Get handler
            handler = self._handlers.get(task.name)
            if not handler:
                raise ValueError(f"No handler for task: {task.name}")

            # Execute with timeout
            if task.timeout_seconds:
                result = await asyncio.wait_for(
                    handler(task.payload),
                    timeout=task.timeout_seconds,
                )
            else:
                result = await handler(task.payload)

            # Mark as completed
            await self.queue.complete(task.id, result, self.worker_id)

            execution_time = (time.time() - start_time) * 1000
            self.stats.tasks_completed += 1
            self.stats.total_execution_time_ms += execution_time
            self.stats.last_task_at = datetime.now()

            logger.info(
                "Task completed",
                worker_id=self.worker_id,
                task_id=task.id,
                execution_time_ms=execution_time,
            )

        except asyncio.TimeoutError:
            error_msg = f"Task timed out after {task.timeout_seconds}s"
            await self.queue.fail(task.id, error_msg, worker_id=self.worker_id)
            self.stats.tasks_failed += 1
            logger.error("Task timeout", task_id=task.id, timeout=task.timeout_seconds)

        except Exception as e:
            error_msg = str(e)
            error_tb = traceback.format_exc()
            await self.queue.fail(task.id, error_msg, error_tb, self.worker_id)
            self.stats.tasks_failed += 1
            logger.error("Task failed", task_id=task.id, error=error_msg)

        finally:
            self.current_tasks.pop(task.id, None)

    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeats."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(self.heartbeat_interval)
                self.stats.update_uptime()

                logger.debug(
                    "Worker heartbeat",
                    worker_id=self.worker_id,
                    status=self.status.value,
                    tasks_completed=self.stats.tasks_completed,
                    tasks_failed=self.stats.tasks_failed,
                    active_tasks=len(self.current_tasks),
                )

            except asyncio.CancelledError:
                break

    def get_stats(self) -> Dict[str, Any]:
        """Get worker statistics."""
        self.stats.update_uptime()
        return {
            "worker_id": self.worker_id,
            "status": self.status.value,
            "queues": self.queue_names,
            "concurrency": self.concurrency,
            "active_tasks": len(self.current_tasks),
            "tasks_completed": self.stats.tasks_completed,
            "tasks_failed": self.stats.tasks_failed,
            "avg_execution_time_ms": (
                self.stats.total_execution_time_ms / self.stats.tasks_completed
                if self.stats.tasks_completed > 0 else 0
            ),
            "uptime_seconds": self.stats.uptime_seconds,
            "last_task_at": self.stats.last_task_at.isoformat() if self.stats.last_task_at else None,
        }


class WorkerPool:
    """
    Pool of workers for distributed task processing.

    Features:
    - Auto-scaling based on queue depth
    - Health monitoring
    - Worker affinity
    - Graceful shutdown
    """

    def __init__(
        self,
        queue: TaskQueue,
        min_workers: int = 1,
        max_workers: int = 10,
        worker_concurrency: int = 1,
        queue_names: Optional[List[str]] = None,
        auto_scale: bool = True,
        scale_up_threshold: int = 100,  # Queue depth to trigger scale up
        scale_down_threshold: int = 10,  # Queue depth to trigger scale down
    ):
        self.queue = queue
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.worker_concurrency = worker_concurrency
        self.queue_names = queue_names or ["default"]
        self.auto_scale = auto_scale
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold

        # Workers
        self._workers: Dict[str, Worker] = {}
        self._handlers: Dict[str, Callable] = {}

        # Control
        self._shutdown_event = asyncio.Event()
        self._scale_task: Optional[asyncio.Task] = None

    def register_handler(
        self,
        task_name: str,
        handler: Callable,
    ) -> None:
        """Register a handler for all workers."""
        self._handlers[task_name] = handler

        # Register with existing workers
        for worker in self._workers.values():
            worker.register_handler(task_name, handler)

    async def start(self) -> None:
        """Start the worker pool."""
        logger.info(
            "Starting worker pool",
            min_workers=self.min_workers,
            max_workers=self.max_workers,
        )

        self._shutdown_event.clear()

        # Start minimum workers
        for _ in range(self.min_workers):
            await self._add_worker()

        # Start auto-scaler
        if self.auto_scale:
            self._scale_task = asyncio.create_task(self._scale_loop())

        logger.info(f"Worker pool started with {len(self._workers)} workers")

    async def stop(self, timeout: float = 30.0) -> None:
        """Stop the worker pool."""
        logger.info("Stopping worker pool")

        self._shutdown_event.set()

        # Stop auto-scaler
        if self._scale_task:
            self._scale_task.cancel()
            try:
                await self._scale_task
            except asyncio.CancelledError:
                pass

        # Stop all workers
        await asyncio.gather(*[
            worker.stop(timeout=timeout)
            for worker in self._workers.values()
        ])

        self._workers.clear()
        logger.info("Worker pool stopped")

    async def _add_worker(self) -> Worker:
        """Add a new worker to the pool."""
        if len(self._workers) >= self.max_workers:
            raise RuntimeError("Maximum workers reached")

        worker = Worker(
            queue=self.queue,
            queue_names=self.queue_names,
            concurrency=self.worker_concurrency,
        )

        # Register handlers
        for task_name, handler in self._handlers.items():
            worker.register_handler(task_name, handler)

        await worker.start()
        self._workers[worker.worker_id] = worker

        logger.info(f"Added worker: {worker.worker_id}, total: {len(self._workers)}")
        return worker

    async def _remove_worker(self) -> bool:
        """Remove a worker from the pool."""
        if len(self._workers) <= self.min_workers:
            return False

        # Find an idle worker to remove
        for worker_id, worker in list(self._workers.items()):
            if worker.status == WorkerStatus.IDLE:
                await worker.stop()
                del self._workers[worker_id]
                logger.info(f"Removed worker: {worker_id}, total: {len(self._workers)}")
                return True

        return False

    async def _scale_loop(self) -> None:
        """Auto-scaling loop."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(10)

                # Get total queue depth
                total_pending = 0
                for queue_name in self.queue_names:
                    stats = await self.queue.get_queue_stats(queue_name)
                    total_pending += stats.get("pending_count", 0)

                current_workers = len(self._workers)

                # Scale up
                if total_pending > self.scale_up_threshold and current_workers < self.max_workers:
                    await self._add_worker()
                    logger.info(f"Scaled up: queue_depth={total_pending}")

                # Scale down
                elif total_pending < self.scale_down_threshold and current_workers > self.min_workers:
                    await self._remove_worker()
                    logger.info(f"Scaled down: queue_depth={total_pending}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Scale loop error", error=str(e))

    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        worker_stats = [w.get_stats() for w in self._workers.values()]

        return {
            "worker_count": len(self._workers),
            "min_workers": self.min_workers,
            "max_workers": self.max_workers,
            "auto_scale": self.auto_scale,
            "total_tasks_completed": sum(w["tasks_completed"] for w in worker_stats),
            "total_tasks_failed": sum(w["tasks_failed"] for w in worker_stats),
            "total_active_tasks": sum(w["active_tasks"] for w in worker_stats),
            "workers": worker_stats,
        }

    async def scale_to(self, target: int) -> None:
        """Scale to a specific number of workers."""
        target = max(self.min_workers, min(self.max_workers, target))

        while len(self._workers) < target:
            await self._add_worker()

        while len(self._workers) > target:
            if not await self._remove_worker():
                break

        logger.info(f"Scaled to {len(self._workers)} workers")


class WorkflowWorker(Worker):
    """
    Specialized worker for workflow step execution.

    Integrates with the event store for durability.
    """

    def __init__(
        self,
        queue: TaskQueue,
        event_store: Optional["EventStore"] = None,
        **kwargs,
    ):
        super().__init__(queue, **kwargs)
        self.event_store = event_store

    async def _execute_task(self, task: Task) -> None:
        """Execute task with event sourcing."""
        # Record start event
        if self.event_store and task.execution_id:
            from aion.automation.execution.event_store import EventType
            await self.event_store.append(
                task.execution_id,
                EventType.STEP_STARTED,
                {
                    "step_id": task.step_id,
                    "task_id": task.id,
                    "worker_id": self.worker_id,
                },
            )

        try:
            await super()._execute_task(task)

            # Record completion event
            if self.event_store and task.execution_id:
                await self.event_store.append(
                    task.execution_id,
                    EventType.STEP_COMPLETED,
                    {
                        "step_id": task.step_id,
                        "task_id": task.id,
                        "worker_id": self.worker_id,
                    },
                )

        except Exception as e:
            # Record failure event
            if self.event_store and task.execution_id:
                await self.event_store.append(
                    task.execution_id,
                    EventType.STEP_FAILED,
                    {
                        "step_id": task.step_id,
                        "task_id": task.id,
                        "worker_id": self.worker_id,
                        "error": str(e),
                    },
                )
            raise
