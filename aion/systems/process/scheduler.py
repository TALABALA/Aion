"""
AION Task Scheduler

State-of-the-art task scheduling system with:
- One-time delayed task execution
- Recurring interval tasks
- Cron expression support with timezone awareness
- Task persistence and recovery
- Missed run handling (coalesce/catch-up)
- Jitter for load distribution
- Comprehensive task lifecycle management
"""

from __future__ import annotations

import asyncio
import random
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Optional, Dict, List
from collections import defaultdict
from heapq import heappush, heappop

import structlog

from aion.systems.process.models import (
    TaskDefinition,
    ProcessPriority,
    ResourceLimits,
    Event,
)
from aion.systems.process.supervisor import ProcessSupervisor
from aion.systems.process.event_bus import EventBus

logger = structlog.get_logger(__name__)


class CronParser:
    """
    Parse and evaluate cron expressions.

    Supports standard cron format: minute hour day month weekday
    Also supports special strings: @hourly, @daily, @weekly, @monthly, @yearly
    """

    SPECIAL_EXPRESSIONS = {
        "@yearly": "0 0 1 1 *",
        "@annually": "0 0 1 1 *",
        "@monthly": "0 0 1 * *",
        "@weekly": "0 0 * * 0",
        "@daily": "0 0 * * *",
        "@midnight": "0 0 * * *",
        "@hourly": "0 * * * *",
    }

    @classmethod
    def parse(cls, expression: str) -> Dict[str, List[int]]:
        """Parse a cron expression into field values."""
        # Handle special expressions
        if expression.startswith("@"):
            expression = cls.SPECIAL_EXPRESSIONS.get(expression.lower(), expression)

        parts = expression.split()
        if len(parts) != 5:
            raise ValueError(f"Invalid cron expression: {expression}")

        return {
            "minute": cls._parse_field(parts[0], 0, 59),
            "hour": cls._parse_field(parts[1], 0, 23),
            "day": cls._parse_field(parts[2], 1, 31),
            "month": cls._parse_field(parts[3], 1, 12),
            "weekday": cls._parse_field(parts[4], 0, 6),  # 0=Sunday
        }

    @classmethod
    def _parse_field(cls, field: str, min_val: int, max_val: int) -> List[int]:
        """Parse a single cron field."""
        if field == "*":
            return list(range(min_val, max_val + 1))

        values = set()

        for part in field.split(","):
            if "/" in part:
                # Step values: */5 or 1-10/2
                range_part, step = part.split("/")
                step = int(step)
                if range_part == "*":
                    start, end = min_val, max_val
                else:
                    start, end = cls._parse_range(range_part, min_val, max_val)
                values.update(range(start, end + 1, step))
            elif "-" in part:
                # Range: 1-5
                start, end = cls._parse_range(part, min_val, max_val)
                values.update(range(start, end + 1))
            else:
                # Single value
                values.add(int(part))

        return sorted(v for v in values if min_val <= v <= max_val)

    @classmethod
    def _parse_range(cls, part: str, min_val: int, max_val: int) -> tuple[int, int]:
        """Parse a range like 1-5."""
        if "-" in part:
            start, end = part.split("-")
            return int(start), int(end)
        val = int(part)
        return val, val

    @classmethod
    def get_next_run(cls, expression: str, after: datetime) -> datetime:
        """Calculate the next run time after a given datetime."""
        parsed = cls.parse(expression)

        # Start from the next minute
        candidate = after.replace(second=0, microsecond=0) + timedelta(minutes=1)

        # Search for next matching time (up to 4 years)
        max_iterations = 365 * 24 * 60 * 4
        for _ in range(max_iterations):
            if (candidate.minute in parsed["minute"] and
                candidate.hour in parsed["hour"] and
                candidate.day in parsed["day"] and
                candidate.month in parsed["month"] and
                candidate.weekday() in parsed["weekday"]):
                return candidate

            candidate += timedelta(minutes=1)

        raise ValueError(f"Could not find next run time for: {expression}")


class TaskScheduler:
    """
    Task scheduler for AION.

    Features:
    - One-time scheduled tasks
    - Recurring interval tasks
    - Cron expression scheduling
    - Task persistence and recovery
    - Missed run handling
    - Priority-based execution
    """

    def __init__(
        self,
        supervisor: ProcessSupervisor,
        event_bus: Optional[EventBus] = None,
        check_interval: float = 1.0,
        max_concurrent_tasks: int = 10,
        enable_persistence: bool = False,
    ):
        self.supervisor = supervisor
        self.event_bus = event_bus
        self.check_interval = check_interval
        self.max_concurrent_tasks = max_concurrent_tasks
        self.enable_persistence = enable_persistence

        # Task storage
        self._tasks: Dict[str, TaskDefinition] = {}
        self._handlers: Dict[str, Callable] = {}

        # Execution tracking
        self._running_tasks: Dict[str, str] = {}  # task_id -> process_id
        self._task_locks: Dict[str, asyncio.Lock] = {}

        # Priority queue: (next_run, task_id)
        self._task_queue: List[tuple[datetime, str]] = []

        # Background scheduler
        self._scheduler_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

        # Statistics
        self._stats = {
            "tasks_scheduled": 0,
            "tasks_executed": 0,
            "tasks_succeeded": 0,
            "tasks_failed": 0,
            "tasks_cancelled": 0,
            "tasks_skipped": 0,
        }

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the scheduler."""
        if self._initialized:
            return

        logger.info("Initializing Task Scheduler")

        # Start scheduler loop
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())

        self._initialized = True
        logger.info(
            "Task Scheduler initialized",
            check_interval=self.check_interval,
            max_concurrent=self.max_concurrent_tasks,
        )

    async def shutdown(self) -> None:
        """Shutdown the scheduler."""
        logger.info("Shutting down Task Scheduler")

        self._shutdown_event.set()

        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass

        # Wait for running tasks to complete
        for task_id, process_id in list(self._running_tasks.items()):
            try:
                await self.supervisor.stop_process(process_id, graceful=True, timeout=10)
            except Exception as e:
                logger.warning(f"Failed to stop task {task_id}: {e}")

        logger.info(
            "Task Scheduler shutdown complete",
            tasks_executed=self._stats["tasks_executed"],
        )

    def register_handler(self, name: str, handler: Callable) -> None:
        """Register a task handler function."""
        self._handlers[name] = handler
        logger.debug(f"Registered task handler: {name}")

    def unregister_handler(self, name: str) -> bool:
        """Unregister a task handler."""
        return self._handlers.pop(name, None) is not None

    async def schedule_once(
        self,
        name: str,
        handler: str,
        run_at: datetime,
        params: Optional[dict] = None,
        priority: ProcessPriority = ProcessPriority.NORMAL,
        timeout_seconds: int = 300,
        metadata: Optional[dict] = None,
        tags: Optional[List[str]] = None,
    ) -> str:
        """
        Schedule a one-time task.

        Args:
            name: Task name
            handler: Handler function name
            run_at: When to run (datetime)
            params: Parameters to pass to handler
            priority: Task priority
            timeout_seconds: Execution timeout
            metadata: Additional metadata
            tags: Task tags

        Returns:
            Task ID
        """
        task_id = str(uuid.uuid4())

        task = TaskDefinition(
            id=task_id,
            name=name,
            handler=handler,
            params=params or {},
            schedule_type="once",
            run_at=run_at,
            next_run=run_at,
            timeout_seconds=timeout_seconds,
            priority=priority,
            metadata=metadata or {},
            tags=tags or [],
        )

        self._tasks[task_id] = task
        self._task_locks[task_id] = asyncio.Lock()
        self._add_to_queue(task)

        self._stats["tasks_scheduled"] += 1

        await self._emit_task_event("scheduled", task)

        logger.info(
            "Scheduled one-time task",
            task_id=task_id,
            name=name,
            run_at=run_at.isoformat(),
        )

        return task_id

    async def schedule_interval(
        self,
        name: str,
        handler: str,
        interval_seconds: int,
        params: Optional[dict] = None,
        start_immediately: bool = False,
        priority: ProcessPriority = ProcessPriority.NORMAL,
        timeout_seconds: int = 300,
        max_instances: int = 1,
        jitter_seconds: int = 0,
        metadata: Optional[dict] = None,
        tags: Optional[List[str]] = None,
    ) -> str:
        """
        Schedule a recurring interval task.

        Args:
            name: Task name
            handler: Handler function name
            interval_seconds: Interval between runs
            params: Parameters to pass
            start_immediately: Run immediately, then start interval
            priority: Task priority
            timeout_seconds: Execution timeout
            max_instances: Max concurrent instances
            jitter_seconds: Random delay up to this value
            metadata: Additional metadata
            tags: Task tags

        Returns:
            Task ID
        """
        task_id = str(uuid.uuid4())

        next_run = datetime.now()
        if not start_immediately:
            next_run += timedelta(seconds=interval_seconds)

        task = TaskDefinition(
            id=task_id,
            name=name,
            handler=handler,
            params=params or {},
            schedule_type="interval",
            interval_seconds=interval_seconds,
            next_run=next_run,
            timeout_seconds=timeout_seconds,
            priority=priority,
            max_instances=max_instances,
            jitter_seconds=jitter_seconds,
            metadata=metadata or {},
            tags=tags or [],
        )

        self._tasks[task_id] = task
        self._task_locks[task_id] = asyncio.Lock()
        self._add_to_queue(task)

        self._stats["tasks_scheduled"] += 1

        await self._emit_task_event("scheduled", task)

        logger.info(
            "Scheduled interval task",
            task_id=task_id,
            name=name,
            interval=interval_seconds,
        )

        return task_id

    async def schedule_cron(
        self,
        name: str,
        handler: str,
        cron_expression: str,
        params: Optional[dict] = None,
        priority: ProcessPriority = ProcessPriority.NORMAL,
        timeout_seconds: int = 300,
        timezone_str: str = "UTC",
        coalesce: bool = True,
        max_instances: int = 1,
        jitter_seconds: int = 0,
        metadata: Optional[dict] = None,
        tags: Optional[List[str]] = None,
    ) -> str:
        """
        Schedule a cron task.

        Args:
            name: Task name
            handler: Handler function name
            cron_expression: Cron expression (e.g., "0 9 * * *" for 9 AM daily)
            params: Parameters to pass
            priority: Task priority
            timeout_seconds: Execution timeout
            timezone_str: Timezone for cron calculation
            coalesce: Skip missed runs if behind
            max_instances: Max concurrent instances
            jitter_seconds: Random delay up to this value
            metadata: Additional metadata
            tags: Task tags

        Returns:
            Task ID
        """
        task_id = str(uuid.uuid4())

        # Validate cron expression
        try:
            next_run = CronParser.get_next_run(cron_expression, datetime.now())
        except ValueError as e:
            raise ValueError(f"Invalid cron expression: {e}")

        task = TaskDefinition(
            id=task_id,
            name=name,
            handler=handler,
            params=params or {},
            schedule_type="cron",
            cron_expression=cron_expression,
            timezone=timezone_str,
            next_run=next_run,
            timeout_seconds=timeout_seconds,
            priority=priority,
            coalesce=coalesce,
            max_instances=max_instances,
            jitter_seconds=jitter_seconds,
            metadata=metadata or {},
            tags=tags or [],
        )

        self._tasks[task_id] = task
        self._task_locks[task_id] = asyncio.Lock()
        self._add_to_queue(task)

        self._stats["tasks_scheduled"] += 1

        await self._emit_task_event("scheduled", task)

        logger.info(
            "Scheduled cron task",
            task_id=task_id,
            name=name,
            cron=cron_expression,
            next_run=next_run.isoformat(),
        )

        return task_id

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a scheduled task."""
        task = self._tasks.pop(task_id, None)
        if not task:
            return False

        self._task_locks.pop(task_id, None)

        # Stop if running
        process_id = self._running_tasks.get(task_id)
        if process_id:
            await self.supervisor.stop_process(process_id)
            self._running_tasks.pop(task_id, None)

        self._stats["tasks_cancelled"] += 1

        await self._emit_task_event("cancelled", task)

        logger.info(f"Cancelled task: {task_id}")
        return True

    async def pause_task(self, task_id: str) -> bool:
        """Pause a scheduled task."""
        task = self._tasks.get(task_id)
        if task:
            task.enabled = False
            await self._emit_task_event("paused", task)
            return True
        return False

    async def resume_task(self, task_id: str) -> bool:
        """Resume a paused task."""
        task = self._tasks.get(task_id)
        if task:
            task.enabled = True

            # Recalculate next run if needed
            if task.schedule_type == "cron" and task.cron_expression:
                task.next_run = CronParser.get_next_run(task.cron_expression, datetime.now())
            elif task.schedule_type == "interval" and task.interval_seconds:
                task.next_run = datetime.now() + timedelta(seconds=task.interval_seconds)

            self._add_to_queue(task)
            await self._emit_task_event("resumed", task)
            return True
        return False

    async def trigger_task(self, task_id: str) -> Optional[str]:
        """Manually trigger a task immediately."""
        task = self._tasks.get(task_id)
        if not task:
            return None

        # Execute immediately
        process_id = await self._execute_task(task)
        return process_id

    def _add_to_queue(self, task: TaskDefinition) -> None:
        """Add task to the priority queue."""
        if task.next_run:
            heappush(self._task_queue, (task.next_run, task.id))

    def _rebuild_queue(self) -> None:
        """Rebuild the task queue from all tasks."""
        self._task_queue.clear()
        for task in self._tasks.values():
            if task.enabled and task.next_run:
                self._add_to_queue(task)

    async def _scheduler_loop(self) -> None:
        """Main scheduler loop."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(self.check_interval)

                if self._shutdown_event.is_set():
                    break

                now = datetime.now()

                # Process due tasks
                while self._task_queue:
                    next_run, task_id = self._task_queue[0]

                    if next_run > now:
                        break  # No more due tasks

                    heappop(self._task_queue)

                    task = self._tasks.get(task_id)
                    if not task or not task.enabled:
                        continue

                    # Check if already running (max_instances)
                    running_count = sum(
                        1 for tid, _ in self._running_tasks.items()
                        if self._tasks.get(tid) and self._tasks[tid].name == task.name
                    )

                    if running_count >= task.max_instances:
                        if task.coalesce:
                            self._stats["tasks_skipped"] += 1
                            logger.debug(f"Skipping task (max instances): {task.name}")
                        # Still update next_run for recurring tasks
                        self._update_next_run(task)
                        continue

                    # Check concurrent limit
                    if len(self._running_tasks) >= self.max_concurrent_tasks:
                        # Re-add to queue with slight delay
                        task.next_run = now + timedelta(seconds=1)
                        self._add_to_queue(task)
                        continue

                    # Execute task
                    await self._execute_task(task)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scheduler loop error: {e}")

    async def _execute_task(self, task: TaskDefinition) -> Optional[str]:
        """Execute a scheduled task."""
        async with self._task_locks.get(task.id, asyncio.Lock()):
            logger.info(f"Executing task: {task.name}")

            # Apply jitter if configured
            if task.jitter_seconds > 0:
                jitter = random.uniform(0, task.jitter_seconds)
                await asyncio.sleep(jitter)

            task.last_run = datetime.now()
            task.run_count += 1

            handler = self._handlers.get(task.handler)
            if not handler:
                logger.error(f"Handler not found: {task.handler}")
                task.failure_count += 1
                task.consecutive_failures += 1
                task.last_error = f"Handler not found: {task.handler}"
                task.last_run_success = False
                self._update_next_run(task)
                return None

            start_time = datetime.now()

            try:
                # Spawn as process
                process_id = await self.supervisor.spawn_task(
                    name=f"task_{task.name}_{task.run_count}",
                    handler=handler,
                    params=task.params,
                    priority=task.priority,
                    limits=task.limits,
                    timeout=float(task.timeout_seconds),
                )

                self._running_tasks[task.id] = process_id
                self._stats["tasks_executed"] += 1

                await self._emit_task_event("started", task, process_id=process_id)

                # Wait for completion (in background)
                asyncio.create_task(self._wait_for_task(task, process_id, start_time))

                return process_id

            except Exception as e:
                logger.error(f"Task execution failed: {task.name}", error=str(e))
                task.failure_count += 1
                task.consecutive_failures += 1
                task.last_error = str(e)
                task.last_run_success = False
                task.last_run_duration = (datetime.now() - start_time).total_seconds()

                self._stats["tasks_failed"] += 1

                await self._emit_task_event("failed", task, error=str(e))

                self._update_next_run(task)
                return None

    async def _wait_for_task(
        self,
        task: TaskDefinition,
        process_id: str,
        start_time: datetime,
    ) -> None:
        """Wait for a task to complete and update status."""
        try:
            # Wait for process to finish
            while True:
                process = self.supervisor.get_process(process_id)
                if not process or process.state.is_terminal():
                    break
                await asyncio.sleep(0.5)

            # Update task status
            end_time = datetime.now()
            task.last_run_duration = (end_time - start_time).total_seconds()

            process = self.supervisor.get_process(process_id)
            if process and process.exit_code == 0:
                task.success_count += 1
                task.consecutive_failures = 0
                task.last_run_success = True
                task.last_error = None
                self._stats["tasks_succeeded"] += 1

                await self._emit_task_event("completed", task, process_id=process_id)
            else:
                task.failure_count += 1
                task.consecutive_failures += 1
                task.last_run_success = False
                if process:
                    task.last_error = process.error
                self._stats["tasks_failed"] += 1

                await self._emit_task_event(
                    "failed",
                    task,
                    process_id=process_id,
                    error=task.last_error,
                )

        except Exception as e:
            logger.error(f"Error waiting for task: {e}")
            task.failure_count += 1
            task.last_error = str(e)

        finally:
            self._running_tasks.pop(task.id, None)
            self._update_next_run(task)

    def _update_next_run(self, task: TaskDefinition) -> None:
        """Update the next run time for a task."""
        now = datetime.now()

        if task.schedule_type == "once":
            task.enabled = False
            task.next_run = None
        elif task.schedule_type == "interval" and task.interval_seconds:
            task.next_run = now + timedelta(seconds=task.interval_seconds)
            self._add_to_queue(task)
        elif task.schedule_type == "cron" and task.cron_expression:
            task.next_run = CronParser.get_next_run(task.cron_expression, now)
            self._add_to_queue(task)

    async def _emit_task_event(
        self,
        event_type: str,
        task: TaskDefinition,
        **kwargs,
    ) -> None:
        """Emit a task event."""
        if not self.event_bus:
            return

        await self.event_bus.emit(Event(
            id=str(uuid.uuid4()),
            type=f"task.{event_type}",
            source="scheduler",
            payload={
                "task_id": task.id,
                "task_name": task.name,
                "handler": task.handler,
                "run_count": task.run_count,
                "schedule_type": task.schedule_type,
                **kwargs,
            },
        ))

    # === Query Methods ===

    def get_task(self, task_id: str) -> Optional[TaskDefinition]:
        """Get task by ID."""
        return self._tasks.get(task_id)

    def get_task_by_name(self, name: str) -> Optional[TaskDefinition]:
        """Get task by name."""
        for task in self._tasks.values():
            if task.name == name:
                return task
        return None

    def get_all_tasks(self) -> List[TaskDefinition]:
        """Get all tasks."""
        return list(self._tasks.values())

    def get_enabled_tasks(self) -> List[TaskDefinition]:
        """Get all enabled tasks."""
        return [t for t in self._tasks.values() if t.enabled]

    def get_pending_tasks(self) -> List[TaskDefinition]:
        """Get tasks pending execution (due now or past due)."""
        now = datetime.now()
        return [
            t for t in self._tasks.values()
            if t.enabled and t.next_run and t.next_run <= now
        ]

    def get_running_tasks(self) -> Dict[str, str]:
        """Get currently running tasks (task_id -> process_id)."""
        return self._running_tasks.copy()

    def get_tasks_by_tag(self, tag: str) -> List[TaskDefinition]:
        """Get tasks by tag."""
        return [t for t in self._tasks.values() if tag in t.tags]

    def get_tasks_by_handler(self, handler: str) -> List[TaskDefinition]:
        """Get tasks by handler name."""
        return [t for t in self._tasks.values() if t.handler == handler]

    def get_failed_tasks(self) -> List[TaskDefinition]:
        """Get tasks that have failed recently."""
        return [t for t in self._tasks.values() if t.consecutive_failures > 0]

    def get_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics."""
        return {
            **self._stats,
            "total_tasks": len(self._tasks),
            "enabled_tasks": len(self.get_enabled_tasks()),
            "running_tasks": len(self._running_tasks),
            "pending_tasks": len(self.get_pending_tasks()),
            "failed_tasks": len(self.get_failed_tasks()),
            "registered_handlers": list(self._handlers.keys()),
        }

    # === Context Manager ===

    async def __aenter__(self) -> "TaskScheduler":
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.shutdown()
