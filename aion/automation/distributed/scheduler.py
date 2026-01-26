"""
AION Distributed Scheduler

Distributed task scheduling with leader election and persistence.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

import structlog

from aion.automation.distributed.queue import TaskQueue, TaskPriority

logger = structlog.get_logger(__name__)


@dataclass
class ScheduleEntry:
    """A scheduled task entry."""
    id: str
    name: str
    task_name: str
    payload: Dict[str, Any]

    # Schedule type (one of these should be set)
    cron_expression: Optional[str] = None
    interval_seconds: Optional[int] = None
    run_at: Optional[datetime] = None

    # Execution settings
    queue_name: str = "default"
    priority: TaskPriority = TaskPriority.NORMAL
    timeout_seconds: Optional[int] = None
    max_retries: int = 3

    # State
    enabled: bool = True
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    run_count: int = 0
    failure_count: int = 0

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "task_name": self.task_name,
            "payload": self.payload,
            "cron_expression": self.cron_expression,
            "interval_seconds": self.interval_seconds,
            "run_at": self.run_at.isoformat() if self.run_at else None,
            "queue_name": self.queue_name,
            "priority": self.priority.value,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "enabled": self.enabled,
            "last_run": self.last_run.isoformat() if self.last_run else None,
            "next_run": self.next_run.isoformat() if self.next_run else None,
            "run_count": self.run_count,
            "failure_count": self.failure_count,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScheduleEntry":
        return cls(
            id=data["id"],
            name=data["name"],
            task_name=data["task_name"],
            payload=data["payload"],
            cron_expression=data.get("cron_expression"),
            interval_seconds=data.get("interval_seconds"),
            run_at=datetime.fromisoformat(data["run_at"]) if data.get("run_at") else None,
            queue_name=data.get("queue_name", "default"),
            priority=TaskPriority(data.get("priority", TaskPriority.NORMAL.value)),
            timeout_seconds=data.get("timeout_seconds"),
            max_retries=data.get("max_retries", 3),
            enabled=data.get("enabled", True),
            last_run=datetime.fromisoformat(data["last_run"]) if data.get("last_run") else None,
            next_run=datetime.fromisoformat(data["next_run"]) if data.get("next_run") else None,
            run_count=data.get("run_count", 0),
            failure_count=data.get("failure_count", 0),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(),
            metadata=data.get("metadata", {}),
        )

    def calculate_next_run(self, from_time: Optional[datetime] = None) -> Optional[datetime]:
        """Calculate the next run time."""
        from_time = from_time or datetime.now()

        if self.cron_expression:
            try:
                from croniter import croniter
                cron = croniter(self.cron_expression, from_time)
                return cron.get_next(datetime)
            except ImportError:
                logger.warning("croniter not available, cron schedules disabled")
                return None

        elif self.interval_seconds:
            if self.last_run:
                return self.last_run + timedelta(seconds=self.interval_seconds)
            return from_time + timedelta(seconds=self.interval_seconds)

        elif self.run_at:
            if self.run_at > from_time:
                return self.run_at
            return None  # One-time schedule already passed

        return None


class DistributedScheduler:
    """
    Distributed scheduler with leader election.

    Features:
    - Cron-based scheduling
    - Interval-based scheduling
    - One-time schedules
    - Distributed lock for leader election
    - Persistence
    - Catch-up for missed schedules
    """

    def __init__(
        self,
        queue: TaskQueue,
        scheduler_id: Optional[str] = None,
        check_interval: float = 1.0,
        lock_backend: Optional["LockBackend"] = None,
        storage_backend: Optional["SchedulerStorageBackend"] = None,
    ):
        self.queue = queue
        self.scheduler_id = scheduler_id or f"scheduler-{uuid.uuid4().hex[:8]}"
        self.check_interval = check_interval

        # Use provided backends or defaults
        self.lock = lock_backend or InMemoryLock()
        self.storage = storage_backend or InMemorySchedulerStorage()

        # State
        self._schedules: Dict[str, ScheduleEntry] = {}
        self._is_leader = False

        # Control
        self._shutdown_event = asyncio.Event()
        self._scheduler_task: Optional[asyncio.Task] = None
        self._leader_task: Optional[asyncio.Task] = None

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the scheduler."""
        if self._initialized:
            return

        await self.storage.initialize()

        # Load existing schedules
        schedules = await self.storage.load_all()
        for schedule in schedules:
            self._schedules[schedule.id] = schedule

        # Start leader election
        self._leader_task = asyncio.create_task(self._leader_election_loop())

        self._initialized = True
        logger.info(
            "Distributed scheduler initialized",
            scheduler_id=self.scheduler_id,
            schedules=len(self._schedules),
        )

    async def shutdown(self) -> None:
        """Shutdown the scheduler."""
        self._shutdown_event.set()

        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass

        if self._leader_task:
            self._leader_task.cancel()
            try:
                await self._leader_task
            except asyncio.CancelledError:
                pass

        # Release leadership
        if self._is_leader:
            await self.lock.release("scheduler_leader")

        await self.storage.shutdown()
        self._initialized = False
        logger.info("Distributed scheduler shutdown")

    async def _leader_election_loop(self) -> None:
        """Continuously try to become/stay leader."""
        while not self._shutdown_event.is_set():
            try:
                # Try to acquire leadership
                acquired = await self.lock.acquire(
                    "scheduler_leader",
                    holder=self.scheduler_id,
                    ttl_seconds=30,
                )

                if acquired and not self._is_leader:
                    # Became leader
                    self._is_leader = True
                    logger.info("Became scheduler leader", scheduler_id=self.scheduler_id)

                    # Start scheduler loop
                    self._scheduler_task = asyncio.create_task(self._scheduler_loop())

                elif not acquired and self._is_leader:
                    # Lost leadership
                    self._is_leader = False
                    logger.warning("Lost scheduler leadership", scheduler_id=self.scheduler_id)

                    if self._scheduler_task:
                        self._scheduler_task.cancel()

                elif self._is_leader:
                    # Refresh lock
                    await self.lock.refresh("scheduler_leader", holder=self.scheduler_id)

                await asyncio.sleep(10)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Leader election error", error=str(e))
                await asyncio.sleep(5)

    async def _scheduler_loop(self) -> None:
        """Main scheduler loop (runs on leader only)."""
        while not self._shutdown_event.is_set() and self._is_leader:
            try:
                now = datetime.now()

                for schedule in list(self._schedules.values()):
                    if not schedule.enabled:
                        continue

                    if schedule.next_run and schedule.next_run <= now:
                        await self._trigger_schedule(schedule)

                await asyncio.sleep(self.check_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Scheduler loop error", error=str(e))
                await asyncio.sleep(1)

    async def _trigger_schedule(self, schedule: ScheduleEntry) -> None:
        """Trigger a scheduled task."""
        logger.info(
            "Triggering scheduled task",
            schedule_id=schedule.id,
            name=schedule.name,
            task_name=schedule.task_name,
        )

        try:
            # Enqueue the task
            await self.queue.enqueue(
                name=schedule.task_name,
                payload=schedule.payload,
                priority=schedule.priority,
                queue_name=schedule.queue_name,
                timeout_seconds=schedule.timeout_seconds,
                max_retries=schedule.max_retries,
                metadata={
                    "schedule_id": schedule.id,
                    "schedule_name": schedule.name,
                    "scheduled_at": schedule.next_run.isoformat() if schedule.next_run else None,
                },
            )

            # Update schedule state
            schedule.last_run = datetime.now()
            schedule.run_count += 1
            schedule.next_run = schedule.calculate_next_run()

            # If one-time schedule, disable it
            if schedule.run_at and not schedule.next_run:
                schedule.enabled = False

            await self.storage.save(schedule)

        except Exception as e:
            logger.error(
                "Failed to trigger schedule",
                schedule_id=schedule.id,
                error=str(e),
            )
            schedule.failure_count += 1
            await self.storage.save(schedule)

    async def add_schedule(
        self,
        name: str,
        task_name: str,
        payload: Dict[str, Any],
        cron_expression: Optional[str] = None,
        interval_seconds: Optional[int] = None,
        run_at: Optional[datetime] = None,
        queue_name: str = "default",
        priority: TaskPriority = TaskPriority.NORMAL,
        timeout_seconds: Optional[int] = None,
        max_retries: int = 3,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ScheduleEntry:
        """Add a new schedule."""
        schedule = ScheduleEntry(
            id=str(uuid.uuid4()),
            name=name,
            task_name=task_name,
            payload=payload,
            cron_expression=cron_expression,
            interval_seconds=interval_seconds,
            run_at=run_at,
            queue_name=queue_name,
            priority=priority,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            metadata=metadata or {},
        )

        # Calculate initial next_run
        schedule.next_run = schedule.calculate_next_run()

        # Store
        self._schedules[schedule.id] = schedule
        await self.storage.save(schedule)

        logger.info(
            "Schedule added",
            schedule_id=schedule.id,
            name=name,
            next_run=schedule.next_run.isoformat() if schedule.next_run else None,
        )

        return schedule

    async def remove_schedule(self, schedule_id: str) -> bool:
        """Remove a schedule."""
        if schedule_id not in self._schedules:
            return False

        del self._schedules[schedule_id]
        await self.storage.delete(schedule_id)

        logger.info("Schedule removed", schedule_id=schedule_id)
        return True

    async def enable_schedule(self, schedule_id: str) -> bool:
        """Enable a schedule."""
        schedule = self._schedules.get(schedule_id)
        if not schedule:
            return False

        schedule.enabled = True
        schedule.next_run = schedule.calculate_next_run()
        await self.storage.save(schedule)

        logger.info("Schedule enabled", schedule_id=schedule_id)
        return True

    async def disable_schedule(self, schedule_id: str) -> bool:
        """Disable a schedule."""
        schedule = self._schedules.get(schedule_id)
        if not schedule:
            return False

        schedule.enabled = False
        await self.storage.save(schedule)

        logger.info("Schedule disabled", schedule_id=schedule_id)
        return True

    async def get_schedule(self, schedule_id: str) -> Optional[ScheduleEntry]:
        """Get a schedule by ID."""
        return self._schedules.get(schedule_id)

    async def list_schedules(
        self,
        enabled_only: bool = False,
    ) -> List[ScheduleEntry]:
        """List all schedules."""
        schedules = list(self._schedules.values())
        if enabled_only:
            schedules = [s for s in schedules if s.enabled]
        return schedules

    async def trigger_now(self, schedule_id: str) -> bool:
        """Manually trigger a schedule immediately."""
        schedule = self._schedules.get(schedule_id)
        if not schedule:
            return False

        await self._trigger_schedule(schedule)
        return True

    def get_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics."""
        schedules = list(self._schedules.values())
        return {
            "scheduler_id": self.scheduler_id,
            "is_leader": self._is_leader,
            "total_schedules": len(schedules),
            "enabled_schedules": len([s for s in schedules if s.enabled]),
            "total_runs": sum(s.run_count for s in schedules),
            "total_failures": sum(s.failure_count for s in schedules),
        }


# Lock backends for leader election

class LockBackend:
    """Abstract lock backend for distributed locking."""

    async def acquire(
        self,
        key: str,
        holder: str,
        ttl_seconds: int = 30,
    ) -> bool:
        raise NotImplementedError

    async def release(self, key: str) -> bool:
        raise NotImplementedError

    async def refresh(
        self,
        key: str,
        holder: str,
        ttl_seconds: int = 30,
    ) -> bool:
        raise NotImplementedError


class InMemoryLock(LockBackend):
    """In-memory lock for single-node deployments."""

    def __init__(self):
        self._locks: Dict[str, tuple[str, float]] = {}  # key -> (holder, expires_at)

    async def acquire(
        self,
        key: str,
        holder: str,
        ttl_seconds: int = 30,
    ) -> bool:
        now = time.time()

        if key in self._locks:
            current_holder, expires_at = self._locks[key]
            if expires_at > now and current_holder != holder:
                return False

        self._locks[key] = (holder, now + ttl_seconds)
        return True

    async def release(self, key: str) -> bool:
        if key in self._locks:
            del self._locks[key]
            return True
        return False

    async def refresh(
        self,
        key: str,
        holder: str,
        ttl_seconds: int = 30,
    ) -> bool:
        now = time.time()

        if key not in self._locks:
            return False

        current_holder, _ = self._locks[key]
        if current_holder != holder:
            return False

        self._locks[key] = (holder, now + ttl_seconds)
        return True


class RedisLock(LockBackend):
    """Redis-based distributed lock."""

    def __init__(self, redis_url: str = "redis://localhost:6379", prefix: str = "aion:lock:"):
        self.redis_url = redis_url
        self.prefix = prefix
        self._client = None

    async def _get_client(self):
        if self._client is None:
            try:
                import redis.asyncio as redis
                self._client = redis.from_url(self.redis_url)
            except ImportError:
                raise ImportError("redis package required for RedisLock")
        return self._client

    async def acquire(
        self,
        key: str,
        holder: str,
        ttl_seconds: int = 30,
    ) -> bool:
        client = await self._get_client()
        lock_key = f"{self.prefix}{key}"

        # Use SET NX EX for atomic lock acquisition
        result = await client.set(
            lock_key,
            holder,
            nx=True,
            ex=ttl_seconds,
        )

        return result is not None

    async def release(self, key: str) -> bool:
        client = await self._get_client()
        lock_key = f"{self.prefix}{key}"

        result = await client.delete(lock_key)
        return result > 0

    async def refresh(
        self,
        key: str,
        holder: str,
        ttl_seconds: int = 30,
    ) -> bool:
        client = await self._get_client()
        lock_key = f"{self.prefix}{key}"

        # Verify holder and refresh
        current = await client.get(lock_key)
        if current and current.decode() == holder:
            await client.expire(lock_key, ttl_seconds)
            return True
        return False


# Storage backends for schedule persistence

class SchedulerStorageBackend:
    """Abstract storage backend for schedules."""

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    async def save(self, schedule: ScheduleEntry) -> None:
        raise NotImplementedError

    async def delete(self, schedule_id: str) -> bool:
        raise NotImplementedError

    async def load(self, schedule_id: str) -> Optional[ScheduleEntry]:
        raise NotImplementedError

    async def load_all(self) -> List[ScheduleEntry]:
        raise NotImplementedError


class InMemorySchedulerStorage(SchedulerStorageBackend):
    """In-memory schedule storage."""

    def __init__(self):
        self._schedules: Dict[str, ScheduleEntry] = {}

    async def save(self, schedule: ScheduleEntry) -> None:
        self._schedules[schedule.id] = schedule

    async def delete(self, schedule_id: str) -> bool:
        if schedule_id in self._schedules:
            del self._schedules[schedule_id]
            return True
        return False

    async def load(self, schedule_id: str) -> Optional[ScheduleEntry]:
        return self._schedules.get(schedule_id)

    async def load_all(self) -> List[ScheduleEntry]:
        return list(self._schedules.values())


class RedisSchedulerStorage(SchedulerStorageBackend):
    """Redis-based schedule storage."""

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        prefix: str = "aion:schedules:",
    ):
        self.redis_url = redis_url
        self.prefix = prefix
        self._client = None

    async def _get_client(self):
        if self._client is None:
            try:
                import redis.asyncio as redis
                self._client = redis.from_url(self.redis_url)
            except ImportError:
                raise ImportError("redis package required for RedisSchedulerStorage")
        return self._client

    async def initialize(self) -> None:
        await self._get_client()

    async def shutdown(self) -> None:
        if self._client:
            await self._client.close()

    async def save(self, schedule: ScheduleEntry) -> None:
        client = await self._get_client()
        key = f"{self.prefix}{schedule.id}"
        await client.set(key, json.dumps(schedule.to_dict()))

    async def delete(self, schedule_id: str) -> bool:
        client = await self._get_client()
        key = f"{self.prefix}{schedule_id}"
        result = await client.delete(key)
        return result > 0

    async def load(self, schedule_id: str) -> Optional[ScheduleEntry]:
        client = await self._get_client()
        key = f"{self.prefix}{schedule_id}"
        data = await client.get(key)
        if data:
            return ScheduleEntry.from_dict(json.loads(data))
        return None

    async def load_all(self) -> List[ScheduleEntry]:
        client = await self._get_client()
        keys = await client.keys(f"{self.prefix}*")
        schedules = []
        for key in keys:
            data = await client.get(key)
            if data:
                schedules.append(ScheduleEntry.from_dict(json.loads(data)))
        return schedules
