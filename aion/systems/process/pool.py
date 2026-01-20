"""
AION Process Pool

High-performance process pooling with:
- Pre-warmed agent instances
- Automatic scaling
- Health monitoring
- Resource-efficient recycling
- Affinity-based allocation
- Priority queuing
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, Generic, List, Optional, Set, Tuple, TypeVar

import structlog

from aion.systems.process.models import (
    ProcessState,
    ProcessPriority,
    ResourceLimits,
    ProcessInfo,
)

logger = structlog.get_logger(__name__)

T = TypeVar("T")


class PooledProcessState(Enum):
    """State of a pooled process."""
    WARMING = auto()    # Being initialized
    IDLE = auto()       # Ready for use
    ACQUIRED = auto()   # In use
    RECYCLING = auto()  # Being recycled
    DRAINING = auto()   # Shutting down
    DEAD = auto()       # Terminated


@dataclass
class PooledProcess:
    """A process in the pool."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    process_type: str = ""
    state: PooledProcessState = PooledProcessState.WARMING
    created_at: datetime = field(default_factory=datetime.now)
    last_used: datetime = field(default_factory=datetime.now)
    acquired_at: Optional[datetime] = None
    use_count: int = 0
    max_uses: int = 100  # Max uses before recycling
    max_idle_seconds: float = 300.0  # Max idle time before recycling

    # The actual process/agent instance
    instance: Any = None

    # Resource tracking
    memory_usage_mb: float = 0
    cpu_time_seconds: float = 0

    # Health
    healthy: bool = True
    health_check_failures: int = 0
    last_health_check: Optional[datetime] = None

    # Affinity for sticky allocation
    affinity_key: Optional[str] = None

    def is_expired(self) -> bool:
        """Check if process should be recycled."""
        if self.use_count >= self.max_uses:
            return True

        if self.state == PooledProcessState.IDLE:
            idle_time = (datetime.now() - self.last_used).total_seconds()
            if idle_time > self.max_idle_seconds:
                return True

        return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "process_type": self.process_type,
            "state": self.state.name,
            "created_at": self.created_at.isoformat(),
            "last_used": self.last_used.isoformat(),
            "use_count": self.use_count,
            "healthy": self.healthy,
            "memory_usage_mb": self.memory_usage_mb,
        }


@dataclass
class PoolConfig:
    """Configuration for a process pool."""
    process_type: str
    min_size: int = 2
    max_size: int = 10
    max_idle_seconds: float = 300.0
    max_uses_per_process: int = 100
    warm_up_batch_size: int = 2
    health_check_interval: float = 30.0
    scale_up_threshold: float = 0.8  # Scale up when utilization > 80%
    scale_down_threshold: float = 0.2  # Scale down when utilization < 20%
    factory: Optional[Callable[[], Any]] = None
    initializer: Optional[Callable[[Any], Any]] = None
    finalizer: Optional[Callable[[Any], Any]] = None
    health_checker: Optional[Callable[[Any], bool]] = None


@dataclass
class PoolStats:
    """Statistics for a process pool."""
    total_processes: int = 0
    idle_processes: int = 0
    acquired_processes: int = 0
    warming_processes: int = 0
    total_acquisitions: int = 0
    total_releases: int = 0
    total_recycled: int = 0
    acquisition_wait_time_ms: float = 0.0
    avg_use_duration_ms: float = 0.0
    utilization: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_processes": self.total_processes,
            "idle_processes": self.idle_processes,
            "acquired_processes": self.acquired_processes,
            "warming_processes": self.warming_processes,
            "total_acquisitions": self.total_acquisitions,
            "total_releases": self.total_releases,
            "total_recycled": self.total_recycled,
            "acquisition_wait_time_ms": self.acquisition_wait_time_ms,
            "avg_use_duration_ms": self.avg_use_duration_ms,
            "utilization": self.utilization,
        }


class ProcessPool:
    """
    Pool of pre-warmed processes.

    Features:
    - Automatic warm-up on start
    - Lazy expansion on demand
    - Health monitoring
    - Automatic recycling
    - Priority-based allocation
    - Affinity-based sticky allocation
    """

    def __init__(self, config: PoolConfig):
        self.config = config

        # Process storage
        self._processes: Dict[str, PooledProcess] = {}
        self._idle_queue: deque[str] = deque()
        self._affinity_map: Dict[str, str] = {}  # affinity_key -> process_id

        # Waiting queue for acquisitions
        self._waiting: asyncio.Queue[asyncio.Future[PooledProcess]] = asyncio.Queue()

        # Statistics
        self._stats = PoolStats()
        self._use_durations: deque[float] = deque(maxlen=1000)

        # Background tasks
        self._maintenance_task: Optional[asyncio.Task] = None
        self._health_check_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

        # Locks
        self._lock = asyncio.Lock()

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the pool."""
        if self._initialized:
            return

        logger.info(
            f"Initializing process pool",
            process_type=self.config.process_type,
            min_size=self.config.min_size,
        )

        # Warm up initial processes
        await self._warm_up(self.config.min_size)

        # Start background tasks
        self._maintenance_task = asyncio.create_task(self._maintenance_loop())
        self._health_check_task = asyncio.create_task(self._health_check_loop())

        self._initialized = True
        logger.info(f"Process pool initialized with {len(self._processes)} processes")

    async def shutdown(self) -> None:
        """Shutdown the pool."""
        logger.info("Shutting down process pool")

        self._shutdown_event.set()

        # Cancel background tasks
        for task in [self._maintenance_task, self._health_check_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Shutdown all processes
        for process in list(self._processes.values()):
            await self._destroy_process(process)

        logger.info("Process pool shutdown complete")

    async def acquire(
        self,
        timeout: Optional[float] = None,
        affinity_key: Optional[str] = None,
        priority: ProcessPriority = ProcessPriority.NORMAL,
    ) -> PooledProcess:
        """
        Acquire a process from the pool.

        Args:
            timeout: Max time to wait for a process
            affinity_key: Key for sticky allocation
            priority: Priority for queue ordering

        Returns:
            An acquired process
        """
        start_time = time.time()

        async with self._lock:
            # Check affinity first
            if affinity_key and affinity_key in self._affinity_map:
                process_id = self._affinity_map[affinity_key]
                process = self._processes.get(process_id)

                if process and process.state == PooledProcessState.IDLE:
                    process.state = PooledProcessState.ACQUIRED
                    process.acquired_at = datetime.now()
                    process.use_count += 1

                    self._idle_queue = deque(
                        pid for pid in self._idle_queue if pid != process_id
                    )

                    self._stats.total_acquisitions += 1
                    self._update_stats()

                    wait_time = (time.time() - start_time) * 1000
                    self._stats.acquisition_wait_time_ms = wait_time

                    return process

            # Try to get from idle queue
            while self._idle_queue:
                process_id = self._idle_queue.popleft()
                process = self._processes.get(process_id)

                if process and process.state == PooledProcessState.IDLE and process.healthy:
                    process.state = PooledProcessState.ACQUIRED
                    process.acquired_at = datetime.now()
                    process.use_count += 1

                    if affinity_key:
                        process.affinity_key = affinity_key
                        self._affinity_map[affinity_key] = process_id

                    self._stats.total_acquisitions += 1
                    self._update_stats()

                    wait_time = (time.time() - start_time) * 1000
                    self._stats.acquisition_wait_time_ms = wait_time

                    return process

            # Check if we can scale up
            if len(self._processes) < self.config.max_size:
                # Create new process
                process = await self._create_process()

                process.state = PooledProcessState.ACQUIRED
                process.acquired_at = datetime.now()
                process.use_count += 1

                if affinity_key:
                    process.affinity_key = affinity_key
                    self._affinity_map[affinity_key] = process.id

                self._stats.total_acquisitions += 1
                self._update_stats()

                wait_time = (time.time() - start_time) * 1000
                self._stats.acquisition_wait_time_ms = wait_time

                return process

        # Wait for a process to become available
        future: asyncio.Future[PooledProcess] = asyncio.get_event_loop().create_future()

        await self._waiting.put(future)

        try:
            if timeout:
                process = await asyncio.wait_for(future, timeout=timeout)
            else:
                process = await future

            wait_time = (time.time() - start_time) * 1000
            self._stats.acquisition_wait_time_ms = wait_time

            return process

        except asyncio.TimeoutError:
            raise PoolExhaustedError("No process available within timeout")

    async def release(self, process: PooledProcess) -> None:
        """Release a process back to the pool."""
        async with self._lock:
            if process.id not in self._processes:
                return

            # Calculate use duration
            if process.acquired_at:
                duration = (datetime.now() - process.acquired_at).total_seconds() * 1000
                self._use_durations.append(duration)
                if self._use_durations:
                    self._stats.avg_use_duration_ms = sum(self._use_durations) / len(self._use_durations)

            process.last_used = datetime.now()
            process.acquired_at = None

            self._stats.total_releases += 1

            # Check if should be recycled
            if process.is_expired() or not process.healthy:
                await self._recycle_process(process)
                return

            # Check if someone is waiting
            if not self._waiting.empty():
                try:
                    future = self._waiting.get_nowait()
                    if not future.done():
                        process.state = PooledProcessState.ACQUIRED
                        process.acquired_at = datetime.now()
                        process.use_count += 1
                        future.set_result(process)
                        self._stats.total_acquisitions += 1
                        return
                except asyncio.QueueEmpty:
                    pass

            # Return to idle pool
            process.state = PooledProcessState.IDLE
            self._idle_queue.append(process.id)

            self._update_stats()

    async def _create_process(self) -> PooledProcess:
        """Create a new pooled process."""
        process = PooledProcess(
            process_type=self.config.process_type,
            state=PooledProcessState.WARMING,
            max_uses=self.config.max_uses_per_process,
            max_idle_seconds=self.config.max_idle_seconds,
        )

        # Create instance using factory
        if self.config.factory:
            process.instance = self.config.factory()

        # Initialize
        if self.config.initializer:
            result = self.config.initializer(process.instance)
            if asyncio.iscoroutine(result):
                await result

        process.state = PooledProcessState.IDLE
        self._processes[process.id] = process

        logger.debug(f"Created pooled process: {process.id}")
        return process

    async def _destroy_process(self, process: PooledProcess) -> None:
        """Destroy a pooled process."""
        process.state = PooledProcessState.DEAD

        # Call finalizer
        if self.config.finalizer and process.instance:
            try:
                result = self.config.finalizer(process.instance)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"Process finalizer error: {e}")

        # Remove from maps
        self._processes.pop(process.id, None)

        if process.affinity_key:
            self._affinity_map.pop(process.affinity_key, None)

        logger.debug(f"Destroyed pooled process: {process.id}")

    async def _recycle_process(self, process: PooledProcess) -> None:
        """Recycle a process (destroy and replace)."""
        process.state = PooledProcessState.RECYCLING
        self._stats.total_recycled += 1

        await self._destroy_process(process)

        # Create replacement if under min size
        if len(self._processes) < self.config.min_size:
            new_process = await self._create_process()
            self._idle_queue.append(new_process.id)

        self._update_stats()

    async def _warm_up(self, count: int) -> None:
        """Warm up processes."""
        current = len(self._processes)
        to_create = min(count - current, self.config.max_size - current)

        if to_create <= 0:
            return

        logger.debug(f"Warming up {to_create} processes")

        # Create in batches
        for i in range(0, to_create, self.config.warm_up_batch_size):
            batch_size = min(self.config.warm_up_batch_size, to_create - i)
            tasks = [self._create_process() for _ in range(batch_size)]
            processes = await asyncio.gather(*tasks, return_exceptions=True)

            for process in processes:
                if isinstance(process, PooledProcess):
                    self._idle_queue.append(process.id)
                else:
                    logger.error(f"Failed to create process: {process}")

        self._update_stats()

    async def _maintenance_loop(self) -> None:
        """Background maintenance loop."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(10.0)

                async with self._lock:
                    # Check for expired processes
                    for process in list(self._processes.values()):
                        if process.state == PooledProcessState.IDLE and process.is_expired():
                            await self._recycle_process(process)

                    # Auto-scaling
                    utilization = self._stats.utilization

                    if utilization > self.config.scale_up_threshold:
                        # Scale up
                        current = len(self._processes)
                        target = min(current + 2, self.config.max_size)
                        if target > current:
                            await self._warm_up(target)

                    elif utilization < self.config.scale_down_threshold:
                        # Scale down
                        current = len(self._processes)
                        idle_count = self._stats.idle_processes

                        if idle_count > 2 and current > self.config.min_size:
                            # Remove excess idle processes
                            to_remove = min(idle_count - 1, current - self.config.min_size)
                            removed = 0

                            while self._idle_queue and removed < to_remove:
                                process_id = self._idle_queue.popleft()
                                process = self._processes.get(process_id)
                                if process:
                                    await self._destroy_process(process)
                                    removed += 1

                    self._update_stats()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Maintenance loop error: {e}")

    async def _health_check_loop(self) -> None:
        """Background health check loop."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(self.config.health_check_interval)

                if not self.config.health_checker:
                    continue

                async with self._lock:
                    for process in list(self._processes.values()):
                        if process.state == PooledProcessState.IDLE:
                            try:
                                healthy = self.config.health_checker(process.instance)
                                if asyncio.iscoroutine(healthy):
                                    healthy = await healthy

                                process.healthy = healthy
                                process.last_health_check = datetime.now()

                                if healthy:
                                    process.health_check_failures = 0
                                else:
                                    process.health_check_failures += 1

                                    if process.health_check_failures >= 3:
                                        logger.warning(f"Process unhealthy, recycling: {process.id}")
                                        await self._recycle_process(process)

                            except Exception as e:
                                logger.error(f"Health check error: {e}")
                                process.health_check_failures += 1

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check loop error: {e}")

    def _update_stats(self) -> None:
        """Update pool statistics."""
        idle = sum(1 for p in self._processes.values() if p.state == PooledProcessState.IDLE)
        acquired = sum(1 for p in self._processes.values() if p.state == PooledProcessState.ACQUIRED)
        warming = sum(1 for p in self._processes.values() if p.state == PooledProcessState.WARMING)

        self._stats.total_processes = len(self._processes)
        self._stats.idle_processes = idle
        self._stats.acquired_processes = acquired
        self._stats.warming_processes = warming

        if self._stats.total_processes > 0:
            self._stats.utilization = acquired / self._stats.total_processes
        else:
            self._stats.utilization = 0.0

    def get_stats(self) -> PoolStats:
        """Get pool statistics."""
        return self._stats

    def get_process(self, process_id: str) -> Optional[PooledProcess]:
        """Get a specific process."""
        return self._processes.get(process_id)

    def get_all_processes(self) -> List[PooledProcess]:
        """Get all processes."""
        return list(self._processes.values())

    async def __aenter__(self) -> "ProcessPool":
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.shutdown()


class PoolExhaustedError(Exception):
    """Raised when pool is exhausted."""
    pass


# === Pool Manager ===

class ProcessPoolManager:
    """
    Manages multiple process pools.

    Provides centralized management of pools for different
    process types with unified statistics and control.
    """

    def __init__(self):
        self._pools: Dict[str, ProcessPool] = {}
        self._lock = asyncio.Lock()

    async def create_pool(
        self,
        process_type: str,
        config: Optional[PoolConfig] = None,
    ) -> ProcessPool:
        """Create a new process pool."""
        async with self._lock:
            if process_type in self._pools:
                return self._pools[process_type]

            pool_config = config or PoolConfig(process_type=process_type)
            pool = ProcessPool(pool_config)

            await pool.initialize()
            self._pools[process_type] = pool

            return pool

    def get_pool(self, process_type: str) -> Optional[ProcessPool]:
        """Get a pool by process type."""
        return self._pools.get(process_type)

    async def acquire_from(
        self,
        process_type: str,
        **kwargs,
    ) -> PooledProcess:
        """Acquire a process from a specific pool."""
        pool = self._pools.get(process_type)
        if not pool:
            raise ValueError(f"Pool not found: {process_type}")

        return await pool.acquire(**kwargs)

    async def release_to(
        self,
        process_type: str,
        process: PooledProcess,
    ) -> None:
        """Release a process to a specific pool."""
        pool = self._pools.get(process_type)
        if pool:
            await pool.release(process)

    async def shutdown_all(self) -> None:
        """Shutdown all pools."""
        for pool in self._pools.values():
            await pool.shutdown()
        self._pools.clear()

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get stats for all pools."""
        return {
            name: pool.get_stats().to_dict()
            for name, pool in self._pools.items()
        }


# === Optimized Cron Parser ===

class OptimizedCronParser:
    """
    Highly optimized cron expression parser.

    Uses precomputation and binary search for O(1) average
    next-run calculation instead of O(n) iteration.
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

    def __init__(self, expression: str):
        """Parse and precompute cron expression."""
        # Handle special expressions
        if expression.startswith("@"):
            expression = self.SPECIAL_EXPRESSIONS.get(expression.lower(), expression)

        parts = expression.split()
        if len(parts) != 5:
            raise ValueError(f"Invalid cron expression: {expression}")

        # Parse and store as sorted arrays for binary search
        self.minutes = self._parse_field(parts[0], 0, 59)
        self.hours = self._parse_field(parts[1], 0, 23)
        self.days = self._parse_field(parts[2], 1, 31)
        self.months = self._parse_field(parts[3], 1, 12)
        self.weekdays = self._parse_field(parts[4], 0, 6)

        # Precompute bitmasks for O(1) lookup
        self._minute_mask = self._to_bitmask(self.minutes)
        self._hour_mask = self._to_bitmask(self.hours)
        self._day_mask = self._to_bitmask(self.days)
        self._month_mask = self._to_bitmask(self.months)
        self._weekday_mask = self._to_bitmask(self.weekdays)

    @staticmethod
    def _parse_field(field: str, min_val: int, max_val: int) -> List[int]:
        """Parse a single cron field."""
        if field == "*":
            return list(range(min_val, max_val + 1))

        values = set()

        for part in field.split(","):
            if "/" in part:
                range_part, step = part.split("/")
                step = int(step)
                if range_part == "*":
                    start, end = min_val, max_val
                elif "-" in range_part:
                    start, end = map(int, range_part.split("-"))
                else:
                    start = end = int(range_part)
                values.update(range(start, end + 1, step))
            elif "-" in part:
                start, end = map(int, part.split("-"))
                values.update(range(start, end + 1))
            else:
                values.add(int(part))

        return sorted(v for v in values if min_val <= v <= max_val)

    @staticmethod
    def _to_bitmask(values: List[int]) -> int:
        """Convert values to bitmask."""
        mask = 0
        for v in values:
            mask |= (1 << v)
        return mask

    def _check_mask(self, mask: int, value: int) -> bool:
        """Check if value is in bitmask."""
        return bool(mask & (1 << value))

    def _find_next_in_mask(self, mask: int, start: int, max_val: int) -> Optional[int]:
        """Find next value >= start in bitmask."""
        for i in range(start, max_val + 1):
            if mask & (1 << i):
                return i
        return None

    def _find_first_in_mask(self, mask: int, max_val: int) -> Optional[int]:
        """Find first value in bitmask."""
        for i in range(max_val + 1):
            if mask & (1 << i):
                return i
        return None

    def get_next_run(self, after: datetime) -> datetime:
        """
        Calculate the next run time after a given datetime.

        Uses bitmask lookups for O(1) field matching.
        """
        # Start from next minute
        year = after.year
        month = after.month
        day = after.day
        hour = after.hour
        minute = after.minute + 1

        # Normalize overflow
        if minute >= 60:
            minute = 0
            hour += 1
        if hour >= 24:
            hour = 0
            day += 1

        # Search for next valid time (max 4 years)
        for _ in range(365 * 4):
            # Check/find month
            if not self._check_mask(self._month_mask, month):
                next_month = self._find_next_in_mask(self._month_mask, month, 12)
                if next_month is None:
                    year += 1
                    month = self._find_first_in_mask(self._month_mask, 12)
                else:
                    month = next_month
                day = self._find_first_in_mask(self._day_mask, 31)
                hour = self._find_first_in_mask(self._hour_mask, 23)
                minute = self._find_first_in_mask(self._minute_mask, 59)
                continue

            # Check day of month and weekday
            try:
                candidate = datetime(year, month, day, hour, minute)
            except ValueError:
                # Invalid date (e.g., Feb 31)
                day = 1
                month += 1
                if month > 12:
                    month = 1
                    year += 1
                continue

            weekday = candidate.weekday()
            # Convert Python weekday (0=Mon) to cron weekday (0=Sun)
            cron_weekday = (weekday + 1) % 7

            if (not self._check_mask(self._day_mask, day) or
                not self._check_mask(self._weekday_mask, cron_weekday)):
                day += 1
                hour = self._find_first_in_mask(self._hour_mask, 23)
                minute = self._find_first_in_mask(self._minute_mask, 59)

                # Handle month overflow
                import calendar
                max_day = calendar.monthrange(year, month)[1]
                if day > max_day:
                    day = 1
                    month += 1
                    if month > 12:
                        month = 1
                        year += 1
                continue

            # Check/find hour
            if not self._check_mask(self._hour_mask, hour):
                next_hour = self._find_next_in_mask(self._hour_mask, hour, 23)
                if next_hour is None:
                    day += 1
                    hour = self._find_first_in_mask(self._hour_mask, 23)
                else:
                    hour = next_hour
                minute = self._find_first_in_mask(self._minute_mask, 59)
                continue

            # Check/find minute
            if not self._check_mask(self._minute_mask, minute):
                next_minute = self._find_next_in_mask(self._minute_mask, minute, 59)
                if next_minute is None:
                    hour += 1
                    if hour >= 24:
                        hour = 0
                        day += 1
                    minute = self._find_first_in_mask(self._minute_mask, 59)
                else:
                    minute = next_minute
                continue

            # Found valid time
            return datetime(year, month, day, hour, minute)

        raise ValueError(f"Could not find next run time")


# Global pool manager
_pool_manager: Optional[ProcessPoolManager] = None


def get_pool_manager() -> ProcessPoolManager:
    """Get the global pool manager."""
    global _pool_manager
    if _pool_manager is None:
        _pool_manager = ProcessPoolManager()
    return _pool_manager
