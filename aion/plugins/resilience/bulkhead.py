"""
Bulkhead Pattern Implementation

Isolates plugin resources to prevent one plugin from exhausting
system resources and affecting others.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, TypeVar
from functools import wraps

import structlog

logger = structlog.get_logger(__name__)

T = TypeVar("T")


class BulkheadFullError(Exception):
    """Raised when bulkhead capacity is exhausted."""

    def __init__(self, name: str, max_concurrent: int, queue_size: int):
        self.name = name
        self.max_concurrent = max_concurrent
        self.queue_size = queue_size
        super().__init__(
            f"Bulkhead '{name}' is full: "
            f"max_concurrent={max_concurrent}, queue_size={queue_size}"
        )


@dataclass
class BulkheadConfig:
    """Configuration for a bulkhead."""

    # Maximum concurrent executions
    max_concurrent: int = 10

    # Maximum waiting queue size (0 = no queue)
    max_queue_size: int = 100

    # Queue timeout in seconds (0 = wait forever)
    queue_timeout: float = 30.0

    # Execution timeout in seconds (0 = no timeout)
    execution_timeout: float = 60.0

    # Callbacks
    on_full: Optional[Callable[[], None]] = None
    on_slot_released: Optional[Callable[[], None]] = None


@dataclass
class BulkheadMetrics:
    """Metrics for a bulkhead."""

    total_calls: int = 0
    successful_calls: int = 0
    rejected_calls: int = 0
    timed_out_calls: int = 0
    current_concurrent: int = 0
    current_queue_size: int = 0
    max_concurrent_seen: int = 0
    max_queue_seen: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "rejected_calls": self.rejected_calls,
            "timed_out_calls": self.timed_out_calls,
            "current_concurrent": self.current_concurrent,
            "current_queue_size": self.current_queue_size,
            "max_concurrent_seen": self.max_concurrent_seen,
            "max_queue_seen": self.max_queue_seen,
        }


class Bulkhead:
    """
    Bulkhead implementation for resource isolation.

    Limits concurrent executions and provides queuing for overflow,
    preventing resource exhaustion.

    Usage:
        bulkhead = Bulkhead("my-plugin", BulkheadConfig(max_concurrent=5))

        async with bulkhead:
            result = await resource_intensive_operation()

        # Or as decorator
        @bulkhead
        async def my_function():
            ...
    """

    def __init__(
        self,
        name: str,
        config: Optional[BulkheadConfig] = None,
    ):
        self.name = name
        self.config = config or BulkheadConfig()
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent)
        self._metrics = BulkheadMetrics()
        self._queue: asyncio.Queue = asyncio.Queue(
            maxsize=self.config.max_queue_size if self.config.max_queue_size > 0 else 0
        )
        self._lock = asyncio.Lock()

    @property
    def metrics(self) -> BulkheadMetrics:
        """Get bulkhead metrics."""
        return self._metrics

    def available_permits(self) -> int:
        """Get number of available execution slots."""
        # Note: _value is internal but there's no public API for this
        return max(0, self.config.max_concurrent - self._metrics.current_concurrent)

    def queue_space(self) -> int:
        """Get remaining queue space."""
        if self.config.max_queue_size == 0:
            return float("inf")
        return max(0, self.config.max_queue_size - self._metrics.current_queue_size)

    async def __aenter__(self):
        """Enter context manager."""
        await self._acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        await self._release()
        if exc_type is None:
            self._metrics.successful_calls += 1
        return False

    def __call__(self, func: Callable) -> Callable:
        """Use as decorator."""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            async with self:
                if self.config.execution_timeout > 0:
                    return await asyncio.wait_for(
                        func(*args, **kwargs),
                        timeout=self.config.execution_timeout,
                    )
                return await func(*args, **kwargs)
        return wrapper

    async def execute(
        self,
        func: Callable[..., T],
        *args,
        **kwargs,
    ) -> T:
        """Execute a function through the bulkhead."""
        async with self:
            if self.config.execution_timeout > 0:
                try:
                    return await asyncio.wait_for(
                        func(*args, **kwargs),
                        timeout=self.config.execution_timeout,
                    )
                except asyncio.TimeoutError:
                    self._metrics.timed_out_calls += 1
                    raise
            return await func(*args, **kwargs)

    async def _acquire(self) -> None:
        """Acquire a slot in the bulkhead."""
        async with self._lock:
            self._metrics.total_calls += 1

            # Check if we can acquire immediately
            if self._semaphore.locked():
                # Need to queue
                if self.config.max_queue_size > 0:
                    if self._metrics.current_queue_size >= self.config.max_queue_size:
                        self._metrics.rejected_calls += 1
                        if self.config.on_full:
                            self.config.on_full()
                        raise BulkheadFullError(
                            self.name,
                            self.config.max_concurrent,
                            self.config.max_queue_size,
                        )

                self._metrics.current_queue_size += 1
                self._metrics.max_queue_seen = max(
                    self._metrics.max_queue_seen,
                    self._metrics.current_queue_size,
                )

        # Wait for semaphore (outside lock)
        try:
            if self.config.queue_timeout > 0:
                await asyncio.wait_for(
                    self._semaphore.acquire(),
                    timeout=self.config.queue_timeout,
                )
            else:
                await self._semaphore.acquire()

            async with self._lock:
                if self._metrics.current_queue_size > 0:
                    self._metrics.current_queue_size -= 1
                self._metrics.current_concurrent += 1
                self._metrics.max_concurrent_seen = max(
                    self._metrics.max_concurrent_seen,
                    self._metrics.current_concurrent,
                )

        except asyncio.TimeoutError:
            async with self._lock:
                self._metrics.current_queue_size -= 1
                self._metrics.timed_out_calls += 1
            raise

    async def _release(self) -> None:
        """Release a slot in the bulkhead."""
        self._semaphore.release()
        async with self._lock:
            self._metrics.current_concurrent -= 1
            if self.config.on_slot_released:
                self.config.on_slot_released()

    def get_stats(self) -> dict[str, Any]:
        """Get bulkhead statistics."""
        return {
            "name": self.name,
            "config": {
                "max_concurrent": self.config.max_concurrent,
                "max_queue_size": self.config.max_queue_size,
                "queue_timeout": self.config.queue_timeout,
                "execution_timeout": self.config.execution_timeout,
            },
            "metrics": self._metrics.to_dict(),
            "available_permits": self.available_permits(),
            "queue_space": self.queue_space(),
        }


class BulkheadRegistry:
    """
    Registry for managing multiple bulkheads.

    Provides centralized management of bulkheads across plugins.
    """

    def __init__(self):
        self._bulkheads: dict[str, Bulkhead] = {}
        self._lock = asyncio.Lock()

    async def get_or_create(
        self,
        name: str,
        config: Optional[BulkheadConfig] = None,
    ) -> Bulkhead:
        """Get existing or create new bulkhead."""
        async with self._lock:
            if name not in self._bulkheads:
                self._bulkheads[name] = Bulkhead(name, config)
            return self._bulkheads[name]

    def get(self, name: str) -> Optional[Bulkhead]:
        """Get bulkhead by name."""
        return self._bulkheads.get(name)

    def remove(self, name: str) -> None:
        """Remove a bulkhead."""
        self._bulkheads.pop(name, None)

    def get_stats(self) -> dict[str, Any]:
        """Get statistics for all bulkheads."""
        return {
            "total": len(self._bulkheads),
            "bulkheads": {
                name: bulkhead.get_stats()
                for name, bulkhead in self._bulkheads.items()
            },
        }


class CompositeBulkhead:
    """
    Combines multiple bulkheads for hierarchical resource limiting.

    Example: Per-plugin limit + global system limit
    """

    def __init__(self, bulkheads: list[Bulkhead]):
        self.bulkheads = bulkheads

    async def __aenter__(self):
        """Acquire all bulkheads in order."""
        acquired = []
        try:
            for bulkhead in self.bulkheads:
                await bulkhead._acquire()
                acquired.append(bulkhead)
            return self
        except Exception:
            # Release any acquired bulkheads on failure
            for bulkhead in reversed(acquired):
                await bulkhead._release()
            raise

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Release all bulkheads in reverse order."""
        for bulkhead in reversed(self.bulkheads):
            await bulkhead._release()
            if exc_type is None:
                bulkhead._metrics.successful_calls += 1
        return False
