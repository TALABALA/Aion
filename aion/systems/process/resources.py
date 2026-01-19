"""
AION Resource Management

State-of-the-art resource management and quota enforcement:
- Per-process resource limits
- System-wide resource budgets
- Token rate limiting
- Memory pressure handling
- CPU quota enforcement
- Graceful degradation under pressure
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable
from enum import Enum

import structlog

from aion.systems.process.models import (
    ResourceLimits,
    ResourceUsage,
    ProcessPriority,
)

logger = structlog.get_logger(__name__)


class ResourcePressure(Enum):
    """System resource pressure levels."""
    NONE = "none"           # All resources available
    LOW = "low"             # Some pressure, prefer conservation
    MEDIUM = "medium"       # Significant pressure, limit new processes
    HIGH = "high"           # High pressure, pause low-priority processes
    CRITICAL = "critical"   # Critical, terminate non-essential processes


@dataclass
class TokenBucket:
    """
    Token bucket for rate limiting.

    Implements the token bucket algorithm for smooth rate limiting.
    """
    capacity: int  # Max tokens
    refill_rate: float  # Tokens per second
    tokens: float = 0.0
    last_refill: float = field(default_factory=time.time)

    def __post_init__(self):
        self.tokens = float(self.capacity)

    def consume(self, amount: int = 1) -> bool:
        """
        Try to consume tokens.

        Returns True if tokens were available, False otherwise.
        """
        self._refill()

        if self.tokens >= amount:
            self.tokens -= amount
            return True
        return False

    async def consume_async(self, amount: int = 1, timeout: float = 60.0) -> bool:
        """
        Async consume with wait.

        Waits up to timeout for tokens to become available.
        """
        start = time.time()

        while (time.time() - start) < timeout:
            if self.consume(amount):
                return True

            # Calculate wait time for tokens
            wait_time = (amount - self.tokens) / self.refill_rate
            wait_time = min(wait_time, timeout - (time.time() - start))

            if wait_time <= 0:
                return False

            await asyncio.sleep(min(wait_time, 0.1))

        return False

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now

    def get_available(self) -> float:
        """Get number of available tokens."""
        self._refill()
        return self.tokens


@dataclass
class SlidingWindowRateLimiter:
    """
    Sliding window rate limiter.

    More accurate than token bucket for bursty traffic.
    """
    max_requests: int
    window_seconds: float
    _requests: deque = field(default_factory=deque)

    def allow(self) -> bool:
        """Check if request is allowed."""
        now = time.time()
        cutoff = now - self.window_seconds

        # Remove old requests
        while self._requests and self._requests[0] < cutoff:
            self._requests.popleft()

        if len(self._requests) < self.max_requests:
            self._requests.append(now)
            return True

        return False

    def get_current_rate(self) -> float:
        """Get current request rate per second."""
        now = time.time()
        cutoff = now - self.window_seconds

        while self._requests and self._requests[0] < cutoff:
            self._requests.popleft()

        if not self._requests:
            return 0.0

        return len(self._requests) / self.window_seconds


@dataclass
class ResourceQuota:
    """
    Resource quota for a process or system.

    Tracks usage against limits with soft and hard thresholds.
    """
    name: str
    limit: float
    soft_threshold: float = 0.8  # Warning at 80%
    hard_threshold: float = 0.95  # Critical at 95%
    current: float = 0.0

    @property
    def usage_percent(self) -> float:
        """Get usage as percentage of limit."""
        if self.limit <= 0:
            return 0.0
        return (self.current / self.limit) * 100

    @property
    def remaining(self) -> float:
        """Get remaining quota."""
        return max(0, self.limit - self.current)

    @property
    def is_soft_exceeded(self) -> bool:
        """Check if soft threshold exceeded."""
        return self.current >= (self.limit * self.soft_threshold)

    @property
    def is_hard_exceeded(self) -> bool:
        """Check if hard threshold exceeded."""
        return self.current >= (self.limit * self.hard_threshold)

    @property
    def is_exceeded(self) -> bool:
        """Check if limit exceeded."""
        return self.current >= self.limit

    def consume(self, amount: float) -> bool:
        """Consume quota. Returns False if exceeded."""
        if self.current + amount > self.limit:
            return False
        self.current += amount
        return True

    def release(self, amount: float) -> None:
        """Release quota."""
        self.current = max(0, self.current - amount)

    def reset(self) -> None:
        """Reset quota to zero."""
        self.current = 0.0


class ResourceManager:
    """
    Central resource manager for AION.

    Manages:
    - System-wide resource budgets
    - Per-process resource allocation
    - Token rate limiting
    - Resource pressure detection
    - Graceful degradation policies
    """

    def __init__(
        self,
        system_limits: Optional[ResourceLimits] = None,
        enable_memory_monitoring: bool = True,
        enable_token_limiting: bool = True,
        pressure_check_interval: float = 5.0,
    ):
        self.system_limits = system_limits or ResourceLimits(
            max_memory_mb=4096,
            max_cpu_percent=80.0,
            max_tokens_per_minute=100000,
            max_tokens_total=10000000,
        )
        self.enable_memory_monitoring = enable_memory_monitoring
        self.enable_token_limiting = enable_token_limiting
        self.pressure_check_interval = pressure_check_interval

        # Resource quotas
        self._quotas: Dict[str, ResourceQuota] = {}

        # Token rate limiters (per process)
        self._token_buckets: Dict[str, TokenBucket] = {}
        self._system_token_bucket: Optional[TokenBucket] = None

        # Resource tracking
        self._process_usage: Dict[str, ResourceUsage] = {}

        # Pressure state
        self._pressure = ResourcePressure.NONE
        self._pressure_callbacks: List[Callable] = []

        # Background monitoring
        self._monitor_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the resource manager."""
        if self._initialized:
            return

        logger.info("Initializing Resource Manager")

        # Setup system quotas
        self._quotas["memory"] = ResourceQuota(
            name="memory",
            limit=self.system_limits.max_memory_mb or float("inf"),
        )

        self._quotas["tokens_total"] = ResourceQuota(
            name="tokens_total",
            limit=self.system_limits.max_tokens_total or float("inf"),
        )

        # Setup system token bucket
        if self.enable_token_limiting and self.system_limits.max_tokens_per_minute:
            self._system_token_bucket = TokenBucket(
                capacity=self.system_limits.max_tokens_per_minute,
                refill_rate=self.system_limits.max_tokens_per_minute / 60.0,
            )

        # Start pressure monitoring
        self._monitor_task = asyncio.create_task(self._pressure_monitor_loop())

        self._initialized = True
        logger.info("Resource Manager initialized")

    async def shutdown(self) -> None:
        """Shutdown the resource manager."""
        logger.info("Shutting down Resource Manager")

        self._shutdown_event.set()

        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

    def allocate_for_process(
        self,
        process_id: str,
        limits: ResourceLimits,
    ) -> bool:
        """
        Allocate resources for a process.

        Returns True if allocation successful.
        """
        # Create token bucket for process
        if self.enable_token_limiting and limits.max_tokens_per_minute:
            self._token_buckets[process_id] = TokenBucket(
                capacity=limits.max_tokens_per_minute,
                refill_rate=limits.max_tokens_per_minute / 60.0,
            )

        # Initialize usage tracking
        self._process_usage[process_id] = ResourceUsage()

        logger.debug(f"Allocated resources for process {process_id}")
        return True

    def release_for_process(self, process_id: str) -> None:
        """Release resources allocated to a process."""
        self._token_buckets.pop(process_id, None)

        usage = self._process_usage.pop(process_id, None)
        if usage:
            # Update system quotas
            self._quotas["memory"].release(usage.memory_mb)

        logger.debug(f"Released resources for process {process_id}")

    async def request_tokens(
        self,
        process_id: str,
        amount: int,
        timeout: float = 60.0,
    ) -> bool:
        """
        Request tokens for a process.

        Enforces both per-process and system-wide limits.
        Returns True if tokens granted.
        """
        # Check system token bucket
        if self._system_token_bucket:
            if not await self._system_token_bucket.consume_async(amount, timeout / 2):
                logger.debug(f"System token limit hit for process {process_id}")
                return False

        # Check process token bucket
        process_bucket = self._token_buckets.get(process_id)
        if process_bucket:
            if not await process_bucket.consume_async(amount, timeout / 2):
                logger.debug(f"Process token limit hit for {process_id}")
                return False

        # Update total tokens quota
        self._quotas["tokens_total"].consume(amount)

        # Update process usage
        if process_id in self._process_usage:
            self._process_usage[process_id].tokens_used += amount

        return True

    def record_usage(
        self,
        process_id: str,
        memory_mb: Optional[float] = None,
        cpu_percent: Optional[float] = None,
        tokens: Optional[int] = None,
    ) -> None:
        """Record resource usage for a process."""
        usage = self._process_usage.get(process_id)
        if not usage:
            return

        old_memory = usage.memory_mb

        if memory_mb is not None:
            usage.memory_mb = memory_mb
            # Update memory quota
            self._quotas["memory"].release(old_memory)
            self._quotas["memory"].consume(memory_mb)

        if cpu_percent is not None:
            usage.cpu_percent = cpu_percent

        if tokens is not None:
            usage.tokens_used += tokens
            self._quotas["tokens_total"].consume(tokens)

        usage.last_activity = datetime.now()

    def get_usage(self, process_id: str) -> Optional[ResourceUsage]:
        """Get resource usage for a process."""
        return self._process_usage.get(process_id)

    def check_limits(
        self,
        process_id: str,
        limits: ResourceLimits,
    ) -> tuple[bool, Optional[str]]:
        """
        Check if a process exceeds its limits.

        Returns (exceeded, reason).
        """
        usage = self._process_usage.get(process_id)
        if not usage:
            return False, None

        return usage.exceeds_limits(limits)

    def get_pressure(self) -> ResourcePressure:
        """Get current resource pressure level."""
        return self._pressure

    def get_quota(self, name: str) -> Optional[ResourceQuota]:
        """Get a resource quota by name."""
        return self._quotas.get(name)

    def get_all_quotas(self) -> Dict[str, ResourceQuota]:
        """Get all resource quotas."""
        return self._quotas.copy()

    def get_token_bucket(self, process_id: str) -> Optional[TokenBucket]:
        """Get token bucket for a process."""
        return self._token_buckets.get(process_id)

    def on_pressure_change(self, callback: Callable) -> None:
        """Register a callback for pressure changes."""
        self._pressure_callbacks.append(callback)

    async def _pressure_monitor_loop(self) -> None:
        """Monitor system pressure levels."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(self.pressure_check_interval)

                if self._shutdown_event.is_set():
                    break

                new_pressure = await self._calculate_pressure()

                if new_pressure != self._pressure:
                    old_pressure = self._pressure
                    self._pressure = new_pressure

                    logger.info(
                        "Resource pressure changed",
                        old=old_pressure.value,
                        new=new_pressure.value,
                    )

                    # Notify callbacks
                    for callback in self._pressure_callbacks:
                        try:
                            result = callback(old_pressure, new_pressure)
                            if asyncio.iscoroutine(result):
                                await result
                        except Exception as e:
                            logger.warning(f"Pressure callback failed: {e}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Pressure monitor error: {e}")

    async def _calculate_pressure(self) -> ResourcePressure:
        """Calculate current resource pressure."""
        try:
            import psutil

            # Get system metrics
            memory = psutil.virtual_memory()
            cpu = psutil.cpu_percent(interval=0.1)

            memory_percent = memory.percent
            cpu_percent = cpu

            # Check quotas
            tokens_quota = self._quotas.get("tokens_total")
            tokens_pressure = 0
            if tokens_quota and tokens_quota.limit > 0:
                tokens_pressure = tokens_quota.usage_percent

            # Calculate overall pressure
            max_pressure = max(memory_percent, cpu_percent, tokens_pressure)

            if max_pressure >= 95:
                return ResourcePressure.CRITICAL
            elif max_pressure >= 85:
                return ResourcePressure.HIGH
            elif max_pressure >= 70:
                return ResourcePressure.MEDIUM
            elif max_pressure >= 50:
                return ResourcePressure.LOW
            else:
                return ResourcePressure.NONE

        except ImportError:
            return ResourcePressure.NONE
        except Exception as e:
            logger.warning(f"Failed to calculate pressure: {e}")
            return ResourcePressure.NONE

    def get_stats(self) -> Dict[str, Any]:
        """Get resource manager statistics."""
        quotas = {
            name: {
                "current": q.current,
                "limit": q.limit,
                "usage_percent": q.usage_percent,
                "is_soft_exceeded": q.is_soft_exceeded,
                "is_hard_exceeded": q.is_hard_exceeded,
            }
            for name, q in self._quotas.items()
        }

        return {
            "pressure": self._pressure.value,
            "quotas": quotas,
            "tracked_processes": len(self._process_usage),
            "system_token_bucket_available": (
                self._system_token_bucket.get_available()
                if self._system_token_bucket else None
            ),
        }

    # === Context Manager ===

    async def __aenter__(self) -> "ResourceManager":
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.shutdown()


class ResourceThrottler:
    """
    Adaptive resource throttler.

    Automatically adjusts processing rate based on resource pressure.
    """

    def __init__(
        self,
        resource_manager: ResourceManager,
        base_rate: float = 1.0,
        min_rate: float = 0.1,
        pressure_multipliers: Optional[Dict[ResourcePressure, float]] = None,
    ):
        self.resource_manager = resource_manager
        self.base_rate = base_rate
        self.min_rate = min_rate
        self.pressure_multipliers = pressure_multipliers or {
            ResourcePressure.NONE: 1.0,
            ResourcePressure.LOW: 0.8,
            ResourcePressure.MEDIUM: 0.5,
            ResourcePressure.HIGH: 0.25,
            ResourcePressure.CRITICAL: 0.1,
        }

    def get_current_rate(self) -> float:
        """Get current throttled rate."""
        pressure = self.resource_manager.get_pressure()
        multiplier = self.pressure_multipliers.get(pressure, 1.0)
        return max(self.min_rate, self.base_rate * multiplier)

    async def throttle(self) -> None:
        """Wait according to current throttle rate."""
        rate = self.get_current_rate()
        if rate < self.base_rate:
            delay = (1.0 / rate) - (1.0 / self.base_rate)
            await asyncio.sleep(delay)

    def should_skip(self, priority: ProcessPriority) -> bool:
        """Check if operation should be skipped due to pressure."""
        pressure = self.resource_manager.get_pressure()

        # Skip low priority work under high pressure
        if pressure == ResourcePressure.CRITICAL:
            return priority.value >= ProcessPriority.NORMAL.value
        elif pressure == ResourcePressure.HIGH:
            return priority.value >= ProcessPriority.LOW.value

        return False
