"""
AION Real-time Usage Tracker

State-of-the-art usage tracking with:
- Redis-based atomic counters for real-time metrics
- Sliding window rate limiting
- Hierarchical time-based keys
- Automatic TTL management
- Lua scripts for atomic operations
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
import json

import structlog

from aion.usage.models import (
    UsageMetric,
    UsagePeriod,
    SubscriptionTier,
    TierLimits,
    get_tier_limits,
)

logger = structlog.get_logger(__name__)


# =============================================================================
# Abstract Base Tracker
# =============================================================================

class UsageTracker(ABC):
    """
    Abstract base class for usage tracking.

    Implements the Strategy pattern for different storage backends.
    """

    @abstractmethod
    async def increment(
        self,
        user_id: str,
        metric: UsageMetric,
        amount: float = 1.0,
        dimensions: Optional[Dict[str, str]] = None,
    ) -> float:
        """
        Atomically increment a usage counter.

        Args:
            user_id: User identifier
            metric: The metric to increment
            amount: Amount to add (supports decimals for storage)
            dimensions: Additional dimensions (expert_type, etc.)

        Returns:
            New total value after increment
        """
        pass

    @abstractmethod
    async def get_current(
        self,
        user_id: str,
        metric: UsageMetric,
        period: UsagePeriod = UsagePeriod.MONTHLY,
    ) -> float:
        """
        Get current usage for a metric in a time period.

        Args:
            user_id: User identifier
            metric: The metric to retrieve
            period: Time period (hourly, daily, monthly)

        Returns:
            Current usage value
        """
        pass

    @abstractmethod
    async def get_all_metrics(
        self,
        user_id: str,
        period: UsagePeriod = UsagePeriod.MONTHLY,
    ) -> Dict[UsageMetric, float]:
        """
        Get all metrics for a user in a time period.

        Returns:
            Dictionary of metric -> value
        """
        pass

    @abstractmethod
    async def check_limit(
        self,
        user_id: str,
        metric: UsageMetric,
        limit: Optional[float],
        period: UsagePeriod = UsagePeriod.MONTHLY,
    ) -> Tuple[bool, float, float]:
        """
        Check if usage is within limits.

        Returns:
            Tuple of (within_limit, current_value, percentage)
        """
        pass

    @abstractmethod
    async def reset_metric(
        self,
        user_id: str,
        metric: UsageMetric,
        period: UsagePeriod = UsagePeriod.MONTHLY,
    ) -> None:
        """Reset a metric counter."""
        pass

    @abstractmethod
    async def get_breakdown(
        self,
        user_id: str,
        metric: UsageMetric,
        dimension: str,
        period: UsagePeriod = UsagePeriod.MONTHLY,
    ) -> Dict[str, float]:
        """
        Get usage breakdown by dimension.

        Args:
            dimension: The dimension to break down by (e.g., "expert_type")

        Returns:
            Dictionary of dimension_value -> usage
        """
        pass


# =============================================================================
# Redis Usage Tracker
# =============================================================================

# Lua script for atomic increment with limit checking
INCR_WITH_LIMIT_SCRIPT = """
local key = KEYS[1]
local amount = tonumber(ARGV[1])
local limit = tonumber(ARGV[2])
local ttl = tonumber(ARGV[3])

local current = redis.call('GET', key)
current = current and tonumber(current) or 0

local new_value = current + amount

-- Check limit if provided
if limit and limit > 0 and new_value > limit then
    return {0, current, limit}  -- Over limit, don't increment
end

-- Increment
redis.call('INCRBYFLOAT', key, amount)

-- Set TTL if key is new
if current == 0 and ttl > 0 then
    redis.call('EXPIRE', key, ttl)
end

return {1, new_value, limit or -1}
"""

# Lua script for sliding window rate limiting
SLIDING_WINDOW_SCRIPT = """
local key = KEYS[1]
local window_start = tonumber(ARGV[1])
local window_end = tonumber(ARGV[2])
local limit = tonumber(ARGV[3])
local timestamp = tonumber(ARGV[4])
local ttl = tonumber(ARGV[5])

-- Remove old entries
redis.call('ZREMRANGEBYSCORE', key, 0, window_start)

-- Count current entries
local count = redis.call('ZCOUNT', key, window_start, window_end)

if limit and limit > 0 and count >= limit then
    return {0, count}  -- Over limit
end

-- Add new entry
redis.call('ZADD', key, timestamp, timestamp)
redis.call('EXPIRE', key, ttl)

return {1, count + 1}
"""


class RedisUsageTracker(UsageTracker):
    """
    Redis-based usage tracker for production deployments.

    Features:
    - Atomic operations with Lua scripts
    - Hierarchical time-based keys: usage:{user}:{metric}:{period}
    - Automatic TTL based on period
    - Sliding window rate limiting
    - Multi-dimensional tracking with sorted sets
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        prefix: str = "aion:usage:",
        extra_ttl_days: int = 7,  # Extra retention after period ends
    ):
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.prefix = prefix
        self.extra_ttl_days = extra_ttl_days

        self._redis = None
        self._connected = False
        self._incr_script = None
        self._sliding_script = None

    async def connect(self) -> None:
        """Initialize Redis connection."""
        if self._connected:
            return

        try:
            import redis.asyncio as redis

            self._redis = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                decode_responses=True,
            )

            # Test connection
            await self._redis.ping()

            # Register Lua scripts
            self._incr_script = self._redis.register_script(INCR_WITH_LIMIT_SCRIPT)
            self._sliding_script = self._redis.register_script(SLIDING_WINDOW_SCRIPT)

            self._connected = True
            logger.info(f"Redis usage tracker connected to {self.host}:{self.port}")

        except ImportError:
            logger.error("redis package not installed")
            raise
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    async def disconnect(self) -> None:
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()
            self._connected = False

    def _get_period_key(self, period: UsagePeriod) -> str:
        """Get the time component for a period key."""
        now = datetime.utcnow()

        if period == UsagePeriod.HOURLY:
            return now.strftime("%Y-%m-%d-%H")
        elif period == UsagePeriod.DAILY:
            return now.strftime("%Y-%m-%d")
        elif period == UsagePeriod.WEEKLY:
            # ISO week
            return now.strftime("%Y-W%W")
        elif period == UsagePeriod.MONTHLY:
            return now.strftime("%Y-%m")
        elif period == UsagePeriod.YEARLY:
            return now.strftime("%Y")
        else:
            return "all"

    def _get_ttl_seconds(self, period: UsagePeriod) -> int:
        """Calculate TTL for a period with extra retention."""
        extra = self.extra_ttl_days * 86400

        if period == UsagePeriod.HOURLY:
            return 3600 + extra
        elif period == UsagePeriod.DAILY:
            return 86400 + extra
        elif period == UsagePeriod.WEEKLY:
            return 7 * 86400 + extra
        elif period == UsagePeriod.MONTHLY:
            return 31 * 86400 + extra
        elif period == UsagePeriod.YEARLY:
            return 366 * 86400 + extra
        else:
            return 0  # No expiry

    def _build_key(
        self,
        user_id: str,
        metric: UsageMetric,
        period: UsagePeriod,
        dimension_suffix: str = "",
    ) -> str:
        """Build a Redis key for usage tracking."""
        period_key = self._get_period_key(period)
        base_key = f"{self.prefix}{user_id}:{metric.value}:{period_key}"

        if dimension_suffix:
            return f"{base_key}:{dimension_suffix}"
        return base_key

    async def increment(
        self,
        user_id: str,
        metric: UsageMetric,
        amount: float = 1.0,
        dimensions: Optional[Dict[str, str]] = None,
        limit: Optional[float] = None,
        period: UsagePeriod = UsagePeriod.MONTHLY,
    ) -> float:
        """Atomically increment usage counter."""
        if not self._connected:
            await self.connect()

        key = self._build_key(user_id, metric, period)
        ttl = self._get_ttl_seconds(period)

        # Use Lua script for atomic increment with optional limit check
        result = await self._incr_script(
            keys=[key],
            args=[amount, limit or 0, ttl],
        )

        success, new_value, _ = result

        # Track dimensional breakdown if provided
        if dimensions:
            for dim_name, dim_value in dimensions.items():
                dim_key = f"{key}:by_{dim_name}"
                await self._redis.hincrbyfloat(dim_key, dim_value, amount)
                await self._redis.expire(dim_key, ttl)

        # Also update daily counters for trend analysis
        if period != UsagePeriod.DAILY:
            daily_key = self._build_key(user_id, metric, UsagePeriod.DAILY)
            daily_ttl = self._get_ttl_seconds(UsagePeriod.DAILY)
            await self._redis.incrbyfloat(daily_key, amount)
            await self._redis.expire(daily_key, daily_ttl)

        return float(new_value)

    async def get_current(
        self,
        user_id: str,
        metric: UsageMetric,
        period: UsagePeriod = UsagePeriod.MONTHLY,
    ) -> float:
        """Get current usage value."""
        if not self._connected:
            await self.connect()

        key = self._build_key(user_id, metric, period)
        value = await self._redis.get(key)

        return float(value) if value else 0.0

    async def get_all_metrics(
        self,
        user_id: str,
        period: UsagePeriod = UsagePeriod.MONTHLY,
    ) -> Dict[UsageMetric, float]:
        """Get all tracked metrics for a user."""
        if not self._connected:
            await self.connect()

        result = {}
        period_key = self._get_period_key(period)
        pattern = f"{self.prefix}{user_id}:*:{period_key}"

        async for key in self._redis.scan_iter(match=pattern):
            # Parse metric from key
            parts = key.split(":")
            if len(parts) >= 4:
                metric_str = parts[3]  # aion:usage:{user}:{metric}:{period}
                try:
                    metric = UsageMetric(metric_str)
                    value = await self._redis.get(key)
                    result[metric] = float(value) if value else 0.0
                except ValueError:
                    continue

        return result

    async def check_limit(
        self,
        user_id: str,
        metric: UsageMetric,
        limit: Optional[float],
        period: UsagePeriod = UsagePeriod.MONTHLY,
    ) -> Tuple[bool, float, float]:
        """
        Check if usage is within limits.

        Returns:
            (within_limit, current_value, percentage)
        """
        current = await self.get_current(user_id, metric, period)

        if limit is None or limit <= 0:
            return True, current, 0.0

        percentage = (current / limit) * 100
        within_limit = current < limit

        return within_limit, current, percentage

    async def reset_metric(
        self,
        user_id: str,
        metric: UsageMetric,
        period: UsagePeriod = UsagePeriod.MONTHLY,
    ) -> None:
        """Reset a usage counter."""
        if not self._connected:
            await self.connect()

        key = self._build_key(user_id, metric, period)
        await self._redis.delete(key)

        # Also delete dimensional breakdowns
        pattern = f"{key}:by_*"
        async for dim_key in self._redis.scan_iter(match=pattern):
            await self._redis.delete(dim_key)

    async def get_breakdown(
        self,
        user_id: str,
        metric: UsageMetric,
        dimension: str,
        period: UsagePeriod = UsagePeriod.MONTHLY,
    ) -> Dict[str, float]:
        """Get usage breakdown by dimension."""
        if not self._connected:
            await self.connect()

        key = self._build_key(user_id, metric, period)
        dim_key = f"{key}:by_{dimension}"

        breakdown = await self._redis.hgetall(dim_key)
        return {k: float(v) for k, v in breakdown.items()}

    async def get_history(
        self,
        user_id: str,
        metric: UsageMetric,
        days: int = 30,
    ) -> List[Tuple[str, float]]:
        """
        Get daily usage history for trend analysis.

        Returns:
            List of (date_string, value) tuples
        """
        if not self._connected:
            await self.connect()

        history = []
        base_date = datetime.utcnow()

        for i in range(days):
            date = base_date - timedelta(days=i)
            date_str = date.strftime("%Y-%m-%d")
            key = f"{self.prefix}{user_id}:{metric.value}:{date_str}"

            value = await self._redis.get(key)
            history.append((date_str, float(value) if value else 0.0))

        return list(reversed(history))

    async def sliding_window_check(
        self,
        user_id: str,
        metric: UsageMetric,
        limit: int,
        window_seconds: int,
    ) -> Tuple[bool, int]:
        """
        Sliding window rate limit check.

        Returns:
            (allowed, current_count)
        """
        if not self._connected:
            await self.connect()

        now = time.time()
        window_start = now - window_seconds
        key = f"{self.prefix}{user_id}:{metric.value}:sliding"
        ttl = window_seconds * 2  # Keep data for 2x window

        result = await self._sliding_script(
            keys=[key],
            args=[window_start, now, limit, now, ttl],
        )

        allowed, count = result
        return bool(allowed), int(count)

    async def batch_increment(
        self,
        user_id: str,
        metrics: Dict[UsageMetric, float],
        period: UsagePeriod = UsagePeriod.MONTHLY,
    ) -> Dict[UsageMetric, float]:
        """
        Batch increment multiple metrics atomically.

        More efficient than individual increments.
        """
        if not self._connected:
            await self.connect()

        results = {}
        ttl = self._get_ttl_seconds(period)

        pipe = self._redis.pipeline()
        keys = []

        for metric, amount in metrics.items():
            key = self._build_key(user_id, metric, period)
            keys.append((key, metric))
            pipe.incrbyfloat(key, amount)
            pipe.expire(key, ttl)

        responses = await pipe.execute()

        # Parse results (every 2 responses: incr result, expire result)
        for i, (key, metric) in enumerate(keys):
            results[metric] = float(responses[i * 2])

        return results

    # === Context Manager ===

    async def __aenter__(self) -> "RedisUsageTracker":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.disconnect()


# =============================================================================
# Memory Usage Tracker (for testing/development)
# =============================================================================

class MemoryUsageTracker(UsageTracker):
    """
    In-memory usage tracker for development and testing.
    """

    def __init__(self):
        self._counters: Dict[str, float] = defaultdict(float)
        self._breakdowns: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._lock = asyncio.Lock()

    def _build_key(
        self,
        user_id: str,
        metric: UsageMetric,
        period: UsagePeriod,
    ) -> str:
        """Build a storage key."""
        now = datetime.utcnow()
        if period == UsagePeriod.MONTHLY:
            period_key = now.strftime("%Y-%m")
        elif period == UsagePeriod.DAILY:
            period_key = now.strftime("%Y-%m-%d")
        else:
            period_key = "all"

        return f"{user_id}:{metric.value}:{period_key}"

    async def increment(
        self,
        user_id: str,
        metric: UsageMetric,
        amount: float = 1.0,
        dimensions: Optional[Dict[str, str]] = None,
        limit: Optional[float] = None,
        period: UsagePeriod = UsagePeriod.MONTHLY,
    ) -> float:
        """Increment counter."""
        async with self._lock:
            key = self._build_key(user_id, metric, period)

            if limit and self._counters[key] + amount > limit:
                return self._counters[key]

            self._counters[key] += amount

            # Track dimensions
            if dimensions:
                for dim_name, dim_value in dimensions.items():
                    dim_key = f"{key}:by_{dim_name}"
                    self._breakdowns[dim_key][dim_value] += amount

            return self._counters[key]

    async def get_current(
        self,
        user_id: str,
        metric: UsageMetric,
        period: UsagePeriod = UsagePeriod.MONTHLY,
    ) -> float:
        """Get current value."""
        key = self._build_key(user_id, metric, period)
        return self._counters.get(key, 0.0)

    async def get_all_metrics(
        self,
        user_id: str,
        period: UsagePeriod = UsagePeriod.MONTHLY,
    ) -> Dict[UsageMetric, float]:
        """Get all metrics."""
        result = {}
        now = datetime.utcnow()
        if period == UsagePeriod.MONTHLY:
            period_key = now.strftime("%Y-%m")
        elif period == UsagePeriod.DAILY:
            period_key = now.strftime("%Y-%m-%d")
        else:
            period_key = "all"

        prefix = f"{user_id}:"

        for key, value in self._counters.items():
            if key.startswith(prefix) and period_key in key:
                parts = key.split(":")
                if len(parts) >= 2:
                    try:
                        metric = UsageMetric(parts[1])
                        result[metric] = value
                    except ValueError:
                        continue

        return result

    async def check_limit(
        self,
        user_id: str,
        metric: UsageMetric,
        limit: Optional[float],
        period: UsagePeriod = UsagePeriod.MONTHLY,
    ) -> Tuple[bool, float, float]:
        """Check limit."""
        current = await self.get_current(user_id, metric, period)

        if limit is None or limit <= 0:
            return True, current, 0.0

        percentage = (current / limit) * 100
        return current < limit, current, percentage

    async def reset_metric(
        self,
        user_id: str,
        metric: UsageMetric,
        period: UsagePeriod = UsagePeriod.MONTHLY,
    ) -> None:
        """Reset counter."""
        async with self._lock:
            key = self._build_key(user_id, metric, period)
            self._counters[key] = 0.0

            # Clear breakdowns
            dim_prefix = f"{key}:by_"
            to_delete = [k for k in self._breakdowns if k.startswith(dim_prefix)]
            for k in to_delete:
                del self._breakdowns[k]

    async def get_breakdown(
        self,
        user_id: str,
        metric: UsageMetric,
        dimension: str,
        period: UsagePeriod = UsagePeriod.MONTHLY,
    ) -> Dict[str, float]:
        """Get breakdown."""
        key = self._build_key(user_id, metric, period)
        dim_key = f"{key}:by_{dimension}"
        return dict(self._breakdowns.get(dim_key, {}))

    async def clear_all(self) -> None:
        """Clear all counters (for testing)."""
        async with self._lock:
            self._counters.clear()
            self._breakdowns.clear()


# =============================================================================
# Factory
# =============================================================================

def create_usage_tracker(
    backend: str = "memory",
    **kwargs: Any,
) -> UsageTracker:
    """
    Create a usage tracker instance.

    Args:
        backend: "memory" or "redis"
        **kwargs: Backend-specific configuration

    Returns:
        UsageTracker instance
    """
    if backend == "memory":
        return MemoryUsageTracker()
    elif backend == "redis":
        return RedisUsageTracker(**kwargs)
    else:
        raise ValueError(f"Unknown tracker backend: {backend}")
