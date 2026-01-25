"""
AION Rate Limiter

Enterprise-grade rate limiting with multiple strategies.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import structlog

from aion.security.types import RateLimitConfig, RateLimitResult, RateLimitState

logger = structlog.get_logger(__name__)


class RateLimitStrategy(str, Enum):
    """Rate limiting strategies."""

    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"
    LEAKY_BUCKET = "leaky_bucket"


class RateLimiter:
    """
    Enterprise-grade rate limiter.

    Features:
    - Multiple strategies (fixed window, sliding window, token bucket)
    - Per-user, per-tenant, per-IP limiting
    - Configurable limits per endpoint/resource
    - Burst protection
    - Penalty system for abuse
    - Real-time statistics
    """

    def __init__(
        self,
        default_config: Optional[RateLimitConfig] = None,
        strategy: RateLimitStrategy = RateLimitStrategy.TOKEN_BUCKET,
    ):
        self.default_config = default_config or RateLimitConfig()
        self.strategy = strategy

        # State storage by key
        self._state: Dict[str, RateLimitState] = defaultdict(
            lambda: RateLimitState(tokens=self.default_config.bucket_size)
        )

        # Custom configs per key pattern
        self._configs: Dict[str, RateLimitConfig] = {}

        # Request history for sliding window
        self._request_history: Dict[str, List[float]] = defaultdict(list)

        # Penalty tracking
        self._penalties: Dict[str, datetime] = {}
        self._violation_counts: Dict[str, int] = defaultdict(int)

        # Statistics
        self._stats = {
            "total_requests": 0,
            "allowed_requests": 0,
            "denied_requests": 0,
            "by_limit_type": defaultdict(int),
        }

        # Cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize rate limiter."""
        if self._initialized:
            return

        logger.info("Initializing Rate Limiter", strategy=self.strategy.value)

        # Start cleanup loop
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

        self._initialized = True

    async def shutdown(self) -> None:
        """Shutdown rate limiter."""
        self._shutdown_event.set()

        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        self._initialized = False

    def set_config(self, key_pattern: str, config: RateLimitConfig) -> None:
        """Set rate limit config for a key pattern."""
        self._configs[key_pattern] = config

    def _get_config(self, key: str) -> RateLimitConfig:
        """Get config for a key (with pattern matching)."""
        for pattern, config in self._configs.items():
            if key.startswith(pattern) or pattern == "*":
                return config
        return self.default_config

    # =========================================================================
    # Main Rate Limiting
    # =========================================================================

    async def check(
        self,
        key: str,
        cost: int = 1,
    ) -> RateLimitResult:
        """
        Check if request is allowed.

        Args:
            key: Rate limit key (e.g., "user:123", "tenant:456", "ip:1.2.3.4")
            cost: Cost of the request (default 1)

        Returns:
            RateLimitResult with allow/deny decision and limit info
        """
        self._stats["total_requests"] += 1

        config = self._get_config(key)
        state = self._state[key]
        now = datetime.now()

        # Check for penalty
        if key in self._penalties:
            if now < self._penalties[key]:
                self._stats["denied_requests"] += 1
                self._stats["by_limit_type"]["penalty"] += 1
                return RateLimitResult(
                    allowed=False,
                    limit_type="penalty",
                    current=0,
                    limit=0,
                    remaining=0,
                    retry_after_seconds=(self._penalties[key] - now).total_seconds(),
                )
            else:
                del self._penalties[key]

        # Apply rate limiting based on strategy
        if self.strategy == RateLimitStrategy.TOKEN_BUCKET:
            result = self._check_token_bucket(key, config, state, cost, now)
        elif self.strategy == RateLimitStrategy.SLIDING_WINDOW:
            result = self._check_sliding_window(key, config, cost, now)
        elif self.strategy == RateLimitStrategy.FIXED_WINDOW:
            result = self._check_fixed_window(key, config, state, cost, now)
        else:
            result = self._check_token_bucket(key, config, state, cost, now)

        # Track statistics
        if result.allowed:
            self._stats["allowed_requests"] += 1
        else:
            self._stats["denied_requests"] += 1
            self._stats["by_limit_type"][result.limit_type or "unknown"] += 1

            # Track violations for penalty
            self._violation_counts[key] += 1
            if self._violation_counts[key] >= 10:
                self._apply_penalty(key, minutes=5)

        return result

    async def consume(self, key: str, cost: int = 1) -> bool:
        """Consume rate limit quota (convenience method)."""
        result = await self.check(key, cost)
        return result.allowed

    # =========================================================================
    # Strategy Implementations
    # =========================================================================

    def _check_token_bucket(
        self,
        key: str,
        config: RateLimitConfig,
        state: RateLimitState,
        cost: int,
        now: datetime,
    ) -> RateLimitResult:
        """Token bucket rate limiting."""
        # Refill tokens
        time_passed = (now - state.last_refill).total_seconds()
        tokens_to_add = time_passed * config.refill_rate
        state.tokens = min(state.tokens + tokens_to_add, config.bucket_size)
        state.last_refill = now

        # Check if we have enough tokens
        if state.tokens >= cost:
            state.tokens -= cost
            return RateLimitResult(
                allowed=True,
                limit_type="token_bucket",
                current=int(config.bucket_size - state.tokens),
                limit=config.bucket_size,
                remaining=int(state.tokens),
            )
        else:
            # Calculate retry after
            tokens_needed = cost - state.tokens
            retry_after = tokens_needed / config.refill_rate

            return RateLimitResult(
                allowed=False,
                limit_type="token_bucket",
                current=int(config.bucket_size - state.tokens),
                limit=config.bucket_size,
                remaining=int(state.tokens),
                retry_after_seconds=retry_after,
            )

    def _check_sliding_window(
        self,
        key: str,
        config: RateLimitConfig,
        cost: int,
        now: datetime,
    ) -> RateLimitResult:
        """Sliding window rate limiting."""
        window_start = time.time() - config.window_size_seconds
        history = self._request_history[key]

        # Remove old entries
        history[:] = [ts for ts in history if ts > window_start]

        current_count = len(history)

        if current_count + cost <= config.window_limit:
            # Add new request(s)
            for _ in range(cost):
                history.append(time.time())

            return RateLimitResult(
                allowed=True,
                limit_type="sliding_window",
                current=current_count + cost,
                limit=config.window_limit,
                remaining=config.window_limit - current_count - cost,
            )
        else:
            # Calculate when oldest request will expire
            if history:
                oldest = min(history)
                retry_after = oldest + config.window_size_seconds - time.time()
            else:
                retry_after = config.window_size_seconds

            return RateLimitResult(
                allowed=False,
                limit_type="sliding_window",
                current=current_count,
                limit=config.window_limit,
                remaining=0,
                retry_after_seconds=max(0, retry_after),
            )

    def _check_fixed_window(
        self,
        key: str,
        config: RateLimitConfig,
        state: RateLimitState,
        cost: int,
        now: datetime,
    ) -> RateLimitResult:
        """Fixed window rate limiting."""
        # Reset counters if window has passed
        self._reset_fixed_windows(state, now)

        # Check each window
        checks = [
            ("second", state.second_count, config.requests_per_second, state.second_reset)
            if config.requests_per_second else None,
            ("minute", state.minute_count, config.requests_per_minute, state.minute_reset),
            ("hour", state.hour_count, config.requests_per_hour, state.hour_reset),
            ("day", state.day_count, config.requests_per_day, state.day_reset),
        ]

        for check in checks:
            if check is None:
                continue

            window, count, limit, reset = check

            if count + cost > limit:
                return RateLimitResult(
                    allowed=False,
                    limit_type=f"fixed_window_{window}",
                    current=count,
                    limit=limit,
                    remaining=0,
                    reset_at=reset,
                    retry_after_seconds=(reset - now).total_seconds(),
                )

        # All windows allow - update counters
        if config.requests_per_second:
            state.second_count += cost
        state.minute_count += cost
        state.hour_count += cost
        state.day_count += cost

        # Return info for most restrictive window
        return RateLimitResult(
            allowed=True,
            limit_type="fixed_window_minute",
            current=state.minute_count,
            limit=config.requests_per_minute,
            remaining=config.requests_per_minute - state.minute_count,
            reset_at=state.minute_reset,
        )

    def _reset_fixed_windows(self, state: RateLimitState, now: datetime) -> None:
        """Reset fixed window counters if needed."""
        # Second reset
        if state.second_reset and now >= state.second_reset:
            state.second_count = 0
            state.second_reset = now + timedelta(seconds=1)

        # Minute reset
        if now >= state.minute_reset:
            state.minute_count = 0
            state.minute_reset = now + timedelta(minutes=1)

        # Hour reset
        if now >= state.hour_reset:
            state.hour_count = 0
            state.hour_reset = now + timedelta(hours=1)

        # Day reset
        if now >= state.day_reset:
            state.day_count = 0
            state.day_reset = now + timedelta(days=1)

    # =========================================================================
    # Concurrency Limiting
    # =========================================================================

    async def acquire_concurrent(self, key: str) -> Tuple[bool, int]:
        """
        Acquire a concurrent slot.

        Returns (acquired, current_count).
        """
        config = self._get_config(key)
        state = self._state[key]

        if state.current_concurrent >= config.max_concurrent:
            return False, state.current_concurrent

        state.current_concurrent += 1
        return True, state.current_concurrent

    async def release_concurrent(self, key: str) -> int:
        """
        Release a concurrent slot.

        Returns new count.
        """
        state = self._state[key]
        state.current_concurrent = max(0, state.current_concurrent - 1)
        return state.current_concurrent

    # =========================================================================
    # Penalty System
    # =========================================================================

    def _apply_penalty(self, key: str, minutes: int = 5) -> None:
        """Apply a temporary penalty to a key."""
        self._penalties[key] = datetime.now() + timedelta(minutes=minutes)
        self._violation_counts[key] = 0

        logger.warning(
            "Rate limit penalty applied",
            key=key,
            duration_minutes=minutes,
        )

    def clear_penalty(self, key: str) -> bool:
        """Clear a penalty for a key."""
        if key in self._penalties:
            del self._penalties[key]
            self._violation_counts[key] = 0
            return True
        return False

    # =========================================================================
    # Status and Monitoring
    # =========================================================================

    def get_status(self, key: str) -> Dict[str, Any]:
        """Get detailed rate limit status for a key."""
        config = self._get_config(key)
        state = self._state[key]
        now = datetime.now()

        self._reset_fixed_windows(state, now)

        status = {
            "key": key,
            "strategy": self.strategy.value,
            "token_bucket": {
                "tokens": state.tokens,
                "bucket_size": config.bucket_size,
                "refill_rate": config.refill_rate,
            },
            "fixed_windows": {
                "minute": {
                    "used": state.minute_count,
                    "limit": config.requests_per_minute,
                    "remaining": config.requests_per_minute - state.minute_count,
                    "reset": state.minute_reset.isoformat(),
                },
                "hour": {
                    "used": state.hour_count,
                    "limit": config.requests_per_hour,
                    "remaining": config.requests_per_hour - state.hour_count,
                    "reset": state.hour_reset.isoformat(),
                },
                "day": {
                    "used": state.day_count,
                    "limit": config.requests_per_day,
                    "remaining": config.requests_per_day - state.day_count,
                    "reset": state.day_reset.isoformat(),
                },
            },
            "concurrent": {
                "current": state.current_concurrent,
                "limit": config.max_concurrent,
            },
            "penalty": {
                "active": key in self._penalties,
                "until": self._penalties.get(key, "").isoformat() if key in self._penalties else None,
                "violations": self._violation_counts.get(key, 0),
            },
        }

        return status

    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics."""
        return {
            **self._stats,
            "active_keys": len(self._state),
            "active_penalties": len(self._penalties),
            "strategy": self.strategy.value,
        }

    # =========================================================================
    # Cleanup
    # =========================================================================

    async def _cleanup_loop(self) -> None:
        """Periodically clean up old state."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(60)  # Every minute
                self._cleanup_state()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Rate limit cleanup error: {e}")

    def _cleanup_state(self) -> None:
        """Clean up old rate limit state."""
        now = datetime.now()
        stale_threshold = timedelta(hours=1)

        # Clean up state for keys with no recent activity
        stale_keys = []
        for key, state in self._state.items():
            if now - state.last_refill > stale_threshold:
                if state.minute_count == 0 and state.tokens >= self.default_config.bucket_size:
                    stale_keys.append(key)

        for key in stale_keys:
            del self._state[key]

        # Clean up old request history
        window_start = time.time() - 3600  # 1 hour
        for key in list(self._request_history.keys()):
            history = self._request_history[key]
            history[:] = [ts for ts in history if ts > window_start]
            if not history:
                del self._request_history[key]

        # Clean up expired penalties
        expired_penalties = [k for k, v in self._penalties.items() if now > v]
        for key in expired_penalties:
            del self._penalties[key]

        if stale_keys or expired_penalties:
            logger.debug(
                "Rate limit cleanup",
                stale_keys=len(stale_keys),
                expired_penalties=len(expired_penalties),
            )
