"""
AION Conversation Rate Limiting Middleware

Rate limiting for conversation endpoints.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Optional

from fastapi import Request, HTTPException
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class RateLimitConfig:
    """Rate limit configuration."""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    burst_limit: int = 10
    burst_window_seconds: int = 1


@dataclass
class RateLimitState:
    """State for rate limit tracking."""
    minute_count: int = 0
    hour_count: int = 0
    day_count: int = 0
    burst_count: int = 0

    minute_reset: datetime = field(default_factory=datetime.now)
    hour_reset: datetime = field(default_factory=datetime.now)
    day_reset: datetime = field(default_factory=datetime.now)
    burst_reset: datetime = field(default_factory=datetime.now)

    def reset_if_needed(self) -> None:
        """Reset counters if time windows have passed."""
        now = datetime.now()

        if now >= self.minute_reset:
            self.minute_count = 0
            self.minute_reset = now + timedelta(minutes=1)

        if now >= self.hour_reset:
            self.hour_count = 0
            self.hour_reset = now + timedelta(hours=1)

        if now >= self.day_reset:
            self.day_count = 0
            self.day_reset = now + timedelta(days=1)

        if now >= self.burst_reset:
            self.burst_count = 0
            self.burst_reset = now + timedelta(seconds=1)


class RateLimiter:
    """
    Rate limiter with multiple time windows.

    Features:
    - Per-minute, per-hour, per-day limits
    - Burst protection
    - Per-client tracking
    - Graceful degradation
    """

    def __init__(
        self,
        config: Optional[RateLimitConfig] = None,
        key_func: Optional[Callable[[Request], str]] = None,
    ):
        self.config = config or RateLimitConfig()
        self.key_func = key_func or self._default_key_func
        self._states: dict[str, RateLimitState] = defaultdict(RateLimitState)
        self._lock = asyncio.Lock()

    def _default_key_func(self, request: Request) -> str:
        """Get rate limit key from request."""
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()

        if request.client:
            return request.client.host

        return "unknown"

    async def check(self, request: Request) -> dict[str, Any]:
        """
        Check rate limits for a request.

        Returns:
            Dict with rate limit info

        Raises:
            HTTPException if rate limited
        """
        key = self.key_func(request)

        async with self._lock:
            state = self._states[key]
            state.reset_if_needed()

            limits_info = {
                "key": key,
                "minute": {
                    "remaining": self.config.requests_per_minute - state.minute_count,
                    "limit": self.config.requests_per_minute,
                    "reset": state.minute_reset.isoformat(),
                },
                "hour": {
                    "remaining": self.config.requests_per_hour - state.hour_count,
                    "limit": self.config.requests_per_hour,
                    "reset": state.hour_reset.isoformat(),
                },
                "day": {
                    "remaining": self.config.requests_per_day - state.day_count,
                    "limit": self.config.requests_per_day,
                    "reset": state.day_reset.isoformat(),
                },
            }

            if state.burst_count >= self.config.burst_limit:
                retry_after = (state.burst_reset - datetime.now()).total_seconds()
                raise HTTPException(
                    status_code=429,
                    detail="Burst limit exceeded",
                    headers={
                        "Retry-After": str(int(max(1, retry_after))),
                        "X-RateLimit-Limit": str(self.config.burst_limit),
                        "X-RateLimit-Remaining": "0",
                    },
                )

            if state.minute_count >= self.config.requests_per_minute:
                retry_after = (state.minute_reset - datetime.now()).total_seconds()
                raise HTTPException(
                    status_code=429,
                    detail="Rate limit exceeded (per minute)",
                    headers={
                        "Retry-After": str(int(max(1, retry_after))),
                        "X-RateLimit-Limit": str(self.config.requests_per_minute),
                        "X-RateLimit-Remaining": "0",
                    },
                )

            if state.hour_count >= self.config.requests_per_hour:
                retry_after = (state.hour_reset - datetime.now()).total_seconds()
                raise HTTPException(
                    status_code=429,
                    detail="Rate limit exceeded (per hour)",
                    headers={
                        "Retry-After": str(int(max(1, retry_after))),
                        "X-RateLimit-Limit": str(self.config.requests_per_hour),
                        "X-RateLimit-Remaining": "0",
                    },
                )

            if state.day_count >= self.config.requests_per_day:
                retry_after = (state.day_reset - datetime.now()).total_seconds()
                raise HTTPException(
                    status_code=429,
                    detail="Rate limit exceeded (per day)",
                    headers={
                        "Retry-After": str(int(max(1, retry_after))),
                        "X-RateLimit-Limit": str(self.config.requests_per_day),
                        "X-RateLimit-Remaining": "0",
                    },
                )

            state.burst_count += 1
            state.minute_count += 1
            state.hour_count += 1
            state.day_count += 1

            return limits_info

    async def reset(self, key: str) -> None:
        """Reset rate limits for a key."""
        async with self._lock:
            if key in self._states:
                del self._states[key]

    def get_stats(self) -> dict[str, Any]:
        """Get rate limiter statistics."""
        return {
            "tracked_clients": len(self._states),
            "config": {
                "requests_per_minute": self.config.requests_per_minute,
                "requests_per_hour": self.config.requests_per_hour,
                "requests_per_day": self.config.requests_per_day,
                "burst_limit": self.config.burst_limit,
            },
        }


class RateLimitMiddleware:
    """
    FastAPI middleware for rate limiting.
    """

    def __init__(
        self,
        limiter: Optional[RateLimiter] = None,
        exclude_paths: Optional[list[str]] = None,
    ):
        self.limiter = limiter or RateLimiter()
        self.exclude_paths = exclude_paths or ["/health", "/ready", "/docs", "/openapi.json"]

    async def __call__(self, request: Request) -> dict[str, Any]:
        """Check rate limits."""
        if any(request.url.path.startswith(p) for p in self.exclude_paths):
            return {"rate_limited": False}

        return await self.limiter.check(request)


def create_rate_limiter(
    requests_per_minute: int = 60,
    requests_per_hour: int = 1000,
    requests_per_day: int = 10000,
    burst_limit: int = 10,
) -> RateLimiter:
    """Create a configured rate limiter."""
    config = RateLimitConfig(
        requests_per_minute=requests_per_minute,
        requests_per_hour=requests_per_hour,
        requests_per_day=requests_per_day,
        burst_limit=burst_limit,
    )
    return RateLimiter(config=config)


class SlidingWindowRateLimiter:
    """
    Sliding window rate limiter for more accurate limiting.
    """

    def __init__(
        self,
        limit: int,
        window_seconds: int,
        key_func: Optional[Callable[[Request], str]] = None,
    ):
        self.limit = limit
        self.window = timedelta(seconds=window_seconds)
        self.key_func = key_func or (lambda r: r.client.host if r.client else "unknown")
        self._requests: dict[str, list[datetime]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def check(self, request: Request) -> bool:
        """
        Check if request is allowed.

        Returns:
            True if allowed, raises HTTPException if rate limited
        """
        key = self.key_func(request)
        now = datetime.now()
        window_start = now - self.window

        async with self._lock:
            self._requests[key] = [
                ts for ts in self._requests[key]
                if ts > window_start
            ]

            if len(self._requests[key]) >= self.limit:
                oldest = min(self._requests[key])
                retry_after = (oldest + self.window - now).total_seconds()

                raise HTTPException(
                    status_code=429,
                    detail="Rate limit exceeded",
                    headers={
                        "Retry-After": str(int(max(1, retry_after))),
                        "X-RateLimit-Limit": str(self.limit),
                        "X-RateLimit-Remaining": "0",
                    },
                )

            self._requests[key].append(now)
            return True
