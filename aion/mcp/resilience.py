"""
AION MCP Resilience Patterns

Production-grade resilience patterns for MCP integration:
- Circuit Breaker with state machine
- Exponential Backoff with jitter
- Rate Limiting (token bucket + sliding window)
- LRU Caching with TTL
- Request Deduplication
- Bulkhead pattern for isolation
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import random
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, Generic, Optional, TypeVar, Union

import structlog

logger = structlog.get_logger(__name__)

T = TypeVar("T")


# ============================================
# Circuit Breaker
# ============================================

class CircuitState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5          # Failures before opening
    success_threshold: int = 3          # Successes in half-open to close
    timeout: float = 30.0               # Seconds before half-open
    half_open_max_calls: int = 3        # Max concurrent calls in half-open
    excluded_exceptions: tuple = ()     # Exceptions that don't count as failures


@dataclass
class CircuitStats:
    """Statistics for circuit breaker."""
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    total_calls: int = 0
    total_failures: int = 0
    total_successes: int = 0
    state_changes: int = 0
    last_state_change: Optional[datetime] = None


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open."""
    def __init__(self, circuit_name: str, retry_after: float):
        self.circuit_name = circuit_name
        self.retry_after = retry_after
        super().__init__(f"Circuit '{circuit_name}' is open. Retry after {retry_after:.1f}s")


class CircuitBreaker:
    """
    Production-grade circuit breaker implementation.

    States:
    - CLOSED: Normal operation, tracking failures
    - OPEN: Too many failures, rejecting all requests
    - HALF_OPEN: Testing if service recovered

    Features:
    - Configurable thresholds
    - Automatic state transitions
    - Half-open state with limited concurrency
    - Exception filtering
    - Statistics and monitoring
    """

    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
    ):
        """
        Initialize circuit breaker.

        Args:
            name: Circuit breaker name (for logging/metrics)
            config: Configuration options
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._half_open_calls = 0

        self._lock = asyncio.Lock()
        self._stats = CircuitStats()

    @property
    def state(self) -> CircuitState:
        """Get current state (may transition from OPEN to HALF_OPEN)."""
        if self._state == CircuitState.OPEN:
            if self._should_attempt_reset():
                return CircuitState.HALF_OPEN
        return self._state

    def _should_attempt_reset(self) -> bool:
        """Check if we should transition to HALF_OPEN."""
        if self._last_failure_time is None:
            return True
        elapsed = time.monotonic() - self._last_failure_time
        return elapsed >= self.config.timeout

    async def __aenter__(self):
        """Context manager entry - check if request allowed."""
        await self._before_call()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - record result."""
        if exc_type is None:
            await self._on_success()
        elif not isinstance(exc_val, self.config.excluded_exceptions):
            await self._on_failure()
        return False

    async def _before_call(self) -> None:
        """Check if call is allowed."""
        async with self._lock:
            current_state = self.state

            if current_state == CircuitState.OPEN:
                retry_after = self.config.timeout
                if self._last_failure_time:
                    elapsed = time.monotonic() - self._last_failure_time
                    retry_after = max(0, self.config.timeout - elapsed)
                raise CircuitBreakerError(self.name, retry_after)

            if current_state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self.config.half_open_max_calls:
                    raise CircuitBreakerError(self.name, 1.0)
                self._half_open_calls += 1

                # Transition to HALF_OPEN if we were OPEN
                if self._state == CircuitState.OPEN:
                    self._transition_to(CircuitState.HALF_OPEN)

            self._stats.total_calls += 1

    async def _on_success(self) -> None:
        """Record successful call."""
        async with self._lock:
            self._stats.total_successes += 1
            self._stats.last_success_time = datetime.now()

            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                self._half_open_calls = max(0, self._half_open_calls - 1)

                if self._success_count >= self.config.success_threshold:
                    self._transition_to(CircuitState.CLOSED)
            else:
                # Reset failure count on success in CLOSED state
                self._failure_count = 0

    async def _on_failure(self) -> None:
        """Record failed call."""
        async with self._lock:
            self._failure_count += 1
            self._stats.failure_count = self._failure_count
            self._stats.total_failures += 1
            self._stats.last_failure_time = datetime.now()
            self._last_failure_time = time.monotonic()

            if self._state == CircuitState.HALF_OPEN:
                # Any failure in half-open returns to open
                self._half_open_calls = 0
                self._transition_to(CircuitState.OPEN)
            elif self._failure_count >= self.config.failure_threshold:
                self._transition_to(CircuitState.OPEN)

    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state."""
        old_state = self._state
        self._state = new_state

        self._stats.state = new_state
        self._stats.state_changes += 1
        self._stats.last_state_change = datetime.now()

        # Reset counters on state change
        if new_state == CircuitState.CLOSED:
            self._failure_count = 0
            self._success_count = 0
        elif new_state == CircuitState.HALF_OPEN:
            self._success_count = 0
            self._half_open_calls = 0
        elif new_state == CircuitState.OPEN:
            self._success_count = 0

        logger.info(
            f"Circuit breaker '{self.name}' state change",
            old_state=old_state.value,
            new_state=new_state.value,
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "total_calls": self._stats.total_calls,
            "total_failures": self._stats.total_failures,
            "total_successes": self._stats.total_successes,
            "state_changes": self._stats.state_changes,
            "last_failure": self._stats.last_failure_time.isoformat() if self._stats.last_failure_time else None,
            "last_success": self._stats.last_success_time.isoformat() if self._stats.last_success_time else None,
        }

    def reset(self) -> None:
        """Reset circuit breaker to closed state."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._half_open_calls = 0
        self._last_failure_time = None


# ============================================
# Exponential Backoff with Jitter
# ============================================

@dataclass
class BackoffConfig:
    """Configuration for exponential backoff."""
    base_delay: float = 1.0             # Initial delay in seconds
    max_delay: float = 60.0             # Maximum delay cap
    max_retries: int = 5                # Maximum retry attempts
    multiplier: float = 2.0             # Delay multiplier per attempt
    jitter: float = 0.1                 # Jitter factor (0-1)
    jitter_mode: str = "full"           # "full", "equal", "decorrelated"


class RetryExhaustedError(Exception):
    """Raised when all retry attempts are exhausted."""
    def __init__(self, attempts: int, last_error: Exception):
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(f"Retry exhausted after {attempts} attempts: {last_error}")


class ExponentialBackoff:
    """
    Production-grade exponential backoff with multiple jitter strategies.

    Jitter Modes:
    - full: delay * random(0, 1)
    - equal: delay/2 + random(0, delay/2)
    - decorrelated: random(base_delay, previous_delay * 3)

    Based on AWS Architecture Blog recommendations.
    """

    def __init__(self, config: Optional[BackoffConfig] = None):
        """
        Initialize backoff calculator.

        Args:
            config: Backoff configuration
        """
        self.config = config or BackoffConfig()
        self._previous_delay = self.config.base_delay

    def calculate_delay(self, attempt: int) -> float:
        """
        Calculate delay for given attempt number.

        Args:
            attempt: Attempt number (0-indexed)

        Returns:
            Delay in seconds
        """
        # Calculate base exponential delay
        delay = self.config.base_delay * (self.config.multiplier ** attempt)
        delay = min(delay, self.config.max_delay)

        # Apply jitter
        if self.config.jitter_mode == "full":
            # Full jitter: random between 0 and delay
            delay = random.uniform(0, delay)
        elif self.config.jitter_mode == "equal":
            # Equal jitter: half fixed + half random
            delay = delay / 2 + random.uniform(0, delay / 2)
        elif self.config.jitter_mode == "decorrelated":
            # Decorrelated jitter
            delay = random.uniform(self.config.base_delay, self._previous_delay * 3)
            delay = min(delay, self.config.max_delay)

        # Add small random jitter to all modes
        jitter_amount = delay * self.config.jitter
        delay += random.uniform(-jitter_amount, jitter_amount)

        # Store for decorrelated mode
        self._previous_delay = delay

        return max(0, delay)

    def reset(self) -> None:
        """Reset backoff state."""
        self._previous_delay = self.config.base_delay


async def retry_with_backoff(
    func: Callable,
    config: Optional[BackoffConfig] = None,
    retryable_exceptions: tuple = (Exception,),
    on_retry: Optional[Callable[[int, Exception, float], None]] = None,
) -> Any:
    """
    Execute function with exponential backoff retry.

    Args:
        func: Async function to execute
        config: Backoff configuration
        retryable_exceptions: Exceptions that trigger retry
        on_retry: Optional callback(attempt, error, delay) called before retry

    Returns:
        Function result

    Raises:
        RetryExhaustedError: If all retries fail
    """
    backoff = ExponentialBackoff(config)
    config = backoff.config
    last_error: Optional[Exception] = None

    for attempt in range(config.max_retries + 1):
        try:
            return await func()
        except retryable_exceptions as e:
            last_error = e

            if attempt >= config.max_retries:
                raise RetryExhaustedError(attempt + 1, e)

            delay = backoff.calculate_delay(attempt)

            if on_retry:
                on_retry(attempt, e, delay)

            logger.debug(
                "Retrying after error",
                attempt=attempt + 1,
                max_retries=config.max_retries,
                delay=delay,
                error=str(e),
            )

            await asyncio.sleep(delay)

    # Should not reach here
    raise RetryExhaustedError(config.max_retries + 1, last_error)


# ============================================
# Rate Limiting
# ============================================

class RateLimitExceededError(Exception):
    """Raised when rate limit is exceeded."""
    def __init__(self, limit_name: str, retry_after: float):
        self.limit_name = limit_name
        self.retry_after = retry_after
        super().__init__(f"Rate limit '{limit_name}' exceeded. Retry after {retry_after:.2f}s")


class TokenBucketRateLimiter:
    """
    Token bucket rate limiter.

    Allows bursts up to bucket capacity while maintaining
    average rate over time.
    """

    def __init__(
        self,
        name: str,
        rate: float,           # Tokens per second
        capacity: int = 10,    # Bucket capacity
    ):
        """
        Initialize rate limiter.

        Args:
            name: Limiter name
            rate: Tokens added per second
            capacity: Maximum tokens in bucket
        """
        self.name = name
        self.rate = rate
        self.capacity = capacity

        self._tokens = float(capacity)
        self._last_update = time.monotonic()
        self._lock = asyncio.Lock()

        # Statistics
        self._total_requests = 0
        self._total_allowed = 0
        self._total_rejected = 0

    async def acquire(self, tokens: int = 1, block: bool = True) -> bool:
        """
        Acquire tokens from bucket.

        Args:
            tokens: Number of tokens to acquire
            block: Whether to wait for tokens

        Returns:
            True if tokens acquired

        Raises:
            RateLimitExceededError: If block=False and no tokens available
        """
        async with self._lock:
            self._total_requests += 1

            # Refill tokens based on elapsed time
            now = time.monotonic()
            elapsed = now - self._last_update
            self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
            self._last_update = now

            if self._tokens >= tokens:
                self._tokens -= tokens
                self._total_allowed += 1
                return True

            if not block:
                self._total_rejected += 1
                # Calculate time until enough tokens
                tokens_needed = tokens - self._tokens
                retry_after = tokens_needed / self.rate
                raise RateLimitExceededError(self.name, retry_after)

            # Wait for tokens
            tokens_needed = tokens - self._tokens
            wait_time = tokens_needed / self.rate

        await asyncio.sleep(wait_time)
        return await self.acquire(tokens, block=False)

    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics."""
        return {
            "name": self.name,
            "rate": self.rate,
            "capacity": self.capacity,
            "current_tokens": self._tokens,
            "total_requests": self._total_requests,
            "total_allowed": self._total_allowed,
            "total_rejected": self._total_rejected,
        }


class SlidingWindowRateLimiter:
    """
    Sliding window rate limiter.

    More accurate than fixed window, prevents burst at window boundaries.
    """

    def __init__(
        self,
        name: str,
        limit: int,            # Maximum requests
        window_seconds: float, # Window duration
    ):
        """
        Initialize rate limiter.

        Args:
            name: Limiter name
            limit: Maximum requests per window
            window_seconds: Window duration in seconds
        """
        self.name = name
        self.limit = limit
        self.window_seconds = window_seconds

        self._requests: list[float] = []
        self._lock = asyncio.Lock()

        # Statistics
        self._total_requests = 0
        self._total_allowed = 0
        self._total_rejected = 0

    async def acquire(self, block: bool = True) -> bool:
        """
        Attempt to acquire a request slot.

        Args:
            block: Whether to wait if limit exceeded

        Returns:
            True if request allowed
        """
        async with self._lock:
            self._total_requests += 1
            now = time.monotonic()
            cutoff = now - self.window_seconds

            # Remove expired requests
            self._requests = [t for t in self._requests if t > cutoff]

            if len(self._requests) < self.limit:
                self._requests.append(now)
                self._total_allowed += 1
                return True

            if not block:
                self._total_rejected += 1
                # Estimate when oldest request expires
                oldest = min(self._requests)
                retry_after = oldest + self.window_seconds - now
                raise RateLimitExceededError(self.name, max(0, retry_after))

        # Wait for a slot
        oldest = min(self._requests)
        wait_time = oldest + self.window_seconds - time.monotonic()

        if wait_time > 0:
            await asyncio.sleep(wait_time)

        return await self.acquire(block=False)

    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics."""
        return {
            "name": self.name,
            "limit": self.limit,
            "window_seconds": self.window_seconds,
            "current_requests": len(self._requests),
            "total_requests": self._total_requests,
            "total_allowed": self._total_allowed,
            "total_rejected": self._total_rejected,
        }


# ============================================
# LRU Cache with TTL
# ============================================

@dataclass
class CacheEntry(Generic[T]):
    """Cache entry with metadata."""
    value: T
    created_at: float
    last_accessed: float
    access_count: int = 0
    ttl: Optional[float] = None

    def is_expired(self, now: float) -> bool:
        """Check if entry is expired."""
        if self.ttl is None:
            return False
        return (now - self.created_at) > self.ttl


class LRUCache(Generic[T]):
    """
    LRU cache with TTL support.

    Features:
    - Configurable max size
    - Per-entry TTL
    - Automatic expiration cleanup
    - Statistics tracking
    """

    def __init__(
        self,
        name: str,
        max_size: int = 1000,
        default_ttl: Optional[float] = None,
        cleanup_interval: float = 60.0,
    ):
        """
        Initialize cache.

        Args:
            name: Cache name
            max_size: Maximum entries
            default_ttl: Default TTL in seconds
            cleanup_interval: Interval for cleanup task
        """
        self.name = name
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cleanup_interval = cleanup_interval

        self._cache: OrderedDict[str, CacheEntry[T]] = OrderedDict()
        self._lock = asyncio.Lock()

        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0

        # Cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start background cleanup task."""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def stop(self) -> None:
        """Stop background cleanup task."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None

    async def _cleanup_loop(self) -> None:
        """Background cleanup of expired entries."""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self._cleanup_expired()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")

    async def _cleanup_expired(self) -> None:
        """Remove expired entries."""
        async with self._lock:
            now = time.monotonic()
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired(now)
            ]

            for key in expired_keys:
                del self._cache[key]
                self._evictions += 1

    async def get(self, key: str) -> Optional[T]:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None
        """
        async with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                self._misses += 1
                return None

            now = time.monotonic()
            if entry.is_expired(now):
                del self._cache[key]
                self._misses += 1
                self._evictions += 1
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.last_accessed = now
            entry.access_count += 1

            self._hits += 1
            return entry.value

    async def set(
        self,
        key: str,
        value: T,
        ttl: Optional[float] = None,
    ) -> None:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Optional TTL (uses default if not specified)
        """
        async with self._lock:
            now = time.monotonic()

            # Evict if at capacity
            while len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)
                self._evictions += 1

            self._cache[key] = CacheEntry(
                value=value,
                created_at=now,
                last_accessed=now,
                ttl=ttl if ttl is not None else self.default_ttl,
            )

    async def delete(self, key: str) -> bool:
        """
        Delete entry from cache.

        Args:
            key: Cache key

        Returns:
            True if entry was deleted
        """
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    async def clear(self) -> None:
        """Clear all cache entries."""
        async with self._lock:
            self._cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0

        return {
            "name": self.name,
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "evictions": self._evictions,
        }


# ============================================
# Request Deduplication
# ============================================

class RequestDeduplicator:
    """
    Deduplicates concurrent identical requests.

    If multiple identical requests come in simultaneously,
    only one is executed and all get the same result.
    """

    def __init__(self, name: str, ttl: float = 5.0):
        """
        Initialize deduplicator.

        Args:
            name: Deduplicator name
            ttl: How long to cache results
        """
        self.name = name
        self.ttl = ttl

        self._pending: Dict[str, asyncio.Future] = {}
        self._results: Dict[str, tuple[Any, float]] = {}
        self._lock = asyncio.Lock()

        # Statistics
        self._total_requests = 0
        self._deduplicated = 0
        self._cached = 0

    def _make_key(self, *args, **kwargs) -> str:
        """Create cache key from arguments."""
        key_data = str((args, sorted(kwargs.items())))
        return hashlib.sha256(key_data.encode()).hexdigest()[:16]

    async def execute(
        self,
        func: Callable,
        *args,
        **kwargs,
    ) -> Any:
        """
        Execute function with deduplication.

        Args:
            func: Async function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result
        """
        key = self._make_key(*args, **kwargs)

        async with self._lock:
            self._total_requests += 1
            now = time.monotonic()

            # Check cached results
            if key in self._results:
                result, cached_at = self._results[key]
                if now - cached_at < self.ttl:
                    self._cached += 1
                    return result
                else:
                    del self._results[key]

            # Check pending requests
            if key in self._pending:
                self._deduplicated += 1
                future = self._pending[key]

            else:
                # Create new future for this request
                future = asyncio.get_event_loop().create_future()
                self._pending[key] = future

        # If we created the future, execute the function
        if key in self._pending and self._pending[key] is future:
            try:
                result = await func(*args, **kwargs)

                async with self._lock:
                    self._results[key] = (result, time.monotonic())
                    del self._pending[key]

                future.set_result(result)
                return result

            except Exception as e:
                async with self._lock:
                    if key in self._pending:
                        del self._pending[key]

                future.set_exception(e)
                raise

        # Wait for pending request
        return await future

    def get_stats(self) -> Dict[str, Any]:
        """Get deduplication statistics."""
        return {
            "name": self.name,
            "pending_requests": len(self._pending),
            "cached_results": len(self._results),
            "total_requests": self._total_requests,
            "deduplicated": self._deduplicated,
            "cached": self._cached,
        }


# ============================================
# Bulkhead Pattern
# ============================================

class BulkheadFullError(Exception):
    """Raised when bulkhead is at capacity."""
    def __init__(self, bulkhead_name: str, current: int, max_concurrent: int):
        self.bulkhead_name = bulkhead_name
        self.current = current
        self.max_concurrent = max_concurrent
        super().__init__(
            f"Bulkhead '{bulkhead_name}' at capacity: {current}/{max_concurrent}"
        )


class Bulkhead:
    """
    Bulkhead pattern for isolation.

    Limits concurrent executions to prevent resource exhaustion
    and isolate failures.
    """

    def __init__(
        self,
        name: str,
        max_concurrent: int = 10,
        max_wait: float = 0.0,
    ):
        """
        Initialize bulkhead.

        Args:
            name: Bulkhead name
            max_concurrent: Maximum concurrent executions
            max_wait: Maximum time to wait for a slot (0 = no waiting)
        """
        self.name = name
        self.max_concurrent = max_concurrent
        self.max_wait = max_wait

        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._current = 0
        self._lock = asyncio.Lock()

        # Statistics
        self._total_requests = 0
        self._total_rejected = 0
        self._peak_concurrent = 0

    async def __aenter__(self):
        """Enter bulkhead."""
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit bulkhead."""
        await self.release()
        return False

    async def acquire(self) -> None:
        """Acquire a slot in the bulkhead."""
        async with self._lock:
            self._total_requests += 1

            if self.max_wait <= 0:
                if self._current >= self.max_concurrent:
                    self._total_rejected += 1
                    raise BulkheadFullError(self.name, self._current, self.max_concurrent)

        try:
            await asyncio.wait_for(
                self._semaphore.acquire(),
                timeout=self.max_wait if self.max_wait > 0 else None,
            )
        except asyncio.TimeoutError:
            async with self._lock:
                self._total_rejected += 1
            raise BulkheadFullError(self.name, self._current, self.max_concurrent)

        async with self._lock:
            self._current += 1
            self._peak_concurrent = max(self._peak_concurrent, self._current)

    async def release(self) -> None:
        """Release a slot in the bulkhead."""
        self._semaphore.release()
        async with self._lock:
            self._current = max(0, self._current - 1)

    def get_stats(self) -> Dict[str, Any]:
        """Get bulkhead statistics."""
        return {
            "name": self.name,
            "max_concurrent": self.max_concurrent,
            "current": self._current,
            "peak_concurrent": self._peak_concurrent,
            "total_requests": self._total_requests,
            "total_rejected": self._total_rejected,
        }


# ============================================
# Combined Resilience Policy
# ============================================

@dataclass
class ResilienceConfig:
    """Combined resilience configuration."""
    circuit_breaker: Optional[CircuitBreakerConfig] = None
    backoff: Optional[BackoffConfig] = None
    rate_limit_rate: Optional[float] = None
    rate_limit_capacity: Optional[int] = None
    cache_max_size: Optional[int] = None
    cache_ttl: Optional[float] = None
    bulkhead_max_concurrent: Optional[int] = None


class ResiliencePolicy:
    """
    Combined resilience policy applying multiple patterns.

    Order of application:
    1. Rate limiting
    2. Bulkhead
    3. Circuit breaker
    4. Retry with backoff
    5. Cache
    """

    def __init__(
        self,
        name: str,
        config: Optional[ResilienceConfig] = None,
    ):
        """
        Initialize resilience policy.

        Args:
            name: Policy name
            config: Configuration options
        """
        self.name = name
        self.config = config or ResilienceConfig()

        # Initialize components based on config
        self._circuit_breaker: Optional[CircuitBreaker] = None
        self._rate_limiter: Optional[TokenBucketRateLimiter] = None
        self._bulkhead: Optional[Bulkhead] = None
        self._cache: Optional[LRUCache] = None
        self._backoff_config: Optional[BackoffConfig] = None

        if self.config.circuit_breaker:
            self._circuit_breaker = CircuitBreaker(f"{name}_cb", self.config.circuit_breaker)

        if self.config.rate_limit_rate:
            self._rate_limiter = TokenBucketRateLimiter(
                f"{name}_rl",
                rate=self.config.rate_limit_rate,
                capacity=self.config.rate_limit_capacity or 10,
            )

        if self.config.bulkhead_max_concurrent:
            self._bulkhead = Bulkhead(f"{name}_bh", self.config.bulkhead_max_concurrent)

        if self.config.cache_max_size:
            self._cache = LRUCache(
                f"{name}_cache",
                max_size=self.config.cache_max_size,
                default_ttl=self.config.cache_ttl,
            )

        if self.config.backoff:
            self._backoff_config = self.config.backoff

    async def execute(
        self,
        func: Callable,
        cache_key: Optional[str] = None,
    ) -> Any:
        """
        Execute function with full resilience policy.

        Args:
            func: Async function to execute
            cache_key: Optional cache key (skips cache if None)

        Returns:
            Function result
        """
        # Check cache first
        if self._cache and cache_key:
            cached = await self._cache.get(cache_key)
            if cached is not None:
                return cached

        # Rate limiting
        if self._rate_limiter:
            await self._rate_limiter.acquire()

        # Bulkhead
        if self._bulkhead:
            await self._bulkhead.acquire()

        try:
            result = await self._execute_with_resilience(func)

            # Cache result
            if self._cache and cache_key:
                await self._cache.set(cache_key, result)

            return result

        finally:
            if self._bulkhead:
                await self._bulkhead.release()

    async def _execute_with_resilience(self, func: Callable) -> Any:
        """Execute with circuit breaker and retry."""

        async def execute_with_cb():
            if self._circuit_breaker:
                async with self._circuit_breaker:
                    return await func()
            else:
                return await func()

        if self._backoff_config:
            return await retry_with_backoff(
                execute_with_cb,
                config=self._backoff_config,
            )
        else:
            return await execute_with_cb()

    def get_stats(self) -> Dict[str, Any]:
        """Get combined statistics."""
        stats = {"name": self.name}

        if self._circuit_breaker:
            stats["circuit_breaker"] = self._circuit_breaker.get_stats()
        if self._rate_limiter:
            stats["rate_limiter"] = self._rate_limiter.get_stats()
        if self._bulkhead:
            stats["bulkhead"] = self._bulkhead.get_stats()
        if self._cache:
            stats["cache"] = self._cache.get_stats()

        return stats
