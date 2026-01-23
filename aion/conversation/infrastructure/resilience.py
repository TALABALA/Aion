"""
AION SOTA Resilience Patterns

State-of-the-art resilience patterns featuring:
- Circuit Breaker for fault isolation
- Exponential backoff retry with jitter
- Bulkhead pattern for resource isolation
- Fallback strategies
- Health monitoring
"""

from __future__ import annotations

import asyncio
import functools
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar, Union

import structlog

logger = structlog.get_logger(__name__)

T = TypeVar("T")


# =============================================================================
# Circuit Breaker
# =============================================================================

class CircuitState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, rejecting requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5          # Failures before opening
    success_threshold: int = 3          # Successes to close from half-open
    timeout_seconds: float = 30.0       # Time to wait before half-open
    half_open_max_calls: int = 3        # Max concurrent calls in half-open
    excluded_exceptions: tuple = ()     # Exceptions that don't count as failures


@dataclass
class CircuitBreakerStats:
    """Statistics for circuit breaker."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    state_transitions: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0


class CircuitBreaker:
    """
    Circuit Breaker pattern for fault isolation.

    Prevents cascading failures by failing fast when a service is unhealthy.

    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Service unhealthy, requests are rejected immediately
    - HALF_OPEN: Testing if service recovered, limited requests allowed
    """

    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()

        self._state = CircuitState.CLOSED
        self._stats = CircuitBreakerStats()
        self._last_state_change = datetime.now()
        self._half_open_calls = 0
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        return self._state

    @property
    def stats(self) -> CircuitBreakerStats:
        """Get circuit breaker statistics."""
        return self._stats

    async def is_available(self) -> bool:
        """Check if circuit allows requests."""
        async with self._lock:
            if self._state == CircuitState.CLOSED:
                return True

            if self._state == CircuitState.OPEN:
                # Check if timeout has passed
                elapsed = (datetime.now() - self._last_state_change).total_seconds()
                if elapsed >= self.config.timeout_seconds:
                    await self._transition_to(CircuitState.HALF_OPEN)
                    return True
                return False

            if self._state == CircuitState.HALF_OPEN:
                return self._half_open_calls < self.config.half_open_max_calls

            return False

    async def record_success(self) -> None:
        """Record a successful call."""
        async with self._lock:
            self._stats.total_calls += 1
            self._stats.successful_calls += 1
            self._stats.consecutive_successes += 1
            self._stats.consecutive_failures = 0
            self._stats.last_success_time = datetime.now()

            if self._state == CircuitState.HALF_OPEN:
                self._half_open_calls = max(0, self._half_open_calls - 1)

                if self._stats.consecutive_successes >= self.config.success_threshold:
                    await self._transition_to(CircuitState.CLOSED)

    async def record_failure(self, exception: Optional[Exception] = None) -> None:
        """Record a failed call."""
        # Check if exception should be excluded
        if exception and isinstance(exception, self.config.excluded_exceptions):
            return

        async with self._lock:
            self._stats.total_calls += 1
            self._stats.failed_calls += 1
            self._stats.consecutive_failures += 1
            self._stats.consecutive_successes = 0
            self._stats.last_failure_time = datetime.now()

            if self._state == CircuitState.HALF_OPEN:
                self._half_open_calls = max(0, self._half_open_calls - 1)
                await self._transition_to(CircuitState.OPEN)

            elif self._state == CircuitState.CLOSED:
                if self._stats.consecutive_failures >= self.config.failure_threshold:
                    await self._transition_to(CircuitState.OPEN)

    async def record_rejected(self) -> None:
        """Record a rejected call (circuit open)."""
        async with self._lock:
            self._stats.total_calls += 1
            self._stats.rejected_calls += 1

    async def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state."""
        old_state = self._state
        self._state = new_state
        self._last_state_change = datetime.now()
        self._stats.state_transitions += 1

        if new_state == CircuitState.HALF_OPEN:
            self._half_open_calls = 0
            self._stats.consecutive_successes = 0

        elif new_state == CircuitState.CLOSED:
            self._stats.consecutive_failures = 0

        logger.info(
            f"Circuit breaker '{self.name}' state change",
            old_state=old_state.value,
            new_state=new_state.value,
        )

    async def execute(
        self,
        func: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """
        Execute a function with circuit breaker protection.

        Raises:
            CircuitOpenError: If circuit is open
        """
        if not await self.is_available():
            await self.record_rejected()
            raise CircuitOpenError(f"Circuit '{self.name}' is open")

        if self._state == CircuitState.HALF_OPEN:
            self._half_open_calls += 1

        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            await self.record_success()
            return result

        except Exception as e:
            await self.record_failure(e)
            raise

    def reset(self) -> None:
        """Reset the circuit breaker to initial state."""
        self._state = CircuitState.CLOSED
        self._stats = CircuitBreakerStats()
        self._last_state_change = datetime.now()
        self._half_open_calls = 0


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass


# =============================================================================
# Retry Policy
# =============================================================================

@dataclass
class RetryConfig:
    """Configuration for retry policy."""
    max_retries: int = 3
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    jitter_factor: float = 0.1
    retryable_exceptions: tuple = (Exception,)
    non_retryable_exceptions: tuple = ()


class RetryPolicy:
    """
    Retry policy with exponential backoff and jitter.

    Features:
    - Exponential backoff
    - Optional jitter to prevent thundering herd
    - Configurable retryable exceptions
    - Callback hooks for retry events
    """

    def __init__(
        self,
        config: Optional[RetryConfig] = None,
        on_retry: Optional[Callable[[int, Exception, float], None]] = None,
    ):
        self.config = config or RetryConfig()
        self.on_retry = on_retry

    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for a retry attempt."""
        delay = min(
            self.config.base_delay_seconds * (self.config.exponential_base ** attempt),
            self.config.max_delay_seconds,
        )

        if self.config.jitter:
            jitter_range = delay * self.config.jitter_factor
            delay = delay + random.uniform(-jitter_range, jitter_range)

        return max(0, delay)

    def should_retry(self, exception: Exception, attempt: int) -> bool:
        """Check if should retry for given exception and attempt."""
        if attempt >= self.config.max_retries:
            return False

        if isinstance(exception, self.config.non_retryable_exceptions):
            return False

        return isinstance(exception, self.config.retryable_exceptions)

    async def execute(
        self,
        func: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """Execute a function with retry policy."""
        last_exception: Optional[Exception] = None

        for attempt in range(self.config.max_retries + 1):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)

            except Exception as e:
                last_exception = e

                if not self.should_retry(e, attempt):
                    raise

                delay = self.calculate_delay(attempt)

                logger.warning(
                    f"Retry attempt {attempt + 1}/{self.config.max_retries}",
                    error=str(e),
                    delay=f"{delay:.2f}s",
                )

                if self.on_retry:
                    self.on_retry(attempt, e, delay)

                await asyncio.sleep(delay)

        raise last_exception or Exception("Retry failed with no exception")


# =============================================================================
# Fallback
# =============================================================================

class Fallback(Generic[T]):
    """
    Fallback pattern for graceful degradation.

    Provides alternative responses when primary operation fails.
    """

    def __init__(
        self,
        default_value: Optional[T] = None,
        fallback_func: Optional[Callable[..., T]] = None,
        cache_fallback: bool = True,
    ):
        self.default_value = default_value
        self.fallback_func = fallback_func
        self.cache_fallback = cache_fallback
        self._cached_value: Optional[T] = None
        self._last_success_value: Optional[T] = None

    async def execute(
        self,
        func: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """Execute with fallback on failure."""
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            # Cache successful value
            if self.cache_fallback:
                self._last_success_value = result

            return result

        except Exception as e:
            logger.warning(f"Primary operation failed, using fallback: {e}")
            return await self._get_fallback(*args, **kwargs)

    async def _get_fallback(self, *args: Any, **kwargs: Any) -> T:
        """Get fallback value."""
        # Try fallback function
        if self.fallback_func:
            try:
                if asyncio.iscoroutinefunction(self.fallback_func):
                    return await self.fallback_func(*args, **kwargs)
                else:
                    return self.fallback_func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Fallback function failed: {e}")

        # Try cached value
        if self._last_success_value is not None:
            return self._last_success_value

        # Use default value
        if self.default_value is not None:
            return self.default_value

        raise Exception("No fallback available")


# =============================================================================
# Bulkhead (Resource Isolation)
# =============================================================================

class Bulkhead:
    """
    Bulkhead pattern for resource isolation.

    Limits concurrent executions to prevent resource exhaustion.
    """

    def __init__(
        self,
        name: str,
        max_concurrent: int = 10,
        max_wait_seconds: float = 30.0,
    ):
        self.name = name
        self.max_concurrent = max_concurrent
        self.max_wait_seconds = max_wait_seconds

        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._active_count = 0
        self._rejected_count = 0

    @property
    def active_count(self) -> int:
        """Get number of active executions."""
        return self._active_count

    @property
    def available_permits(self) -> int:
        """Get number of available permits."""
        return self.max_concurrent - self._active_count

    async def execute(
        self,
        func: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """Execute with bulkhead protection."""
        try:
            acquired = await asyncio.wait_for(
                self._semaphore.acquire(),
                timeout=self.max_wait_seconds,
            )

            if not acquired:
                self._rejected_count += 1
                raise BulkheadFullError(f"Bulkhead '{self.name}' is full")

        except asyncio.TimeoutError:
            self._rejected_count += 1
            raise BulkheadFullError(f"Bulkhead '{self.name}' wait timeout")

        self._active_count += 1

        try:
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        finally:
            self._active_count -= 1
            self._semaphore.release()


class BulkheadFullError(Exception):
    """Raised when bulkhead is at capacity."""
    pass


# =============================================================================
# Resilient Executor
# =============================================================================

class ResilientExecutor:
    """
    Combines all resilience patterns into a single executor.

    Order of execution:
    1. Bulkhead (limit concurrency)
    2. Circuit Breaker (fail fast if unhealthy)
    3. Retry (with exponential backoff)
    4. Fallback (if all else fails)
    """

    def __init__(
        self,
        name: str,
        circuit_breaker: Optional[CircuitBreaker] = None,
        retry_policy: Optional[RetryPolicy] = None,
        fallback: Optional[Fallback] = None,
        bulkhead: Optional[Bulkhead] = None,
    ):
        self.name = name
        self.circuit_breaker = circuit_breaker
        self.retry_policy = retry_policy
        self.fallback = fallback
        self.bulkhead = bulkhead

        # Metrics
        self._total_calls = 0
        self._successful_calls = 0
        self._failed_calls = 0
        self._fallback_calls = 0

    @property
    def stats(self) -> Dict[str, Any]:
        """Get executor statistics."""
        return {
            "total_calls": self._total_calls,
            "successful_calls": self._successful_calls,
            "failed_calls": self._failed_calls,
            "fallback_calls": self._fallback_calls,
            "success_rate": self._successful_calls / max(self._total_calls, 1),
            "circuit_breaker": self.circuit_breaker.stats if self.circuit_breaker else None,
        }

    async def execute(
        self,
        func: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """Execute a function with all resilience patterns."""
        self._total_calls += 1

        async def _execute_inner() -> T:
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)

        try:
            # Apply bulkhead
            if self.bulkhead:
                _execute_inner = functools.partial(
                    self.bulkhead.execute,
                    _execute_inner,
                )

            # Apply circuit breaker
            if self.circuit_breaker:
                _execute_inner = functools.partial(
                    self.circuit_breaker.execute,
                    _execute_inner,
                )

            # Apply retry
            if self.retry_policy:
                result = await self.retry_policy.execute(_execute_inner)
            else:
                result = await _execute_inner()

            self._successful_calls += 1
            return result

        except Exception as e:
            self._failed_calls += 1

            # Try fallback
            if self.fallback:
                try:
                    self._fallback_calls += 1
                    return await self.fallback.execute(
                        lambda: None,  # Dummy that will fail
                    )
                except Exception:
                    pass

            raise

    @classmethod
    def create_default(
        cls,
        name: str,
        max_retries: int = 3,
        circuit_failure_threshold: int = 5,
        max_concurrent: int = 10,
        default_value: Any = None,
    ) -> "ResilientExecutor":
        """Create a resilient executor with sensible defaults."""
        return cls(
            name=name,
            circuit_breaker=CircuitBreaker(
                name=f"{name}_circuit",
                config=CircuitBreakerConfig(
                    failure_threshold=circuit_failure_threshold,
                ),
            ),
            retry_policy=RetryPolicy(
                config=RetryConfig(
                    max_retries=max_retries,
                ),
            ),
            fallback=Fallback(default_value=default_value) if default_value else None,
            bulkhead=Bulkhead(
                name=f"{name}_bulkhead",
                max_concurrent=max_concurrent,
            ),
        )


# =============================================================================
# Decorators
# =============================================================================

def with_retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    exceptions: tuple = (Exception,),
) -> Callable:
    """Decorator to add retry logic to a function."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        policy = RetryPolicy(
            config=RetryConfig(
                max_retries=max_retries,
                base_delay_seconds=base_delay,
                retryable_exceptions=exceptions,
            )
        )

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            return await policy.execute(func, *args, **kwargs)

        return wrapper
    return decorator


def with_circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    timeout_seconds: float = 30.0,
) -> Callable:
    """Decorator to add circuit breaker to a function."""
    breaker = CircuitBreaker(
        name=name,
        config=CircuitBreakerConfig(
            failure_threshold=failure_threshold,
            timeout_seconds=timeout_seconds,
        ),
    )

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            return await breaker.execute(func, *args, **kwargs)

        return wrapper
    return decorator


def with_fallback(
    default_value: Any = None,
    fallback_func: Optional[Callable] = None,
) -> Callable:
    """Decorator to add fallback to a function."""
    fb = Fallback(
        default_value=default_value,
        fallback_func=fallback_func,
    )

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            return await fb.execute(func, *args, **kwargs)

        return wrapper
    return decorator
