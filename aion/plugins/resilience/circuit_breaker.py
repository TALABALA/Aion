"""
Circuit Breaker Pattern Implementation

Prevents cascading failures by stopping calls to failing plugins
and allowing recovery time.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, TypeVar, Generic
from functools import wraps

import structlog

logger = structlog.get_logger(__name__)

T = TypeVar("T")


class CircuitState(str, Enum):
    """State of the circuit breaker."""

    CLOSED = "closed"  # Normal operation, calls pass through
    OPEN = "open"  # Circuit tripped, calls fail fast
    HALF_OPEN = "half_open"  # Testing if service has recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for a circuit breaker."""

    # Failure threshold to open circuit
    failure_threshold: int = 5
    failure_rate_threshold: float = 0.5  # 50% failure rate

    # Time window for counting failures
    failure_window_seconds: float = 60.0

    # How long to wait before trying again
    recovery_timeout: float = 30.0

    # Number of successful calls needed to close circuit
    success_threshold: int = 3

    # Call timeout (0 = no timeout)
    call_timeout: float = 30.0

    # Exceptions that should trip the circuit
    trip_exceptions: tuple = (Exception,)

    # Exceptions that should NOT trip the circuit
    ignore_exceptions: tuple = ()

    # Callbacks
    on_open: Optional[Callable[[], None]] = None
    on_close: Optional[Callable[[], None]] = None
    on_half_open: Optional[Callable[[], None]] = None


@dataclass
class CircuitMetrics:
    """Metrics for a circuit breaker."""

    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    timeouts: int = 0

    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    state_change_time: float = field(default_factory=time.time)

    # Rolling window of recent calls (timestamp, success)
    recent_calls: list[tuple[float, bool]] = field(default_factory=list)

    def record_success(self) -> None:
        """Record a successful call."""
        now = time.time()
        self.total_calls += 1
        self.successful_calls += 1
        self.last_success_time = now
        self.recent_calls.append((now, True))

    def record_failure(self) -> None:
        """Record a failed call."""
        now = time.time()
        self.total_calls += 1
        self.failed_calls += 1
        self.last_failure_time = now
        self.recent_calls.append((now, False))

    def record_rejection(self) -> None:
        """Record a rejected call (circuit open)."""
        self.rejected_calls += 1

    def record_timeout(self) -> None:
        """Record a timeout."""
        self.timeouts += 1
        self.record_failure()

    def get_failure_rate(self, window_seconds: float) -> float:
        """Get failure rate in the recent window."""
        cutoff = time.time() - window_seconds
        self.recent_calls = [(t, s) for t, s in self.recent_calls if t > cutoff]

        if not self.recent_calls:
            return 0.0

        failures = sum(1 for _, success in self.recent_calls if not success)
        return failures / len(self.recent_calls)

    def get_failure_count(self, window_seconds: float) -> int:
        """Get failure count in the recent window."""
        cutoff = time.time() - window_seconds
        return sum(1 for t, s in self.recent_calls if t > cutoff and not s)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "rejected_calls": self.rejected_calls,
            "timeouts": self.timeouts,
            "last_failure_time": self.last_failure_time,
            "last_success_time": self.last_success_time,
            "state_change_time": self.state_change_time,
        }


class CircuitOpenError(Exception):
    """Raised when circuit is open and call is rejected."""

    def __init__(self, circuit_name: str, time_until_retry: float):
        self.circuit_name = circuit_name
        self.time_until_retry = time_until_retry
        super().__init__(
            f"Circuit '{circuit_name}' is open. "
            f"Retry in {time_until_retry:.1f}s"
        )


class CircuitBreaker:
    """
    Circuit Breaker implementation.

    Monitors calls and opens the circuit when failure threshold is reached,
    preventing cascading failures.

    Usage:
        breaker = CircuitBreaker("my-service")

        async with breaker:
            result = await risky_operation()

        # Or as decorator
        @breaker
        async def my_function():
            ...
    """

    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._metrics = CircuitMetrics()
        self._half_open_successes = 0
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        return self._state

    @property
    def metrics(self) -> CircuitMetrics:
        """Get circuit metrics."""
        return self._metrics

    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self._state == CircuitState.CLOSED

    def is_open(self) -> bool:
        """Check if circuit is open (failing fast)."""
        return self._state == CircuitState.OPEN

    def is_half_open(self) -> bool:
        """Check if circuit is half-open (testing recovery)."""
        return self._state == CircuitState.HALF_OPEN

    async def __aenter__(self):
        """Enter context manager."""
        await self._before_call()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        if exc_type is None:
            await self._on_success()
        elif self._should_trip(exc_type):
            await self._on_failure(exc_val)
        return False  # Don't suppress exceptions

    def __call__(self, func: Callable) -> Callable:
        """Use as decorator."""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            async with self:
                if self.config.call_timeout > 0:
                    return await asyncio.wait_for(
                        func(*args, **kwargs),
                        timeout=self.config.call_timeout,
                    )
                return await func(*args, **kwargs)
        return wrapper

    async def call(
        self,
        func: Callable[..., T],
        *args,
        **kwargs,
    ) -> T:
        """Execute a function through the circuit breaker."""
        async with self:
            if self.config.call_timeout > 0:
                try:
                    return await asyncio.wait_for(
                        func(*args, **kwargs),
                        timeout=self.config.call_timeout,
                    )
                except asyncio.TimeoutError:
                    self._metrics.record_timeout()
                    raise
            return await func(*args, **kwargs)

    async def _before_call(self) -> None:
        """Check circuit state before allowing call."""
        async with self._lock:
            if self._state == CircuitState.OPEN:
                # Check if recovery timeout has passed
                elapsed = time.time() - self._metrics.state_change_time
                if elapsed >= self.config.recovery_timeout:
                    self._transition_to(CircuitState.HALF_OPEN)
                else:
                    self._metrics.record_rejection()
                    raise CircuitOpenError(
                        self.name,
                        self.config.recovery_timeout - elapsed,
                    )

    async def _on_success(self) -> None:
        """Handle successful call."""
        async with self._lock:
            self._metrics.record_success()

            if self._state == CircuitState.HALF_OPEN:
                self._half_open_successes += 1
                if self._half_open_successes >= self.config.success_threshold:
                    self._transition_to(CircuitState.CLOSED)

    async def _on_failure(self, error: Exception) -> None:
        """Handle failed call."""
        async with self._lock:
            self._metrics.record_failure()

            if self._state == CircuitState.HALF_OPEN:
                # Any failure in half-open reopens the circuit
                self._transition_to(CircuitState.OPEN)

            elif self._state == CircuitState.CLOSED:
                # Check if we should open the circuit
                failure_count = self._metrics.get_failure_count(
                    self.config.failure_window_seconds
                )
                failure_rate = self._metrics.get_failure_rate(
                    self.config.failure_window_seconds
                )

                should_open = (
                    failure_count >= self.config.failure_threshold or
                    failure_rate >= self.config.failure_rate_threshold
                )

                if should_open:
                    self._transition_to(CircuitState.OPEN)

    def _should_trip(self, exc_type: type) -> bool:
        """Check if exception should trip the circuit."""
        if issubclass(exc_type, self.config.ignore_exceptions):
            return False
        return issubclass(exc_type, self.config.trip_exceptions)

    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state."""
        if self._state == new_state:
            return

        old_state = self._state
        self._state = new_state
        self._metrics.state_change_time = time.time()

        logger.info(
            "Circuit breaker state change",
            name=self.name,
            old_state=old_state.value,
            new_state=new_state.value,
        )

        if new_state == CircuitState.OPEN:
            self._half_open_successes = 0
            if self.config.on_open:
                self.config.on_open()

        elif new_state == CircuitState.CLOSED:
            self._half_open_successes = 0
            if self.config.on_close:
                self.config.on_close()

        elif new_state == CircuitState.HALF_OPEN:
            self._half_open_successes = 0
            if self.config.on_half_open:
                self.config.on_half_open()

    def reset(self) -> None:
        """Manually reset the circuit to closed state."""
        self._state = CircuitState.CLOSED
        self._half_open_successes = 0
        self._metrics = CircuitMetrics()

    def get_stats(self) -> dict[str, Any]:
        """Get circuit breaker statistics."""
        return {
            "name": self.name,
            "state": self._state.value,
            "metrics": self._metrics.to_dict(),
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "failure_rate_threshold": self.config.failure_rate_threshold,
                "recovery_timeout": self.config.recovery_timeout,
                "success_threshold": self.config.success_threshold,
            },
        }


class CircuitBreakerRegistry:
    """
    Registry for managing multiple circuit breakers.

    Provides centralized management of circuit breakers across plugins.
    """

    def __init__(self):
        self._breakers: dict[str, CircuitBreaker] = {}
        self._lock = asyncio.Lock()

    async def get_or_create(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
    ) -> CircuitBreaker:
        """Get existing or create new circuit breaker."""
        async with self._lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(name, config)
            return self._breakers[name]

    def get(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name."""
        return self._breakers.get(name)

    def remove(self, name: str) -> None:
        """Remove a circuit breaker."""
        self._breakers.pop(name, None)

    def reset_all(self) -> None:
        """Reset all circuit breakers."""
        for breaker in self._breakers.values():
            breaker.reset()

    def get_stats(self) -> dict[str, Any]:
        """Get statistics for all circuit breakers."""
        return {
            "total": len(self._breakers),
            "open": sum(1 for b in self._breakers.values() if b.is_open()),
            "half_open": sum(1 for b in self._breakers.values() if b.is_half_open()),
            "closed": sum(1 for b in self._breakers.values() if b.is_closed()),
            "breakers": {
                name: breaker.get_stats()
                for name, breaker in self._breakers.items()
            },
        }
