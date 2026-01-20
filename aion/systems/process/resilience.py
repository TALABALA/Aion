"""
AION Resilience Patterns

Enterprise-grade fault tolerance with:
- Circuit Breaker (prevent cascade failures)
- Bulkhead Isolation (limit concurrent operations)
- Retry with backoff
- Timeout handling
- Fallback mechanisms
- Saga pattern (distributed transactions with compensation)
- Rate limiting
"""

from __future__ import annotations

import asyncio
import random
import time
import uuid
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from functools import wraps
from typing import (
    Any, Callable, Dict, Generic, List, Optional, Set, Tuple, TypeVar, Union
)

import structlog

logger = structlog.get_logger(__name__)

T = TypeVar("T")
F = TypeVar("F", bound=Callable)


# === Circuit Breaker ===

class CircuitState(Enum):
    """State of a circuit breaker."""
    CLOSED = auto()     # Normal operation
    OPEN = auto()       # Failing, reject calls
    HALF_OPEN = auto()  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5
    success_threshold: int = 2
    timeout_seconds: float = 30.0
    half_open_max_calls: int = 3
    excluded_exceptions: Set[type] = field(default_factory=set)
    fallback: Optional[Callable] = None


@dataclass
class CircuitBreakerMetrics:
    """Metrics for circuit breaker."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    state_changes: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "rejected_calls": self.rejected_calls,
            "state_changes": self.state_changes,
            "consecutive_failures": self.consecutive_failures,
            "consecutive_successes": self.consecutive_successes,
        }


class CircuitBreaker:
    """
    Circuit Breaker pattern implementation.

    Prevents cascade failures by detecting failures and stopping
    calls to a failing service.

    States:
    - CLOSED: Normal operation, calls pass through
    - OPEN: Failures detected, calls rejected immediately
    - HALF_OPEN: Testing if service recovered
    """

    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()

        self._state = CircuitState.CLOSED
        self._metrics = CircuitBreakerMetrics()
        self._half_open_calls = 0
        self._opened_at: Optional[datetime] = None
        self._lock = asyncio.Lock()

        # Callbacks
        self._on_state_change: List[Callable[[CircuitState, CircuitState], Any]] = []
        self._on_failure: List[Callable[[Exception], Any]] = []

    @property
    def state(self) -> CircuitState:
        return self._state

    @property
    def is_closed(self) -> bool:
        return self._state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        return self._state == CircuitState.OPEN

    def on_state_change(self, callback: Callable[[CircuitState, CircuitState], Any]) -> None:
        """Register state change callback."""
        self._on_state_change.append(callback)

    def on_failure(self, callback: Callable[[Exception], Any]) -> None:
        """Register failure callback."""
        self._on_failure.append(callback)

    async def _set_state(self, new_state: CircuitState) -> None:
        """Set circuit state."""
        if new_state == self._state:
            return

        old_state = self._state
        self._state = new_state
        self._metrics.state_changes += 1

        if new_state == CircuitState.OPEN:
            self._opened_at = datetime.now()
            self._half_open_calls = 0
        elif new_state == CircuitState.CLOSED:
            self._metrics.consecutive_failures = 0

        logger.info(
            f"Circuit breaker {self.name} state changed",
            old_state=old_state.name,
            new_state=new_state.name,
        )

        # Notify callbacks
        for callback in self._on_state_change:
            try:
                result = callback(old_state, new_state)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"State change callback error: {e}")

    async def _check_state(self) -> bool:
        """Check if call should be allowed."""
        async with self._lock:
            if self._state == CircuitState.CLOSED:
                return True

            if self._state == CircuitState.OPEN:
                # Check if timeout has passed
                if self._opened_at:
                    elapsed = (datetime.now() - self._opened_at).total_seconds()
                    if elapsed >= self.config.timeout_seconds:
                        await self._set_state(CircuitState.HALF_OPEN)
                        self._half_open_calls = 0
                        return True
                return False

            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls < self.config.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False

        return False

    async def _record_success(self) -> None:
        """Record a successful call."""
        async with self._lock:
            self._metrics.total_calls += 1
            self._metrics.successful_calls += 1
            self._metrics.consecutive_successes += 1
            self._metrics.consecutive_failures = 0
            self._metrics.last_success_time = datetime.now()

            if self._state == CircuitState.HALF_OPEN:
                if self._metrics.consecutive_successes >= self.config.success_threshold:
                    await self._set_state(CircuitState.CLOSED)

    async def _record_failure(self, error: Exception) -> None:
        """Record a failed call."""
        async with self._lock:
            self._metrics.total_calls += 1
            self._metrics.failed_calls += 1
            self._metrics.consecutive_failures += 1
            self._metrics.consecutive_successes = 0
            self._metrics.last_failure_time = datetime.now()

            # Notify failure callbacks
            for callback in self._on_failure:
                try:
                    result = callback(error)
                    if asyncio.iscoroutine(result):
                        await result
                except Exception as e:
                    logger.error(f"Failure callback error: {e}")

            if self._state == CircuitState.CLOSED:
                if self._metrics.consecutive_failures >= self.config.failure_threshold:
                    await self._set_state(CircuitState.OPEN)
            elif self._state == CircuitState.HALF_OPEN:
                await self._set_state(CircuitState.OPEN)

    async def execute(
        self,
        func: Callable[..., T],
        *args,
        **kwargs,
    ) -> T:
        """Execute a function through the circuit breaker."""
        # Check if call is allowed
        if not await self._check_state():
            self._metrics.rejected_calls += 1

            if self.config.fallback:
                return self.config.fallback(*args, **kwargs)

            raise CircuitOpenError(f"Circuit breaker {self.name} is open")

        try:
            # Execute
            result = func(*args, **kwargs)
            if asyncio.iscoroutine(result):
                result = await result

            await self._record_success()
            return result

        except Exception as e:
            # Check if exception should be excluded
            if type(e) in self.config.excluded_exceptions:
                await self._record_success()
                raise

            await self._record_failure(e)
            raise

    def __call__(self, func: F) -> F:
        """Decorator to wrap a function with circuit breaker."""
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await self.execute(func, *args, **kwargs)
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                return asyncio.get_event_loop().run_until_complete(
                    self.execute(func, *args, **kwargs)
                )
            return sync_wrapper

    def get_metrics(self) -> CircuitBreakerMetrics:
        """Get circuit breaker metrics."""
        return self._metrics

    def reset(self) -> None:
        """Reset the circuit breaker."""
        self._state = CircuitState.CLOSED
        self._metrics = CircuitBreakerMetrics()
        self._half_open_calls = 0
        self._opened_at = None


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass


# === Bulkhead Isolation ===

@dataclass
class BulkheadConfig:
    """Configuration for bulkhead."""
    max_concurrent: int = 10
    max_wait_time: float = 0.0  # 0 = no waiting
    queue_size: int = 0  # 0 = no queue


class Bulkhead:
    """
    Bulkhead pattern implementation.

    Isolates components by limiting concurrent operations,
    preventing resource exhaustion.
    """

    def __init__(
        self,
        name: str,
        config: Optional[BulkheadConfig] = None,
    ):
        self.name = name
        self.config = config or BulkheadConfig()

        self._semaphore = asyncio.Semaphore(self.config.max_concurrent)
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=self.config.queue_size or 0)
        self._active_count = 0
        self._rejected_count = 0
        self._queued_count = 0
        self._lock = asyncio.Lock()

    @property
    def active_count(self) -> int:
        return self._active_count

    @property
    def available_permits(self) -> int:
        return self.config.max_concurrent - self._active_count

    async def acquire(self) -> bool:
        """Acquire a permit."""
        async with self._lock:
            if self._active_count < self.config.max_concurrent:
                self._active_count += 1
                return True

            if self.config.max_wait_time > 0:
                try:
                    await asyncio.wait_for(
                        self._semaphore.acquire(),
                        timeout=self.config.max_wait_time,
                    )
                    self._active_count += 1
                    self._queued_count += 1
                    return True
                except asyncio.TimeoutError:
                    pass

            self._rejected_count += 1
            return False

    def release(self) -> None:
        """Release a permit."""
        self._active_count = max(0, self._active_count - 1)
        self._semaphore.release()

    async def execute(
        self,
        func: Callable[..., T],
        *args,
        **kwargs,
    ) -> T:
        """Execute a function within the bulkhead."""
        if not await self.acquire():
            raise BulkheadFullError(f"Bulkhead {self.name} is full")

        try:
            result = func(*args, **kwargs)
            if asyncio.iscoroutine(result):
                result = await result
            return result
        finally:
            self.release()

    def __call__(self, func: F) -> F:
        """Decorator to wrap a function with bulkhead."""
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await self.execute(func, *args, **kwargs)
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                return asyncio.get_event_loop().run_until_complete(
                    self.execute(func, *args, **kwargs)
                )
            return sync_wrapper

    def get_stats(self) -> Dict[str, Any]:
        """Get bulkhead statistics."""
        return {
            "name": self.name,
            "max_concurrent": self.config.max_concurrent,
            "active_count": self._active_count,
            "available_permits": self.available_permits,
            "rejected_count": self._rejected_count,
            "queued_count": self._queued_count,
        }


class BulkheadFullError(Exception):
    """Raised when bulkhead is full."""
    pass


# === Retry with Backoff ===

class BackoffStrategy(Enum):
    """Backoff strategies."""
    FIXED = auto()
    LINEAR = auto()
    EXPONENTIAL = auto()
    EXPONENTIAL_JITTER = auto()


@dataclass
class RetryConfig:
    """Configuration for retry."""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_strategy: BackoffStrategy = BackoffStrategy.EXPONENTIAL_JITTER
    backoff_multiplier: float = 2.0
    retryable_exceptions: Set[type] = field(default_factory=lambda: {Exception})
    non_retryable_exceptions: Set[type] = field(default_factory=set)


class Retry:
    """
    Retry pattern with configurable backoff.

    Automatically retries failed operations with
    intelligent backoff strategies.
    """

    def __init__(
        self,
        name: str = "default",
        config: Optional[RetryConfig] = None,
    ):
        self.name = name
        self.config = config or RetryConfig()

        self._total_retries = 0
        self._successful_retries = 0
        self._exhausted_retries = 0

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for a retry attempt."""
        strategy = self.config.backoff_strategy
        base = self.config.base_delay
        multiplier = self.config.backoff_multiplier
        max_delay = self.config.max_delay

        if strategy == BackoffStrategy.FIXED:
            delay = base
        elif strategy == BackoffStrategy.LINEAR:
            delay = base * attempt
        elif strategy == BackoffStrategy.EXPONENTIAL:
            delay = base * (multiplier ** (attempt - 1))
        elif strategy == BackoffStrategy.EXPONENTIAL_JITTER:
            exp_delay = base * (multiplier ** (attempt - 1))
            delay = exp_delay * (0.5 + random.random())
        else:
            delay = base

        return min(delay, max_delay)

    def _should_retry(self, exception: Exception) -> bool:
        """Check if exception should trigger retry."""
        exc_type = type(exception)

        # Check non-retryable first
        if exc_type in self.config.non_retryable_exceptions:
            return False

        # Check retryable
        for retryable in self.config.retryable_exceptions:
            if isinstance(exception, retryable):
                return True

        return False

    async def execute(
        self,
        func: Callable[..., T],
        *args,
        **kwargs,
    ) -> T:
        """Execute a function with retry."""
        last_exception: Optional[Exception] = None

        for attempt in range(1, self.config.max_retries + 2):
            try:
                result = func(*args, **kwargs)
                if asyncio.iscoroutine(result):
                    result = await result

                if attempt > 1:
                    self._successful_retries += 1
                    logger.info(f"Retry succeeded on attempt {attempt}")

                return result

            except Exception as e:
                last_exception = e

                if not self._should_retry(e):
                    raise

                if attempt > self.config.max_retries:
                    self._exhausted_retries += 1
                    logger.warning(f"Retries exhausted after {attempt} attempts")
                    raise

                self._total_retries += 1
                delay = self._calculate_delay(attempt)

                logger.warning(
                    f"Retry attempt {attempt}/{self.config.max_retries}",
                    error=str(e),
                    delay=delay,
                )

                await asyncio.sleep(delay)

        raise last_exception or Exception("Retry failed")

    def __call__(self, func: F) -> F:
        """Decorator to wrap a function with retry."""
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await self.execute(func, *args, **kwargs)
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                return asyncio.get_event_loop().run_until_complete(
                    self.execute(func, *args, **kwargs)
                )
            return sync_wrapper

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_retries": self._total_retries,
            "successful_retries": self._successful_retries,
            "exhausted_retries": self._exhausted_retries,
        }


# === Timeout ===

class Timeout:
    """
    Timeout wrapper for operations.

    Ensures operations complete within a time limit.
    """

    def __init__(
        self,
        timeout_seconds: float,
        fallback: Optional[Callable] = None,
    ):
        self.timeout_seconds = timeout_seconds
        self.fallback = fallback
        self._timeout_count = 0

    async def execute(
        self,
        func: Callable[..., T],
        *args,
        **kwargs,
    ) -> T:
        """Execute a function with timeout."""
        try:
            result = func(*args, **kwargs)
            if asyncio.iscoroutine(result):
                result = await asyncio.wait_for(result, timeout=self.timeout_seconds)
            return result
        except asyncio.TimeoutError:
            self._timeout_count += 1

            if self.fallback:
                return self.fallback(*args, **kwargs)

            raise TimeoutError(f"Operation timed out after {self.timeout_seconds}s")

    def __call__(self, func: F) -> F:
        """Decorator to wrap a function with timeout."""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await self.execute(func, *args, **kwargs)
        return wrapper


# === Saga Pattern ===

class SagaStepStatus(Enum):
    """Status of a saga step."""
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    COMPENSATING = auto()
    COMPENSATED = auto()


@dataclass
class SagaStep:
    """A step in a saga."""
    name: str
    action: Callable[..., Any]
    compensation: Callable[..., Any]
    status: SagaStepStatus = SagaStepStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    async def execute(self, context: Dict[str, Any]) -> Any:
        """Execute the action."""
        self.status = SagaStepStatus.RUNNING
        self.started_at = datetime.now()

        try:
            result = self.action(context)
            if asyncio.iscoroutine(result):
                result = await result

            self.result = result
            self.status = SagaStepStatus.COMPLETED
            self.completed_at = datetime.now()
            return result

        except Exception as e:
            self.error = str(e)
            self.status = SagaStepStatus.FAILED
            self.completed_at = datetime.now()
            raise

    async def compensate(self, context: Dict[str, Any]) -> None:
        """Execute the compensation."""
        if self.status != SagaStepStatus.COMPLETED:
            return

        self.status = SagaStepStatus.COMPENSATING

        try:
            result = self.compensation(context)
            if asyncio.iscoroutine(result):
                await result

            self.status = SagaStepStatus.COMPENSATED

        except Exception as e:
            logger.error(f"Compensation failed for {self.name}: {e}")
            # Compensation failure is logged but saga continues


@dataclass
class SagaDefinition:
    """Definition of a saga (distributed transaction)."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    steps: List[SagaStep] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    status: str = "pending"  # pending, running, completed, failed, compensating, compensated
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    error: Optional[str] = None

    def add_step(
        self,
        name: str,
        action: Callable,
        compensation: Callable,
    ) -> "SagaDefinition":
        """Add a step to the saga (builder pattern)."""
        self.steps.append(SagaStep(
            name=name,
            action=action,
            compensation=compensation,
        ))
        return self


class SagaOrchestrator:
    """
    Saga pattern orchestrator.

    Coordinates distributed transactions with compensation
    for maintaining consistency across services.

    Example:
        saga = SagaDefinition(name="order_saga")
        saga.add_step("create_order", create_order, cancel_order)
        saga.add_step("reserve_inventory", reserve, release)
        saga.add_step("process_payment", charge, refund)

        result = await orchestrator.execute(saga, {"order_id": "123"})
    """

    def __init__(self):
        self._active_sagas: Dict[str, SagaDefinition] = {}
        self._completed_sagas: deque[SagaDefinition] = deque(maxlen=1000)

        self._stats = {
            "sagas_started": 0,
            "sagas_completed": 0,
            "sagas_compensated": 0,
            "sagas_failed": 0,
            "steps_executed": 0,
            "compensations_executed": 0,
        }

    async def execute(
        self,
        saga: SagaDefinition,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute a saga.

        Returns the final context with all step results.
        """
        saga.context = context or {}
        saga.status = "running"
        self._active_sagas[saga.id] = saga
        self._stats["sagas_started"] += 1

        completed_steps: List[SagaStep] = []

        try:
            for step in saga.steps:
                logger.info(f"Executing saga step: {step.name}")

                result = await step.execute(saga.context)
                self._stats["steps_executed"] += 1

                # Store result in context for subsequent steps
                saga.context[f"{step.name}_result"] = result
                completed_steps.append(step)

            saga.status = "completed"
            saga.completed_at = datetime.now()
            self._stats["sagas_completed"] += 1

            logger.info(f"Saga completed: {saga.name}")
            return saga.context

        except Exception as e:
            saga.error = str(e)
            saga.status = "compensating"

            logger.error(f"Saga failed, starting compensation: {saga.name}", error=str(e))

            # Compensate in reverse order
            for step in reversed(completed_steps):
                try:
                    await step.compensate(saga.context)
                    self._stats["compensations_executed"] += 1
                except Exception as comp_error:
                    logger.error(f"Compensation error: {comp_error}")

            saga.status = "compensated"
            saga.completed_at = datetime.now()
            self._stats["sagas_compensated"] += 1

            raise SagaFailedError(f"Saga {saga.name} failed: {e}") from e

        finally:
            self._active_sagas.pop(saga.id, None)
            self._completed_sagas.append(saga)

    def get_saga(self, saga_id: str) -> Optional[SagaDefinition]:
        """Get saga by ID."""
        return self._active_sagas.get(saga_id)

    def get_active_sagas(self) -> List[SagaDefinition]:
        """Get all active sagas."""
        return list(self._active_sagas.values())

    def get_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics."""
        return {
            **self._stats,
            "active_sagas": len(self._active_sagas),
        }


class SagaFailedError(Exception):
    """Raised when a saga fails."""
    pass


# === Combined Resilience Decorator ===

def resilient(
    circuit_breaker: Optional[CircuitBreaker] = None,
    bulkhead: Optional[Bulkhead] = None,
    retry: Optional[Retry] = None,
    timeout: Optional[Timeout] = None,
) -> Callable[[F], F]:
    """
    Combined resilience decorator.

    Applies multiple resilience patterns in the correct order:
    1. Bulkhead (limit concurrency)
    2. Circuit Breaker (prevent cascade failures)
    3. Timeout (limit duration)
    4. Retry (handle transient failures)
    """
    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            async def execute():
                result = func(*args, **kwargs)
                if asyncio.iscoroutine(result):
                    return await result
                return result

            operation = execute

            # Apply in reverse order (innermost first)
            if retry:
                original = operation
                async def with_retry():
                    return await retry.execute(original)
                operation = with_retry

            if timeout:
                original = operation
                async def with_timeout():
                    return await timeout.execute(original)
                operation = with_timeout

            if circuit_breaker:
                original = operation
                async def with_circuit():
                    return await circuit_breaker.execute(original)
                operation = with_circuit

            if bulkhead:
                original = operation
                async def with_bulkhead():
                    return await bulkhead.execute(original)
                operation = with_bulkhead

            return await operation()

        return wrapper

    return decorator


# === Resilience Manager ===

class ResilienceManager:
    """
    Central manager for resilience patterns.

    Provides a registry of circuit breakers, bulkheads, etc.
    with easy access and monitoring.
    """

    def __init__(self):
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._bulkheads: Dict[str, Bulkhead] = {}
        self._retries: Dict[str, Retry] = {}
        self._saga_orchestrator = SagaOrchestrator()

    def circuit_breaker(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
    ) -> CircuitBreaker:
        """Get or create a circuit breaker."""
        if name not in self._circuit_breakers:
            self._circuit_breakers[name] = CircuitBreaker(name, config)
        return self._circuit_breakers[name]

    def bulkhead(
        self,
        name: str,
        config: Optional[BulkheadConfig] = None,
    ) -> Bulkhead:
        """Get or create a bulkhead."""
        if name not in self._bulkheads:
            self._bulkheads[name] = Bulkhead(name, config)
        return self._bulkheads[name]

    def retry(
        self,
        name: str,
        config: Optional[RetryConfig] = None,
    ) -> Retry:
        """Get or create a retry policy."""
        if name not in self._retries:
            self._retries[name] = Retry(name, config)
        return self._retries[name]

    @property
    def sagas(self) -> SagaOrchestrator:
        """Get the saga orchestrator."""
        return self._saga_orchestrator

    def get_all_stats(self) -> Dict[str, Any]:
        """Get stats for all resilience components."""
        return {
            "circuit_breakers": {
                name: {
                    "state": cb.state.name,
                    **cb.get_metrics().to_dict(),
                }
                for name, cb in self._circuit_breakers.items()
            },
            "bulkheads": {
                name: bh.get_stats()
                for name, bh in self._bulkheads.items()
            },
            "retries": {
                name: r.get_stats()
                for name, r in self._retries.items()
            },
            "sagas": self._saga_orchestrator.get_stats(),
        }

    def reset_all(self) -> None:
        """Reset all resilience components."""
        for cb in self._circuit_breakers.values():
            cb.reset()


# Global instance
_resilience_manager: Optional[ResilienceManager] = None


def get_resilience_manager() -> ResilienceManager:
    """Get the global resilience manager."""
    global _resilience_manager
    if _resilience_manager is None:
        _resilience_manager = ResilienceManager()
    return _resilience_manager
