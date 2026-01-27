"""
Retry Policy Implementation

Provides configurable retry strategies with exponential backoff
for transient failures.
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, TypeVar, Type
from functools import wraps

import structlog

logger = structlog.get_logger(__name__)

T = TypeVar("T")


@dataclass
class RetryConfig:
    """Configuration for retry policy."""

    # Maximum retry attempts (0 = no retries)
    max_attempts: int = 3

    # Base delay between retries (seconds)
    base_delay: float = 1.0

    # Maximum delay between retries (seconds)
    max_delay: float = 60.0

    # Delay multiplier for exponential backoff
    multiplier: float = 2.0

    # Add randomization to prevent thundering herd
    jitter: bool = True
    jitter_factor: float = 0.1  # ±10%

    # Exceptions to retry on
    retry_on: tuple[Type[Exception], ...] = (Exception,)

    # Exceptions to NOT retry on
    no_retry_on: tuple[Type[Exception], ...] = ()

    # Custom retry predicate
    retry_predicate: Optional[Callable[[Exception], bool]] = None

    # Callbacks
    on_retry: Optional[Callable[[int, Exception, float], None]] = None
    on_success: Optional[Callable[[int], None]] = None
    on_failure: Optional[Callable[[int, Exception], None]] = None


@dataclass
class RetryMetrics:
    """Metrics for retry operations."""

    total_calls: int = 0
    successful_first_try: int = 0
    successful_after_retry: int = 0
    failed_all_retries: int = 0
    total_retries: int = 0
    total_delay_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_calls": self.total_calls,
            "successful_first_try": self.successful_first_try,
            "successful_after_retry": self.successful_after_retry,
            "failed_all_retries": self.failed_all_retries,
            "total_retries": self.total_retries,
            "total_delay_seconds": self.total_delay_seconds,
            "retry_rate": (
                self.total_retries / self.total_calls
                if self.total_calls > 0 else 0.0
            ),
        }


class ExponentialBackoff:
    """
    Exponential backoff calculator.

    Computes delay for each retry attempt with optional jitter.
    """

    def __init__(
        self,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        multiplier: float = 2.0,
        jitter: bool = True,
        jitter_factor: float = 0.1,
    ):
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.multiplier = multiplier
        self.jitter = jitter
        self.jitter_factor = jitter_factor

    def get_delay(self, attempt: int) -> float:
        """
        Get delay for a given attempt number.

        Args:
            attempt: Attempt number (0-indexed)

        Returns:
            Delay in seconds
        """
        # Calculate base exponential delay
        delay = self.base_delay * (self.multiplier ** attempt)

        # Cap at max delay
        delay = min(delay, self.max_delay)

        # Add jitter
        if self.jitter:
            jitter_range = delay * self.jitter_factor
            delay += random.uniform(-jitter_range, jitter_range)

        return max(0, delay)

    def __iter__(self):
        """Iterate over delays."""
        attempt = 0
        while True:
            yield self.get_delay(attempt)
            attempt += 1


class RetryPolicy:
    """
    Retry policy implementation.

    Automatically retries failed operations with configurable backoff.

    Usage:
        policy = RetryPolicy(RetryConfig(max_attempts=3))

        @policy
        async def flaky_operation():
            ...

        # Or manually
        result = await policy.execute(flaky_operation, arg1, arg2)
    """

    def __init__(self, config: Optional[RetryConfig] = None):
        self.config = config or RetryConfig()
        self._metrics = RetryMetrics()
        self._backoff = ExponentialBackoff(
            base_delay=self.config.base_delay,
            max_delay=self.config.max_delay,
            multiplier=self.config.multiplier,
            jitter=self.config.jitter,
            jitter_factor=self.config.jitter_factor,
        )

    @property
    def metrics(self) -> RetryMetrics:
        """Get retry metrics."""
        return self._metrics

    def __call__(self, func: Callable) -> Callable:
        """Use as decorator."""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await self.execute(func, *args, **kwargs)
        return wrapper

    async def execute(
        self,
        func: Callable[..., T],
        *args,
        **kwargs,
    ) -> T:
        """
        Execute a function with retry logic.

        Args:
            func: Async function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            Last exception if all retries fail
        """
        self._metrics.total_calls += 1
        last_exception: Optional[Exception] = None
        attempts = 0

        while attempts <= self.config.max_attempts:
            try:
                result = await func(*args, **kwargs)

                # Success
                if attempts == 0:
                    self._metrics.successful_first_try += 1
                else:
                    self._metrics.successful_after_retry += 1

                if self.config.on_success:
                    self.config.on_success(attempts)

                return result

            except Exception as e:
                last_exception = e

                # Check if we should retry
                if not self._should_retry(e):
                    self._metrics.failed_all_retries += 1
                    if self.config.on_failure:
                        self.config.on_failure(attempts, e)
                    raise

                # Check if we have retries left
                if attempts >= self.config.max_attempts:
                    self._metrics.failed_all_retries += 1
                    if self.config.on_failure:
                        self.config.on_failure(attempts, e)
                    raise

                # Calculate delay
                delay = self._backoff.get_delay(attempts)
                self._metrics.total_retries += 1
                self._metrics.total_delay_seconds += delay

                logger.warning(
                    "Retrying operation",
                    attempt=attempts + 1,
                    max_attempts=self.config.max_attempts,
                    delay=delay,
                    error=str(e),
                )

                if self.config.on_retry:
                    self.config.on_retry(attempts, e, delay)

                # Wait before retry
                await asyncio.sleep(delay)
                attempts += 1

        # Should not reach here, but just in case
        if last_exception:
            raise last_exception
        raise RuntimeError("Retry loop exited unexpectedly")

    def _should_retry(self, exc: Exception) -> bool:
        """Check if exception should trigger a retry."""
        # Check custom predicate first
        if self.config.retry_predicate:
            return self.config.retry_predicate(exc)

        # Check no-retry list
        if isinstance(exc, self.config.no_retry_on):
            return False

        # Check retry list
        return isinstance(exc, self.config.retry_on)

    def get_stats(self) -> dict[str, Any]:
        """Get retry policy statistics."""
        return {
            "config": {
                "max_attempts": self.config.max_attempts,
                "base_delay": self.config.base_delay,
                "max_delay": self.config.max_delay,
                "multiplier": self.config.multiplier,
            },
            "metrics": self._metrics.to_dict(),
        }


class RetryPolicyRegistry:
    """Registry for managing retry policies."""

    def __init__(self):
        self._policies: dict[str, RetryPolicy] = {}

    def register(self, name: str, config: Optional[RetryConfig] = None) -> RetryPolicy:
        """Register a new retry policy."""
        policy = RetryPolicy(config)
        self._policies[name] = policy
        return policy

    def get(self, name: str) -> Optional[RetryPolicy]:
        """Get a retry policy by name."""
        return self._policies.get(name)

    def get_stats(self) -> dict[str, Any]:
        """Get statistics for all policies."""
        return {
            name: policy.get_stats()
            for name, policy in self._policies.items()
        }
