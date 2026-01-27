"""
Plugin Resilience Module

Implements circuit breaker and bulkhead patterns for fault isolation,
preventing cascading failures across plugins.
"""

from aion.plugins.resilience.circuit_breaker import (
    CircuitBreaker,
    CircuitState,
    CircuitBreakerConfig,
    CircuitBreakerRegistry,
)
from aion.plugins.resilience.bulkhead import (
    Bulkhead,
    BulkheadConfig,
    BulkheadRegistry,
    BulkheadFullError,
)
from aion.plugins.resilience.retry import (
    RetryPolicy,
    RetryConfig,
    ExponentialBackoff,
)

__all__ = [
    "CircuitBreaker",
    "CircuitState",
    "CircuitBreakerConfig",
    "CircuitBreakerRegistry",
    "Bulkhead",
    "BulkheadConfig",
    "BulkheadRegistry",
    "BulkheadFullError",
    "RetryPolicy",
    "RetryConfig",
    "ExponentialBackoff",
]
