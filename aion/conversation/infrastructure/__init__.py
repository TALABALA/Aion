"""
AION Infrastructure Components

State-of-the-art infrastructure for:
- Resilience patterns (circuit breakers, retry, fallback)
- Observability (OpenTelemetry, Prometheus metrics)
- Safety guardrails (content moderation, PII detection)
- Distributed session storage (Redis, PostgreSQL)
- Semantic caching
"""

from aion.conversation.infrastructure.resilience import (
    CircuitBreaker,
    CircuitState,
    RetryPolicy,
    Fallback,
    ResilientExecutor,
)
from aion.conversation.infrastructure.observability import (
    MetricsCollector,
    Tracer,
    ConversationMetrics,
)
from aion.conversation.infrastructure.safety import (
    ContentModerator,
    PIIDetector,
    SafetyGuard,
    SafetyLevel,
)
from aion.conversation.infrastructure.storage import (
    SessionStore,
    RedisSessionStore,
    PostgresSessionStore,
)
from aion.conversation.infrastructure.caching import (
    SemanticCache,
    CacheEntry,
)

__all__ = [
    # Resilience
    "CircuitBreaker",
    "CircuitState",
    "RetryPolicy",
    "Fallback",
    "ResilientExecutor",
    # Observability
    "MetricsCollector",
    "Tracer",
    "ConversationMetrics",
    # Safety
    "ContentModerator",
    "PIIDetector",
    "SafetyGuard",
    "SafetyLevel",
    # Storage
    "SessionStore",
    "RedisSessionStore",
    "PostgresSessionStore",
    # Caching
    "SemanticCache",
    "CacheEntry",
]
