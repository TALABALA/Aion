"""
AION MCP Advanced Observability

SOTA observability features:
- Correlation ID propagation across service boundaries
- Trace sampling strategies (probabilistic, rate-limiting, adaptive)
- Exemplars linking metrics to traces
- Structured context propagation
- Baggage for cross-cutting concerns
"""

from __future__ import annotations

import asyncio
import contextvars
import functools
import hashlib
import random
import time
import uuid
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

import structlog

logger = structlog.get_logger(__name__)

T = TypeVar("T")


# ============================================
# Correlation Context
# ============================================

# Context variables for correlation
_correlation_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "correlation_id", default=None
)
_trace_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "trace_id", default=None
)
_span_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "span_id", default=None
)
_parent_span_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "parent_span_id", default=None
)
_baggage: contextvars.ContextVar[Dict[str, str]] = contextvars.ContextVar(
    "baggage", default={}
)


def generate_correlation_id() -> str:
    """Generate a new correlation ID."""
    return str(uuid.uuid4())


def generate_trace_id() -> str:
    """Generate a new trace ID (32 hex chars)."""
    return uuid.uuid4().hex


def generate_span_id() -> str:
    """Generate a new span ID (16 hex chars)."""
    return uuid.uuid4().hex[:16]


def get_correlation_id() -> Optional[str]:
    """Get current correlation ID."""
    return _correlation_id.get()


def set_correlation_id(cid: str) -> contextvars.Token:
    """Set correlation ID, returns token for reset."""
    return _correlation_id.set(cid)


def get_trace_context() -> Dict[str, Optional[str]]:
    """Get current trace context."""
    return {
        "correlation_id": _correlation_id.get(),
        "trace_id": _trace_id.get(),
        "span_id": _span_id.get(),
        "parent_span_id": _parent_span_id.get(),
    }


def get_baggage() -> Dict[str, str]:
    """Get current baggage."""
    return _baggage.get().copy()


def set_baggage_item(key: str, value: str) -> None:
    """Set a baggage item."""
    current = _baggage.get().copy()
    current[key] = value
    _baggage.set(current)


def get_baggage_item(key: str) -> Optional[str]:
    """Get a baggage item."""
    return _baggage.get().get(key)


@dataclass
class CorrelationContext:
    """Complete correlation context for propagation."""
    correlation_id: str
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    baggage: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    service_name: Optional[str] = None
    operation_name: Optional[str] = None

    def to_headers(self) -> Dict[str, str]:
        """Convert to HTTP headers for propagation."""
        headers = {
            "X-Correlation-ID": self.correlation_id,
            "X-Trace-ID": self.trace_id,
            "X-Span-ID": self.span_id,
        }

        if self.parent_span_id:
            headers["X-Parent-Span-ID"] = self.parent_span_id

        # W3C Trace Context format
        headers["traceparent"] = (
            f"00-{self.trace_id}-{self.span_id}-01"
        )

        # Baggage as W3C Baggage header
        if self.baggage:
            baggage_items = [f"{k}={v}" for k, v in self.baggage.items()]
            headers["baggage"] = ",".join(baggage_items)

        return headers

    @classmethod
    def from_headers(cls, headers: Dict[str, str]) -> "CorrelationContext":
        """Create context from HTTP headers."""
        correlation_id = headers.get("X-Correlation-ID", generate_correlation_id())

        # Try W3C traceparent first
        traceparent = headers.get("traceparent", "")
        if traceparent:
            parts = traceparent.split("-")
            if len(parts) >= 3:
                trace_id = parts[1]
                span_id = parts[2]
            else:
                trace_id = generate_trace_id()
                span_id = generate_span_id()
        else:
            trace_id = headers.get("X-Trace-ID", generate_trace_id())
            span_id = headers.get("X-Span-ID", generate_span_id())

        parent_span_id = headers.get("X-Parent-Span-ID")

        # Parse baggage
        baggage = {}
        baggage_header = headers.get("baggage", "")
        if baggage_header:
            for item in baggage_header.split(","):
                if "=" in item:
                    k, v = item.split("=", 1)
                    baggage[k.strip()] = v.strip()

        return cls(
            correlation_id=correlation_id,
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            baggage=baggage,
        )

    @classmethod
    def new(
        cls,
        operation_name: Optional[str] = None,
        service_name: Optional[str] = None,
    ) -> "CorrelationContext":
        """Create new root context."""
        return cls(
            correlation_id=generate_correlation_id(),
            trace_id=generate_trace_id(),
            span_id=generate_span_id(),
            operation_name=operation_name,
            service_name=service_name,
        )

    def create_child(
        self,
        operation_name: Optional[str] = None,
    ) -> "CorrelationContext":
        """Create child context for nested operations."""
        return CorrelationContext(
            correlation_id=self.correlation_id,
            trace_id=self.trace_id,
            span_id=generate_span_id(),
            parent_span_id=self.span_id,
            baggage=self.baggage.copy(),
            service_name=self.service_name,
            operation_name=operation_name,
        )


@asynccontextmanager
async def correlation_context(
    correlation_id: Optional[str] = None,
    operation_name: Optional[str] = None,
    inherit: bool = True,
):
    """
    Context manager for correlation context.

    Args:
        correlation_id: Override correlation ID
        operation_name: Name of the operation
        inherit: Whether to inherit from parent context

    Yields:
        CorrelationContext
    """
    # Get or create context
    if inherit and _correlation_id.get():
        ctx = CorrelationContext(
            correlation_id=correlation_id or _correlation_id.get(),
            trace_id=_trace_id.get() or generate_trace_id(),
            span_id=generate_span_id(),
            parent_span_id=_span_id.get(),
            baggage=_baggage.get().copy(),
            operation_name=operation_name,
        )
    else:
        ctx = CorrelationContext.new(operation_name=operation_name)
        if correlation_id:
            ctx.correlation_id = correlation_id

    # Set context variables
    tokens = [
        _correlation_id.set(ctx.correlation_id),
        _trace_id.set(ctx.trace_id),
        _span_id.set(ctx.span_id),
        _parent_span_id.set(ctx.parent_span_id),
        _baggage.set(ctx.baggage),
    ]

    try:
        yield ctx
    finally:
        # Reset context variables
        _correlation_id.reset(tokens[0])
        _trace_id.reset(tokens[1])
        _span_id.reset(tokens[2])
        _parent_span_id.reset(tokens[3])
        _baggage.reset(tokens[4])


# ============================================
# Trace Sampling Strategies
# ============================================

class SamplingDecision(Enum):
    """Sampling decision result."""
    DROP = "drop"
    RECORD_ONLY = "record_only"
    RECORD_AND_SAMPLE = "record_and_sample"


@dataclass
class SamplingResult:
    """Result of a sampling decision."""
    decision: SamplingDecision
    attributes: Dict[str, Any] = field(default_factory=dict)


class Sampler(ABC):
    """Abstract base for trace samplers."""

    @abstractmethod
    def should_sample(
        self,
        trace_id: str,
        span_name: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> SamplingResult:
        """Determine if a trace should be sampled."""
        pass


class AlwaysOnSampler(Sampler):
    """Always sample."""

    def should_sample(
        self,
        trace_id: str,
        span_name: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> SamplingResult:
        return SamplingResult(decision=SamplingDecision.RECORD_AND_SAMPLE)


class AlwaysOffSampler(Sampler):
    """Never sample."""

    def should_sample(
        self,
        trace_id: str,
        span_name: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> SamplingResult:
        return SamplingResult(decision=SamplingDecision.DROP)


class ProbabilitySampler(Sampler):
    """
    Probabilistic sampler based on trace ID.

    Ensures consistent sampling across services for the same trace.
    """

    def __init__(self, probability: float = 0.1):
        """
        Initialize probabilistic sampler.

        Args:
            probability: Sampling probability (0.0 to 1.0)
        """
        self.probability = max(0.0, min(1.0, probability))
        self._threshold = int(self.probability * (2**64 - 1))

    def should_sample(
        self,
        trace_id: str,
        span_name: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> SamplingResult:
        # Use trace ID for consistent sampling
        trace_hash = int(hashlib.md5(trace_id.encode()).hexdigest()[:16], 16)

        if trace_hash < self._threshold:
            return SamplingResult(
                decision=SamplingDecision.RECORD_AND_SAMPLE,
                attributes={"sampler.probability": self.probability},
            )

        return SamplingResult(decision=SamplingDecision.DROP)


class RateLimitingSampler(Sampler):
    """
    Rate-limiting sampler.

    Allows up to N traces per second.
    """

    def __init__(self, max_traces_per_second: float = 10.0):
        """
        Initialize rate-limiting sampler.

        Args:
            max_traces_per_second: Maximum traces to sample per second
        """
        self.max_traces_per_second = max_traces_per_second
        self._tokens = max_traces_per_second
        self._last_update = time.monotonic()
        self._lock = asyncio.Lock()

    def should_sample(
        self,
        trace_id: str,
        span_name: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> SamplingResult:
        now = time.monotonic()
        elapsed = now - self._last_update

        # Refill tokens
        self._tokens = min(
            self.max_traces_per_second,
            self._tokens + elapsed * self.max_traces_per_second,
        )
        self._last_update = now

        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return SamplingResult(
                decision=SamplingDecision.RECORD_AND_SAMPLE,
                attributes={"sampler.type": "rate_limiting"},
            )

        return SamplingResult(decision=SamplingDecision.DROP)


class AdaptiveSampler(Sampler):
    """
    Adaptive sampler that adjusts based on load.

    - Samples more during low traffic
    - Samples less during high traffic
    - Always samples errors
    """

    def __init__(
        self,
        target_traces_per_second: float = 10.0,
        min_probability: float = 0.01,
        max_probability: float = 1.0,
        window_seconds: float = 60.0,
    ):
        """
        Initialize adaptive sampler.

        Args:
            target_traces_per_second: Target trace rate
            min_probability: Minimum sampling probability
            max_probability: Maximum sampling probability
            window_seconds: Window for rate calculation
        """
        self.target_rate = target_traces_per_second
        self.min_probability = min_probability
        self.max_probability = max_probability
        self.window_seconds = window_seconds

        self._request_count = 0
        self._sampled_count = 0
        self._window_start = time.monotonic()
        self._current_probability = max_probability

    def should_sample(
        self,
        trace_id: str,
        span_name: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> SamplingResult:
        now = time.monotonic()

        # Check for window reset
        if now - self._window_start > self.window_seconds:
            self._update_probability()
            self._window_start = now
            self._request_count = 0
            self._sampled_count = 0

        self._request_count += 1

        # Always sample errors
        if attributes and attributes.get("error"):
            self._sampled_count += 1
            return SamplingResult(
                decision=SamplingDecision.RECORD_AND_SAMPLE,
                attributes={
                    "sampler.type": "adaptive",
                    "sampler.reason": "error",
                },
            )

        # Probabilistic sampling
        if random.random() < self._current_probability:
            self._sampled_count += 1
            return SamplingResult(
                decision=SamplingDecision.RECORD_AND_SAMPLE,
                attributes={
                    "sampler.type": "adaptive",
                    "sampler.probability": self._current_probability,
                },
            )

        return SamplingResult(decision=SamplingDecision.DROP)

    def _update_probability(self) -> None:
        """Update sampling probability based on observed rate."""
        if self._request_count == 0:
            return

        elapsed = max(0.001, time.monotonic() - self._window_start)
        observed_rate = self._request_count / elapsed

        if observed_rate > 0:
            # Adjust probability to hit target rate
            desired_probability = self.target_rate / observed_rate
            self._current_probability = max(
                self.min_probability,
                min(self.max_probability, desired_probability),
            )


class ParentBasedSampler(Sampler):
    """
    Parent-based sampler.

    Respects parent sampling decision for consistent traces.
    """

    def __init__(
        self,
        root_sampler: Sampler,
        remote_parent_sampled: Optional[Sampler] = None,
        remote_parent_not_sampled: Optional[Sampler] = None,
    ):
        """
        Initialize parent-based sampler.

        Args:
            root_sampler: Sampler for root spans
            remote_parent_sampled: Sampler when parent was sampled
            remote_parent_not_sampled: Sampler when parent was not sampled
        """
        self.root_sampler = root_sampler
        self.remote_parent_sampled = remote_parent_sampled or AlwaysOnSampler()
        self.remote_parent_not_sampled = remote_parent_not_sampled or AlwaysOffSampler()

    def should_sample(
        self,
        trace_id: str,
        span_name: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> SamplingResult:
        # Check if we have a parent
        parent_span_id = _parent_span_id.get()

        if parent_span_id is None:
            # Root span
            return self.root_sampler.should_sample(trace_id, span_name, attributes)

        # Check parent's sampling decision (from attributes/context)
        parent_sampled = attributes.get("parent_sampled", True) if attributes else True

        if parent_sampled:
            return self.remote_parent_sampled.should_sample(
                trace_id, span_name, attributes
            )
        else:
            return self.remote_parent_not_sampled.should_sample(
                trace_id, span_name, attributes
            )


# ============================================
# Exemplars
# ============================================

@dataclass
class Exemplar:
    """
    Exemplar linking a metric observation to a trace.

    Exemplars allow you to correlate metric data points
    to specific traces for debugging.
    """
    value: float
    timestamp: datetime
    trace_id: str
    span_id: str
    labels: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "labels": self.labels,
        }


class ExemplarReservoir:
    """
    Reservoir for storing exemplars.

    Uses reservoir sampling to maintain a representative sample.
    """

    def __init__(self, max_size: int = 10):
        """
        Initialize reservoir.

        Args:
            max_size: Maximum number of exemplars to store
        """
        self.max_size = max_size
        self._exemplars: List[Exemplar] = []
        self._count = 0
        self._lock = asyncio.Lock()

    async def add(
        self,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Add an exemplar with current trace context."""
        trace_id = _trace_id.get()
        span_id = _span_id.get()

        if not trace_id or not span_id:
            return

        exemplar = Exemplar(
            value=value,
            timestamp=datetime.now(),
            trace_id=trace_id,
            span_id=span_id,
            labels=labels or {},
        )

        async with self._lock:
            self._count += 1

            if len(self._exemplars) < self.max_size:
                self._exemplars.append(exemplar)
            else:
                # Reservoir sampling
                idx = random.randint(0, self._count - 1)
                if idx < self.max_size:
                    self._exemplars[idx] = exemplar

    def get_exemplars(self) -> List[Exemplar]:
        """Get all stored exemplars."""
        return self._exemplars.copy()

    def clear(self) -> None:
        """Clear all exemplars."""
        self._exemplars.clear()
        self._count = 0


class MetricWithExemplars:
    """
    Metric wrapper that collects exemplars.

    Links metric observations to traces for debugging.
    """

    def __init__(
        self,
        name: str,
        description: str,
        reservoir_size: int = 10,
    ):
        """
        Initialize metric with exemplars.

        Args:
            name: Metric name
            description: Metric description
            reservoir_size: Max exemplars to keep
        """
        self.name = name
        self.description = description
        self.reservoir = ExemplarReservoir(reservoir_size)
        self._observations: List[Tuple[float, Dict[str, str]]] = []

    async def observe(
        self,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Record an observation with exemplar."""
        await self.reservoir.add(value, labels)
        self._observations.append((value, labels or {}))

    def get_exemplars(self) -> List[Exemplar]:
        """Get exemplars for this metric."""
        return self.reservoir.get_exemplars()


# ============================================
# Structured Context Logger
# ============================================

class ContextualLogger:
    """
    Logger that automatically includes correlation context.

    Integrates with structlog for structured logging.
    """

    def __init__(self, name: str):
        """
        Initialize contextual logger.

        Args:
            name: Logger name
        """
        self.name = name
        self._logger = structlog.get_logger(name)

    def _get_context(self) -> Dict[str, Any]:
        """Get current correlation context for logging."""
        ctx = {
            "correlation_id": _correlation_id.get(),
            "trace_id": _trace_id.get(),
            "span_id": _span_id.get(),
        }
        return {k: v for k, v in ctx.items() if v is not None}

    def debug(self, msg: str, **kwargs) -> None:
        self._logger.debug(msg, **self._get_context(), **kwargs)

    def info(self, msg: str, **kwargs) -> None:
        self._logger.info(msg, **self._get_context(), **kwargs)

    def warning(self, msg: str, **kwargs) -> None:
        self._logger.warning(msg, **self._get_context(), **kwargs)

    def error(self, msg: str, **kwargs) -> None:
        self._logger.error(msg, **self._get_context(), **kwargs)

    def exception(self, msg: str, **kwargs) -> None:
        self._logger.exception(msg, **self._get_context(), **kwargs)


# ============================================
# Observability Middleware
# ============================================

def with_correlation(
    operation_name: Optional[str] = None,
    sample: bool = True,
    sampler: Optional[Sampler] = None,
):
    """
    Decorator to add correlation context to functions.

    Args:
        operation_name: Name of the operation
        sample: Whether to apply sampling
        sampler: Custom sampler to use
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            op_name = operation_name or func.__name__

            async with correlation_context(operation_name=op_name) as ctx:
                # Apply sampling
                if sample and sampler:
                    result = sampler.should_sample(
                        ctx.trace_id,
                        op_name,
                    )
                    if result.decision == SamplingDecision.DROP:
                        # Still execute, just don't trace
                        pass

                return await func(*args, **kwargs)

        return wrapper
    return decorator


# ============================================
# Global Sampler
# ============================================

_global_sampler: Optional[Sampler] = None


def get_sampler() -> Sampler:
    """Get global sampler."""
    global _global_sampler
    if _global_sampler is None:
        _global_sampler = ProbabilitySampler(0.1)
    return _global_sampler


def set_sampler(sampler: Sampler) -> None:
    """Set global sampler."""
    global _global_sampler
    _global_sampler = sampler
