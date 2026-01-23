"""
AION SOTA Observability

State-of-the-art observability featuring:
- OpenTelemetry distributed tracing
- Prometheus metrics
- Structured logging
- Performance monitoring
- Error tracking
"""

from __future__ import annotations

import asyncio
import functools
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TypeVar

import structlog

logger = structlog.get_logger(__name__)

T = TypeVar("T")


# =============================================================================
# Metrics Types
# =============================================================================

class MetricType(str, Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricValue:
    """A metric value with metadata."""
    name: str
    value: float
    metric_type: MetricType
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    description: str = ""


@dataclass
class HistogramBucket:
    """A histogram bucket."""
    le: float  # Less than or equal
    count: int = 0


# =============================================================================
# Metrics Collector
# =============================================================================

class MetricsCollector:
    """
    Metrics collector with Prometheus-compatible interface.

    Supports:
    - Counters (monotonically increasing values)
    - Gauges (values that can go up and down)
    - Histograms (distribution of values)
    - Summaries (quantiles)
    """

    DEFAULT_HISTOGRAM_BUCKETS = [
        0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, float("inf")
    ]

    def __init__(self, namespace: str = "aion"):
        self.namespace = namespace

        # Metric storage
        self._counters: Dict[str, Dict[str, float]] = {}
        self._gauges: Dict[str, Dict[str, float]] = {}
        self._histograms: Dict[str, Dict[str, List[HistogramBucket]]] = {}
        self._histogram_sums: Dict[str, Dict[str, float]] = {}
        self._histogram_counts: Dict[str, Dict[str, int]] = {}

        # Prometheus client (if available)
        self._prom_counters: Dict[str, Any] = {}
        self._prom_gauges: Dict[str, Any] = {}
        self._prom_histograms: Dict[str, Any] = {}
        self._prometheus_available = False

        self._initialize_prometheus()

    def _initialize_prometheus(self) -> None:
        """Initialize Prometheus client if available."""
        try:
            from prometheus_client import Counter, Gauge, Histogram, REGISTRY
            self._prometheus_available = True
            self._prom_registry = REGISTRY
            logger.info("Prometheus client initialized")
        except ImportError:
            logger.warning("Prometheus client not available, using internal metrics")

    def _get_label_key(self, labels: Dict[str, str]) -> str:
        """Create a unique key from labels."""
        return "|".join(f"{k}={v}" for k, v in sorted(labels.items()))

    # -------------------------------------------------------------------------
    # Counter Operations
    # -------------------------------------------------------------------------

    def counter_inc(
        self,
        name: str,
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None,
        description: str = "",
    ) -> None:
        """Increment a counter."""
        full_name = f"{self.namespace}_{name}"
        labels = labels or {}
        label_key = self._get_label_key(labels)

        # Internal storage
        if full_name not in self._counters:
            self._counters[full_name] = {}
        self._counters[full_name][label_key] = self._counters[full_name].get(label_key, 0) + value

        # Prometheus
        if self._prometheus_available:
            try:
                from prometheus_client import Counter

                if full_name not in self._prom_counters:
                    self._prom_counters[full_name] = Counter(
                        full_name,
                        description or name,
                        list(labels.keys()) if labels else [],
                    )

                if labels:
                    self._prom_counters[full_name].labels(**labels).inc(value)
                else:
                    self._prom_counters[full_name].inc(value)
            except Exception as e:
                logger.debug(f"Prometheus counter error: {e}")

    def counter_get(
        self,
        name: str,
        labels: Optional[Dict[str, str]] = None,
    ) -> float:
        """Get counter value."""
        full_name = f"{self.namespace}_{name}"
        label_key = self._get_label_key(labels or {})
        return self._counters.get(full_name, {}).get(label_key, 0.0)

    # -------------------------------------------------------------------------
    # Gauge Operations
    # -------------------------------------------------------------------------

    def gauge_set(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
        description: str = "",
    ) -> None:
        """Set a gauge value."""
        full_name = f"{self.namespace}_{name}"
        labels = labels or {}
        label_key = self._get_label_key(labels)

        # Internal storage
        if full_name not in self._gauges:
            self._gauges[full_name] = {}
        self._gauges[full_name][label_key] = value

        # Prometheus
        if self._prometheus_available:
            try:
                from prometheus_client import Gauge

                if full_name not in self._prom_gauges:
                    self._prom_gauges[full_name] = Gauge(
                        full_name,
                        description or name,
                        list(labels.keys()) if labels else [],
                    )

                if labels:
                    self._prom_gauges[full_name].labels(**labels).set(value)
                else:
                    self._prom_gauges[full_name].set(value)
            except Exception as e:
                logger.debug(f"Prometheus gauge error: {e}")

    def gauge_inc(
        self,
        name: str,
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Increment a gauge."""
        full_name = f"{self.namespace}_{name}"
        labels = labels or {}
        label_key = self._get_label_key(labels)

        if full_name not in self._gauges:
            self._gauges[full_name] = {}
        current = self._gauges[full_name].get(label_key, 0)
        self._gauges[full_name][label_key] = current + value

    def gauge_dec(
        self,
        name: str,
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Decrement a gauge."""
        self.gauge_inc(name, -value, labels)

    def gauge_get(
        self,
        name: str,
        labels: Optional[Dict[str, str]] = None,
    ) -> float:
        """Get gauge value."""
        full_name = f"{self.namespace}_{name}"
        label_key = self._get_label_key(labels or {})
        return self._gauges.get(full_name, {}).get(label_key, 0.0)

    # -------------------------------------------------------------------------
    # Histogram Operations
    # -------------------------------------------------------------------------

    def histogram_observe(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
        buckets: Optional[List[float]] = None,
        description: str = "",
    ) -> None:
        """Record a histogram observation."""
        full_name = f"{self.namespace}_{name}"
        labels = labels or {}
        label_key = self._get_label_key(labels)
        buckets = buckets or self.DEFAULT_HISTOGRAM_BUCKETS

        # Internal storage
        if full_name not in self._histograms:
            self._histograms[full_name] = {}
            self._histogram_sums[full_name] = {}
            self._histogram_counts[full_name] = {}

        if label_key not in self._histograms[full_name]:
            self._histograms[full_name][label_key] = [
                HistogramBucket(le=b) for b in buckets
            ]
            self._histogram_sums[full_name][label_key] = 0.0
            self._histogram_counts[full_name][label_key] = 0

        # Update buckets
        for bucket in self._histograms[full_name][label_key]:
            if value <= bucket.le:
                bucket.count += 1

        self._histogram_sums[full_name][label_key] += value
        self._histogram_counts[full_name][label_key] += 1

        # Prometheus
        if self._prometheus_available:
            try:
                from prometheus_client import Histogram

                if full_name not in self._prom_histograms:
                    self._prom_histograms[full_name] = Histogram(
                        full_name,
                        description or name,
                        list(labels.keys()) if labels else [],
                        buckets=buckets[:-1],  # Remove inf
                    )

                if labels:
                    self._prom_histograms[full_name].labels(**labels).observe(value)
                else:
                    self._prom_histograms[full_name].observe(value)
            except Exception as e:
                logger.debug(f"Prometheus histogram error: {e}")

    def histogram_get_stats(
        self,
        name: str,
        labels: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Get histogram statistics."""
        full_name = f"{self.namespace}_{name}"
        label_key = self._get_label_key(labels or {})

        if full_name not in self._histograms:
            return {}

        return {
            "sum": self._histogram_sums.get(full_name, {}).get(label_key, 0),
            "count": self._histogram_counts.get(full_name, {}).get(label_key, 0),
            "buckets": [
                {"le": b.le, "count": b.count}
                for b in self._histograms.get(full_name, {}).get(label_key, [])
            ],
        }

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics as a dictionary."""
        return {
            "counters": dict(self._counters),
            "gauges": dict(self._gauges),
            "histograms": {
                name: {
                    label: self.histogram_get_stats(name.replace(f"{self.namespace}_", ""), dict(pair.split("=") for pair in label.split("|") if "=" in pair) if label else None)
                    for label in labels.keys()
                }
                for name, labels in self._histograms.items()
            },
        }

    def clear(self) -> None:
        """Clear all metrics."""
        self._counters.clear()
        self._gauges.clear()
        self._histograms.clear()
        self._histogram_sums.clear()
        self._histogram_counts.clear()


# =============================================================================
# Tracer (OpenTelemetry Compatible)
# =============================================================================

@dataclass
class SpanContext:
    """Context for a trace span."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None


@dataclass
class Span:
    """A trace span."""
    name: str
    context: SpanContext
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "OK"
    status_message: str = ""

    def set_attribute(self, key: str, value: Any) -> None:
        """Set an attribute."""
        self.attributes[key] = value

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        """Add an event."""
        self.events.append({
            "name": name,
            "timestamp": datetime.now().isoformat(),
            "attributes": attributes or {},
        })

    def set_status(self, status: str, message: str = "") -> None:
        """Set span status."""
        self.status = status
        self.status_message = message

    def end(self) -> None:
        """End the span."""
        self.end_time = datetime.now()

    @property
    def duration_ms(self) -> float:
        """Get span duration in milliseconds."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds() * 1000
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "trace_id": self.context.trace_id,
            "span_id": self.context.span_id,
            "parent_span_id": self.context.parent_span_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "attributes": self.attributes,
            "events": self.events,
            "status": self.status,
            "status_message": self.status_message,
        }


class Tracer:
    """
    Distributed tracer compatible with OpenTelemetry.

    Provides span-based tracing for request flows across services.
    """

    def __init__(
        self,
        service_name: str = "aion-conversation",
        exporter: Optional[Any] = None,
    ):
        self.service_name = service_name
        self.exporter = exporter

        # OpenTelemetry tracer (if available)
        self._otel_tracer = None
        self._otel_available = False

        # Internal span storage
        self._spans: Dict[str, Span] = {}
        self._active_span: Optional[Span] = None

        self._initialize_otel()

    def _initialize_otel(self) -> None:
        """Initialize OpenTelemetry if available."""
        try:
            from opentelemetry import trace
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.resources import Resource

            resource = Resource.create({"service.name": self.service_name})
            provider = TracerProvider(resource=resource)
            trace.set_tracer_provider(provider)

            self._otel_tracer = trace.get_tracer(self.service_name)
            self._otel_available = True
            logger.info("OpenTelemetry tracer initialized")
        except ImportError:
            logger.warning("OpenTelemetry not available, using internal tracing")

    def _generate_id(self, length: int = 16) -> str:
        """Generate a random ID."""
        import secrets
        return secrets.token_hex(length)

    def start_span(
        self,
        name: str,
        parent: Optional[Span] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Span:
        """Start a new span."""
        # Create context
        trace_id = parent.context.trace_id if parent else self._generate_id(16)
        span_id = self._generate_id(8)
        parent_span_id = parent.context.span_id if parent else None

        context = SpanContext(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
        )

        # Create span
        span = Span(
            name=name,
            context=context,
            attributes=attributes or {},
        )

        span.set_attribute("service.name", self.service_name)

        # Store span
        self._spans[span_id] = span
        self._active_span = span

        logger.debug(f"Started span: {name}", trace_id=trace_id[:8], span_id=span_id[:8])

        return span

    def end_span(self, span: Span) -> None:
        """End a span."""
        span.end()

        if self._active_span == span:
            self._active_span = None

        logger.debug(
            f"Ended span: {span.name}",
            duration_ms=f"{span.duration_ms:.2f}",
            status=span.status,
        )

        # Export if exporter is configured
        if self.exporter:
            try:
                self.exporter.export([span])
            except Exception as e:
                logger.warning(f"Failed to export span: {e}")

    @asynccontextmanager
    async def trace(
        self,
        name: str,
        parent: Optional[Span] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ):
        """Context manager for tracing."""
        span = self.start_span(name, parent, attributes)
        try:
            yield span
        except Exception as e:
            span.set_status("ERROR", str(e))
            span.add_event("exception", {"type": type(e).__name__, "message": str(e)})
            raise
        finally:
            self.end_span(span)

    def get_active_span(self) -> Optional[Span]:
        """Get the currently active span."""
        return self._active_span

    def get_span(self, span_id: str) -> Optional[Span]:
        """Get a span by ID."""
        return self._spans.get(span_id)


# =============================================================================
# Conversation Metrics
# =============================================================================

class ConversationMetrics:
    """
    Predefined metrics for conversation system.
    """

    def __init__(self, collector: Optional[MetricsCollector] = None):
        self.collector = collector or MetricsCollector()

    def record_request(
        self,
        endpoint: str,
        method: str = "POST",
        status_code: int = 200,
    ) -> None:
        """Record an API request."""
        self.collector.counter_inc(
            "http_requests_total",
            labels={"endpoint": endpoint, "method": method, "status": str(status_code)},
        )

    def record_response_time(
        self,
        endpoint: str,
        duration_seconds: float,
    ) -> None:
        """Record response time."""
        self.collector.histogram_observe(
            "http_request_duration_seconds",
            duration_seconds,
            labels={"endpoint": endpoint},
        )

    def record_llm_request(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        duration_seconds: float,
        success: bool = True,
    ) -> None:
        """Record an LLM request."""
        self.collector.counter_inc(
            "llm_requests_total",
            labels={"model": model, "success": str(success)},
        )

        self.collector.counter_inc(
            "llm_tokens_total",
            prompt_tokens + completion_tokens,
            labels={"model": model, "type": "total"},
        )

        self.collector.counter_inc(
            "llm_prompt_tokens_total",
            prompt_tokens,
            labels={"model": model},
        )

        self.collector.counter_inc(
            "llm_completion_tokens_total",
            completion_tokens,
            labels={"model": model},
        )

        self.collector.histogram_observe(
            "llm_request_duration_seconds",
            duration_seconds,
            labels={"model": model},
        )

    def record_conversation(
        self,
        conversation_id: str,
        message_count: int,
        duration_seconds: float,
    ) -> None:
        """Record conversation metrics."""
        self.collector.counter_inc("conversations_total")

        self.collector.histogram_observe(
            "conversation_messages_count",
            message_count,
        )

        self.collector.histogram_observe(
            "conversation_duration_seconds",
            duration_seconds,
        )

    def record_tool_execution(
        self,
        tool_name: str,
        duration_seconds: float,
        success: bool = True,
    ) -> None:
        """Record tool execution."""
        self.collector.counter_inc(
            "tool_executions_total",
            labels={"tool": tool_name, "success": str(success)},
        )

        self.collector.histogram_observe(
            "tool_execution_duration_seconds",
            duration_seconds,
            labels={"tool": tool_name},
        )

    def record_memory_operation(
        self,
        operation: str,
        memory_type: str,
        duration_seconds: float,
    ) -> None:
        """Record memory operation."""
        self.collector.counter_inc(
            "memory_operations_total",
            labels={"operation": operation, "type": memory_type},
        )

        self.collector.histogram_observe(
            "memory_operation_duration_seconds",
            duration_seconds,
            labels={"operation": operation, "type": memory_type},
        )

    def record_cache_hit(self, cache_name: str, hit: bool) -> None:
        """Record cache hit/miss."""
        self.collector.counter_inc(
            "cache_requests_total",
            labels={"cache": cache_name, "hit": str(hit)},
        )

    def record_active_sessions(self, count: int) -> None:
        """Record active session count."""
        self.collector.gauge_set("active_sessions", count)

    def record_error(
        self,
        error_type: str,
        component: str,
    ) -> None:
        """Record an error."""
        self.collector.counter_inc(
            "errors_total",
            labels={"type": error_type, "component": component},
        )


# =============================================================================
# Decorators
# =============================================================================

def traced(
    tracer: Tracer,
    name: Optional[str] = None,
) -> Callable:
    """Decorator to trace a function."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        span_name = name or func.__name__

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> T:
            async with tracer.trace(span_name) as span:
                span.set_attribute("function", func.__name__)
                return await func(*args, **kwargs)

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> T:
            span = tracer.start_span(span_name)
            span.set_attribute("function", func.__name__)
            try:
                return func(*args, **kwargs)
            except Exception as e:
                span.set_status("ERROR", str(e))
                raise
            finally:
                tracer.end_span(span)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def timed(
    metrics: MetricsCollector,
    metric_name: str,
    labels: Optional[Dict[str, str]] = None,
) -> Callable:
    """Decorator to time a function."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> T:
            start = time.time()
            try:
                return await func(*args, **kwargs)
            finally:
                duration = time.time() - start
                metrics.histogram_observe(metric_name, duration, labels)

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> T:
            start = time.time()
            try:
                return func(*args, **kwargs)
            finally:
                duration = time.time() - start
                metrics.histogram_observe(metric_name, duration, labels)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator
