"""
AION Observability Stack

Enterprise-grade observability with:
- OpenTelemetry integration (traces, metrics, logs)
- Prometheus metrics export
- Distributed tracing with context propagation
- Automatic span instrumentation
- Custom metrics and gauges
- Health check endpoints
"""

from __future__ import annotations

import asyncio
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from functools import wraps
from typing import Any, Callable, Dict, Generator, List, Optional, Set, Tuple, TypeVar, Union

import structlog

logger = structlog.get_logger(__name__)

F = TypeVar("F", bound=Callable)


# === OpenTelemetry-compatible Tracing ===

class SpanKind(Enum):
    """Type of span."""
    INTERNAL = auto()
    SERVER = auto()
    CLIENT = auto()
    PRODUCER = auto()
    CONSUMER = auto()


class SpanStatus(Enum):
    """Status of a span."""
    UNSET = auto()
    OK = auto()
    ERROR = auto()


@dataclass
class SpanContext:
    """Context for distributed tracing."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    trace_flags: int = 1  # Sampled
    trace_state: Dict[str, str] = field(default_factory=dict)
    baggage: Dict[str, str] = field(default_factory=dict)

    def to_header(self) -> str:
        """Convert to W3C traceparent format."""
        return f"00-{self.trace_id}-{self.span_id}-{self.trace_flags:02x}"

    @classmethod
    def from_header(cls, header: str) -> Optional["SpanContext"]:
        """Parse W3C traceparent format."""
        try:
            parts = header.split("-")
            if len(parts) != 4:
                return None
            return cls(
                trace_id=parts[1],
                span_id=parts[2],
                trace_flags=int(parts[3], 16),
            )
        except Exception:
            return None

    @classmethod
    def generate(cls, parent: Optional["SpanContext"] = None) -> "SpanContext":
        """Generate a new span context."""
        return cls(
            trace_id=parent.trace_id if parent else uuid.uuid4().hex,
            span_id=uuid.uuid4().hex[:16],
            parent_span_id=parent.span_id if parent else None,
            trace_flags=parent.trace_flags if parent else 1,
            trace_state=dict(parent.trace_state) if parent else {},
            baggage=dict(parent.baggage) if parent else {},
        )


@dataclass
class SpanEvent:
    """Event within a span."""
    name: str
    timestamp: datetime = field(default_factory=datetime.now)
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Span:
    """A tracing span."""
    name: str
    context: SpanContext
    kind: SpanKind = SpanKind.INTERNAL
    status: SpanStatus = SpanStatus.UNSET
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[SpanEvent] = field(default_factory=list)
    links: List[SpanContext] = field(default_factory=list)

    # Resource attributes
    service_name: str = "aion"
    service_version: str = ""
    host_name: str = ""

    def set_attribute(self, key: str, value: Any) -> None:
        """Set a span attribute."""
        self.attributes[key] = value

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        """Add an event to the span."""
        self.events.append(SpanEvent(
            name=name,
            attributes=attributes or {},
        ))

    def set_status(self, status: SpanStatus, description: Optional[str] = None) -> None:
        """Set span status."""
        self.status = status
        if description:
            self.attributes["status.description"] = description

    def end(self, end_time: Optional[datetime] = None) -> None:
        """End the span."""
        self.end_time = end_time or datetime.now()

    @property
    def duration_ms(self) -> float:
        """Get span duration in milliseconds."""
        if not self.end_time:
            return 0
        return (self.end_time - self.start_time).total_seconds() * 1000

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for export."""
        return {
            "name": self.name,
            "trace_id": self.context.trace_id,
            "span_id": self.context.span_id,
            "parent_span_id": self.context.parent_span_id,
            "kind": self.kind.name,
            "status": self.status.name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "attributes": self.attributes,
            "events": [
                {"name": e.name, "timestamp": e.timestamp.isoformat(), "attributes": e.attributes}
                for e in self.events
            ],
            "service": {
                "name": self.service_name,
                "version": self.service_version,
            },
        }


class SpanExporter(ABC):
    """Abstract span exporter."""

    @abstractmethod
    async def export(self, spans: List[Span]) -> None:
        """Export spans."""
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the exporter."""
        pass


class ConsoleSpanExporter(SpanExporter):
    """Export spans to console."""

    async def export(self, spans: List[Span]) -> None:
        for span in spans:
            logger.info(
                "Span",
                name=span.name,
                trace_id=span.context.trace_id,
                span_id=span.context.span_id,
                duration_ms=span.duration_ms,
                status=span.status.name,
            )

    async def shutdown(self) -> None:
        pass


class OTLPSpanExporter(SpanExporter):
    """Export spans in OTLP format (OpenTelemetry Protocol)."""

    def __init__(
        self,
        endpoint: str = "http://localhost:4318/v1/traces",
        headers: Optional[Dict[str, str]] = None,
    ):
        self.endpoint = endpoint
        self.headers = headers or {}

    async def export(self, spans: List[Span]) -> None:
        """Export spans to OTLP endpoint."""
        try:
            import aiohttp

            payload = {
                "resourceSpans": [{
                    "resource": {
                        "attributes": [
                            {"key": "service.name", "value": {"stringValue": spans[0].service_name if spans else "aion"}},
                        ]
                    },
                    "scopeSpans": [{
                        "spans": [self._convert_span(s) for s in spans]
                    }]
                }]
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.endpoint,
                    json=payload,
                    headers={"Content-Type": "application/json", **self.headers},
                ) as response:
                    if response.status != 200:
                        logger.warning(f"OTLP export failed: {response.status}")

        except ImportError:
            logger.warning("aiohttp not installed, OTLP export disabled")
        except Exception as e:
            logger.error(f"OTLP export error: {e}")

    def _convert_span(self, span: Span) -> Dict[str, Any]:
        """Convert span to OTLP format."""
        return {
            "traceId": span.context.trace_id,
            "spanId": span.context.span_id,
            "parentSpanId": span.context.parent_span_id or "",
            "name": span.name,
            "kind": span.kind.value,
            "startTimeUnixNano": int(span.start_time.timestamp() * 1e9),
            "endTimeUnixNano": int(span.end_time.timestamp() * 1e9) if span.end_time else 0,
            "attributes": [
                {"key": k, "value": {"stringValue": str(v)}}
                for k, v in span.attributes.items()
            ],
            "status": {"code": span.status.value},
        }

    async def shutdown(self) -> None:
        pass


class Tracer:
    """
    Distributed tracer with OpenTelemetry compatibility.

    Features:
    - Context propagation
    - Automatic span correlation
    - Multiple exporters
    - Sampling
    """

    # Thread-local context storage
    _current_context: Optional[SpanContext] = None
    _current_span: Optional[Span] = None

    def __init__(
        self,
        service_name: str = "aion",
        service_version: str = "",
        exporters: Optional[List[SpanExporter]] = None,
        sample_rate: float = 1.0,
        max_spans_per_trace: int = 1000,
        batch_size: int = 100,
        flush_interval: float = 5.0,
    ):
        self.service_name = service_name
        self.service_version = service_version
        self.exporters = exporters or [ConsoleSpanExporter()]
        self.sample_rate = sample_rate
        self.max_spans_per_trace = max_spans_per_trace
        self.batch_size = batch_size
        self.flush_interval = flush_interval

        # Span buffer
        self._span_buffer: List[Span] = []
        self._buffer_lock = asyncio.Lock()

        # Background flush
        self._flush_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

        # Stats
        self._stats = {
            "spans_created": 0,
            "spans_exported": 0,
            "spans_dropped": 0,
        }

    async def start(self) -> None:
        """Start the tracer."""
        self._flush_task = asyncio.create_task(self._flush_loop())
        logger.info("Tracer started", service=self.service_name)

    async def shutdown(self) -> None:
        """Shutdown the tracer."""
        self._shutdown_event.set()

        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass

        # Final flush
        await self._flush_spans()

        # Shutdown exporters
        for exporter in self.exporters:
            await exporter.shutdown()

        logger.info("Tracer shutdown", spans_exported=self._stats["spans_exported"])

    def _should_sample(self) -> bool:
        """Determine if trace should be sampled."""
        import random
        return random.random() < self.sample_rate

    @contextmanager
    def start_span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Optional[Dict[str, Any]] = None,
        parent: Optional[SpanContext] = None,
    ) -> Generator[Span, None, None]:
        """Start a new span (sync context manager)."""
        if not self._should_sample():
            # Return dummy span
            yield Span(name=name, context=SpanContext.generate())
            return

        parent_context = parent or Tracer._current_context
        context = SpanContext.generate(parent_context)

        span = Span(
            name=name,
            context=context,
            kind=kind,
            attributes=attributes or {},
            service_name=self.service_name,
            service_version=self.service_version,
        )

        # Set as current
        old_context = Tracer._current_context
        old_span = Tracer._current_span
        Tracer._current_context = context
        Tracer._current_span = span

        self._stats["spans_created"] += 1

        try:
            yield span
            if span.status == SpanStatus.UNSET:
                span.set_status(SpanStatus.OK)
        except Exception as e:
            span.set_status(SpanStatus.ERROR, str(e))
            span.add_event("exception", {"message": str(e), "type": type(e).__name__})
            raise
        finally:
            span.end()
            Tracer._current_context = old_context
            Tracer._current_span = old_span

            # Buffer for export
            asyncio.create_task(self._buffer_span(span))

    @asynccontextmanager
    async def start_async_span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Optional[Dict[str, Any]] = None,
        parent: Optional[SpanContext] = None,
    ):
        """Start a new span (async context manager)."""
        if not self._should_sample():
            yield Span(name=name, context=SpanContext.generate())
            return

        parent_context = parent or Tracer._current_context
        context = SpanContext.generate(parent_context)

        span = Span(
            name=name,
            context=context,
            kind=kind,
            attributes=attributes or {},
            service_name=self.service_name,
            service_version=self.service_version,
        )

        old_context = Tracer._current_context
        old_span = Tracer._current_span
        Tracer._current_context = context
        Tracer._current_span = span

        self._stats["spans_created"] += 1

        try:
            yield span
            if span.status == SpanStatus.UNSET:
                span.set_status(SpanStatus.OK)
        except Exception as e:
            span.set_status(SpanStatus.ERROR, str(e))
            span.add_event("exception", {"message": str(e), "type": type(e).__name__})
            raise
        finally:
            span.end()
            Tracer._current_context = old_context
            Tracer._current_span = old_span
            await self._buffer_span(span)

    def trace(
        self,
        name: Optional[str] = None,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Callable[[F], F]:
        """Decorator to trace a function."""
        def decorator(func: F) -> F:
            span_name = name or func.__name__

            if asyncio.iscoroutinefunction(func):
                @wraps(func)
                async def async_wrapper(*args, **kwargs):
                    async with self.start_async_span(span_name, kind, attributes) as span:
                        span.set_attribute("function", func.__name__)
                        span.set_attribute("module", func.__module__)
                        return await func(*args, **kwargs)
                return async_wrapper
            else:
                @wraps(func)
                def sync_wrapper(*args, **kwargs):
                    with self.start_span(span_name, kind, attributes) as span:
                        span.set_attribute("function", func.__name__)
                        span.set_attribute("module", func.__module__)
                        return func(*args, **kwargs)
                return sync_wrapper

        return decorator

    async def _buffer_span(self, span: Span) -> None:
        """Add span to buffer."""
        async with self._buffer_lock:
            self._span_buffer.append(span)

            if len(self._span_buffer) >= self.batch_size:
                await self._flush_spans()

    async def _flush_spans(self) -> None:
        """Flush buffered spans to exporters."""
        async with self._buffer_lock:
            if not self._span_buffer:
                return

            spans = self._span_buffer[:]
            self._span_buffer.clear()

        for exporter in self.exporters:
            try:
                await exporter.export(spans)
                self._stats["spans_exported"] += len(spans)
            except Exception as e:
                logger.error(f"Span export error: {e}")
                self._stats["spans_dropped"] += len(spans)

    async def _flush_loop(self) -> None:
        """Background flush loop."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(self.flush_interval)
                await self._flush_spans()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Flush loop error: {e}")

    @classmethod
    def get_current_context(cls) -> Optional[SpanContext]:
        """Get current span context."""
        return cls._current_context

    @classmethod
    def get_current_span(cls) -> Optional[Span]:
        """Get current span."""
        return cls._current_span

    def get_stats(self) -> Dict[str, Any]:
        """Get tracer statistics."""
        return {
            **self._stats,
            "buffer_size": len(self._span_buffer),
        }


# === Prometheus-compatible Metrics ===

class MetricType(Enum):
    """Type of metric."""
    COUNTER = auto()
    GAUGE = auto()
    HISTOGRAM = auto()
    SUMMARY = auto()


@dataclass
class MetricSample:
    """A metric sample."""
    name: str
    labels: Dict[str, str]
    value: float
    timestamp: datetime = field(default_factory=datetime.now)


class Counter:
    """Prometheus-compatible counter."""

    def __init__(self, name: str, description: str, labels: Optional[List[str]] = None):
        self.name = name
        self.description = description
        self.label_names = labels or []
        self._values: Dict[tuple, float] = defaultdict(float)
        self._lock = asyncio.Lock()

    def inc(self, amount: float = 1, **labels) -> None:
        """Increment the counter."""
        key = self._label_key(labels)
        self._values[key] += amount

    def _label_key(self, labels: Dict[str, str]) -> tuple:
        """Create a hashable key from labels."""
        return tuple(sorted(labels.items()))

    def collect(self) -> List[MetricSample]:
        """Collect all samples."""
        samples = []
        for key, value in self._values.items():
            labels = dict(key)
            samples.append(MetricSample(
                name=self.name,
                labels=labels,
                value=value,
            ))
        return samples


class Gauge:
    """Prometheus-compatible gauge."""

    def __init__(self, name: str, description: str, labels: Optional[List[str]] = None):
        self.name = name
        self.description = description
        self.label_names = labels or []
        self._values: Dict[tuple, float] = {}

    def set(self, value: float, **labels) -> None:
        """Set the gauge value."""
        key = self._label_key(labels)
        self._values[key] = value

    def inc(self, amount: float = 1, **labels) -> None:
        """Increment the gauge."""
        key = self._label_key(labels)
        self._values[key] = self._values.get(key, 0) + amount

    def dec(self, amount: float = 1, **labels) -> None:
        """Decrement the gauge."""
        key = self._label_key(labels)
        self._values[key] = self._values.get(key, 0) - amount

    def _label_key(self, labels: Dict[str, str]) -> tuple:
        return tuple(sorted(labels.items()))

    def collect(self) -> List[MetricSample]:
        samples = []
        for key, value in self._values.items():
            labels = dict(key)
            samples.append(MetricSample(
                name=self.name,
                labels=labels,
                value=value,
            ))
        return samples


class Histogram:
    """Prometheus-compatible histogram."""

    DEFAULT_BUCKETS = (
        0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0,
        2.5, 5.0, 7.5, 10.0, float("inf")
    )

    def __init__(
        self,
        name: str,
        description: str,
        labels: Optional[List[str]] = None,
        buckets: Optional[Tuple[float, ...]] = None,
    ):
        self.name = name
        self.description = description
        self.label_names = labels or []
        self.buckets = buckets or self.DEFAULT_BUCKETS

        self._bucket_counts: Dict[tuple, Dict[float, int]] = defaultdict(
            lambda: {b: 0 for b in self.buckets}
        )
        self._sums: Dict[tuple, float] = defaultdict(float)
        self._counts: Dict[tuple, int] = defaultdict(int)

    def observe(self, value: float, **labels) -> None:
        """Observe a value."""
        key = self._label_key(labels)
        self._sums[key] += value
        self._counts[key] += 1

        for bucket in self.buckets:
            if value <= bucket:
                self._bucket_counts[key][bucket] += 1

    @contextmanager
    def time(self, **labels):
        """Context manager to time operations."""
        start = time.time()
        try:
            yield
        finally:
            self.observe(time.time() - start, **labels)

    def _label_key(self, labels: Dict[str, str]) -> tuple:
        return tuple(sorted(labels.items()))

    def collect(self) -> List[MetricSample]:
        samples = []

        for key in set(self._sums.keys()) | set(self._counts.keys()):
            labels = dict(key)

            # Bucket samples
            for bucket, count in self._bucket_counts[key].items():
                bucket_labels = {**labels, "le": str(bucket)}
                samples.append(MetricSample(
                    name=f"{self.name}_bucket",
                    labels=bucket_labels,
                    value=count,
                ))

            # Sum
            samples.append(MetricSample(
                name=f"{self.name}_sum",
                labels=labels,
                value=self._sums[key],
            ))

            # Count
            samples.append(MetricSample(
                name=f"{self.name}_count",
                labels=labels,
                value=self._counts[key],
            ))

        return samples


class Summary:
    """Prometheus-compatible summary with quantiles."""

    DEFAULT_QUANTILES = (0.5, 0.9, 0.95, 0.99)

    def __init__(
        self,
        name: str,
        description: str,
        labels: Optional[List[str]] = None,
        quantiles: Optional[Tuple[float, ...]] = None,
        max_age_seconds: float = 600,
    ):
        self.name = name
        self.description = description
        self.label_names = labels or []
        self.quantiles = quantiles or self.DEFAULT_QUANTILES
        self.max_age_seconds = max_age_seconds

        self._observations: Dict[tuple, List[Tuple[float, float]]] = defaultdict(list)
        self._sums: Dict[tuple, float] = defaultdict(float)
        self._counts: Dict[tuple, int] = defaultdict(int)

    def observe(self, value: float, **labels) -> None:
        """Observe a value."""
        key = self._label_key(labels)
        now = time.time()

        self._observations[key].append((now, value))
        self._sums[key] += value
        self._counts[key] += 1

        # Clean old observations
        cutoff = now - self.max_age_seconds
        self._observations[key] = [
            (t, v) for t, v in self._observations[key]
            if t >= cutoff
        ]

    def _label_key(self, labels: Dict[str, str]) -> tuple:
        return tuple(sorted(labels.items()))

    def _calculate_quantile(self, values: List[float], q: float) -> float:
        """Calculate a quantile."""
        if not values:
            return 0
        sorted_values = sorted(values)
        idx = int(q * len(sorted_values))
        return sorted_values[min(idx, len(sorted_values) - 1)]

    def collect(self) -> List[MetricSample]:
        samples = []

        for key in set(self._sums.keys()) | set(self._counts.keys()):
            labels = dict(key)
            values = [v for _, v in self._observations[key]]

            # Quantile samples
            for quantile in self.quantiles:
                q_value = self._calculate_quantile(values, quantile)
                q_labels = {**labels, "quantile": str(quantile)}
                samples.append(MetricSample(
                    name=self.name,
                    labels=q_labels,
                    value=q_value,
                ))

            # Sum
            samples.append(MetricSample(
                name=f"{self.name}_sum",
                labels=labels,
                value=self._sums[key],
            ))

            # Count
            samples.append(MetricSample(
                name=f"{self.name}_count",
                labels=labels,
                value=self._counts[key],
            ))

        return samples


class MetricsRegistry:
    """
    Registry for Prometheus-compatible metrics.

    Features:
    - Metric registration
    - Prometheus exposition format
    - OpenMetrics format
    """

    def __init__(self, namespace: str = "aion"):
        self.namespace = namespace
        self._metrics: Dict[str, Union[Counter, Gauge, Histogram, Summary]] = {}

    def counter(
        self,
        name: str,
        description: str,
        labels: Optional[List[str]] = None,
    ) -> Counter:
        """Create and register a counter."""
        full_name = f"{self.namespace}_{name}"
        counter = Counter(full_name, description, labels)
        self._metrics[full_name] = counter
        return counter

    def gauge(
        self,
        name: str,
        description: str,
        labels: Optional[List[str]] = None,
    ) -> Gauge:
        """Create and register a gauge."""
        full_name = f"{self.namespace}_{name}"
        gauge = Gauge(full_name, description, labels)
        self._metrics[full_name] = gauge
        return gauge

    def histogram(
        self,
        name: str,
        description: str,
        labels: Optional[List[str]] = None,
        buckets: Optional[Tuple[float, ...]] = None,
    ) -> Histogram:
        """Create and register a histogram."""
        full_name = f"{self.namespace}_{name}"
        histogram = Histogram(full_name, description, labels, buckets)
        self._metrics[full_name] = histogram
        return histogram

    def summary(
        self,
        name: str,
        description: str,
        labels: Optional[List[str]] = None,
        quantiles: Optional[Tuple[float, ...]] = None,
    ) -> Summary:
        """Create and register a summary."""
        full_name = f"{self.namespace}_{name}"
        summary = Summary(full_name, description, labels, quantiles)
        self._metrics[full_name] = summary
        return summary

    def collect_all(self) -> List[MetricSample]:
        """Collect all metric samples."""
        samples = []
        for metric in self._metrics.values():
            samples.extend(metric.collect())
        return samples

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus exposition format."""
        lines = []
        samples = self.collect_all()

        # Group by metric name
        by_name: Dict[str, List[MetricSample]] = defaultdict(list)
        for sample in samples:
            base_name = sample.name.rsplit("_", 1)[0] if "_" in sample.name else sample.name
            by_name[sample.name].append(sample)

        for name, metric_samples in by_name.items():
            for sample in metric_samples:
                if sample.labels:
                    label_str = ",".join(f'{k}="{v}"' for k, v in sample.labels.items())
                    lines.append(f"{sample.name}{{{label_str}}} {sample.value}")
                else:
                    lines.append(f"{sample.name} {sample.value}")

        return "\n".join(lines) + "\n"


# === Pre-configured AION Metrics ===

class AIONMetrics:
    """Pre-configured metrics for AION process manager."""

    def __init__(self, registry: Optional[MetricsRegistry] = None):
        self.registry = registry or MetricsRegistry()

        # Process metrics
        self.processes_total = self.registry.counter(
            "processes_total",
            "Total number of processes spawned",
            ["type", "status"],
        )
        self.processes_active = self.registry.gauge(
            "processes_active",
            "Number of active processes",
            ["type", "state"],
        )
        self.process_duration = self.registry.histogram(
            "process_duration_seconds",
            "Process execution duration",
            ["type"],
        )

        # Task metrics
        self.tasks_total = self.registry.counter(
            "tasks_total",
            "Total number of tasks executed",
            ["handler", "status"],
        )
        self.tasks_active = self.registry.gauge(
            "tasks_active",
            "Number of active tasks",
        )
        self.task_duration = self.registry.histogram(
            "task_duration_seconds",
            "Task execution duration",
            ["handler"],
        )

        # Event bus metrics
        self.events_total = self.registry.counter(
            "events_total",
            "Total events emitted",
            ["type"],
        )
        self.events_delivered = self.registry.counter(
            "events_delivered",
            "Events successfully delivered",
            ["type"],
        )
        self.event_latency = self.registry.histogram(
            "event_latency_seconds",
            "Event delivery latency",
        )

        # Resource metrics
        self.memory_usage = self.registry.gauge(
            "memory_usage_bytes",
            "Memory usage in bytes",
            ["process"],
        )
        self.cpu_usage = self.registry.gauge(
            "cpu_usage_percent",
            "CPU usage percentage",
            ["process"],
        )
        self.token_usage = self.registry.counter(
            "token_usage_total",
            "Total tokens consumed",
            ["process", "type"],
        )

        # Cluster metrics
        self.cluster_nodes = self.registry.gauge(
            "cluster_nodes",
            "Number of cluster nodes",
            ["state"],
        )
        self.cluster_leader = self.registry.gauge(
            "cluster_is_leader",
            "Whether this node is the leader",
        )

    def export(self) -> str:
        """Export all metrics in Prometheus format."""
        return self.registry.export_prometheus()


# === Global Instances ===

_default_tracer: Optional[Tracer] = None
_default_metrics: Optional[AIONMetrics] = None


def get_tracer() -> Tracer:
    """Get the default tracer."""
    global _default_tracer
    if _default_tracer is None:
        _default_tracer = Tracer()
    return _default_tracer


def get_metrics() -> AIONMetrics:
    """Get the default metrics."""
    global _default_metrics
    if _default_metrics is None:
        _default_metrics = AIONMetrics()
    return _default_metrics


def set_tracer(tracer: Tracer) -> None:
    """Set the default tracer."""
    global _default_tracer
    _default_tracer = tracer


def set_metrics(metrics: AIONMetrics) -> None:
    """Set the default metrics."""
    global _default_metrics
    _default_metrics = metrics
