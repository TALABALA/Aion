"""
AION Observability - True SOTA Implementation

Production-grade observability with:
- Full OpenTelemetry instrumentation
- Distributed trace context propagation
- Structured logging with trace correlation
- Custom metrics with dimensional labels
- Automatic span creation for workflows
- Baggage propagation for context
- Exemplars linking metrics to traces
"""

from __future__ import annotations

import asyncio
import functools
import json
import time
import uuid
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    TypeVar,
    Union,
)

import structlog

logger = structlog.get_logger(__name__)


# =============================================================================
# Trace Context
# =============================================================================


@dataclass
class TraceContext:
    """
    Distributed trace context for propagation.

    Compatible with W3C Trace Context specification.
    """
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    trace_flags: int = 1  # 1 = sampled
    trace_state: Dict[str, str] = field(default_factory=dict)

    # Baggage (arbitrary key-value pairs)
    baggage: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if not self.trace_id:
            self.trace_id = uuid.uuid4().hex
        if not self.span_id:
            self.span_id = uuid.uuid4().hex[:16]

    @property
    def is_sampled(self) -> bool:
        return bool(self.trace_flags & 1)

    def create_child(self) -> "TraceContext":
        """Create a child span context."""
        return TraceContext(
            trace_id=self.trace_id,
            span_id=uuid.uuid4().hex[:16],
            parent_span_id=self.span_id,
            trace_flags=self.trace_flags,
            trace_state=self.trace_state.copy(),
            baggage=self.baggage.copy(),
        )

    def to_traceparent(self) -> str:
        """Serialize to W3C traceparent header."""
        return f"00-{self.trace_id}-{self.span_id}-{self.trace_flags:02x}"

    def to_tracestate(self) -> str:
        """Serialize to W3C tracestate header."""
        return ",".join(f"{k}={v}" for k, v in self.trace_state.items())

    def to_baggage(self) -> str:
        """Serialize baggage header."""
        return ",".join(f"{k}={v}" for k, v in self.baggage.items())

    @classmethod
    def from_traceparent(cls, header: str) -> Optional["TraceContext"]:
        """Parse from W3C traceparent header."""
        try:
            parts = header.split("-")
            if len(parts) != 4 or parts[0] != "00":
                return None

            return cls(
                trace_id=parts[1],
                span_id=parts[2],
                trace_flags=int(parts[3], 16),
            )
        except Exception:
            return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "trace_flags": self.trace_flags,
            "trace_state": self.trace_state,
            "baggage": self.baggage,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TraceContext":
        return cls(
            trace_id=data.get("trace_id", ""),
            span_id=data.get("span_id", ""),
            parent_span_id=data.get("parent_span_id"),
            trace_flags=data.get("trace_flags", 1),
            trace_state=data.get("trace_state", {}),
            baggage=data.get("baggage", {}),
        )

    @classmethod
    def extract_from_task(cls, task_metadata: Dict[str, Any]) -> Optional["TraceContext"]:
        """Extract trace context from task metadata."""
        if "trace_id" in task_metadata:
            return cls.from_dict(task_metadata)
        return None

    def inject_into_task(self, task_metadata: Dict[str, Any]) -> None:
        """Inject trace context into task metadata."""
        task_metadata.update(self.to_dict())


# Context variable for current trace
_current_context: Optional[TraceContext] = None


def get_current_context() -> Optional[TraceContext]:
    """Get current trace context."""
    return _current_context


def set_current_context(ctx: Optional[TraceContext]) -> None:
    """Set current trace context."""
    global _current_context
    _current_context = ctx


# =============================================================================
# Span
# =============================================================================


class SpanKind(str, Enum):
    """Types of spans."""
    INTERNAL = "internal"
    SERVER = "server"
    CLIENT = "client"
    PRODUCER = "producer"
    CONSUMER = "consumer"


class SpanStatus(str, Enum):
    """Span status codes."""
    UNSET = "unset"
    OK = "ok"
    ERROR = "error"


@dataclass
class SpanEvent:
    """Event recorded during a span."""
    name: str
    timestamp: datetime
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SpanLink:
    """Link to another span."""
    trace_id: str
    span_id: str
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Span:
    """
    A span representing a unit of work.

    Follows OpenTelemetry span model.
    """
    name: str
    context: TraceContext
    kind: SpanKind = SpanKind.INTERNAL

    # Timing
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None

    # Attributes
    attributes: Dict[str, Any] = field(default_factory=dict)

    # Status
    status: SpanStatus = SpanStatus.UNSET
    status_message: Optional[str] = None

    # Events
    events: List[SpanEvent] = field(default_factory=list)

    # Links
    links: List[SpanLink] = field(default_factory=list)

    # Resource
    resource: Dict[str, Any] = field(default_factory=dict)

    def set_attribute(self, key: str, value: Any) -> None:
        """Set a span attribute."""
        self.attributes[key] = value

    def set_attributes(self, attributes: Dict[str, Any]) -> None:
        """Set multiple attributes."""
        self.attributes.update(attributes)

    def add_event(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add an event to the span."""
        self.events.append(SpanEvent(
            name=name,
            timestamp=datetime.now(),
            attributes=attributes or {},
        ))

    def add_link(
        self,
        trace_id: str,
        span_id: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a link to another span."""
        self.links.append(SpanLink(
            trace_id=trace_id,
            span_id=span_id,
            attributes=attributes or {},
        ))

    def set_status(self, status: SpanStatus, message: Optional[str] = None) -> None:
        """Set span status."""
        self.status = status
        self.status_message = message

    def record_exception(
        self,
        exception: Exception,
        escaped: bool = True,
    ) -> None:
        """Record an exception as an event."""
        self.add_event(
            "exception",
            {
                "exception.type": type(exception).__name__,
                "exception.message": str(exception),
                "exception.escaped": escaped,
            },
        )
        self.set_status(SpanStatus.ERROR, str(exception))

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
        return {
            "name": self.name,
            "trace_id": self.context.trace_id,
            "span_id": self.context.span_id,
            "parent_span_id": self.context.parent_span_id,
            "kind": self.kind.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "attributes": self.attributes,
            "status": self.status.value,
            "status_message": self.status_message,
            "events": [
                {
                    "name": e.name,
                    "timestamp": e.timestamp.isoformat(),
                    "attributes": e.attributes,
                }
                for e in self.events
            ],
            "links": [
                {
                    "trace_id": l.trace_id,
                    "span_id": l.span_id,
                    "attributes": l.attributes,
                }
                for l in self.links
            ],
            "resource": self.resource,
        }


# =============================================================================
# Span Exporter Interface
# =============================================================================


class SpanExporter:
    """Base class for span exporters."""

    async def export(self, spans: List[Span]) -> bool:
        """Export spans. Returns True on success."""
        raise NotImplementedError

    async def shutdown(self) -> None:
        """Shutdown the exporter."""
        pass


class ConsoleSpanExporter(SpanExporter):
    """Export spans to console (for debugging)."""

    async def export(self, spans: List[Span]) -> bool:
        for span in spans:
            print(f"[SPAN] {span.name}")
            print(f"  trace_id: {span.context.trace_id}")
            print(f"  span_id: {span.context.span_id}")
            print(f"  duration: {span.duration_ms:.2f}ms")
            print(f"  status: {span.status.value}")
            if span.attributes:
                print(f"  attributes: {span.attributes}")
        return True


class OTLPSpanExporter(SpanExporter):
    """Export spans via OTLP protocol."""

    def __init__(
        self,
        endpoint: str = "http://localhost:4317",
        headers: Optional[Dict[str, str]] = None,
    ):
        self.endpoint = endpoint
        self.headers = headers or {}
        self._client = None

    async def export(self, spans: List[Span]) -> bool:
        try:
            # Convert to OTLP format and send
            # This is a simplified implementation
            import aiohttp

            async with aiohttp.ClientSession() as session:
                payload = {
                    "resourceSpans": [
                        {
                            "scopeSpans": [
                                {
                                    "spans": [s.to_dict() for s in spans]
                                }
                            ]
                        }
                    ]
                }

                async with session.post(
                    f"{self.endpoint}/v1/traces",
                    json=payload,
                    headers=self.headers,
                ) as response:
                    return response.status == 200

        except Exception as e:
            logger.error("OTLP export failed", error=str(e))
            return False


# =============================================================================
# Tracer
# =============================================================================


class TracerProvider:
    """
    Provider for creating tracers.

    Manages span processing and export.
    """

    def __init__(
        self,
        service_name: str,
        service_version: str = "1.0.0",
        environment: str = "development",
        exporters: Optional[List[SpanExporter]] = None,
        sample_rate: float = 1.0,
    ):
        self.service_name = service_name
        self.service_version = service_version
        self.environment = environment
        self.exporters = exporters or []
        self.sample_rate = sample_rate

        # Resource attributes
        self.resource = {
            "service.name": service_name,
            "service.version": service_version,
            "deployment.environment": environment,
        }

        # Span buffer for batching
        self._span_buffer: List[Span] = []
        self._buffer_lock = asyncio.Lock()
        self._export_task: Optional[asyncio.Task] = None

        self._shutdown = False

    async def start(self) -> None:
        """Start the tracer provider."""
        self._export_task = asyncio.create_task(self._export_loop())
        logger.info("Tracer provider started", service=self.service_name)

    async def shutdown(self) -> None:
        """Shutdown and flush remaining spans."""
        self._shutdown = True

        if self._export_task:
            self._export_task.cancel()
            try:
                await self._export_task
            except asyncio.CancelledError:
                pass

        # Final flush
        await self._flush()

        for exporter in self.exporters:
            await exporter.shutdown()

        logger.info("Tracer provider shutdown")

    def get_tracer(self, name: str) -> "Tracer":
        """Get a tracer instance."""
        return Tracer(self, name)

    async def _record_span(self, span: Span) -> None:
        """Record a completed span."""
        span.resource = self.resource

        async with self._buffer_lock:
            self._span_buffer.append(span)

    async def _export_loop(self) -> None:
        """Background loop to export spans."""
        while not self._shutdown:
            try:
                await asyncio.sleep(5)  # Export every 5 seconds
                await self._flush()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Export loop error", error=str(e))

    async def _flush(self) -> None:
        """Flush buffered spans to exporters."""
        async with self._buffer_lock:
            if not self._span_buffer:
                return

            spans = self._span_buffer.copy()
            self._span_buffer.clear()

        for exporter in self.exporters:
            try:
                await exporter.export(spans)
            except Exception as e:
                logger.error("Span export failed", error=str(e))


class Tracer:
    """
    Creates and manages spans.
    """

    def __init__(self, provider: TracerProvider, name: str):
        self.provider = provider
        self.name = name

    def start_span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        parent: Optional[TraceContext] = None,
        attributes: Optional[Dict[str, Any]] = None,
        links: Optional[List[SpanLink]] = None,
    ) -> Span:
        """Start a new span."""
        # Get or create context
        if parent:
            context = parent.create_child()
        elif get_current_context():
            context = get_current_context().create_child()
        else:
            context = TraceContext()

        span = Span(
            name=name,
            context=context,
            kind=kind,
            attributes=attributes or {},
            links=links or [],
        )

        span.set_attribute("otel.library.name", self.name)

        return span

    @asynccontextmanager
    async def span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Optional[Dict[str, Any]] = None,
    ):
        """Context manager for automatic span management."""
        span = self.start_span(name, kind, attributes=attributes)
        previous_context = get_current_context()
        set_current_context(span.context)

        try:
            yield span
            if span.status == SpanStatus.UNSET:
                span.set_status(SpanStatus.OK)
        except Exception as e:
            span.record_exception(e)
            raise
        finally:
            span.end()
            set_current_context(previous_context)
            await self.provider._record_span(span)

    @contextmanager
    def span_sync(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Iterator[Span]:
        """Synchronous context manager for spans."""
        span = self.start_span(name, kind, attributes=attributes)
        previous_context = get_current_context()
        set_current_context(span.context)

        try:
            yield span
            if span.status == SpanStatus.UNSET:
                span.set_status(SpanStatus.OK)
        except Exception as e:
            span.record_exception(e)
            raise
        finally:
            span.end()
            set_current_context(previous_context)
            # For sync, we can't await, so we schedule
            asyncio.create_task(self.provider._record_span(span))


# =============================================================================
# Metrics
# =============================================================================


class MetricKind(str, Enum):
    """Types of metrics."""
    COUNTER = "counter"
    UP_DOWN_COUNTER = "up_down_counter"
    HISTOGRAM = "histogram"
    GAUGE = "gauge"


@dataclass
class MetricPoint:
    """A single metric measurement."""
    timestamp: datetime
    value: float
    attributes: Dict[str, str] = field(default_factory=dict)
    exemplar_trace_id: Optional[str] = None
    exemplar_span_id: Optional[str] = None


class Metric:
    """Base metric class."""

    def __init__(
        self,
        name: str,
        description: str,
        unit: str = "",
        kind: MetricKind = MetricKind.COUNTER,
    ):
        self.name = name
        self.description = description
        self.unit = unit
        self.kind = kind
        self._points: List[MetricPoint] = []
        self._lock = asyncio.Lock()

    async def _record(
        self,
        value: float,
        attributes: Optional[Dict[str, str]] = None,
    ) -> None:
        """Record a metric point with optional trace exemplar."""
        ctx = get_current_context()

        point = MetricPoint(
            timestamp=datetime.now(),
            value=value,
            attributes=attributes or {},
            exemplar_trace_id=ctx.trace_id if ctx else None,
            exemplar_span_id=ctx.span_id if ctx else None,
        )

        async with self._lock:
            self._points.append(point)

    async def get_points(self) -> List[MetricPoint]:
        """Get and clear recorded points."""
        async with self._lock:
            points = self._points.copy()
            self._points.clear()
            return points


class Counter(Metric):
    """Monotonic counter metric."""

    def __init__(self, name: str, description: str, unit: str = ""):
        super().__init__(name, description, unit, MetricKind.COUNTER)

    async def add(
        self,
        value: float,
        attributes: Optional[Dict[str, str]] = None,
    ) -> None:
        """Add to the counter."""
        if value < 0:
            raise ValueError("Counter can only be incremented")
        await self._record(value, attributes)


class Histogram(Metric):
    """Histogram metric for distributions."""

    def __init__(
        self,
        name: str,
        description: str,
        unit: str = "",
        boundaries: Optional[List[float]] = None,
    ):
        super().__init__(name, description, unit, MetricKind.HISTOGRAM)
        self.boundaries = boundaries or [
            0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0,
            2.5, 5.0, 7.5, 10.0
        ]

    async def record(
        self,
        value: float,
        attributes: Optional[Dict[str, str]] = None,
    ) -> None:
        """Record a value in the histogram."""
        await self._record(value, attributes)


class Gauge(Metric):
    """Gauge metric for current values."""

    def __init__(self, name: str, description: str, unit: str = ""):
        super().__init__(name, description, unit, MetricKind.GAUGE)

    async def set(
        self,
        value: float,
        attributes: Optional[Dict[str, str]] = None,
    ) -> None:
        """Set the gauge value."""
        await self._record(value, attributes)


# =============================================================================
# Meter Provider
# =============================================================================


class MeterProvider:
    """
    Provider for creating meters.
    """

    def __init__(self, service_name: str):
        self.service_name = service_name
        self._metrics: Dict[str, Metric] = {}

    def get_meter(self, name: str) -> "Meter":
        """Get a meter instance."""
        return Meter(self, name)

    def _register_metric(self, metric: Metric) -> None:
        """Register a metric."""
        self._metrics[metric.name] = metric

    def get_all_metrics(self) -> Dict[str, Metric]:
        """Get all registered metrics."""
        return self._metrics.copy()


class Meter:
    """Creates metrics."""

    def __init__(self, provider: MeterProvider, name: str):
        self.provider = provider
        self.name = name

    def create_counter(
        self,
        name: str,
        description: str = "",
        unit: str = "",
    ) -> Counter:
        """Create a counter metric."""
        counter = Counter(name, description, unit)
        self.provider._register_metric(counter)
        return counter

    def create_histogram(
        self,
        name: str,
        description: str = "",
        unit: str = "",
        boundaries: Optional[List[float]] = None,
    ) -> Histogram:
        """Create a histogram metric."""
        histogram = Histogram(name, description, unit, boundaries)
        self.provider._register_metric(histogram)
        return histogram

    def create_gauge(
        self,
        name: str,
        description: str = "",
        unit: str = "",
    ) -> Gauge:
        """Create a gauge metric."""
        gauge = Gauge(name, description, unit)
        self.provider._register_metric(gauge)
        return gauge


# =============================================================================
# Workflow Instrumentation
# =============================================================================


class WorkflowInstrumentation:
    """
    Automatic instrumentation for workflow execution.

    Provides pre-built spans and metrics for:
    - Workflow execution
    - Step execution
    - Action invocation
    - Approval handling
    """

    def __init__(
        self,
        tracer: Tracer,
        meter: Meter,
    ):
        self.tracer = tracer
        self.meter = meter

        # Pre-create metrics
        self.workflow_executions = meter.create_counter(
            "aion.workflow.executions",
            "Total workflow executions",
        )
        self.workflow_duration = meter.create_histogram(
            "aion.workflow.duration",
            "Workflow execution duration",
            "ms",
        )
        self.step_executions = meter.create_counter(
            "aion.step.executions",
            "Total step executions",
        )
        self.step_duration = meter.create_histogram(
            "aion.step.duration",
            "Step execution duration",
            "ms",
        )
        self.active_workflows = meter.create_gauge(
            "aion.workflow.active",
            "Currently active workflows",
        )
        self.queue_depth = meter.create_gauge(
            "aion.queue.depth",
            "Task queue depth",
        )

    @asynccontextmanager
    async def workflow_span(
        self,
        workflow_id: str,
        workflow_name: str,
        execution_id: str,
        inputs: Optional[Dict[str, Any]] = None,
    ):
        """Create an instrumented workflow span."""
        async with self.tracer.span(
            f"workflow:{workflow_name}",
            kind=SpanKind.INTERNAL,
            attributes={
                "aion.workflow.id": workflow_id,
                "aion.workflow.name": workflow_name,
                "aion.execution.id": execution_id,
            },
        ) as span:
            start_time = time.time()

            # Record start
            await self.workflow_executions.add(1, {
                "workflow_name": workflow_name,
                "status": "started",
            })

            try:
                yield span

                # Record success
                duration = (time.time() - start_time) * 1000
                await self.workflow_duration.record(duration, {
                    "workflow_name": workflow_name,
                    "status": "success",
                })
                await self.workflow_executions.add(1, {
                    "workflow_name": workflow_name,
                    "status": "completed",
                })

            except Exception as e:
                # Record failure
                duration = (time.time() - start_time) * 1000
                await self.workflow_duration.record(duration, {
                    "workflow_name": workflow_name,
                    "status": "error",
                })
                await self.workflow_executions.add(1, {
                    "workflow_name": workflow_name,
                    "status": "failed",
                })
                raise

    @asynccontextmanager
    async def step_span(
        self,
        step_id: str,
        step_name: str,
        step_type: str,
    ):
        """Create an instrumented step span."""
        async with self.tracer.span(
            f"step:{step_name}",
            kind=SpanKind.INTERNAL,
            attributes={
                "aion.step.id": step_id,
                "aion.step.name": step_name,
                "aion.step.type": step_type,
            },
        ) as span:
            start_time = time.time()

            await self.step_executions.add(1, {
                "step_name": step_name,
                "step_type": step_type,
                "status": "started",
            })

            try:
                yield span

                duration = (time.time() - start_time) * 1000
                await self.step_duration.record(duration, {
                    "step_name": step_name,
                    "step_type": step_type,
                    "status": "success",
                })

            except Exception as e:
                duration = (time.time() - start_time) * 1000
                await self.step_duration.record(duration, {
                    "step_name": step_name,
                    "step_type": step_type,
                    "status": "error",
                })
                raise


# =============================================================================
# Structured Logging with Trace Correlation
# =============================================================================


def configure_logging_with_traces(service_name: str) -> None:
    """Configure structlog with trace correlation."""

    def add_trace_context(
        logger: Any,
        method_name: str,
        event_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Add trace context to log entries."""
        ctx = get_current_context()
        if ctx:
            event_dict["trace_id"] = ctx.trace_id
            event_dict["span_id"] = ctx.span_id
            if ctx.parent_span_id:
                event_dict["parent_span_id"] = ctx.parent_span_id

        event_dict["service"] = service_name
        return event_dict

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            add_trace_context,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


# =============================================================================
# Decorators for Easy Instrumentation
# =============================================================================


def trace_async(
    name: Optional[str] = None,
    kind: SpanKind = SpanKind.INTERNAL,
    attributes: Optional[Dict[str, Any]] = None,
):
    """Decorator to trace async functions."""
    def decorator(func):
        span_name = name or func.__name__

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Get tracer from first arg if it has one
            tracer = None
            if args and hasattr(args[0], 'tracer'):
                tracer = args[0].tracer
            elif args and hasattr(args[0], '_tracer'):
                tracer = args[0]._tracer

            if tracer:
                async with tracer.span(span_name, kind, attributes):
                    return await func(*args, **kwargs)
            else:
                return await func(*args, **kwargs)

        return wrapper
    return decorator


def trace_sync(
    name: Optional[str] = None,
    kind: SpanKind = SpanKind.INTERNAL,
    attributes: Optional[Dict[str, Any]] = None,
):
    """Decorator to trace sync functions."""
    def decorator(func):
        span_name = name or func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tracer = None
            if args and hasattr(args[0], 'tracer'):
                tracer = args[0].tracer

            if tracer:
                with tracer.span_sync(span_name, kind, attributes):
                    return func(*args, **kwargs)
            else:
                return func(*args, **kwargs)

        return wrapper
    return decorator


# =============================================================================
# Factory Functions
# =============================================================================


async def create_observability(
    service_name: str,
    service_version: str = "1.0.0",
    environment: str = "development",
    otlp_endpoint: Optional[str] = None,
) -> Tuple[TracerProvider, MeterProvider, WorkflowInstrumentation]:
    """Create complete observability setup."""
    # Configure logging
    configure_logging_with_traces(service_name)

    # Create exporters
    exporters = [ConsoleSpanExporter()]
    if otlp_endpoint:
        exporters.append(OTLPSpanExporter(endpoint=otlp_endpoint))

    # Create providers
    tracer_provider = TracerProvider(
        service_name=service_name,
        service_version=service_version,
        environment=environment,
        exporters=exporters,
    )
    await tracer_provider.start()

    meter_provider = MeterProvider(service_name)

    # Create instrumentation
    tracer = tracer_provider.get_tracer("aion.workflow")
    meter = meter_provider.get_meter("aion.workflow")
    instrumentation = WorkflowInstrumentation(tracer, meter)

    return tracer_provider, meter_provider, instrumentation
