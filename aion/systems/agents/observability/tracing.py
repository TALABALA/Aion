"""
Distributed Tracing

OpenTelemetry-compatible distributed tracing for agent operations.
"""

import asyncio
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, AsyncIterator, Optional
import structlog

logger = structlog.get_logger()


class SpanKind(str, Enum):
    """Type of span."""

    INTERNAL = "internal"
    CLIENT = "client"
    SERVER = "server"
    PRODUCER = "producer"
    CONSUMER = "consumer"


class SpanStatus(str, Enum):
    """Status of a span."""

    UNSET = "unset"
    OK = "ok"
    ERROR = "error"


@dataclass
class SpanContext:
    """Context for trace propagation."""

    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    baggage: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for propagation."""
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "baggage": self.baggage,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SpanContext":
        """Create from dictionary."""
        return cls(
            trace_id=data["trace_id"],
            span_id=data["span_id"],
            parent_span_id=data.get("parent_span_id"),
            baggage=data.get("baggage", {}),
        )


@dataclass
class SpanEvent:
    """An event within a span."""

    name: str
    timestamp: datetime = field(default_factory=datetime.now)
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class Span:
    """A trace span representing a unit of work."""

    name: str
    context: SpanContext
    kind: SpanKind = SpanKind.INTERNAL
    status: SpanStatus = SpanStatus.UNSET
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[SpanEvent] = field(default_factory=list)
    links: list[SpanContext] = field(default_factory=list)

    def set_attribute(self, key: str, value: Any) -> None:
        """Set a span attribute."""
        self.attributes[key] = value

    def add_event(
        self,
        name: str,
        attributes: Optional[dict[str, Any]] = None,
    ) -> None:
        """Add an event to the span."""
        self.events.append(SpanEvent(
            name=name,
            attributes=attributes or {},
        ))

    def set_status(self, status: SpanStatus, description: str = "") -> None:
        """Set span status."""
        self.status = status
        if description:
            self.attributes["status_description"] = description

    def end(self) -> None:
        """End the span."""
        if not self.end_time:
            self.end_time = datetime.now()

    @property
    def duration_ms(self) -> float:
        """Get span duration in milliseconds."""
        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds() * 1000

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "trace_id": self.context.trace_id,
            "span_id": self.context.span_id,
            "parent_span_id": self.context.parent_span_id,
            "kind": self.kind.value,
            "status": self.status.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "attributes": self.attributes,
            "events": [
                {
                    "name": e.name,
                    "timestamp": e.timestamp.isoformat(),
                    "attributes": e.attributes,
                }
                for e in self.events
            ],
        }


class TraceExporter:
    """Base class for trace exporters."""

    async def export(self, spans: list[Span]) -> bool:
        """Export spans to backend."""
        raise NotImplementedError

    async def shutdown(self) -> None:
        """Shutdown exporter."""
        pass


class ConsoleExporter(TraceExporter):
    """Export traces to console."""

    async def export(self, spans: list[Span]) -> bool:
        """Export spans to console."""
        for span in spans:
            logger.info(
                "trace_span",
                name=span.name,
                trace_id=span.context.trace_id,
                duration_ms=span.duration_ms,
                status=span.status.value,
            )
        return True


class InMemoryExporter(TraceExporter):
    """Store traces in memory."""

    def __init__(self, max_spans: int = 10000):
        self.max_spans = max_spans
        self.spans: list[Span] = []

    async def export(self, spans: list[Span]) -> bool:
        """Store spans in memory."""
        self.spans.extend(spans)

        # Trim if needed
        if len(self.spans) > self.max_spans:
            self.spans = self.spans[-self.max_spans:]

        return True

    def get_trace(self, trace_id: str) -> list[Span]:
        """Get all spans for a trace."""
        return [s for s in self.spans if s.context.trace_id == trace_id]

    def get_recent(self, limit: int = 100) -> list[Span]:
        """Get recent spans."""
        return self.spans[-limit:]

    def clear(self) -> None:
        """Clear stored spans."""
        self.spans.clear()


class Tracer:
    """
    Creates and manages spans.

    Features:
    - Span creation and management
    - Context propagation
    - Automatic parent-child relationships
    - Async context manager support
    """

    def __init__(
        self,
        service_name: str,
        exporter: Optional[TraceExporter] = None,
    ):
        self.service_name = service_name
        self.exporter = exporter or ConsoleExporter()
        self._current_span: Optional[Span] = None
        self._span_stack: list[Span] = []
        self._completed_spans: list[Span] = []
        self._batch_size = 100
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize tracer."""
        self._initialized = True
        logger.info("tracer_initialized", service=self.service_name)

    async def shutdown(self) -> None:
        """Shutdown tracer and flush pending spans."""
        if self._completed_spans:
            await self.exporter.export(self._completed_spans)
            self._completed_spans.clear()

        await self.exporter.shutdown()
        self._initialized = False

    def start_span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        parent: Optional[SpanContext] = None,
        attributes: Optional[dict[str, Any]] = None,
    ) -> Span:
        """Start a new span."""
        # Determine parent
        parent_context = parent
        if not parent_context and self._current_span:
            parent_context = self._current_span.context

        # Create span context
        trace_id = parent_context.trace_id if parent_context else str(uuid.uuid4())
        span_id = str(uuid.uuid4())[:16]
        parent_span_id = parent_context.span_id if parent_context else None

        context = SpanContext(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
        )

        span = Span(
            name=name,
            context=context,
            kind=kind,
            attributes=attributes or {},
        )

        # Add service name
        span.set_attribute("service.name", self.service_name)

        # Update current span
        if self._current_span:
            self._span_stack.append(self._current_span)
        self._current_span = span

        return span

    def end_span(self, span: Span) -> None:
        """End a span."""
        span.end()

        # Add to completed
        self._completed_spans.append(span)

        # Export if batch is full
        if len(self._completed_spans) >= self._batch_size:
            asyncio.create_task(self._flush())

        # Update current span
        if self._current_span == span:
            self._current_span = self._span_stack.pop() if self._span_stack else None

    @asynccontextmanager
    async def span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Optional[dict[str, Any]] = None,
    ) -> AsyncIterator[Span]:
        """Context manager for spans."""
        span = self.start_span(name, kind=kind, attributes=attributes)
        try:
            yield span
            if span.status == SpanStatus.UNSET:
                span.set_status(SpanStatus.OK)
        except Exception as e:
            span.set_status(SpanStatus.ERROR, str(e))
            span.set_attribute("error.type", type(e).__name__)
            span.set_attribute("error.message", str(e))
            raise
        finally:
            self.end_span(span)

    async def _flush(self) -> None:
        """Flush completed spans to exporter."""
        if self._completed_spans:
            spans = self._completed_spans.copy()
            self._completed_spans.clear()
            await self.exporter.export(spans)

    def get_current_span(self) -> Optional[Span]:
        """Get current active span."""
        return self._current_span

    def get_current_context(self) -> Optional[SpanContext]:
        """Get current span context for propagation."""
        return self._current_span.context if self._current_span else None


class TracingManager:
    """
    Manages tracing across multiple agents.

    Features:
    - Multi-tracer management
    - Cross-agent trace correlation
    - Trace analysis and querying
    """

    def __init__(self):
        self._tracers: dict[str, Tracer] = {}
        self._exporter = InMemoryExporter()
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize tracing manager."""
        self._initialized = True
        logger.info("tracing_manager_initialized")

    async def shutdown(self) -> None:
        """Shutdown all tracers."""
        for tracer in self._tracers.values():
            await tracer.shutdown()
        self._initialized = False

    def get_tracer(self, service_name: str) -> Tracer:
        """Get or create tracer for a service."""
        if service_name not in self._tracers:
            tracer = Tracer(service_name, self._exporter)
            self._tracers[service_name] = tracer

        return self._tracers[service_name]

    def get_trace(self, trace_id: str) -> list[dict[str, Any]]:
        """Get all spans for a trace."""
        spans = self._exporter.get_trace(trace_id)
        return [s.to_dict() for s in spans]

    def get_trace_tree(self, trace_id: str) -> dict[str, Any]:
        """Get trace as a hierarchical tree."""
        spans = self._exporter.get_trace(trace_id)

        if not spans:
            return {}

        # Build tree
        span_map = {s.context.span_id: s.to_dict() for s in spans}

        for span_dict in span_map.values():
            span_dict["children"] = []

        roots = []
        for span_dict in span_map.values():
            parent_id = span_dict.get("parent_span_id")
            if parent_id and parent_id in span_map:
                span_map[parent_id]["children"].append(span_dict)
            else:
                roots.append(span_dict)

        return {
            "trace_id": trace_id,
            "spans": roots,
            "total_spans": len(spans),
        }

    def get_recent_traces(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get recent trace summaries."""
        spans = self._exporter.get_recent(limit * 10)

        # Group by trace
        traces: dict[str, list[Span]] = {}
        for span in spans:
            trace_id = span.context.trace_id
            if trace_id not in traces:
                traces[trace_id] = []
            traces[trace_id].append(span)

        # Create summaries
        summaries = []
        for trace_id, trace_spans in list(traces.items())[-limit:]:
            root_spans = [s for s in trace_spans if not s.context.parent_span_id]
            root = root_spans[0] if root_spans else trace_spans[0]

            summaries.append({
                "trace_id": trace_id,
                "name": root.name,
                "span_count": len(trace_spans),
                "duration_ms": sum(s.duration_ms for s in trace_spans),
                "status": root.status.value,
                "start_time": min(s.start_time for s in trace_spans).isoformat(),
            })

        return summaries

    def search_traces(
        self,
        service_name: Optional[str] = None,
        operation_name: Optional[str] = None,
        min_duration_ms: Optional[float] = None,
        status: Optional[SpanStatus] = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Search traces by criteria."""
        spans = self._exporter.spans

        results = []
        for span in spans:
            # Filter by service
            if service_name:
                if span.attributes.get("service.name") != service_name:
                    continue

            # Filter by operation
            if operation_name and operation_name not in span.name:
                continue

            # Filter by duration
            if min_duration_ms and span.duration_ms < min_duration_ms:
                continue

            # Filter by status
            if status and span.status != status:
                continue

            results.append(span.to_dict())

            if len(results) >= limit:
                break

        return results

    def get_stats(self) -> dict[str, Any]:
        """Get tracing statistics."""
        spans = self._exporter.spans

        # Calculate stats
        total_spans = len(spans)
        error_spans = sum(1 for s in spans if s.status == SpanStatus.ERROR)
        avg_duration = sum(s.duration_ms for s in spans) / max(1, total_spans)

        # Group by service
        by_service: dict[str, int] = {}
        for span in spans:
            service = span.attributes.get("service.name", "unknown")
            by_service[service] = by_service.get(service, 0) + 1

        return {
            "total_spans": total_spans,
            "total_traces": len({s.context.trace_id for s in spans}),
            "error_rate": error_spans / max(1, total_spans),
            "avg_duration_ms": avg_duration,
            "spans_by_service": by_service,
            "tracer_count": len(self._tracers),
        }
