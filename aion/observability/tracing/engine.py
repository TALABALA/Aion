"""
AION Tracing Engine

SOTA distributed tracing with:
- OpenTelemetry compatibility
- W3C Trace Context propagation
- Automatic context management
- Multiple exporters (Jaeger, Zipkin, OTLP)
- Tail-based sampling
- Span linking for causality
"""

from __future__ import annotations

import asyncio
import functools
import time
import threading
from contextlib import contextmanager, asynccontextmanager
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Generator, List, Optional, TypeVar, Union
import uuid

import structlog

from aion.observability.types import (
    Span, SpanKind, SpanStatus, SpanEvent, SpanLink, SpanContext,
    Trace, SamplingDecision,
)
from aion.observability.collector import TelemetryCollector
from aion.observability.context import (
    _current_span, get_context_manager, ObservabilityContext,
)

logger = structlog.get_logger(__name__)

T = TypeVar('T')


class TracingEngine:
    """
    SOTA Distributed tracing engine.

    Features:
    - Automatic context propagation
    - Span creation and management
    - Multiple sampling strategies
    - Multiple exporters (Jaeger, Zipkin, OTLP)
    - Span linking for complex workflows
    - Tail-based sampling support
    """

    def __init__(
        self,
        collector: TelemetryCollector,
        service_name: str = "aion",
        service_version: str = "",
        sampler: "Sampler" = None,
        max_spans_per_trace: int = 1000,
        max_attributes_per_span: int = 128,
        max_events_per_span: int = 128,
        max_links_per_span: int = 32,
    ):
        self.collector = collector
        self.service_name = service_name
        self.service_version = service_version
        self.max_spans_per_trace = max_spans_per_trace
        self.max_attributes_per_span = max_attributes_per_span
        self.max_events_per_span = max_events_per_span
        self.max_links_per_span = max_links_per_span

        # Sampler
        if sampler is None:
            from aion.observability.tracing.sampling import AlwaysOnSampler
            sampler = AlwaysOnSampler()
        self.sampler = sampler

        # Active traces
        self._traces: Dict[str, Trace] = {}
        self._trace_locks: Dict[str, threading.Lock] = {}

        # Statistics
        self._stats = {
            "spans_started": 0,
            "spans_ended": 0,
            "spans_dropped": 0,
            "traces_completed": 0,
        }

        # Cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the tracing engine."""
        if self._initialized:
            return

        logger.info(
            "Initializing Tracing Engine",
            service_name=self.service_name,
        )

        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

        self._initialized = True

    async def shutdown(self) -> None:
        """Shutdown the tracing engine."""
        logger.info("Shutting down Tracing Engine")

        self._shutdown_event.set()

        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        # End any active spans
        for trace in list(self._traces.values()):
            for span in trace.spans:
                if not span.end_time:
                    span.end()
                    self.collector.collect_span(span)

        self._initialized = False

    def _generate_trace_id(self) -> str:
        """Generate a W3C-compatible trace ID (32 hex chars)."""
        return uuid.uuid4().hex

    def _generate_span_id(self) -> str:
        """Generate a W3C-compatible span ID (16 hex chars)."""
        return uuid.uuid4().hex[:16]

    def start_span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        parent: Optional[Union[Span, SpanContext]] = None,
        attributes: Dict[str, Any] = None,
        links: List[SpanLink] = None,
        start_time: datetime = None,
        trace_id: Optional[str] = None,
    ) -> Span:
        """
        Start a new span.

        Args:
            name: Span name (operation name)
            kind: Span kind (internal, server, client, producer, consumer)
            parent: Parent span or span context
            attributes: Initial span attributes
            links: Links to other spans
            start_time: Custom start time
            trace_id: Force specific trace ID

        Returns:
            New span (already set as current)
        """
        # Get parent from context if not provided
        if parent is None:
            parent = _current_span.get()

        # Determine trace ID and parent span ID
        if trace_id:
            tid = trace_id
            parent_span_id = None
        elif isinstance(parent, Span):
            tid = parent.trace_id
            parent_span_id = parent.span_id
        elif isinstance(parent, SpanContext):
            tid = parent.trace_id
            parent_span_id = parent.span_id
        else:
            tid = self._generate_trace_id()
            parent_span_id = None

        # Make sampling decision
        sampling_decision = self.sampler.should_sample(
            trace_id=tid,
            parent_context=parent.context if isinstance(parent, Span) else parent,
            name=name,
            kind=kind,
            attributes=attributes,
            links=links,
        )

        # Create span
        span = Span(
            trace_id=tid,
            span_id=self._generate_span_id(),
            parent_span_id=parent_span_id,
            name=name,
            kind=kind,
            start_time=start_time or datetime.utcnow(),
            attributes=attributes or {},
            links=links or [],
            service_name=self.service_name,
            service_version=self.service_version,
            sampling_decision=sampling_decision,
        )

        # Track in trace
        if tid not in self._traces:
            self._traces[tid] = Trace(trace_id=tid)
            self._trace_locks[tid] = threading.Lock()

        trace = self._traces[tid]

        # Check span limit
        if len(trace.spans) >= self.max_spans_per_trace:
            self._stats["spans_dropped"] += 1
            span.dropped_attributes_count += 1
        else:
            with self._trace_locks[tid]:
                trace.spans.append(span)

        # Set as current span
        _current_span.set(span)

        self._stats["spans_started"] += 1

        return span

    def end_span(
        self,
        span: Span,
        end_time: datetime = None,
        status: SpanStatus = None,
        status_message: str = "",
    ) -> None:
        """End a span."""
        if span.end_time:
            logger.warning(f"Span {span.name} already ended")
            return

        span.end_time = end_time or datetime.utcnow()

        if status:
            span.status = status
            span.status_message = status_message

        # Export if sampled
        if span.sampling_decision == SamplingDecision.RECORD_AND_SAMPLE:
            self.collector.collect_span(span)

        # Restore parent context
        if span.parent_span_id:
            trace = self._traces.get(span.trace_id)
            if trace:
                parent = next(
                    (s for s in trace.spans if s.span_id == span.parent_span_id),
                    None
                )
                _current_span.set(parent)
        else:
            _current_span.set(None)

        self._stats["spans_ended"] += 1

    def add_event(
        self,
        span: Span,
        name: str,
        attributes: Dict[str, Any] = None,
        timestamp: datetime = None,
    ) -> None:
        """Add an event to a span."""
        if len(span.events) >= self.max_events_per_span:
            span.dropped_events_count += 1
            return

        event = SpanEvent(
            name=name,
            attributes=attributes or {},
            timestamp=timestamp or datetime.utcnow(),
        )
        span.events.append(event)

    def add_link(
        self,
        span: Span,
        link: SpanLink,
    ) -> None:
        """Add a link to another span."""
        if len(span.links) >= self.max_links_per_span:
            span.dropped_links_count += 1
            return

        span.links.append(link)

    def set_attribute(
        self,
        span: Span,
        key: str,
        value: Any,
    ) -> None:
        """Set a span attribute."""
        if len(span.attributes) >= self.max_attributes_per_span:
            span.dropped_attributes_count += 1
            return

        span.attributes[key] = value

    def set_attributes(
        self,
        span: Span,
        attributes: Dict[str, Any],
    ) -> None:
        """Set multiple span attributes."""
        for key, value in attributes.items():
            self.set_attribute(span, key, value)

    def set_status(
        self,
        span: Span,
        status: SpanStatus,
        message: str = "",
    ) -> None:
        """Set span status."""
        span.status = status
        span.status_message = message

    def record_exception(
        self,
        span: Span,
        exception: Exception,
        escaped: bool = False,
    ) -> None:
        """Record an exception on a span."""
        import traceback

        self.add_event(span, "exception", {
            "exception.type": type(exception).__name__,
            "exception.message": str(exception),
            "exception.stacktrace": traceback.format_exc(),
            "exception.escaped": escaped,
        })

        if not escaped:
            self.set_status(span, SpanStatus.ERROR, str(exception))

    def get_current_span(self) -> Optional[Span]:
        """Get the current span from context."""
        return _current_span.get()

    def get_current_trace_id(self) -> Optional[str]:
        """Get current trace ID."""
        span = self.get_current_span()
        return span.trace_id if span else None

    def get_current_span_context(self) -> Optional[SpanContext]:
        """Get current span context for propagation."""
        span = self.get_current_span()
        return span.context if span else None

    def get_trace(self, trace_id: str) -> Optional[Trace]:
        """Get a trace by ID."""
        return self._traces.get(trace_id)

    def get_span(self, trace_id: str, span_id: str) -> Optional[Span]:
        """Get a specific span."""
        trace = self._traces.get(trace_id)
        if trace:
            for span in trace.spans:
                if span.span_id == span_id:
                    return span
        return None

    # === Context Propagation ===

    def inject_context(
        self,
        headers: Dict[str, str],
        span: Optional[Span] = None,
    ) -> Dict[str, str]:
        """Inject trace context into headers (W3C format)."""
        span = span or self.get_current_span()

        if span:
            context = span.context
            headers["traceparent"] = context.to_traceparent()
            if context.trace_state:
                headers["tracestate"] = context.trace_state

        return headers

    def extract_context(self, headers: Dict[str, str]) -> Optional[SpanContext]:
        """Extract trace context from headers."""
        traceparent = headers.get("traceparent")
        if traceparent:
            context = SpanContext.from_traceparent(traceparent)
            if context:
                context.trace_state = headers.get("tracestate", "")
                return context
        return None

    def create_child_span_from_headers(
        self,
        name: str,
        headers: Dict[str, str],
        kind: SpanKind = SpanKind.SERVER,
        attributes: Dict[str, Any] = None,
    ) -> Span:
        """Create a child span from propagated headers."""
        parent_context = self.extract_context(headers)

        if parent_context:
            return self.start_span(
                name=name,
                kind=kind,
                parent=parent_context,
                attributes=attributes,
                trace_id=parent_context.trace_id,
            )
        else:
            return self.start_span(
                name=name,
                kind=kind,
                attributes=attributes,
            )

    # === Cleanup ===

    async def _cleanup_loop(self) -> None:
        """Background cleanup of old traces."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(300)  # 5 minutes
                await self._cleanup_traces()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Trace cleanup error: {e}")

    async def _cleanup_traces(self) -> None:
        """Remove old completed traces."""
        cutoff = datetime.utcnow() - timedelta(minutes=30)

        for trace_id in list(self._traces.keys()):
            trace = self._traces.get(trace_id)
            if not trace:
                continue

            # Check if all spans are ended and old
            all_ended = all(s.end_time for s in trace.spans)
            if all_ended and trace.spans:
                latest_end = max(s.end_time for s in trace.spans)
                if latest_end < cutoff:
                    del self._traces[trace_id]
                    if trace_id in self._trace_locks:
                        del self._trace_locks[trace_id]
                    self._stats["traces_completed"] += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get tracing statistics."""
        return {
            **self._stats,
            "active_traces": len(self._traces),
            "total_active_spans": sum(len(t.spans) for t in self._traces.values()),
        }


class SpanContextManager:
    """Context manager for spans."""

    def __init__(
        self,
        engine: TracingEngine,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Dict[str, Any] = None,
        record_exception: bool = True,
        set_status_on_error: bool = True,
    ):
        self.engine = engine
        self.name = name
        self.kind = kind
        self.attributes = attributes
        self.record_exception = record_exception
        self.set_status_on_error = set_status_on_error
        self.span: Optional[Span] = None

    def __enter__(self) -> Span:
        self.span = self.engine.start_span(
            self.name,
            self.kind,
            attributes=self.attributes,
        )
        return self.span

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val and self.record_exception:
            self.engine.record_exception(self.span, exc_val)
        elif exc_val and self.set_status_on_error:
            self.engine.set_status(self.span, SpanStatus.ERROR, str(exc_val))

        status = SpanStatus.ERROR if exc_val else SpanStatus.OK
        self.engine.end_span(self.span, status=status)

    async def __aenter__(self) -> Span:
        return self.__enter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.__exit__(exc_type, exc_val, exc_tb)


def traced(
    name: Optional[str] = None,
    kind: SpanKind = SpanKind.INTERNAL,
    attributes: Dict[str, Any] = None,
    record_exception: bool = True,
):
    """
    Decorator to trace a function.

    Usage:
        @traced("process_request", kind=SpanKind.SERVER)
        async def process_request(request):
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        span_name = name or func.__name__

        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                from aion.observability import get_tracing_engine
                engine = get_tracing_engine()

                async with SpanContextManager(
                    engine,
                    span_name,
                    kind,
                    attributes,
                    record_exception,
                ) as span:
                    span.set_attribute("code.function", func.__name__)
                    span.set_attribute("code.filepath", func.__code__.co_filename)
                    return await func(*args, **kwargs)

            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                from aion.observability import get_tracing_engine
                engine = get_tracing_engine()

                with SpanContextManager(
                    engine,
                    span_name,
                    kind,
                    attributes,
                    record_exception,
                ) as span:
                    span.set_attribute("code.function", func.__name__)
                    span.set_attribute("code.filepath", func.__code__.co_filename)
                    return func(*args, **kwargs)

            return sync_wrapper

    return decorator


def trace_method(
    kind: SpanKind = SpanKind.INTERNAL,
    attributes: Dict[str, Any] = None,
):
    """
    Decorator to trace a method (includes class name in span name).

    Usage:
        class MyService:
            @trace_method()
            async def process(self):
                ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(self, *args, **kwargs):
                from aion.observability import get_tracing_engine
                engine = get_tracing_engine()

                span_name = f"{self.__class__.__name__}.{func.__name__}"

                async with SpanContextManager(
                    engine,
                    span_name,
                    kind,
                    attributes,
                ) as span:
                    span.set_attribute("code.namespace", self.__class__.__name__)
                    span.set_attribute("code.function", func.__name__)
                    return await func(self, *args, **kwargs)

            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(self, *args, **kwargs):
                from aion.observability import get_tracing_engine
                engine = get_tracing_engine()

                span_name = f"{self.__class__.__name__}.{func.__name__}"

                with SpanContextManager(
                    engine,
                    span_name,
                    kind,
                    attributes,
                ) as span:
                    span.set_attribute("code.namespace", self.__class__.__name__)
                    span.set_attribute("code.function", func.__name__)
                    return func(self, *args, **kwargs)

            return sync_wrapper

    return decorator


@contextmanager
def trace_block(
    name: str,
    attributes: Dict[str, Any] = None,
) -> Generator[Span, None, None]:
    """
    Context manager for tracing a code block.

    Usage:
        with trace_block("process_items") as span:
            for item in items:
                process(item)
    """
    from aion.observability import get_tracing_engine
    engine = get_tracing_engine()

    span = engine.start_span(name, attributes=attributes)
    try:
        yield span
        engine.end_span(span, status=SpanStatus.OK)
    except Exception as e:
        engine.record_exception(span, e)
        engine.end_span(span, status=SpanStatus.ERROR)
        raise


@asynccontextmanager
async def async_trace_block(
    name: str,
    attributes: Dict[str, Any] = None,
) -> Generator[Span, None, None]:
    """Async version of trace_block."""
    from aion.observability import get_tracing_engine
    engine = get_tracing_engine()

    span = engine.start_span(name, attributes=attributes)
    try:
        yield span
        engine.end_span(span, status=SpanStatus.OK)
    except Exception as e:
        engine.record_exception(span, e)
        engine.end_span(span, status=SpanStatus.ERROR)
        raise
