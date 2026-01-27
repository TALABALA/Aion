"""
Distributed Tracing for Plugins

Implements OpenTelemetry-compatible tracing for plugin operations,
enabling cross-plugin trace correlation and performance analysis.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Optional, Dict, List, TypeVar

import structlog

logger = structlog.get_logger(__name__)

T = TypeVar("T")


@dataclass
class TracingConfig:
    """Configuration for plugin tracing."""

    enabled: bool = True
    service_name: str = "aion-plugins"

    # Sampling
    sample_rate: float = 1.0  # 1.0 = 100% sampling

    # Export
    export_enabled: bool = False
    export_endpoint: Optional[str] = None  # e.g., "http://localhost:4317"
    export_format: str = "otlp"  # otlp, jaeger, zipkin

    # Span limits
    max_attributes: int = 128
    max_events: int = 128
    max_links: int = 128

    # Context propagation
    propagate_context: bool = True


@dataclass
class SpanContext:
    """Context for a trace span."""

    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    trace_flags: int = 1  # 1 = sampled
    trace_state: dict[str, str] = field(default_factory=dict)

    @classmethod
    def generate(cls, parent: Optional["SpanContext"] = None) -> "SpanContext":
        """Generate a new span context."""
        return cls(
            trace_id=parent.trace_id if parent else uuid.uuid4().hex,
            span_id=uuid.uuid4().hex[:16],
            parent_span_id=parent.span_id if parent else None,
        )

    def to_headers(self) -> dict[str, str]:
        """Convert to W3C Trace Context headers."""
        return {
            "traceparent": f"00-{self.trace_id}-{self.span_id}-{self.trace_flags:02x}",
            "tracestate": ",".join(f"{k}={v}" for k, v in self.trace_state.items()),
        }

    @classmethod
    def from_headers(cls, headers: dict[str, str]) -> Optional["SpanContext"]:
        """Parse from W3C Trace Context headers."""
        traceparent = headers.get("traceparent")
        if not traceparent:
            return None

        try:
            parts = traceparent.split("-")
            if len(parts) != 4:
                return None

            trace_state = {}
            if "tracestate" in headers:
                for item in headers["tracestate"].split(","):
                    if "=" in item:
                        k, v = item.split("=", 1)
                        trace_state[k.strip()] = v.strip()

            return cls(
                trace_id=parts[1],
                span_id=parts[2],
                trace_flags=int(parts[3], 16),
                trace_state=trace_state,
            )
        except Exception:
            return None


@dataclass
class SpanEvent:
    """An event within a span."""

    name: str
    timestamp: float
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class SpanLink:
    """A link to another span."""

    context: SpanContext
    attributes: dict[str, Any] = field(default_factory=dict)


class Span:
    """
    A trace span representing a unit of work.

    Compatible with OpenTelemetry span semantics.
    """

    def __init__(
        self,
        name: str,
        context: SpanContext,
        kind: str = "internal",  # internal, client, server, producer, consumer
        attributes: Optional[dict[str, Any]] = None,
    ):
        self.name = name
        self.context = context
        self.kind = kind
        self.attributes: dict[str, Any] = attributes or {}
        self.events: list[SpanEvent] = []
        self.links: list[SpanLink] = []
        self.status: str = "unset"  # unset, ok, error
        self.status_message: Optional[str] = None

        self.start_time: float = time.time()
        self.end_time: Optional[float] = None
        self._ended = False

    @property
    def duration_ms(self) -> float:
        """Get span duration in milliseconds."""
        end = self.end_time or time.time()
        return (end - self.start_time) * 1000

    def set_attribute(self, key: str, value: Any) -> "Span":
        """Set a span attribute."""
        self.attributes[key] = value
        return self

    def set_attributes(self, attributes: dict[str, Any]) -> "Span":
        """Set multiple attributes."""
        self.attributes.update(attributes)
        return self

    def add_event(
        self,
        name: str,
        attributes: Optional[dict[str, Any]] = None,
    ) -> "Span":
        """Add an event to the span."""
        self.events.append(SpanEvent(
            name=name,
            timestamp=time.time(),
            attributes=attributes or {},
        ))
        return self

    def add_link(
        self,
        context: SpanContext,
        attributes: Optional[dict[str, Any]] = None,
    ) -> "Span":
        """Add a link to another span."""
        self.links.append(SpanLink(
            context=context,
            attributes=attributes or {},
        ))
        return self

    def set_status(self, status: str, message: Optional[str] = None) -> "Span":
        """Set span status."""
        self.status = status
        self.status_message = message
        return self

    def record_exception(self, exception: Exception) -> "Span":
        """Record an exception as an event."""
        self.add_event(
            "exception",
            {
                "exception.type": type(exception).__name__,
                "exception.message": str(exception),
            },
        )
        self.set_status("error", str(exception))
        return self

    def end(self) -> None:
        """End the span."""
        if not self._ended:
            self.end_time = time.time()
            self._ended = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for export."""
        return {
            "name": self.name,
            "trace_id": self.context.trace_id,
            "span_id": self.context.span_id,
            "parent_span_id": self.context.parent_span_id,
            "kind": self.kind,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "attributes": self.attributes,
            "events": [
                {
                    "name": e.name,
                    "timestamp": e.timestamp,
                    "attributes": e.attributes,
                }
                for e in self.events
            ],
            "links": [
                {
                    "trace_id": l.context.trace_id,
                    "span_id": l.context.span_id,
                    "attributes": l.attributes,
                }
                for l in self.links
            ],
            "status": self.status,
            "status_message": self.status_message,
        }


class PluginTracer:
    """
    Tracer for plugin operations.

    Creates and manages spans for tracing plugin execution.
    """

    def __init__(
        self,
        plugin_id: str,
        config: Optional[TracingConfig] = None,
    ):
        self.plugin_id = plugin_id
        self.config = config or TracingConfig()
        self._spans: list[Span] = []
        self._active_spans: dict[str, Span] = {}
        self._current_context: Optional[SpanContext] = None
        self._lock = asyncio.Lock()

        # Optional OpenTelemetry integration
        self._otel_tracer = None
        self._setup_otel()

    def _setup_otel(self) -> None:
        """Set up OpenTelemetry if available."""
        if not self.config.export_enabled:
            return

        try:
            from opentelemetry import trace
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            if self.config.export_format == "otlp":
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                    OTLPSpanExporter,
                )
                exporter = OTLPSpanExporter(endpoint=self.config.export_endpoint)
            else:
                # Fallback to console exporter
                from opentelemetry.sdk.trace.export import ConsoleSpanExporter
                exporter = ConsoleSpanExporter()

            provider = TracerProvider()
            provider.add_span_processor(BatchSpanProcessor(exporter))
            trace.set_tracer_provider(provider)

            self._otel_tracer = trace.get_tracer(
                self.config.service_name,
                schema_url="https://opentelemetry.io/schemas/1.11.0",
            )

            logger.info(
                "OpenTelemetry tracing initialized",
                plugin_id=self.plugin_id,
            )

        except ImportError:
            logger.debug("OpenTelemetry not available")
        except Exception as e:
            logger.warning(f"Failed to initialize OpenTelemetry: {e}")

    @contextmanager
    def start_span(
        self,
        name: str,
        kind: str = "internal",
        attributes: Optional[dict[str, Any]] = None,
    ):
        """
        Start a new span as a context manager.

        Usage:
            with tracer.start_span("operation") as span:
                span.set_attribute("key", "value")
                # ... do work ...
        """
        if not self.config.enabled:
            yield None
            return

        # Create span context
        parent = self._current_context
        context = SpanContext.generate(parent)

        # Create span
        span = Span(
            name=name,
            context=context,
            kind=kind,
            attributes={
                "plugin.id": self.plugin_id,
                **(attributes or {}),
            },
        )

        # Set as current context
        previous_context = self._current_context
        self._current_context = context
        self._active_spans[context.span_id] = span

        try:
            yield span
            if span.status == "unset":
                span.set_status("ok")
        except Exception as e:
            span.record_exception(e)
            raise
        finally:
            span.end()
            self._current_context = previous_context
            self._active_spans.pop(context.span_id, None)
            self._spans.append(span)

            # Keep only recent spans
            if len(self._spans) > 1000:
                self._spans = self._spans[-1000:]

    @asynccontextmanager
    async def start_span_async(
        self,
        name: str,
        kind: str = "internal",
        attributes: Optional[dict[str, Any]] = None,
    ):
        """Async version of start_span."""
        if not self.config.enabled:
            yield None
            return

        async with self._lock:
            parent = self._current_context
            context = SpanContext.generate(parent)

            span = Span(
                name=name,
                context=context,
                kind=kind,
                attributes={
                    "plugin.id": self.plugin_id,
                    **(attributes or {}),
                },
            )

            previous_context = self._current_context
            self._current_context = context
            self._active_spans[context.span_id] = span

        try:
            yield span
            if span.status == "unset":
                span.set_status("ok")
        except Exception as e:
            span.record_exception(e)
            raise
        finally:
            span.end()
            async with self._lock:
                self._current_context = previous_context
                self._active_spans.pop(context.span_id, None)
                self._spans.append(span)

                if len(self._spans) > 1000:
                    self._spans = self._spans[-1000:]

    def get_current_context(self) -> Optional[SpanContext]:
        """Get current span context."""
        return self._current_context

    def set_context(self, context: SpanContext) -> None:
        """Set current context (for context propagation)."""
        self._current_context = context

    def get_recent_spans(self, limit: int = 100) -> list[dict[str, Any]]:
        """Get recent completed spans."""
        return [span.to_dict() for span in self._spans[-limit:]]

    def get_stats(self) -> dict[str, Any]:
        """Get tracing statistics."""
        total_duration = sum(s.duration_ms for s in self._spans)
        error_count = sum(1 for s in self._spans if s.status == "error")

        return {
            "plugin_id": self.plugin_id,
            "enabled": self.config.enabled,
            "total_spans": len(self._spans),
            "active_spans": len(self._active_spans),
            "total_duration_ms": total_duration,
            "average_duration_ms": total_duration / len(self._spans) if self._spans else 0,
            "error_count": error_count,
            "error_rate": error_count / len(self._spans) if self._spans else 0,
        }


def trace_plugin_operation(
    tracer: PluginTracer,
    name: Optional[str] = None,
    kind: str = "internal",
    attributes: Optional[dict[str, Any]] = None,
):
    """
    Decorator to trace a plugin operation.

    Usage:
        @trace_plugin_operation(tracer, "my_operation")
        async def my_function():
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        span_name = name or func.__name__

        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                async with tracer.start_span_async(span_name, kind, attributes):
                    return await func(*args, **kwargs)
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                with tracer.start_span(span_name, kind, attributes):
                    return func(*args, **kwargs)
            return sync_wrapper

    return decorator


class TracingManager:
    """
    Manages tracing across multiple plugins.

    Provides centralized trace management and correlation.
    """

    def __init__(self, config: Optional[TracingConfig] = None):
        self.config = config or TracingConfig()
        self._tracers: dict[str, PluginTracer] = {}
        self._global_spans: list[Span] = []

    def get_tracer(self, plugin_id: str) -> PluginTracer:
        """Get or create tracer for a plugin."""
        if plugin_id not in self._tracers:
            self._tracers[plugin_id] = PluginTracer(plugin_id, self.config)
        return self._tracers[plugin_id]

    def remove_tracer(self, plugin_id: str) -> None:
        """Remove tracer for a plugin."""
        self._tracers.pop(plugin_id, None)

    def get_all_spans(self, limit: int = 1000) -> list[dict[str, Any]]:
        """Get all spans across plugins."""
        all_spans = []
        for tracer in self._tracers.values():
            all_spans.extend(tracer._spans)

        # Sort by start time
        all_spans.sort(key=lambda s: s.start_time, reverse=True)
        return [s.to_dict() for s in all_spans[:limit]]

    def get_trace(self, trace_id: str) -> list[dict[str, Any]]:
        """Get all spans for a specific trace."""
        spans = []
        for tracer in self._tracers.values():
            for span in tracer._spans:
                if span.context.trace_id == trace_id:
                    spans.append(span.to_dict())

        # Sort by start time
        spans.sort(key=lambda s: s["start_time"])
        return spans

    def get_stats(self) -> dict[str, Any]:
        """Get statistics for all tracers."""
        return {
            "total_tracers": len(self._tracers),
            "config": {
                "enabled": self.config.enabled,
                "sample_rate": self.config.sample_rate,
                "export_enabled": self.config.export_enabled,
            },
            "per_plugin": {
                pid: tracer.get_stats()
                for pid, tracer in self._tracers.items()
            },
        }
