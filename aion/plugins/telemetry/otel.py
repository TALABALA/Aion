"""
OpenTelemetry SDK Integration

Real integration with the OpenTelemetry SDK for production-grade
distributed tracing and metrics collection.

Requires:
    pip install opentelemetry-api opentelemetry-sdk
    pip install opentelemetry-exporter-otlp  # For OTLP export
    pip install opentelemetry-exporter-jaeger  # For Jaeger export
    pip install opentelemetry-exporter-prometheus  # For Prometheus metrics
"""

from __future__ import annotations

import atexit
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Dict, Iterator, Optional, TypeVar

import structlog

logger = structlog.get_logger(__name__)

# Try to import OpenTelemetry SDK
try:
    from opentelemetry import trace as otel_trace
    from opentelemetry import metrics as otel_metrics
    from opentelemetry.sdk.trace import TracerProvider, SpanProcessor
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        ConsoleSpanExporter,
        SpanExporter,
    )
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import (
        PeriodicExportingMetricReader,
        ConsoleMetricExporter,
        MetricExporter,
    )
    from opentelemetry.sdk.resources import Resource, SERVICE_NAME
    from opentelemetry.trace import Status, StatusCode, Span, Tracer
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
    from opentelemetry.propagate import set_global_textmap, inject, extract
    from opentelemetry.context import Context
    from opentelemetry.semconv.trace import SpanAttributes
    from opentelemetry.metrics import Meter, Counter as OTelCounter, Histogram as OTelHistogram

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    logger.warning("OpenTelemetry SDK not installed. Using fallback implementation.")

# Optional exporters
try:
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    OTLP_AVAILABLE = True
except ImportError:
    OTLP_AVAILABLE = False

try:
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    JAEGER_AVAILABLE = True
except ImportError:
    JAEGER_AVAILABLE = False

try:
    from opentelemetry.exporter.prometheus import PrometheusMetricReader
    from prometheus_client import start_http_server as start_prometheus_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


T = TypeVar("T")


@dataclass
class OTelConfig:
    """Configuration for OpenTelemetry integration."""

    # Service identification
    service_name: str = "aion-plugins"
    service_version: str = "1.0.0"
    deployment_environment: str = "development"

    # Tracing
    tracing_enabled: bool = True
    trace_sample_rate: float = 1.0  # 1.0 = 100% sampling

    # Trace export
    trace_exporter: str = "console"  # console, otlp, jaeger, none
    otlp_endpoint: str = "http://localhost:4317"
    jaeger_agent_host: str = "localhost"
    jaeger_agent_port: int = 6831

    # Metrics
    metrics_enabled: bool = True
    metrics_export_interval_ms: int = 60000  # 1 minute

    # Metrics export
    metrics_exporter: str = "console"  # console, otlp, prometheus, none
    prometheus_port: int = 9464

    # Propagation
    propagation_format: str = "tracecontext"  # W3C Trace Context

    # Batching
    batch_max_size: int = 512
    batch_schedule_delay_ms: int = 5000


class OTelManager:
    """
    Manages OpenTelemetry SDK initialization and providers.

    Provides a central point for configuring tracing and metrics
    with automatic cleanup on shutdown.
    """

    _instance: Optional["OTelManager"] = None

    def __init__(self, config: Optional[OTelConfig] = None):
        self.config = config or OTelConfig()
        self._tracer_provider: Optional[TracerProvider] = None
        self._meter_provider: Optional[MeterProvider] = None
        self._tracers: Dict[str, Tracer] = {}
        self._meters: Dict[str, Meter] = {}
        self._initialized = False

    @classmethod
    def get_instance(cls) -> "OTelManager":
        """Get the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def configure(cls, config: OTelConfig) -> "OTelManager":
        """Configure and return the singleton instance."""
        cls._instance = cls(config)
        return cls._instance

    def initialize(self) -> None:
        """Initialize OpenTelemetry providers."""
        if self._initialized:
            return

        if not OTEL_AVAILABLE:
            logger.warning("OpenTelemetry SDK not available")
            self._initialized = True
            return

        # Create resource
        resource = Resource.create({
            SERVICE_NAME: self.config.service_name,
            "service.version": self.config.service_version,
            "deployment.environment": self.config.deployment_environment,
        })

        # Initialize tracing
        if self.config.tracing_enabled:
            self._init_tracing(resource)

        # Initialize metrics
        if self.config.metrics_enabled:
            self._init_metrics(resource)

        # Set up propagation
        set_global_textmap(TraceContextTextMapPropagator())

        # Register shutdown handler
        atexit.register(self.shutdown)

        self._initialized = True
        logger.info(
            "OpenTelemetry initialized",
            service=self.config.service_name,
            tracing=self.config.tracing_enabled,
            metrics=self.config.metrics_enabled,
        )

    def _init_tracing(self, resource: Resource) -> None:
        """Initialize tracing provider."""
        # Create span exporter
        exporter = self._create_span_exporter()

        # Create and configure tracer provider
        self._tracer_provider = TracerProvider(resource=resource)

        if exporter:
            processor = BatchSpanProcessor(
                exporter,
                max_export_batch_size=self.config.batch_max_size,
                schedule_delay_millis=self.config.batch_schedule_delay_ms,
            )
            self._tracer_provider.add_span_processor(processor)

        # Set as global provider
        otel_trace.set_tracer_provider(self._tracer_provider)

    def _init_metrics(self, resource: Resource) -> None:
        """Initialize metrics provider."""
        readers = []

        # Create metric reader based on exporter config
        if self.config.metrics_exporter == "console":
            readers.append(PeriodicExportingMetricReader(
                ConsoleMetricExporter(),
                export_interval_millis=self.config.metrics_export_interval_ms,
            ))
        elif self.config.metrics_exporter == "otlp" and OTLP_AVAILABLE:
            readers.append(PeriodicExportingMetricReader(
                OTLPMetricExporter(endpoint=self.config.otlp_endpoint),
                export_interval_millis=self.config.metrics_export_interval_ms,
            ))
        elif self.config.metrics_exporter == "prometheus" and PROMETHEUS_AVAILABLE:
            readers.append(PrometheusMetricReader())
            start_prometheus_server(self.config.prometheus_port)
            logger.info(
                "Prometheus metrics server started",
                port=self.config.prometheus_port,
            )

        # Create meter provider
        self._meter_provider = MeterProvider(
            resource=resource,
            metric_readers=readers,
        )

        # Set as global provider
        otel_metrics.set_meter_provider(self._meter_provider)

    def _create_span_exporter(self) -> Optional[SpanExporter]:
        """Create span exporter based on config."""
        exporter_type = self.config.trace_exporter

        if exporter_type == "console":
            return ConsoleSpanExporter()

        elif exporter_type == "otlp" and OTLP_AVAILABLE:
            return OTLPSpanExporter(endpoint=self.config.otlp_endpoint)

        elif exporter_type == "jaeger" and JAEGER_AVAILABLE:
            return JaegerExporter(
                agent_host_name=self.config.jaeger_agent_host,
                agent_port=self.config.jaeger_agent_port,
            )

        elif exporter_type == "none":
            return None

        else:
            logger.warning(f"Unknown trace exporter: {exporter_type}, using console")
            return ConsoleSpanExporter()

    def get_tracer(self, name: str) -> "PluginTracer":
        """Get a tracer for a plugin."""
        if not self._initialized:
            self.initialize()

        if name not in self._tracers and OTEL_AVAILABLE and self._tracer_provider:
            self._tracers[name] = self._tracer_provider.get_tracer(
                name,
                version=self.config.service_version,
            )

        return PluginTracer(name, self._tracers.get(name))

    def get_meter(self, name: str) -> "PluginMeter":
        """Get a meter for a plugin."""
        if not self._initialized:
            self.initialize()

        if name not in self._meters and OTEL_AVAILABLE and self._meter_provider:
            self._meters[name] = self._meter_provider.get_meter(
                name,
                version=self.config.service_version,
            )

        return PluginMeter(name, self._meters.get(name))

    def shutdown(self) -> None:
        """Shutdown OpenTelemetry providers."""
        if not self._initialized:
            return

        if self._tracer_provider and OTEL_AVAILABLE:
            self._tracer_provider.shutdown()

        if self._meter_provider and OTEL_AVAILABLE:
            self._meter_provider.shutdown()

        self._initialized = False
        logger.info("OpenTelemetry shutdown complete")


class PluginTracer:
    """
    Tracer wrapper for plugins.

    Provides a simplified API for creating spans with automatic
    context propagation.
    """

    def __init__(self, name: str, tracer: Optional[Tracer] = None):
        self.name = name
        self._tracer = tracer

    @contextmanager
    def start_span(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
        kind: Optional[str] = None,
    ) -> Iterator[Optional[Span]]:
        """
        Start a new span.

        Usage:
            with tracer.start_span("operation", attributes={"key": "value"}) as span:
                # ... do work ...
                span.set_attribute("result", "success")
        """
        if not OTEL_AVAILABLE or not self._tracer:
            yield None
            return

        span_kind = otel_trace.SpanKind.INTERNAL
        if kind == "client":
            span_kind = otel_trace.SpanKind.CLIENT
        elif kind == "server":
            span_kind = otel_trace.SpanKind.SERVER
        elif kind == "producer":
            span_kind = otel_trace.SpanKind.PRODUCER
        elif kind == "consumer":
            span_kind = otel_trace.SpanKind.CONSUMER

        with self._tracer.start_as_current_span(
            name,
            kind=span_kind,
            attributes=attributes,
        ) as span:
            try:
                yield span
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

    def start_span_from_context(
        self,
        name: str,
        carrier: Dict[str, str],
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Iterator[Optional[Span]]:
        """
        Start a span with context extracted from carrier headers.

        Usage:
            headers = request.headers
            with tracer.start_span_from_context("handle_request", headers) as span:
                # ... handle request ...
        """
        if not OTEL_AVAILABLE or not self._tracer:
            yield None
            return

        # Extract context from carrier
        ctx = extract(carrier)

        with self._tracer.start_as_current_span(
            name,
            context=ctx,
            attributes=attributes,
        ) as span:
            try:
                yield span
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

    def inject_context(self, carrier: Dict[str, str]) -> Dict[str, str]:
        """
        Inject current trace context into carrier for propagation.

        Usage:
            headers = {}
            tracer.inject_context(headers)
            # headers now contains traceparent, tracestate
        """
        if OTEL_AVAILABLE:
            inject(carrier)
        return carrier

    def trace(
        self,
        name: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Callable:
        """
        Decorator for tracing functions.

        Usage:
            @tracer.trace("my_function")
            async def my_function():
                ...
        """
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            span_name = name or func.__name__

            @wraps(func)
            async def async_wrapper(*args, **kwargs) -> T:
                with self.start_span(span_name, attributes) as span:
                    if span:
                        span.set_attribute("function.name", func.__name__)
                        span.set_attribute("function.module", func.__module__)
                    return await func(*args, **kwargs)

            @wraps(func)
            def sync_wrapper(*args, **kwargs) -> T:
                with self.start_span(span_name, attributes) as span:
                    if span:
                        span.set_attribute("function.name", func.__name__)
                        span.set_attribute("function.module", func.__module__)
                    return func(*args, **kwargs)

            import asyncio
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            return sync_wrapper

        return decorator


class PluginMeter:
    """
    Meter wrapper for plugins.

    Provides a simplified API for creating and recording metrics.
    """

    def __init__(self, name: str, meter: Optional[Meter] = None):
        self.name = name
        self._meter = meter
        self._counters: Dict[str, OTelCounter] = {}
        self._histograms: Dict[str, OTelHistogram] = {}

    def counter(
        self,
        name: str,
        description: str = "",
        unit: str = "1",
    ) -> "MetricCounter":
        """Create or get a counter."""
        if name not in self._counters and OTEL_AVAILABLE and self._meter:
            self._counters[name] = self._meter.create_counter(
                name,
                description=description,
                unit=unit,
            )
        return MetricCounter(name, self._counters.get(name))

    def histogram(
        self,
        name: str,
        description: str = "",
        unit: str = "ms",
    ) -> "MetricHistogram":
        """Create or get a histogram."""
        if name not in self._histograms and OTEL_AVAILABLE and self._meter:
            self._histograms[name] = self._meter.create_histogram(
                name,
                description=description,
                unit=unit,
            )
        return MetricHistogram(name, self._histograms.get(name))


class MetricCounter:
    """Counter metric wrapper."""

    def __init__(self, name: str, counter: Optional[OTelCounter] = None):
        self.name = name
        self._counter = counter
        self._fallback_value = 0

    def add(self, value: int = 1, attributes: Optional[Dict[str, str]] = None) -> None:
        """Add to the counter."""
        if self._counter and OTEL_AVAILABLE:
            self._counter.add(value, attributes or {})
        else:
            self._fallback_value += value

    def inc(self, attributes: Optional[Dict[str, str]] = None) -> None:
        """Increment by 1."""
        self.add(1, attributes)


class MetricHistogram:
    """Histogram metric wrapper."""

    def __init__(self, name: str, histogram: Optional[OTelHistogram] = None):
        self.name = name
        self._histogram = histogram
        self._fallback_values: list[float] = []

    def record(self, value: float, attributes: Optional[Dict[str, str]] = None) -> None:
        """Record a value."""
        if self._histogram and OTEL_AVAILABLE:
            self._histogram.record(value, attributes or {})
        else:
            self._fallback_values.append(value)

    @contextmanager
    def time(self, attributes: Optional[Dict[str, str]] = None) -> Iterator[None]:
        """Context manager for timing operations."""
        import time
        start = time.perf_counter()
        try:
            yield
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            self.record(duration_ms, attributes)


# Convenience functions
def get_tracer(name: str) -> PluginTracer:
    """Get a tracer for a plugin."""
    return OTelManager.get_instance().get_tracer(name)


def get_meter(name: str) -> PluginMeter:
    """Get a meter for a plugin."""
    return OTelManager.get_instance().get_meter(name)


def configure_otel(config: OTelConfig) -> OTelManager:
    """Configure OpenTelemetry with the given config."""
    manager = OTelManager.configure(config)
    manager.initialize()
    return manager


def trace_operation(
    name: str,
    plugin_id: str,
    attributes: Optional[Dict[str, Any]] = None,
) -> Callable:
    """
    Decorator for tracing plugin operations.

    Usage:
        @trace_operation("fetch_data", "my-plugin")
        async def fetch_data():
            ...
    """
    tracer = get_tracer(plugin_id)
    attrs = {"plugin.id": plugin_id}
    if attributes:
        attrs.update(attributes)
    return tracer.trace(name, attrs)
