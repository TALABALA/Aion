"""
AION Telemetry Provider

OpenTelemetry configuration and setup for workflows.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import structlog

logger = structlog.get_logger(__name__)

# Global telemetry state
_tracer = None
_meter = None
_provider = None


class ExporterType(str, Enum):
    """Supported telemetry exporters."""
    CONSOLE = "console"
    OTLP = "otlp"
    JAEGER = "jaeger"
    ZIPKIN = "zipkin"
    PROMETHEUS = "prometheus"


@dataclass
class TracingConfig:
    """Tracing configuration."""
    enabled: bool = True
    service_name: str = "aion-automation"
    service_version: str = "1.0.0"

    # Exporter settings
    exporter: ExporterType = ExporterType.OTLP
    otlp_endpoint: str = "http://localhost:4317"
    jaeger_endpoint: str = "http://localhost:14268/api/traces"
    zipkin_endpoint: str = "http://localhost:9411/api/v2/spans"

    # Sampling
    sample_rate: float = 1.0  # 1.0 = 100% sampling
    always_sample_errors: bool = True

    # Batch settings
    max_queue_size: int = 2048
    max_export_batch_size: int = 512
    export_timeout_millis: int = 30000

    # Additional attributes
    resource_attributes: Dict[str, str] = field(default_factory=dict)


@dataclass
class MetricsConfig:
    """Metrics configuration."""
    enabled: bool = True
    service_name: str = "aion-automation"

    # Exporter settings
    exporter: ExporterType = ExporterType.PROMETHEUS
    otlp_endpoint: str = "http://localhost:4317"
    prometheus_port: int = 9090

    # Collection interval
    export_interval_millis: int = 60000

    # Metric prefixes
    metric_prefix: str = "aion_automation"


class TelemetryProvider:
    """
    Central telemetry provider for workflow observability.

    Configures and manages OpenTelemetry tracing and metrics.
    """

    def __init__(
        self,
        tracing_config: Optional[TracingConfig] = None,
        metrics_config: Optional[MetricsConfig] = None,
    ):
        self.tracing_config = tracing_config or TracingConfig()
        self.metrics_config = metrics_config or MetricsConfig()

        self._tracer_provider = None
        self._meter_provider = None
        self._tracer = None
        self._meter = None

        self._initialized = False

    def initialize(self) -> None:
        """Initialize telemetry providers."""
        if self._initialized:
            return

        global _tracer, _meter, _provider

        try:
            self._setup_tracing()
            self._setup_metrics()

            _tracer = self._tracer
            _meter = self._meter
            _provider = self

            self._initialized = True
            logger.info(
                "Telemetry initialized",
                tracing_enabled=self.tracing_config.enabled,
                metrics_enabled=self.metrics_config.enabled,
            )

        except ImportError as e:
            logger.warning(f"OpenTelemetry not available: {e}")
            self._setup_noop_telemetry()

    def _setup_tracing(self) -> None:
        """Setup OpenTelemetry tracing."""
        if not self.tracing_config.enabled:
            self._setup_noop_tracer()
            return

        try:
            from opentelemetry import trace
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
            from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION

            # Create resource
            resource_attributes = {
                SERVICE_NAME: self.tracing_config.service_name,
                SERVICE_VERSION: self.tracing_config.service_version,
                **self.tracing_config.resource_attributes,
            }
            resource = Resource.create(resource_attributes)

            # Create provider
            self._tracer_provider = TracerProvider(resource=resource)

            # Setup exporter
            exporter = self._create_trace_exporter()
            processor = BatchSpanProcessor(
                exporter,
                max_queue_size=self.tracing_config.max_queue_size,
                max_export_batch_size=self.tracing_config.max_export_batch_size,
                export_timeout_millis=self.tracing_config.export_timeout_millis,
            )
            self._tracer_provider.add_span_processor(processor)

            # Set as global
            trace.set_tracer_provider(self._tracer_provider)
            self._tracer = trace.get_tracer(
                self.tracing_config.service_name,
                self.tracing_config.service_version,
            )

            logger.info(
                "Tracing configured",
                exporter=self.tracing_config.exporter.value,
            )

        except ImportError:
            logger.warning("OpenTelemetry SDK not available, using noop tracer")
            self._setup_noop_tracer()

    def _create_trace_exporter(self):
        """Create the appropriate trace exporter."""
        exporter_type = self.tracing_config.exporter

        if exporter_type == ExporterType.CONSOLE:
            from opentelemetry.sdk.trace.export import ConsoleSpanExporter
            return ConsoleSpanExporter()

        elif exporter_type == ExporterType.OTLP:
            try:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
                return OTLPSpanExporter(endpoint=self.tracing_config.otlp_endpoint)
            except ImportError:
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
                return OTLPSpanExporter(endpoint=self.tracing_config.otlp_endpoint)

        elif exporter_type == ExporterType.JAEGER:
            from opentelemetry.exporter.jaeger.thrift import JaegerExporter
            return JaegerExporter(
                collector_endpoint=self.tracing_config.jaeger_endpoint,
            )

        elif exporter_type == ExporterType.ZIPKIN:
            from opentelemetry.exporter.zipkin.json import ZipkinExporter
            return ZipkinExporter(endpoint=self.tracing_config.zipkin_endpoint)

        else:
            from opentelemetry.sdk.trace.export import ConsoleSpanExporter
            return ConsoleSpanExporter()

    def _setup_noop_tracer(self) -> None:
        """Setup no-op tracer when OpenTelemetry is not available."""
        self._tracer = NoopTracer()

    def _setup_metrics(self) -> None:
        """Setup OpenTelemetry metrics."""
        if not self.metrics_config.enabled:
            self._setup_noop_meter()
            return

        try:
            from opentelemetry import metrics
            from opentelemetry.sdk.metrics import MeterProvider
            from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
            from opentelemetry.sdk.resources import Resource, SERVICE_NAME

            # Create resource
            resource = Resource.create({SERVICE_NAME: self.metrics_config.service_name})

            # Setup exporter
            exporter = self._create_metrics_exporter()
            reader = PeriodicExportingMetricReader(
                exporter,
                export_interval_millis=self.metrics_config.export_interval_millis,
            )

            # Create provider
            self._meter_provider = MeterProvider(resource=resource, metric_readers=[reader])

            # Set as global
            metrics.set_meter_provider(self._meter_provider)
            self._meter = metrics.get_meter(self.metrics_config.service_name)

            logger.info(
                "Metrics configured",
                exporter=self.metrics_config.exporter.value,
            )

        except ImportError:
            logger.warning("OpenTelemetry metrics not available, using noop meter")
            self._setup_noop_meter()

    def _create_metrics_exporter(self):
        """Create the appropriate metrics exporter."""
        exporter_type = self.metrics_config.exporter

        if exporter_type == ExporterType.CONSOLE:
            from opentelemetry.sdk.metrics.export import ConsoleMetricExporter
            return ConsoleMetricExporter()

        elif exporter_type == ExporterType.OTLP:
            try:
                from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
                return OTLPMetricExporter(endpoint=self.metrics_config.otlp_endpoint)
            except ImportError:
                from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
                return OTLPMetricExporter(endpoint=self.metrics_config.otlp_endpoint)

        elif exporter_type == ExporterType.PROMETHEUS:
            from opentelemetry.exporter.prometheus import PrometheusMetricReader
            # Note: Prometheus uses a pull model, not a push exporter
            # This would typically start a /metrics endpoint
            from opentelemetry.sdk.metrics.export import ConsoleMetricExporter
            return ConsoleMetricExporter()  # Fallback for simplicity

        else:
            from opentelemetry.sdk.metrics.export import ConsoleMetricExporter
            return ConsoleMetricExporter()

    def _setup_noop_meter(self) -> None:
        """Setup no-op meter when OpenTelemetry is not available."""
        self._meter = NoopMeter()

    def _setup_noop_telemetry(self) -> None:
        """Setup no-op implementations when OpenTelemetry is not available."""
        self._setup_noop_tracer()
        self._setup_noop_meter()
        self._initialized = True

    def shutdown(self) -> None:
        """Shutdown telemetry providers."""
        if self._tracer_provider:
            self._tracer_provider.shutdown()
        if self._meter_provider:
            self._meter_provider.shutdown()

        self._initialized = False
        logger.info("Telemetry shutdown")

    @property
    def tracer(self):
        """Get the tracer instance."""
        return self._tracer

    @property
    def meter(self):
        """Get the meter instance."""
        return self._meter


# No-op implementations for when OpenTelemetry is not available

class NoopSpan:
    """No-op span implementation."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def set_attributes(self, attributes: Dict[str, Any]) -> None:
        pass

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        pass

    def record_exception(self, exception: Exception) -> None:
        pass

    def set_status(self, status: Any) -> None:
        pass

    def end(self, end_time: Optional[int] = None) -> None:
        pass

    def is_recording(self) -> bool:
        return False

    def get_span_context(self):
        return NoopSpanContext()


class NoopSpanContext:
    """No-op span context."""
    trace_id = 0
    span_id = 0
    is_valid = False


class NoopTracer:
    """No-op tracer implementation."""

    def start_span(self, name: str, **kwargs) -> NoopSpan:
        return NoopSpan()

    def start_as_current_span(self, name: str, **kwargs):
        return NoopSpan()


class NoopCounter:
    """No-op counter metric."""

    def add(self, amount: int, attributes: Optional[Dict[str, Any]] = None) -> None:
        pass


class NoopHistogram:
    """No-op histogram metric."""

    def record(self, value: float, attributes: Optional[Dict[str, Any]] = None) -> None:
        pass


class NoopUpDownCounter:
    """No-op up-down counter metric."""

    def add(self, amount: int, attributes: Optional[Dict[str, Any]] = None) -> None:
        pass


class NoopGauge:
    """No-op gauge metric."""

    def set(self, value: float, attributes: Optional[Dict[str, Any]] = None) -> None:
        pass


class NoopMeter:
    """No-op meter implementation."""

    def create_counter(self, name: str, **kwargs) -> NoopCounter:
        return NoopCounter()

    def create_histogram(self, name: str, **kwargs) -> NoopHistogram:
        return NoopHistogram()

    def create_up_down_counter(self, name: str, **kwargs) -> NoopUpDownCounter:
        return NoopUpDownCounter()

    def create_observable_gauge(self, name: str, callbacks: List, **kwargs) -> NoopGauge:
        return NoopGauge()


def configure_telemetry(
    tracing_config: Optional[TracingConfig] = None,
    metrics_config: Optional[MetricsConfig] = None,
) -> TelemetryProvider:
    """
    Configure and initialize telemetry.

    This is the main entry point for setting up observability.
    """
    global _provider

    provider = TelemetryProvider(
        tracing_config=tracing_config,
        metrics_config=metrics_config,
    )
    provider.initialize()
    _provider = provider

    return provider


def get_tracer():
    """Get the configured tracer."""
    global _tracer
    if _tracer is None:
        _tracer = NoopTracer()
    return _tracer


def get_meter():
    """Get the configured meter."""
    global _meter
    if _meter is None:
        _meter = NoopMeter()
    return _meter


def get_provider() -> Optional[TelemetryProvider]:
    """Get the telemetry provider."""
    return _provider
