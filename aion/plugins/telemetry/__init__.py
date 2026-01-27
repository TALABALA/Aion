"""
Plugin Telemetry Module

Provides OpenTelemetry integration for distributed tracing,
metrics, and observability of plugin operations.

Includes:
- Standalone tracing implementation (no dependencies)
- Real OpenTelemetry SDK integration (requires opentelemetry-sdk)
"""

# Standalone implementation (always available)
from aion.plugins.telemetry.tracing import (
    PluginTracer,
    SpanContext,
    TracingConfig,
    trace_plugin_operation,
)
from aion.plugins.telemetry.metrics import (
    PluginMetrics,
    MetricsConfig,
    Counter,
    Gauge,
    Histogram,
    MetricsManager,
)

# Real OpenTelemetry SDK integration
from aion.plugins.telemetry.otel import (
    OTelManager,
    OTelConfig,
    PluginTracer as OTelPluginTracer,
    PluginMeter,
    MetricCounter,
    MetricHistogram,
    get_tracer,
    get_meter,
    configure_otel,
    trace_operation,
    OTEL_AVAILABLE,
    OTLP_AVAILABLE,
    JAEGER_AVAILABLE,
    PROMETHEUS_AVAILABLE,
)

__all__ = [
    # Standalone implementation
    "PluginTracer",
    "SpanContext",
    "TracingConfig",
    "trace_plugin_operation",
    "PluginMetrics",
    "MetricsConfig",
    "Counter",
    "Gauge",
    "Histogram",
    "MetricsManager",

    # OpenTelemetry SDK integration
    "OTelManager",
    "OTelConfig",
    "OTelPluginTracer",
    "PluginMeter",
    "MetricCounter",
    "MetricHistogram",
    "get_tracer",
    "get_meter",
    "configure_otel",
    "trace_operation",

    # Feature flags
    "OTEL_AVAILABLE",
    "OTLP_AVAILABLE",
    "JAEGER_AVAILABLE",
    "PROMETHEUS_AVAILABLE",
]
