"""
Plugin Telemetry Module

Provides OpenTelemetry integration for distributed tracing,
metrics, and observability of plugin operations.
"""

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
)

__all__ = [
    "PluginTracer",
    "SpanContext",
    "TracingConfig",
    "trace_plugin_operation",
    "PluginMetrics",
    "MetricsConfig",
    "Counter",
    "Gauge",
    "Histogram",
]
