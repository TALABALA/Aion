"""
AION Observability - OpenTelemetry Integration

Complete observability stack for workflow automation:
- Distributed tracing
- Metrics collection
- Log correlation
"""

from aion.automation.observability.telemetry import (
    TelemetryProvider,
    TracingConfig,
    MetricsConfig,
    configure_telemetry,
    get_tracer,
    get_meter,
)
from aion.automation.observability.tracing import (
    WorkflowTracer,
    trace_workflow,
    trace_step,
    trace_action,
)
from aion.automation.observability.metrics import (
    WorkflowMetrics,
    MetricRegistry,
)
from aion.automation.observability.dashboards import (
    DashboardGenerator,
    GrafanaDashboard,
)

__all__ = [
    "TelemetryProvider",
    "TracingConfig",
    "MetricsConfig",
    "configure_telemetry",
    "get_tracer",
    "get_meter",
    "WorkflowTracer",
    "trace_workflow",
    "trace_step",
    "trace_action",
    "WorkflowMetrics",
    "MetricRegistry",
    "DashboardGenerator",
    "GrafanaDashboard",
]
