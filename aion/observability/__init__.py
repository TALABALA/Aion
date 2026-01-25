"""
AION Observability System

Comprehensive monitoring, tracing, logging, and alerting for AION.
Provides OpenTelemetry-compatible distributed tracing, Prometheus-compatible
metrics, structured logging with trace correlation, and intelligent alerting.

Usage:
    from aion.observability import (
        # Manager
        get_observability,
        ObservabilityManager,
        ObservabilityConfig,

        # Tracing
        traced,
        get_tracing_engine,
        TracingEngine,

        # Metrics
        get_metrics_engine,
        MetricsEngine,
        Counter, Gauge, Histogram, Summary,

        # Logging
        get_logging_engine,
        LoggingEngine,

        # Alerting
        get_alert_engine,
        AlertEngine,

        # Analysis
        get_cost_tracker,
        get_anomaly_detector,
        get_profiler,
        CostTracker,
        AnomalyDetector,
        Profiler,

        # Health
        get_health_checker,
        HealthChecker,

        # Decorators
        metered, logged, profiled, observable,
    )

    # Initialize
    obs = get_observability()
    await obs.initialize()

    # Use decorators
    @traced("my_operation")
    @metered("my_operation")
    async def my_function():
        pass
"""

from __future__ import annotations

# =============================================================================
# Types - Core data structures
# =============================================================================
from aion.observability.types import (
    # Metrics
    Metric,
    MetricType,
    MetricDefinition,
    # Tracing
    Span,
    SpanKind,
    SpanStatus,
    SpanContext,
    SpanEvent,
    SpanLink,
    Trace,
    # Logging
    LogEntry,
    LogLevel,
    # Alerting
    Alert,
    AlertRule,
    AlertSeverity,
    AlertState,
    # Health
    HealthCheck,
    HealthStatus,
    SystemHealth,
    # Cost
    CostRecord,
    CostBudget,
    ResourceType,
    # Anomaly
    Anomaly,
    AnomalyType,
    # Profiling
    ProfileSample,
    ProfileReport,
    # Audit
    AuditEvent,
    AuditAction,
)

# =============================================================================
# Collector - Central telemetry ingestion
# =============================================================================
from aion.observability.collector import (
    TelemetryCollector,
    ExporterConfig,
    CollectorStats,
)

# =============================================================================
# Context - Request context propagation
# =============================================================================
from aion.observability.context import (
    ObservabilityContext,
    ContextManager,
    get_current_context,
    get_current_span,
    get_current_trace_id,
    get_request_id,
    set_request_id,
    get_user_id,
    set_user_id,
    get_session_id,
    set_session_id,
    get_agent_id,
    set_agent_id,
    get_goal_id,
    set_goal_id,
    get_baggage,
    set_baggage,
    with_context,
    with_request,
)

# =============================================================================
# Metrics - Metrics collection and export
# =============================================================================
from aion.observability.metrics import (
    MetricsEngine,
)
from aion.observability.metrics.registry import (
    MetricBase,
    LabeledMetric,
    Counter,
    Gauge,
    Histogram,
    Summary,
    Info,
    MetricRegistry,
    get_default_registry,
    set_default_registry,
    # Pre-defined metrics
    REQUESTS_TOTAL,
    REQUEST_DURATION,
    ACTIVE_REQUESTS,
    TOKENS_USED,
    TOKEN_COST,
    LLM_LATENCY,
    LLM_REQUESTS,
    TOOL_CALLS,
    TOOL_DURATION,
    ACTIVE_AGENTS,
    AGENT_TASKS,
    ACTIVE_GOALS,
    GOALS_TOTAL,
    MEMORY_OPERATIONS,
    MEMORY_SEARCH_DURATION,
    ERRORS,
    KG_QUERIES,
    KG_ENTITIES,
    SPANS_TOTAL,
    SPAN_DURATION,
)
from aion.observability.metrics.exporters import (
    MetricExporter,
    PrometheusExporter,
    StatsDBatchExporter,
    OTLPMetricExporter,
    HTTPPushExporter,
    ConsoleExporter,
    InMemoryExporter,
    MultiExporter,
)

# =============================================================================
# Tracing - Distributed tracing
# =============================================================================
from aion.observability.tracing import (
    TracingEngine,
    SpanContextManager,
    traced,
    # Propagation
    ContextPropagator,
    W3CTraceContextPropagator,
    B3Propagator,
    CompositePropagator,
    # Sampling
    Sampler,
    AlwaysOnSampler,
    AlwaysOffSampler,
    TraceIdRatioSampler,
    ParentBasedSampler,
    RateLimitingSampler,
    AdaptiveSampler,
)

# =============================================================================
# Logging - Structured logging with trace correlation
# =============================================================================
from aion.observability.logging import (
    LoggingEngine,
    ObservabilityLogger,
    ObservabilityHandler,
)
from aion.observability.logging.correlation import (
    CorrelatedLogger,
    SpanLogger,
    add_trace_context_processor,
    add_service_context_processor,
    configure_correlated_logging,
)

# =============================================================================
# Alerting - Alert processing and notification
# =============================================================================
from aion.observability.alerting import (
    AlertEngine,
)
from aion.observability.alerting.channels import (
    AlertChannel,
    WebhookChannel,
    SlackChannel,
    EmailChannel,
    PagerDutyChannel,
    DiscordChannel,
    OpsGenieChannel,
    LogChannel,
    ConsoleChannel,
    CompositeChannel,
)

# =============================================================================
# Analysis - Cost, Anomaly, Profiling
# =============================================================================
from aion.observability.analysis import (
    CostTracker,
    AnomalyDetector,
    Profiler,
)
from aion.observability.analysis.cost import (
    TokenPricing,
    CostAggregation,
)
from aion.observability.analysis.anomaly import (
    MetricState,
    IsolationForestDetector,
)
from aion.observability.analysis.profiler import (
    OperationProfile,
    profile,
    profile_method,
)

# =============================================================================
# Health - System health monitoring
# =============================================================================
from aion.observability.health import (
    HealthChecker,
)

# =============================================================================
# Storage - Persistence backends
# =============================================================================
from aion.observability.storage import (
    StorageBackend,
    MetricStore,
    TraceStore,
    LogStore,
    InMemoryMetricStore,
    InMemoryTraceStore,
    InMemoryLogStore,
)

# =============================================================================
# Instrumentation - Decorators and middleware
# =============================================================================
from aion.observability.instrumentation import (
    # Decorators
    metered,
    logged,
    profiled,
    observable,
    with_cost_tracking,
    # Middleware
    ObservabilityMiddleware,
    WebSocketObservabilityMiddleware,
)

# =============================================================================
# Manager - Central coordinator
# =============================================================================
from aion.observability.manager import (
    ObservabilityManager,
    ObservabilityConfig,
    get_observability,
    get_tracing_engine,
    get_metrics_engine,
    get_logging_engine,
    get_alert_engine,
    get_cost_tracker,
    get_anomaly_detector,
    get_profiler,
    get_health_checker,
)


# =============================================================================
# Version
# =============================================================================
__version__ = "1.0.0"


# =============================================================================
# Public API
# =============================================================================
__all__ = [
    # Version
    "__version__",

    # Manager
    "ObservabilityManager",
    "ObservabilityConfig",
    "get_observability",

    # Types - Metrics
    "Metric",
    "MetricType",
    "MetricDefinition",

    # Types - Tracing
    "Span",
    "SpanKind",
    "SpanStatus",
    "SpanContext",
    "SpanEvent",
    "SpanLink",
    "Trace",

    # Types - Logging
    "LogEntry",
    "LogLevel",

    # Types - Alerting
    "Alert",
    "AlertRule",
    "AlertSeverity",
    "AlertState",

    # Types - Health
    "HealthCheck",
    "HealthStatus",
    "SystemHealth",

    # Types - Cost
    "CostRecord",
    "CostBudget",
    "ResourceType",

    # Types - Anomaly
    "Anomaly",
    "AnomalyType",

    # Types - Profiling
    "ProfileSample",
    "ProfileReport",

    # Types - Audit
    "AuditEvent",
    "AuditAction",

    # Collector
    "TelemetryCollector",
    "ExporterConfig",
    "CollectorStats",

    # Context
    "ObservabilityContext",
    "ContextManager",
    "get_current_context",
    "get_current_span",
    "get_current_trace_id",
    "get_request_id",
    "set_request_id",
    "get_user_id",
    "set_user_id",
    "get_session_id",
    "set_session_id",
    "get_agent_id",
    "set_agent_id",
    "get_goal_id",
    "set_goal_id",
    "get_baggage",
    "set_baggage",
    "with_context",
    "with_request",

    # Metrics Engine
    "MetricsEngine",
    "get_metrics_engine",

    # Metrics Registry
    "MetricBase",
    "LabeledMetric",
    "Counter",
    "Gauge",
    "Histogram",
    "Summary",
    "Info",
    "MetricRegistry",
    "get_default_registry",
    "set_default_registry",

    # Pre-defined Metrics
    "REQUESTS_TOTAL",
    "REQUEST_DURATION",
    "ACTIVE_REQUESTS",
    "TOKENS_USED",
    "TOKEN_COST",
    "LLM_LATENCY",
    "LLM_REQUESTS",
    "TOOL_CALLS",
    "TOOL_DURATION",
    "ACTIVE_AGENTS",
    "AGENT_TASKS",
    "ACTIVE_GOALS",
    "GOALS_TOTAL",
    "MEMORY_OPERATIONS",
    "MEMORY_SEARCH_DURATION",
    "ERRORS",
    "KG_QUERIES",
    "KG_ENTITIES",
    "SPANS_TOTAL",
    "SPAN_DURATION",

    # Metric Exporters
    "MetricExporter",
    "PrometheusExporter",
    "StatsDBatchExporter",
    "OTLPMetricExporter",
    "HTTPPushExporter",
    "ConsoleExporter",
    "InMemoryExporter",
    "MultiExporter",

    # Tracing
    "TracingEngine",
    "SpanContextManager",
    "traced",
    "get_tracing_engine",

    # Propagation
    "ContextPropagator",
    "W3CTraceContextPropagator",
    "B3Propagator",
    "CompositePropagator",

    # Sampling
    "Sampler",
    "AlwaysOnSampler",
    "AlwaysOffSampler",
    "TraceIdRatioSampler",
    "ParentBasedSampler",
    "RateLimitingSampler",
    "AdaptiveSampler",

    # Logging
    "LoggingEngine",
    "ObservabilityLogger",
    "ObservabilityHandler",
    "get_logging_engine",

    # Log Correlation
    "CorrelatedLogger",
    "SpanLogger",
    "add_trace_context_processor",
    "add_service_context_processor",
    "configure_correlated_logging",

    # Alerting
    "AlertEngine",
    "get_alert_engine",

    # Alert Channels
    "AlertChannel",
    "WebhookChannel",
    "SlackChannel",
    "EmailChannel",
    "PagerDutyChannel",
    "DiscordChannel",
    "OpsGenieChannel",
    "LogChannel",
    "ConsoleChannel",
    "CompositeChannel",

    # Cost Tracking
    "CostTracker",
    "TokenPricing",
    "CostAggregation",
    "get_cost_tracker",

    # Anomaly Detection
    "AnomalyDetector",
    "MetricState",
    "IsolationForestDetector",
    "get_anomaly_detector",

    # Profiling
    "Profiler",
    "OperationProfile",
    "profile",
    "profile_method",
    "get_profiler",

    # Health
    "HealthChecker",
    "get_health_checker",

    # Storage
    "StorageBackend",
    "MetricStore",
    "TraceStore",
    "LogStore",
    "InMemoryMetricStore",
    "InMemoryTraceStore",
    "InMemoryLogStore",

    # Instrumentation Decorators
    "metered",
    "logged",
    "profiled",
    "observable",
    "with_cost_tracking",

    # Middleware
    "ObservabilityMiddleware",
    "WebSocketObservabilityMiddleware",
]
