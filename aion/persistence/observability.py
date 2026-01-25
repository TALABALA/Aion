"""
AION Persistence Observability

True SOTA implementation with:
- OpenTelemetry integration for distributed tracing
- Prometheus metrics export
- Query performance analytics
- Connection pool monitoring
- Real-time health dashboards
- Anomaly detection for slow queries
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Optional, Protocol
import statistics

logger = logging.getLogger(__name__)


# ==================== OpenTelemetry Integration ====================

try:
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode, SpanKind
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.semconv.trace import SpanAttributes
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    trace = None


class DatabaseOperation(str, Enum):
    """Database operation types for tracing."""
    QUERY = "query"
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
    TRANSACTION_BEGIN = "transaction.begin"
    TRANSACTION_COMMIT = "transaction.commit"
    TRANSACTION_ROLLBACK = "transaction.rollback"
    CONNECTION_ACQUIRE = "connection.acquire"
    CONNECTION_RELEASE = "connection.release"


@dataclass
class SpanContext:
    """Context for a traced span."""
    operation: str
    table: Optional[str] = None
    query: Optional[str] = None
    params_count: int = 0
    rows_affected: int = 0
    error: Optional[str] = None
    start_time: float = field(default_factory=time.monotonic)
    end_time: Optional[float] = None

    @property
    def duration_ms(self) -> float:
        if self.end_time is None:
            return (time.monotonic() - self.start_time) * 1000
        return (self.end_time - self.start_time) * 1000


class TracingManager:
    """
    Manages distributed tracing for database operations.

    Features:
    - OpenTelemetry integration
    - Automatic span creation
    - Query parameterization (privacy)
    - Error tracking
    - Latency histograms
    """

    def __init__(
        self,
        service_name: str = "aion-persistence",
        enable_query_logging: bool = False,
        sample_rate: float = 1.0,
    ):
        self.service_name = service_name
        self.enable_query_logging = enable_query_logging
        self.sample_rate = sample_rate
        self._tracer = None
        self._initialized = False

    def initialize(self) -> None:
        """Initialize the tracer."""
        if not OTEL_AVAILABLE:
            logger.warning("OpenTelemetry not available, tracing disabled")
            return

        resource = Resource.create({
            "service.name": self.service_name,
            "service.namespace": "aion",
        })

        provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(provider)
        self._tracer = trace.get_tracer(__name__)
        self._initialized = True

    @asynccontextmanager
    async def trace_operation(
        self,
        operation: DatabaseOperation,
        table: Optional[str] = None,
        query: Optional[str] = None,
    ):
        """
        Context manager for tracing database operations.

        Usage:
            async with tracing.trace_operation(DatabaseOperation.QUERY, "users") as span:
                result = await db.execute("SELECT * FROM users")
                span.rows_affected = len(result)
        """
        context = SpanContext(
            operation=operation.value,
            table=table,
            query=query if self.enable_query_logging else None,
        )

        if not self._initialized or not self._tracer:
            yield context
            return

        with self._tracer.start_as_current_span(
            f"db.{operation.value}",
            kind=SpanKind.CLIENT,
        ) as span:
            try:
                # Set standard attributes
                span.set_attribute("db.system", "aion")
                span.set_attribute("db.operation", operation.value)
                if table:
                    span.set_attribute("db.sql.table", table)
                if query and self.enable_query_logging:
                    # Sanitize query (remove potential sensitive data)
                    span.set_attribute("db.statement", self._sanitize_query(query))

                yield context

                # Set result attributes
                span.set_attribute("db.rows_affected", context.rows_affected)
                span.set_status(Status(StatusCode.OK))

            except Exception as e:
                context.error = str(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise
            finally:
                context.end_time = time.monotonic()

    def _sanitize_query(self, query: str) -> str:
        """Sanitize query by removing potential sensitive data."""
        # Replace string literals with placeholders
        import re
        sanitized = re.sub(r"'[^']*'", "'?'", query)
        sanitized = re.sub(r'"[^"]*"', '"?"', sanitized)
        # Truncate if too long
        if len(sanitized) > 1000:
            sanitized = sanitized[:1000] + "..."
        return sanitized


# ==================== Metrics Collection ====================

@dataclass
class QueryMetrics:
    """Metrics for a single query type."""
    query_pattern: str
    total_calls: int = 0
    total_time_ms: float = 0.0
    min_time_ms: float = float('inf')
    max_time_ms: float = 0.0
    error_count: int = 0
    last_execution: Optional[datetime] = None
    recent_times: deque = field(default_factory=lambda: deque(maxlen=100))

    @property
    def avg_time_ms(self) -> float:
        return self.total_time_ms / self.total_calls if self.total_calls > 0 else 0

    @property
    def p50_time_ms(self) -> float:
        if not self.recent_times:
            return 0
        sorted_times = sorted(self.recent_times)
        return sorted_times[len(sorted_times) // 2]

    @property
    def p95_time_ms(self) -> float:
        if not self.recent_times:
            return 0
        sorted_times = sorted(self.recent_times)
        idx = int(len(sorted_times) * 0.95)
        return sorted_times[min(idx, len(sorted_times) - 1)]

    @property
    def p99_time_ms(self) -> float:
        if not self.recent_times:
            return 0
        sorted_times = sorted(self.recent_times)
        idx = int(len(sorted_times) * 0.99)
        return sorted_times[min(idx, len(sorted_times) - 1)]

    @property
    def std_dev_ms(self) -> float:
        if len(self.recent_times) < 2:
            return 0
        return statistics.stdev(self.recent_times)


@dataclass
class ConnectionPoolMetrics:
    """Metrics for connection pool."""
    pool_size: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    waiting_requests: int = 0
    total_acquisitions: int = 0
    total_releases: int = 0
    acquisition_time_ms: float = 0.0
    timeout_count: int = 0
    health_check_failures: int = 0


class MetricsCollector:
    """
    Collects and exposes persistence layer metrics.

    Features:
    - Query performance tracking
    - Connection pool monitoring
    - Error rate tracking
    - Prometheus-compatible export
    - Real-time analytics
    """

    def __init__(
        self,
        slow_query_threshold_ms: float = 100.0,
        enable_detailed_metrics: bool = True,
    ):
        self.slow_query_threshold_ms = slow_query_threshold_ms
        self.enable_detailed_metrics = enable_detailed_metrics

        self._query_metrics: dict[str, QueryMetrics] = {}
        self._pool_metrics = ConnectionPoolMetrics()
        self._slow_queries: deque = deque(maxlen=1000)
        self._errors: deque = deque(maxlen=1000)
        self._start_time = datetime.utcnow()

    def record_query(
        self,
        query_pattern: str,
        duration_ms: float,
        error: Optional[str] = None,
        rows_affected: int = 0,
    ) -> None:
        """Record a query execution."""
        if query_pattern not in self._query_metrics:
            self._query_metrics[query_pattern] = QueryMetrics(query_pattern=query_pattern)

        metrics = self._query_metrics[query_pattern]
        metrics.total_calls += 1
        metrics.total_time_ms += duration_ms
        metrics.min_time_ms = min(metrics.min_time_ms, duration_ms)
        metrics.max_time_ms = max(metrics.max_time_ms, duration_ms)
        metrics.last_execution = datetime.utcnow()
        metrics.recent_times.append(duration_ms)

        if error:
            metrics.error_count += 1
            self._errors.append({
                "query": query_pattern,
                "error": error,
                "time": datetime.utcnow().isoformat(),
            })

        # Track slow queries
        if duration_ms > self.slow_query_threshold_ms:
            self._slow_queries.append({
                "query": query_pattern,
                "duration_ms": duration_ms,
                "time": datetime.utcnow().isoformat(),
            })

    def record_connection_acquired(self, duration_ms: float) -> None:
        """Record a connection acquisition."""
        self._pool_metrics.total_acquisitions += 1
        self._pool_metrics.acquisition_time_ms = (
            (self._pool_metrics.acquisition_time_ms * (self._pool_metrics.total_acquisitions - 1) + duration_ms)
            / self._pool_metrics.total_acquisitions
        )

    def record_connection_released(self) -> None:
        """Record a connection release."""
        self._pool_metrics.total_releases += 1

    def record_connection_timeout(self) -> None:
        """Record a connection timeout."""
        self._pool_metrics.timeout_count += 1

    def update_pool_status(
        self,
        pool_size: int,
        active: int,
        idle: int,
        waiting: int,
    ) -> None:
        """Update connection pool status."""
        self._pool_metrics.pool_size = pool_size
        self._pool_metrics.active_connections = active
        self._pool_metrics.idle_connections = idle
        self._pool_metrics.waiting_requests = waiting

    def get_query_metrics(self) -> dict[str, dict[str, Any]]:
        """Get all query metrics."""
        return {
            pattern: {
                "total_calls": m.total_calls,
                "total_time_ms": m.total_time_ms,
                "avg_time_ms": m.avg_time_ms,
                "min_time_ms": m.min_time_ms if m.min_time_ms != float('inf') else 0,
                "max_time_ms": m.max_time_ms,
                "p50_time_ms": m.p50_time_ms,
                "p95_time_ms": m.p95_time_ms,
                "p99_time_ms": m.p99_time_ms,
                "error_count": m.error_count,
                "error_rate": m.error_count / m.total_calls if m.total_calls > 0 else 0,
                "last_execution": m.last_execution.isoformat() if m.last_execution else None,
            }
            for pattern, m in self._query_metrics.items()
        }

    def get_pool_metrics(self) -> dict[str, Any]:
        """Get connection pool metrics."""
        return {
            "pool_size": self._pool_metrics.pool_size,
            "active_connections": self._pool_metrics.active_connections,
            "idle_connections": self._pool_metrics.idle_connections,
            "waiting_requests": self._pool_metrics.waiting_requests,
            "utilization": (
                self._pool_metrics.active_connections / self._pool_metrics.pool_size
                if self._pool_metrics.pool_size > 0 else 0
            ),
            "total_acquisitions": self._pool_metrics.total_acquisitions,
            "avg_acquisition_time_ms": self._pool_metrics.acquisition_time_ms,
            "timeout_count": self._pool_metrics.timeout_count,
        }

    def get_slow_queries(self, limit: int = 100) -> list[dict]:
        """Get recent slow queries."""
        return list(self._slow_queries)[-limit:]

    def get_recent_errors(self, limit: int = 100) -> list[dict]:
        """Get recent errors."""
        return list(self._errors)[-limit:]

    def get_summary(self) -> dict[str, Any]:
        """Get overall metrics summary."""
        total_queries = sum(m.total_calls for m in self._query_metrics.values())
        total_errors = sum(m.error_count for m in self._query_metrics.values())
        total_time = sum(m.total_time_ms for m in self._query_metrics.values())

        return {
            "uptime_seconds": (datetime.utcnow() - self._start_time).total_seconds(),
            "total_queries": total_queries,
            "total_errors": total_errors,
            "error_rate": total_errors / total_queries if total_queries > 0 else 0,
            "avg_query_time_ms": total_time / total_queries if total_queries > 0 else 0,
            "slow_query_count": len(self._slow_queries),
            "unique_query_patterns": len(self._query_metrics),
            "pool": self.get_pool_metrics(),
        }

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []

        # Query metrics
        for pattern, m in self._query_metrics.items():
            safe_pattern = pattern.replace('"', '\\"')[:100]
            lines.append(f'aion_db_query_total{{pattern="{safe_pattern}"}} {m.total_calls}')
            lines.append(f'aion_db_query_duration_ms_sum{{pattern="{safe_pattern}"}} {m.total_time_ms}')
            lines.append(f'aion_db_query_errors{{pattern="{safe_pattern}"}} {m.error_count}')

        # Pool metrics
        pm = self._pool_metrics
        lines.append(f'aion_db_pool_size {pm.pool_size}')
        lines.append(f'aion_db_pool_active {pm.active_connections}')
        lines.append(f'aion_db_pool_idle {pm.idle_connections}')
        lines.append(f'aion_db_pool_waiting {pm.waiting_requests}')
        lines.append(f'aion_db_pool_acquisitions_total {pm.total_acquisitions}')
        lines.append(f'aion_db_pool_timeouts_total {pm.timeout_count}')

        return "\n".join(lines)


# ==================== Anomaly Detection ====================

class QueryAnomalyDetector:
    """
    Detects anomalies in query performance.

    Features:
    - Statistical anomaly detection
    - Trend analysis
    - Alert generation
    - Baseline learning
    """

    def __init__(
        self,
        z_score_threshold: float = 3.0,
        min_samples: int = 30,
        alert_callback: Optional[Callable[[dict], None]] = None,
    ):
        self.z_score_threshold = z_score_threshold
        self.min_samples = min_samples
        self.alert_callback = alert_callback
        self._baselines: dict[str, dict[str, float]] = {}

    def update_baseline(self, query_pattern: str, metrics: QueryMetrics) -> None:
        """Update baseline for a query pattern."""
        if len(metrics.recent_times) < self.min_samples:
            return

        times = list(metrics.recent_times)
        self._baselines[query_pattern] = {
            "mean": statistics.mean(times),
            "std": statistics.stdev(times) if len(times) > 1 else 0,
            "updated_at": datetime.utcnow().isoformat(),
        }

    def check_anomaly(
        self,
        query_pattern: str,
        duration_ms: float,
    ) -> Optional[dict[str, Any]]:
        """
        Check if a query execution time is anomalous.

        Returns anomaly details if detected, None otherwise.
        """
        baseline = self._baselines.get(query_pattern)
        if not baseline or baseline["std"] == 0:
            return None

        z_score = (duration_ms - baseline["mean"]) / baseline["std"]

        if abs(z_score) > self.z_score_threshold:
            anomaly = {
                "query_pattern": query_pattern,
                "duration_ms": duration_ms,
                "baseline_mean": baseline["mean"],
                "baseline_std": baseline["std"],
                "z_score": z_score,
                "severity": "high" if abs(z_score) > self.z_score_threshold * 2 else "medium",
                "detected_at": datetime.utcnow().isoformat(),
            }

            if self.alert_callback:
                self.alert_callback(anomaly)

            return anomaly

        return None

    def get_baselines(self) -> dict[str, dict[str, float]]:
        """Get all baselines."""
        return self._baselines.copy()


# ==================== Health Dashboard ====================

class HealthDashboard:
    """
    Provides real-time health information for the persistence layer.

    Features:
    - Component health status
    - Performance indicators
    - Resource utilization
    - Alerts and warnings
    """

    def __init__(
        self,
        metrics_collector: MetricsCollector,
        anomaly_detector: Optional[QueryAnomalyDetector] = None,
    ):
        self.metrics = metrics_collector
        self.anomaly_detector = anomaly_detector
        self._alerts: deque = deque(maxlen=100)
        self._component_status: dict[str, str] = {}

    def update_component_status(self, component: str, status: str) -> None:
        """Update status of a component."""
        self._component_status[component] = status

    def add_alert(self, alert: dict[str, Any]) -> None:
        """Add an alert."""
        alert["timestamp"] = datetime.utcnow().isoformat()
        self._alerts.append(alert)

    def get_health_status(self) -> dict[str, Any]:
        """Get comprehensive health status."""
        summary = self.metrics.get_summary()

        # Determine overall health
        health = "healthy"
        issues = []

        # Check error rate
        if summary["error_rate"] > 0.05:
            health = "degraded"
            issues.append(f"High error rate: {summary['error_rate']:.2%}")

        # Check pool utilization
        pool = summary["pool"]
        if pool["utilization"] > 0.9:
            health = "degraded"
            issues.append(f"High pool utilization: {pool['utilization']:.2%}")

        # Check for waiting requests
        if pool["waiting_requests"] > 10:
            health = "degraded"
            issues.append(f"Requests waiting for connections: {pool['waiting_requests']}")

        # Check slow queries
        if summary["slow_query_count"] > 100:
            issues.append(f"Many slow queries: {summary['slow_query_count']}")

        return {
            "status": health,
            "issues": issues,
            "components": self._component_status,
            "metrics": summary,
            "alerts": list(self._alerts)[-10:],
            "timestamp": datetime.utcnow().isoformat(),
        }

    def get_dashboard_data(self) -> dict[str, Any]:
        """Get data for rendering a dashboard."""
        return {
            "health": self.get_health_status(),
            "queries": self.metrics.get_query_metrics(),
            "pool": self.metrics.get_pool_metrics(),
            "slow_queries": self.metrics.get_slow_queries(20),
            "recent_errors": self.metrics.get_recent_errors(20),
            "baselines": (
                self.anomaly_detector.get_baselines()
                if self.anomaly_detector else {}
            ),
        }


# ==================== Observability Coordinator ====================

class ObservabilityCoordinator:
    """
    Coordinates all observability components.

    Provides a unified interface for:
    - Tracing
    - Metrics
    - Anomaly detection
    - Health monitoring
    """

    def __init__(
        self,
        service_name: str = "aion-persistence",
        slow_query_threshold_ms: float = 100.0,
        enable_tracing: bool = True,
        enable_anomaly_detection: bool = True,
    ):
        self.tracing = TracingManager(
            service_name=service_name,
            enable_query_logging=False,  # Privacy by default
        )
        self.metrics = MetricsCollector(
            slow_query_threshold_ms=slow_query_threshold_ms,
        )
        self.anomaly_detector = (
            QueryAnomalyDetector(alert_callback=self._on_anomaly)
            if enable_anomaly_detection else None
        )
        self.dashboard = HealthDashboard(
            self.metrics,
            self.anomaly_detector,
        )

        if enable_tracing:
            self.tracing.initialize()

    def _on_anomaly(self, anomaly: dict[str, Any]) -> None:
        """Handle detected anomaly."""
        self.dashboard.add_alert({
            "type": "query_anomaly",
            "severity": anomaly["severity"],
            "details": anomaly,
        })
        logger.warning(f"Query anomaly detected: {anomaly}")

    @asynccontextmanager
    async def observe_query(
        self,
        operation: DatabaseOperation,
        query_pattern: str,
        table: Optional[str] = None,
    ):
        """
        Observe a database query with full instrumentation.

        Usage:
            async with observability.observe_query(
                DatabaseOperation.QUERY,
                "SELECT * FROM users WHERE id = ?",
                "users"
            ) as ctx:
                result = await db.execute(query, params)
                ctx.rows_affected = len(result)
        """
        async with self.tracing.trace_operation(operation, table, query_pattern) as span:
            try:
                yield span
            finally:
                # Record metrics
                self.metrics.record_query(
                    query_pattern,
                    span.duration_ms,
                    error=span.error,
                    rows_affected=span.rows_affected,
                )

                # Check for anomalies
                if self.anomaly_detector and not span.error:
                    self.anomaly_detector.check_anomaly(query_pattern, span.duration_ms)

    def get_status(self) -> dict[str, Any]:
        """Get observability status."""
        return self.dashboard.get_health_status()

    def export_metrics(self, format: str = "prometheus") -> str:
        """Export metrics in specified format."""
        if format == "prometheus":
            return self.metrics.export_prometheus()
        elif format == "json":
            import json
            return json.dumps(self.dashboard.get_dashboard_data(), indent=2)
        else:
            raise ValueError(f"Unknown format: {format}")
