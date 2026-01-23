"""
AION MCP Metrics and Observability

Production-grade observability with:
- OpenTelemetry distributed tracing
- Prometheus metrics
- Structured logging integration
- Health check endpoints
"""

from __future__ import annotations

import asyncio
import functools
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

import structlog

logger = structlog.get_logger(__name__)


# ============================================
# OpenTelemetry Integration
# ============================================

# Try to import OpenTelemetry
try:
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode, SpanKind
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.semconv.trace import SpanAttributes

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    trace = None
    Status = None
    StatusCode = None
    SpanKind = None


# Try to import Prometheus client
try:
    from prometheus_client import (
        Counter,
        Histogram,
        Gauge,
        Summary,
        Info,
        CollectorRegistry,
        generate_latest,
        CONTENT_TYPE_LATEST,
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    Counter = None
    Histogram = None
    Gauge = None


class MCPTracer:
    """
    OpenTelemetry tracer for MCP operations.

    Provides distributed tracing across:
    - Server connections
    - Tool calls
    - Resource reads
    - Prompt retrievals
    """

    def __init__(
        self,
        service_name: str = "aion-mcp",
        service_version: str = "1.0.0",
        exporter: Optional[Any] = None,
    ):
        """
        Initialize tracer.

        Args:
            service_name: Service name for traces
            service_version: Service version
            exporter: Optional span exporter (e.g., JaegerExporter, OTLPSpanExporter)
        """
        self.service_name = service_name
        self.service_version = service_version
        self._tracer: Optional[Any] = None
        self._provider: Optional[Any] = None

        if OTEL_AVAILABLE:
            self._initialize_otel(exporter)

    def _initialize_otel(self, exporter: Optional[Any]) -> None:
        """Initialize OpenTelemetry provider."""
        resource = Resource.create({
            "service.name": self.service_name,
            "service.version": self.service_version,
        })

        self._provider = TracerProvider(resource=resource)

        if exporter:
            processor = BatchSpanProcessor(exporter)
            self._provider.add_span_processor(processor)

        trace.set_tracer_provider(self._provider)
        self._tracer = trace.get_tracer(self.service_name, self.service_version)

        logger.info("OpenTelemetry tracer initialized", service=self.service_name)

    @asynccontextmanager
    async def span(
        self,
        name: str,
        kind: Optional[Any] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ):
        """
        Create a traced span.

        Args:
            name: Span name
            kind: Span kind (CLIENT, SERVER, INTERNAL, etc.)
            attributes: Span attributes

        Yields:
            Span object (or None if tracing unavailable)
        """
        if not self._tracer:
            yield None
            return

        span_kind = kind or SpanKind.INTERNAL

        with self._tracer.start_as_current_span(
            name,
            kind=span_kind,
            attributes=attributes or {},
        ) as span:
            try:
                yield span
            except Exception as e:
                if span:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                raise

    def trace_tool_call(
        self,
        server_name: str,
        tool_name: str,
    ) -> Callable:
        """
        Decorator for tracing tool calls.

        Args:
            server_name: MCP server name
            tool_name: Tool name
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                attributes = {
                    "mcp.server.name": server_name,
                    "mcp.tool.name": tool_name,
                    "mcp.operation.type": "tool_call",
                }

                async with self.span(
                    f"mcp.tool.{tool_name}",
                    kind=SpanKind.CLIENT if OTEL_AVAILABLE else None,
                    attributes=attributes,
                ) as span:
                    start_time = time.monotonic()
                    try:
                        result = await func(*args, **kwargs)

                        if span:
                            span.set_attribute("mcp.tool.success", True)
                            span.set_status(Status(StatusCode.OK))

                        return result

                    except Exception as e:
                        if span:
                            span.set_attribute("mcp.tool.success", False)
                            span.set_attribute("mcp.tool.error", str(e))

                        raise

                    finally:
                        duration_ms = (time.monotonic() - start_time) * 1000
                        if span:
                            span.set_attribute("mcp.tool.duration_ms", duration_ms)

            return wrapper
        return decorator

    def trace_server_connection(
        self,
        server_name: str,
        transport: str,
    ) -> Callable:
        """
        Decorator for tracing server connections.

        Args:
            server_name: MCP server name
            transport: Transport type
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                attributes = {
                    "mcp.server.name": server_name,
                    "mcp.server.transport": transport,
                    "mcp.operation.type": "connect",
                }

                async with self.span(
                    f"mcp.server.connect.{server_name}",
                    kind=SpanKind.CLIENT if OTEL_AVAILABLE else None,
                    attributes=attributes,
                ) as span:
                    try:
                        result = await func(*args, **kwargs)

                        if span:
                            span.set_attribute("mcp.server.connected", True)
                            span.set_status(Status(StatusCode.OK))

                        return result

                    except Exception as e:
                        if span:
                            span.set_attribute("mcp.server.connected", False)
                            span.set_attribute("mcp.server.error", str(e))

                        raise

            return wrapper
        return decorator

    def inject_context(self, carrier: Dict[str, str]) -> None:
        """Inject trace context into carrier for propagation."""
        if OTEL_AVAILABLE:
            propagator = TraceContextTextMapPropagator()
            propagator.inject(carrier)

    def extract_context(self, carrier: Dict[str, str]) -> Optional[Any]:
        """Extract trace context from carrier."""
        if OTEL_AVAILABLE:
            propagator = TraceContextTextMapPropagator()
            return propagator.extract(carrier)
        return None

    def shutdown(self) -> None:
        """Shutdown tracer and flush pending spans."""
        if self._provider:
            self._provider.shutdown()


# ============================================
# Prometheus Metrics
# ============================================

class MCPMetrics:
    """
    Prometheus metrics for MCP operations.

    Metrics collected:
    - Connection counts and status
    - Tool call latencies and counts
    - Error rates
    - Resource usage
    """

    def __init__(
        self,
        namespace: str = "aion_mcp",
        registry: Optional[Any] = None,
    ):
        """
        Initialize metrics.

        Args:
            namespace: Metrics namespace prefix
            registry: Optional Prometheus registry
        """
        self.namespace = namespace
        self._registry = registry

        if not PROMETHEUS_AVAILABLE:
            logger.warning("Prometheus client not available - metrics disabled")
            return

        self._init_metrics()

    def _init_metrics(self) -> None:
        """Initialize Prometheus metrics."""
        kwargs = {}
        if self._registry:
            kwargs["registry"] = self._registry

        # Connection metrics
        self.connections_total = Counter(
            f"{self.namespace}_connections_total",
            "Total MCP server connections",
            ["server", "transport", "status"],
            **kwargs,
        )

        self.connections_active = Gauge(
            f"{self.namespace}_connections_active",
            "Currently active MCP connections",
            ["server"],
            **kwargs,
        )

        self.connection_duration = Histogram(
            f"{self.namespace}_connection_duration_seconds",
            "MCP connection setup duration",
            ["server", "transport"],
            buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
            **kwargs,
        )

        # Tool metrics
        self.tool_calls_total = Counter(
            f"{self.namespace}_tool_calls_total",
            "Total MCP tool calls",
            ["server", "tool", "status"],
            **kwargs,
        )

        self.tool_call_duration = Histogram(
            f"{self.namespace}_tool_call_duration_seconds",
            "MCP tool call duration",
            ["server", "tool"],
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
            **kwargs,
        )

        self.tool_call_errors = Counter(
            f"{self.namespace}_tool_call_errors_total",
            "Total MCP tool call errors",
            ["server", "tool", "error_type"],
            **kwargs,
        )

        # Resource metrics
        self.resource_reads_total = Counter(
            f"{self.namespace}_resource_reads_total",
            "Total MCP resource reads",
            ["server", "status"],
            **kwargs,
        )

        self.resource_read_duration = Histogram(
            f"{self.namespace}_resource_read_duration_seconds",
            "MCP resource read duration",
            ["server"],
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
            **kwargs,
        )

        # Prompt metrics
        self.prompt_gets_total = Counter(
            f"{self.namespace}_prompt_gets_total",
            "Total MCP prompt retrievals",
            ["server", "status"],
            **kwargs,
        )

        # Circuit breaker metrics
        self.circuit_breaker_state = Gauge(
            f"{self.namespace}_circuit_breaker_state",
            "Circuit breaker state (0=closed, 1=half_open, 2=open)",
            ["server"],
            **kwargs,
        )

        self.circuit_breaker_failures = Counter(
            f"{self.namespace}_circuit_breaker_failures_total",
            "Circuit breaker failure count",
            ["server"],
            **kwargs,
        )

        # Rate limiter metrics
        self.rate_limit_rejected = Counter(
            f"{self.namespace}_rate_limit_rejected_total",
            "Requests rejected by rate limiter",
            ["server"],
            **kwargs,
        )

        # Cache metrics
        self.cache_hits = Counter(
            f"{self.namespace}_cache_hits_total",
            "Cache hit count",
            ["cache_name"],
            **kwargs,
        )

        self.cache_misses = Counter(
            f"{self.namespace}_cache_misses_total",
            "Cache miss count",
            ["cache_name"],
            **kwargs,
        )

        # Info metric
        self.info = Info(
            f"{self.namespace}_info",
            "MCP integration info",
            **kwargs,
        )

        logger.info("Prometheus metrics initialized", namespace=self.namespace)

    def record_connection(
        self,
        server: str,
        transport: str,
        success: bool,
        duration: float,
    ) -> None:
        """Record a connection attempt."""
        if not PROMETHEUS_AVAILABLE:
            return

        status = "success" if success else "failure"
        self.connections_total.labels(server=server, transport=transport, status=status).inc()
        self.connection_duration.labels(server=server, transport=transport).observe(duration)

        if success:
            self.connections_active.labels(server=server).inc()

    def record_disconnection(self, server: str) -> None:
        """Record a disconnection."""
        if not PROMETHEUS_AVAILABLE:
            return

        self.connections_active.labels(server=server).dec()

    def record_tool_call(
        self,
        server: str,
        tool: str,
        success: bool,
        duration: float,
        error_type: Optional[str] = None,
    ) -> None:
        """Record a tool call."""
        if not PROMETHEUS_AVAILABLE:
            return

        status = "success" if success else "failure"
        self.tool_calls_total.labels(server=server, tool=tool, status=status).inc()
        self.tool_call_duration.labels(server=server, tool=tool).observe(duration)

        if not success and error_type:
            self.tool_call_errors.labels(server=server, tool=tool, error_type=error_type).inc()

    def record_resource_read(
        self,
        server: str,
        success: bool,
        duration: float,
    ) -> None:
        """Record a resource read."""
        if not PROMETHEUS_AVAILABLE:
            return

        status = "success" if success else "failure"
        self.resource_reads_total.labels(server=server, status=status).inc()
        self.resource_read_duration.labels(server=server).observe(duration)

    def record_prompt_get(
        self,
        server: str,
        success: bool,
    ) -> None:
        """Record a prompt retrieval."""
        if not PROMETHEUS_AVAILABLE:
            return

        status = "success" if success else "failure"
        self.prompt_gets_total.labels(server=server, status=status).inc()

    def record_circuit_breaker_state(
        self,
        server: str,
        state: str,
    ) -> None:
        """Record circuit breaker state change."""
        if not PROMETHEUS_AVAILABLE:
            return

        state_value = {"closed": 0, "half_open": 1, "open": 2}.get(state, -1)
        self.circuit_breaker_state.labels(server=server).set(state_value)

    def record_circuit_breaker_failure(self, server: str) -> None:
        """Record circuit breaker failure."""
        if not PROMETHEUS_AVAILABLE:
            return

        self.circuit_breaker_failures.labels(server=server).inc()

    def record_rate_limit_rejected(self, server: str) -> None:
        """Record rate limit rejection."""
        if not PROMETHEUS_AVAILABLE:
            return

        self.rate_limit_rejected.labels(server=server).inc()

    def record_cache_hit(self, cache_name: str) -> None:
        """Record cache hit."""
        if not PROMETHEUS_AVAILABLE:
            return

        self.cache_hits.labels(cache_name=cache_name).inc()

    def record_cache_miss(self, cache_name: str) -> None:
        """Record cache miss."""
        if not PROMETHEUS_AVAILABLE:
            return

        self.cache_misses.labels(cache_name=cache_name).inc()

    def set_info(self, info: Dict[str, str]) -> None:
        """Set info metric."""
        if not PROMETHEUS_AVAILABLE:
            return

        self.info.info(info)

    def get_metrics(self) -> bytes:
        """Get Prometheus metrics in text format."""
        if not PROMETHEUS_AVAILABLE:
            return b""

        if self._registry:
            return generate_latest(self._registry)
        return generate_latest()


# ============================================
# Health Checks
# ============================================

class HealthStatus(str, Enum):
    """Health check status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    name: str
    status: HealthStatus
    message: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "details": self.details,
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp.isoformat(),
        }


class MCPHealthChecker:
    """
    Health checker for MCP integration.

    Checks:
    - Server connectivity
    - Circuit breaker status
    - Rate limiter capacity
    - Cache health
    """

    def __init__(self, manager: Any):
        """
        Initialize health checker.

        Args:
            manager: MCPManager instance
        """
        self._manager = manager
        self._checks: List[Callable] = []

    def add_check(self, check: Callable) -> None:
        """Add a custom health check."""
        self._checks.append(check)

    async def check_all(self) -> Dict[str, Any]:
        """
        Run all health checks.

        Returns:
            Combined health check results
        """
        results = []

        # Check server connectivity
        results.append(await self._check_servers())

        # Run custom checks
        for check in self._checks:
            try:
                result = await check()
                results.append(result)
            except Exception as e:
                results.append(HealthCheckResult(
                    name="custom_check",
                    status=HealthStatus.UNHEALTHY,
                    message=str(e),
                ))

        # Determine overall status
        overall_status = HealthStatus.HEALTHY

        for result in results:
            if result.status == HealthStatus.UNHEALTHY:
                overall_status = HealthStatus.UNHEALTHY
                break
            elif result.status == HealthStatus.DEGRADED:
                overall_status = HealthStatus.DEGRADED

        return {
            "status": overall_status.value,
            "checks": [r.to_dict() for r in results],
            "timestamp": datetime.now().isoformat(),
        }

    async def _check_servers(self) -> HealthCheckResult:
        """Check MCP server health."""
        start = time.monotonic()

        try:
            connected = self._manager.get_connected_servers()
            total = len(self._manager.registry.get_enabled_servers())

            if not total:
                status = HealthStatus.HEALTHY
                message = "No servers configured"
            elif len(connected) == total:
                status = HealthStatus.HEALTHY
                message = f"All {total} servers connected"
            elif len(connected) > 0:
                status = HealthStatus.DEGRADED
                message = f"{len(connected)}/{total} servers connected"
            else:
                status = HealthStatus.UNHEALTHY
                message = "No servers connected"

            return HealthCheckResult(
                name="mcp_servers",
                status=status,
                message=message,
                details={
                    "connected": connected,
                    "total_enabled": total,
                },
                duration_ms=(time.monotonic() - start) * 1000,
            )

        except Exception as e:
            return HealthCheckResult(
                name="mcp_servers",
                status=HealthStatus.UNHEALTHY,
                message=str(e),
                duration_ms=(time.monotonic() - start) * 1000,
            )


# ============================================
# Instrumentation Decorator
# ============================================

def instrument(
    tracer: Optional[MCPTracer] = None,
    metrics: Optional[MCPMetrics] = None,
    operation: str = "unknown",
    server: str = "unknown",
):
    """
    Combined instrumentation decorator.

    Adds both tracing and metrics to a function.

    Args:
        tracer: MCPTracer instance
        metrics: MCPMetrics instance
        operation: Operation type (tool_call, resource_read, etc.)
        server: Server name
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.monotonic()
            success = False
            error_type = None

            try:
                # Start trace span
                span_context = None
                if tracer:
                    span_context = tracer.span(
                        f"mcp.{operation}",
                        attributes={
                            "mcp.server": server,
                            "mcp.operation": operation,
                        },
                    )

                async with span_context if span_context else asynccontextmanager(lambda: iter([None]))():
                    result = await func(*args, **kwargs)
                    success = True
                    return result

            except Exception as e:
                error_type = type(e).__name__
                raise

            finally:
                duration = time.monotonic() - start_time

                # Record metrics
                if metrics:
                    if operation == "tool_call":
                        tool_name = kwargs.get("tool_name", "unknown")
                        metrics.record_tool_call(
                            server=server,
                            tool=tool_name,
                            success=success,
                            duration=duration,
                            error_type=error_type,
                        )
                    elif operation == "resource_read":
                        metrics.record_resource_read(
                            server=server,
                            success=success,
                            duration=duration,
                        )
                    elif operation == "prompt_get":
                        metrics.record_prompt_get(
                            server=server,
                            success=success,
                        )

        return wrapper
    return decorator


# ============================================
# Global Instances
# ============================================

# Global tracer and metrics instances (initialized by MCPManager)
_global_tracer: Optional[MCPTracer] = None
_global_metrics: Optional[MCPMetrics] = None


def get_tracer() -> Optional[MCPTracer]:
    """Get global tracer instance."""
    return _global_tracer


def get_metrics() -> Optional[MCPMetrics]:
    """Get global metrics instance."""
    return _global_metrics


def init_observability(
    tracer: Optional[MCPTracer] = None,
    metrics: Optional[MCPMetrics] = None,
) -> None:
    """
    Initialize global observability instances.

    Args:
        tracer: MCPTracer instance
        metrics: MCPMetrics instance
    """
    global _global_tracer, _global_metrics
    _global_tracer = tracer
    _global_metrics = metrics

    logger.info(
        "MCP observability initialized",
        tracing_enabled=tracer is not None,
        metrics_enabled=metrics is not None,
    )
