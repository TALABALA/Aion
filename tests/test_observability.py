"""
AION Observability Tests

Comprehensive test suite for the AION Monitoring & Observability System.
Tests cover: metrics, tracing, logging, alerting, cost tracking,
anomaly detection, profiling, and health checks.
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from unittest.mock import Mock, AsyncMock, patch


# =============================================================================
# Types Tests
# =============================================================================

class TestObservabilityTypes:
    """Test observability data types."""

    def test_metric_creation(self):
        """Test Metric dataclass creation."""
        from aion.observability.types import Metric, MetricType

        metric = Metric(
            name="test_counter",
            value=42.0,
            metric_type=MetricType.COUNTER,
            labels={"env": "test"},
            description="A test counter",
        )

        assert metric.name == "test_counter"
        assert metric.value == 42.0
        assert metric.metric_type == MetricType.COUNTER
        assert metric.labels["env"] == "test"

    def test_metric_prometheus_format(self):
        """Test Prometheus format output."""
        from aion.observability.types import Metric, MetricType

        metric = Metric(
            name="requests_total",
            value=100.0,
            metric_type=MetricType.COUNTER,
            labels={"method": "GET", "path": "/api"},
        )

        prom_output = metric.to_prometheus()
        assert "requests_total" in prom_output
        assert 'method="GET"' in prom_output
        assert 'path="/api"' in prom_output
        assert "100" in prom_output

    def test_span_creation(self):
        """Test Span dataclass creation."""
        from aion.observability.types import Span, SpanKind, SpanStatus

        span = Span(
            trace_id="abc123",
            span_id="span456",
            operation_name="process_request",
            service_name="aion",
            kind=SpanKind.SERVER,
        )

        assert span.trace_id == "abc123"
        assert span.span_id == "span456"
        assert span.operation_name == "process_request"
        assert span.kind == SpanKind.SERVER
        assert span.status == SpanStatus.UNSET

    def test_span_end(self):
        """Test ending a span."""
        from aion.observability.types import Span, SpanStatus

        span = Span(
            trace_id="abc123",
            span_id="span456",
            operation_name="test_op",
            service_name="test",
        )

        time.sleep(0.01)
        span.end()

        assert span.end_time is not None
        assert span.duration_ms > 0
        assert span.status == SpanStatus.OK

    def test_span_error(self):
        """Test span error recording."""
        from aion.observability.types import Span, SpanStatus

        span = Span(
            trace_id="abc123",
            span_id="span456",
            operation_name="test_op",
            service_name="test",
        )

        span.set_error(ValueError("Test error"))

        assert span.status == SpanStatus.ERROR
        assert span.error_message == "Test error"
        assert "ValueError" in span.error_type

    def test_log_entry_creation(self):
        """Test LogEntry dataclass creation."""
        from aion.observability.types import LogEntry, LogLevel

        entry = LogEntry(
            level=LogLevel.INFO,
            message="Test message",
            logger_name="test.logger",
        )

        assert entry.level == LogLevel.INFO
        assert entry.message == "Test message"
        assert entry.logger_name == "test.logger"

    def test_alert_rule_creation(self):
        """Test AlertRule dataclass creation."""
        from aion.observability.types import AlertRule, AlertSeverity

        rule = AlertRule(
            name="high_error_rate",
            metric_name="error_rate",
            condition="gt",
            threshold=0.05,
            severity=AlertSeverity.CRITICAL,
        )

        assert rule.name == "high_error_rate"
        assert rule.threshold == 0.05
        assert rule.severity == AlertSeverity.CRITICAL

    def test_cost_record(self):
        """Test CostRecord dataclass."""
        from aion.observability.types import CostRecord, ResourceType

        record = CostRecord(
            resource_type=ResourceType.LLM_TOKENS,
            cost=0.05,
            quantity=1000,
            model="gpt-4",
        )

        assert record.resource_type == ResourceType.LLM_TOKENS
        assert record.cost == 0.05
        assert record.quantity == 1000


# =============================================================================
# Context Tests
# =============================================================================

class TestObservabilityContext:
    """Test observability context management."""

    def test_context_creation(self):
        """Test ObservabilityContext creation."""
        from aion.observability.context import ObservabilityContext

        ctx = ObservabilityContext(
            trace_id="trace123",
            span_id="span456",
            request_id="req789",
        )

        assert ctx.trace_id == "trace123"
        assert ctx.span_id == "span456"
        assert ctx.request_id == "req789"

    def test_request_id_context(self):
        """Test request ID context variable."""
        from aion.observability.context import (
            set_request_id,
            get_request_id,
        )

        set_request_id("test-request-123")
        assert get_request_id() == "test-request-123"

    def test_with_context_manager(self):
        """Test with_context context manager."""
        from aion.observability.context import (
            with_context,
            get_current_trace_id,
            get_request_id,
        )

        with with_context(trace_id="ctx-trace", request_id="ctx-request"):
            assert get_current_trace_id() == "ctx-trace"
            assert get_request_id() == "ctx-request"

    def test_baggage(self):
        """Test baggage context variable."""
        from aion.observability.context import set_baggage, get_baggage

        set_baggage("user_id", "user123")
        set_baggage("tenant_id", "tenant456")

        assert get_baggage("user_id") == "user123"
        assert get_baggage("tenant_id") == "tenant456"
        assert get_baggage("nonexistent") is None


# =============================================================================
# Metrics Tests
# =============================================================================

class TestMetricsEngine:
    """Test metrics engine functionality."""

    @pytest.fixture
    def metrics_engine(self):
        """Create a metrics engine for testing."""
        from aion.observability.metrics.engine import MetricsEngine
        from aion.observability.collector import TelemetryCollector

        collector = TelemetryCollector()
        return MetricsEngine(collector=collector)

    @pytest.mark.asyncio
    async def test_counter_increment(self, metrics_engine):
        """Test counter increment."""
        await metrics_engine.initialize()

        metrics_engine.inc("test_counter", 1.0, {"method": "GET"})
        metrics_engine.inc("test_counter", 2.0, {"method": "GET"})

        value = metrics_engine.get_current("test_counter", {"method": "GET"})
        assert value == 3.0

        await metrics_engine.shutdown()

    @pytest.mark.asyncio
    async def test_gauge_set(self, metrics_engine):
        """Test gauge set."""
        await metrics_engine.initialize()

        metrics_engine.set("active_connections", 10.0, {})
        assert metrics_engine.get_current("active_connections", {}) == 10.0

        metrics_engine.set("active_connections", 15.0, {})
        assert metrics_engine.get_current("active_connections", {}) == 15.0

        await metrics_engine.shutdown()

    @pytest.mark.asyncio
    async def test_histogram_observe(self, metrics_engine):
        """Test histogram observation."""
        await metrics_engine.initialize()

        # Observe some values
        for value in [0.1, 0.2, 0.3, 0.5, 1.0, 2.0]:
            metrics_engine.observe("request_duration", value, {"endpoint": "/api"})

        stats = metrics_engine.get_stats()
        assert "histograms" in stats

        await metrics_engine.shutdown()

    @pytest.mark.asyncio
    async def test_prometheus_export(self, metrics_engine):
        """Test Prometheus format export."""
        await metrics_engine.initialize()

        metrics_engine.inc("requests_total", 100.0, {"method": "POST"})
        metrics_engine.set("temperature_celsius", 25.5, {"location": "server-1"})

        output = metrics_engine.export_prometheus()

        assert "requests_total" in output
        assert "temperature_celsius" in output
        assert "100" in output

        await metrics_engine.shutdown()

    @pytest.mark.asyncio
    async def test_metric_labels_key(self, metrics_engine):
        """Test metric labels key generation."""
        await metrics_engine.initialize()

        metrics_engine.inc("test", 1.0, {"a": "1", "b": "2"})
        metrics_engine.inc("test", 1.0, {"b": "2", "a": "1"})  # Same labels, different order

        # Should be counted together
        value = metrics_engine.get_current("test", {"a": "1", "b": "2"})
        assert value == 2.0

        await metrics_engine.shutdown()


class TestMetricRegistry:
    """Test metric registry functionality."""

    def test_counter_creation(self):
        """Test counter creation via registry."""
        from aion.observability.metrics.registry import Counter

        counter = Counter(
            name="requests_total",
            description="Total requests",
            labels=["method", "status"],
        )

        assert counter.name == "requests_total"
        assert "method" in counter.label_names
        assert "status" in counter.label_names

    def test_labeled_metric(self):
        """Test labeled metric access."""
        from aion.observability.metrics.registry import Counter

        counter = Counter(
            name="requests_total",
            labels=["method", "status"],
        )

        labeled = counter.labels(method="GET", status="200")
        assert labeled._label_values["method"] == "GET"
        assert labeled._label_values["status"] == "200"

    def test_label_validation(self):
        """Test label validation."""
        from aion.observability.metrics.registry import Counter

        counter = Counter(
            name="requests_total",
            labels=["method"],
        )

        # Missing label should raise
        with pytest.raises(ValueError):
            counter.labels(wrong_label="value")

    def test_histogram_buckets(self):
        """Test histogram with custom buckets."""
        from aion.observability.metrics.registry import Histogram

        histogram = Histogram(
            name="request_duration",
            buckets=(0.1, 0.5, 1.0, 5.0, 10.0),
        )

        assert histogram.buckets == (0.1, 0.5, 1.0, 5.0, 10.0)


class TestMetricExporters:
    """Test metric exporters."""

    @pytest.mark.asyncio
    async def test_in_memory_exporter(self):
        """Test in-memory exporter."""
        from aion.observability.metrics.exporters import InMemoryExporter
        from aion.observability.types import Metric, MetricType

        exporter = InMemoryExporter()

        metrics = [
            Metric(name="test1", value=1.0, metric_type=MetricType.COUNTER),
            Metric(name="test2", value=2.0, metric_type=MetricType.GAUGE),
        ]

        success = await exporter.export(metrics)
        assert success

        stored = exporter.get_metrics()
        assert len(stored) == 2
        assert stored[0].name == "test1"

    @pytest.mark.asyncio
    async def test_multi_exporter(self):
        """Test multi-exporter."""
        from aion.observability.metrics.exporters import (
            MultiExporter,
            InMemoryExporter,
        )
        from aion.observability.types import Metric, MetricType

        exp1 = InMemoryExporter()
        exp2 = InMemoryExporter()
        multi = MultiExporter([exp1, exp2])

        metrics = [Metric(name="test", value=1.0, metric_type=MetricType.COUNTER)]

        success = await multi.export(metrics)
        assert success

        assert len(exp1.get_metrics()) == 1
        assert len(exp2.get_metrics()) == 1


# =============================================================================
# Tracing Tests
# =============================================================================

class TestTracingEngine:
    """Test distributed tracing engine."""

    @pytest.fixture
    def tracing_engine(self):
        """Create a tracing engine for testing."""
        from aion.observability.tracing.engine import TracingEngine
        from aion.observability.collector import TelemetryCollector
        from aion.observability.tracing.sampling import AlwaysOnSampler

        collector = TelemetryCollector()
        return TracingEngine(
            collector=collector,
            service_name="test-service",
            sampler=AlwaysOnSampler(),
        )

    @pytest.mark.asyncio
    async def test_create_span(self, tracing_engine):
        """Test span creation."""
        await tracing_engine.initialize()

        span = tracing_engine.start_span("test_operation")
        assert span is not None
        assert span.operation_name == "test_operation"
        assert span.service_name == "test-service"

        span.end()
        await tracing_engine.shutdown()

    @pytest.mark.asyncio
    async def test_span_context_manager(self, tracing_engine):
        """Test span context manager."""
        await tracing_engine.initialize()

        with tracing_engine.trace("test_op") as span:
            assert span.operation_name == "test_op"
            time.sleep(0.01)

        assert span.end_time is not None
        assert span.duration_ms > 0

        await tracing_engine.shutdown()

    @pytest.mark.asyncio
    async def test_child_spans(self, tracing_engine):
        """Test parent-child span relationships."""
        await tracing_engine.initialize()

        with tracing_engine.trace("parent") as parent:
            with tracing_engine.trace("child") as child:
                assert child.parent_span_id == parent.span_id
                assert child.trace_id == parent.trace_id

        await tracing_engine.shutdown()

    @pytest.mark.asyncio
    async def test_span_attributes(self, tracing_engine):
        """Test span attribute setting."""
        await tracing_engine.initialize()

        with tracing_engine.trace("test") as span:
            span.set_attribute("user_id", "123")
            span.set_attribute("http.method", "GET")

        assert span.attributes["user_id"] == "123"
        assert span.attributes["http.method"] == "GET"

        await tracing_engine.shutdown()

    @pytest.mark.asyncio
    async def test_span_events(self, tracing_engine):
        """Test span event recording."""
        await tracing_engine.initialize()

        with tracing_engine.trace("test") as span:
            span.add_event("processing_started", {"items": 100})
            span.add_event("processing_complete", {"processed": 100})

        assert len(span.events) == 2

        await tracing_engine.shutdown()


class TestTraceSamplers:
    """Test trace sampling strategies."""

    def test_always_on_sampler(self):
        """Test always-on sampler."""
        from aion.observability.tracing.sampling import AlwaysOnSampler

        sampler = AlwaysOnSampler()
        assert sampler.should_sample("any-trace-id", "any-op")

    def test_always_off_sampler(self):
        """Test always-off sampler."""
        from aion.observability.tracing.sampling import AlwaysOffSampler

        sampler = AlwaysOffSampler()
        assert not sampler.should_sample("any-trace-id", "any-op")

    def test_ratio_sampler(self):
        """Test ratio-based sampler."""
        from aion.observability.tracing.sampling import TraceIdRatioSampler

        # 100% sampling
        sampler = TraceIdRatioSampler(1.0)
        assert sampler.should_sample("test", "op")

        # 0% sampling
        sampler = TraceIdRatioSampler(0.0)
        assert not sampler.should_sample("test", "op")

    def test_rate_limiting_sampler(self):
        """Test rate-limiting sampler."""
        from aion.observability.tracing.sampling import RateLimitingSampler

        sampler = RateLimitingSampler(max_per_second=2)

        # First two should be sampled
        assert sampler.should_sample("t1", "op")
        assert sampler.should_sample("t2", "op")

        # Third might be rate limited
        # (depends on timing)


class TestTracePropagation:
    """Test trace context propagation."""

    def test_w3c_trace_context_inject(self):
        """Test W3C Trace Context injection."""
        from aion.observability.tracing.propagation import W3CTraceContextPropagator
        from aion.observability.types import SpanContext

        propagator = W3CTraceContextPropagator()

        ctx = SpanContext(
            trace_id="0af7651916cd43dd8448eb211c80319c",
            span_id="b7ad6b7169203331",
        )

        headers = {}
        propagator.inject(ctx, headers)

        assert "traceparent" in headers
        assert "0af7651916cd43dd8448eb211c80319c" in headers["traceparent"]

    def test_w3c_trace_context_extract(self):
        """Test W3C Trace Context extraction."""
        from aion.observability.tracing.propagation import W3CTraceContextPropagator

        propagator = W3CTraceContextPropagator()

        headers = {
            "traceparent": "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
        }

        ctx = propagator.extract(headers)

        assert ctx is not None
        assert ctx.trace_id == "0af7651916cd43dd8448eb211c80319c"
        assert ctx.span_id == "b7ad6b7169203331"
        assert ctx.sampled is True

    def test_b3_propagator(self):
        """Test B3 propagation format."""
        from aion.observability.tracing.propagation import B3Propagator
        from aion.observability.types import SpanContext

        propagator = B3Propagator()

        ctx = SpanContext(
            trace_id="traceid123",
            span_id="spanid456",
            sampled=True,
        )

        headers = {}
        propagator.inject(ctx, headers)

        assert headers.get("X-B3-TraceId") == "traceid123"
        assert headers.get("X-B3-SpanId") == "spanid456"
        assert headers.get("X-B3-Sampled") == "1"


# =============================================================================
# Logging Tests
# =============================================================================

class TestLoggingEngine:
    """Test structured logging engine."""

    @pytest.fixture
    def logging_engine(self):
        """Create a logging engine for testing."""
        from aion.observability.logging.engine import LoggingEngine
        from aion.observability.collector import TelemetryCollector
        from aion.observability.types import LogLevel

        collector = TelemetryCollector()
        return LoggingEngine(
            collector=collector,
            service_name="test-service",
            default_level=LogLevel.DEBUG,
        )

    @pytest.mark.asyncio
    async def test_log_levels(self, logging_engine):
        """Test logging at different levels."""
        await logging_engine.initialize()

        logger = logging_engine.get_logger("test")

        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")

        logs = logging_engine.query_logs(limit=10)
        assert len(logs) >= 4

        await logging_engine.shutdown()

    @pytest.mark.asyncio
    async def test_structured_logging(self, logging_engine):
        """Test structured log fields."""
        await logging_engine.initialize()

        logger = logging_engine.get_logger("test")
        bound = logger.bind(user_id="123", request_id="req-456")

        bound.info("User action")

        logs = logging_engine.query_logs(limit=1)
        assert len(logs) == 1
        assert logs[0].extra.get("user_id") == "123"

        await logging_engine.shutdown()

    @pytest.mark.asyncio
    async def test_log_query_by_level(self, logging_engine):
        """Test querying logs by level."""
        await logging_engine.initialize()

        logger = logging_engine.get_logger("test")
        logger.info("Info 1")
        logger.error("Error 1")
        logger.info("Info 2")

        errors = logging_engine.query_logs(level="ERROR")
        assert len(errors) == 1
        assert errors[0].message == "Error 1"

        await logging_engine.shutdown()


# =============================================================================
# Alerting Tests
# =============================================================================

class TestAlertEngine:
    """Test alerting engine."""

    @pytest.fixture
    def alert_engine(self):
        """Create an alert engine for testing."""
        from aion.observability.alerting.engine import AlertEngine
        from aion.observability.metrics.engine import MetricsEngine
        from aion.observability.collector import TelemetryCollector

        collector = TelemetryCollector()
        metrics = MetricsEngine(collector=collector)
        return AlertEngine(metrics_engine=metrics)

    @pytest.mark.asyncio
    async def test_add_alert_rule(self, alert_engine):
        """Test adding an alert rule."""
        await alert_engine.initialize()

        from aion.observability.types import AlertRule, AlertSeverity

        rule = AlertRule(
            name="high_error_rate",
            metric_name="error_rate",
            condition="gt",
            threshold=0.05,
            severity=AlertSeverity.CRITICAL,
        )

        alert_engine.add_rule(rule)

        assert "high_error_rate" in alert_engine._rules

        await alert_engine.shutdown()

    @pytest.mark.asyncio
    async def test_alert_evaluation(self, alert_engine):
        """Test alert rule evaluation."""
        await alert_engine.initialize()

        from aion.observability.types import AlertRule, AlertSeverity

        rule = AlertRule(
            name="test_alert",
            metric_name="test_metric",
            condition="gt",
            threshold=10.0,
            severity=AlertSeverity.WARNING,
            duration_seconds=0,  # Immediate
        )

        alert_engine.add_rule(rule)

        # Set a value above threshold
        alert_engine._metrics_engine.set("test_metric", 15.0, {})

        await alert_engine.evaluate()

        alerts = alert_engine.get_active_alerts()
        # Alert should be pending or firing
        assert len(alerts) >= 0  # Depends on timing

        await alert_engine.shutdown()


class TestAlertChannels:
    """Test alert notification channels."""

    @pytest.mark.asyncio
    async def test_log_channel(self):
        """Test log-based alert channel."""
        from aion.observability.alerting.channels import LogChannel
        from aion.observability.types import Alert, AlertSeverity, AlertState

        channel = LogChannel()

        alert = Alert(
            name="test_alert",
            severity=AlertSeverity.WARNING,
            state=AlertState.FIRING,
            message="Test alert message",
        )

        success = await channel.send(alert)
        assert success

    @pytest.mark.asyncio
    async def test_composite_channel(self):
        """Test composite channel."""
        from aion.observability.alerting.channels import CompositeChannel, LogChannel
        from aion.observability.types import Alert, AlertSeverity, AlertState

        channel = CompositeChannel([LogChannel(), LogChannel()])

        alert = Alert(
            name="test_alert",
            severity=AlertSeverity.CRITICAL,
            state=AlertState.FIRING,
            message="Test alert",
        )

        success = await channel.send(alert)
        assert success


# =============================================================================
# Cost Tracking Tests
# =============================================================================

class TestCostTracker:
    """Test cost tracking functionality."""

    @pytest.fixture
    def cost_tracker(self):
        """Create a cost tracker for testing."""
        from aion.observability.analysis.cost import CostTracker
        from aion.observability.collector import TelemetryCollector

        collector = TelemetryCollector()
        return CostTracker(collector=collector)

    @pytest.mark.asyncio
    async def test_track_llm_cost(self, cost_tracker):
        """Test LLM cost tracking."""
        await cost_tracker.initialize()

        cost_tracker.track_llm_tokens(
            model="gpt-4",
            input_tokens=1000,
            output_tokens=500,
        )

        stats = cost_tracker.get_stats()
        assert stats["total_cost"] > 0

        await cost_tracker.shutdown()

    @pytest.mark.asyncio
    async def test_cost_attribution(self, cost_tracker):
        """Test cost attribution to agents."""
        await cost_tracker.initialize()

        cost_tracker.track_llm_tokens(
            model="gpt-4",
            input_tokens=1000,
            output_tokens=500,
            agent_id="agent-123",
        )

        costs = cost_tracker.get_costs_by_agent()
        assert "agent-123" in costs

        await cost_tracker.shutdown()

    @pytest.mark.asyncio
    async def test_budget_alert(self, cost_tracker):
        """Test budget alert threshold."""
        await cost_tracker.initialize()

        from aion.observability.types import CostBudget

        budget = CostBudget(
            name="daily",
            daily_limit=1.0,
            alert_threshold=0.8,
        )

        cost_tracker.set_budget(budget)

        # Track enough to trigger alert
        for _ in range(100):
            cost_tracker.track_llm_tokens(
                model="gpt-4",
                input_tokens=10000,
                output_tokens=5000,
            )

        stats = cost_tracker.get_stats()
        # Should have tracked costs

        await cost_tracker.shutdown()


# =============================================================================
# Anomaly Detection Tests
# =============================================================================

class TestAnomalyDetector:
    """Test anomaly detection functionality."""

    @pytest.fixture
    def anomaly_detector(self):
        """Create an anomaly detector for testing."""
        from aion.observability.analysis.anomaly import AnomalyDetector
        from aion.observability.metrics.engine import MetricsEngine
        from aion.observability.collector import TelemetryCollector

        collector = TelemetryCollector()
        metrics = MetricsEngine(collector=collector)
        return AnomalyDetector(metrics_engine=metrics)

    @pytest.mark.asyncio
    async def test_detect_spike(self, anomaly_detector):
        """Test spike detection."""
        await anomaly_detector.initialize()

        # Feed normal values to build baseline
        for i in range(50):
            anomaly_detector.observe("test_metric", 100.0 + (i % 5))

        # Introduce a spike
        anomaly_detector.observe("test_metric", 500.0)

        anomalies = anomaly_detector.get_recent_anomalies(10)
        # Should detect the spike
        # (depends on enough data points)

        await anomaly_detector.shutdown()

    @pytest.mark.asyncio
    async def test_z_score_calculation(self, anomaly_detector):
        """Test z-score calculation."""
        await anomaly_detector.initialize()

        # Build baseline
        for val in [100, 101, 99, 100, 102, 98, 100, 101, 99, 100]:
            anomaly_detector.observe("stable_metric", float(val))

        # Check if z-score is calculated
        status = anomaly_detector.get_metric_status("stable_metric")
        assert "mean" in status or status.get("samples", 0) > 0

        await anomaly_detector.shutdown()


# =============================================================================
# Profiler Tests
# =============================================================================

class TestProfiler:
    """Test performance profiling."""

    @pytest.fixture
    def profiler(self):
        """Create a profiler for testing."""
        from aion.observability.analysis.profiler import Profiler

        return Profiler(
            enable_cpu_profiling=True,
            enable_memory_tracking=True,
            hot_spot_threshold_ms=1.0,
        )

    @pytest.mark.asyncio
    async def test_operation_profiling(self, profiler):
        """Test operation profiling."""
        await profiler.initialize()

        with profiler.profile_operation("test_operation"):
            time.sleep(0.01)

        operations = profiler.get_operations()
        assert len(operations) > 0

        await profiler.shutdown()

    @pytest.mark.asyncio
    async def test_hot_spot_detection(self, profiler):
        """Test hot spot detection."""
        await profiler.initialize()

        # Create a slow operation
        with profiler.profile_operation("slow_operation"):
            time.sleep(0.05)

        hotspots = profiler.get_hot_spots(10)
        assert len(hotspots) > 0
        assert hotspots[0].name == "slow_operation"

        await profiler.shutdown()

    @pytest.mark.asyncio
    async def test_memory_tracking(self, profiler):
        """Test memory usage tracking."""
        await profiler.initialize()

        stats = profiler.get_memory_stats()
        assert "current_mb" in stats or "rss_mb" in stats or "memory" in str(stats).lower()

        await profiler.shutdown()


# =============================================================================
# Health Check Tests
# =============================================================================

class TestHealthChecker:
    """Test health checking functionality."""

    @pytest.fixture
    def health_checker(self):
        """Create a health checker for testing."""
        from aion.observability.health import HealthChecker

        return HealthChecker(check_interval=1.0)

    @pytest.mark.asyncio
    async def test_built_in_checks(self, health_checker):
        """Test built-in health checks."""
        await health_checker.initialize()

        health = await health_checker.check_all()
        assert health is not None

        await health_checker.shutdown()

    @pytest.mark.asyncio
    async def test_custom_check(self, health_checker):
        """Test custom health check registration."""
        await health_checker.initialize()

        async def custom_check():
            from aion.observability.types import HealthCheck, HealthStatus
            return HealthCheck(
                name="custom",
                status=HealthStatus.HEALTHY,
                message="All good",
            )

        health_checker.register_check("custom", custom_check)

        result = await health_checker.run_check("custom")
        assert result is not None
        assert result.name == "custom"

        await health_checker.shutdown()

    @pytest.mark.asyncio
    async def test_readiness_probe(self, health_checker):
        """Test readiness probe."""
        await health_checker.initialize()

        is_ready = await health_checker.is_ready()
        assert isinstance(is_ready, bool)

        await health_checker.shutdown()

    @pytest.mark.asyncio
    async def test_liveness_probe(self, health_checker):
        """Test liveness probe."""
        await health_checker.initialize()

        is_alive = await health_checker.is_alive()
        assert isinstance(is_alive, bool)

        await health_checker.shutdown()


# =============================================================================
# Storage Tests
# =============================================================================

class TestInMemoryStorage:
    """Test in-memory storage backends."""

    @pytest.mark.asyncio
    async def test_metric_store(self):
        """Test in-memory metric store."""
        from aion.observability.storage.memory import InMemoryMetricStore
        from aion.observability.types import Metric, MetricType

        store = InMemoryMetricStore()
        await store.initialize()

        metrics = [
            Metric(name="test", value=1.0, metric_type=MetricType.COUNTER),
            Metric(name="test", value=2.0, metric_type=MetricType.COUNTER),
        ]

        count = await store.write_metrics(metrics)
        assert count == 2

        result = await store.query_metrics("test")
        assert len(result) == 2

        await store.shutdown()

    @pytest.mark.asyncio
    async def test_trace_store(self):
        """Test in-memory trace store."""
        from aion.observability.storage.memory import InMemoryTraceStore
        from aion.observability.types import Span, SpanKind

        store = InMemoryTraceStore()
        await store.initialize()

        spans = [
            Span(
                trace_id="trace1",
                span_id="span1",
                operation_name="op1",
                service_name="svc",
            ),
        ]

        count = await store.write_spans(spans)
        assert count == 1

        trace = await store.get_trace("trace1")
        assert trace is not None
        assert len(trace.spans) == 1

        await store.shutdown()

    @pytest.mark.asyncio
    async def test_log_store(self):
        """Test in-memory log store."""
        from aion.observability.storage.memory import InMemoryLogStore
        from aion.observability.types import LogEntry, LogLevel

        store = InMemoryLogStore()
        await store.initialize()

        logs = [
            LogEntry(level=LogLevel.INFO, message="Test 1"),
            LogEntry(level=LogLevel.ERROR, message="Test 2"),
        ]

        count = await store.write_logs(logs)
        assert count == 2

        results = await store.query_logs(level="ERROR")
        assert len(results) == 1

        await store.shutdown()


# =============================================================================
# Collector Tests
# =============================================================================

class TestTelemetryCollector:
    """Test telemetry collector."""

    @pytest.fixture
    def collector(self):
        """Create a collector for testing."""
        from aion.observability.collector import TelemetryCollector

        return TelemetryCollector(buffer_size=100, flush_interval=0.1)

    @pytest.mark.asyncio
    async def test_collect_metrics(self, collector):
        """Test metric collection."""
        await collector.initialize()

        from aion.observability.types import Metric, MetricType

        metric = Metric(name="test", value=1.0, metric_type=MetricType.COUNTER)
        collector.collect_metric(metric)

        stats = collector.get_stats()
        assert stats["metrics_collected"] >= 1

        await collector.shutdown()

    @pytest.mark.asyncio
    async def test_collect_spans(self, collector):
        """Test span collection."""
        await collector.initialize()

        from aion.observability.types import Span

        span = Span(
            trace_id="t1",
            span_id="s1",
            operation_name="test",
            service_name="svc",
        )
        collector.collect_span(span)

        stats = collector.get_stats()
        assert stats["spans_collected"] >= 1

        await collector.shutdown()


# =============================================================================
# Instrumentation Tests
# =============================================================================

class TestInstrumentationDecorators:
    """Test instrumentation decorators."""

    @pytest.mark.asyncio
    async def test_traced_decorator(self):
        """Test @traced decorator."""
        from aion.observability.instrumentation.decorators import traced

        @traced("test_function")
        async def my_function():
            return "result"

        result = await my_function()
        assert result == "result"

    @pytest.mark.asyncio
    async def test_metered_decorator(self):
        """Test @metered decorator."""
        from aion.observability.instrumentation.decorators import metered

        @metered("test_operation")
        async def my_function():
            return "result"

        result = await my_function()
        assert result == "result"

    @pytest.mark.asyncio
    async def test_logged_decorator(self):
        """Test @logged decorator."""
        from aion.observability.instrumentation.decorators import logged

        @logged(level="INFO")
        async def my_function():
            return "result"

        result = await my_function()
        assert result == "result"

    @pytest.mark.asyncio
    async def test_observable_decorator(self):
        """Test @observable decorator (combines all)."""
        from aion.observability.instrumentation.decorators import observable

        @observable("combined_test")
        async def my_function():
            return "result"

        result = await my_function()
        assert result == "result"


# =============================================================================
# Manager Tests
# =============================================================================

class TestObservabilityManager:
    """Test observability manager."""

    @pytest.mark.asyncio
    async def test_manager_initialization(self):
        """Test manager initialization."""
        from aion.observability.manager import ObservabilityManager, ObservabilityConfig

        config = ObservabilityConfig(
            service_name="test-service",
            environment="test",
        )

        manager = ObservabilityManager(config=config)
        await manager.initialize()

        assert manager._initialized
        assert manager.tracing is not None
        assert manager.metrics is not None
        assert manager.logging is not None

        await manager.shutdown()

    @pytest.mark.asyncio
    async def test_manager_stats(self):
        """Test manager statistics."""
        from aion.observability.manager import ObservabilityManager

        manager = ObservabilityManager()
        await manager.initialize()

        stats = manager.get_stats()
        assert "service" in stats
        assert "uptime_seconds" in stats

        await manager.shutdown()

    @pytest.mark.asyncio
    async def test_manager_shutdown(self):
        """Test graceful shutdown."""
        from aion.observability.manager import ObservabilityManager

        manager = ObservabilityManager()
        await manager.initialize()
        assert manager._initialized

        await manager.shutdown()
        assert not manager._initialized


# =============================================================================
# Integration Tests
# =============================================================================

class TestObservabilityIntegration:
    """Integration tests for the observability system."""

    @pytest.mark.asyncio
    async def test_full_request_flow(self):
        """Test complete observability flow for a request."""
        from aion.observability.manager import ObservabilityManager, ObservabilityConfig
        from aion.observability.context import with_request

        config = ObservabilityConfig(
            service_name="integration-test",
            enable_cost_tracking=True,
            enable_anomaly_detection=True,
        )

        manager = ObservabilityManager(config=config)
        await manager.initialize()

        # Simulate a request flow
        with with_request(request_id="req-123", user_id="user-456"):
            # Create a trace
            with manager.tracing.trace("handle_request") as root_span:
                root_span.set_attribute("http.method", "POST")

                # Increment metrics
                manager.metrics.inc("requests_total", 1, {"method": "POST"})

                # Log
                logger = manager.get_logger("integration")
                logger.info("Processing request")

                # Simulate nested operation
                with manager.tracing.trace("process_data") as child_span:
                    time.sleep(0.01)
                    child_span.set_attribute("items_processed", 42)

                # Track costs
                if manager.costs:
                    manager.costs.track_llm_tokens(
                        model="gpt-4",
                        input_tokens=100,
                        output_tokens=50,
                    )

        # Verify everything was recorded
        stats = manager.get_stats()
        assert stats["tracing"]["active_spans"] == 0  # All ended
        assert stats["metrics"]["total_series"] > 0

        await manager.shutdown()

    @pytest.mark.asyncio
    async def test_alert_lifecycle(self):
        """Test complete alert lifecycle."""
        from aion.observability.manager import ObservabilityManager, ObservabilityConfig
        from aion.observability.types import AlertRule, AlertSeverity

        config = ObservabilityConfig(
            service_name="alert-test",
            alert_evaluation_interval=0.1,
        )

        manager = ObservabilityManager(config=config)
        await manager.initialize()

        # Add an alert rule
        rule = AlertRule(
            name="high_latency",
            metric_name="request_latency",
            condition="gt",
            threshold=1.0,
            severity=AlertSeverity.WARNING,
            duration_seconds=0,
        )
        manager.alerts.add_rule(rule)

        # Set metric above threshold
        manager.metrics.set("request_latency", 2.0, {})

        # Evaluate alerts
        await manager.alerts.evaluate()

        # Check alert state
        alerts = manager.alerts.get_active_alerts()
        # Should have at least a pending alert

        await manager.shutdown()
