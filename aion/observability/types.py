"""
AION Observability Types

State-of-the-art dataclasses for metrics, traces, logs, alerts, and cost tracking.
Designed for production-grade observability with OpenTelemetry compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union, Callable
import uuid
import hashlib
import json


# === Metrics ===

class MetricType(str, Enum):
    """Types of metrics following Prometheus conventions."""
    COUNTER = "counter"          # Monotonically increasing
    GAUGE = "gauge"              # Point-in-time value
    HISTOGRAM = "histogram"      # Distribution of values
    SUMMARY = "summary"          # Quantiles over time
    INFO = "info"                # Static information
    STATESET = "stateset"        # Set of states


class AggregationType(str, Enum):
    """Aggregation methods for metrics."""
    SUM = "sum"
    AVG = "avg"
    MIN = "min"
    MAX = "max"
    LAST = "last"
    COUNT = "count"
    P50 = "p50"
    P90 = "p90"
    P95 = "p95"
    P99 = "p99"
    RATE = "rate"
    INCREASE = "increase"


@dataclass
class MetricLabel:
    """A label/tag for a metric with validation."""
    key: str
    value: str

    def __post_init__(self):
        # Sanitize label key to be Prometheus-compatible
        self.key = self._sanitize_label_name(self.key)

    @staticmethod
    def _sanitize_label_name(name: str) -> str:
        """Sanitize label name for Prometheus compatibility."""
        import re
        # Replace invalid characters with underscores
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', name)
        # Ensure doesn't start with digit
        if sanitized and sanitized[0].isdigit():
            sanitized = '_' + sanitized
        return sanitized


@dataclass
class Metric:
    """A single metric measurement with full metadata."""
    name: str
    value: float
    metric_type: MetricType = MetricType.GAUGE
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    unit: str = ""  # e.g., "seconds", "bytes", "requests"
    description: str = ""

    # For histograms
    bucket: Optional[float] = None  # Upper bound of histogram bucket

    # For summaries
    quantile: Optional[float] = None

    # Exemplar for high-cardinality traces
    exemplar_trace_id: Optional[str] = None
    exemplar_span_id: Optional[str] = None
    exemplar_value: Optional[float] = None

    def __post_init__(self):
        # Sanitize metric name
        self.name = self._sanitize_metric_name(self.name)

    @staticmethod
    def _sanitize_metric_name(name: str) -> str:
        """Sanitize metric name for Prometheus compatibility."""
        import re
        sanitized = re.sub(r'[^a-zA-Z0-9_:]', '_', name)
        if sanitized and sanitized[0].isdigit():
            sanitized = '_' + sanitized
        return sanitized

    def labels_key(self) -> str:
        """Generate a unique key from labels for deduplication."""
        return ",".join(f"{k}={v}" for k, v in sorted(self.labels.items()))

    def full_name(self) -> str:
        """Get full metric name with labels."""
        if self.labels:
            labels_str = ",".join(f'{k}="{v}"' for k, v in sorted(self.labels.items()))
            return f"{self.name}{{{labels_str}}}"
        return self.name

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "value": self.value,
            "type": self.metric_type.value,
            "labels": self.labels,
            "timestamp": self.timestamp.isoformat(),
            "unit": self.unit,
            "description": self.description,
        }

    def to_prometheus(self) -> str:
        """Export in Prometheus exposition format."""
        if self.labels:
            labels_str = ",".join(f'{k}="{v}"' for k, v in sorted(self.labels.items()))
            return f"{self.name}{{{labels_str}}} {self.value}"
        return f"{self.name} {self.value}"


@dataclass
class MetricDefinition:
    """Definition of a metric with full configuration."""
    name: str
    metric_type: MetricType
    description: str
    unit: str = ""
    labels: List[str] = field(default_factory=list)  # Expected label keys

    # Histogram buckets
    buckets: List[float] = field(default_factory=lambda: [
        0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, float('inf')
    ])

    # Summary objectives
    objectives: Dict[float, float] = field(default_factory=lambda: {
        0.5: 0.05, 0.9: 0.01, 0.99: 0.001
    })

    # Aggregation settings
    aggregation: AggregationType = AggregationType.LAST
    retention_days: int = 30

    # Alert thresholds
    warning_threshold: Optional[float] = None
    critical_threshold: Optional[float] = None

    # Cardinality limits
    max_label_cardinality: int = 1000


@dataclass
class MetricSeries:
    """A time series of metric values."""
    name: str
    labels: Dict[str, str]
    points: List[tuple] = field(default_factory=list)  # (timestamp, value)

    def add_point(self, value: float, timestamp: datetime = None):
        """Add a data point to the series."""
        timestamp = timestamp or datetime.utcnow()
        self.points.append((timestamp, value))

    def get_rate(self, duration: timedelta) -> float:
        """Calculate rate of change over duration."""
        if len(self.points) < 2:
            return 0.0
        cutoff = datetime.utcnow() - duration
        recent = [(t, v) for t, v in self.points if t >= cutoff]
        if len(recent) < 2:
            return 0.0
        delta_value = recent[-1][1] - recent[0][1]
        delta_time = (recent[-1][0] - recent[0][0]).total_seconds()
        return delta_value / delta_time if delta_time > 0 else 0.0


# === Tracing ===

class SpanKind(str, Enum):
    """Kind of span following OpenTelemetry conventions."""
    INTERNAL = "internal"    # Default internal operation
    SERVER = "server"        # Server-side of an RPC
    CLIENT = "client"        # Client-side of an RPC
    PRODUCER = "producer"    # Message producer
    CONSUMER = "consumer"    # Message consumer


class SpanStatus(str, Enum):
    """Status of a span."""
    UNSET = "unset"
    OK = "ok"
    ERROR = "error"


class SamplingDecision(str, Enum):
    """Sampling decision for a trace."""
    NOT_RECORD = "not_record"       # Don't record, don't sample
    RECORD = "record"               # Record but don't sample
    RECORD_AND_SAMPLE = "sample"    # Record and sample


@dataclass
class SpanEvent:
    """An event within a span (annotation)."""
    name: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "timestamp": self.timestamp.isoformat(),
            "attributes": self.attributes,
        }


@dataclass
class SpanLink:
    """Link to another span for causality tracking."""
    trace_id: str
    span_id: str
    trace_state: str = ""
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "trace_state": self.trace_state,
            "attributes": self.attributes,
        }


@dataclass
class SpanContext:
    """Context for span propagation (W3C Trace Context compatible)."""
    trace_id: str
    span_id: str
    trace_flags: int = 1  # 1 = sampled
    trace_state: str = ""
    is_remote: bool = False

    @property
    def is_valid(self) -> bool:
        """Check if context is valid."""
        return bool(self.trace_id and self.span_id)

    @property
    def is_sampled(self) -> bool:
        """Check if trace is sampled."""
        return bool(self.trace_flags & 1)

    def to_traceparent(self) -> str:
        """Convert to W3C traceparent header."""
        return f"00-{self.trace_id}-{self.span_id}-{self.trace_flags:02x}"

    @classmethod
    def from_traceparent(cls, traceparent: str) -> Optional["SpanContext"]:
        """Parse W3C traceparent header."""
        try:
            parts = traceparent.split("-")
            if len(parts) >= 4 and parts[0] == "00":
                return cls(
                    trace_id=parts[1],
                    span_id=parts[2],
                    trace_flags=int(parts[3], 16),
                    is_remote=True,
                )
        except Exception:
            pass
        return None


@dataclass
class Span:
    """A span in a distributed trace with full OpenTelemetry compatibility."""
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    parent_span_id: Optional[str] = None

    # Identity
    name: str = ""
    kind: SpanKind = SpanKind.INTERNAL

    # Timing
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None

    # Status
    status: SpanStatus = SpanStatus.UNSET
    status_message: str = ""

    # Attributes (OpenTelemetry semantic conventions)
    attributes: Dict[str, Any] = field(default_factory=dict)

    # Events and links
    events: List[SpanEvent] = field(default_factory=list)
    links: List[SpanLink] = field(default_factory=list)

    # Resource info
    service_name: str = "aion"
    service_version: str = ""
    service_instance_id: str = ""

    # Instrumentation scope
    instrumentation_scope_name: str = ""
    instrumentation_scope_version: str = ""

    # Sampling
    sampling_decision: SamplingDecision = SamplingDecision.RECORD_AND_SAMPLE

    # Dropped counts (for cardinality limits)
    dropped_attributes_count: int = 0
    dropped_events_count: int = 0
    dropped_links_count: int = 0

    @property
    def duration_ms(self) -> float:
        """Get span duration in milliseconds."""
        if not self.end_time:
            return (datetime.utcnow() - self.start_time).total_seconds() * 1000
        return (self.end_time - self.start_time).total_seconds() * 1000

    @property
    def duration_ns(self) -> int:
        """Get span duration in nanoseconds."""
        return int(self.duration_ms * 1_000_000)

    @property
    def is_recording(self) -> bool:
        """Check if span is recording."""
        return self.sampling_decision != SamplingDecision.NOT_RECORD

    @property
    def context(self) -> SpanContext:
        """Get span context for propagation."""
        return SpanContext(
            trace_id=self.trace_id,
            span_id=self.span_id,
            trace_flags=1 if self.sampling_decision == SamplingDecision.RECORD_AND_SAMPLE else 0,
        )

    def add_event(self, name: str, attributes: Dict[str, Any] = None) -> None:
        """Add an event to the span."""
        self.events.append(SpanEvent(
            name=name,
            attributes=attributes or {},
        ))

    def set_attribute(self, key: str, value: Any) -> None:
        """Set a span attribute."""
        self.attributes[key] = value

    def set_status(self, status: SpanStatus, message: str = "") -> None:
        """Set span status."""
        self.status = status
        self.status_message = message

    def record_exception(self, exception: Exception) -> None:
        """Record an exception as an event."""
        import traceback
        self.add_event("exception", {
            "exception.type": type(exception).__name__,
            "exception.message": str(exception),
            "exception.stacktrace": traceback.format_exc(),
        })
        self.set_status(SpanStatus.ERROR, str(exception))

    def end(self, end_time: datetime = None) -> None:
        """End the span."""
        self.end_time = end_time or datetime.utcnow()

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "name": self.name,
            "kind": self.kind.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "status": self.status.value,
            "status_message": self.status_message,
            "attributes": self.attributes,
            "events": [e.to_dict() for e in self.events],
            "links": [l.to_dict() for l in self.links],
            "service_name": self.service_name,
        }

    def to_otlp(self) -> dict:
        """Export in OTLP format."""
        return {
            "traceId": self.trace_id,
            "spanId": self.span_id,
            "parentSpanId": self.parent_span_id or "",
            "name": self.name,
            "kind": self.kind.value.upper(),
            "startTimeUnixNano": int(self.start_time.timestamp() * 1e9),
            "endTimeUnixNano": int(self.end_time.timestamp() * 1e9) if self.end_time else 0,
            "attributes": [{"key": k, "value": {"stringValue": str(v)}} for k, v in self.attributes.items()],
            "events": [{"name": e.name, "timeUnixNano": int(e.timestamp.timestamp() * 1e9)} for e in self.events],
            "status": {"code": 1 if self.status == SpanStatus.OK else 2 if self.status == SpanStatus.ERROR else 0},
        }


@dataclass
class Trace:
    """A complete distributed trace."""
    trace_id: str
    spans: List[Span] = field(default_factory=list)

    # Metadata
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    @property
    def root_span(self) -> Optional[Span]:
        """Get the root span."""
        for span in self.spans:
            if not span.parent_span_id:
                return span
        return self.spans[0] if self.spans else None

    @property
    def duration_ms(self) -> float:
        """Get total trace duration."""
        if not self.spans:
            return 0.0
        start = min(s.start_time for s in self.spans)
        end = max(s.end_time or s.start_time for s in self.spans)
        return (end - start).total_seconds() * 1000

    @property
    def span_count(self) -> int:
        """Get number of spans."""
        return len(self.spans)

    @property
    def error_count(self) -> int:
        """Get number of error spans."""
        return sum(1 for s in self.spans if s.status == SpanStatus.ERROR)

    def get_span(self, span_id: str) -> Optional[Span]:
        """Get a span by ID."""
        for span in self.spans:
            if span.span_id == span_id:
                return span
        return None

    def get_children(self, span_id: str) -> List[Span]:
        """Get child spans of a span."""
        return [s for s in self.spans if s.parent_span_id == span_id]

    def to_tree(self) -> dict:
        """Convert trace to tree structure."""
        def build_tree(span: Span) -> dict:
            return {
                "span": span.to_dict(),
                "children": [build_tree(child) for child in self.get_children(span.span_id)],
            }
        root = self.root_span
        return build_tree(root) if root else {}


# === Logging ===

class LogLevel(str, Enum):
    """Log levels following syslog severity."""
    TRACE = "trace"      # Finest level
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

    @property
    def severity_number(self) -> int:
        """Get OTel severity number."""
        return {
            LogLevel.TRACE: 1,
            LogLevel.DEBUG: 5,
            LogLevel.INFO: 9,
            LogLevel.WARNING: 13,
            LogLevel.ERROR: 17,
            LogLevel.CRITICAL: 21,
        }.get(self, 0)


@dataclass
class LogEntry:
    """A structured log entry with full context."""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    level: LogLevel = LogLevel.INFO
    message: str = ""

    # Context
    logger_name: str = ""
    trace_id: Optional[str] = None
    span_id: Optional[str] = None

    # Structured data
    attributes: Dict[str, Any] = field(default_factory=dict)

    # Resource
    service_name: str = "aion"
    service_version: str = ""
    host_name: str = ""

    # Source location
    source_file: str = ""
    source_line: int = 0
    source_function: str = ""

    # Error info
    exception_type: Optional[str] = None
    exception_message: Optional[str] = None
    exception_traceback: Optional[str] = None

    # Identifiers
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    agent_id: Optional[str] = None

    def to_dict(self) -> dict:
        result = {
            "timestamp": self.timestamp.isoformat(),
            "level": self.level.value,
            "severity_number": self.level.severity_number,
            "message": self.message,
            "logger": self.logger_name,
            "service": self.service_name,
        }

        if self.trace_id:
            result["trace_id"] = self.trace_id
        if self.span_id:
            result["span_id"] = self.span_id
        if self.attributes:
            result["attributes"] = self.attributes
        if self.source_file:
            result["source"] = {
                "file": self.source_file,
                "line": self.source_line,
                "function": self.source_function,
            }
        if self.exception_type:
            result["exception"] = {
                "type": self.exception_type,
                "message": self.exception_message,
                "traceback": self.exception_traceback,
            }
        if self.request_id:
            result["request_id"] = self.request_id
        if self.user_id:
            result["user_id"] = self.user_id
        if self.agent_id:
            result["agent_id"] = self.agent_id

        return result

    def to_json(self) -> str:
        """Export as JSON string."""
        return json.dumps(self.to_dict())

    def to_otlp(self) -> dict:
        """Export in OTLP format."""
        return {
            "timeUnixNano": int(self.timestamp.timestamp() * 1e9),
            "severityNumber": self.level.severity_number,
            "severityText": self.level.value.upper(),
            "body": {"stringValue": self.message},
            "attributes": [{"key": k, "value": {"stringValue": str(v)}} for k, v in self.attributes.items()],
            "traceId": self.trace_id or "",
            "spanId": self.span_id or "",
        }


# === Alerting ===

class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

    @property
    def priority(self) -> int:
        """Get priority (higher = more severe)."""
        return {"info": 1, "warning": 2, "error": 3, "critical": 4}.get(self.value, 0)


class AlertState(str, Enum):
    """Alert states following Prometheus alerting."""
    INACTIVE = "inactive"      # Rule not matching
    PENDING = "pending"        # Matching but waiting for duration
    FIRING = "firing"          # Active and notified
    RESOLVED = "resolved"      # Was firing, now resolved


class AlertCondition(str, Enum):
    """Alert condition operators."""
    GREATER_THAN = ">"
    LESS_THAN = "<"
    GREATER_EQUAL = ">="
    LESS_EQUAL = "<="
    EQUAL = "=="
    NOT_EQUAL = "!="
    ABSENT = "absent"          # Metric missing
    PRESENT = "present"        # Metric exists


@dataclass
class AlertRule:
    """An alert rule definition with full configuration."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""

    # Condition (PromQL-style)
    metric_name: str = ""
    condition: AlertCondition = AlertCondition.GREATER_THAN
    threshold: float = 0.0

    # Optional label matchers
    label_matchers: Dict[str, str] = field(default_factory=dict)

    # Aggregation
    aggregation: Optional[AggregationType] = None
    aggregation_window: timedelta = field(default_factory=lambda: timedelta(minutes=5))

    # Timing
    for_duration: timedelta = field(default_factory=lambda: timedelta(seconds=0))
    evaluation_interval: timedelta = field(default_factory=lambda: timedelta(minutes=1))

    # Severity
    severity: AlertSeverity = AlertSeverity.WARNING

    # Labels and annotations (templated)
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)

    # Actions
    channels: List[str] = field(default_factory=list)  # Channel IDs to notify
    runbook_url: str = ""

    # Status
    enabled: bool = True

    # Silencing
    silenced_until: Optional[datetime] = None

    def is_silenced(self) -> bool:
        """Check if rule is currently silenced."""
        if self.silenced_until:
            return datetime.utcnow() < self.silenced_until
        return False

    def check_condition(self, value: float) -> bool:
        """Check if condition is met."""
        ops = {
            AlertCondition.GREATER_THAN: lambda v, t: v > t,
            AlertCondition.LESS_THAN: lambda v, t: v < t,
            AlertCondition.GREATER_EQUAL: lambda v, t: v >= t,
            AlertCondition.LESS_EQUAL: lambda v, t: v <= t,
            AlertCondition.EQUAL: lambda v, t: v == t,
            AlertCondition.NOT_EQUAL: lambda v, t: v != t,
        }
        return ops.get(self.condition, lambda v, t: False)(value, self.threshold)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "metric_name": self.metric_name,
            "condition": f"{self.condition.value} {self.threshold}",
            "severity": self.severity.value,
            "enabled": self.enabled,
            "for_duration": str(self.for_duration),
        }


@dataclass
class Alert:
    """An active alert instance."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    fingerprint: str = ""  # Unique identifier based on labels

    # Rule info
    rule_id: str = ""
    rule_name: str = ""

    # State
    state: AlertState = AlertState.PENDING
    severity: AlertSeverity = AlertSeverity.WARNING

    # Details
    message: str = ""
    current_value: float = 0.0
    threshold: float = 0.0

    # Labels and annotations
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)

    # Timestamps
    started_at: datetime = field(default_factory=datetime.utcnow)
    fired_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    last_evaluated_at: datetime = field(default_factory=datetime.utcnow)

    # Evaluation history
    evaluation_count: int = 0
    consecutive_fires: int = 0

    # Notifications
    notified_channels: List[str] = field(default_factory=list)
    last_notified_at: Optional[datetime] = None
    notification_count: int = 0

    # Acknowledgement
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None

    def __post_init__(self):
        if not self.fingerprint:
            self.fingerprint = self._generate_fingerprint()

    def _generate_fingerprint(self) -> str:
        """Generate unique fingerprint from labels."""
        label_str = json.dumps(self.labels, sort_keys=True)
        return hashlib.md5(f"{self.rule_id}:{label_str}".encode()).hexdigest()[:16]

    @property
    def duration(self) -> timedelta:
        """Get alert duration."""
        end = self.resolved_at or datetime.utcnow()
        return end - self.started_at

    @property
    def is_active(self) -> bool:
        """Check if alert is still active."""
        return self.state in (AlertState.PENDING, AlertState.FIRING)

    def fire(self) -> None:
        """Transition to firing state."""
        self.state = AlertState.FIRING
        self.fired_at = datetime.utcnow()
        self.consecutive_fires += 1

    def resolve(self) -> None:
        """Transition to resolved state."""
        self.state = AlertState.RESOLVED
        self.resolved_at = datetime.utcnow()

    def acknowledge(self, user: str) -> None:
        """Acknowledge the alert."""
        self.acknowledged = True
        self.acknowledged_by = user
        self.acknowledged_at = datetime.utcnow()

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "fingerprint": self.fingerprint,
            "rule_name": self.rule_name,
            "state": self.state.value,
            "severity": self.severity.value,
            "message": self.message,
            "current_value": self.current_value,
            "threshold": self.threshold,
            "labels": self.labels,
            "annotations": self.annotations,
            "started_at": self.started_at.isoformat(),
            "fired_at": self.fired_at.isoformat() if self.fired_at else None,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "duration_seconds": self.duration.total_seconds(),
            "acknowledged": self.acknowledged,
        }


# === Health ===

class HealthStatus(str, Enum):
    """Health status values."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """A health check result."""
    name: str
    status: HealthStatus = HealthStatus.UNKNOWN
    message: str = ""
    latency_ms: float = 0.0
    checked_at: datetime = field(default_factory=datetime.utcnow)

    # Details
    details: Dict[str, Any] = field(default_factory=dict)

    # Dependencies
    dependencies: List[str] = field(default_factory=list)

    # Thresholds
    warning_threshold_ms: float = 1000.0
    critical_threshold_ms: float = 5000.0

    def check_latency_status(self) -> HealthStatus:
        """Determine status based on latency."""
        if self.latency_ms >= self.critical_threshold_ms:
            return HealthStatus.UNHEALTHY
        elif self.latency_ms >= self.warning_threshold_ms:
            return HealthStatus.DEGRADED
        return HealthStatus.HEALTHY

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "latency_ms": self.latency_ms,
            "checked_at": self.checked_at.isoformat(),
            "details": self.details,
        }


@dataclass
class SystemHealth:
    """Overall system health aggregation."""
    status: HealthStatus = HealthStatus.UNKNOWN
    checks: List[HealthCheck] = field(default_factory=list)
    checked_at: datetime = field(default_factory=datetime.utcnow)

    # Service metadata
    service_name: str = "aion"
    version: str = ""
    uptime_seconds: float = 0.0

    def aggregate_status(self) -> HealthStatus:
        """Aggregate status from all checks."""
        if not self.checks:
            return HealthStatus.UNKNOWN

        statuses = [c.status for c in self.checks]

        if HealthStatus.UNHEALTHY in statuses:
            return HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        elif all(s == HealthStatus.HEALTHY for s in statuses):
            return HealthStatus.HEALTHY
        return HealthStatus.UNKNOWN

    @property
    def healthy_count(self) -> int:
        """Count of healthy checks."""
        return sum(1 for c in self.checks if c.status == HealthStatus.HEALTHY)

    @property
    def unhealthy_count(self) -> int:
        """Count of unhealthy checks."""
        return sum(1 for c in self.checks if c.status == HealthStatus.UNHEALTHY)

    def to_dict(self) -> dict:
        return {
            "status": self.status.value,
            "service": self.service_name,
            "version": self.version,
            "uptime_seconds": self.uptime_seconds,
            "checks": [c.to_dict() for c in self.checks],
            "healthy_count": self.healthy_count,
            "unhealthy_count": self.unhealthy_count,
            "checked_at": self.checked_at.isoformat(),
        }


# === Cost Tracking ===

class ResourceType(str, Enum):
    """Types of billable resources."""
    TOKENS_INPUT = "tokens_input"
    TOKENS_OUTPUT = "tokens_output"
    API_CALL = "api_call"
    STORAGE = "storage"
    COMPUTE = "compute"
    NETWORK = "network"
    EMBEDDING = "embedding"
    SEARCH = "search"


@dataclass
class CostRecord:
    """A record of resource cost with full attribution."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # What
    resource_type: ResourceType = ResourceType.TOKENS_INPUT
    resource_name: str = ""  # Specific resource (model name, API name, etc.)

    # How much
    quantity: float = 0.0
    unit: str = ""
    unit_cost: float = 0.0
    total_cost: float = 0.0
    currency: str = "USD"

    # Context for attribution
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    agent_id: Optional[str] = None
    goal_id: Optional[str] = None
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    session_id: Optional[str] = None

    # Labels for grouping
    labels: Dict[str, str] = field(default_factory=dict)

    # Cost allocation
    cost_center: str = ""
    project: str = ""
    environment: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "resource_type": self.resource_type.value,
            "resource_name": self.resource_name,
            "quantity": self.quantity,
            "unit": self.unit,
            "unit_cost": self.unit_cost,
            "total_cost": self.total_cost,
            "currency": self.currency,
            "trace_id": self.trace_id,
            "agent_id": self.agent_id,
            "labels": self.labels,
        }


@dataclass
class CostBudget:
    """Budget configuration for cost control."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""

    # Budget amount
    amount: float = 0.0
    currency: str = "USD"

    # Period
    period: str = "monthly"  # daily, weekly, monthly, yearly
    period_start: Optional[datetime] = None

    # Scope
    scope_type: str = "global"  # global, user, agent, project
    scope_id: Optional[str] = None

    # Alerts
    warning_threshold_percent: float = 80.0
    critical_threshold_percent: float = 95.0

    # Current usage
    current_usage: float = 0.0

    @property
    def remaining(self) -> float:
        """Get remaining budget."""
        return max(0, self.amount - self.current_usage)

    @property
    def usage_percent(self) -> float:
        """Get usage as percentage."""
        return (self.current_usage / self.amount * 100) if self.amount > 0 else 0

    @property
    def is_over_budget(self) -> bool:
        """Check if over budget."""
        return self.current_usage > self.amount

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "amount": self.amount,
            "currency": self.currency,
            "period": self.period,
            "current_usage": self.current_usage,
            "remaining": self.remaining,
            "usage_percent": self.usage_percent,
            "is_over_budget": self.is_over_budget,
        }


# === Anomaly Detection ===

class AnomalyType(str, Enum):
    """Types of anomalies."""
    SPIKE = "spike"                # Sudden increase
    DROP = "drop"                  # Sudden decrease
    TREND_CHANGE = "trend_change"  # Change in trend
    SEASONALITY = "seasonality"    # Deviation from pattern
    OUTLIER = "outlier"            # Statistical outlier
    MISSING = "missing"            # Missing data


@dataclass
class Anomaly:
    """Detected anomaly in metrics."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # What
    metric_name: str = ""
    labels: Dict[str, str] = field(default_factory=dict)
    anomaly_type: AnomalyType = AnomalyType.OUTLIER

    # When
    detected_at: datetime = field(default_factory=datetime.utcnow)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    # Values
    expected_value: float = 0.0
    actual_value: float = 0.0
    deviation: float = 0.0  # Standard deviations from mean
    confidence: float = 0.0  # 0-1 confidence score

    # Context
    description: str = ""
    possible_causes: List[str] = field(default_factory=list)

    # Related
    related_metrics: List[str] = field(default_factory=list)
    related_spans: List[str] = field(default_factory=list)

    @property
    def severity(self) -> AlertSeverity:
        """Determine severity based on deviation."""
        if abs(self.deviation) >= 5:
            return AlertSeverity.CRITICAL
        elif abs(self.deviation) >= 3:
            return AlertSeverity.ERROR
        elif abs(self.deviation) >= 2:
            return AlertSeverity.WARNING
        return AlertSeverity.INFO

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "metric_name": self.metric_name,
            "anomaly_type": self.anomaly_type.value,
            "detected_at": self.detected_at.isoformat(),
            "expected_value": self.expected_value,
            "actual_value": self.actual_value,
            "deviation": self.deviation,
            "confidence": self.confidence,
            "severity": self.severity.value,
            "description": self.description,
        }


# === Profiling ===

@dataclass
class ProfileSample:
    """A single profile sample."""
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Location
    function_name: str = ""
    file_name: str = ""
    line_number: int = 0

    # Metrics
    cpu_time_ns: int = 0
    wall_time_ns: int = 0
    memory_bytes: int = 0
    allocations: int = 0

    # Context
    trace_id: Optional[str] = None
    span_id: Optional[str] = None


@dataclass
class ProfileReport:
    """Aggregated profiling report."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""

    # Period
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None

    # Samples
    samples: List[ProfileSample] = field(default_factory=list)
    sample_count: int = 0

    # Aggregated metrics
    total_cpu_time_ns: int = 0
    total_wall_time_ns: int = 0
    peak_memory_bytes: int = 0
    total_allocations: int = 0

    # Hot spots (function -> time)
    hot_spots: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "sample_count": self.sample_count,
            "total_cpu_time_ms": self.total_cpu_time_ns / 1_000_000,
            "peak_memory_mb": self.peak_memory_bytes / (1024 * 1024),
            "hot_spots": dict(sorted(self.hot_spots.items(), key=lambda x: -x[1])[:10]),
        }


# === Audit Trail ===

class AuditAction(str, Enum):
    """Types of auditable actions."""
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    EXECUTE = "execute"
    APPROVE = "approve"
    REJECT = "reject"
    LOGIN = "login"
    LOGOUT = "logout"
    CONFIG_CHANGE = "config_change"
    PERMISSION_CHANGE = "permission_change"


@dataclass
class AuditEvent:
    """An audit trail event for compliance."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Who
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    service_name: str = "aion"

    # What
    action: AuditAction = AuditAction.READ
    resource_type: str = ""
    resource_id: str = ""

    # Details
    description: str = ""
    old_value: Optional[Any] = None
    new_value: Optional[Any] = None

    # Context
    trace_id: Optional[str] = None
    request_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None

    # Result
    success: bool = True
    error_message: Optional[str] = None

    # Compliance
    sensitive: bool = False
    retention_days: int = 365

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "action": self.action.value,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "description": self.description,
            "trace_id": self.trace_id,
            "success": self.success,
        }
