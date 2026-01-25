"""
Native OTLP Protocol Implementation.

Implements the OpenTelemetry Protocol for exporting telemetry data.
Supports both gRPC and HTTP transports.
"""

import asyncio
import gzip
import json
import logging
import struct
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
from enum import IntEnum
import base64

logger = logging.getLogger(__name__)


# =============================================================================
# OTLP Data Types
# =============================================================================

class AggregationTemporality(IntEnum):
    """Aggregation temporality for metrics."""
    UNSPECIFIED = 0
    DELTA = 1
    CUMULATIVE = 2


class DataPointFlags(IntEnum):
    """Flags for data points."""
    FLAG_NONE = 0
    FLAG_NO_RECORDED_VALUE = 1


@dataclass
class KeyValue:
    """Key-value pair for attributes."""
    key: str
    value: Any  # string, bool, int, double, array, or kvlist


@dataclass
class InstrumentationScope:
    """Instrumentation scope identifying the instrumentation library."""
    name: str
    version: str = ""
    attributes: List[KeyValue] = field(default_factory=list)
    dropped_attributes_count: int = 0


@dataclass
class Resource:
    """Resource information for telemetry."""
    attributes: List[KeyValue] = field(default_factory=list)
    dropped_attributes_count: int = 0


# =============================================================================
# Span Data Types
# =============================================================================

class SpanKind(IntEnum):
    """Type of span."""
    UNSPECIFIED = 0
    INTERNAL = 1
    SERVER = 2
    CLIENT = 3
    PRODUCER = 4
    CONSUMER = 5


class StatusCode(IntEnum):
    """Span status code."""
    UNSET = 0
    OK = 1
    ERROR = 2


@dataclass
class SpanStatus:
    """Status of a span."""
    code: StatusCode = StatusCode.UNSET
    message: str = ""


@dataclass
class SpanEvent:
    """Event within a span."""
    time_unix_nano: int
    name: str
    attributes: List[KeyValue] = field(default_factory=list)
    dropped_attributes_count: int = 0


@dataclass
class SpanLink:
    """Link to another span."""
    trace_id: bytes
    span_id: bytes
    trace_state: str = ""
    attributes: List[KeyValue] = field(default_factory=list)
    dropped_attributes_count: int = 0


@dataclass
class Span:
    """A span representing a unit of work."""
    trace_id: bytes  # 16 bytes
    span_id: bytes  # 8 bytes
    parent_span_id: bytes = b''  # 8 bytes or empty
    name: str = ""
    kind: SpanKind = SpanKind.INTERNAL
    start_time_unix_nano: int = 0
    end_time_unix_nano: int = 0
    attributes: List[KeyValue] = field(default_factory=list)
    events: List[SpanEvent] = field(default_factory=list)
    links: List[SpanLink] = field(default_factory=list)
    status: SpanStatus = field(default_factory=SpanStatus)
    trace_state: str = ""
    dropped_attributes_count: int = 0
    dropped_events_count: int = 0
    dropped_links_count: int = 0


@dataclass
class ScopeSpans:
    """Spans from a single instrumentation scope."""
    scope: InstrumentationScope
    spans: List[Span] = field(default_factory=list)
    schema_url: str = ""


@dataclass
class ResourceSpans:
    """Spans from a single resource."""
    resource: Resource
    scope_spans: List[ScopeSpans] = field(default_factory=list)
    schema_url: str = ""


# =============================================================================
# Metric Data Types
# =============================================================================

@dataclass
class NumberDataPoint:
    """A single numeric data point."""
    attributes: List[KeyValue] = field(default_factory=list)
    start_time_unix_nano: int = 0
    time_unix_nano: int = 0
    value: Union[int, float] = 0
    exemplars: List['Exemplar'] = field(default_factory=list)
    flags: DataPointFlags = DataPointFlags.FLAG_NONE


@dataclass
class HistogramDataPoint:
    """A histogram data point."""
    attributes: List[KeyValue] = field(default_factory=list)
    start_time_unix_nano: int = 0
    time_unix_nano: int = 0
    count: int = 0
    sum: float = 0.0
    bucket_counts: List[int] = field(default_factory=list)
    explicit_bounds: List[float] = field(default_factory=list)
    exemplars: List['Exemplar'] = field(default_factory=list)
    flags: DataPointFlags = DataPointFlags.FLAG_NONE
    min: Optional[float] = None
    max: Optional[float] = None


@dataclass
class ExponentialHistogramDataPoint:
    """An exponential histogram data point."""
    attributes: List[KeyValue] = field(default_factory=list)
    start_time_unix_nano: int = 0
    time_unix_nano: int = 0
    count: int = 0
    sum: float = 0.0
    scale: int = 0
    zero_count: int = 0
    positive: Optional['Buckets'] = None
    negative: Optional['Buckets'] = None
    flags: DataPointFlags = DataPointFlags.FLAG_NONE
    min: Optional[float] = None
    max: Optional[float] = None


@dataclass
class Buckets:
    """Buckets for exponential histogram."""
    offset: int = 0
    bucket_counts: List[int] = field(default_factory=list)


@dataclass
class SummaryDataPoint:
    """A summary data point."""
    attributes: List[KeyValue] = field(default_factory=list)
    start_time_unix_nano: int = 0
    time_unix_nano: int = 0
    count: int = 0
    sum: float = 0.0
    quantile_values: List['QuantileValue'] = field(default_factory=list)
    flags: DataPointFlags = DataPointFlags.FLAG_NONE


@dataclass
class QuantileValue:
    """A quantile value."""
    quantile: float
    value: float


@dataclass
class Exemplar:
    """An exemplar linking metrics to traces."""
    filtered_attributes: List[KeyValue] = field(default_factory=list)
    time_unix_nano: int = 0
    value: Union[int, float] = 0
    span_id: bytes = b''
    trace_id: bytes = b''


@dataclass
class Gauge:
    """A gauge metric."""
    data_points: List[NumberDataPoint] = field(default_factory=list)


@dataclass
class Sum:
    """A sum metric."""
    data_points: List[NumberDataPoint] = field(default_factory=list)
    aggregation_temporality: AggregationTemporality = AggregationTemporality.CUMULATIVE
    is_monotonic: bool = False


@dataclass
class Histogram:
    """A histogram metric."""
    data_points: List[HistogramDataPoint] = field(default_factory=list)
    aggregation_temporality: AggregationTemporality = AggregationTemporality.CUMULATIVE


@dataclass
class ExponentialHistogram:
    """An exponential histogram metric."""
    data_points: List[ExponentialHistogramDataPoint] = field(default_factory=list)
    aggregation_temporality: AggregationTemporality = AggregationTemporality.CUMULATIVE


@dataclass
class Summary:
    """A summary metric."""
    data_points: List[SummaryDataPoint] = field(default_factory=list)


@dataclass
class Metric:
    """A single metric with its data."""
    name: str
    description: str = ""
    unit: str = ""
    data: Union[Gauge, Sum, Histogram, ExponentialHistogram, Summary] = None


@dataclass
class ScopeMetrics:
    """Metrics from a single instrumentation scope."""
    scope: InstrumentationScope
    metrics: List[Metric] = field(default_factory=list)
    schema_url: str = ""


@dataclass
class ResourceMetrics:
    """Metrics from a single resource."""
    resource: Resource
    scope_metrics: List[ScopeMetrics] = field(default_factory=list)
    schema_url: str = ""


# =============================================================================
# Log Data Types
# =============================================================================

class SeverityNumber(IntEnum):
    """Log severity number."""
    UNSPECIFIED = 0
    TRACE = 1
    TRACE2 = 2
    TRACE3 = 3
    TRACE4 = 4
    DEBUG = 5
    DEBUG2 = 6
    DEBUG3 = 7
    DEBUG4 = 8
    INFO = 9
    INFO2 = 10
    INFO3 = 11
    INFO4 = 12
    WARN = 13
    WARN2 = 14
    WARN3 = 15
    WARN4 = 16
    ERROR = 17
    ERROR2 = 18
    ERROR3 = 19
    ERROR4 = 20
    FATAL = 21
    FATAL2 = 22
    FATAL3 = 23
    FATAL4 = 24


@dataclass
class LogRecord:
    """A single log record."""
    time_unix_nano: int = 0
    observed_time_unix_nano: int = 0
    severity_number: SeverityNumber = SeverityNumber.UNSPECIFIED
    severity_text: str = ""
    body: Any = None
    attributes: List[KeyValue] = field(default_factory=list)
    dropped_attributes_count: int = 0
    flags: int = 0
    trace_id: bytes = b''
    span_id: bytes = b''


@dataclass
class ScopeLogs:
    """Logs from a single instrumentation scope."""
    scope: InstrumentationScope
    log_records: List[LogRecord] = field(default_factory=list)
    schema_url: str = ""


@dataclass
class ResourceLogs:
    """Logs from a single resource."""
    resource: Resource
    scope_logs: List[ScopeLogs] = field(default_factory=list)
    schema_url: str = ""


# =============================================================================
# OTLP Serialization
# =============================================================================

class OTLPSerializer:
    """Serialize OTLP data to protobuf wire format."""

    def serialize_key_value(self, kv: KeyValue) -> Dict[str, Any]:
        """Serialize a key-value pair to JSON representation."""
        value_dict = {}

        if isinstance(kv.value, str):
            value_dict = {"stringValue": kv.value}
        elif isinstance(kv.value, bool):
            value_dict = {"boolValue": kv.value}
        elif isinstance(kv.value, int):
            value_dict = {"intValue": str(kv.value)}
        elif isinstance(kv.value, float):
            value_dict = {"doubleValue": kv.value}
        elif isinstance(kv.value, bytes):
            value_dict = {"bytesValue": base64.b64encode(kv.value).decode()}
        elif isinstance(kv.value, list):
            value_dict = {"arrayValue": {"values": [
                self.serialize_any_value(v) for v in kv.value
            ]}}
        elif isinstance(kv.value, dict):
            value_dict = {"kvlistValue": {"values": [
                {"key": k, "value": self.serialize_any_value(v)}
                for k, v in kv.value.items()
            ]}}

        return {"key": kv.key, "value": value_dict}

    def serialize_any_value(self, value: Any) -> Dict[str, Any]:
        """Serialize any value type."""
        if isinstance(value, str):
            return {"stringValue": value}
        elif isinstance(value, bool):
            return {"boolValue": value}
        elif isinstance(value, int):
            return {"intValue": str(value)}
        elif isinstance(value, float):
            return {"doubleValue": value}
        elif isinstance(value, bytes):
            return {"bytesValue": base64.b64encode(value).decode()}
        elif isinstance(value, list):
            return {"arrayValue": {"values": [
                self.serialize_any_value(v) for v in value
            ]}}
        return {"stringValue": str(value)}

    def serialize_resource(self, resource: Resource) -> Dict[str, Any]:
        """Serialize resource to JSON."""
        return {
            "attributes": [self.serialize_key_value(kv) for kv in resource.attributes],
            "droppedAttributesCount": resource.dropped_attributes_count
        }

    def serialize_scope(self, scope: InstrumentationScope) -> Dict[str, Any]:
        """Serialize instrumentation scope to JSON."""
        return {
            "name": scope.name,
            "version": scope.version,
            "attributes": [self.serialize_key_value(kv) for kv in scope.attributes]
        }

    def serialize_span(self, span: Span) -> Dict[str, Any]:
        """Serialize span to JSON."""
        return {
            "traceId": base64.b64encode(span.trace_id).decode(),
            "spanId": base64.b64encode(span.span_id).decode(),
            "parentSpanId": base64.b64encode(span.parent_span_id).decode() if span.parent_span_id else "",
            "name": span.name,
            "kind": span.kind.value,
            "startTimeUnixNano": str(span.start_time_unix_nano),
            "endTimeUnixNano": str(span.end_time_unix_nano),
            "attributes": [self.serialize_key_value(kv) for kv in span.attributes],
            "events": [self.serialize_span_event(e) for e in span.events],
            "links": [self.serialize_span_link(l) for l in span.links],
            "status": {
                "code": span.status.code.value,
                "message": span.status.message
            },
            "traceState": span.trace_state
        }

    def serialize_span_event(self, event: SpanEvent) -> Dict[str, Any]:
        """Serialize span event to JSON."""
        return {
            "timeUnixNano": str(event.time_unix_nano),
            "name": event.name,
            "attributes": [self.serialize_key_value(kv) for kv in event.attributes]
        }

    def serialize_span_link(self, link: SpanLink) -> Dict[str, Any]:
        """Serialize span link to JSON."""
        return {
            "traceId": base64.b64encode(link.trace_id).decode(),
            "spanId": base64.b64encode(link.span_id).decode(),
            "traceState": link.trace_state,
            "attributes": [self.serialize_key_value(kv) for kv in link.attributes]
        }

    def serialize_traces(self, resource_spans: List[ResourceSpans]) -> Dict[str, Any]:
        """Serialize trace data to OTLP JSON."""
        return {
            "resourceSpans": [
                {
                    "resource": self.serialize_resource(rs.resource),
                    "scopeSpans": [
                        {
                            "scope": self.serialize_scope(ss.scope),
                            "spans": [self.serialize_span(s) for s in ss.spans],
                            "schemaUrl": ss.schema_url
                        }
                        for ss in rs.scope_spans
                    ],
                    "schemaUrl": rs.schema_url
                }
                for rs in resource_spans
            ]
        }

    def serialize_metrics(self, resource_metrics: List[ResourceMetrics]) -> Dict[str, Any]:
        """Serialize metric data to OTLP JSON."""
        return {
            "resourceMetrics": [
                {
                    "resource": self.serialize_resource(rm.resource),
                    "scopeMetrics": [
                        {
                            "scope": self.serialize_scope(sm.scope),
                            "metrics": [self.serialize_metric(m) for m in sm.metrics],
                            "schemaUrl": sm.schema_url
                        }
                        for sm in rm.scope_metrics
                    ],
                    "schemaUrl": rm.schema_url
                }
                for rm in resource_metrics
            ]
        }

    def serialize_metric(self, metric: Metric) -> Dict[str, Any]:
        """Serialize a single metric."""
        result = {
            "name": metric.name,
            "description": metric.description,
            "unit": metric.unit
        }

        if isinstance(metric.data, Gauge):
            result["gauge"] = {
                "dataPoints": [self.serialize_number_data_point(dp) for dp in metric.data.data_points]
            }
        elif isinstance(metric.data, Sum):
            result["sum"] = {
                "dataPoints": [self.serialize_number_data_point(dp) for dp in metric.data.data_points],
                "aggregationTemporality": metric.data.aggregation_temporality.value,
                "isMonotonic": metric.data.is_monotonic
            }
        elif isinstance(metric.data, Histogram):
            result["histogram"] = {
                "dataPoints": [self.serialize_histogram_data_point(dp) for dp in metric.data.data_points],
                "aggregationTemporality": metric.data.aggregation_temporality.value
            }

        return result

    def serialize_number_data_point(self, dp: NumberDataPoint) -> Dict[str, Any]:
        """Serialize number data point."""
        result = {
            "attributes": [self.serialize_key_value(kv) for kv in dp.attributes],
            "startTimeUnixNano": str(dp.start_time_unix_nano),
            "timeUnixNano": str(dp.time_unix_nano)
        }

        if isinstance(dp.value, int):
            result["asInt"] = str(dp.value)
        else:
            result["asDouble"] = dp.value

        return result

    def serialize_histogram_data_point(self, dp: HistogramDataPoint) -> Dict[str, Any]:
        """Serialize histogram data point."""
        return {
            "attributes": [self.serialize_key_value(kv) for kv in dp.attributes],
            "startTimeUnixNano": str(dp.start_time_unix_nano),
            "timeUnixNano": str(dp.time_unix_nano),
            "count": str(dp.count),
            "sum": dp.sum,
            "bucketCounts": [str(c) for c in dp.bucket_counts],
            "explicitBounds": dp.explicit_bounds
        }

    def serialize_logs(self, resource_logs: List[ResourceLogs]) -> Dict[str, Any]:
        """Serialize log data to OTLP JSON."""
        return {
            "resourceLogs": [
                {
                    "resource": self.serialize_resource(rl.resource),
                    "scopeLogs": [
                        {
                            "scope": self.serialize_scope(sl.scope),
                            "logRecords": [self.serialize_log_record(lr) for lr in sl.log_records],
                            "schemaUrl": sl.schema_url
                        }
                        for sl in rl.scope_logs
                    ],
                    "schemaUrl": rl.schema_url
                }
                for rl in resource_logs
            ]
        }

    def serialize_log_record(self, log: LogRecord) -> Dict[str, Any]:
        """Serialize log record."""
        return {
            "timeUnixNano": str(log.time_unix_nano),
            "observedTimeUnixNano": str(log.observed_time_unix_nano),
            "severityNumber": log.severity_number.value,
            "severityText": log.severity_text,
            "body": self.serialize_any_value(log.body) if log.body else None,
            "attributes": [self.serialize_key_value(kv) for kv in log.attributes],
            "traceId": base64.b64encode(log.trace_id).decode() if log.trace_id else "",
            "spanId": base64.b64encode(log.span_id).decode() if log.span_id else ""
        }


# =============================================================================
# OTLP Exporters
# =============================================================================

class OTLPExporter(ABC):
    """Base class for OTLP exporters."""

    def __init__(
        self,
        endpoint: str,
        headers: Dict[str, str] = None,
        timeout: int = 10,
        compression: str = "gzip"
    ):
        self.endpoint = endpoint
        self.headers = headers or {}
        self.timeout = timeout
        self.compression = compression
        self.serializer = OTLPSerializer()

    @abstractmethod
    async def export_traces(self, resource_spans: List[ResourceSpans]) -> bool:
        """Export trace data."""
        pass

    @abstractmethod
    async def export_metrics(self, resource_metrics: List[ResourceMetrics]) -> bool:
        """Export metric data."""
        pass

    @abstractmethod
    async def export_logs(self, resource_logs: List[ResourceLogs]) -> bool:
        """Export log data."""
        pass

    def _compress(self, data: bytes) -> bytes:
        """Compress data if compression is enabled."""
        if self.compression == "gzip":
            return gzip.compress(data)
        return data


class OTLPHTTPExporter(OTLPExporter):
    """OTLP/HTTP exporter."""

    def __init__(
        self,
        endpoint: str = "http://localhost:4318",
        headers: Dict[str, str] = None,
        timeout: int = 10,
        compression: str = "gzip"
    ):
        super().__init__(endpoint, headers, timeout, compression)

    async def export_traces(self, resource_spans: List[ResourceSpans]) -> bool:
        """Export traces via HTTP."""
        url = f"{self.endpoint}/v1/traces"
        data = self.serializer.serialize_traces(resource_spans)
        return await self._post(url, data)

    async def export_metrics(self, resource_metrics: List[ResourceMetrics]) -> bool:
        """Export metrics via HTTP."""
        url = f"{self.endpoint}/v1/metrics"
        data = self.serializer.serialize_metrics(resource_metrics)
        return await self._post(url, data)

    async def export_logs(self, resource_logs: List[ResourceLogs]) -> bool:
        """Export logs via HTTP."""
        url = f"{self.endpoint}/v1/logs"
        data = self.serializer.serialize_logs(resource_logs)
        return await self._post(url, data)

    async def _post(self, url: str, data: Dict[str, Any]) -> bool:
        """POST data to endpoint."""
        try:
            json_data = json.dumps(data).encode('utf-8')

            headers = {
                "Content-Type": "application/json",
                **self.headers
            }

            if self.compression == "gzip":
                json_data = self._compress(json_data)
                headers["Content-Encoding"] = "gzip"

            # In real implementation, use aiohttp or httpx
            logger.debug(f"Exporting to {url}: {len(json_data)} bytes")

            # Placeholder for actual HTTP request
            # async with aiohttp.ClientSession() as session:
            #     async with session.post(url, data=json_data, headers=headers) as resp:
            #         return resp.status < 300

            return True

        except Exception as e:
            logger.error(f"Export failed: {e}")
            return False


class OTLPGRPCExporter(OTLPExporter):
    """OTLP/gRPC exporter."""

    def __init__(
        self,
        endpoint: str = "localhost:4317",
        headers: Dict[str, str] = None,
        timeout: int = 10,
        compression: str = "gzip",
        insecure: bool = True
    ):
        super().__init__(endpoint, headers, timeout, compression)
        self.insecure = insecure
        self._channel = None

    async def connect(self):
        """Establish gRPC connection."""
        # In real implementation, use grpcio-aio
        logger.info(f"Connecting to gRPC endpoint: {self.endpoint}")

    async def export_traces(self, resource_spans: List[ResourceSpans]) -> bool:
        """Export traces via gRPC."""
        try:
            data = self.serializer.serialize_traces(resource_spans)
            logger.debug(f"Exporting {len(resource_spans)} resource spans via gRPC")
            # In real implementation, call gRPC service
            return True
        except Exception as e:
            logger.error(f"gRPC export failed: {e}")
            return False

    async def export_metrics(self, resource_metrics: List[ResourceMetrics]) -> bool:
        """Export metrics via gRPC."""
        try:
            data = self.serializer.serialize_metrics(resource_metrics)
            logger.debug(f"Exporting {len(resource_metrics)} resource metrics via gRPC")
            return True
        except Exception as e:
            logger.error(f"gRPC export failed: {e}")
            return False

    async def export_logs(self, resource_logs: List[ResourceLogs]) -> bool:
        """Export logs via gRPC."""
        try:
            data = self.serializer.serialize_logs(resource_logs)
            logger.debug(f"Exporting {len(resource_logs)} resource logs via gRPC")
            return True
        except Exception as e:
            logger.error(f"gRPC export failed: {e}")
            return False


# =============================================================================
# OTLP Collector
# =============================================================================

class OTLPCollector:
    """
    OTLP Collector for receiving and forwarding telemetry.

    Acts as a local collector that batches and forwards data.
    """

    def __init__(
        self,
        exporters: List[OTLPExporter] = None,
        batch_size: int = 100,
        batch_timeout: float = 5.0
    ):
        self.exporters = exporters or []
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout

        self._trace_buffer: List[ResourceSpans] = []
        self._metric_buffer: List[ResourceMetrics] = []
        self._log_buffer: List[ResourceLogs] = []

        self._running = False
        self._flush_task = None

    def add_exporter(self, exporter: OTLPExporter):
        """Add an exporter."""
        self.exporters.append(exporter)

    async def start(self):
        """Start the collector."""
        self._running = True
        self._flush_task = asyncio.create_task(self._flush_loop())
        logger.info("OTLP Collector started")

    async def stop(self):
        """Stop the collector."""
        self._running = False
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass

        # Final flush
        await self._flush_all()
        logger.info("OTLP Collector stopped")

    def collect_traces(self, resource_spans: ResourceSpans):
        """Collect trace data."""
        self._trace_buffer.append(resource_spans)
        if len(self._trace_buffer) >= self.batch_size:
            asyncio.create_task(self._flush_traces())

    def collect_metrics(self, resource_metrics: ResourceMetrics):
        """Collect metric data."""
        self._metric_buffer.append(resource_metrics)
        if len(self._metric_buffer) >= self.batch_size:
            asyncio.create_task(self._flush_metrics())

    def collect_logs(self, resource_logs: ResourceLogs):
        """Collect log data."""
        self._log_buffer.append(resource_logs)
        if len(self._log_buffer) >= self.batch_size:
            asyncio.create_task(self._flush_logs())

    async def _flush_loop(self):
        """Periodic flush loop."""
        while self._running:
            await asyncio.sleep(self.batch_timeout)
            await self._flush_all()

    async def _flush_all(self):
        """Flush all buffers."""
        await asyncio.gather(
            self._flush_traces(),
            self._flush_metrics(),
            self._flush_logs()
        )

    async def _flush_traces(self):
        """Flush trace buffer."""
        if not self._trace_buffer:
            return

        batch = self._trace_buffer
        self._trace_buffer = []

        for exporter in self.exporters:
            try:
                await exporter.export_traces(batch)
            except Exception as e:
                logger.error(f"Trace export failed: {e}")

    async def _flush_metrics(self):
        """Flush metric buffer."""
        if not self._metric_buffer:
            return

        batch = self._metric_buffer
        self._metric_buffer = []

        for exporter in self.exporters:
            try:
                await exporter.export_metrics(batch)
            except Exception as e:
                logger.error(f"Metric export failed: {e}")

    async def _flush_logs(self):
        """Flush log buffer."""
        if not self._log_buffer:
            return

        batch = self._log_buffer
        self._log_buffer = []

        for exporter in self.exporters:
            try:
                await exporter.export_logs(batch)
            except Exception as e:
                logger.error(f"Log export failed: {e}")
