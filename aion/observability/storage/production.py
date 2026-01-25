"""
Production Storage Backends

SOTA storage backends for observability data:
- Time-series databases (InfluxDB, TimescaleDB, VictoriaMetrics)
- Distributed tracing backends (Jaeger, Tempo)
- Log storage (Elasticsearch, Loki)
- Unified storage (ClickHouse)
"""

from __future__ import annotations

import asyncio
import gzip
import json
import logging
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

from aion.observability.types import Metric, Span, LogEntry, Trace
from aion.observability.storage.base import MetricStore, TraceStore, LogStore

logger = logging.getLogger(__name__)


# =============================================================================
# InfluxDB Storage
# =============================================================================

@dataclass
class InfluxDBConfig:
    """InfluxDB connection configuration."""
    url: str = "http://localhost:8086"
    token: str = ""
    org: str = "aion"
    bucket: str = "observability"
    batch_size: int = 5000
    flush_interval_seconds: float = 10.0
    retention_days: int = 30


class InfluxDBMetricStore(MetricStore):
    """
    InfluxDB-based metric storage.

    Features:
    - High-throughput writes with batching
    - Efficient time-series queries
    - Built-in downsampling
    - Flux query language support
    """

    def __init__(self, config: InfluxDBConfig):
        self.config = config
        self._client = None
        self._write_api = None
        self._query_api = None
        self._batch: List[str] = []
        self._last_flush = time.time()
        self._running = False

    async def initialize(self) -> None:
        """Initialize InfluxDB connection."""
        try:
            from influxdb_client import InfluxDBClient
            from influxdb_client.client.write_api import SYNCHRONOUS

            self._client = InfluxDBClient(
                url=self.config.url,
                token=self.config.token,
                org=self.config.org,
            )
            self._write_api = self._client.write_api(write_options=SYNCHRONOUS)
            self._query_api = self._client.query_api()
            self._running = True

            # Start batch flush task
            asyncio.create_task(self._flush_loop())

            logger.info("InfluxDB metric store initialized")

        except ImportError:
            logger.warning("influxdb-client not installed, using mock mode")
            self._running = True

    async def shutdown(self) -> None:
        """Shutdown InfluxDB connection."""
        self._running = False
        await self._flush()

        if self._client:
            self._client.close()

    async def write_metrics(self, metrics: List[Metric]) -> int:
        """Write metrics to InfluxDB."""
        count = 0

        for metric in metrics:
            # Convert to InfluxDB line protocol
            line = self._to_line_protocol(metric)
            self._batch.append(line)
            count += 1

            if len(self._batch) >= self.config.batch_size:
                await self._flush()

        return count

    def _to_line_protocol(self, metric: Metric) -> str:
        """Convert metric to InfluxDB line protocol."""
        # measurement,tag1=value1,tag2=value2 field=value timestamp
        tags = ",".join(f"{k}={v}" for k, v in sorted(metric.labels.items()))
        if tags:
            measurement = f"{metric.name},{tags}"
        else:
            measurement = metric.name

        timestamp_ns = int(metric.timestamp.timestamp() * 1e9)

        return f"{measurement} value={metric.value} {timestamp_ns}"

    async def _flush(self) -> None:
        """Flush batch to InfluxDB."""
        if not self._batch:
            return

        batch = self._batch
        self._batch = []

        if self._write_api:
            try:
                data = "\n".join(batch)
                self._write_api.write(
                    bucket=self.config.bucket,
                    record=data,
                )
            except Exception as e:
                logger.error(f"Error flushing to InfluxDB: {e}")
                # Re-add failed batch
                self._batch = batch + self._batch

        self._last_flush = time.time()

    async def _flush_loop(self) -> None:
        """Periodic flush loop."""
        while self._running:
            await asyncio.sleep(1.0)

            if time.time() - self._last_flush >= self.config.flush_interval_seconds:
                await self._flush()

    async def query_metrics(
        self,
        name: str,
        labels: Optional[Dict[str, str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000,
    ) -> List[Tuple[datetime, float]]:
        """Query metrics from InfluxDB."""
        if not self._query_api:
            return []

        # Build Flux query
        start = start_time or datetime.utcnow() - timedelta(hours=1)
        end = end_time or datetime.utcnow()

        query = f'''
from(bucket: "{self.config.bucket}")
  |> range(start: {start.isoformat()}Z, stop: {end.isoformat()}Z)
  |> filter(fn: (r) => r["_measurement"] == "{name}")
'''

        if labels:
            for k, v in labels.items():
                query += f'  |> filter(fn: (r) => r["{k}"] == "{v}")\n'

        query += f'  |> limit(n: {limit})'

        try:
            tables = self._query_api.query(query)
            results = []

            for table in tables:
                for record in table.records:
                    results.append((record.get_time(), record.get_value()))

            return results

        except Exception as e:
            logger.error(f"Error querying InfluxDB: {e}")
            return []


# =============================================================================
# TimescaleDB Storage
# =============================================================================

@dataclass
class TimescaleDBConfig:
    """TimescaleDB connection configuration."""
    host: str = "localhost"
    port: int = 5432
    database: str = "observability"
    user: str = "aion"
    password: str = ""
    pool_size: int = 10
    chunk_time_interval: str = "1 day"
    retention_days: int = 30


class TimescaleDBMetricStore(MetricStore):
    """
    TimescaleDB-based metric storage.

    Features:
    - Automatic hypertable chunking
    - Continuous aggregates
    - Compression policies
    - PostgreSQL compatibility
    """

    def __init__(self, config: TimescaleDBConfig):
        self.config = config
        self._pool = None

    async def initialize(self) -> None:
        """Initialize TimescaleDB connection pool."""
        try:
            import asyncpg

            self._pool = await asyncpg.create_pool(
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
                user=self.config.user,
                password=self.config.password,
                min_size=1,
                max_size=self.config.pool_size,
            )

            # Create tables if not exist
            await self._create_schema()

            logger.info("TimescaleDB metric store initialized")

        except ImportError:
            logger.warning("asyncpg not installed, using mock mode")

    async def _create_schema(self) -> None:
        """Create TimescaleDB schema."""
        if not self._pool:
            return

        async with self._pool.acquire() as conn:
            # Metrics table
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS metrics (
                    time TIMESTAMPTZ NOT NULL,
                    name TEXT NOT NULL,
                    value DOUBLE PRECISION NOT NULL,
                    labels JSONB
                );

                SELECT create_hypertable('metrics', 'time',
                    if_not_exists => TRUE,
                    chunk_time_interval => INTERVAL '1 day'
                );

                CREATE INDEX IF NOT EXISTS metrics_name_idx ON metrics (name, time DESC);
            ''')

    async def shutdown(self) -> None:
        """Shutdown connection pool."""
        if self._pool:
            await self._pool.close()

    async def write_metrics(self, metrics: List[Metric]) -> int:
        """Write metrics to TimescaleDB."""
        if not self._pool:
            return 0

        async with self._pool.acquire() as conn:
            # Batch insert
            values = [
                (
                    metric.timestamp,
                    metric.name,
                    metric.value,
                    json.dumps(metric.labels) if metric.labels else None,
                )
                for metric in metrics
            ]

            await conn.executemany(
                '''
                INSERT INTO metrics (time, name, value, labels)
                VALUES ($1, $2, $3, $4)
                ''',
                values,
            )

            return len(metrics)

    async def query_metrics(
        self,
        name: str,
        labels: Optional[Dict[str, str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000,
    ) -> List[Tuple[datetime, float]]:
        """Query metrics from TimescaleDB."""
        if not self._pool:
            return []

        start = start_time or datetime.utcnow() - timedelta(hours=1)
        end = end_time or datetime.utcnow()

        query = '''
            SELECT time, value FROM metrics
            WHERE name = $1
            AND time >= $2 AND time <= $3
        '''
        params = [name, start, end]

        if labels:
            query += " AND labels @> $4"
            params.append(json.dumps(labels))

        query += f" ORDER BY time DESC LIMIT {limit}"

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
            return [(row["time"], row["value"]) for row in rows]


# =============================================================================
# Jaeger Trace Storage
# =============================================================================

@dataclass
class JaegerConfig:
    """Jaeger connection configuration."""
    collector_endpoint: str = "http://localhost:14268/api/traces"
    query_endpoint: str = "http://localhost:16686"
    agent_host: str = "localhost"
    agent_port: int = 6831
    service_name: str = "aion"
    batch_size: int = 100


class JaegerTraceStore(TraceStore):
    """
    Jaeger-based distributed trace storage.

    Features:
    - OpenTracing/OpenTelemetry compatible
    - Span batching
    - DAG visualization
    - Service dependency analysis
    """

    def __init__(self, config: JaegerConfig):
        self.config = config
        self._batch: List[Dict] = []
        self._running = False

    async def initialize(self) -> None:
        """Initialize Jaeger connection."""
        self._running = True
        asyncio.create_task(self._flush_loop())
        logger.info("Jaeger trace store initialized")

    async def shutdown(self) -> None:
        """Shutdown Jaeger connection."""
        self._running = False
        await self._flush()

    async def write_spans(self, spans: List[Span]) -> int:
        """Write spans to Jaeger."""
        for span in spans:
            jaeger_span = self._to_jaeger_format(span)
            self._batch.append(jaeger_span)

            if len(self._batch) >= self.config.batch_size:
                await self._flush()

        return len(spans)

    def _to_jaeger_format(self, span: Span) -> Dict[str, Any]:
        """Convert span to Jaeger Thrift format."""
        return {
            "traceIdLow": int(span.trace_id[:16], 16) if len(span.trace_id) >= 16 else 0,
            "traceIdHigh": int(span.trace_id[16:32], 16) if len(span.trace_id) >= 32 else 0,
            "spanId": int(span.span_id[:16], 16) if span.span_id else 0,
            "parentSpanId": int(span.parent_span_id[:16], 16) if span.parent_span_id else 0,
            "operationName": span.operation_name,
            "startTime": int(span.start_time.timestamp() * 1_000_000),
            "duration": int((span.duration_ms or 0) * 1000),
            "tags": [
                {"key": k, "vStr": str(v)}
                for k, v in span.attributes.items()
            ],
            "logs": [
                {
                    "timestamp": int(e.timestamp.timestamp() * 1_000_000),
                    "fields": [{"key": "event", "vStr": e.name}],
                }
                for e in span.events
            ],
        }

    async def _flush(self) -> None:
        """Flush batch to Jaeger."""
        if not self._batch:
            return

        batch = self._batch
        self._batch = []

        try:
            import aiohttp

            payload = {
                "process": {
                    "serviceName": self.config.service_name,
                },
                "spans": batch,
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.config.collector_endpoint,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                ) as response:
                    if response.status != 202:
                        logger.error(f"Jaeger flush failed: {response.status}")
                        self._batch = batch + self._batch

        except Exception as e:
            logger.error(f"Error flushing to Jaeger: {e}")
            self._batch = batch + self._batch

    async def _flush_loop(self) -> None:
        """Periodic flush loop."""
        while self._running:
            await asyncio.sleep(5.0)
            await self._flush()

    async def get_trace(self, trace_id: str) -> Optional[Trace]:
        """Get a trace from Jaeger."""
        try:
            import aiohttp

            url = f"{self.config.query_endpoint}/api/traces/{trace_id}"

            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._from_jaeger_format(data)

        except Exception as e:
            logger.error(f"Error fetching trace from Jaeger: {e}")

        return None

    def _from_jaeger_format(self, data: Dict) -> Trace:
        """Convert Jaeger format to Trace."""
        # Simplified conversion
        trace_id = data.get("traceID", "")
        spans = []

        for span_data in data.get("spans", []):
            span = Span(
                trace_id=trace_id,
                span_id=span_data.get("spanID", ""),
                operation_name=span_data.get("operationName", ""),
                service_name=data.get("processes", {}).get(
                    span_data.get("processID", ""), {}
                ).get("serviceName", ""),
            )
            spans.append(span)

        return Trace(trace_id=trace_id, spans=spans)

    async def query_traces(
        self,
        service_name: Optional[str] = None,
        operation_name: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Trace]:
        """Query traces from Jaeger."""
        try:
            import aiohttp

            params = {"limit": limit}
            if service_name:
                params["service"] = service_name
            if operation_name:
                params["operation"] = operation_name
            if start_time:
                params["start"] = int(start_time.timestamp() * 1_000_000)
            if end_time:
                params["end"] = int(end_time.timestamp() * 1_000_000)

            url = f"{self.config.query_endpoint}/api/traces"

            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return [self._from_jaeger_format(t) for t in data.get("data", [])]

        except Exception as e:
            logger.error(f"Error querying Jaeger: {e}")

        return []


# =============================================================================
# Elasticsearch/Loki Log Storage
# =============================================================================

@dataclass
class ElasticsearchConfig:
    """Elasticsearch connection configuration."""
    hosts: List[str] = field(default_factory=lambda: ["http://localhost:9200"])
    index_prefix: str = "aion-logs"
    username: str = ""
    password: str = ""
    batch_size: int = 1000
    flush_interval_seconds: float = 5.0


class ElasticsearchLogStore(LogStore):
    """
    Elasticsearch-based log storage.

    Features:
    - Full-text search
    - Aggregations
    - Index lifecycle management
    - Distributed storage
    """

    def __init__(self, config: ElasticsearchConfig):
        self.config = config
        self._client = None
        self._batch: List[Dict] = []
        self._running = False

    async def initialize(self) -> None:
        """Initialize Elasticsearch connection."""
        try:
            from elasticsearch import AsyncElasticsearch

            auth = None
            if self.config.username and self.config.password:
                auth = (self.config.username, self.config.password)

            self._client = AsyncElasticsearch(
                hosts=self.config.hosts,
                basic_auth=auth,
            )

            # Create index template
            await self._create_index_template()

            self._running = True
            asyncio.create_task(self._flush_loop())

            logger.info("Elasticsearch log store initialized")

        except ImportError:
            logger.warning("elasticsearch not installed, using mock mode")
            self._running = True

    async def _create_index_template(self) -> None:
        """Create index template for logs."""
        if not self._client:
            return

        template = {
            "index_patterns": [f"{self.config.index_prefix}-*"],
            "template": {
                "settings": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0,
                },
                "mappings": {
                    "properties": {
                        "@timestamp": {"type": "date"},
                        "level": {"type": "keyword"},
                        "logger": {"type": "keyword"},
                        "message": {"type": "text"},
                        "trace_id": {"type": "keyword"},
                        "span_id": {"type": "keyword"},
                        "extra": {"type": "object", "dynamic": True},
                    },
                },
            },
        }

        try:
            await self._client.indices.put_index_template(
                name=f"{self.config.index_prefix}-template",
                body=template,
            )
        except Exception as e:
            logger.error(f"Error creating ES template: {e}")

    async def shutdown(self) -> None:
        """Shutdown Elasticsearch connection."""
        self._running = False
        await self._flush()

        if self._client:
            await self._client.close()

    async def write_logs(self, logs: List[LogEntry]) -> int:
        """Write logs to Elasticsearch."""
        for log in logs:
            doc = self._to_es_doc(log)
            self._batch.append(doc)

            if len(self._batch) >= self.config.batch_size:
                await self._flush()

        return len(logs)

    def _to_es_doc(self, log: LogEntry) -> Dict[str, Any]:
        """Convert log entry to ES document."""
        return {
            "_index": f"{self.config.index_prefix}-{log.timestamp.strftime('%Y.%m.%d')}",
            "_source": {
                "@timestamp": log.timestamp.isoformat(),
                "level": log.level.value,
                "logger": log.logger_name,
                "message": log.message,
                "trace_id": log.trace_id,
                "span_id": log.span_id,
                "extra": log.extra,
            },
        }

    async def _flush(self) -> None:
        """Flush batch to Elasticsearch."""
        if not self._batch or not self._client:
            return

        batch = self._batch
        self._batch = []

        try:
            from elasticsearch.helpers import async_bulk

            success, failed = await async_bulk(
                self._client,
                batch,
                raise_on_error=False,
            )

            if failed:
                logger.error(f"ES bulk insert failed: {len(failed)} docs")

        except Exception as e:
            logger.error(f"Error flushing to Elasticsearch: {e}")

    async def _flush_loop(self) -> None:
        """Periodic flush loop."""
        while self._running:
            await asyncio.sleep(self.config.flush_interval_seconds)
            await self._flush()

    async def query_logs(
        self,
        level: Optional[str] = None,
        logger_name: Optional[str] = None,
        message_contains: Optional[str] = None,
        trace_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000,
    ) -> List[LogEntry]:
        """Query logs from Elasticsearch."""
        if not self._client:
            return []

        query = {"bool": {"must": []}}

        if level:
            query["bool"]["must"].append({"term": {"level": level}})
        if logger_name:
            query["bool"]["must"].append({"term": {"logger": logger_name}})
        if message_contains:
            query["bool"]["must"].append({"match": {"message": message_contains}})
        if trace_id:
            query["bool"]["must"].append({"term": {"trace_id": trace_id}})
        if start_time or end_time:
            range_query = {"@timestamp": {}}
            if start_time:
                range_query["@timestamp"]["gte"] = start_time.isoformat()
            if end_time:
                range_query["@timestamp"]["lte"] = end_time.isoformat()
            query["bool"]["must"].append({"range": range_query})

        try:
            result = await self._client.search(
                index=f"{self.config.index_prefix}-*",
                body={
                    "query": query if query["bool"]["must"] else {"match_all": {}},
                    "size": limit,
                    "sort": [{"@timestamp": "desc"}],
                },
            )

            logs = []
            for hit in result["hits"]["hits"]:
                source = hit["_source"]
                from aion.observability.types import LogLevel

                log = LogEntry(
                    timestamp=datetime.fromisoformat(source["@timestamp"].replace("Z", "+00:00")),
                    level=LogLevel(source.get("level", "INFO")),
                    message=source.get("message", ""),
                    logger_name=source.get("logger", ""),
                    trace_id=source.get("trace_id"),
                    span_id=source.get("span_id"),
                    extra=source.get("extra", {}),
                )
                logs.append(log)

            return logs

        except Exception as e:
            logger.error(f"Error querying Elasticsearch: {e}")
            return []


# =============================================================================
# ClickHouse Unified Storage
# =============================================================================

@dataclass
class ClickHouseConfig:
    """ClickHouse connection configuration."""
    host: str = "localhost"
    port: int = 9000
    database: str = "observability"
    user: str = "default"
    password: str = ""
    cluster: str = ""


class ClickHouseStore:
    """
    ClickHouse-based unified storage for all observability data.

    Features:
    - Column-oriented storage
    - Excellent compression
    - Materialized views
    - Distributed queries
    """

    def __init__(self, config: ClickHouseConfig):
        self.config = config
        self._client = None

    async def initialize(self) -> None:
        """Initialize ClickHouse connection."""
        try:
            from clickhouse_driver import Client

            self._client = Client(
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
                user=self.config.user,
                password=self.config.password,
            )

            await self._create_schema()

            logger.info("ClickHouse store initialized")

        except ImportError:
            logger.warning("clickhouse-driver not installed, using mock mode")

    async def _create_schema(self) -> None:
        """Create ClickHouse schema."""
        if not self._client:
            return

        # Metrics table
        self._client.execute('''
            CREATE TABLE IF NOT EXISTS metrics (
                timestamp DateTime64(3),
                name String,
                value Float64,
                labels Map(String, String)
            ) ENGINE = MergeTree()
            ORDER BY (name, timestamp)
            TTL timestamp + INTERVAL 30 DAY
        ''')

        # Traces table
        self._client.execute('''
            CREATE TABLE IF NOT EXISTS spans (
                timestamp DateTime64(3),
                trace_id String,
                span_id String,
                parent_span_id String,
                operation_name String,
                service_name String,
                duration_ms Float64,
                status UInt8,
                attributes Map(String, String)
            ) ENGINE = MergeTree()
            ORDER BY (trace_id, timestamp)
            TTL timestamp + INTERVAL 7 DAY
        ''')

        # Logs table
        self._client.execute('''
            CREATE TABLE IF NOT EXISTS logs (
                timestamp DateTime64(3),
                level String,
                logger String,
                message String,
                trace_id Nullable(String),
                span_id Nullable(String),
                extra String
            ) ENGINE = MergeTree()
            ORDER BY (timestamp)
            TTL timestamp + INTERVAL 14 DAY
        ''')

    async def shutdown(self) -> None:
        """Shutdown ClickHouse connection."""
        if self._client:
            self._client.disconnect()

    async def write_metrics(self, metrics: List[Metric]) -> int:
        """Write metrics to ClickHouse."""
        if not self._client:
            return 0

        data = [
            (
                metric.timestamp,
                metric.name,
                metric.value,
                metric.labels,
            )
            for metric in metrics
        ]

        self._client.execute(
            "INSERT INTO metrics (timestamp, name, value, labels) VALUES",
            data,
        )

        return len(metrics)

    async def write_spans(self, spans: List[Span]) -> int:
        """Write spans to ClickHouse."""
        if not self._client:
            return 0

        data = [
            (
                span.start_time,
                span.trace_id,
                span.span_id,
                span.parent_span_id or "",
                span.operation_name,
                span.service_name,
                span.duration_ms or 0,
                1 if span.status.is_error else 0,
                span.attributes,
            )
            for span in spans
        ]

        self._client.execute(
            '''INSERT INTO spans
            (timestamp, trace_id, span_id, parent_span_id, operation_name,
             service_name, duration_ms, status, attributes) VALUES''',
            data,
        )

        return len(spans)

    async def write_logs(self, logs: List[LogEntry]) -> int:
        """Write logs to ClickHouse."""
        if not self._client:
            return 0

        data = [
            (
                log.timestamp,
                log.level.value,
                log.logger_name,
                log.message,
                log.trace_id,
                log.span_id,
                json.dumps(log.extra) if log.extra else "{}",
            )
            for log in logs
        ]

        self._client.execute(
            '''INSERT INTO logs
            (timestamp, level, logger, message, trace_id, span_id, extra) VALUES''',
            data,
        )

        return len(logs)

    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        if not self._client:
            return {}

        stats = {}

        for table in ["metrics", "spans", "logs"]:
            result = self._client.execute(
                f"SELECT count(), sum(bytes) FROM system.parts WHERE table = '{table}'"
            )
            if result:
                stats[table] = {
                    "rows": result[0][0],
                    "bytes": result[0][1],
                }

        return stats


# =============================================================================
# Storage Factory
# =============================================================================

def create_metric_store(backend: str, **config) -> MetricStore:
    """Create a metric store based on backend type."""
    if backend == "influxdb":
        return InfluxDBMetricStore(InfluxDBConfig(**config))
    elif backend == "timescaledb":
        return TimescaleDBMetricStore(TimescaleDBConfig(**config))
    elif backend == "clickhouse":
        return ClickHouseStore(ClickHouseConfig(**config))
    else:
        from aion.observability.storage.memory import InMemoryMetricStore
        return InMemoryMetricStore()


def create_trace_store(backend: str, **config) -> TraceStore:
    """Create a trace store based on backend type."""
    if backend == "jaeger":
        return JaegerTraceStore(JaegerConfig(**config))
    elif backend == "clickhouse":
        return ClickHouseStore(ClickHouseConfig(**config))
    else:
        from aion.observability.storage.memory import InMemoryTraceStore
        return InMemoryTraceStore()


def create_log_store(backend: str, **config) -> LogStore:
    """Create a log store based on backend type."""
    if backend == "elasticsearch":
        return ElasticsearchLogStore(ElasticsearchConfig(**config))
    elif backend == "clickhouse":
        return ClickHouseStore(ClickHouseConfig(**config))
    else:
        from aion.observability.storage.memory import InMemoryLogStore
        return InMemoryLogStore()
