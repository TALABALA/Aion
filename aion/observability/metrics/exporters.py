"""
AION Metrics Exporters

Export metrics to various backends:
- Prometheus (pull-based)
- StatsD (push-based)
- OTLP (OpenTelemetry Protocol)
- Custom HTTP endpoints
"""

from __future__ import annotations

import asyncio
import json
import socket
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import structlog

from aion.observability.types import Metric, MetricType

logger = structlog.get_logger(__name__)


class MetricExporter(ABC):
    """Base class for metric exporters."""

    @abstractmethod
    async def export(self, metrics: List[Metric]) -> bool:
        """Export metrics. Returns True on success."""
        pass

    async def shutdown(self) -> None:
        """Cleanup resources."""
        pass


class PrometheusExporter(MetricExporter):
    """
    Prometheus exporter that serves metrics via HTTP.

    Provides a /metrics endpoint in Prometheus exposition format.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 9090,
        path: str = "/metrics",
    ):
        self.host = host
        self.port = port
        self.path = path

        self._metrics_cache: Dict[str, Metric] = {}
        self._server = None
        self._running = False

    async def start_server(self) -> None:
        """Start the HTTP server for Prometheus scraping."""
        from aiohttp import web

        app = web.Application()
        app.router.add_get(self.path, self._handle_metrics)
        app.router.add_get("/", self._handle_health)

        runner = web.AppRunner(app)
        await runner.setup()

        self._server = web.TCPSite(runner, self.host, self.port)
        await self._server.start()

        self._running = True
        logger.info(f"Prometheus exporter started on {self.host}:{self.port}{self.path}")

    async def _handle_metrics(self, request) -> "web.Response":
        """Handle /metrics endpoint."""
        from aiohttp import web

        output = self._format_prometheus()
        return web.Response(
            text=output,
            content_type="text/plain; version=0.0.4; charset=utf-8",
        )

    async def _handle_health(self, request) -> "web.Response":
        """Handle health check."""
        from aiohttp import web
        return web.Response(text="OK")

    async def export(self, metrics: List[Metric]) -> bool:
        """Cache metrics for Prometheus scraping."""
        for metric in metrics:
            key = f"{metric.name}:{metric.labels_key()}"
            self._metrics_cache[key] = metric
        return True

    def _format_prometheus(self) -> str:
        """Format metrics in Prometheus exposition format."""
        lines = []
        metrics_by_name: Dict[str, List[Metric]] = {}

        # Group by name
        for metric in self._metrics_cache.values():
            if metric.name not in metrics_by_name:
                metrics_by_name[metric.name] = []
            metrics_by_name[metric.name].append(metric)

        for name, metric_list in sorted(metrics_by_name.items()):
            sample = metric_list[0]

            # Add HELP and TYPE
            if sample.description:
                lines.append(f"# HELP {name} {sample.description}")
            lines.append(f"# TYPE {name} {sample.metric_type.value}")

            for metric in metric_list:
                lines.append(metric.to_prometheus())

        return "\n".join(lines) + "\n"

    async def shutdown(self) -> None:
        """Stop the HTTP server."""
        if self._server:
            await self._server.stop()
            self._running = False


class StatsDBatchExporter(MetricExporter):
    """
    StatsD exporter with batching support.

    Sends metrics to a StatsD server (Datadog, etc).
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8125,
        prefix: str = "aion",
        batch_size: int = 50,
        flush_interval: float = 1.0,
    ):
        self.host = host
        self.port = port
        self.prefix = prefix
        self.batch_size = batch_size
        self.flush_interval = flush_interval

        self._socket: Optional[socket.socket] = None
        self._buffer: List[str] = []
        self._lock = asyncio.Lock()

    async def _ensure_socket(self) -> None:
        """Ensure UDP socket is created."""
        if self._socket is None:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    async def export(self, metrics: List[Metric]) -> bool:
        """Export metrics to StatsD."""
        try:
            await self._ensure_socket()

            async with self._lock:
                for metric in metrics:
                    statsd_line = self._format_statsd(metric)
                    self._buffer.append(statsd_line)

                    if len(self._buffer) >= self.batch_size:
                        await self._flush()

            return True
        except Exception as e:
            logger.error(f"StatsD export error: {e}")
            return False

    async def _flush(self) -> None:
        """Flush buffer to StatsD."""
        if not self._buffer or not self._socket:
            return

        data = "\n".join(self._buffer).encode('utf-8')
        try:
            self._socket.sendto(data, (self.host, self.port))
        except Exception as e:
            logger.error(f"StatsD send error: {e}")

        self._buffer.clear()

    def _format_statsd(self, metric: Metric) -> str:
        """Format metric for StatsD."""
        name = f"{self.prefix}.{metric.name}"

        # Add labels as tags
        if metric.labels:
            tags = ",".join(f"{k}:{v}" for k, v in metric.labels.items())
            name = f"{name}|#{tags}"

        type_char = {
            MetricType.COUNTER: "c",
            MetricType.GAUGE: "g",
            MetricType.HISTOGRAM: "h",
            MetricType.SUMMARY: "ms",
        }.get(metric.metric_type, "g")

        return f"{name}:{metric.value}|{type_char}"

    async def shutdown(self) -> None:
        """Flush and close socket."""
        async with self._lock:
            await self._flush()
            if self._socket:
                self._socket.close()
                self._socket = None


class OTLPMetricExporter(MetricExporter):
    """
    OpenTelemetry Protocol (OTLP) metric exporter.

    Exports metrics via HTTP/protobuf or HTTP/JSON.
    """

    def __init__(
        self,
        endpoint: str = "http://localhost:4318/v1/metrics",
        headers: Dict[str, str] = None,
        timeout: float = 30.0,
        compression: str = "gzip",
    ):
        self.endpoint = endpoint
        self.headers = headers or {}
        self.timeout = timeout
        self.compression = compression

        self._session = None

    async def _ensure_session(self) -> None:
        """Ensure HTTP session exists."""
        if self._session is None:
            import httpx
            self._session = httpx.AsyncClient(timeout=self.timeout)

    async def export(self, metrics: List[Metric]) -> bool:
        """Export metrics via OTLP."""
        if not metrics:
            return True

        try:
            await self._ensure_session()

            payload = self._build_otlp_payload(metrics)

            headers = {
                "Content-Type": "application/json",
                **self.headers,
            }

            response = await self._session.post(
                self.endpoint,
                json=payload,
                headers=headers,
            )

            if response.status_code >= 400:
                logger.error(f"OTLP export failed: {response.status_code}")
                return False

            return True
        except Exception as e:
            logger.error(f"OTLP export error: {e}")
            return False

    def _build_otlp_payload(self, metrics: List[Metric]) -> dict:
        """Build OTLP payload."""
        resource_metrics = {
            "resource": {
                "attributes": [
                    {"key": "service.name", "value": {"stringValue": "aion"}},
                ],
            },
            "scopeMetrics": [
                {
                    "scope": {"name": "aion.observability"},
                    "metrics": [self._metric_to_otlp(m) for m in metrics],
                }
            ],
        }

        return {"resourceMetrics": [resource_metrics]}

    def _metric_to_otlp(self, metric: Metric) -> dict:
        """Convert metric to OTLP format."""
        base = {
            "name": metric.name,
            "description": metric.description,
            "unit": metric.unit,
        }

        data_point = {
            "timeUnixNano": int(metric.timestamp.timestamp() * 1e9),
            "attributes": [
                {"key": k, "value": {"stringValue": v}}
                for k, v in metric.labels.items()
            ],
        }

        if metric.metric_type == MetricType.COUNTER:
            data_point["asInt"] = int(metric.value)
            base["sum"] = {
                "dataPoints": [data_point],
                "aggregationTemporality": 2,  # CUMULATIVE
                "isMonotonic": True,
            }
        elif metric.metric_type == MetricType.GAUGE:
            data_point["asDouble"] = metric.value
            base["gauge"] = {"dataPoints": [data_point]}
        elif metric.metric_type == MetricType.HISTOGRAM:
            data_point["asDouble"] = metric.value
            base["histogram"] = {
                "dataPoints": [data_point],
                "aggregationTemporality": 2,
            }
        else:
            data_point["asDouble"] = metric.value
            base["gauge"] = {"dataPoints": [data_point]}

        return base

    async def shutdown(self) -> None:
        """Close HTTP session."""
        if self._session:
            await self._session.aclose()
            self._session = None


class HTTPPushExporter(MetricExporter):
    """
    Generic HTTP push exporter.

    Pushes metrics to any HTTP endpoint as JSON.
    """

    def __init__(
        self,
        endpoint: str,
        headers: Dict[str, str] = None,
        method: str = "POST",
        timeout: float = 30.0,
        batch_size: int = 100,
    ):
        self.endpoint = endpoint
        self.headers = headers or {}
        self.method = method
        self.timeout = timeout
        self.batch_size = batch_size

        self._session = None

    async def _ensure_session(self) -> None:
        """Ensure HTTP session exists."""
        if self._session is None:
            import httpx
            self._session = httpx.AsyncClient(timeout=self.timeout)

    async def export(self, metrics: List[Metric]) -> bool:
        """Export metrics via HTTP."""
        if not metrics:
            return True

        try:
            await self._ensure_session()

            # Export in batches
            for i in range(0, len(metrics), self.batch_size):
                batch = metrics[i:i + self.batch_size]
                payload = {
                    "metrics": [m.to_dict() for m in batch],
                    "timestamp": datetime.utcnow().isoformat(),
                }

                headers = {
                    "Content-Type": "application/json",
                    **self.headers,
                }

                response = await self._session.request(
                    self.method,
                    self.endpoint,
                    json=payload,
                    headers=headers,
                )

                if response.status_code >= 400:
                    logger.error(f"HTTP export failed: {response.status_code}")
                    return False

            return True
        except Exception as e:
            logger.error(f"HTTP export error: {e}")
            return False

    async def shutdown(self) -> None:
        """Close HTTP session."""
        if self._session:
            await self._session.aclose()
            self._session = None


class ConsoleExporter(MetricExporter):
    """
    Console exporter for debugging.

    Prints metrics to stdout.
    """

    def __init__(self, format: str = "text"):
        self.format = format

    async def export(self, metrics: List[Metric]) -> bool:
        """Print metrics to console."""
        for metric in metrics:
            if self.format == "json":
                print(json.dumps(metric.to_dict()))
            else:
                print(f"[{metric.timestamp.isoformat()}] {metric.full_name()} = {metric.value}")
        return True


class InMemoryExporter(MetricExporter):
    """
    In-memory exporter for testing.

    Stores metrics in memory for inspection.
    """

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self._metrics: List[Metric] = []

    async def export(self, metrics: List[Metric]) -> bool:
        """Store metrics in memory."""
        self._metrics.extend(metrics)

        # Trim if needed
        if len(self._metrics) > self.max_size:
            self._metrics = self._metrics[-self.max_size:]

        return True

    def get_metrics(self) -> List[Metric]:
        """Get all stored metrics."""
        return list(self._metrics)

    def get_by_name(self, name: str) -> List[Metric]:
        """Get metrics by name."""
        return [m for m in self._metrics if m.name == name]

    def clear(self) -> None:
        """Clear all stored metrics."""
        self._metrics.clear()


class MultiExporter(MetricExporter):
    """
    Export to multiple backends simultaneously.
    """

    def __init__(self, exporters: List[MetricExporter] = None):
        self.exporters = exporters or []

    def add_exporter(self, exporter: MetricExporter) -> None:
        """Add an exporter."""
        self.exporters.append(exporter)

    async def export(self, metrics: List[Metric]) -> bool:
        """Export to all backends."""
        results = await asyncio.gather(
            *[e.export(metrics) for e in self.exporters],
            return_exceptions=True,
        )

        # Log any failures
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Exporter {i} failed: {result}")
            elif not result:
                logger.warning(f"Exporter {i} returned False")

        # Return True if at least one succeeded
        return any(r is True for r in results)

    async def shutdown(self) -> None:
        """Shutdown all exporters."""
        await asyncio.gather(
            *[e.shutdown() for e in self.exporters],
            return_exceptions=True,
        )
