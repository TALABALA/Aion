"""
Plugin Metrics Collection

Provides metrics collection for plugin operations including
counters, gauges, and histograms.
"""

from __future__ import annotations

import asyncio
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class MetricsConfig:
    """Configuration for plugin metrics."""

    enabled: bool = True
    service_name: str = "aion-plugins"

    # Collection
    collection_interval: float = 60.0  # seconds

    # Export
    export_enabled: bool = False
    export_endpoint: Optional[str] = None
    export_format: str = "prometheus"  # prometheus, otlp

    # Histogram buckets (for latency in ms)
    histogram_buckets: list[float] = field(default_factory=lambda: [
        1, 5, 10, 25, 50, 75, 100, 250, 500, 750, 1000, 2500, 5000, 10000
    ])


class Counter:
    """
    A monotonically increasing counter.

    Usage:
        counter = Counter("requests_total", ["method", "status"])
        counter.inc(labels={"method": "GET", "status": "200"})
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        labels: Optional[list[str]] = None,
    ):
        self.name = name
        self.description = description
        self.label_names = labels or []
        self._values: Dict[tuple, float] = defaultdict(float)
        self._created = time.time()

    def inc(self, value: float = 1.0, labels: Optional[dict[str, str]] = None) -> None:
        """Increment the counter."""
        if value < 0:
            raise ValueError("Counter can only be incremented")
        key = self._labels_to_key(labels)
        self._values[key] += value

    def get(self, labels: Optional[dict[str, str]] = None) -> float:
        """Get counter value."""
        key = self._labels_to_key(labels)
        return self._values.get(key, 0.0)

    def _labels_to_key(self, labels: Optional[dict[str, str]]) -> tuple:
        """Convert labels to hashable key."""
        if not labels:
            return ()
        return tuple(labels.get(name, "") for name in self.label_names)

    def collect(self) -> list[dict[str, Any]]:
        """Collect all metric values."""
        results = []
        for key, value in self._values.items():
            labels = dict(zip(self.label_names, key))
            results.append({
                "name": self.name,
                "type": "counter",
                "value": value,
                "labels": labels,
                "timestamp": time.time(),
            })
        return results


class Gauge:
    """
    A gauge that can go up or down.

    Usage:
        gauge = Gauge("active_connections", ["host"])
        gauge.set(10, labels={"host": "localhost"})
        gauge.inc(labels={"host": "localhost"})
        gauge.dec(labels={"host": "localhost"})
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        labels: Optional[list[str]] = None,
    ):
        self.name = name
        self.description = description
        self.label_names = labels or []
        self._values: Dict[tuple, float] = defaultdict(float)

    def set(self, value: float, labels: Optional[dict[str, str]] = None) -> None:
        """Set gauge value."""
        key = self._labels_to_key(labels)
        self._values[key] = value

    def inc(self, value: float = 1.0, labels: Optional[dict[str, str]] = None) -> None:
        """Increment gauge."""
        key = self._labels_to_key(labels)
        self._values[key] += value

    def dec(self, value: float = 1.0, labels: Optional[dict[str, str]] = None) -> None:
        """Decrement gauge."""
        key = self._labels_to_key(labels)
        self._values[key] -= value

    def get(self, labels: Optional[dict[str, str]] = None) -> float:
        """Get gauge value."""
        key = self._labels_to_key(labels)
        return self._values.get(key, 0.0)

    def _labels_to_key(self, labels: Optional[dict[str, str]]) -> tuple:
        """Convert labels to hashable key."""
        if not labels:
            return ()
        return tuple(labels.get(name, "") for name in self.label_names)

    def collect(self) -> list[dict[str, Any]]:
        """Collect all metric values."""
        results = []
        for key, value in self._values.items():
            labels = dict(zip(self.label_names, key))
            results.append({
                "name": self.name,
                "type": "gauge",
                "value": value,
                "labels": labels,
                "timestamp": time.time(),
            })
        return results


class Histogram:
    """
    A histogram for measuring distributions.

    Usage:
        histogram = Histogram("request_latency_ms", ["method"])
        histogram.observe(150.5, labels={"method": "GET"})
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        labels: Optional[list[str]] = None,
        buckets: Optional[list[float]] = None,
    ):
        self.name = name
        self.description = description
        self.label_names = labels or []
        self.buckets = sorted(buckets or [10, 25, 50, 100, 250, 500, 1000])

        self._observations: Dict[tuple, list[float]] = defaultdict(list)
        self._bucket_counts: Dict[tuple, Dict[float, int]] = defaultdict(
            lambda: {b: 0 for b in self.buckets}
        )
        self._sum: Dict[tuple, float] = defaultdict(float)
        self._count: Dict[tuple, int] = defaultdict(int)

    def observe(self, value: float, labels: Optional[dict[str, str]] = None) -> None:
        """Observe a value."""
        key = self._labels_to_key(labels)
        self._observations[key].append(value)
        self._sum[key] += value
        self._count[key] += 1

        # Update bucket counts
        for bucket in self.buckets:
            if value <= bucket:
                self._bucket_counts[key][bucket] += 1

        # Keep only recent observations for percentile calculation
        if len(self._observations[key]) > 10000:
            self._observations[key] = self._observations[key][-10000:]

    def get_percentile(
        self,
        percentile: float,
        labels: Optional[dict[str, str]] = None,
    ) -> float:
        """Get percentile value."""
        key = self._labels_to_key(labels)
        observations = self._observations.get(key, [])
        if not observations:
            return 0.0

        sorted_obs = sorted(observations)
        idx = int(len(sorted_obs) * percentile / 100)
        return sorted_obs[min(idx, len(sorted_obs) - 1)]

    def _labels_to_key(self, labels: Optional[dict[str, str]]) -> tuple:
        """Convert labels to hashable key."""
        if not labels:
            return ()
        return tuple(labels.get(name, "") for name in self.label_names)

    def collect(self) -> list[dict[str, Any]]:
        """Collect all metric values."""
        results = []
        for key in self._observations:
            labels = dict(zip(self.label_names, key))
            observations = self._observations[key]

            if observations:
                results.append({
                    "name": self.name,
                    "type": "histogram",
                    "labels": labels,
                    "timestamp": time.time(),
                    "count": self._count[key],
                    "sum": self._sum[key],
                    "buckets": {
                        str(b): self._bucket_counts[key][b]
                        for b in self.buckets
                    },
                    "p50": self.get_percentile(50, labels),
                    "p90": self.get_percentile(90, labels),
                    "p95": self.get_percentile(95, labels),
                    "p99": self.get_percentile(99, labels),
                    "min": min(observations),
                    "max": max(observations),
                    "mean": statistics.mean(observations),
                })
        return results


class Timer:
    """
    Context manager for timing operations.

    Usage:
        with metrics.timer("operation_duration", labels={"op": "fetch"}):
            # ... timed operation ...
    """

    def __init__(
        self,
        histogram: Histogram,
        labels: Optional[dict[str, str]] = None,
    ):
        self.histogram = histogram
        self.labels = labels
        self._start: Optional[float] = None

    def __enter__(self) -> "Timer":
        self._start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._start is not None:
            duration_ms = (time.time() - self._start) * 1000
            self.histogram.observe(duration_ms, self.labels)

    async def __aenter__(self) -> "Timer":
        self._start = time.time()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._start is not None:
            duration_ms = (time.time() - self._start) * 1000
            self.histogram.observe(duration_ms, self.labels)


class PluginMetrics:
    """
    Metrics collection for a single plugin.

    Provides pre-defined metrics for common plugin operations.
    """

    def __init__(
        self,
        plugin_id: str,
        config: Optional[MetricsConfig] = None,
    ):
        self.plugin_id = plugin_id
        self.config = config or MetricsConfig()

        # Pre-defined metrics
        self.calls_total = Counter(
            "plugin_calls_total",
            "Total number of plugin calls",
            ["plugin_id", "method", "status"],
        )
        self.call_duration = Histogram(
            "plugin_call_duration_ms",
            "Plugin call duration in milliseconds",
            ["plugin_id", "method"],
            buckets=self.config.histogram_buckets,
        )
        self.errors_total = Counter(
            "plugin_errors_total",
            "Total number of plugin errors",
            ["plugin_id", "error_type"],
        )
        self.active_operations = Gauge(
            "plugin_active_operations",
            "Number of active operations",
            ["plugin_id"],
        )

        # Custom metrics
        self._custom_counters: dict[str, Counter] = {}
        self._custom_gauges: dict[str, Gauge] = {}
        self._custom_histograms: dict[str, Histogram] = {}

    def record_call(
        self,
        method: str,
        duration_ms: float,
        success: bool = True,
    ) -> None:
        """Record a plugin call."""
        labels = {
            "plugin_id": self.plugin_id,
            "method": method,
            "status": "success" if success else "error",
        }
        self.calls_total.inc(labels=labels)
        self.call_duration.observe(
            duration_ms,
            labels={"plugin_id": self.plugin_id, "method": method},
        )

    def record_error(self, error_type: str) -> None:
        """Record a plugin error."""
        self.errors_total.inc(labels={
            "plugin_id": self.plugin_id,
            "error_type": error_type,
        })

    def timer(self, method: str) -> Timer:
        """Create a timer for measuring operation duration."""
        return Timer(
            self.call_duration,
            labels={"plugin_id": self.plugin_id, "method": method},
        )

    def create_counter(
        self,
        name: str,
        description: str = "",
        labels: Optional[list[str]] = None,
    ) -> Counter:
        """Create a custom counter."""
        counter = Counter(name, description, labels)
        self._custom_counters[name] = counter
        return counter

    def create_gauge(
        self,
        name: str,
        description: str = "",
        labels: Optional[list[str]] = None,
    ) -> Gauge:
        """Create a custom gauge."""
        gauge = Gauge(name, description, labels)
        self._custom_gauges[name] = gauge
        return gauge

    def create_histogram(
        self,
        name: str,
        description: str = "",
        labels: Optional[list[str]] = None,
        buckets: Optional[list[float]] = None,
    ) -> Histogram:
        """Create a custom histogram."""
        histogram = Histogram(name, description, labels, buckets)
        self._custom_histograms[name] = histogram
        return histogram

    def collect_all(self) -> list[dict[str, Any]]:
        """Collect all metrics."""
        metrics = []
        metrics.extend(self.calls_total.collect())
        metrics.extend(self.call_duration.collect())
        metrics.extend(self.errors_total.collect())
        metrics.extend(self.active_operations.collect())

        for counter in self._custom_counters.values():
            metrics.extend(counter.collect())
        for gauge in self._custom_gauges.values():
            metrics.extend(gauge.collect())
        for histogram in self._custom_histograms.values():
            metrics.extend(histogram.collect())

        return metrics

    def get_stats(self) -> dict[str, Any]:
        """Get metrics statistics."""
        return {
            "plugin_id": self.plugin_id,
            "total_calls": sum(
                v for k, v in self.calls_total._values.items()
            ),
            "total_errors": sum(
                v for k, v in self.errors_total._values.items()
            ),
            "custom_metrics": {
                "counters": list(self._custom_counters.keys()),
                "gauges": list(self._custom_gauges.keys()),
                "histograms": list(self._custom_histograms.keys()),
            },
        }


class MetricsManager:
    """
    Manages metrics across multiple plugins.

    Provides centralized metrics collection and export.
    """

    def __init__(self, config: Optional[MetricsConfig] = None):
        self.config = config or MetricsConfig()
        self._metrics: dict[str, PluginMetrics] = {}
        self._export_task: Optional[asyncio.Task] = None

    def get_metrics(self, plugin_id: str) -> PluginMetrics:
        """Get or create metrics for a plugin."""
        if plugin_id not in self._metrics:
            self._metrics[plugin_id] = PluginMetrics(plugin_id, self.config)
        return self._metrics[plugin_id]

    def remove_metrics(self, plugin_id: str) -> None:
        """Remove metrics for a plugin."""
        self._metrics.pop(plugin_id, None)

    def collect_all(self) -> list[dict[str, Any]]:
        """Collect all metrics from all plugins."""
        all_metrics = []
        for plugin_metrics in self._metrics.values():
            all_metrics.extend(plugin_metrics.collect_all())
        return all_metrics

    def to_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        metrics = self.collect_all()

        for metric in metrics:
            name = metric["name"].replace(".", "_")
            labels_str = ",".join(
                f'{k}="{v}"' for k, v in metric.get("labels", {}).items()
            )
            if labels_str:
                labels_str = f"{{{labels_str}}}"

            if metric["type"] == "counter":
                lines.append(f"# TYPE {name} counter")
                lines.append(f"{name}{labels_str} {metric['value']}")

            elif metric["type"] == "gauge":
                lines.append(f"# TYPE {name} gauge")
                lines.append(f"{name}{labels_str} {metric['value']}")

            elif metric["type"] == "histogram":
                lines.append(f"# TYPE {name} histogram")
                for bucket, count in metric.get("buckets", {}).items():
                    bucket_labels = f'{labels_str[:-1]},le="{bucket}"}}'
                    lines.append(f"{name}_bucket{bucket_labels} {count}")
                lines.append(f'{name}_sum{labels_str} {metric["sum"]}')
                lines.append(f'{name}_count{labels_str} {metric["count"]}')

        return "\n".join(lines)

    async def start_export(self) -> None:
        """Start periodic metrics export."""
        if not self.config.export_enabled:
            return

        self._export_task = asyncio.create_task(self._export_loop())

    async def stop_export(self) -> None:
        """Stop metrics export."""
        if self._export_task:
            self._export_task.cancel()
            try:
                await self._export_task
            except asyncio.CancelledError:
                pass

    async def _export_loop(self) -> None:
        """Periodic export loop."""
        while True:
            try:
                await asyncio.sleep(self.config.collection_interval)
                # Export logic would go here
                logger.debug("Metrics exported", count=len(self.collect_all()))
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Metrics export failed: {e}")

    def get_stats(self) -> dict[str, Any]:
        """Get statistics for all plugins."""
        return {
            "total_plugins": len(self._metrics),
            "config": {
                "enabled": self.config.enabled,
                "export_enabled": self.config.export_enabled,
            },
            "per_plugin": {
                pid: pm.get_stats()
                for pid, pm in self._metrics.items()
            },
        }
