"""
AION Metrics Engine

SOTA metrics processing with:
- Metric registration and validation
- Aggregation (sum, avg, min, max, percentiles)
- Time-series storage with rollups
- Prometheus-compatible export
- High cardinality protection
"""

from __future__ import annotations

import asyncio
import statistics
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import structlog

from aion.observability.types import (
    Metric, MetricType, MetricDefinition, MetricSeries, AggregationType,
)
from aion.observability.collector import TelemetryCollector

logger = structlog.get_logger(__name__)


@dataclass
class HistogramBuckets:
    """Histogram bucket storage."""
    buckets: List[float] = field(default_factory=lambda: [
        0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, float('inf')
    ])
    counts: Dict[float, int] = field(default_factory=dict)
    sum: float = 0.0
    count: int = 0

    def __post_init__(self):
        for bucket in self.buckets:
            self.counts[bucket] = 0

    def observe(self, value: float) -> None:
        """Record an observation."""
        self.sum += value
        self.count += 1
        for bucket in self.buckets:
            if value <= bucket:
                self.counts[bucket] += 1

    def get_percentile(self, p: float) -> float:
        """Estimate percentile from histogram."""
        if self.count == 0:
            return 0.0
        target = p * self.count
        for i, bucket in enumerate(self.buckets[:-1]):
            if self.counts[bucket] >= target:
                if i == 0:
                    return bucket / 2
                prev_bucket = self.buckets[i - 1]
                prev_count = self.counts[prev_bucket]
                curr_count = self.counts[bucket]
                ratio = (target - prev_count) / (curr_count - prev_count) if curr_count > prev_count else 0
                return prev_bucket + ratio * (bucket - prev_bucket)
        return self.buckets[-2]


@dataclass
class SummaryQuantiles:
    """Summary quantile storage using reservoir sampling."""
    max_samples: int = 10000
    samples: List[float] = field(default_factory=list)
    count: int = 0
    sum: float = 0.0

    def observe(self, value: float) -> None:
        """Record an observation."""
        self.sum += value
        self.count += 1

        if len(self.samples) < self.max_samples:
            self.samples.append(value)
        else:
            # Reservoir sampling
            import random
            idx = random.randint(0, self.count - 1)
            if idx < self.max_samples:
                self.samples[idx] = value

    def get_quantile(self, q: float) -> float:
        """Get quantile value."""
        if not self.samples:
            return 0.0
        sorted_samples = sorted(self.samples)
        idx = int(q * len(sorted_samples))
        return sorted_samples[min(idx, len(sorted_samples) - 1)]


class MetricsEngine:
    """
    SOTA Metrics processing engine.

    Features:
    - Metric registration and validation
    - Aggregation (sum, avg, min, max, percentiles)
    - Time-series storage with rollups
    - Prometheus-compatible export
    - High cardinality protection
    - Exemplar support for trace correlation
    """

    def __init__(
        self,
        collector: TelemetryCollector,
        aggregation_interval: float = 60.0,
        max_cardinality: int = 10000,
        retention_raw: timedelta = timedelta(hours=1),
        retention_1m: timedelta = timedelta(hours=24),
        retention_1h: timedelta = timedelta(days=7),
    ):
        self.collector = collector
        self.aggregation_interval = aggregation_interval
        self.max_cardinality = max_cardinality
        self.retention_raw = retention_raw
        self.retention_1m = retention_1m
        self.retention_1h = retention_1h

        # Metric definitions
        self._definitions: Dict[str, MetricDefinition] = {}

        # Current values (for counters and gauges)
        self._counters: Dict[str, Dict[str, float]] = defaultdict(dict)
        self._gauges: Dict[str, Dict[str, Tuple[float, datetime]]] = defaultdict(dict)

        # Histograms and summaries
        self._histograms: Dict[str, Dict[str, HistogramBuckets]] = defaultdict(dict)
        self._summaries: Dict[str, Dict[str, SummaryQuantiles]] = defaultdict(dict)

        # Time series storage
        self._series_raw: Dict[str, List[Tuple[datetime, float]]] = defaultdict(list)
        self._series_1m: Dict[str, List[Tuple[datetime, float]]] = defaultdict(list)
        self._series_1h: Dict[str, List[Tuple[datetime, float]]] = defaultdict(list)

        # Exemplars (trace links for metrics)
        self._exemplars: Dict[str, Dict[str, Tuple[str, str, float, datetime]]] = defaultdict(dict)

        # Cardinality tracking
        self._label_cardinality: Dict[str, Set[str]] = defaultdict(set)

        # Lock for thread safety
        self._lock = threading.RLock()

        # Background aggregation
        self._aggregation_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

        # Register built-in metrics
        self._register_builtin_metrics()

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the metrics engine."""
        if self._initialized:
            return

        logger.info("Initializing Metrics Engine")

        # Start background tasks
        self._aggregation_task = asyncio.create_task(self._aggregation_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

        self._initialized = True

    async def shutdown(self) -> None:
        """Shutdown the metrics engine."""
        logger.info("Shutting down Metrics Engine")

        self._shutdown_event.set()

        for task in [self._aggregation_task, self._cleanup_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        self._initialized = False

    def _register_builtin_metrics(self) -> None:
        """Register built-in AION metrics."""
        builtins = [
            MetricDefinition(
                name="aion_requests_total",
                metric_type=MetricType.COUNTER,
                description="Total number of requests",
                labels=["method", "endpoint", "status"],
            ),
            MetricDefinition(
                name="aion_request_duration_seconds",
                metric_type=MetricType.HISTOGRAM,
                description="Request duration in seconds",
                unit="seconds",
                labels=["method", "endpoint"],
            ),
            MetricDefinition(
                name="aion_tokens_used_total",
                metric_type=MetricType.COUNTER,
                description="Total tokens used",
                labels=["model", "type"],
            ),
            MetricDefinition(
                name="aion_tokens_cost_dollars",
                metric_type=MetricType.COUNTER,
                description="Total token cost in dollars",
                unit="dollars",
                labels=["model"],
            ),
            MetricDefinition(
                name="aion_tool_calls_total",
                metric_type=MetricType.COUNTER,
                description="Total tool calls",
                labels=["tool", "status"],
            ),
            MetricDefinition(
                name="aion_tool_duration_seconds",
                metric_type=MetricType.HISTOGRAM,
                description="Tool execution duration",
                unit="seconds",
                labels=["tool"],
            ),
            MetricDefinition(
                name="aion_agents_active",
                metric_type=MetricType.GAUGE,
                description="Number of active agents",
            ),
            MetricDefinition(
                name="aion_goals_active",
                metric_type=MetricType.GAUGE,
                description="Number of active goals",
            ),
            MetricDefinition(
                name="aion_goals_total",
                metric_type=MetricType.COUNTER,
                description="Total goals created",
                labels=["status"],
            ),
            MetricDefinition(
                name="aion_memory_operations_total",
                metric_type=MetricType.COUNTER,
                description="Memory operations",
                labels=["operation"],
            ),
            MetricDefinition(
                name="aion_memory_search_duration_seconds",
                metric_type=MetricType.HISTOGRAM,
                description="Memory search duration",
                unit="seconds",
            ),
            MetricDefinition(
                name="aion_llm_latency_seconds",
                metric_type=MetricType.HISTOGRAM,
                description="LLM API latency",
                unit="seconds",
                labels=["model", "operation"],
            ),
            MetricDefinition(
                name="aion_llm_requests_total",
                metric_type=MetricType.COUNTER,
                description="Total LLM requests",
                labels=["model", "status"],
            ),
            MetricDefinition(
                name="aion_errors_total",
                metric_type=MetricType.COUNTER,
                description="Total errors",
                labels=["component", "error_type"],
            ),
            MetricDefinition(
                name="aion_knowledge_queries_total",
                metric_type=MetricType.COUNTER,
                description="Knowledge graph queries",
                labels=["query_type"],
            ),
            MetricDefinition(
                name="aion_knowledge_entities_total",
                metric_type=MetricType.GAUGE,
                description="Total knowledge graph entities",
            ),
            MetricDefinition(
                name="aion_process_cpu_percent",
                metric_type=MetricType.GAUGE,
                description="Process CPU usage percent",
            ),
            MetricDefinition(
                name="aion_process_memory_bytes",
                metric_type=MetricType.GAUGE,
                description="Process memory usage",
                unit="bytes",
            ),
            MetricDefinition(
                name="aion_spans_total",
                metric_type=MetricType.COUNTER,
                description="Total spans created",
                labels=["service", "operation"],
            ),
            MetricDefinition(
                name="aion_span_duration_seconds",
                metric_type=MetricType.HISTOGRAM,
                description="Span duration",
                unit="seconds",
                labels=["service", "operation"],
            ),
        ]

        for metric_def in builtins:
            self._definitions[metric_def.name] = metric_def

    def register_metric(self, definition: MetricDefinition) -> None:
        """Register a new metric definition."""
        with self._lock:
            self._definitions[definition.name] = definition
            logger.debug(f"Registered metric: {definition.name}")

    def _labels_key(self, labels: Dict[str, str]) -> str:
        """Create a unique key from labels."""
        if not labels:
            return ""
        return ",".join(f"{k}={v}" for k, v in sorted(labels.items()))

    def _check_cardinality(self, name: str, labels_key: str) -> bool:
        """Check if adding this label set would exceed cardinality limit."""
        if labels_key in self._label_cardinality[name]:
            return True
        if len(self._label_cardinality[name]) >= self.max_cardinality:
            logger.warning(
                f"Metric {name} exceeded cardinality limit",
                current=len(self._label_cardinality[name]),
                limit=self.max_cardinality,
            )
            return False
        self._label_cardinality[name].add(labels_key)
        return True

    # === Recording Methods ===

    def inc(
        self,
        name: str,
        value: float = 1.0,
        labels: Dict[str, str] = None,
        exemplar_trace_id: str = None,
        exemplar_span_id: str = None,
    ) -> None:
        """Increment a counter."""
        labels = labels or {}
        key = self._labels_key(labels)

        with self._lock:
            if not self._check_cardinality(name, key):
                return

            current = self._counters[name].get(key, 0.0)
            self._counters[name][key] = current + value

            # Store exemplar
            if exemplar_trace_id and exemplar_span_id:
                self._exemplars[name][key] = (
                    exemplar_trace_id,
                    exemplar_span_id,
                    value,
                    datetime.utcnow(),
                )

        # Send to collector
        metric = Metric(
            name=name,
            value=self._counters[name][key],
            metric_type=MetricType.COUNTER,
            labels=labels,
            exemplar_trace_id=exemplar_trace_id,
            exemplar_span_id=exemplar_span_id,
        )
        self.collector.collect_metric(metric)

    def dec(
        self,
        name: str,
        value: float = 1.0,
        labels: Dict[str, str] = None,
    ) -> None:
        """Decrement a gauge (convenience method)."""
        self.set(name, self.get_current(name, labels) - value, labels)

    def set(
        self,
        name: str,
        value: float,
        labels: Dict[str, str] = None,
    ) -> None:
        """Set a gauge value."""
        labels = labels or {}
        key = self._labels_key(labels)
        now = datetime.utcnow()

        with self._lock:
            if not self._check_cardinality(name, key):
                return

            self._gauges[name][key] = (value, now)

            # Add to time series
            series_key = f"{name}:{key}"
            self._series_raw[series_key].append((now, value))

        # Send to collector
        metric = Metric(
            name=name,
            value=value,
            metric_type=MetricType.GAUGE,
            labels=labels,
        )
        self.collector.collect_metric(metric)

    def observe(
        self,
        name: str,
        value: float,
        labels: Dict[str, str] = None,
        exemplar_trace_id: str = None,
        exemplar_span_id: str = None,
    ) -> None:
        """Record a histogram observation."""
        labels = labels or {}
        key = self._labels_key(labels)

        with self._lock:
            if not self._check_cardinality(name, key):
                return

            if key not in self._histograms[name]:
                definition = self._definitions.get(name)
                buckets = definition.buckets if definition else None
                self._histograms[name][key] = HistogramBuckets(
                    buckets=buckets or [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, float('inf')]
                )

            self._histograms[name][key].observe(value)

            # Store exemplar for high values
            if exemplar_trace_id and exemplar_span_id:
                self._exemplars[name][key] = (
                    exemplar_trace_id,
                    exemplar_span_id,
                    value,
                    datetime.utcnow(),
                )

        # Send to collector
        metric = Metric(
            name=name,
            value=value,
            metric_type=MetricType.HISTOGRAM,
            labels=labels,
            exemplar_trace_id=exemplar_trace_id,
            exemplar_span_id=exemplar_span_id,
        )
        self.collector.collect_metric(metric)

    def observe_summary(
        self,
        name: str,
        value: float,
        labels: Dict[str, str] = None,
    ) -> None:
        """Record a summary observation."""
        labels = labels or {}
        key = self._labels_key(labels)

        with self._lock:
            if not self._check_cardinality(name, key):
                return

            if key not in self._summaries[name]:
                self._summaries[name][key] = SummaryQuantiles()

            self._summaries[name][key].observe(value)

        metric = Metric(
            name=name,
            value=value,
            metric_type=MetricType.SUMMARY,
            labels=labels,
        )
        self.collector.collect_metric(metric)

    # === Query Methods ===

    def get_current(
        self,
        name: str,
        labels: Dict[str, str] = None,
    ) -> float:
        """Get current value of a metric."""
        labels = labels or {}
        key = self._labels_key(labels)

        with self._lock:
            # Check counters
            if name in self._counters and key in self._counters[name]:
                return self._counters[name][key]

            # Check gauges
            if name in self._gauges and key in self._gauges[name]:
                return self._gauges[name][key][0]

            return 0.0

    def get_all_values(self, name: str) -> Dict[str, float]:
        """Get all current values for a metric (all label combinations)."""
        with self._lock:
            result = {}

            if name in self._counters:
                for key, value in self._counters[name].items():
                    result[key] = value

            if name in self._gauges:
                for key, (value, _) in self._gauges[name].items():
                    result[key] = value

            return result

    def get_histogram_stats(
        self,
        name: str,
        labels: Dict[str, str] = None,
    ) -> Dict[str, float]:
        """Get histogram statistics."""
        labels = labels or {}
        key = self._labels_key(labels)

        with self._lock:
            if name not in self._histograms or key not in self._histograms[name]:
                return {}

            hist = self._histograms[name][key]

            if hist.count == 0:
                return {"count": 0, "sum": 0.0}

            return {
                "count": hist.count,
                "sum": hist.sum,
                "avg": hist.sum / hist.count,
                "p50": hist.get_percentile(0.5),
                "p90": hist.get_percentile(0.9),
                "p95": hist.get_percentile(0.95),
                "p99": hist.get_percentile(0.99),
                "buckets": dict(hist.counts),
            }

    def get_summary_stats(
        self,
        name: str,
        labels: Dict[str, str] = None,
    ) -> Dict[str, float]:
        """Get summary statistics."""
        labels = labels or {}
        key = self._labels_key(labels)

        with self._lock:
            if name not in self._summaries or key not in self._summaries[name]:
                return {}

            summary = self._summaries[name][key]

            if summary.count == 0:
                return {"count": 0, "sum": 0.0}

            return {
                "count": summary.count,
                "sum": summary.sum,
                "avg": summary.sum / summary.count,
                "p50": summary.get_quantile(0.5),
                "p90": summary.get_quantile(0.9),
                "p95": summary.get_quantile(0.95),
                "p99": summary.get_quantile(0.99),
            }

    def get_time_series(
        self,
        name: str,
        labels: Dict[str, str] = None,
        duration: timedelta = None,
        resolution: str = "raw",  # raw, 1m, 1h
    ) -> List[Tuple[datetime, float]]:
        """Get time series data."""
        labels = labels or {}
        key = self._labels_key(labels)
        series_key = f"{name}:{key}"

        with self._lock:
            if resolution == "1h":
                series = self._series_1h.get(series_key, [])
            elif resolution == "1m":
                series = self._series_1m.get(series_key, [])
            else:
                series = self._series_raw.get(series_key, [])

            if duration:
                cutoff = datetime.utcnow() - duration
                series = [(t, v) for t, v in series if t >= cutoff]

            return list(series)

    def get_rate(
        self,
        name: str,
        labels: Dict[str, str] = None,
        duration: timedelta = timedelta(minutes=5),
    ) -> float:
        """Calculate rate of change over duration."""
        series = self.get_time_series(name, labels, duration)

        if len(series) < 2:
            return 0.0

        delta_value = series[-1][1] - series[0][1]
        delta_time = (series[-1][0] - series[0][0]).total_seconds()

        return delta_value / delta_time if delta_time > 0 else 0.0

    # === Aggregation ===

    async def _aggregation_loop(self) -> None:
        """Background aggregation loop."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(self.aggregation_interval)
                await self._aggregate()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Aggregation error: {e}")

    async def _aggregate(self) -> None:
        """Perform metric aggregation (rollups)."""
        now = datetime.utcnow()
        minute_start = now.replace(second=0, microsecond=0)

        with self._lock:
            for series_key, points in self._series_raw.items():
                if not points:
                    continue

                # Get points from last minute
                minute_points = [
                    v for t, v in points
                    if t >= minute_start - timedelta(minutes=1) and t < minute_start
                ]

                if minute_points:
                    # Store 1-minute average
                    avg = statistics.mean(minute_points)
                    self._series_1m[series_key].append((minute_start, avg))

            # Hourly rollup
            hour_start = now.replace(minute=0, second=0, microsecond=0)
            for series_key, points in self._series_1m.items():
                if not points:
                    continue

                hour_points = [
                    v for t, v in points
                    if t >= hour_start - timedelta(hours=1) and t < hour_start
                ]

                if hour_points:
                    avg = statistics.mean(hour_points)
                    self._series_1h[series_key].append((hour_start, avg))

    async def _cleanup_loop(self) -> None:
        """Background cleanup loop for retention."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(3600)  # Run hourly
                await self._cleanup()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup error: {e}")

    async def _cleanup(self) -> None:
        """Clean up old data based on retention policies."""
        now = datetime.utcnow()

        with self._lock:
            # Clean raw data
            raw_cutoff = now - self.retention_raw
            for key in list(self._series_raw.keys()):
                self._series_raw[key] = [
                    (t, v) for t, v in self._series_raw[key] if t >= raw_cutoff
                ]

            # Clean 1m data
            m1_cutoff = now - self.retention_1m
            for key in list(self._series_1m.keys()):
                self._series_1m[key] = [
                    (t, v) for t, v in self._series_1m[key] if t >= m1_cutoff
                ]

            # Clean 1h data
            h1_cutoff = now - self.retention_1h
            for key in list(self._series_1h.keys()):
                self._series_1h[key] = [
                    (t, v) for t, v in self._series_1h[key] if t >= h1_cutoff
                ]

    # === Prometheus Export ===

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus exposition format."""
        lines = []

        with self._lock:
            # Export counters
            for name, values in self._counters.items():
                definition = self._definitions.get(name)
                if definition:
                    lines.append(f"# HELP {name} {definition.description}")
                    lines.append(f"# TYPE {name} counter")

                for labels_key, value in values.items():
                    if labels_key:
                        lines.append(f"{name}{{{labels_key}}} {value}")
                    else:
                        lines.append(f"{name} {value}")

            # Export gauges
            for name, values in self._gauges.items():
                definition = self._definitions.get(name)
                if definition:
                    lines.append(f"# HELP {name} {definition.description}")
                    lines.append(f"# TYPE {name} gauge")

                for labels_key, (value, _) in values.items():
                    if labels_key:
                        lines.append(f"{name}{{{labels_key}}} {value}")
                    else:
                        lines.append(f"{name} {value}")

            # Export histograms
            for name, buckets_dict in self._histograms.items():
                definition = self._definitions.get(name)
                if definition:
                    lines.append(f"# HELP {name} {definition.description}")
                    lines.append(f"# TYPE {name} histogram")

                for labels_key, hist in buckets_dict.items():
                    base_labels = labels_key
                    for bucket, count in sorted(hist.counts.items()):
                        if bucket == float('inf'):
                            bucket_label = '+Inf'
                        else:
                            bucket_label = str(bucket)
                        if base_labels:
                            lines.append(f'{name}_bucket{{{base_labels},le="{bucket_label}"}} {count}')
                        else:
                            lines.append(f'{name}_bucket{{le="{bucket_label}"}} {count}')

                    if labels_key:
                        lines.append(f"{name}_sum{{{labels_key}}} {hist.sum}")
                        lines.append(f"{name}_count{{{labels_key}}} {hist.count}")
                    else:
                        lines.append(f"{name}_sum {hist.sum}")
                        lines.append(f"{name}_count {hist.count}")

        return "\n".join(lines)

    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all current metrics as a dictionary."""
        result = {}

        with self._lock:
            for name in self._definitions:
                metric_data = {
                    "definition": {
                        "name": self._definitions[name].name,
                        "type": self._definitions[name].metric_type.value,
                        "description": self._definitions[name].description,
                        "unit": self._definitions[name].unit,
                    },
                    "values": {},
                }

                # Get values based on type
                if name in self._counters:
                    metric_data["values"] = dict(self._counters[name])
                elif name in self._gauges:
                    metric_data["values"] = {k: v[0] for k, v in self._gauges[name].items()}

                if name in self._histograms:
                    metric_data["histogram_stats"] = {
                        key: {
                            "count": hist.count,
                            "sum": hist.sum,
                            "avg": hist.sum / hist.count if hist.count > 0 else 0,
                        }
                        for key, hist in self._histograms[name].items()
                    }

                result[name] = metric_data

        return result

    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        with self._lock:
            return {
                "registered_metrics": len(self._definitions),
                "counter_series": sum(len(v) for v in self._counters.values()),
                "gauge_series": sum(len(v) for v in self._gauges.values()),
                "histogram_series": sum(len(v) for v in self._histograms.values()),
                "summary_series": sum(len(v) for v in self._summaries.values()),
                "total_cardinality": sum(len(v) for v in self._label_cardinality.values()),
                "time_series_raw": len(self._series_raw),
                "time_series_1m": len(self._series_1m),
                "time_series_1h": len(self._series_1h),
            }
