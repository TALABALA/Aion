"""
Metrics Collection

Prometheus-compatible metrics for agent monitoring.
"""

import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional
import structlog

logger = structlog.get_logger()


class MetricType(str, Enum):
    """Type of metric."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricValue:
    """A metric data point."""

    value: float
    timestamp: datetime = field(default_factory=datetime.now)
    labels: dict[str, str] = field(default_factory=dict)


class Metric:
    """Base class for metrics."""

    def __init__(
        self,
        name: str,
        description: str,
        labels: Optional[list[str]] = None,
    ):
        self.name = name
        self.description = description
        self.label_names = labels or []
        self._values: dict[tuple, MetricValue] = {}

    def _label_key(self, labels: dict[str, str]) -> tuple:
        """Create a hashable key from labels."""
        return tuple(sorted(labels.items()))

    def get_value(self, labels: Optional[dict[str, str]] = None) -> float:
        """Get current value for labels."""
        key = self._label_key(labels or {})
        return self._values.get(key, MetricValue(0.0)).value

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "type": self.metric_type.value,
            "values": [
                {
                    "labels": dict(key),
                    "value": mv.value,
                    "timestamp": mv.timestamp.isoformat(),
                }
                for key, mv in self._values.items()
            ],
        }

    @property
    def metric_type(self) -> MetricType:
        """Get metric type."""
        raise NotImplementedError


class Counter(Metric):
    """
    Counter metric - monotonically increasing value.

    Used for: requests, errors, completions.
    """

    @property
    def metric_type(self) -> MetricType:
        return MetricType.COUNTER

    def inc(self, value: float = 1.0, labels: Optional[dict[str, str]] = None) -> None:
        """Increment counter."""
        if value < 0:
            raise ValueError("Counter can only increase")

        key = self._label_key(labels or {})
        current = self._values.get(key, MetricValue(0.0))
        self._values[key] = MetricValue(
            value=current.value + value,
            labels=labels or {},
        )


class Gauge(Metric):
    """
    Gauge metric - value that can go up and down.

    Used for: queue size, memory usage, active connections.
    """

    @property
    def metric_type(self) -> MetricType:
        return MetricType.GAUGE

    def set(self, value: float, labels: Optional[dict[str, str]] = None) -> None:
        """Set gauge value."""
        key = self._label_key(labels or {})
        self._values[key] = MetricValue(value=value, labels=labels or {})

    def inc(self, value: float = 1.0, labels: Optional[dict[str, str]] = None) -> None:
        """Increment gauge."""
        key = self._label_key(labels or {})
        current = self._values.get(key, MetricValue(0.0))
        self._values[key] = MetricValue(
            value=current.value + value,
            labels=labels or {},
        )

    def dec(self, value: float = 1.0, labels: Optional[dict[str, str]] = None) -> None:
        """Decrement gauge."""
        self.inc(-value, labels)


class Histogram(Metric):
    """
    Histogram metric - distribution of values.

    Used for: request durations, response sizes.
    """

    DEFAULT_BUCKETS = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10]

    def __init__(
        self,
        name: str,
        description: str,
        labels: Optional[list[str]] = None,
        buckets: Optional[list[float]] = None,
    ):
        super().__init__(name, description, labels)
        self.buckets = sorted(buckets or self.DEFAULT_BUCKETS)
        self._observations: dict[tuple, list[float]] = {}
        self._bucket_counts: dict[tuple, dict[float, int]] = {}

    @property
    def metric_type(self) -> MetricType:
        return MetricType.HISTOGRAM

    def observe(self, value: float, labels: Optional[dict[str, str]] = None) -> None:
        """Record an observation."""
        key = self._label_key(labels or {})

        # Store observation
        if key not in self._observations:
            self._observations[key] = []
            self._bucket_counts[key] = {b: 0 for b in self.buckets}

        self._observations[key].append(value)

        # Update buckets
        for bucket in self.buckets:
            if value <= bucket:
                self._bucket_counts[key][bucket] += 1

    def get_bucket_counts(
        self,
        labels: Optional[dict[str, str]] = None,
    ) -> dict[float, int]:
        """Get bucket counts."""
        key = self._label_key(labels or {})
        return self._bucket_counts.get(key, {})

    def get_percentile(
        self,
        percentile: float,
        labels: Optional[dict[str, str]] = None,
    ) -> float:
        """Get percentile value."""
        key = self._label_key(labels or {})
        observations = self._observations.get(key, [])

        if not observations:
            return 0.0

        sorted_obs = sorted(observations)
        idx = int(len(sorted_obs) * percentile / 100)
        return sorted_obs[min(idx, len(sorted_obs) - 1)]

    def get_stats(
        self,
        labels: Optional[dict[str, str]] = None,
    ) -> dict[str, float]:
        """Get histogram statistics."""
        key = self._label_key(labels or {})
        observations = self._observations.get(key, [])

        if not observations:
            return {"count": 0, "sum": 0, "avg": 0, "min": 0, "max": 0}

        return {
            "count": len(observations),
            "sum": sum(observations),
            "avg": sum(observations) / len(observations),
            "min": min(observations),
            "max": max(observations),
            "p50": self.get_percentile(50, labels),
            "p95": self.get_percentile(95, labels),
            "p99": self.get_percentile(99, labels),
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = super().to_dict()
        result["buckets"] = self.buckets
        result["stats"] = {
            str(dict(key)): self.get_stats(dict(key))
            for key in self._observations.keys()
        }
        return result


class MetricsCollector:
    """
    Collects metrics for an agent.

    Features:
    - Standard agent metrics
    - Custom metric registration
    - Periodic collection
    """

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self._metrics: dict[str, Metric] = {}
        self._collectors: list[Callable[[], dict[str, float]]] = []

        # Register standard metrics
        self._register_standard_metrics()

    def _register_standard_metrics(self) -> None:
        """Register standard agent metrics."""
        # Task metrics
        self.register(Counter(
            "agent_tasks_total",
            "Total tasks processed",
            ["status"],
        ))
        self.register(Histogram(
            "agent_task_duration_seconds",
            "Task duration in seconds",
            ["type"],
        ))

        # Message metrics
        self.register(Counter(
            "agent_messages_total",
            "Total messages sent/received",
            ["direction", "type"],
        ))

        # Tool metrics
        self.register(Counter(
            "agent_tool_calls_total",
            "Total tool calls",
            ["tool", "status"],
        ))
        self.register(Histogram(
            "agent_tool_duration_seconds",
            "Tool execution duration",
            ["tool"],
        ))

        # Memory metrics
        self.register(Gauge(
            "agent_memory_items",
            "Number of items in memory",
            ["type"],
        ))

        # Learning metrics
        self.register(Gauge(
            "agent_learning_rate",
            "Current learning rate",
        ))
        self.register(Counter(
            "agent_adaptations_total",
            "Total strategy adaptations",
        ))

    def register(self, metric: Metric) -> None:
        """Register a metric."""
        self._metrics[metric.name] = metric

    def get(self, name: str) -> Optional[Metric]:
        """Get a metric by name."""
        return self._metrics.get(name)

    def counter(self, name: str) -> Optional[Counter]:
        """Get a counter metric."""
        metric = self._metrics.get(name)
        return metric if isinstance(metric, Counter) else None

    def gauge(self, name: str) -> Optional[Gauge]:
        """Get a gauge metric."""
        metric = self._metrics.get(name)
        return metric if isinstance(metric, Gauge) else None

    def histogram(self, name: str) -> Optional[Histogram]:
        """Get a histogram metric."""
        metric = self._metrics.get(name)
        return metric if isinstance(metric, Histogram) else None

    def add_collector(self, collector: Callable[[], dict[str, float]]) -> None:
        """Add a custom collector function."""
        self._collectors.append(collector)

    def collect(self) -> dict[str, Any]:
        """Collect all metrics."""
        # Run custom collectors
        for collector in self._collectors:
            try:
                values = collector()
                for name, value in values.items():
                    metric = self._metrics.get(name)
                    if metric and isinstance(metric, Gauge):
                        metric.set(value)
            except Exception as e:
                logger.warning("collector_error", error=str(e))

        return {
            "agent_id": self.agent_id,
            "timestamp": datetime.now().isoformat(),
            "metrics": {
                name: metric.to_dict()
                for name, metric in self._metrics.items()
            },
        }


class MetricsRegistry:
    """
    Central registry for all agent metrics.

    Features:
    - Multi-agent metrics aggregation
    - Query and filtering
    - Export to various formats
    """

    def __init__(self):
        self._collectors: dict[str, MetricsCollector] = {}
        self._global_metrics: dict[str, Metric] = {}
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize registry."""
        # Register global metrics
        self._global_metrics["active_agents"] = Gauge(
            "active_agents",
            "Number of active agents",
        )
        self._global_metrics["total_tasks"] = Counter(
            "total_tasks",
            "Total tasks across all agents",
            ["status"],
        )

        self._initialized = True
        logger.info("metrics_registry_initialized")

    async def shutdown(self) -> None:
        """Shutdown registry."""
        self._initialized = False
        logger.info("metrics_registry_shutdown")

    def get_collector(self, agent_id: str) -> MetricsCollector:
        """Get or create collector for an agent."""
        if agent_id not in self._collectors:
            self._collectors[agent_id] = MetricsCollector(agent_id)

            # Update global metrics
            active = self._global_metrics.get("active_agents")
            if isinstance(active, Gauge):
                active.set(len(self._collectors))

        return self._collectors[agent_id]

    def remove_collector(self, agent_id: str) -> bool:
        """Remove a collector."""
        if agent_id in self._collectors:
            del self._collectors[agent_id]

            active = self._global_metrics.get("active_agents")
            if isinstance(active, Gauge):
                active.set(len(self._collectors))

            return True
        return False

    def collect_all(self) -> dict[str, Any]:
        """Collect metrics from all agents."""
        return {
            "timestamp": datetime.now().isoformat(),
            "global": {
                name: metric.to_dict()
                for name, metric in self._global_metrics.items()
            },
            "agents": {
                agent_id: collector.collect()
                for agent_id, collector in self._collectors.items()
            },
        }

    def query(
        self,
        metric_name: str,
        agent_id: Optional[str] = None,
        labels: Optional[dict[str, str]] = None,
    ) -> list[dict[str, Any]]:
        """Query metrics by name."""
        results = []

        # Check global metrics
        if metric_name in self._global_metrics:
            metric = self._global_metrics[metric_name]
            results.append({
                "source": "global",
                "metric": metric.to_dict(),
            })

        # Check agent metrics
        collectors = self._collectors.items()
        if agent_id:
            collectors = [(agent_id, self._collectors.get(agent_id))]

        for aid, collector in collectors:
            if collector:
                metric = collector.get(metric_name)
                if metric:
                    results.append({
                        "source": aid,
                        "metric": metric.to_dict(),
                    })

        return results

    def get_summary(self) -> dict[str, Any]:
        """Get metrics summary."""
        all_metrics = self.collect_all()

        # Aggregate key metrics
        total_tasks = 0
        total_errors = 0

        for agent_data in all_metrics["agents"].values():
            metrics = agent_data.get("metrics", {})

            tasks = metrics.get("agent_tasks_total", {})
            for value_data in tasks.get("values", []):
                total_tasks += value_data.get("value", 0)
                if value_data.get("labels", {}).get("status") == "error":
                    total_errors += value_data.get("value", 0)

        return {
            "active_agents": len(self._collectors),
            "total_tasks": total_tasks,
            "error_rate": total_errors / max(1, total_tasks),
            "metrics_count": sum(
                len(c._metrics) for c in self._collectors.values()
            ) + len(self._global_metrics),
        }

    def to_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []

        # Export global metrics
        for name, metric in self._global_metrics.items():
            lines.append(f"# HELP {name} {metric.description}")
            lines.append(f"# TYPE {name} {metric.metric_type.value}")

            for key, mv in metric._values.items():
                label_str = ",".join(f'{k}="{v}"' for k, v in key)
                if label_str:
                    lines.append(f"{name}{{{label_str}}} {mv.value}")
                else:
                    lines.append(f"{name} {mv.value}")

        # Export agent metrics
        for agent_id, collector in self._collectors.items():
            for name, metric in collector._metrics.items():
                lines.append(f"# HELP {name} {metric.description}")
                lines.append(f"# TYPE {name} {metric.metric_type.value}")

                for key, mv in metric._values.items():
                    labels = dict(key)
                    labels["agent_id"] = agent_id
                    label_str = ",".join(f'{k}="{v}"' for k, v in labels.items())
                    lines.append(f"{name}{{{label_str}}} {mv.value}")

        return "\n".join(lines)
