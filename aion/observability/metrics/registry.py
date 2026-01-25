"""
AION Metrics Registry

High-level metric types with Prometheus-compatible API.
Provides Counter, Gauge, Histogram, and Summary classes.
"""

from __future__ import annotations

import time
import threading
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, Generator, List, Optional, TypeVar

import structlog

from aion.observability.types import MetricType, MetricDefinition

logger = structlog.get_logger(__name__)

T = TypeVar('T')


class MetricBase(ABC):
    """Base class for all metric types."""

    def __init__(
        self,
        name: str,
        description: str = "",
        labels: List[str] = None,
        unit: str = "",
        registry: "MetricRegistry" = None,
    ):
        self.name = name
        self.description = description
        self.label_names = labels or []
        self.unit = unit
        self._registry = registry

        # Validate name
        if not name:
            raise ValueError("Metric name cannot be empty")

    @property
    @abstractmethod
    def metric_type(self) -> MetricType:
        """Get the metric type."""
        pass

    def labels(self, **label_values) -> "LabeledMetric":
        """Create a labeled metric instance."""
        # Validate labels
        provided = set(label_values.keys())
        expected = set(self.label_names)

        if provided != expected:
            missing = expected - provided
            extra = provided - expected
            raise ValueError(
                f"Label mismatch for {self.name}. "
                f"Missing: {missing}, Extra: {extra}"
            )

        return LabeledMetric(self, label_values)

    def _get_engine(self):
        """Get the metrics engine."""
        if self._registry:
            return self._registry._engine
        # Lazy import to avoid circular dependency
        from aion.observability import get_metrics_engine
        return get_metrics_engine()


class LabeledMetric:
    """A metric instance with specific label values."""

    def __init__(self, metric: MetricBase, label_values: Dict[str, str]):
        self._metric = metric
        self._label_values = label_values

    def __getattr__(self, name: str):
        """Forward method calls to the parent metric."""
        method = getattr(self._metric, name, None)
        if callable(method):
            def wrapper(*args, **kwargs):
                return method(*args, labels=self._label_values, **kwargs)
            return wrapper
        raise AttributeError(f"'{type(self._metric).__name__}' has no attribute '{name}'")


class Counter(MetricBase):
    """
    A counter metric that only goes up.

    Usage:
        requests_total = Counter(
            'requests_total',
            'Total requests',
            labels=['method', 'endpoint']
        )
        requests_total.labels(method='GET', endpoint='/api').inc()
    """

    @property
    def metric_type(self) -> MetricType:
        return MetricType.COUNTER

    def inc(self, value: float = 1.0, labels: Dict[str, str] = None) -> None:
        """Increment the counter."""
        if value < 0:
            raise ValueError("Counter can only be incremented")

        engine = self._get_engine()
        if engine:
            engine.inc(self.name, value, labels or {})

    def count_exceptions(self, exception: type = Exception):
        """Decorator to count exceptions."""
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except exception:
                    self.inc()
                    raise
            return wrapper
        return decorator


class Gauge(MetricBase):
    """
    A gauge metric that can go up and down.

    Usage:
        active_requests = Gauge(
            'active_requests',
            'Currently active requests'
        )
        active_requests.inc()
        active_requests.dec()
        active_requests.set(42)
    """

    @property
    def metric_type(self) -> MetricType:
        return MetricType.GAUGE

    def inc(self, value: float = 1.0, labels: Dict[str, str] = None) -> None:
        """Increment the gauge."""
        engine = self._get_engine()
        if engine:
            current = engine.get_current(self.name, labels)
            engine.set(self.name, current + value, labels or {})

    def dec(self, value: float = 1.0, labels: Dict[str, str] = None) -> None:
        """Decrement the gauge."""
        engine = self._get_engine()
        if engine:
            current = engine.get_current(self.name, labels)
            engine.set(self.name, current - value, labels or {})

    def set(self, value: float, labels: Dict[str, str] = None) -> None:
        """Set the gauge to a value."""
        engine = self._get_engine()
        if engine:
            engine.set(self.name, value, labels or {})

    def set_to_current_time(self, labels: Dict[str, str] = None) -> None:
        """Set gauge to current Unix timestamp."""
        self.set(time.time(), labels)

    @contextmanager
    def track_inprogress(self, labels: Dict[str, str] = None) -> Generator[None, None, None]:
        """Context manager to track in-progress operations."""
        self.inc(labels=labels)
        try:
            yield
        finally:
            self.dec(labels=labels)

    def time(self, labels: Dict[str, str] = None):
        """Decorator to time a function and set gauge to last duration."""
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            def wrapper(*args, **kwargs):
                start = time.perf_counter()
                try:
                    return func(*args, **kwargs)
                finally:
                    duration = time.perf_counter() - start
                    self.set(duration, labels)
            return wrapper
        return decorator


class Histogram(MetricBase):
    """
    A histogram metric for measuring distributions.

    Usage:
        request_duration = Histogram(
            'request_duration_seconds',
            'Request duration',
            labels=['method'],
            buckets=[.005, .01, .025, .05, .1, .25, .5, 1, 2.5, 5, 10]
        )
        request_duration.labels(method='GET').observe(0.42)
    """

    DEFAULT_BUCKETS = (
        .005, .01, .025, .05, .1, .25, .5, 1, 2.5, 5, 10, float('inf')
    )

    def __init__(
        self,
        name: str,
        description: str = "",
        labels: List[str] = None,
        unit: str = "",
        buckets: tuple = None,
        registry: "MetricRegistry" = None,
    ):
        super().__init__(name, description, labels, unit, registry)
        self.buckets = buckets or self.DEFAULT_BUCKETS

    @property
    def metric_type(self) -> MetricType:
        return MetricType.HISTOGRAM

    def observe(self, value: float, labels: Dict[str, str] = None) -> None:
        """Record an observation."""
        engine = self._get_engine()
        if engine:
            engine.observe(self.name, value, labels or {})

    @contextmanager
    def time(self, labels: Dict[str, str] = None) -> Generator[None, None, None]:
        """Context manager to time an operation."""
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            self.observe(duration, labels)

    def time_decorator(self, labels: Dict[str, str] = None):
        """Decorator to time a function."""
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            def wrapper(*args, **kwargs):
                with self.time(labels):
                    return func(*args, **kwargs)
            return wrapper
        return decorator


class Summary(MetricBase):
    """
    A summary metric for calculating quantiles.

    Usage:
        response_size = Summary(
            'response_size_bytes',
            'Response size in bytes'
        )
        response_size.observe(1024)
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        labels: List[str] = None,
        unit: str = "",
        max_age_seconds: float = 600,
        age_buckets: int = 5,
        registry: "MetricRegistry" = None,
    ):
        super().__init__(name, description, labels, unit, registry)
        self.max_age_seconds = max_age_seconds
        self.age_buckets = age_buckets

    @property
    def metric_type(self) -> MetricType:
        return MetricType.SUMMARY

    def observe(self, value: float, labels: Dict[str, str] = None) -> None:
        """Record an observation."""
        engine = self._get_engine()
        if engine:
            engine.observe_summary(self.name, value, labels or {})

    @contextmanager
    def time(self, labels: Dict[str, str] = None) -> Generator[None, None, None]:
        """Context manager to time an operation."""
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            self.observe(duration, labels)


class Info(MetricBase):
    """
    An info metric for static information.

    Usage:
        build_info = Info(
            'build_info',
            'Build information'
        )
        build_info.info({'version': '1.0.0', 'commit': 'abc123'})
    """

    @property
    def metric_type(self) -> MetricType:
        return MetricType.INFO

    def info(self, labels: Dict[str, str]) -> None:
        """Set info metric labels."""
        engine = self._get_engine()
        if engine:
            engine.set(self.name, 1.0, labels)


class MetricRegistry:
    """
    Registry for managing metrics.

    Usage:
        registry = MetricRegistry()
        counter = registry.counter('requests_total', 'Total requests')
        gauge = registry.gauge('active_requests', 'Active requests')
    """

    def __init__(self, engine=None):
        self._engine = engine
        self._metrics: Dict[str, MetricBase] = {}
        self._lock = threading.Lock()

    def set_engine(self, engine) -> None:
        """Set the metrics engine."""
        self._engine = engine

    def counter(
        self,
        name: str,
        description: str = "",
        labels: List[str] = None,
        unit: str = "",
    ) -> Counter:
        """Create or get a counter."""
        with self._lock:
            if name in self._metrics:
                metric = self._metrics[name]
                if not isinstance(metric, Counter):
                    raise ValueError(f"Metric {name} already exists as {type(metric).__name__}")
                return metric

            counter = Counter(name, description, labels, unit, self)
            self._metrics[name] = counter
            self._register_definition(counter)
            return counter

    def gauge(
        self,
        name: str,
        description: str = "",
        labels: List[str] = None,
        unit: str = "",
    ) -> Gauge:
        """Create or get a gauge."""
        with self._lock:
            if name in self._metrics:
                metric = self._metrics[name]
                if not isinstance(metric, Gauge):
                    raise ValueError(f"Metric {name} already exists as {type(metric).__name__}")
                return metric

            gauge = Gauge(name, description, labels, unit, self)
            self._metrics[name] = gauge
            self._register_definition(gauge)
            return gauge

    def histogram(
        self,
        name: str,
        description: str = "",
        labels: List[str] = None,
        unit: str = "",
        buckets: tuple = None,
    ) -> Histogram:
        """Create or get a histogram."""
        with self._lock:
            if name in self._metrics:
                metric = self._metrics[name]
                if not isinstance(metric, Histogram):
                    raise ValueError(f"Metric {name} already exists as {type(metric).__name__}")
                return metric

            histogram = Histogram(name, description, labels, unit, buckets, self)
            self._metrics[name] = histogram
            self._register_definition(histogram)
            return histogram

    def summary(
        self,
        name: str,
        description: str = "",
        labels: List[str] = None,
        unit: str = "",
    ) -> Summary:
        """Create or get a summary."""
        with self._lock:
            if name in self._metrics:
                metric = self._metrics[name]
                if not isinstance(metric, Summary):
                    raise ValueError(f"Metric {name} already exists as {type(metric).__name__}")
                return metric

            summary = Summary(name, description, labels, unit, registry=self)
            self._metrics[name] = summary
            self._register_definition(summary)
            return summary

    def info(
        self,
        name: str,
        description: str = "",
    ) -> Info:
        """Create or get an info metric."""
        with self._lock:
            if name in self._metrics:
                metric = self._metrics[name]
                if not isinstance(metric, Info):
                    raise ValueError(f"Metric {name} already exists as {type(metric).__name__}")
                return metric

            info = Info(name, description, registry=self)
            self._metrics[name] = info
            self._register_definition(info)
            return info

    def _register_definition(self, metric: MetricBase) -> None:
        """Register metric definition with engine."""
        if self._engine:
            definition = MetricDefinition(
                name=metric.name,
                metric_type=metric.metric_type,
                description=metric.description,
                unit=metric.unit,
                labels=metric.label_names,
            )
            self._engine.register_metric(definition)

    def get_metric(self, name: str) -> Optional[MetricBase]:
        """Get a metric by name."""
        return self._metrics.get(name)

    def get_all_metrics(self) -> Dict[str, MetricBase]:
        """Get all registered metrics."""
        return dict(self._metrics)

    def unregister(self, name: str) -> bool:
        """Unregister a metric."""
        with self._lock:
            if name in self._metrics:
                del self._metrics[name]
                return True
            return False


# Global registry
_default_registry: Optional[MetricRegistry] = None


def get_default_registry() -> MetricRegistry:
    """Get the default metric registry."""
    global _default_registry
    if _default_registry is None:
        _default_registry = MetricRegistry()
    return _default_registry


def set_default_registry(registry: MetricRegistry) -> None:
    """Set the default metric registry."""
    global _default_registry
    _default_registry = registry


# === Pre-defined AION Metrics ===

# Request metrics
REQUESTS_TOTAL = Counter(
    "aion_requests_total",
    "Total number of requests",
    labels=["method", "endpoint", "status"],
)

REQUEST_DURATION = Histogram(
    "aion_request_duration_seconds",
    "Request duration in seconds",
    labels=["method", "endpoint"],
    unit="seconds",
)

ACTIVE_REQUESTS = Gauge(
    "aion_active_requests",
    "Number of active requests",
)

# Token metrics
TOKENS_USED = Counter(
    "aion_tokens_used_total",
    "Total tokens used",
    labels=["model", "type"],
)

TOKEN_COST = Counter(
    "aion_tokens_cost_dollars",
    "Total token cost in dollars",
    labels=["model"],
    unit="dollars",
)

# LLM metrics
LLM_LATENCY = Histogram(
    "aion_llm_latency_seconds",
    "LLM API latency",
    labels=["model", "operation"],
    unit="seconds",
)

LLM_REQUESTS = Counter(
    "aion_llm_requests_total",
    "Total LLM requests",
    labels=["model", "status"],
)

# Tool metrics
TOOL_CALLS = Counter(
    "aion_tool_calls_total",
    "Total tool calls",
    labels=["tool", "status"],
)

TOOL_DURATION = Histogram(
    "aion_tool_duration_seconds",
    "Tool execution duration",
    labels=["tool"],
    unit="seconds",
)

# Agent metrics
ACTIVE_AGENTS = Gauge(
    "aion_agents_active",
    "Number of active agents",
)

AGENT_TASKS = Counter(
    "aion_agent_tasks_total",
    "Total agent tasks",
    labels=["agent_type", "status"],
)

# Goal metrics
ACTIVE_GOALS = Gauge(
    "aion_goals_active",
    "Number of active goals",
)

GOALS_TOTAL = Counter(
    "aion_goals_total",
    "Total goals created",
    labels=["status"],
)

# Memory metrics
MEMORY_OPERATIONS = Counter(
    "aion_memory_operations_total",
    "Memory operations",
    labels=["operation"],
)

MEMORY_SEARCH_DURATION = Histogram(
    "aion_memory_search_duration_seconds",
    "Memory search duration",
    unit="seconds",
)

# Error metrics
ERRORS = Counter(
    "aion_errors_total",
    "Total errors",
    labels=["component", "error_type"],
)

# Knowledge graph metrics
KG_QUERIES = Counter(
    "aion_knowledge_queries_total",
    "Knowledge graph queries",
    labels=["query_type"],
)

KG_ENTITIES = Gauge(
    "aion_knowledge_entities_total",
    "Total knowledge graph entities",
)

# Tracing metrics
SPANS_TOTAL = Counter(
    "aion_spans_total",
    "Total spans created",
    labels=["service", "operation"],
)

SPAN_DURATION = Histogram(
    "aion_span_duration_seconds",
    "Span duration",
    labels=["service", "operation"],
    unit="seconds",
)
