"""
AION Telemetry Collector

Central ingestion point for all telemetry data with SOTA features:
- Buffered collection with async flush
- Backpressure handling
- Sampling and filtering
- Multiple backend support
- Batch export with retry
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set, TypeVar
import threading
import weakref

import structlog

from aion.observability.types import (
    Metric, Span, LogEntry, CostRecord, Alert, AuditEvent, Anomaly,
    MetricType, LogLevel,
)

logger = structlog.get_logger(__name__)

T = TypeVar('T')


class ExporterType(str, Enum):
    """Types of telemetry exporters."""
    METRICS = "metrics"
    SPANS = "spans"
    LOGS = "logs"
    COSTS = "costs"
    ALERTS = "alerts"
    AUDITS = "audits"


@dataclass
class ExporterConfig:
    """Configuration for an exporter."""
    name: str
    exporter_type: ExporterType
    exporter_fn: Callable
    batch_size: int = 100
    timeout_ms: int = 30000
    retry_count: int = 3
    retry_delay_ms: int = 1000
    enabled: bool = True


@dataclass
class CollectorStats:
    """Statistics for the collector."""
    metrics_collected: int = 0
    spans_collected: int = 0
    logs_collected: int = 0
    costs_collected: int = 0
    alerts_collected: int = 0
    audits_collected: int = 0

    metrics_dropped: int = 0
    spans_dropped: int = 0
    logs_dropped: int = 0

    flush_count: int = 0
    export_errors: int = 0
    export_success: int = 0

    last_flush_time: Optional[datetime] = None
    last_export_duration_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "metrics_collected": self.metrics_collected,
            "spans_collected": self.spans_collected,
            "logs_collected": self.logs_collected,
            "costs_collected": self.costs_collected,
            "alerts_collected": self.alerts_collected,
            "audits_collected": self.audits_collected,
            "metrics_dropped": self.metrics_dropped,
            "spans_dropped": self.spans_dropped,
            "logs_dropped": self.logs_dropped,
            "flush_count": self.flush_count,
            "export_errors": self.export_errors,
            "export_success": self.export_success,
            "last_flush_time": self.last_flush_time.isoformat() if self.last_flush_time else None,
            "last_export_duration_ms": self.last_export_duration_ms,
        }


class RingBuffer:
    """Thread-safe ring buffer with overflow handling."""

    def __init__(self, maxsize: int = 10000):
        self._buffer: deque = deque(maxlen=maxsize)
        self._lock = threading.Lock()
        self._dropped = 0

    def put(self, item: Any) -> bool:
        """Add item to buffer. Returns True if added, False if dropped."""
        with self._lock:
            if len(self._buffer) >= self._buffer.maxlen:
                self._dropped += 1
                return False
            self._buffer.append(item)
            return True

    def get_all(self) -> List[Any]:
        """Get and clear all items from buffer."""
        with self._lock:
            items = list(self._buffer)
            self._buffer.clear()
            return items

    def get_batch(self, batch_size: int) -> List[Any]:
        """Get a batch of items without clearing."""
        with self._lock:
            return list(self._buffer)[:batch_size]

    def remove_batch(self, count: int) -> None:
        """Remove items from front of buffer."""
        with self._lock:
            for _ in range(min(count, len(self._buffer))):
                self._buffer.popleft()

    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)

    @property
    def dropped_count(self) -> int:
        return self._dropped


class TelemetryCollector:
    """
    Central collector for all telemetry data.

    Features:
    - Buffered collection with async flush
    - Multiple backend support
    - Sampling and filtering
    - Backpressure handling
    - Graceful shutdown with drain
    """

    def __init__(
        self,
        buffer_size: int = 10000,
        flush_interval: float = 5.0,
        max_export_batch_size: int = 500,
        shutdown_timeout: float = 30.0,
    ):
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        self.max_export_batch_size = max_export_batch_size
        self.shutdown_timeout = shutdown_timeout

        # Buffers for each telemetry type
        self._metrics_buffer = RingBuffer(buffer_size)
        self._spans_buffer = RingBuffer(buffer_size)
        self._logs_buffer = RingBuffer(buffer_size)
        self._costs_buffer = RingBuffer(buffer_size // 10)
        self._alerts_buffer = RingBuffer(buffer_size // 10)
        self._audits_buffer = RingBuffer(buffer_size // 10)

        # Exporters
        self._exporters: Dict[ExporterType, List[ExporterConfig]] = {
            ExporterType.METRICS: [],
            ExporterType.SPANS: [],
            ExporterType.LOGS: [],
            ExporterType.COSTS: [],
            ExporterType.ALERTS: [],
            ExporterType.AUDITS: [],
        }

        # Processors (transformations before export)
        self._metric_processors: List[Callable[[Metric], Optional[Metric]]] = []
        self._span_processors: List[Callable[[Span], Optional[Span]]] = []
        self._log_processors: List[Callable[[LogEntry], Optional[LogEntry]]] = []

        # Filters
        self._metric_filters: List[Callable[[Metric], bool]] = []
        self._span_filters: List[Callable[[Span], bool]] = []
        self._log_filters: List[Callable[[LogEntry], bool]] = []

        # Sampling
        self._metric_sample_rate: float = 1.0
        self._span_sample_rate: float = 1.0
        self._log_sample_rate: float = 1.0

        # Background task
        self._flush_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        self._flush_lock = asyncio.Lock()

        # Statistics
        self._stats = CollectorStats()

        # Weak reference to avoid circular dependencies
        self._weak_refs: Set[weakref.ref] = set()

        self._initialized = False
        self._shutting_down = False

    async def initialize(self) -> None:
        """Initialize the collector and start background tasks."""
        if self._initialized:
            return

        logger.info(
            "Initializing Telemetry Collector",
            buffer_size=self.buffer_size,
            flush_interval=self.flush_interval,
        )

        # Start flush loop
        self._flush_task = asyncio.create_task(self._flush_loop())

        self._initialized = True

    async def shutdown(self, drain: bool = True) -> None:
        """Shutdown and optionally drain remaining data."""
        if not self._initialized:
            return

        logger.info("Shutting down Telemetry Collector", drain=drain)
        self._shutting_down = True
        self._shutdown_event.set()

        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass

        # Drain remaining data
        if drain:
            logger.info("Draining remaining telemetry data")
            try:
                await asyncio.wait_for(
                    self._flush_all(),
                    timeout=self.shutdown_timeout,
                )
            except asyncio.TimeoutError:
                logger.warning("Shutdown drain timed out")

        self._initialized = False
        logger.info(
            "Telemetry Collector shutdown complete",
            stats=self._stats.to_dict(),
        )

    # === Collection Methods ===

    def collect_metric(self, metric: Metric) -> bool:
        """Collect a metric. Returns True if accepted."""
        if self._shutting_down:
            return False

        # Apply sampling
        if not self._should_sample(self._metric_sample_rate):
            return False

        # Apply filters
        for filter_fn in self._metric_filters:
            if not filter_fn(metric):
                return False

        # Apply processors
        processed = metric
        for processor in self._metric_processors:
            processed = processor(processed)
            if processed is None:
                return False

        if self._metrics_buffer.put(processed):
            self._stats.metrics_collected += 1
            return True
        else:
            self._stats.metrics_dropped += 1
            return False

    def collect_span(self, span: Span) -> bool:
        """Collect a span. Returns True if accepted."""
        if self._shutting_down:
            return False

        # Apply sampling (spans usually have their own sampling decision)
        if not self._should_sample(self._span_sample_rate):
            return False

        # Apply filters
        for filter_fn in self._span_filters:
            if not filter_fn(span):
                return False

        # Apply processors
        processed = span
        for processor in self._span_processors:
            processed = processor(processed)
            if processed is None:
                return False

        if self._spans_buffer.put(processed):
            self._stats.spans_collected += 1
            return True
        else:
            self._stats.spans_dropped += 1
            return False

    def collect_log(self, entry: LogEntry) -> bool:
        """Collect a log entry. Returns True if accepted."""
        if self._shutting_down:
            return False

        # Apply sampling (usually only for debug/trace)
        if entry.level in (LogLevel.DEBUG, LogLevel.TRACE):
            if not self._should_sample(self._log_sample_rate):
                return False

        # Apply filters
        for filter_fn in self._log_filters:
            if not filter_fn(entry):
                return False

        # Apply processors
        processed = entry
        for processor in self._log_processors:
            processed = processor(processed)
            if processed is None:
                return False

        if self._logs_buffer.put(processed):
            self._stats.logs_collected += 1
            return True
        else:
            self._stats.logs_dropped += 1
            return False

    def collect_cost(self, cost: CostRecord) -> bool:
        """Collect a cost record."""
        if self._shutting_down:
            return False

        if self._costs_buffer.put(cost):
            self._stats.costs_collected += 1
            return True
        return False

    def collect_alert(self, alert: Alert) -> bool:
        """Collect an alert."""
        if self._shutting_down:
            return False

        if self._alerts_buffer.put(alert):
            self._stats.alerts_collected += 1
            return True
        return False

    def collect_audit(self, audit: AuditEvent) -> bool:
        """Collect an audit event."""
        if self._shutting_down:
            return False

        if self._audits_buffer.put(audit):
            self._stats.audits_collected += 1
            return True
        return False

    # === Convenience Methods ===

    def record_metric(
        self,
        name: str,
        value: float,
        labels: Dict[str, str] = None,
        metric_type: MetricType = MetricType.GAUGE,
        unit: str = "",
    ) -> None:
        """Record a metric with simple API."""
        metric = Metric(
            name=name,
            value=value,
            labels=labels or {},
            metric_type=metric_type,
            unit=unit,
        )
        self.collect_metric(metric)

    def record_counter(
        self,
        name: str,
        value: float = 1.0,
        labels: Dict[str, str] = None,
    ) -> None:
        """Increment a counter."""
        self.record_metric(name, value, labels, MetricType.COUNTER)

    def record_gauge(
        self,
        name: str,
        value: float,
        labels: Dict[str, str] = None,
    ) -> None:
        """Set a gauge value."""
        self.record_metric(name, value, labels, MetricType.GAUGE)

    def record_histogram(
        self,
        name: str,
        value: float,
        labels: Dict[str, str] = None,
    ) -> None:
        """Record a histogram observation."""
        self.record_metric(name, value, labels, MetricType.HISTOGRAM)

    def record_duration(
        self,
        name: str,
        duration_seconds: float,
        labels: Dict[str, str] = None,
    ) -> None:
        """Record a duration metric."""
        self.record_metric(name, duration_seconds, labels, MetricType.HISTOGRAM, "seconds")

    @asynccontextmanager
    async def timed_operation(
        self,
        metric_name: str,
        labels: Dict[str, str] = None,
    ):
        """Context manager that records operation duration."""
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            self.record_duration(metric_name, duration, labels)

    # === Exporter Registration ===

    def add_exporter(self, config: ExporterConfig) -> None:
        """Add an exporter."""
        self._exporters[config.exporter_type].append(config)
        logger.info(f"Added {config.exporter_type.value} exporter: {config.name}")

    def add_metric_exporter(
        self,
        name: str,
        exporter_fn: Callable[[List[Metric]], Coroutine],
        **kwargs,
    ) -> None:
        """Add a metric exporter."""
        self.add_exporter(ExporterConfig(
            name=name,
            exporter_type=ExporterType.METRICS,
            exporter_fn=exporter_fn,
            **kwargs,
        ))

    def add_span_exporter(
        self,
        name: str,
        exporter_fn: Callable[[List[Span]], Coroutine],
        **kwargs,
    ) -> None:
        """Add a span exporter."""
        self.add_exporter(ExporterConfig(
            name=name,
            exporter_type=ExporterType.SPANS,
            exporter_fn=exporter_fn,
            **kwargs,
        ))

    def add_log_exporter(
        self,
        name: str,
        exporter_fn: Callable[[List[LogEntry]], Coroutine],
        **kwargs,
    ) -> None:
        """Add a log exporter."""
        self.add_exporter(ExporterConfig(
            name=name,
            exporter_type=ExporterType.LOGS,
            exporter_fn=exporter_fn,
            **kwargs,
        ))

    # === Processor Registration ===

    def add_metric_processor(self, processor: Callable[[Metric], Optional[Metric]]) -> None:
        """Add a metric processor."""
        self._metric_processors.append(processor)

    def add_span_processor(self, processor: Callable[[Span], Optional[Span]]) -> None:
        """Add a span processor."""
        self._span_processors.append(processor)

    def add_log_processor(self, processor: Callable[[LogEntry], Optional[LogEntry]]) -> None:
        """Add a log processor."""
        self._log_processors.append(processor)

    # === Filter Registration ===

    def add_metric_filter(self, filter_fn: Callable[[Metric], bool]) -> None:
        """Add a metric filter. Return True to keep, False to drop."""
        self._metric_filters.append(filter_fn)

    def add_span_filter(self, filter_fn: Callable[[Span], bool]) -> None:
        """Add a span filter."""
        self._span_filters.append(filter_fn)

    def add_log_filter(self, filter_fn: Callable[[LogEntry], bool]) -> None:
        """Add a log filter."""
        self._log_filters.append(filter_fn)

    # === Sampling ===

    def set_sample_rates(
        self,
        metrics: float = 1.0,
        spans: float = 1.0,
        logs: float = 1.0,
    ) -> None:
        """Set sampling rates (0.0-1.0)."""
        self._metric_sample_rate = max(0.0, min(1.0, metrics))
        self._span_sample_rate = max(0.0, min(1.0, spans))
        self._log_sample_rate = max(0.0, min(1.0, logs))

    def _should_sample(self, rate: float) -> bool:
        """Check if should sample based on rate."""
        if rate >= 1.0:
            return True
        if rate <= 0.0:
            return False
        import random
        return random.random() < rate

    # === Flush Logic ===

    async def _flush_loop(self) -> None:
        """Background flush loop."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(self.flush_interval)
                await self._flush_all()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Flush error: {e}")
                self._stats.export_errors += 1

    async def _flush_all(self) -> None:
        """Flush all buffers."""
        async with self._flush_lock:
            start_time = time.perf_counter()

            await asyncio.gather(
                self._flush_buffer(
                    self._metrics_buffer,
                    self._exporters[ExporterType.METRICS],
                ),
                self._flush_buffer(
                    self._spans_buffer,
                    self._exporters[ExporterType.SPANS],
                ),
                self._flush_buffer(
                    self._logs_buffer,
                    self._exporters[ExporterType.LOGS],
                ),
                self._flush_buffer(
                    self._costs_buffer,
                    self._exporters[ExporterType.COSTS],
                ),
                self._flush_buffer(
                    self._alerts_buffer,
                    self._exporters[ExporterType.ALERTS],
                ),
                self._flush_buffer(
                    self._audits_buffer,
                    self._exporters[ExporterType.AUDITS],
                ),
                return_exceptions=True,
            )

            self._stats.flush_count += 1
            self._stats.last_flush_time = datetime.utcnow()
            self._stats.last_export_duration_ms = (time.perf_counter() - start_time) * 1000

    async def _flush_buffer(
        self,
        buffer: RingBuffer,
        exporters: List[ExporterConfig],
    ) -> None:
        """Flush a single buffer to its exporters."""
        if not exporters or len(buffer) == 0:
            return

        items = buffer.get_all()
        if not items:
            return

        # Process in batches
        for i in range(0, len(items), self.max_export_batch_size):
            batch = items[i:i + self.max_export_batch_size]

            for config in exporters:
                if not config.enabled:
                    continue

                success = await self._export_with_retry(config, batch)
                if success:
                    self._stats.export_success += 1
                else:
                    self._stats.export_errors += 1

    async def _export_with_retry(
        self,
        config: ExporterConfig,
        batch: List[Any],
    ) -> bool:
        """Export with retry logic."""
        for attempt in range(config.retry_count):
            try:
                await asyncio.wait_for(
                    config.exporter_fn(batch),
                    timeout=config.timeout_ms / 1000,
                )
                return True
            except asyncio.TimeoutError:
                logger.warning(
                    f"Export timeout for {config.name}",
                    attempt=attempt + 1,
                    batch_size=len(batch),
                )
            except Exception as e:
                logger.warning(
                    f"Export error for {config.name}: {e}",
                    attempt=attempt + 1,
                )

            if attempt < config.retry_count - 1:
                delay = config.retry_delay_ms * (2 ** attempt) / 1000
                await asyncio.sleep(delay)

        logger.error(f"Export failed for {config.name} after {config.retry_count} attempts")
        return False

    async def force_flush(self) -> None:
        """Force immediate flush of all buffers."""
        await self._flush_all()

    # === Statistics ===

    def get_stats(self) -> Dict[str, Any]:
        """Get collector statistics."""
        return {
            **self._stats.to_dict(),
            "metrics_buffer_size": len(self._metrics_buffer),
            "spans_buffer_size": len(self._spans_buffer),
            "logs_buffer_size": len(self._logs_buffer),
            "costs_buffer_size": len(self._costs_buffer),
            "alerts_buffer_size": len(self._alerts_buffer),
            "audits_buffer_size": len(self._audits_buffer),
            "metrics_dropped_buffer": self._metrics_buffer.dropped_count,
            "spans_dropped_buffer": self._spans_buffer.dropped_count,
            "logs_dropped_buffer": self._logs_buffer.dropped_count,
        }

    def reset_stats(self) -> None:
        """Reset statistics."""
        self._stats = CollectorStats()


# === Default Exporters ===

async def console_metric_exporter(metrics: List[Metric]) -> None:
    """Export metrics to console (for debugging)."""
    for metric in metrics:
        print(f"METRIC: {metric.full_name()} = {metric.value}")


async def console_span_exporter(spans: List[Span]) -> None:
    """Export spans to console (for debugging)."""
    for span in spans:
        print(f"SPAN: {span.name} ({span.duration_ms:.2f}ms) [{span.status.value}]")


async def console_log_exporter(logs: List[LogEntry]) -> None:
    """Export logs to console (for debugging)."""
    for log in logs:
        print(f"LOG [{log.level.value.upper()}]: {log.message}")
