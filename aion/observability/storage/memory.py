"""
AION In-Memory Storage

In-memory storage backends for development and testing.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import threading

from aion.observability.types import Metric, Span, LogEntry, Trace, LogLevel
from aion.observability.storage.base import MetricStore, TraceStore, LogStore


class InMemoryMetricStore(MetricStore):
    """
    In-memory metric storage.

    Suitable for development and single-instance deployments.
    """

    def __init__(
        self,
        max_points_per_series: int = 10000,
        retention: timedelta = timedelta(hours=24),
    ):
        self.max_points_per_series = max_points_per_series
        self.retention = retention

        # Storage: metric_key -> [(timestamp, value), ...]
        self._data: Dict[str, List[Tuple[datetime, float]]] = defaultdict(list)

        # Latest values
        self._latest: Dict[str, Tuple[datetime, float]] = {}

        # Lock for thread safety
        self._lock = threading.RLock()

        self._initialized = False

    async def initialize(self) -> None:
        self._initialized = True

    async def shutdown(self) -> None:
        self._initialized = False

    async def health_check(self) -> bool:
        return self._initialized

    def _metric_key(self, name: str, labels: Dict[str, str] = None) -> str:
        """Create a unique key for a metric series."""
        if not labels:
            return name
        labels_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}:{labels_str}"

    async def write_metrics(self, metrics: List[Metric]) -> int:
        """Write metrics to storage."""
        count = 0
        with self._lock:
            for metric in metrics:
                key = self._metric_key(metric.name, metric.labels)

                # Add to time series
                self._data[key].append((metric.timestamp, metric.value))

                # Trim if needed
                if len(self._data[key]) > self.max_points_per_series:
                    self._data[key] = self._data[key][-self.max_points_per_series:]

                # Update latest
                self._latest[key] = (metric.timestamp, metric.value)

                count += 1

        return count

    async def query_metrics(
        self,
        name: str,
        labels: Dict[str, str] = None,
        start_time: datetime = None,
        end_time: datetime = None,
        limit: int = 1000,
    ) -> List[Tuple[datetime, float]]:
        """Query metric time series."""
        key = self._metric_key(name, labels)

        with self._lock:
            series = self._data.get(key, [])

            # Apply time filters
            if start_time:
                series = [(t, v) for t, v in series if t >= start_time]
            if end_time:
                series = [(t, v) for t, v in series if t <= end_time]

            return series[-limit:]

    async def get_latest(
        self,
        name: str,
        labels: Dict[str, str] = None,
    ) -> Optional[float]:
        """Get latest value for a metric."""
        key = self._metric_key(name, labels)

        with self._lock:
            latest = self._latest.get(key)
            return latest[1] if latest else None

    async def list_metrics(self) -> List[str]:
        """List all metric names."""
        with self._lock:
            names = set()
            for key in self._data.keys():
                name = key.split(":")[0]
                names.add(name)
            return list(names)

    async def cleanup(self) -> int:
        """Remove data older than retention period. Returns count removed."""
        cutoff = datetime.utcnow() - self.retention
        removed = 0

        with self._lock:
            for key in list(self._data.keys()):
                original_len = len(self._data[key])
                self._data[key] = [(t, v) for t, v in self._data[key] if t >= cutoff]
                removed += original_len - len(self._data[key])

        return removed


class InMemoryTraceStore(TraceStore):
    """
    In-memory trace storage.

    Suitable for development and single-instance deployments.
    """

    def __init__(
        self,
        max_traces: int = 10000,
        retention: timedelta = timedelta(hours=24),
    ):
        self.max_traces = max_traces
        self.retention = retention

        # Storage: trace_id -> Trace
        self._traces: Dict[str, Trace] = {}

        # Index: span_id -> (trace_id, span)
        self._span_index: Dict[str, Tuple[str, Span]] = {}

        # Service index for querying
        self._service_index: Dict[str, List[str]] = defaultdict(list)

        # Lock
        self._lock = threading.RLock()

        self._initialized = False

    async def initialize(self) -> None:
        self._initialized = True

    async def shutdown(self) -> None:
        self._initialized = False

    async def health_check(self) -> bool:
        return self._initialized

    async def write_spans(self, spans: List[Span]) -> int:
        """Write spans to storage."""
        count = 0
        with self._lock:
            for span in spans:
                trace_id = span.trace_id

                # Create trace if needed
                if trace_id not in self._traces:
                    self._traces[trace_id] = Trace(trace_id=trace_id)

                trace = self._traces[trace_id]
                trace.spans.append(span)

                # Update trace timing
                if trace.start_time is None or span.start_time < trace.start_time:
                    trace.start_time = span.start_time
                if span.end_time:
                    if trace.end_time is None or span.end_time > trace.end_time:
                        trace.end_time = span.end_time

                # Index span
                self._span_index[span.span_id] = (trace_id, span)

                # Service index
                if trace_id not in self._service_index[span.service_name]:
                    self._service_index[span.service_name].append(trace_id)

                count += 1

            # Trim if needed
            if len(self._traces) > self.max_traces:
                self._cleanup_oldest()

        return count

    def _cleanup_oldest(self) -> None:
        """Remove oldest traces to stay within limit."""
        # Sort by start time
        sorted_traces = sorted(
            self._traces.items(),
            key=lambda x: x[1].start_time or datetime.min,
        )

        # Remove oldest
        to_remove = len(self._traces) - self.max_traces
        for trace_id, _ in sorted_traces[:to_remove]:
            self._remove_trace(trace_id)

    def _remove_trace(self, trace_id: str) -> None:
        """Remove a trace and its indexes."""
        trace = self._traces.pop(trace_id, None)
        if trace:
            for span in trace.spans:
                self._span_index.pop(span.span_id, None)
                if trace_id in self._service_index.get(span.service_name, []):
                    self._service_index[span.service_name].remove(trace_id)

    async def get_trace(self, trace_id: str) -> Optional[Trace]:
        """Get a complete trace by ID."""
        with self._lock:
            return self._traces.get(trace_id)

    async def query_traces(
        self,
        service_name: str = None,
        operation_name: str = None,
        start_time: datetime = None,
        end_time: datetime = None,
        min_duration_ms: float = None,
        max_duration_ms: float = None,
        limit: int = 100,
    ) -> List[Trace]:
        """Query traces with filters."""
        with self._lock:
            traces = list(self._traces.values())

            # Filter by service
            if service_name:
                trace_ids = set(self._service_index.get(service_name, []))
                traces = [t for t in traces if t.trace_id in trace_ids]

            # Filter by operation (check root span name)
            if operation_name:
                traces = [
                    t for t in traces
                    if t.root_span and t.root_span.name == operation_name
                ]

            # Filter by time
            if start_time:
                traces = [t for t in traces if t.start_time and t.start_time >= start_time]
            if end_time:
                traces = [t for t in traces if t.end_time and t.end_time <= end_time]

            # Filter by duration
            if min_duration_ms:
                traces = [t for t in traces if t.duration_ms >= min_duration_ms]
            if max_duration_ms:
                traces = [t for t in traces if t.duration_ms <= max_duration_ms]

            # Sort by start time descending
            traces.sort(key=lambda t: t.start_time or datetime.min, reverse=True)

            return traces[:limit]

    async def get_span(self, trace_id: str, span_id: str) -> Optional[Span]:
        """Get a specific span."""
        with self._lock:
            result = self._span_index.get(span_id)
            if result and result[0] == trace_id:
                return result[1]
            return None


class InMemoryLogStore(LogStore):
    """
    In-memory log storage.

    Suitable for development and single-instance deployments.
    """

    def __init__(
        self,
        max_logs: int = 100000,
        retention: timedelta = timedelta(hours=24),
    ):
        self.max_logs = max_logs
        self.retention = retention

        # Storage
        self._logs: List[LogEntry] = []

        # Indexes
        self._by_level: Dict[str, List[int]] = defaultdict(list)
        self._by_trace: Dict[str, List[int]] = defaultdict(list)
        self._by_logger: Dict[str, List[int]] = defaultdict(list)

        # Lock
        self._lock = threading.RLock()

        self._initialized = False

    async def initialize(self) -> None:
        self._initialized = True

    async def shutdown(self) -> None:
        self._initialized = False

    async def health_check(self) -> bool:
        return self._initialized

    async def write_logs(self, logs: List[LogEntry]) -> int:
        """Write logs to storage."""
        count = 0
        with self._lock:
            for log in logs:
                idx = len(self._logs)
                self._logs.append(log)

                # Index by level
                self._by_level[log.level.value].append(idx)

                # Index by trace
                if log.trace_id:
                    self._by_trace[log.trace_id].append(idx)

                # Index by logger
                if log.logger_name:
                    self._by_logger[log.logger_name].append(idx)

                count += 1

            # Trim if needed
            if len(self._logs) > self.max_logs:
                self._trim()

        return count

    def _trim(self) -> None:
        """Trim logs to stay within limit."""
        # Keep last max_logs
        to_remove = len(self._logs) - self.max_logs
        if to_remove <= 0:
            return

        self._logs = self._logs[to_remove:]

        # Rebuild indexes
        self._by_level.clear()
        self._by_trace.clear()
        self._by_logger.clear()

        for idx, log in enumerate(self._logs):
            self._by_level[log.level.value].append(idx)
            if log.trace_id:
                self._by_trace[log.trace_id].append(idx)
            if log.logger_name:
                self._by_logger[log.logger_name].append(idx)

    async def query_logs(
        self,
        level: str = None,
        logger_name: str = None,
        message_contains: str = None,
        trace_id: str = None,
        start_time: datetime = None,
        end_time: datetime = None,
        limit: int = 1000,
    ) -> List[LogEntry]:
        """Query logs with filters."""
        with self._lock:
            # Start with all logs or filtered by index
            if trace_id:
                indexes = self._by_trace.get(trace_id, [])
                logs = [self._logs[i] for i in indexes if i < len(self._logs)]
            elif level:
                indexes = self._by_level.get(level, [])
                logs = [self._logs[i] for i in indexes if i < len(self._logs)]
            elif logger_name:
                indexes = self._by_logger.get(logger_name, [])
                logs = [self._logs[i] for i in indexes if i < len(self._logs)]
            else:
                logs = list(self._logs)

            # Apply remaining filters
            if level and trace_id:
                logs = [l for l in logs if l.level.value == level]
            if logger_name and (trace_id or level):
                logs = [l for l in logs if l.logger_name == logger_name]
            if message_contains:
                logs = [l for l in logs if message_contains.lower() in l.message.lower()]
            if start_time:
                logs = [l for l in logs if l.timestamp >= start_time]
            if end_time:
                logs = [l for l in logs if l.timestamp <= end_time]

            # Sort by timestamp descending
            logs.sort(key=lambda l: l.timestamp, reverse=True)

            return logs[:limit]

    async def get_log_count(
        self,
        level: str = None,
        start_time: datetime = None,
        end_time: datetime = None,
    ) -> int:
        """Get count of logs matching filters."""
        with self._lock:
            if level:
                logs = [self._logs[i] for i in self._by_level.get(level, []) if i < len(self._logs)]
            else:
                logs = self._logs

            if start_time:
                logs = [l for l in logs if l.timestamp >= start_time]
            if end_time:
                logs = [l for l in logs if l.timestamp <= end_time]

            return len(logs)
