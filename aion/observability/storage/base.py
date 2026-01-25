"""
AION Storage Base Classes

Abstract base classes for observability storage backends.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from aion.observability.types import Metric, Span, LogEntry, Trace


class StorageBackend(ABC):
    """Base class for storage backends."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the storage backend."""
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown and cleanup resources."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if storage is healthy."""
        pass


class MetricStore(StorageBackend):
    """Base class for metric storage."""

    @abstractmethod
    async def write_metrics(self, metrics: List[Metric]) -> int:
        """Write metrics to storage. Returns count written."""
        pass

    @abstractmethod
    async def query_metrics(
        self,
        name: str,
        labels: Dict[str, str] = None,
        start_time: datetime = None,
        end_time: datetime = None,
        limit: int = 1000,
    ) -> List[Tuple[datetime, float]]:
        """Query metric time series."""
        pass

    @abstractmethod
    async def get_latest(
        self,
        name: str,
        labels: Dict[str, str] = None,
    ) -> Optional[float]:
        """Get latest value for a metric."""
        pass

    @abstractmethod
    async def list_metrics(self) -> List[str]:
        """List all metric names."""
        pass


class TraceStore(StorageBackend):
    """Base class for trace storage."""

    @abstractmethod
    async def write_spans(self, spans: List[Span]) -> int:
        """Write spans to storage. Returns count written."""
        pass

    @abstractmethod
    async def get_trace(self, trace_id: str) -> Optional[Trace]:
        """Get a complete trace by ID."""
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    async def get_span(self, trace_id: str, span_id: str) -> Optional[Span]:
        """Get a specific span."""
        pass


class LogStore(StorageBackend):
    """Base class for log storage."""

    @abstractmethod
    async def write_logs(self, logs: List[LogEntry]) -> int:
        """Write logs to storage. Returns count written."""
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    async def get_log_count(
        self,
        level: str = None,
        start_time: datetime = None,
        end_time: datetime = None,
    ) -> int:
        """Get count of logs matching filters."""
        pass
