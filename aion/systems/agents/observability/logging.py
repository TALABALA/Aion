"""
Structured Logging

Comprehensive logging system for agent operations.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional
import structlog

logger = structlog.get_logger()


class LogLevel(str, Enum):
    """Log levels."""

    TRACE = "trace"
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

    @property
    def numeric(self) -> int:
        """Get numeric level for comparison."""
        levels = {
            "trace": 5,
            "debug": 10,
            "info": 20,
            "warning": 30,
            "error": 40,
            "critical": 50,
        }
        return levels.get(self.value, 20)


@dataclass
class LogEntry:
    """A structured log entry."""

    level: LogLevel
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    agent_id: Optional[str] = None
    task_id: Optional[str] = None
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    attributes: dict[str, Any] = field(default_factory=dict)
    exception: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "level": self.level.value,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "agent_id": self.agent_id,
            "task_id": self.task_id,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "attributes": self.attributes,
            "exception": self.exception,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


class StructuredLogger:
    """
    Structured logger for an agent.

    Features:
    - Structured log entries
    - Context propagation
    - Log filtering and sampling
    - Multiple output handlers
    """

    def __init__(
        self,
        agent_id: str,
        level: LogLevel = LogLevel.INFO,
    ):
        self.agent_id = agent_id
        self.level = level
        self._context: dict[str, Any] = {}
        self._handlers: list[Callable[[LogEntry], None]] = []
        self._entries: list[LogEntry] = []
        self._max_entries = 1000

    def set_level(self, level: LogLevel) -> None:
        """Set minimum log level."""
        self.level = level

    def set_context(self, **kwargs: Any) -> None:
        """Set context that will be added to all logs."""
        self._context.update(kwargs)

    def clear_context(self) -> None:
        """Clear context."""
        self._context.clear()

    def add_handler(self, handler: Callable[[LogEntry], None]) -> None:
        """Add a log handler."""
        self._handlers.append(handler)

    def _should_log(self, level: LogLevel) -> bool:
        """Check if level should be logged."""
        return level.numeric >= self.level.numeric

    def _log(
        self,
        level: LogLevel,
        message: str,
        exception: Optional[Exception] = None,
        **kwargs: Any,
    ) -> None:
        """Create and emit a log entry."""
        if not self._should_log(level):
            return

        entry = LogEntry(
            level=level,
            message=message,
            agent_id=self.agent_id,
            task_id=self._context.get("task_id"),
            trace_id=self._context.get("trace_id"),
            span_id=self._context.get("span_id"),
            attributes={**self._context, **kwargs},
            exception=str(exception) if exception else None,
        )

        # Store entry
        self._entries.append(entry)
        if len(self._entries) > self._max_entries:
            self._entries = self._entries[-self._max_entries:]

        # Call handlers
        for handler in self._handlers:
            try:
                handler(entry)
            except Exception as e:
                logger.warning("handler_error", error=str(e))

        # Log to structlog
        log_method = getattr(logger, level.value, logger.info)
        log_method(
            message,
            agent_id=self.agent_id,
            **{k: v for k, v in kwargs.items() if not k.startswith("_")},
        )

    def trace(self, message: str, **kwargs: Any) -> None:
        """Log at TRACE level."""
        self._log(LogLevel.TRACE, message, **kwargs)

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log at DEBUG level."""
        self._log(LogLevel.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs: Any) -> None:
        """Log at INFO level."""
        self._log(LogLevel.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log at WARNING level."""
        self._log(LogLevel.WARNING, message, **kwargs)

    def error(self, message: str, exception: Optional[Exception] = None, **kwargs: Any) -> None:
        """Log at ERROR level."""
        self._log(LogLevel.ERROR, message, exception=exception, **kwargs)

    def critical(self, message: str, exception: Optional[Exception] = None, **kwargs: Any) -> None:
        """Log at CRITICAL level."""
        self._log(LogLevel.CRITICAL, message, exception=exception, **kwargs)

    def get_entries(
        self,
        level: Optional[LogLevel] = None,
        limit: int = 100,
    ) -> list[LogEntry]:
        """Get recent log entries."""
        entries = self._entries

        if level:
            entries = [e for e in entries if e.level.numeric >= level.numeric]

        return entries[-limit:]


class LogAggregator:
    """
    Aggregates logs from multiple agents.

    Features:
    - Multi-agent log collection
    - Log searching and filtering
    - Log analysis
    """

    def __init__(self):
        self._loggers: dict[str, StructuredLogger] = {}
        self._all_entries: list[LogEntry] = []
        self._max_entries = 10000
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize aggregator."""
        self._initialized = True
        logger.info("log_aggregator_initialized")

    async def shutdown(self) -> None:
        """Shutdown aggregator."""
        self._initialized = False
        logger.info("log_aggregator_shutdown")

    def get_logger(
        self,
        agent_id: str,
        level: LogLevel = LogLevel.INFO,
    ) -> StructuredLogger:
        """Get or create logger for an agent."""
        if agent_id not in self._loggers:
            agent_logger = StructuredLogger(agent_id, level)

            # Add handler to aggregate
            agent_logger.add_handler(self._handle_entry)

            self._loggers[agent_id] = agent_logger

        return self._loggers[agent_id]

    def _handle_entry(self, entry: LogEntry) -> None:
        """Handle a log entry."""
        self._all_entries.append(entry)

        if len(self._all_entries) > self._max_entries:
            self._all_entries = self._all_entries[-self._max_entries:]

    def search(
        self,
        query: Optional[str] = None,
        agent_id: Optional[str] = None,
        level: Optional[LogLevel] = None,
        task_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> list[LogEntry]:
        """Search log entries."""
        results = self._all_entries

        if agent_id:
            results = [e for e in results if e.agent_id == agent_id]

        if level:
            results = [e for e in results if e.level.numeric >= level.numeric]

        if task_id:
            results = [e for e in results if e.task_id == task_id]

        if trace_id:
            results = [e for e in results if e.trace_id == trace_id]

        if start_time:
            results = [e for e in results if e.timestamp >= start_time]

        if end_time:
            results = [e for e in results if e.timestamp <= end_time]

        if query:
            query_lower = query.lower()
            results = [
                e for e in results
                if query_lower in e.message.lower()
                or any(
                    query_lower in str(v).lower()
                    for v in e.attributes.values()
                )
            ]

        return results[-limit:]

    def get_errors(
        self,
        agent_id: Optional[str] = None,
        limit: int = 100,
    ) -> list[LogEntry]:
        """Get error and critical logs."""
        return self.search(
            agent_id=agent_id,
            level=LogLevel.ERROR,
            limit=limit,
        )

    def get_by_trace(self, trace_id: str) -> list[LogEntry]:
        """Get all logs for a trace."""
        return [e for e in self._all_entries if e.trace_id == trace_id]

    def get_by_task(self, task_id: str) -> list[LogEntry]:
        """Get all logs for a task."""
        return [e for e in self._all_entries if e.task_id == task_id]

    def get_stats(self) -> dict[str, Any]:
        """Get logging statistics."""
        level_counts: dict[str, int] = {}
        agent_counts: dict[str, int] = {}

        for entry in self._all_entries:
            level_counts[entry.level.value] = level_counts.get(entry.level.value, 0) + 1
            if entry.agent_id:
                agent_counts[entry.agent_id] = agent_counts.get(entry.agent_id, 0) + 1

        return {
            "total_entries": len(self._all_entries),
            "loggers": len(self._loggers),
            "by_level": level_counts,
            "by_agent": agent_counts,
            "error_count": level_counts.get("error", 0) + level_counts.get("critical", 0),
        }

    def export_json(
        self,
        agent_id: Optional[str] = None,
        limit: int = 1000,
    ) -> str:
        """Export logs as JSON."""
        entries = self.search(agent_id=agent_id, limit=limit)
        return json.dumps([e.to_dict() for e in entries], indent=2)
