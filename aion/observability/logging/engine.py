"""
AION Logging Engine

Structured logging with:
- Automatic trace correlation
- Multiple output handlers
- Log aggregation and search
- Sampling for high-volume logs
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import sys
import threading
import traceback
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import structlog

from aion.observability.types import LogEntry, LogLevel
from aion.observability.collector import TelemetryCollector
from aion.observability.context import get_context_manager

base_logger = structlog.get_logger(__name__)


class LoggingEngine:
    """
    SOTA Logging engine with observability integration.

    Features:
    - Automatic trace context injection
    - Structured JSON logging
    - Multiple output handlers
    - Log level filtering
    - Sampling for debug logs
    """

    def __init__(
        self,
        collector: TelemetryCollector,
        service_name: str = "aion",
        default_level: LogLevel = LogLevel.INFO,
        debug_sample_rate: float = 0.1,
    ):
        self.collector = collector
        self.service_name = service_name
        self.default_level = default_level
        self.debug_sample_rate = debug_sample_rate

        # Log level filtering
        self._level_filter = self._level_to_int(default_level)

        # Named loggers
        self._loggers: Dict[str, "ObservabilityLogger"] = {}

        # Handlers
        self._handlers: List[Callable[[LogEntry], None]] = []

        # Statistics
        self._stats = {
            "logs_processed": 0,
            "logs_filtered": 0,
            "logs_sampled": 0,
        }

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the logging engine."""
        if self._initialized:
            return

        base_logger.info("Initializing Logging Engine")

        # Configure structlog
        self._configure_structlog()

        self._initialized = True

    async def shutdown(self) -> None:
        """Shutdown the logging engine."""
        base_logger.info("Shutting down Logging Engine")
        self._initialized = False

    def _configure_structlog(self) -> None:
        """Configure structlog with observability processors."""
        structlog.configure(
            processors=[
                structlog.stdlib.add_log_level,
                structlog.stdlib.add_logger_name,
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                self._add_trace_context,
                structlog.processors.JSONRenderer(),
            ],
            wrapper_class=structlog.stdlib.BoundLogger,
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )

    def _add_trace_context(
        self,
        logger: Any,
        method_name: str,
        event_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Add trace context to log events."""
        ctx = get_context_manager()

        trace_id = ctx.get_trace_id()
        if trace_id:
            event_dict["trace_id"] = trace_id

        span = ctx.get_current_span()
        if span:
            event_dict["span_id"] = span.span_id

        request_id = ctx.get_request_id()
        if request_id:
            event_dict["request_id"] = request_id

        agent_id = ctx.get_agent_id()
        if agent_id:
            event_dict["agent_id"] = agent_id

        return event_dict

    def _level_to_int(self, level: LogLevel) -> int:
        """Convert log level to integer for comparison."""
        levels = {
            LogLevel.TRACE: 0,
            LogLevel.DEBUG: 10,
            LogLevel.INFO: 20,
            LogLevel.WARNING: 30,
            LogLevel.ERROR: 40,
            LogLevel.CRITICAL: 50,
        }
        return levels.get(level, 20)

    def set_level(self, level: LogLevel) -> None:
        """Set minimum log level."""
        self.default_level = level
        self._level_filter = self._level_to_int(level)

    def get_logger(self, name: str = "") -> "ObservabilityLogger":
        """Get or create a named logger."""
        if name not in self._loggers:
            self._loggers[name] = ObservabilityLogger(self, name)
        return self._loggers[name]

    def add_handler(self, handler: Callable[[LogEntry], None]) -> None:
        """Add a log handler."""
        self._handlers.append(handler)

    def log(
        self,
        level: LogLevel,
        message: str,
        logger_name: str = "",
        attributes: Dict[str, Any] = None,
        exception: Exception = None,
    ) -> None:
        """Log a message."""
        # Level filtering
        if self._level_to_int(level) < self._level_filter:
            self._stats["logs_filtered"] += 1
            return

        # Sampling for debug logs
        if level in (LogLevel.DEBUG, LogLevel.TRACE):
            import random
            if random.random() > self.debug_sample_rate:
                self._stats["logs_sampled"] += 1
                return

        # Get caller info
        frame = inspect.currentframe()
        caller_frame = frame.f_back.f_back if frame and frame.f_back else None

        source_file = ""
        source_line = 0
        source_function = ""

        if caller_frame:
            source_file = caller_frame.f_code.co_filename
            source_line = caller_frame.f_lineno
            source_function = caller_frame.f_code.co_name

        # Get context
        ctx = get_context_manager()

        # Create log entry
        entry = LogEntry(
            timestamp=datetime.utcnow(),
            level=level,
            message=message,
            logger_name=logger_name,
            trace_id=ctx.get_trace_id(),
            span_id=ctx.get_current_span().span_id if ctx.get_current_span() else None,
            attributes=attributes or {},
            service_name=self.service_name,
            source_file=source_file,
            source_line=source_line,
            source_function=source_function,
            request_id=ctx.get_request_id(),
            user_id=ctx.get_user_id(),
            session_id=ctx.get_session_id(),
            agent_id=ctx.get_agent_id(),
        )

        # Add exception info
        if exception:
            entry.exception_type = type(exception).__name__
            entry.exception_message = str(exception)
            entry.exception_traceback = traceback.format_exc()

        # Send to collector
        self.collector.collect_log(entry)

        # Call handlers
        for handler in self._handlers:
            try:
                handler(entry)
            except Exception as e:
                base_logger.error(f"Log handler error: {e}")

        self._stats["logs_processed"] += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get logging statistics."""
        return {
            **self._stats,
            "loggers_count": len(self._loggers),
            "handlers_count": len(self._handlers),
            "current_level": self.default_level.value,
        }


class ObservabilityLogger:
    """
    Logger with automatic observability context.

    Usage:
        logger = logging_engine.get_logger("my_module")
        logger.info("Processing request", request_id=123)
        logger.error("Failed to process", exception=e)
    """

    def __init__(self, engine: LoggingEngine, name: str = ""):
        self._engine = engine
        self._name = name
        self._bound_attributes: Dict[str, Any] = {}

    def bind(self, **attributes) -> "ObservabilityLogger":
        """Create a new logger with bound attributes."""
        new_logger = ObservabilityLogger(self._engine, self._name)
        new_logger._bound_attributes = {**self._bound_attributes, **attributes}
        return new_logger

    def _log(
        self,
        level: LogLevel,
        message: str,
        exception: Exception = None,
        **kwargs,
    ) -> None:
        """Internal log method."""
        attributes = {**self._bound_attributes, **kwargs}
        self._engine.log(
            level=level,
            message=message,
            logger_name=self._name,
            attributes=attributes,
            exception=exception,
        )

    def trace(self, message: str, **kwargs) -> None:
        """Log at trace level."""
        self._log(LogLevel.TRACE, message, **kwargs)

    def debug(self, message: str, **kwargs) -> None:
        """Log at debug level."""
        self._log(LogLevel.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs) -> None:
        """Log at info level."""
        self._log(LogLevel.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs) -> None:
        """Log at warning level."""
        self._log(LogLevel.WARNING, message, **kwargs)

    def warn(self, message: str, **kwargs) -> None:
        """Alias for warning."""
        self.warning(message, **kwargs)

    def error(self, message: str, exception: Exception = None, **kwargs) -> None:
        """Log at error level."""
        self._log(LogLevel.ERROR, message, exception=exception, **kwargs)

    def critical(self, message: str, exception: Exception = None, **kwargs) -> None:
        """Log at critical level."""
        self._log(LogLevel.CRITICAL, message, exception=exception, **kwargs)

    def exception(self, message: str, **kwargs) -> None:
        """Log an exception with traceback."""
        exc_info = sys.exc_info()
        if exc_info[1]:
            self._log(LogLevel.ERROR, message, exception=exc_info[1], **kwargs)
        else:
            self._log(LogLevel.ERROR, message, **kwargs)


# Standard logging handler integration
class ObservabilityHandler(logging.Handler):
    """
    Python logging handler that forwards to ObservabilityLogger.

    Usage:
        import logging
        handler = ObservabilityHandler(logging_engine)
        logging.getLogger().addHandler(handler)
    """

    LEVEL_MAP = {
        logging.DEBUG: LogLevel.DEBUG,
        logging.INFO: LogLevel.INFO,
        logging.WARNING: LogLevel.WARNING,
        logging.ERROR: LogLevel.ERROR,
        logging.CRITICAL: LogLevel.CRITICAL,
    }

    def __init__(self, engine: LoggingEngine):
        super().__init__()
        self._engine = engine

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record."""
        try:
            level = self.LEVEL_MAP.get(record.levelno, LogLevel.INFO)
            message = self.format(record)

            exception = None
            if record.exc_info and record.exc_info[1]:
                exception = record.exc_info[1]

            self._engine.log(
                level=level,
                message=message,
                logger_name=record.name,
                exception=exception,
                attributes={
                    "pathname": record.pathname,
                    "lineno": record.lineno,
                    "funcName": record.funcName,
                },
            )
        except Exception:
            self.handleError(record)


# Console handler for development
def console_log_handler(entry: LogEntry) -> None:
    """Print logs to console."""
    level_colors = {
        LogLevel.TRACE: "\033[90m",    # Gray
        LogLevel.DEBUG: "\033[36m",    # Cyan
        LogLevel.INFO: "\033[32m",     # Green
        LogLevel.WARNING: "\033[33m",  # Yellow
        LogLevel.ERROR: "\033[31m",    # Red
        LogLevel.CRITICAL: "\033[35m", # Magenta
    }
    reset = "\033[0m"

    color = level_colors.get(entry.level, "")
    level_str = entry.level.value.upper().ljust(8)

    # Format context
    context_parts = []
    if entry.trace_id:
        context_parts.append(f"trace={entry.trace_id[:8]}")
    if entry.request_id:
        context_parts.append(f"req={entry.request_id[:8]}")
    context_str = " ".join(context_parts)

    print(
        f"{color}[{entry.timestamp.strftime('%H:%M:%S')}] "
        f"{level_str} {entry.logger_name or 'root'}: "
        f"{entry.message}"
        f"{' (' + context_str + ')' if context_str else ''}{reset}"
    )

    if entry.exception_traceback:
        print(f"{level_colors[LogLevel.ERROR]}{entry.exception_traceback}{reset}")
