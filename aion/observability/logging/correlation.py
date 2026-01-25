"""
AION Log Correlation

Correlate logs with traces and spans for unified observability.
"""

from __future__ import annotations

import functools
from typing import Any, Callable, Dict, Optional, TypeVar

import structlog

from aion.observability.types import LogEntry, LogLevel, Span
from aion.observability.context import get_context_manager, _current_span

T = TypeVar('T')


class CorrelatedLogger:
    """
    Logger that automatically correlates with the current trace context.

    Provides seamless integration between logging and tracing.
    """

    def __init__(
        self,
        name: str = "",
        service_name: str = "aion",
    ):
        self.name = name
        self.service_name = service_name
        self._base_logger = structlog.get_logger(name)
        self._bound_context: Dict[str, Any] = {}

    def bind(self, **context) -> "CorrelatedLogger":
        """Create a new logger with bound context."""
        new_logger = CorrelatedLogger(self.name, self.service_name)
        new_logger._bound_context = {**self._bound_context, **context}
        return new_logger

    def _get_context(self) -> Dict[str, Any]:
        """Get current trace context."""
        ctx = get_context_manager()
        context = dict(self._bound_context)

        # Add trace context
        trace_id = ctx.get_trace_id()
        if trace_id:
            context["trace_id"] = trace_id

        span = ctx.get_current_span()
        if span:
            context["span_id"] = span.span_id
            context["span_name"] = span.name

        # Add request context
        request_id = ctx.get_request_id()
        if request_id:
            context["request_id"] = request_id

        user_id = ctx.get_user_id()
        if user_id:
            context["user_id"] = user_id

        agent_id = ctx.get_agent_id()
        if agent_id:
            context["agent_id"] = agent_id

        return context

    def _log(self, method: str, event: str, **kwargs) -> None:
        """Internal log method with context injection."""
        context = self._get_context()
        context.update(kwargs)

        log_method = getattr(self._base_logger, method)
        log_method(event, **context)

    def debug(self, event: str, **kwargs) -> None:
        """Log at debug level."""
        self._log("debug", event, **kwargs)

    def info(self, event: str, **kwargs) -> None:
        """Log at info level."""
        self._log("info", event, **kwargs)

    def warning(self, event: str, **kwargs) -> None:
        """Log at warning level."""
        self._log("warning", event, **kwargs)

    def error(self, event: str, **kwargs) -> None:
        """Log at error level."""
        self._log("error", event, **kwargs)

    def critical(self, event: str, **kwargs) -> None:
        """Log at critical level."""
        self._log("critical", event, **kwargs)

    def exception(self, event: str, **kwargs) -> None:
        """Log exception with traceback."""
        self._log("exception", event, **kwargs)

    # === Span Integration ===

    def log_span_start(self, span: Span) -> None:
        """Log span start event."""
        self.debug(
            "span_started",
            span_name=span.name,
            span_id=span.span_id,
            parent_span_id=span.parent_span_id,
            span_kind=span.kind.value,
        )

    def log_span_end(self, span: Span) -> None:
        """Log span end event."""
        self.debug(
            "span_ended",
            span_name=span.name,
            span_id=span.span_id,
            duration_ms=span.duration_ms,
            status=span.status.value,
        )

    def log_span_error(self, span: Span, error: Exception) -> None:
        """Log span error."""
        self.error(
            "span_error",
            span_name=span.name,
            span_id=span.span_id,
            error_type=type(error).__name__,
            error_message=str(error),
        )


def inject_trace_context(log_func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator to inject trace context into log calls.

    Usage:
        @inject_trace_context
        def my_log_function(message, **kwargs):
            print(f"[{kwargs.get('trace_id', 'no-trace')}] {message}")
    """
    @functools.wraps(log_func)
    def wrapper(*args, **kwargs) -> T:
        ctx = get_context_manager()

        # Inject trace context
        trace_id = ctx.get_trace_id()
        if trace_id and "trace_id" not in kwargs:
            kwargs["trace_id"] = trace_id

        span = ctx.get_current_span()
        if span and "span_id" not in kwargs:
            kwargs["span_id"] = span.span_id

        request_id = ctx.get_request_id()
        if request_id and "request_id" not in kwargs:
            kwargs["request_id"] = request_id

        return log_func(*args, **kwargs)

    return wrapper


class SpanLogger:
    """
    Logger that logs events to both logging system and span events.

    Useful for detailed operation logging that should be visible in both logs and traces.
    """

    def __init__(
        self,
        span: Span,
        logger: CorrelatedLogger = None,
    ):
        self.span = span
        self.logger = logger or CorrelatedLogger()

    def log(
        self,
        level: LogLevel,
        message: str,
        **attributes,
    ) -> None:
        """Log to both logger and span."""
        # Log to logger
        log_method = getattr(self.logger, level.value, self.logger.info)
        log_method(message, **attributes)

        # Add as span event
        self.span.add_event(
            message,
            attributes={
                "log.level": level.value,
                **attributes,
            },
        )

    def debug(self, message: str, **attributes) -> None:
        """Log debug message."""
        self.log(LogLevel.DEBUG, message, **attributes)

    def info(self, message: str, **attributes) -> None:
        """Log info message."""
        self.log(LogLevel.INFO, message, **attributes)

    def warning(self, message: str, **attributes) -> None:
        """Log warning message."""
        self.log(LogLevel.WARNING, message, **attributes)

    def error(self, message: str, **attributes) -> None:
        """Log error message."""
        self.log(LogLevel.ERROR, message, **attributes)


# Structlog processors for correlation
def add_trace_context_processor(
    logger: Any,
    method_name: str,
    event_dict: Dict[str, Any],
) -> Dict[str, Any]:
    """Structlog processor to add trace context."""
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

    return event_dict


def add_service_context_processor(
    service_name: str,
    service_version: str = "",
) -> Callable:
    """Create a processor that adds service context."""
    def processor(
        logger: Any,
        method_name: str,
        event_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        event_dict["service"] = service_name
        if service_version:
            event_dict["version"] = service_version
        return event_dict

    return processor


# Configure structlog for correlation
def configure_correlated_logging(
    service_name: str = "aion",
    service_version: str = "",
    json_output: bool = True,
) -> None:
    """
    Configure structlog with trace correlation.

    Call this at application startup.
    """
    processors = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        add_trace_context_processor,
        add_service_context_processor(service_name, service_version),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if json_output:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


# Utility function to get a correlated logger
def get_logger(name: str = "") -> CorrelatedLogger:
    """Get a correlated logger."""
    return CorrelatedLogger(name)
