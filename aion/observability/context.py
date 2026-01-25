"""
AION Observability Context

Request context propagation for distributed tracing and correlation.
Implements OpenTelemetry-compatible context management with async support.
"""

from __future__ import annotations

import asyncio
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, TypeVar
import uuid

import structlog

from aion.observability.types import Span, SpanContext

logger = structlog.get_logger(__name__)

T = TypeVar('T')


# === Context Variables ===

# Current span context
_current_span: ContextVar[Optional[Span]] = ContextVar('current_span', default=None)

# Current trace context
_current_trace_id: ContextVar[Optional[str]] = ContextVar('current_trace_id', default=None)

# Request context
_request_id: ContextVar[Optional[str]] = ContextVar('request_id', default=None)
_user_id: ContextVar[Optional[str]] = ContextVar('user_id', default=None)
_session_id: ContextVar[Optional[str]] = ContextVar('session_id', default=None)
_agent_id: ContextVar[Optional[str]] = ContextVar('agent_id', default=None)
_goal_id: ContextVar[Optional[str]] = ContextVar('goal_id', default=None)

# Baggage (key-value pairs propagated across services)
_baggage: ContextVar[Dict[str, str]] = ContextVar('baggage', default_factory=dict)

# Custom attributes
_custom_attributes: ContextVar[Dict[str, Any]] = ContextVar('custom_attributes', default_factory=dict)


@dataclass
class ObservabilityContext:
    """
    Full observability context for a request/operation.

    Captures all relevant context for metrics, tracing, and logging.
    """

    # Trace context
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    parent_span_id: Optional[str] = None

    # Request context
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    agent_id: Optional[str] = None
    goal_id: Optional[str] = None

    # Service info
    service_name: str = "aion"
    service_version: str = ""
    environment: str = ""

    # Timing
    start_time: datetime = field(default_factory=datetime.utcnow)

    # Baggage
    baggage: Dict[str, str] = field(default_factory=dict)

    # Custom attributes
    attributes: Dict[str, Any] = field(default_factory=dict)

    # Sampling
    sampled: bool = True

    @classmethod
    def current(cls) -> "ObservabilityContext":
        """Get current context from context variables."""
        span = _current_span.get()
        return cls(
            trace_id=span.trace_id if span else _current_trace_id.get(),
            span_id=span.span_id if span else None,
            parent_span_id=span.parent_span_id if span else None,
            request_id=_request_id.get(),
            user_id=_user_id.get(),
            session_id=_session_id.get(),
            agent_id=_agent_id.get(),
            goal_id=_goal_id.get(),
            baggage=_baggage.get().copy() if _baggage.get() else {},
            attributes=_custom_attributes.get().copy() if _custom_attributes.get() else {},
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "request_id": self.request_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "agent_id": self.agent_id,
            "goal_id": self.goal_id,
            "service_name": self.service_name,
            "baggage": self.baggage,
            "attributes": self.attributes,
            "sampled": self.sampled,
        }

    def to_log_context(self) -> Dict[str, Any]:
        """Get context suitable for log enrichment."""
        ctx = {}
        if self.trace_id:
            ctx["trace_id"] = self.trace_id
        if self.span_id:
            ctx["span_id"] = self.span_id
        if self.request_id:
            ctx["request_id"] = self.request_id
        if self.user_id:
            ctx["user_id"] = self.user_id
        if self.session_id:
            ctx["session_id"] = self.session_id
        if self.agent_id:
            ctx["agent_id"] = self.agent_id
        ctx.update(self.attributes)
        return ctx


class ContextManager:
    """
    Manager for observability context operations.

    Provides methods for getting, setting, and propagating context.
    """

    def __init__(self):
        self._tokens: Dict[str, List[Token]] = {
            "span": [],
            "trace": [],
            "request": [],
            "user": [],
            "session": [],
            "agent": [],
            "goal": [],
            "baggage": [],
            "attributes": [],
        }

    # === Span Context ===

    def get_current_span(self) -> Optional[Span]:
        """Get the current span."""
        return _current_span.get()

    def set_current_span(self, span: Optional[Span]) -> Token:
        """Set the current span."""
        token = _current_span.set(span)
        self._tokens["span"].append(token)

        # Also update trace ID
        if span:
            self.set_trace_id(span.trace_id)

        return token

    def reset_span(self, token: Token) -> None:
        """Reset span to previous value."""
        _current_span.reset(token)

    def get_span_context(self) -> Optional[SpanContext]:
        """Get current span context for propagation."""
        span = self.get_current_span()
        return span.context if span else None

    # === Trace Context ===

    def get_trace_id(self) -> Optional[str]:
        """Get current trace ID."""
        span = _current_span.get()
        if span:
            return span.trace_id
        return _current_trace_id.get()

    def set_trace_id(self, trace_id: str) -> Token:
        """Set trace ID."""
        token = _current_trace_id.set(trace_id)
        self._tokens["trace"].append(token)
        return token

    def ensure_trace_id(self) -> str:
        """Ensure there's a trace ID, generating one if needed."""
        trace_id = self.get_trace_id()
        if not trace_id:
            trace_id = uuid.uuid4().hex
            self.set_trace_id(trace_id)
        return trace_id

    # === Request Context ===

    def get_request_id(self) -> Optional[str]:
        """Get current request ID."""
        return _request_id.get()

    def set_request_id(self, request_id: str) -> Token:
        """Set request ID."""
        token = _request_id.set(request_id)
        self._tokens["request"].append(token)
        return token

    def ensure_request_id(self) -> str:
        """Ensure there's a request ID, generating one if needed."""
        req_id = self.get_request_id()
        if not req_id:
            req_id = uuid.uuid4().hex[:16]
            self.set_request_id(req_id)
        return req_id

    # === User Context ===

    def get_user_id(self) -> Optional[str]:
        """Get current user ID."""
        return _user_id.get()

    def set_user_id(self, user_id: str) -> Token:
        """Set user ID."""
        token = _user_id.set(user_id)
        self._tokens["user"].append(token)
        return token

    # === Session Context ===

    def get_session_id(self) -> Optional[str]:
        """Get current session ID."""
        return _session_id.get()

    def set_session_id(self, session_id: str) -> Token:
        """Set session ID."""
        token = _session_id.set(session_id)
        self._tokens["session"].append(token)
        return token

    # === Agent Context ===

    def get_agent_id(self) -> Optional[str]:
        """Get current agent ID."""
        return _agent_id.get()

    def set_agent_id(self, agent_id: str) -> Token:
        """Set agent ID."""
        token = _agent_id.set(agent_id)
        self._tokens["agent"].append(token)
        return token

    # === Goal Context ===

    def get_goal_id(self) -> Optional[str]:
        """Get current goal ID."""
        return _goal_id.get()

    def set_goal_id(self, goal_id: str) -> Token:
        """Set goal ID."""
        token = _goal_id.set(goal_id)
        self._tokens["goal"].append(token)
        return token

    # === Baggage ===

    def get_baggage(self) -> Dict[str, str]:
        """Get all baggage items."""
        return _baggage.get().copy() if _baggage.get() else {}

    def get_baggage_item(self, key: str) -> Optional[str]:
        """Get a baggage item."""
        baggage = _baggage.get()
        return baggage.get(key) if baggage else None

    def set_baggage_item(self, key: str, value: str) -> None:
        """Set a baggage item."""
        baggage = _baggage.get()
        if baggage is None:
            baggage = {}
        baggage[key] = value
        _baggage.set(baggage)

    def remove_baggage_item(self, key: str) -> None:
        """Remove a baggage item."""
        baggage = _baggage.get()
        if baggage and key in baggage:
            del baggage[key]
            _baggage.set(baggage)

    # === Custom Attributes ===

    def get_attributes(self) -> Dict[str, Any]:
        """Get all custom attributes."""
        return _custom_attributes.get().copy() if _custom_attributes.get() else {}

    def get_attribute(self, key: str) -> Optional[Any]:
        """Get a custom attribute."""
        attrs = _custom_attributes.get()
        return attrs.get(key) if attrs else None

    def set_attribute(self, key: str, value: Any) -> None:
        """Set a custom attribute."""
        attrs = _custom_attributes.get()
        if attrs is None:
            attrs = {}
        attrs[key] = value
        _custom_attributes.set(attrs)

    # === Context Propagation ===

    def inject_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Inject context into HTTP headers (W3C Trace Context)."""
        span_context = self.get_span_context()

        if span_context:
            headers["traceparent"] = span_context.to_traceparent()
            if span_context.trace_state:
                headers["tracestate"] = span_context.trace_state

        # Inject request ID
        request_id = self.get_request_id()
        if request_id:
            headers["x-request-id"] = request_id

        # Inject baggage
        baggage = self.get_baggage()
        if baggage:
            baggage_items = [f"{k}={v}" for k, v in baggage.items()]
            headers["baggage"] = ",".join(baggage_items)

        return headers

    def extract_headers(self, headers: Dict[str, str]) -> None:
        """Extract context from HTTP headers."""
        # Extract traceparent
        traceparent = headers.get("traceparent")
        if traceparent:
            span_context = SpanContext.from_traceparent(traceparent)
            if span_context:
                self.set_trace_id(span_context.trace_id)

        # Extract request ID
        request_id = headers.get("x-request-id")
        if request_id:
            self.set_request_id(request_id)

        # Extract baggage
        baggage_header = headers.get("baggage")
        if baggage_header:
            for item in baggage_header.split(","):
                if "=" in item:
                    key, value = item.split("=", 1)
                    self.set_baggage_item(key.strip(), value.strip())

    def get_context(self) -> ObservabilityContext:
        """Get full current context."""
        return ObservabilityContext.current()

    def clear(self) -> None:
        """Clear all context (use at request end)."""
        _current_span.set(None)
        _current_trace_id.set(None)
        _request_id.set(None)
        _user_id.set(None)
        _session_id.set(None)
        _agent_id.set(None)
        _goal_id.set(None)
        _baggage.set({})
        _custom_attributes.set({})


# Global context manager instance
_context_manager = ContextManager()


def get_context_manager() -> ContextManager:
    """Get the global context manager."""
    return _context_manager


# === Convenience Functions ===

def get_current_span() -> Optional[Span]:
    """Get the current span."""
    return _context_manager.get_current_span()


def get_trace_id() -> Optional[str]:
    """Get current trace ID."""
    return _context_manager.get_trace_id()


def get_request_id() -> Optional[str]:
    """Get current request ID."""
    return _context_manager.get_request_id()


def get_context() -> ObservabilityContext:
    """Get current observability context."""
    return _context_manager.get_context()


def set_attribute(key: str, value: Any) -> None:
    """Set a context attribute."""
    _context_manager.set_attribute(key, value)


# === Context Manager Decorators ===

class with_context:
    """
    Context manager/decorator for establishing observability context.

    Can be used as a context manager:
        with with_context(request_id="123", user_id="user"):
            do_something()

    Or as a decorator:
        @with_context(agent_id="agent-1")
        async def process():
            ...
    """

    def __init__(
        self,
        request_id: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        goal_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        **attributes,
    ):
        self.request_id = request_id
        self.user_id = user_id
        self.session_id = session_id
        self.agent_id = agent_id
        self.goal_id = goal_id
        self.trace_id = trace_id
        self.attributes = attributes
        self._tokens: List[Token] = []

    def __enter__(self) -> ObservabilityContext:
        ctx = _context_manager

        if self.trace_id:
            self._tokens.append(ctx.set_trace_id(self.trace_id))
        if self.request_id:
            self._tokens.append(ctx.set_request_id(self.request_id))
        if self.user_id:
            self._tokens.append(ctx.set_user_id(self.user_id))
        if self.session_id:
            self._tokens.append(ctx.set_session_id(self.session_id))
        if self.agent_id:
            self._tokens.append(ctx.set_agent_id(self.agent_id))
        if self.goal_id:
            self._tokens.append(ctx.set_goal_id(self.goal_id))

        for key, value in self.attributes.items():
            ctx.set_attribute(key, value)

        return ctx.get_context()

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Tokens are automatically cleaned up by contextvar
        pass

    async def __aenter__(self) -> ObservabilityContext:
        return self.__enter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.__exit__(exc_type, exc_val, exc_tb)

    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """Use as decorator."""
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                async with self:
                    return await func(*args, **kwargs)
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                with self:
                    return func(*args, **kwargs)
            return sync_wrapper


class with_request:
    """
    Context manager for HTTP request context.

    Sets up request ID, extracts headers, and cleans up on exit.
    """

    def __init__(
        self,
        headers: Dict[str, str] = None,
        request_id: Optional[str] = None,
    ):
        self.headers = headers or {}
        self.request_id = request_id or uuid.uuid4().hex[:16]

    def __enter__(self) -> ObservabilityContext:
        ctx = _context_manager

        # Set request ID
        ctx.set_request_id(self.request_id)

        # Extract from headers
        if self.headers:
            ctx.extract_headers(self.headers)

        # Ensure trace ID
        ctx.ensure_trace_id()

        return ctx.get_context()

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Don't clear - let it propagate through the request
        pass

    async def __aenter__(self) -> ObservabilityContext:
        return self.__enter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.__exit__(exc_type, exc_val, exc_tb)


def copy_context_to_task(coro):
    """
    Decorator to copy current context to a new asyncio task.

    Use when spawning background tasks that should inherit context.
    """
    import copy

    # Capture current context values
    captured = {
        "span": _current_span.get(),
        "trace_id": _current_trace_id.get(),
        "request_id": _request_id.get(),
        "user_id": _user_id.get(),
        "session_id": _session_id.get(),
        "agent_id": _agent_id.get(),
        "goal_id": _goal_id.get(),
        "baggage": _baggage.get().copy() if _baggage.get() else {},
        "attributes": _custom_attributes.get().copy() if _custom_attributes.get() else {},
    }

    async def wrapped():
        # Restore context in new task
        if captured["span"]:
            _current_span.set(captured["span"])
        if captured["trace_id"]:
            _current_trace_id.set(captured["trace_id"])
        if captured["request_id"]:
            _request_id.set(captured["request_id"])
        if captured["user_id"]:
            _user_id.set(captured["user_id"])
        if captured["session_id"]:
            _session_id.set(captured["session_id"])
        if captured["agent_id"]:
            _agent_id.set(captured["agent_id"])
        if captured["goal_id"]:
            _goal_id.set(captured["goal_id"])
        if captured["baggage"]:
            _baggage.set(captured["baggage"])
        if captured["attributes"]:
            _custom_attributes.set(captured["attributes"])

        return await coro

    return wrapped()
