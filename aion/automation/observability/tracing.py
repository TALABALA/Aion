"""
AION Workflow Tracing

Distributed tracing for workflow executions.
"""

from __future__ import annotations

import functools
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, Generator, List, Optional, TypeVar

import structlog

from aion.automation.observability.telemetry import get_tracer, NoopSpan

logger = structlog.get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


@dataclass
class SpanAttributes:
    """Standard span attributes for workflows."""
    # Workflow attributes
    WORKFLOW_ID = "workflow.id"
    WORKFLOW_NAME = "workflow.name"
    WORKFLOW_VERSION = "workflow.version"

    # Execution attributes
    EXECUTION_ID = "execution.id"
    EXECUTION_STATUS = "execution.status"

    # Step attributes
    STEP_ID = "step.id"
    STEP_NAME = "step.name"
    STEP_TYPE = "step.type"
    STEP_INDEX = "step.index"

    # Action attributes
    ACTION_TYPE = "action.type"
    ACTION_NAME = "action.name"

    # Trigger attributes
    TRIGGER_TYPE = "trigger.type"
    TRIGGER_ID = "trigger.id"

    # Error attributes
    ERROR_TYPE = "error.type"
    ERROR_MESSAGE = "error.message"

    # Performance attributes
    DURATION_MS = "duration.ms"
    RETRY_COUNT = "retry.count"


class WorkflowTracer:
    """
    Tracer specifically for workflow operations.

    Provides convenient methods for tracing workflow lifecycles.
    """

    def __init__(self, tracer=None):
        self._tracer = tracer or get_tracer()
        self._active_spans: Dict[str, Any] = {}

    @contextmanager
    def trace_workflow_execution(
        self,
        execution_id: str,
        workflow_id: str,
        workflow_name: str,
        inputs: Optional[Dict[str, Any]] = None,
    ) -> Generator[Any, None, None]:
        """
        Trace an entire workflow execution.

        Creates a root span for the workflow.
        """
        span = self._tracer.start_span(
            f"workflow.execute.{workflow_name}",
            attributes={
                SpanAttributes.EXECUTION_ID: execution_id,
                SpanAttributes.WORKFLOW_ID: workflow_id,
                SpanAttributes.WORKFLOW_NAME: workflow_name,
            },
        )

        # Store for child span creation
        self._active_spans[execution_id] = span

        try:
            with span:
                if inputs:
                    span.add_event("workflow.inputs", {"inputs": str(inputs)[:1000]})
                yield span

        except Exception as e:
            span.record_exception(e)
            span.set_attribute(SpanAttributes.ERROR_TYPE, type(e).__name__)
            span.set_attribute(SpanAttributes.ERROR_MESSAGE, str(e))
            raise

        finally:
            self._active_spans.pop(execution_id, None)

    @contextmanager
    def trace_step(
        self,
        execution_id: str,
        step_id: str,
        step_name: str,
        step_type: str,
        step_index: int = 0,
    ) -> Generator[Any, None, None]:
        """Trace a workflow step."""
        parent_span = self._active_spans.get(execution_id)

        span = self._tracer.start_span(
            f"step.execute.{step_name}",
            attributes={
                SpanAttributes.EXECUTION_ID: execution_id,
                SpanAttributes.STEP_ID: step_id,
                SpanAttributes.STEP_NAME: step_name,
                SpanAttributes.STEP_TYPE: step_type,
                SpanAttributes.STEP_INDEX: step_index,
            },
        )

        start_time = time.time()

        try:
            with span:
                yield span

        except Exception as e:
            span.record_exception(e)
            span.set_attribute(SpanAttributes.ERROR_TYPE, type(e).__name__)
            span.set_attribute(SpanAttributes.ERROR_MESSAGE, str(e))
            raise

        finally:
            duration_ms = (time.time() - start_time) * 1000
            span.set_attribute(SpanAttributes.DURATION_MS, duration_ms)

    @contextmanager
    def trace_action(
        self,
        execution_id: str,
        step_id: str,
        action_type: str,
        action_name: Optional[str] = None,
    ) -> Generator[Any, None, None]:
        """Trace an action execution."""
        span = self._tracer.start_span(
            f"action.execute.{action_type}",
            attributes={
                SpanAttributes.EXECUTION_ID: execution_id,
                SpanAttributes.STEP_ID: step_id,
                SpanAttributes.ACTION_TYPE: action_type,
                SpanAttributes.ACTION_NAME: action_name or action_type,
            },
        )

        start_time = time.time()

        try:
            with span:
                yield span

        except Exception as e:
            span.record_exception(e)
            raise

        finally:
            duration_ms = (time.time() - start_time) * 1000
            span.set_attribute(SpanAttributes.DURATION_MS, duration_ms)

    @contextmanager
    def trace_trigger(
        self,
        trigger_type: str,
        trigger_id: str,
        workflow_id: str,
    ) -> Generator[Any, None, None]:
        """Trace a trigger activation."""
        span = self._tracer.start_span(
            f"trigger.{trigger_type}",
            attributes={
                SpanAttributes.TRIGGER_TYPE: trigger_type,
                SpanAttributes.TRIGGER_ID: trigger_id,
                SpanAttributes.WORKFLOW_ID: workflow_id,
            },
        )

        try:
            with span:
                yield span

        except Exception as e:
            span.record_exception(e)
            raise

    @contextmanager
    def trace_approval(
        self,
        execution_id: str,
        step_id: str,
        approvers: List[str],
    ) -> Generator[Any, None, None]:
        """Trace an approval gate."""
        span = self._tracer.start_span(
            "approval.wait",
            attributes={
                SpanAttributes.EXECUTION_ID: execution_id,
                SpanAttributes.STEP_ID: step_id,
                "approval.approvers": ",".join(approvers),
            },
        )

        try:
            with span:
                yield span

        except Exception as e:
            span.record_exception(e)
            raise

    def add_event(
        self,
        execution_id: str,
        event_name: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add an event to the current workflow span."""
        span = self._active_spans.get(execution_id)
        if span:
            span.add_event(event_name, attributes or {})


# Decorators for tracing

def trace_workflow(
    workflow_id: Optional[str] = None,
    workflow_name: Optional[str] = None,
):
    """Decorator to trace a workflow execution function."""

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            tracer = get_tracer()

            # Try to get execution_id from arguments
            exec_id = kwargs.get("execution_id") or (args[0] if args else "unknown")
            wf_id = workflow_id or kwargs.get("workflow_id", "unknown")
            wf_name = workflow_name or func.__name__

            span = tracer.start_span(
                f"workflow.execute.{wf_name}",
                attributes={
                    SpanAttributes.EXECUTION_ID: str(exec_id),
                    SpanAttributes.WORKFLOW_ID: str(wf_id),
                    SpanAttributes.WORKFLOW_NAME: wf_name,
                },
            )

            try:
                with span:
                    return await func(*args, **kwargs)
            except Exception as e:
                span.record_exception(e)
                raise

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            tracer = get_tracer()

            exec_id = kwargs.get("execution_id") or (args[0] if args else "unknown")
            wf_id = workflow_id or kwargs.get("workflow_id", "unknown")
            wf_name = workflow_name or func.__name__

            span = tracer.start_span(
                f"workflow.execute.{wf_name}",
                attributes={
                    SpanAttributes.EXECUTION_ID: str(exec_id),
                    SpanAttributes.WORKFLOW_ID: str(wf_id),
                    SpanAttributes.WORKFLOW_NAME: wf_name,
                },
            )

            try:
                with span:
                    return func(*args, **kwargs)
            except Exception as e:
                span.record_exception(e)
                raise

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def trace_step(
    step_id: Optional[str] = None,
    step_name: Optional[str] = None,
    step_type: str = "unknown",
):
    """Decorator to trace a step execution function."""

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            tracer = get_tracer()

            s_id = step_id or kwargs.get("step_id", "unknown")
            s_name = step_name or func.__name__
            exec_id = kwargs.get("execution_id", "unknown")

            span = tracer.start_span(
                f"step.execute.{s_name}",
                attributes={
                    SpanAttributes.STEP_ID: str(s_id),
                    SpanAttributes.STEP_NAME: s_name,
                    SpanAttributes.STEP_TYPE: step_type,
                    SpanAttributes.EXECUTION_ID: str(exec_id),
                },
            )

            start_time = time.time()

            try:
                with span:
                    return await func(*args, **kwargs)
            except Exception as e:
                span.record_exception(e)
                raise
            finally:
                duration_ms = (time.time() - start_time) * 1000
                span.set_attribute(SpanAttributes.DURATION_MS, duration_ms)

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            tracer = get_tracer()

            s_id = step_id or kwargs.get("step_id", "unknown")
            s_name = step_name or func.__name__
            exec_id = kwargs.get("execution_id", "unknown")

            span = tracer.start_span(
                f"step.execute.{s_name}",
                attributes={
                    SpanAttributes.STEP_ID: str(s_id),
                    SpanAttributes.STEP_NAME: s_name,
                    SpanAttributes.STEP_TYPE: step_type,
                    SpanAttributes.EXECUTION_ID: str(exec_id),
                },
            )

            start_time = time.time()

            try:
                with span:
                    return func(*args, **kwargs)
            except Exception as e:
                span.record_exception(e)
                raise
            finally:
                duration_ms = (time.time() - start_time) * 1000
                span.set_attribute(SpanAttributes.DURATION_MS, duration_ms)

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def trace_action(action_type: str = "unknown"):
    """Decorator to trace an action execution function."""

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            tracer = get_tracer()

            span = tracer.start_span(
                f"action.execute.{action_type}",
                attributes={
                    SpanAttributes.ACTION_TYPE: action_type,
                    SpanAttributes.ACTION_NAME: func.__name__,
                },
            )

            start_time = time.time()

            try:
                with span:
                    return await func(*args, **kwargs)
            except Exception as e:
                span.record_exception(e)
                raise
            finally:
                duration_ms = (time.time() - start_time) * 1000
                span.set_attribute(SpanAttributes.DURATION_MS, duration_ms)

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            tracer = get_tracer()

            span = tracer.start_span(
                f"action.execute.{action_type}",
                attributes={
                    SpanAttributes.ACTION_TYPE: action_type,
                    SpanAttributes.ACTION_NAME: func.__name__,
                },
            )

            start_time = time.time()

            try:
                with span:
                    return func(*args, **kwargs)
            except Exception as e:
                span.record_exception(e)
                raise
            finally:
                duration_ms = (time.time() - start_time) * 1000
                span.set_attribute(SpanAttributes.DURATION_MS, duration_ms)

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


class TracingContext:
    """
    Context for propagating trace information.

    Used for distributed tracing across service boundaries.
    """

    def __init__(self):
        self._context: Dict[str, str] = {}

    def inject(self) -> Dict[str, str]:
        """Inject trace context into headers/metadata."""
        try:
            from opentelemetry import trace
            from opentelemetry.propagate import inject

            inject(self._context)
            return self._context

        except ImportError:
            return {}

    def extract(self, carrier: Dict[str, str]) -> None:
        """Extract trace context from headers/metadata."""
        try:
            from opentelemetry.propagate import extract

            self._context = extract(carrier)

        except ImportError:
            pass

    @property
    def trace_id(self) -> Optional[str]:
        """Get the current trace ID."""
        try:
            from opentelemetry import trace
            span = trace.get_current_span()
            if span:
                ctx = span.get_span_context()
                if ctx.is_valid:
                    return format(ctx.trace_id, "032x")
        except ImportError:
            pass
        return None

    @property
    def span_id(self) -> Optional[str]:
        """Get the current span ID."""
        try:
            from opentelemetry import trace
            span = trace.get_current_span()
            if span:
                ctx = span.get_span_context()
                if ctx.is_valid:
                    return format(ctx.span_id, "016x")
        except ImportError:
            pass
        return None


# Singleton tracer instance
_workflow_tracer: Optional[WorkflowTracer] = None


def get_workflow_tracer() -> WorkflowTracer:
    """Get the global workflow tracer instance."""
    global _workflow_tracer
    if _workflow_tracer is None:
        _workflow_tracer = WorkflowTracer()
    return _workflow_tracer
