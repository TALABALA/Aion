"""
AION Instrumentation Decorators

Decorators for automatic observability:
- @traced - Add distributed tracing
- @metered - Record metrics
- @logged - Add structured logging
- @profiled - Performance profiling
- @observable - All-in-one observability
"""

from __future__ import annotations

import asyncio
import functools
import time
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

import structlog

from aion.observability.types import SpanKind, SpanStatus, LogLevel

logger = structlog.get_logger(__name__)

T = TypeVar('T')


def traced(
    name: Optional[str] = None,
    kind: SpanKind = SpanKind.INTERNAL,
    attributes: Dict[str, Any] = None,
    record_exception: bool = True,
    record_args: bool = False,
    record_return: bool = False,
):
    """
    Decorator to add distributed tracing to a function.

    Args:
        name: Span name (defaults to function name)
        kind: Span kind
        attributes: Static attributes to add
        record_exception: Whether to record exceptions
        record_args: Whether to record function arguments
        record_return: Whether to record return value

    Usage:
        @traced("process_request", kind=SpanKind.SERVER)
        async def process_request(request):
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        span_name = name or func.__name__
        static_attrs = attributes or {}

        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                from aion.observability import get_tracing_engine
                engine = get_tracing_engine()
                if not engine:
                    return await func(*args, **kwargs)

                attrs = {
                    "code.function": func.__name__,
                    "code.filepath": func.__code__.co_filename,
                    **static_attrs,
                }

                if record_args and args:
                    attrs["function.args_count"] = len(args)

                span = engine.start_span(span_name, kind, attributes=attrs)

                try:
                    result = await func(*args, **kwargs)

                    if record_return and result is not None:
                        span.set_attribute("function.return_type", type(result).__name__)

                    engine.end_span(span, status=SpanStatus.OK)
                    return result
                except Exception as e:
                    if record_exception:
                        engine.record_exception(span, e)
                    engine.end_span(span, status=SpanStatus.ERROR, status_message=str(e))
                    raise

            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                from aion.observability import get_tracing_engine
                engine = get_tracing_engine()
                if not engine:
                    return func(*args, **kwargs)

                attrs = {
                    "code.function": func.__name__,
                    "code.filepath": func.__code__.co_filename,
                    **static_attrs,
                }

                if record_args and args:
                    attrs["function.args_count"] = len(args)

                span = engine.start_span(span_name, kind, attributes=attrs)

                try:
                    result = func(*args, **kwargs)

                    if record_return and result is not None:
                        span.set_attribute("function.return_type", type(result).__name__)

                    engine.end_span(span, status=SpanStatus.OK)
                    return result
                except Exception as e:
                    if record_exception:
                        engine.record_exception(span, e)
                    engine.end_span(span, status=SpanStatus.ERROR, status_message=str(e))
                    raise

            return sync_wrapper

    return decorator


def metered(
    name: Optional[str] = None,
    labels: Dict[str, str] = None,
    record_duration: bool = True,
    record_count: bool = True,
    record_errors: bool = True,
):
    """
    Decorator to record metrics for a function.

    Args:
        name: Base metric name (defaults to function name)
        labels: Static labels to add
        record_duration: Whether to record duration histogram
        record_count: Whether to record call count
        record_errors: Whether to record error count

    Usage:
        @metered("api_request", labels={"endpoint": "/users"})
        async def get_users():
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        metric_name = name or func.__name__
        static_labels = labels or {}

        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                from aion.observability import get_metrics_engine
                engine = get_metrics_engine()

                start_time = time.perf_counter()
                error = False

                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception:
                    error = True
                    raise
                finally:
                    if engine:
                        duration = time.perf_counter() - start_time

                        if record_duration:
                            engine.observe(
                                f"{metric_name}_duration_seconds",
                                duration,
                                static_labels,
                            )

                        if record_count:
                            engine.inc(
                                f"{metric_name}_total",
                                1.0,
                                {**static_labels, "status": "error" if error else "success"},
                            )

                        if record_errors and error:
                            engine.inc(
                                f"{metric_name}_errors_total",
                                1.0,
                                static_labels,
                            )

            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                from aion.observability import get_metrics_engine
                engine = get_metrics_engine()

                start_time = time.perf_counter()
                error = False

                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception:
                    error = True
                    raise
                finally:
                    if engine:
                        duration = time.perf_counter() - start_time

                        if record_duration:
                            engine.observe(
                                f"{metric_name}_duration_seconds",
                                duration,
                                static_labels,
                            )

                        if record_count:
                            engine.inc(
                                f"{metric_name}_total",
                                1.0,
                                {**static_labels, "status": "error" if error else "success"},
                            )

                        if record_errors and error:
                            engine.inc(
                                f"{metric_name}_errors_total",
                                1.0,
                                static_labels,
                            )

            return sync_wrapper

    return decorator


def logged(
    level: LogLevel = LogLevel.INFO,
    log_args: bool = False,
    log_result: bool = False,
    log_duration: bool = True,
):
    """
    Decorator to add structured logging to a function.

    Args:
        level: Log level for entry/exit logs
        log_args: Whether to log function arguments
        log_result: Whether to log return value
        log_duration: Whether to log execution duration

    Usage:
        @logged(level=LogLevel.DEBUG, log_duration=True)
        async def process_item(item):
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        func_name = func.__name__
        func_module = func.__module__

        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                log = structlog.get_logger(func_module)

                log_kwargs = {"function": func_name}
                if log_args:
                    log_kwargs["args_count"] = len(args)
                    log_kwargs["kwargs_keys"] = list(kwargs.keys())

                log.log(level.value, f"Entering {func_name}", **log_kwargs)

                start_time = time.perf_counter()
                try:
                    result = await func(*args, **kwargs)

                    exit_kwargs = {"function": func_name}
                    if log_duration:
                        exit_kwargs["duration_ms"] = (time.perf_counter() - start_time) * 1000
                    if log_result and result is not None:
                        exit_kwargs["result_type"] = type(result).__name__

                    log.log(level.value, f"Exiting {func_name}", **exit_kwargs)
                    return result
                except Exception as e:
                    log.error(
                        f"Error in {func_name}",
                        function=func_name,
                        error=str(e),
                        duration_ms=(time.perf_counter() - start_time) * 1000 if log_duration else None,
                    )
                    raise

            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                log = structlog.get_logger(func_module)

                log_kwargs = {"function": func_name}
                if log_args:
                    log_kwargs["args_count"] = len(args)

                log.log(level.value, f"Entering {func_name}", **log_kwargs)

                start_time = time.perf_counter()
                try:
                    result = func(*args, **kwargs)

                    exit_kwargs = {"function": func_name}
                    if log_duration:
                        exit_kwargs["duration_ms"] = (time.perf_counter() - start_time) * 1000
                    if log_result and result is not None:
                        exit_kwargs["result_type"] = type(result).__name__

                    log.log(level.value, f"Exiting {func_name}", **exit_kwargs)
                    return result
                except Exception as e:
                    log.error(
                        f"Error in {func_name}",
                        function=func_name,
                        error=str(e),
                    )
                    raise

            return sync_wrapper

    return decorator


def profiled(name: Optional[str] = None):
    """
    Decorator to add performance profiling to a function.

    Args:
        name: Profile name (defaults to function name)

    Usage:
        @profiled("process_heavy")
        async def process_heavy_operation():
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        profile_name = name or func.__name__

        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                from aion.observability import get_profiler
                profiler = get_profiler()

                if profiler:
                    with profiler.profile_operation(profile_name):
                        return await func(*args, **kwargs)
                return await func(*args, **kwargs)

            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                from aion.observability import get_profiler
                profiler = get_profiler()

                if profiler:
                    with profiler.profile_operation(profile_name):
                        return func(*args, **kwargs)
                return func(*args, **kwargs)

            return sync_wrapper

    return decorator


def observable(
    name: Optional[str] = None,
    span_kind: SpanKind = SpanKind.INTERNAL,
    labels: Dict[str, str] = None,
    log_level: LogLevel = LogLevel.DEBUG,
):
    """
    All-in-one observability decorator.

    Combines tracing, metrics, logging, and profiling.

    Args:
        name: Operation name
        span_kind: Span kind for tracing
        labels: Metric labels
        log_level: Log level

    Usage:
        @observable("process_request", span_kind=SpanKind.SERVER)
        async def process_request(request):
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        op_name = name or func.__name__
        metric_labels = labels or {}

        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                from aion.observability import (
                    get_tracing_engine,
                    get_metrics_engine,
                    get_profiler,
                )

                log = structlog.get_logger(func.__module__)
                tracing = get_tracing_engine()
                metrics = get_metrics_engine()
                profiler = get_profiler()

                # Start span
                span = None
                if tracing:
                    span = tracing.start_span(op_name, span_kind, attributes={
                        "code.function": func.__name__,
                    })

                log.log(log_level.value, f"Starting {op_name}")
                start_time = time.perf_counter()
                error = False

                try:
                    if profiler:
                        with profiler.profile_operation(op_name):
                            result = await func(*args, **kwargs)
                    else:
                        result = await func(*args, **kwargs)

                    return result
                except Exception as e:
                    error = True
                    if span:
                        tracing.record_exception(span, e)
                    log.error(f"Error in {op_name}", error=str(e))
                    raise
                finally:
                    duration = time.perf_counter() - start_time

                    if span:
                        tracing.end_span(
                            span,
                            status=SpanStatus.ERROR if error else SpanStatus.OK,
                        )

                    if metrics:
                        metrics.observe(
                            f"{op_name}_duration_seconds",
                            duration,
                            metric_labels,
                        )
                        metrics.inc(
                            f"{op_name}_total",
                            1.0,
                            {**metric_labels, "status": "error" if error else "success"},
                        )

                    log.log(
                        log_level.value,
                        f"Completed {op_name}",
                        duration_ms=duration * 1000,
                        error=error,
                    )

            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                from aion.observability import (
                    get_tracing_engine,
                    get_metrics_engine,
                    get_profiler,
                )

                log = structlog.get_logger(func.__module__)
                tracing = get_tracing_engine()
                metrics = get_metrics_engine()
                profiler = get_profiler()

                span = None
                if tracing:
                    span = tracing.start_span(op_name, span_kind, attributes={
                        "code.function": func.__name__,
                    })

                log.log(log_level.value, f"Starting {op_name}")
                start_time = time.perf_counter()
                error = False

                try:
                    if profiler:
                        with profiler.profile_operation(op_name):
                            result = func(*args, **kwargs)
                    else:
                        result = func(*args, **kwargs)

                    return result
                except Exception as e:
                    error = True
                    if span:
                        tracing.record_exception(span, e)
                    log.error(f"Error in {op_name}", error=str(e))
                    raise
                finally:
                    duration = time.perf_counter() - start_time

                    if span:
                        tracing.end_span(
                            span,
                            status=SpanStatus.ERROR if error else SpanStatus.OK,
                        )

                    if metrics:
                        metrics.observe(
                            f"{op_name}_duration_seconds",
                            duration,
                            metric_labels,
                        )
                        metrics.inc(
                            f"{op_name}_total",
                            1.0,
                            {**metric_labels, "status": "error" if error else "success"},
                        )

            return sync_wrapper

    return decorator


def with_cost_tracking(
    model: str = "default",
    input_tokens_key: str = "input_tokens",
    output_tokens_key: str = "output_tokens",
):
    """
    Decorator to track costs for LLM operations.

    Expects the function to return a dict with token counts.

    Args:
        model: Model name for pricing
        input_tokens_key: Key for input tokens in result
        output_tokens_key: Key for output tokens in result

    Usage:
        @with_cost_tracking(model="claude-3-sonnet")
        async def call_llm(prompt):
            result = await client.complete(prompt)
            return {
                "response": result.text,
                "input_tokens": result.usage.input_tokens,
                "output_tokens": result.usage.output_tokens,
            }
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                from aion.observability import get_cost_tracker
                from aion.observability.context import get_context_manager

                result = await func(*args, **kwargs)

                if isinstance(result, dict):
                    input_tokens = result.get(input_tokens_key, 0)
                    output_tokens = result.get(output_tokens_key, 0)

                    if input_tokens or output_tokens:
                        cost_tracker = get_cost_tracker()
                        if cost_tracker:
                            ctx = get_context_manager()
                            cost_tracker.record_tokens(
                                model=model,
                                input_tokens=input_tokens,
                                output_tokens=output_tokens,
                                trace_id=ctx.get_trace_id(),
                                agent_id=ctx.get_agent_id(),
                            )

                return result

            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                from aion.observability import get_cost_tracker
                from aion.observability.context import get_context_manager

                result = func(*args, **kwargs)

                if isinstance(result, dict):
                    input_tokens = result.get(input_tokens_key, 0)
                    output_tokens = result.get(output_tokens_key, 0)

                    if input_tokens or output_tokens:
                        cost_tracker = get_cost_tracker()
                        if cost_tracker:
                            ctx = get_context_manager()
                            cost_tracker.record_tokens(
                                model=model,
                                input_tokens=input_tokens,
                                output_tokens=output_tokens,
                                trace_id=ctx.get_trace_id(),
                                agent_id=ctx.get_agent_id(),
                            )

                return result

            return sync_wrapper

    return decorator
