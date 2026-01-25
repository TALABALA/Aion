"""
AION Observability Middleware

FastAPI middleware for automatic request observability.
"""

from __future__ import annotations

import time
import uuid
from typing import Callable, Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

import structlog

from aion.observability.types import SpanKind, SpanStatus
from aion.observability.context import get_context_manager

logger = structlog.get_logger(__name__)


class ObservabilityMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for automatic observability.

    Features:
    - Request/response tracing
    - Metrics collection
    - Structured logging
    - Context propagation
    """

    def __init__(
        self,
        app: ASGIApp,
        service_name: str = "aion",
        exclude_paths: list = None,
        record_request_body: bool = False,
        record_response_body: bool = False,
    ):
        super().__init__(app)
        self.service_name = service_name
        self.exclude_paths = exclude_paths or ["/health", "/metrics", "/favicon.ico"]
        self.record_request_body = record_request_body
        self.record_response_body = record_response_body

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """Handle request with observability."""
        # Skip excluded paths
        if request.url.path in self.exclude_paths:
            return await call_next(request)

        ctx = get_context_manager()

        # Extract trace context from headers
        headers = dict(request.headers)
        ctx.extract_headers(headers)

        # Generate request ID if not present
        request_id = headers.get("x-request-id") or uuid.uuid4().hex[:16]
        ctx.set_request_id(request_id)

        # Ensure trace ID
        ctx.ensure_trace_id()

        # Start span
        from aion.observability import get_tracing_engine, get_metrics_engine
        tracing = get_tracing_engine()
        metrics = get_metrics_engine()

        span = None
        if tracing:
            span = tracing.start_span(
                name=f"{request.method} {request.url.path}",
                kind=SpanKind.SERVER,
                attributes={
                    "http.method": request.method,
                    "http.url": str(request.url),
                    "http.route": request.url.path,
                    "http.scheme": request.url.scheme,
                    "http.host": request.url.hostname,
                    "http.user_agent": request.headers.get("user-agent", ""),
                    "http.request_id": request_id,
                    "net.peer.ip": request.client.host if request.client else "",
                },
            )

        start_time = time.perf_counter()
        status_code = 500
        error = None

        try:
            response = await call_next(request)
            status_code = response.status_code
            return response

        except Exception as e:
            error = e
            if span:
                tracing.record_exception(span, e)
            raise

        finally:
            duration = time.perf_counter() - start_time

            # End span
            if span:
                span.set_attribute("http.status_code", status_code)
                tracing.end_span(
                    span,
                    status=SpanStatus.ERROR if status_code >= 500 else SpanStatus.OK,
                )

            # Record metrics
            if metrics:
                labels = {
                    "method": request.method,
                    "endpoint": request.url.path,
                    "status": str(status_code),
                }

                metrics.inc("aion_requests_total", 1.0, labels)
                metrics.observe("aion_request_duration_seconds", duration, {
                    "method": request.method,
                    "endpoint": request.url.path,
                })

                if status_code >= 500:
                    metrics.inc("aion_errors_total", 1.0, {
                        "component": "http",
                        "error_type": "server_error",
                    })

            # Log request
            log_ctx = {
                "method": request.method,
                "path": request.url.path,
                "status_code": status_code,
                "duration_ms": duration * 1000,
                "request_id": request_id,
                "trace_id": ctx.get_trace_id(),
            }

            if error:
                logger.error("Request failed", error=str(error), **log_ctx)
            elif status_code >= 500:
                logger.error("Request error", **log_ctx)
            elif status_code >= 400:
                logger.warning("Request client error", **log_ctx)
            else:
                logger.info("Request completed", **log_ctx)


def create_observability_middleware(
    service_name: str = "aion",
    exclude_paths: list = None,
) -> Callable:
    """
    Create observability middleware for FastAPI.

    Usage:
        from fastapi import FastAPI
        from aion.observability.instrumentation import create_observability_middleware

        app = FastAPI()
        app.add_middleware(create_observability_middleware())
    """
    def middleware_factory(app: ASGIApp) -> ObservabilityMiddleware:
        return ObservabilityMiddleware(
            app,
            service_name=service_name,
            exclude_paths=exclude_paths or [],
        )

    return middleware_factory


# WebSocket middleware
class WebSocketObservabilityMiddleware:
    """
    Middleware for WebSocket observability.
    """

    def __init__(self, app: ASGIApp, service_name: str = "aion"):
        self.app = app
        self.service_name = service_name

    async def __call__(self, scope, receive, send):
        if scope["type"] != "websocket":
            await self.app(scope, receive, send)
            return

        ctx = get_context_manager()
        ctx.ensure_trace_id()

        # Extract any headers
        headers = dict(scope.get("headers", []))
        ctx.extract_headers({
            k.decode() if isinstance(k, bytes) else k: v.decode() if isinstance(v, bytes) else v
            for k, v in headers.items()
        })

        from aion.observability import get_tracing_engine, get_metrics_engine
        tracing = get_tracing_engine()
        metrics = get_metrics_engine()

        path = scope.get("path", "/")
        span = None

        if tracing:
            span = tracing.start_span(
                name=f"WebSocket {path}",
                kind=SpanKind.SERVER,
                attributes={
                    "websocket.path": path,
                    "websocket.scheme": scope.get("scheme", "ws"),
                },
            )

        start_time = time.perf_counter()
        message_count = 0
        error = None

        async def receive_wrapper():
            nonlocal message_count
            message = await receive()
            if message["type"] == "websocket.receive":
                message_count += 1
                if span:
                    span.add_event("message_received", {"count": message_count})
            return message

        async def send_wrapper(message):
            if message["type"] == "websocket.send":
                if span:
                    span.add_event("message_sent")
            await send(message)

        try:
            await self.app(scope, receive_wrapper, send_wrapper)
        except Exception as e:
            error = e
            if span:
                tracing.record_exception(span, e)
            raise
        finally:
            duration = time.perf_counter() - start_time

            if span:
                span.set_attribute("websocket.messages_count", message_count)
                tracing.end_span(
                    span,
                    status=SpanStatus.ERROR if error else SpanStatus.OK,
                )

            if metrics:
                metrics.inc("aion_websocket_connections_total", 1.0, {"path": path})
                metrics.observe("aion_websocket_duration_seconds", duration, {"path": path})
                metrics.inc("aion_websocket_messages_total", float(message_count), {"path": path})

            logger.info(
                "WebSocket closed",
                path=path,
                messages=message_count,
                duration_ms=duration * 1000,
                error=str(error) if error else None,
            )
