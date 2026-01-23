"""
AION Conversation Logging Middleware

Structured logging for conversation events.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from datetime import datetime
from typing import Any, Callable, Optional

from fastapi import Request, Response
import structlog

logger = structlog.get_logger(__name__)


class ConversationLogger:
    """
    Structured logger for conversation events.

    Logs:
    - Request/response details
    - Conversation events
    - Tool executions
    - Errors and warnings
    """

    def __init__(
        self,
        log_requests: bool = True,
        log_responses: bool = True,
        log_body: bool = False,
        max_body_length: int = 1000,
        exclude_paths: Optional[list[str]] = None,
    ):
        self.log_requests = log_requests
        self.log_responses = log_responses
        self.log_body = log_body
        self.max_body_length = max_body_length
        self.exclude_paths = exclude_paths or ["/health", "/ready"]

    async def log_request(self, request: Request) -> dict[str, Any]:
        """Log an incoming request."""
        if not self.log_requests:
            return {}

        if any(request.url.path.startswith(p) for p in self.exclude_paths):
            return {}

        request_id = str(uuid.uuid4())[:8]

        log_data = {
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "query": str(request.query_params) if request.query_params else None,
            "client_ip": self._get_client_ip(request),
            "user_agent": request.headers.get("User-Agent"),
            "timestamp": datetime.now().isoformat(),
        }

        if self.log_body and request.method in ("POST", "PUT", "PATCH"):
            try:
                body = await request.body()
                if body:
                    body_str = body.decode()[:self.max_body_length]
                    if len(body) > self.max_body_length:
                        body_str += "..."
                    log_data["body_preview"] = body_str
            except Exception:
                pass

        logger.info("Request received", **log_data)

        return {"request_id": request_id, "start_time": time.time()}

    async def log_response(
        self,
        request: Request,
        response: Response,
        request_context: dict[str, Any],
    ) -> None:
        """Log a response."""
        if not self.log_responses:
            return

        if any(request.url.path.startswith(p) for p in self.exclude_paths):
            return

        request_id = request_context.get("request_id", "unknown")
        start_time = request_context.get("start_time", time.time())
        duration_ms = (time.time() - start_time) * 1000

        log_data = {
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": round(duration_ms, 2),
        }

        if response.status_code >= 400:
            logger.warning("Request failed", **log_data)
        else:
            logger.info("Request completed", **log_data)

    def log_conversation_event(
        self,
        event_type: str,
        conversation_id: str,
        user_id: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Log a conversation event."""
        log_data = {
            "event_type": event_type,
            "conversation_id": conversation_id[:8] if conversation_id else None,
            "user_id": user_id,
            "timestamp": datetime.now().isoformat(),
            **kwargs,
        }

        logger.info("Conversation event", **log_data)

    def log_tool_execution(
        self,
        tool_name: str,
        conversation_id: str,
        execution_time_ms: float,
        success: bool,
        error: Optional[str] = None,
    ) -> None:
        """Log a tool execution."""
        log_data = {
            "tool_name": tool_name,
            "conversation_id": conversation_id[:8] if conversation_id else None,
            "execution_time_ms": round(execution_time_ms, 2),
            "success": success,
            "timestamp": datetime.now().isoformat(),
        }

        if error:
            log_data["error"] = error
            logger.warning("Tool execution failed", **log_data)
        else:
            logger.info("Tool executed", **log_data)

    def log_memory_operation(
        self,
        operation: str,
        conversation_id: Optional[str] = None,
        count: int = 0,
        latency_ms: float = 0,
    ) -> None:
        """Log a memory operation."""
        logger.info(
            "Memory operation",
            operation=operation,
            conversation_id=conversation_id[:8] if conversation_id else None,
            count=count,
            latency_ms=round(latency_ms, 2),
        )

    def log_error(
        self,
        error: Exception,
        context: Optional[dict[str, Any]] = None,
    ) -> None:
        """Log an error."""
        log_data = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "timestamp": datetime.now().isoformat(),
        }

        if context:
            log_data.update(context)

        logger.error("Error occurred", **log_data)

    def _get_client_ip(self, request: Request) -> str:
        """Get client IP from request."""
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()

        if request.client:
            return request.client.host

        return "unknown"


class LoggingMiddleware:
    """
    FastAPI middleware for request/response logging.
    """

    def __init__(self, conv_logger: Optional[ConversationLogger] = None):
        self.logger = conv_logger or ConversationLogger()

    async def __call__(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """Process request with logging."""
        request_context = await self.logger.log_request(request)

        try:
            response = await call_next(request)
            await self.logger.log_response(request, response, request_context)
            return response

        except Exception as e:
            self.logger.log_error(e, {
                "request_id": request_context.get("request_id"),
                "path": request.url.path,
            })
            raise


class AuditLogger:
    """
    Audit logger for security-relevant events.
    """

    def __init__(self, log_file: Optional[str] = None):
        self.log_file = log_file
        self._audit_logger = structlog.get_logger("audit")

    def log_authentication(
        self,
        user_id: Optional[str],
        success: bool,
        method: str,
        ip_address: str,
        reason: Optional[str] = None,
    ) -> None:
        """Log an authentication event."""
        self._audit_logger.info(
            "Authentication",
            user_id=user_id,
            success=success,
            method=method,
            ip_address=ip_address,
            reason=reason,
            timestamp=datetime.now().isoformat(),
        )

    def log_authorization(
        self,
        user_id: str,
        resource: str,
        action: str,
        allowed: bool,
    ) -> None:
        """Log an authorization decision."""
        self._audit_logger.info(
            "Authorization",
            user_id=user_id,
            resource=resource,
            action=action,
            allowed=allowed,
            timestamp=datetime.now().isoformat(),
        )

    def log_data_access(
        self,
        user_id: str,
        resource_type: str,
        resource_id: str,
        action: str,
    ) -> None:
        """Log a data access event."""
        self._audit_logger.info(
            "Data access",
            user_id=user_id,
            resource_type=resource_type,
            resource_id=resource_id,
            action=action,
            timestamp=datetime.now().isoformat(),
        )


def create_conversation_logger(
    log_requests: bool = True,
    log_responses: bool = True,
    log_body: bool = False,
) -> ConversationLogger:
    """Create a configured conversation logger."""
    return ConversationLogger(
        log_requests=log_requests,
        log_responses=log_responses,
        log_body=log_body,
    )
