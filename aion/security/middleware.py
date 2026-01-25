"""
AION Security Middleware

FastAPI middleware for authentication, authorization, and rate limiting.
"""

from __future__ import annotations

import time
from functools import wraps
from typing import Any, Callable, List, Optional, Union

from fastapi import Depends, HTTPException, Request, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader
from starlette.middleware.base import BaseHTTPMiddleware

import structlog

from aion.security.types import (
    AuthMethod,
    Credentials,
    PermissionAction,
    SecurityContext,
)
from aion.security.manager import get_security_manager, SecurityManager

logger = structlog.get_logger(__name__)


# FastAPI security schemes
bearer_scheme = HTTPBearer(auto_error=False)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


class SecurityMiddleware(BaseHTTPMiddleware):
    """
    Security middleware for FastAPI.

    Features:
    - Automatic authentication from headers
    - Rate limiting
    - Request timing
    - Security context injection
    - CORS handling
    """

    def __init__(
        self,
        app,
        security_manager: Optional[SecurityManager] = None,
        exclude_paths: Optional[List[str]] = None,
        require_auth_paths: Optional[List[str]] = None,
    ):
        super().__init__(app)
        self._security = security_manager
        self.exclude_paths = exclude_paths or [
            "/health",
            "/metrics",
            "/docs",
            "/openapi.json",
            "/redoc",
            "/favicon.ico",
        ]
        self.require_auth_paths = require_auth_paths or []

    @property
    def security(self) -> SecurityManager:
        """Get security manager (lazy load)."""
        if self._security is None:
            self._security = get_security_manager()
        return self._security

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request through security middleware."""
        start_time = time.time()

        # Skip excluded paths
        if self._is_excluded(request.url.path):
            return await call_next(request)

        # Extract client info
        ip_address = self._get_client_ip(request)
        user_agent = request.headers.get("user-agent", "")

        # Check rate limit (by IP for unauthenticated)
        rate_key = f"ip:{ip_address}"
        rate_result = await self.security.check_rate_limit(rate_key)

        if not rate_result.allowed:
            return Response(
                content='{"error": "Rate limit exceeded"}',
                status_code=429,
                media_type="application/json",
                headers={
                    "Retry-After": str(int(rate_result.retry_after_seconds or 60)),
                    **rate_result.to_headers(),
                },
            )

        # Extract credentials
        credentials = await self._extract_credentials(request)

        # Authenticate if credentials present
        context = None
        if credentials:
            result = await self.security.authenticate(
                credentials,
                ip_address=ip_address,
                user_agent=user_agent,
            )

            if result.status.value == "success":
                context = result.context

                # Update rate limit key to user-based
                rate_key = f"user:{context.user_id}"

        # Check if authentication is required
        if self._requires_auth(request.url.path) and not context:
            return Response(
                content='{"error": "Authentication required"}',
                status_code=401,
                media_type="application/json",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Store context in request state
        request.state.security_context = context
        request.state.rate_limit_key = rate_key

        # Process request
        try:
            response = await call_next(request)
        except PermissionError as e:
            return Response(
                content=f'{{"error": "{str(e)}"}}',
                status_code=403,
                media_type="application/json",
            )

        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"

        # Add rate limit headers
        rate_status = await self.security.get_rate_limit_status(rate_key)
        if "fixed_windows" in rate_status:
            minute_info = rate_status["fixed_windows"]["minute"]
            response.headers["X-RateLimit-Limit"] = str(minute_info["limit"])
            response.headers["X-RateLimit-Remaining"] = str(minute_info["remaining"])

        # Add timing header
        duration_ms = (time.time() - start_time) * 1000
        response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"

        return response

    async def _extract_credentials(self, request: Request) -> Optional[Credentials]:
        """Extract credentials from request."""
        # Try Authorization header
        auth_header = request.headers.get("authorization", "")

        if auth_header.startswith("Bearer "):
            token = auth_header[7:]

            # Check if JWT or API key
            if token.count(".") == 2:
                return Credentials(method=AuthMethod.JWT, token=token)
            elif token.startswith("aion_"):
                return Credentials(method=AuthMethod.API_KEY, api_key=token)
            else:
                # Could be session token
                return Credentials(method=AuthMethod.SESSION, session_id=token)

        # Try X-API-Key header
        api_key = request.headers.get("x-api-key")
        if api_key:
            return Credentials(method=AuthMethod.API_KEY, api_key=api_key)

        # Try session cookie
        session_id = request.cookies.get("session_id")
        if session_id:
            return Credentials(method=AuthMethod.SESSION, session_id=session_id)

        return None

    def _is_excluded(self, path: str) -> bool:
        """Check if path is excluded from security."""
        for excluded in self.exclude_paths:
            if path.startswith(excluded):
                return True
        return False

    def _requires_auth(self, path: str) -> bool:
        """Check if path requires authentication."""
        if not self.require_auth_paths:
            return False

        for required in self.require_auth_paths:
            if path.startswith(required):
                return True
        return False

    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address."""
        # Check forwarded headers
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip

        if request.client:
            return request.client.host

        return "unknown"


# =============================================================================
# Dependency Injection Helpers
# =============================================================================


async def get_security_context(request: Request) -> Optional[SecurityContext]:
    """Get security context from request (optional)."""
    return getattr(request.state, "security_context", None)


async def require_auth(request: Request) -> SecurityContext:
    """Require authenticated user."""
    context = getattr(request.state, "security_context", None)

    if not context or not context.is_authenticated():
        raise HTTPException(
            status_code=401,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return context


async def require_admin(request: Request) -> SecurityContext:
    """Require admin role."""
    context = await require_auth(request)

    if "admin" not in context.roles:
        raise HTTPException(
            status_code=403,
            detail="Admin access required",
        )

    return context


def require_permission(
    resource: str,
    action: Union[str, PermissionAction],
):
    """
    Dependency to require a specific permission.

    Usage:
        @app.get("/agents")
        async def list_agents(
            context: SecurityContext = Depends(require_permission("agents", "read"))
        ):
            ...
    """
    if isinstance(action, str):
        action = PermissionAction(action)

    async def check_permission(request: Request) -> SecurityContext:
        context = await require_auth(request)
        security = get_security_manager()

        allowed = await security.authorize(context, resource, action)

        if not allowed:
            raise HTTPException(
                status_code=403,
                detail=f"Permission denied: {resource}:{action.value}",
            )

        return context

    return Depends(check_permission)


def require_role(role: str):
    """
    Dependency to require a specific role.

    Usage:
        @app.post("/admin/users")
        async def create_user(
            context: SecurityContext = Depends(require_role("admin"))
        ):
            ...
    """

    async def check_role(request: Request) -> SecurityContext:
        context = await require_auth(request)

        if role not in context.roles:
            raise HTTPException(
                status_code=403,
                detail=f"Role required: {role}",
            )

        return context

    return Depends(check_role)


def require_scope(scope: str):
    """
    Dependency to require an OAuth scope.

    Usage:
        @app.get("/api/data")
        async def get_data(
            context: SecurityContext = Depends(require_scope("read:data"))
        ):
            ...
    """

    async def check_scope(request: Request) -> SecurityContext:
        context = await require_auth(request)

        if not context.has_scope(scope):
            raise HTTPException(
                status_code=403,
                detail=f"Scope required: {scope}",
            )

        return context

    return Depends(check_scope)


# =============================================================================
# Decorators
# =============================================================================


def authenticated(func: Callable) -> Callable:
    """Decorator to require authentication."""

    @wraps(func)
    async def wrapper(request: Request, *args, **kwargs):
        context = getattr(request.state, "security_context", None)

        if not context or not context.is_authenticated():
            raise HTTPException(status_code=401, detail="Authentication required")

        return await func(request, *args, context=context, **kwargs)

    return wrapper


def authorized(resource: str, action: str) -> Callable:
    """Decorator to require authorization."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(request: Request, *args, **kwargs):
            context = getattr(request.state, "security_context", None)

            if not context or not context.is_authenticated():
                raise HTTPException(status_code=401, detail="Authentication required")

            security = get_security_manager()
            allowed = await security.authorize(
                context, resource, PermissionAction(action)
            )

            if not allowed:
                raise HTTPException(
                    status_code=403,
                    detail=f"Permission denied: {resource}:{action}",
                )

            return await func(request, *args, context=context, **kwargs)

        return wrapper

    return decorator


# =============================================================================
# Rate Limiting Decorator
# =============================================================================


def rate_limit(requests_per_minute: int = 60, key_func: Optional[Callable] = None):
    """
    Decorator for custom rate limiting.

    Usage:
        @app.post("/api/expensive")
        @rate_limit(requests_per_minute=10)
        async def expensive_operation():
            ...
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(request: Request, *args, **kwargs):
            security = get_security_manager()

            # Determine rate limit key
            if key_func:
                key = key_func(request)
            else:
                context = getattr(request.state, "security_context", None)
                if context and context.user_id:
                    key = f"endpoint:{func.__name__}:user:{context.user_id}"
                else:
                    ip = request.client.host if request.client else "unknown"
                    key = f"endpoint:{func.__name__}:ip:{ip}"

            # Check rate limit
            from aion.security.types import RateLimitConfig

            security.rate_limiter.set_config(
                f"endpoint:{func.__name__}",
                RateLimitConfig(requests_per_minute=requests_per_minute),
            )

            result = await security.check_rate_limit(key)

            if not result.allowed:
                raise HTTPException(
                    status_code=429,
                    detail="Rate limit exceeded",
                    headers={"Retry-After": str(int(result.retry_after_seconds or 60))},
                )

            return await func(request, *args, **kwargs)

        return wrapper

    return decorator
