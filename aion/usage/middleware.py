"""
AION Usage Limit Middleware

State-of-the-art middleware for:
- Real-time usage limit enforcement
- Tier-based access control
- Graceful limit handling with proper HTTP responses
- Usage tracking decorator for automatic metering
"""

from __future__ import annotations

import asyncio
import functools
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union

from fastapi import Request, Response, HTTPException, Depends
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import structlog

from aion.usage.models import (
    SubscriptionTier,
    UsageMetric,
    UsagePeriod,
    TierLimits,
    get_tier_limits,
    Subscription,
    UsageAlert,
    AlertType,
    AlertSeverity,
)
from aion.usage.tracker import UsageTracker, MemoryUsageTracker
from aion.usage.storage import UsageStore, MemoryUsageStore

logger = structlog.get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


# =============================================================================
# Limit Check Results
# =============================================================================

@dataclass
class LimitCheckResult:
    """Result of a usage limit check."""
    allowed: bool
    metric: UsageMetric
    current: float
    limit: Optional[float]
    percentage: float
    tier: SubscriptionTier

    # For 429 response
    retry_after_seconds: Optional[int] = None
    message: str = ""

    # Warning thresholds
    soft_limit_reached: bool = False
    approaching_limit: bool = False

    def to_headers(self) -> Dict[str, str]:
        """Generate rate limit headers."""
        headers = {
            "X-RateLimit-Metric": self.metric.value,
            "X-RateLimit-Used": str(int(self.current)),
            "X-RateLimit-Remaining": str(int(max(0, (self.limit or 0) - self.current))),
            "X-Usage-Tier": self.tier.value,
        }

        if self.limit:
            headers["X-RateLimit-Limit"] = str(int(self.limit))

        if self.retry_after_seconds:
            headers["Retry-After"] = str(self.retry_after_seconds)

        if self.percentage > 0:
            headers["X-Usage-Percentage"] = f"{self.percentage:.1f}"

        return headers


@dataclass
class UsageContext:
    """Context for tracking usage during request processing."""
    user_id: str
    tier: SubscriptionTier
    limits: TierLimits
    subscription: Optional[Subscription] = None

    # Metrics to track for this request
    metrics_to_track: Dict[UsageMetric, float] = None

    # Request metadata
    session_id: Optional[str] = None
    expert_type: Optional[str] = None
    request_start_time: float = 0.0

    def __post_init__(self):
        if self.metrics_to_track is None:
            self.metrics_to_track = {}
        if self.request_start_time == 0.0:
            self.request_start_time = time.time()

    def track(self, metric: UsageMetric, amount: float = 1.0) -> None:
        """Queue a metric for tracking."""
        if metric not in self.metrics_to_track:
            self.metrics_to_track[metric] = 0.0
        self.metrics_to_track[metric] += amount


# =============================================================================
# Usage Limit Middleware
# =============================================================================

class UsageLimitMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for enforcing usage limits.

    Features:
    - Pre-request limit checking
    - Post-request usage tracking
    - Proper 429 responses with retry headers
    - Soft limit warnings in headers
    - Expert access gating
    """

    def __init__(
        self,
        app,
        tracker: Optional[UsageTracker] = None,
        store: Optional[UsageStore] = None,
        user_resolver: Optional[Callable[[Request], Optional[str]]] = None,
        tier_resolver: Optional[Callable[[str], SubscriptionTier]] = None,
        exclude_paths: Optional[List[str]] = None,
        enable_tracking: bool = True,
        enable_enforcement: bool = True,
    ):
        super().__init__(app)
        self.tracker = tracker or MemoryUsageTracker()
        self.store = store or MemoryUsageStore()
        self.user_resolver = user_resolver or self._default_user_resolver
        self.tier_resolver = tier_resolver
        self.exclude_paths = exclude_paths or [
            "/health",
            "/ready",
            "/docs",
            "/openapi.json",
            "/api/v1/usage",  # Don't limit usage endpoints
        ]
        self.enable_tracking = enable_tracking
        self.enable_enforcement = enable_enforcement

    def _default_user_resolver(self, request: Request) -> Optional[str]:
        """Default user resolver from request state or headers."""
        # Try request state (set by auth middleware)
        if hasattr(request.state, "user") and request.state.user:
            return request.state.user.id

        # Try header
        user_id = request.headers.get("X-User-ID")
        if user_id:
            return user_id

        # Try query param (for API keys)
        user_id = request.query_params.get("user_id")
        if user_id:
            return user_id

        return None

    async def _get_tier(self, user_id: str) -> SubscriptionTier:
        """Get user's subscription tier."""
        if self.tier_resolver:
            return self.tier_resolver(user_id)

        # Try to get from store
        subscription = await self.store.get_subscription(user_id)
        if subscription and subscription.is_active:
            return subscription.tier

        return SubscriptionTier.FREE

    def _should_skip(self, request: Request) -> bool:
        """Check if request should skip limit checking."""
        path = request.url.path
        return any(path.startswith(p) for p in self.exclude_paths)

    def _get_metric_for_endpoint(self, request: Request) -> Optional[UsageMetric]:
        """Determine which metric to check for this endpoint."""
        path = request.url.path
        method = request.method

        # Conversation endpoints → messages
        if "/conversation" in path or "/chat" in path:
            if method == "POST":
                return UsageMetric.MESSAGES_SENT

        # Memory endpoints
        if "/memory" in path or "/memories" in path:
            if method == "POST":
                return UsageMetric.MEMORIES_CREATED
            return UsageMetric.MEMORIES_ACCESSED

        # Expert endpoints
        if "/expert" in path:
            return UsageMetric.EXPERT_INVOCATIONS

        # API calls (generic)
        if path.startswith("/api/"):
            return UsageMetric.API_CALLS

        return None

    async def _check_limit(
        self,
        user_id: str,
        metric: UsageMetric,
        limits: TierLimits,
        tier: SubscriptionTier,
    ) -> LimitCheckResult:
        """Check if user is within limits for a metric."""
        # Determine period and limit
        if metric == UsageMetric.MESSAGES_SENT:
            # Check both daily and monthly
            daily_limit = limits.messages_per_day
            monthly_limit = limits.messages_per_month

            # Check daily first
            if daily_limit:
                within, current, pct = await self.tracker.check_limit(
                    user_id, metric, daily_limit, UsagePeriod.DAILY
                )
                if not within:
                    return LimitCheckResult(
                        allowed=False,
                        metric=metric,
                        current=current,
                        limit=daily_limit,
                        percentage=pct,
                        tier=tier,
                        retry_after_seconds=self._seconds_until_reset(UsagePeriod.DAILY),
                        message=f"Daily message limit ({daily_limit}) reached",
                    )

            # Check monthly
            if monthly_limit:
                within, current, pct = await self.tracker.check_limit(
                    user_id, metric, monthly_limit, UsagePeriod.MONTHLY
                )
                if not within:
                    return LimitCheckResult(
                        allowed=False,
                        metric=metric,
                        current=current,
                        limit=monthly_limit,
                        percentage=pct,
                        tier=tier,
                        retry_after_seconds=self._seconds_until_reset(UsagePeriod.MONTHLY),
                        message=f"Monthly message limit ({monthly_limit}) reached",
                    )

                return LimitCheckResult(
                    allowed=True,
                    metric=metric,
                    current=current,
                    limit=monthly_limit,
                    percentage=pct,
                    tier=tier,
                    soft_limit_reached=pct >= limits.soft_limit_threshold * 100,
                    approaching_limit=pct >= 70,
                )

        elif metric == UsageMetric.API_CALLS:
            monthly_limit = limits.api_calls_per_month
            if monthly_limit is not None:
                within, current, pct = await self.tracker.check_limit(
                    user_id, metric, monthly_limit, UsagePeriod.MONTHLY
                )
                if not within:
                    return LimitCheckResult(
                        allowed=False,
                        metric=metric,
                        current=current,
                        limit=monthly_limit,
                        percentage=pct,
                        tier=tier,
                        retry_after_seconds=self._seconds_until_reset(UsagePeriod.MONTHLY),
                        message=f"Monthly API call limit ({monthly_limit}) reached",
                    )

                return LimitCheckResult(
                    allowed=True,
                    metric=metric,
                    current=current,
                    limit=monthly_limit,
                    percentage=pct,
                    tier=tier,
                    soft_limit_reached=pct >= limits.soft_limit_threshold * 100,
                )

        elif metric == UsageMetric.MEMORIES_TOTAL:
            max_memories = limits.memories_max
            if max_memories:
                current = await self.tracker.get_current(
                    user_id, UsageMetric.MEMORIES_TOTAL, UsagePeriod.ALL_TIME
                )
                if current >= max_memories:
                    return LimitCheckResult(
                        allowed=False,
                        metric=metric,
                        current=current,
                        limit=max_memories,
                        percentage=100.0,
                        tier=tier,
                        message=f"Maximum memories ({max_memories}) reached",
                    )

        elif metric == UsageMetric.EXPERT_INVOCATIONS:
            daily_limit = limits.expert_invocations_per_day
            if daily_limit:
                within, current, pct = await self.tracker.check_limit(
                    user_id, metric, daily_limit, UsagePeriod.DAILY
                )
                if not within:
                    return LimitCheckResult(
                        allowed=False,
                        metric=metric,
                        current=current,
                        limit=daily_limit,
                        percentage=pct,
                        tier=tier,
                        retry_after_seconds=self._seconds_until_reset(UsagePeriod.DAILY),
                        message=f"Daily expert invocation limit ({daily_limit}) reached",
                    )

        # Default: allowed
        return LimitCheckResult(
            allowed=True,
            metric=metric,
            current=0,
            limit=None,
            percentage=0,
            tier=tier,
        )

    def _seconds_until_reset(self, period: UsagePeriod) -> int:
        """Calculate seconds until period reset."""
        now = datetime.utcnow()

        if period == UsagePeriod.DAILY:
            tomorrow = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
            return int((tomorrow - now).total_seconds())

        elif period == UsagePeriod.MONTHLY:
            if now.month == 12:
                next_month = now.replace(year=now.year + 1, month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
            else:
                next_month = now.replace(month=now.month + 1, day=1, hour=0, minute=0, second=0, microsecond=0)
            return int((next_month - now).total_seconds())

        elif period == UsagePeriod.HOURLY:
            next_hour = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
            return int((next_hour - now).total_seconds())

        return 3600  # Default to 1 hour

    async def dispatch(self, request: Request, call_next) -> Response:
        """Process request with usage limit checking."""
        # Skip excluded paths
        if self._should_skip(request):
            return await call_next(request)

        # Get user
        user_id = self.user_resolver(request)
        if not user_id:
            # No user identified, skip limit checking
            return await call_next(request)

        # Get tier and limits
        tier = await self._get_tier(user_id)
        limits = get_tier_limits(tier)

        # Determine metric for this endpoint
        metric = self._get_metric_for_endpoint(request)

        # Check limits before processing
        if self.enable_enforcement and metric:
            result = await self._check_limit(user_id, metric, limits, tier)

            if not result.allowed:
                return self._limit_exceeded_response(result)

            # Add warning headers if approaching limit
            if result.soft_limit_reached:
                logger.info(
                    "User approaching limit",
                    user_id=user_id,
                    metric=metric.value,
                    percentage=result.percentage,
                )

        # Create usage context
        context = UsageContext(
            user_id=user_id,
            tier=tier,
            limits=limits,
            request_start_time=time.time(),
        )
        request.state.usage_context = context

        # Process request
        try:
            response = await call_next(request)

            # Track usage after successful request
            if self.enable_tracking and metric:
                await self._track_request_usage(context, metric, response)

            # Add usage headers to response
            if metric:
                current = await self.tracker.get_current(user_id, metric, UsagePeriod.MONTHLY)
                limit = self._get_limit_for_metric(metric, limits)
                headers = {
                    "X-Usage-Metric": metric.value,
                    "X-Usage-Current": str(int(current)),
                    "X-Usage-Tier": tier.value,
                }
                if limit:
                    headers["X-Usage-Limit"] = str(int(limit))
                    headers["X-Usage-Remaining"] = str(int(max(0, limit - current)))

                for key, value in headers.items():
                    response.headers[key] = value

            return response

        except Exception as e:
            logger.error(f"Request processing error: {e}", user_id=user_id)
            raise

    async def _track_request_usage(
        self,
        context: UsageContext,
        metric: UsageMetric,
        response: Response,
    ) -> None:
        """Track usage for completed request."""
        # Track the primary metric
        await self.tracker.increment(
            context.user_id,
            metric,
            1.0,
            dimensions={
                "expert_type": context.expert_type,
            } if context.expert_type else None,
        )

        # Track any additional metrics queued during request
        for queued_metric, amount in context.metrics_to_track.items():
            if queued_metric != metric:
                await self.tracker.increment(context.user_id, queued_metric, amount)

    def _get_limit_for_metric(self, metric: UsageMetric, limits: TierLimits) -> Optional[float]:
        """Get the limit for a specific metric."""
        mapping = {
            UsageMetric.MESSAGES_SENT: limits.messages_per_month,
            UsageMetric.API_CALLS: limits.api_calls_per_month,
            UsageMetric.MEMORIES_TOTAL: limits.memories_max,
            UsageMetric.EXPERT_INVOCATIONS: limits.expert_invocations_per_day,
            UsageMetric.STORAGE_USED_MB: limits.storage_mb,
            UsageMetric.DOCUMENTS_COUNT: limits.documents_max,
        }
        return mapping.get(metric)

    def _limit_exceeded_response(self, result: LimitCheckResult) -> JSONResponse:
        """Generate 429 response for exceeded limit."""
        return JSONResponse(
            status_code=429,
            content={
                "error": "usage_limit_exceeded",
                "message": result.message,
                "metric": result.metric.value,
                "current": result.current,
                "limit": result.limit,
                "tier": result.tier.value,
                "upgrade_url": "/pricing",
                "retry_after_seconds": result.retry_after_seconds,
            },
            headers=result.to_headers(),
        )


# =============================================================================
# FastAPI Dependency
# =============================================================================

class UsageLimitDependency:
    """
    FastAPI dependency for per-route usage limit checking.

    Use this for fine-grained control over specific endpoints.
    """

    def __init__(
        self,
        tracker: UsageTracker,
        store: UsageStore,
        metric: UsageMetric,
        increment_amount: float = 1.0,
    ):
        self.tracker = tracker
        self.store = store
        self.metric = metric
        self.increment_amount = increment_amount

    async def __call__(self, request: Request) -> UsageContext:
        """Check limits and return usage context."""
        # Get user from request state
        user_id = None
        if hasattr(request.state, "user") and request.state.user:
            user_id = request.state.user.id
        elif hasattr(request.state, "usage_context"):
            return request.state.usage_context

        if not user_id:
            raise HTTPException(status_code=401, detail="Authentication required")

        # Get subscription and limits
        subscription = await self.store.get_subscription(user_id)
        tier = subscription.tier if subscription else SubscriptionTier.FREE
        limits = get_tier_limits(tier)

        # Check limit
        limit = self._get_limit_for_metric(self.metric, limits)
        if limit:
            within, current, pct = await self.tracker.check_limit(
                user_id, self.metric, limit, UsagePeriod.MONTHLY
            )

            if not within:
                raise HTTPException(
                    status_code=429,
                    detail={
                        "error": "usage_limit_exceeded",
                        "metric": self.metric.value,
                        "current": current,
                        "limit": limit,
                    },
                    headers={
                        "X-RateLimit-Limit": str(int(limit)),
                        "X-RateLimit-Remaining": "0",
                    },
                )

        # Create context
        context = UsageContext(
            user_id=user_id,
            tier=tier,
            limits=limits,
            subscription=subscription,
        )

        return context

    def _get_limit_for_metric(self, metric: UsageMetric, limits: TierLimits) -> Optional[float]:
        mapping = {
            UsageMetric.MESSAGES_SENT: limits.messages_per_month,
            UsageMetric.API_CALLS: limits.api_calls_per_month,
            UsageMetric.MEMORIES_TOTAL: limits.memories_max,
        }
        return mapping.get(metric)


def usage_limit_dependency(
    tracker: UsageTracker,
    store: UsageStore,
    metric: UsageMetric,
) -> UsageLimitDependency:
    """Create a usage limit dependency."""
    return UsageLimitDependency(tracker, store, metric)


# =============================================================================
# Decorator for Usage Tracking
# =============================================================================

def track_usage(
    metric: UsageMetric,
    amount: Union[float, Callable[..., float]] = 1.0,
    dimensions_extractor: Optional[Callable[..., Dict[str, str]]] = None,
):
    """
    Decorator for automatic usage tracking.

    Usage:
        @track_usage(UsageMetric.EXPERT_INVOCATIONS)
        async def invoke_expert(request: Request, expert_type: str):
            ...

        @track_usage(
            UsageMetric.TOKENS_TOTAL,
            amount=lambda result: result.get("tokens", 0)
        )
        async def generate_response(...):
            ...
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Get usage context from request
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break

            if not request:
                request = kwargs.get("request")

            context: Optional[UsageContext] = None
            if request and hasattr(request.state, "usage_context"):
                context = request.state.usage_context

            # Execute function
            result = await func(*args, **kwargs)

            # Calculate amount
            actual_amount = amount
            if callable(amount):
                actual_amount = amount(result)

            # Extract dimensions
            dimensions = None
            if dimensions_extractor:
                dimensions = dimensions_extractor(*args, **kwargs)

            # Track usage
            if context:
                context.track(metric, actual_amount)

            return result

        return wrapper
    return decorator


# =============================================================================
# Expert Access Checker
# =============================================================================

class ExpertAccessChecker:
    """
    Check if user has access to specific experts based on tier.
    """

    def __init__(self, store: UsageStore):
        self.store = store

    async def check_access(
        self,
        user_id: str,
        expert_type: str,
    ) -> Tuple[bool, str]:
        """
        Check if user can access an expert.

        Returns:
            (allowed, reason)
        """
        subscription = await self.store.get_subscription(user_id)
        tier = subscription.tier if subscription else SubscriptionTier.FREE
        limits = get_tier_limits(tier)

        # Check expert access level
        if limits.expert_access == ExpertAccess.GENERAL_ONLY:
            if expert_type != "general":
                return False, f"Expert '{expert_type}' requires Pro tier or higher"

        elif limits.expert_access == ExpertAccess.BASIC_EXPERTS:
            # Check against allowed list
            if limits.allowed_experts and expert_type not in limits.allowed_experts:
                return False, f"Expert '{expert_type}' not available in your plan"

        # Check if specific expert is allowed
        if limits.allowed_experts is not None:
            if expert_type not in limits.allowed_experts:
                return False, f"Expert '{expert_type}' not included in your plan"

        return True, "Access granted"

    async def __call__(
        self,
        request: Request,
        expert_type: str,
    ) -> None:
        """FastAPI dependency for expert access checking."""
        user_id = None
        if hasattr(request.state, "user") and request.state.user:
            user_id = request.state.user.id

        if not user_id:
            raise HTTPException(status_code=401, detail="Authentication required")

        allowed, reason = await self.check_access(user_id, expert_type)

        if not allowed:
            raise HTTPException(
                status_code=403,
                detail={
                    "error": "expert_access_denied",
                    "expert_type": expert_type,
                    "message": reason,
                    "upgrade_url": "/pricing",
                },
            )


# Import ExpertAccess for the checker
from aion.usage.models import ExpertAccess
