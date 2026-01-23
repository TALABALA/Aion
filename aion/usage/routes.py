"""
AION Usage API Routes

State-of-the-art usage API endpoints:
- Current usage retrieval
- Historical usage data
- Usage breakdown by dimension
- Subscription management
- Alert management
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Path, Request
from pydantic import BaseModel, Field
import structlog

from aion.usage.models import (
    SubscriptionTier,
    UsageMetric,
    UsagePeriod,
    BillingCycle,
)
from aion.usage.service import UsageService

logger = structlog.get_logger(__name__)


# =============================================================================
# Request/Response Models
# =============================================================================

class UsageMetricResponse(BaseModel):
    """Single metric usage response."""
    used: float
    limit: Optional[float]
    unlimited: bool
    percentage: float = 0.0
    soft_limit_reached: bool = False
    hard_limit_reached: bool = False
    change_from_previous: Optional[float] = None


class UsageSummaryResponse(BaseModel):
    """Current usage summary response."""
    period: str
    usage: Dict[str, UsageMetricResponse]
    tier: str
    billing_cycle_end: Optional[str]
    days_remaining: int = 0
    team_id: Optional[str] = None


class UsageLimitsResponse(BaseModel):
    """User's limits response."""
    tier: str
    limits: Dict[str, Any]
    features: Dict[str, bool]


class UsageHistoryItem(BaseModel):
    """Single history data point."""
    date: str
    value: float


class UsageHistoryResponse(BaseModel):
    """Historical usage response."""
    metric: str
    period: str
    history: List[UsageHistoryItem]


class UsageBreakdownResponse(BaseModel):
    """Usage breakdown by dimension."""
    metric: str
    dimension: str
    period: str
    breakdown: Dict[str, float]
    total: float


class UsageTrendsResponse(BaseModel):
    """Usage trends response."""
    metric: str
    trend: str
    daily_average: float
    total: float
    peak: float
    history: List[UsageHistoryItem]


class UsageForecastResponse(BaseModel):
    """Usage forecast response."""
    metric: str
    current: float
    limit: Optional[float]
    days_elapsed: int
    days_remaining: int
    daily_average: float
    forecasted_total: float
    will_exceed_limit: bool
    days_until_exceeded: Optional[int]
    recommended_daily_limit: Optional[float]


class AlertResponse(BaseModel):
    """Usage alert response."""
    alert_id: str
    alert_type: str
    severity: str
    metric: str
    current_value: float
    limit_value: Optional[float]
    percentage: float
    title: str
    message: str
    action_url: Optional[str]
    action_label: Optional[str]
    created_at: str


class SubscriptionResponse(BaseModel):
    """Subscription details response."""
    user_id: str
    tier: str
    billing_cycle: str
    started_at: str
    current_period_start: str
    current_period_end: Optional[str]
    is_active: bool
    is_trial: bool
    trial_ends_at: Optional[str]
    team_id: Optional[str]
    seat_count: int


class CreateSubscriptionRequest(BaseModel):
    """Request to create/upgrade subscription."""
    tier: str = Field(..., description="Subscription tier: free, pro, executive, team")
    billing_cycle: str = Field(default="monthly", description="Billing cycle: monthly or yearly")
    is_trial: bool = Field(default=False)
    trial_days: int = Field(default=14)


class UpgradeSubscriptionRequest(BaseModel):
    """Request to upgrade subscription."""
    tier: str


# =============================================================================
# Router Factory
# =============================================================================

def create_usage_router(
    usage_service: UsageService,
    require_auth: bool = True,
) -> APIRouter:
    """
    Create the usage API router.

    Args:
        usage_service: The usage service instance
        require_auth: Whether authentication is required

    Returns:
        FastAPI APIRouter
    """
    router = APIRouter(prefix="/api/v1/usage", tags=["usage"])

    # =========================================================================
    # Helper: Get User ID
    # =========================================================================

    async def get_user_id(request: Request) -> str:
        """Extract user ID from request."""
        if hasattr(request.state, "user") and request.state.user:
            return request.state.user.id

        user_id = request.headers.get("X-User-ID")
        if user_id:
            return user_id

        if require_auth:
            raise HTTPException(status_code=401, detail="Authentication required")

        # For testing, allow query param
        user_id = request.query_params.get("user_id")
        if not user_id:
            raise HTTPException(status_code=400, detail="user_id required")

        return user_id

    # =========================================================================
    # Current Usage
    # =========================================================================

    @router.get(
        "/current",
        response_model=UsageSummaryResponse,
        summary="Get current usage",
        description="Get current usage summary for the authenticated user",
    )
    async def get_current_usage(
        request: Request,
        period: str = Query(
            default="monthly",
            description="Period: daily, monthly",
            regex="^(daily|monthly)$"
        ),
    ) -> UsageSummaryResponse:
        """Get current usage for the user."""
        user_id = await get_user_id(request)

        period_enum = UsagePeriod.MONTHLY if period == "monthly" else UsagePeriod.DAILY

        summary = await usage_service.get_current_usage(user_id, period_enum)

        return UsageSummaryResponse(
            period=summary.period,
            usage={
                name: UsageMetricResponse(
                    used=metric.used,
                    limit=metric.limit,
                    unlimited=metric.unlimited,
                    percentage=metric.percentage,
                    soft_limit_reached=metric.soft_limit_reached,
                    hard_limit_reached=metric.hard_limit_reached,
                    change_from_previous=metric.change_from_previous,
                )
                for name, metric in summary.usage.items()
            },
            tier=summary.tier.value,
            billing_cycle_end=summary.billing_cycle_end.isoformat() if summary.billing_cycle_end else None,
            days_remaining=summary.days_remaining,
            team_id=summary.team_id,
        )

    # =========================================================================
    # Limits
    # =========================================================================

    @router.get(
        "/limits",
        response_model=UsageLimitsResponse,
        summary="Get usage limits",
        description="Get all usage limits for the authenticated user's tier",
    )
    async def get_limits(request: Request) -> UsageLimitsResponse:
        """Get user's usage limits."""
        user_id = await get_user_id(request)

        limits_data = await usage_service.get_limits(user_id)

        return UsageLimitsResponse(
            tier=limits_data["tier"],
            limits=limits_data["limits"],
            features=limits_data["features"],
        )

    # =========================================================================
    # History
    # =========================================================================

    @router.get(
        "/history",
        response_model=UsageHistoryResponse,
        summary="Get usage history",
        description="Get historical usage data for a specific metric",
    )
    async def get_history(
        request: Request,
        metric: str = Query(
            ...,
            description="Metric to retrieve history for",
        ),
        days: int = Query(
            default=30,
            ge=1,
            le=365,
            description="Number of days of history",
        ),
    ) -> UsageHistoryResponse:
        """Get usage history for a metric."""
        user_id = await get_user_id(request)

        try:
            metric_enum = UsageMetric(metric)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid metric: {metric}. Valid options: {[m.value for m in UsageMetric]}"
            )

        history = await usage_service.get_usage_history(user_id, metric_enum, days)

        return UsageHistoryResponse(
            metric=metric,
            period=f"last_{days}_days",
            history=[UsageHistoryItem(date=h["date"], value=h["value"]) for h in history],
        )

    # =========================================================================
    # Breakdown
    # =========================================================================

    @router.get(
        "/breakdown",
        response_model=UsageBreakdownResponse,
        summary="Get usage breakdown",
        description="Get usage breakdown by dimension (e.g., by expert type)",
    )
    async def get_breakdown(
        request: Request,
        metric: str = Query(..., description="Metric to break down"),
        dimension: str = Query(
            default="expert_type",
            description="Dimension to break down by: expert_type, memory_type",
        ),
        period: str = Query(default="monthly", description="Period: daily, monthly"),
    ) -> UsageBreakdownResponse:
        """Get usage breakdown by dimension."""
        user_id = await get_user_id(request)

        try:
            metric_enum = UsageMetric(metric)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid metric: {metric}")

        period_enum = UsagePeriod.MONTHLY if period == "monthly" else UsagePeriod.DAILY

        breakdown = await usage_service.get_usage_breakdown(
            user_id, metric_enum, dimension, period_enum
        )

        return UsageBreakdownResponse(
            metric=metric,
            dimension=dimension,
            period=period,
            breakdown=breakdown,
            total=sum(breakdown.values()),
        )

    # =========================================================================
    # Trends & Forecasting
    # =========================================================================

    @router.get(
        "/trends/{metric}",
        response_model=UsageTrendsResponse,
        summary="Get usage trends",
        description="Get usage trends and analysis for a metric",
    )
    async def get_trends(
        request: Request,
        metric: str = Path(..., description="Metric to analyze"),
        days: int = Query(default=30, ge=7, le=90),
    ) -> UsageTrendsResponse:
        """Get usage trends for a metric."""
        user_id = await get_user_id(request)

        try:
            metric_enum = UsageMetric(metric)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid metric: {metric}")

        trends = await usage_service.get_usage_trends(user_id, metric_enum, days)

        return UsageTrendsResponse(
            metric=trends["metric"],
            trend=trends["trend"],
            daily_average=trends["daily_average"],
            total=trends["total"],
            peak=trends["peak"],
            history=[UsageHistoryItem(date=h["date"], value=h["value"]) for h in trends["history"]],
        )

    @router.get(
        "/forecast/{metric}",
        response_model=UsageForecastResponse,
        summary="Get usage forecast",
        description="Forecast usage for the remainder of the billing period",
    )
    async def get_forecast(
        request: Request,
        metric: str = Path(..., description="Metric to forecast"),
    ) -> UsageForecastResponse:
        """Get usage forecast for a metric."""
        user_id = await get_user_id(request)

        try:
            metric_enum = UsageMetric(metric)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid metric: {metric}")

        forecast = await usage_service.forecast_usage(user_id, metric_enum)

        return UsageForecastResponse(**forecast)

    # =========================================================================
    # Alerts
    # =========================================================================

    @router.get(
        "/alerts",
        response_model=List[AlertResponse],
        summary="Get active alerts",
        description="Get active usage alerts for the user",
    )
    async def get_alerts(request: Request) -> List[AlertResponse]:
        """Get active usage alerts."""
        user_id = await get_user_id(request)

        alerts = await usage_service.get_active_alerts(user_id)

        return [
            AlertResponse(
                alert_id=alert.alert_id,
                alert_type=alert.alert_type.value,
                severity=alert.severity.value,
                metric=alert.metric.value,
                current_value=alert.current_value,
                limit_value=alert.limit_value,
                percentage=alert.percentage,
                title=alert.title,
                message=alert.message,
                action_url=alert.action_url,
                action_label=alert.action_label,
                created_at=alert.created_at.isoformat(),
            )
            for alert in alerts
        ]

    @router.post(
        "/alerts/{alert_id}/acknowledge",
        summary="Acknowledge alert",
        description="Mark an alert as acknowledged",
    )
    async def acknowledge_alert(
        request: Request,
        alert_id: str = Path(..., description="Alert ID to acknowledge"),
    ) -> Dict[str, bool]:
        """Acknowledge an alert."""
        user_id = await get_user_id(request)

        success = await usage_service.acknowledge_alert(user_id, alert_id)

        return {"acknowledged": success}

    # =========================================================================
    # Subscription Management
    # =========================================================================

    @router.get(
        "/subscription",
        response_model=Optional[SubscriptionResponse],
        summary="Get subscription",
        description="Get current subscription details",
    )
    async def get_subscription(request: Request) -> Optional[SubscriptionResponse]:
        """Get user's subscription."""
        user_id = await get_user_id(request)

        subscription = await usage_service.get_subscription(user_id)

        if not subscription:
            return None

        return SubscriptionResponse(
            user_id=subscription.user_id,
            tier=subscription.tier.value,
            billing_cycle=subscription.billing_cycle.value,
            started_at=subscription.started_at.isoformat(),
            current_period_start=subscription.current_period_start.isoformat(),
            current_period_end=subscription.current_period_end.isoformat() if subscription.current_period_end else None,
            is_active=subscription.is_active,
            is_trial=subscription.is_trial,
            trial_ends_at=subscription.trial_ends_at.isoformat() if subscription.trial_ends_at else None,
            team_id=subscription.team_id,
            seat_count=subscription.seat_count,
        )

    @router.post(
        "/subscription",
        response_model=SubscriptionResponse,
        summary="Create subscription",
        description="Create or update subscription",
    )
    async def create_subscription(
        request: Request,
        body: CreateSubscriptionRequest,
    ) -> SubscriptionResponse:
        """Create or update subscription."""
        user_id = await get_user_id(request)

        try:
            tier = SubscriptionTier(body.tier.lower())
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid tier: {body.tier}. Valid options: {[t.value for t in SubscriptionTier]}"
            )

        try:
            billing_cycle = BillingCycle(body.billing_cycle.lower())
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid billing cycle: {body.billing_cycle}"
            )

        subscription = await usage_service.create_subscription(
            user_id=user_id,
            tier=tier,
            billing_cycle=billing_cycle,
            is_trial=body.is_trial,
            trial_days=body.trial_days,
        )

        return SubscriptionResponse(
            user_id=subscription.user_id,
            tier=subscription.tier.value,
            billing_cycle=subscription.billing_cycle.value,
            started_at=subscription.started_at.isoformat(),
            current_period_start=subscription.current_period_start.isoformat(),
            current_period_end=subscription.current_period_end.isoformat() if subscription.current_period_end else None,
            is_active=subscription.is_active,
            is_trial=subscription.is_trial,
            trial_ends_at=subscription.trial_ends_at.isoformat() if subscription.trial_ends_at else None,
            team_id=subscription.team_id,
            seat_count=subscription.seat_count,
        )

    @router.put(
        "/subscription/upgrade",
        response_model=SubscriptionResponse,
        summary="Upgrade subscription",
        description="Upgrade to a higher tier",
    )
    async def upgrade_subscription(
        request: Request,
        body: UpgradeSubscriptionRequest,
    ) -> SubscriptionResponse:
        """Upgrade subscription tier."""
        user_id = await get_user_id(request)

        try:
            tier = SubscriptionTier(body.tier.lower())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid tier: {body.tier}")

        subscription = await usage_service.upgrade_subscription(user_id, tier)

        return SubscriptionResponse(
            user_id=subscription.user_id,
            tier=subscription.tier.value,
            billing_cycle=subscription.billing_cycle.value,
            started_at=subscription.started_at.isoformat(),
            current_period_start=subscription.current_period_start.isoformat(),
            current_period_end=subscription.current_period_end.isoformat() if subscription.current_period_end else None,
            is_active=subscription.is_active,
            is_trial=subscription.is_trial,
            trial_ends_at=subscription.trial_ends_at.isoformat() if subscription.trial_ends_at else None,
            team_id=subscription.team_id,
            seat_count=subscription.seat_count,
        )

    @router.delete(
        "/subscription",
        summary="Cancel subscription",
        description="Cancel subscription (will downgrade to free at period end)",
    )
    async def cancel_subscription(request: Request) -> Dict[str, Any]:
        """Cancel subscription."""
        user_id = await get_user_id(request)

        subscription = await usage_service.cancel_subscription(user_id)

        if not subscription:
            raise HTTPException(status_code=404, detail="No active subscription found")

        return {
            "canceled": True,
            "effective_date": subscription.current_period_end.isoformat() if subscription.current_period_end else None,
        }

    # =========================================================================
    # Check Action
    # =========================================================================

    @router.get(
        "/check/{action}",
        summary="Check if action is allowed",
        description="Check if user can perform a specific action",
    )
    async def check_action(
        request: Request,
        action: str = Path(..., description="Action to check: send_message, create_memory, api_call, expert:type"),
    ) -> Dict[str, Any]:
        """Check if an action is allowed."""
        user_id = await get_user_id(request)

        allowed, reason = await usage_service.can_perform_action(user_id, action)

        return {
            "action": action,
            "allowed": allowed,
            "reason": reason if not allowed else None,
        }

    return router
