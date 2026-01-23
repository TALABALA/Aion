"""
AION Usage Service

Central service coordinating:
- Real-time tracking (Redis)
- Historical storage (PostgreSQL)
- Subscription management
- Usage analytics
- Alert generation
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, date
from typing import Any, Dict, List, Optional, Tuple
import uuid

import structlog

from aion.usage.models import (
    SubscriptionTier,
    UsageMetric,
    UsagePeriod,
    UsageRecord,
    UsageSummary,
    UsageMetricSummary,
    UsageAlert,
    AlertType,
    AlertSeverity,
    Subscription,
    BillingCycle,
    TierLimits,
    get_tier_limits,
)
from aion.usage.tracker import UsageTracker, MemoryUsageTracker
from aion.usage.storage import UsageStore, MemoryUsageStore

logger = structlog.get_logger(__name__)


class UsageService:
    """
    Central usage service coordinating tracking, storage, and analytics.

    Features:
    - Unified API for all usage operations
    - Real-time + historical data management
    - Automatic alert generation
    - Usage analytics and forecasting
    """

    def __init__(
        self,
        tracker: Optional[UsageTracker] = None,
        store: Optional[UsageStore] = None,
        alert_thresholds: Optional[Dict[str, float]] = None,
    ):
        self.tracker = tracker or MemoryUsageTracker()
        self.store = store or MemoryUsageStore()
        self.alert_thresholds = alert_thresholds or {
            "warning": 0.8,   # 80%
            "critical": 0.9,  # 90%
            "exceeded": 1.0,  # 100%
        }

        # Alert tracking to avoid duplicates
        self._sent_alerts: Dict[str, datetime] = {}
        self._alert_cooldown = timedelta(hours=4)

    # =========================================================================
    # Core Usage Operations
    # =========================================================================

    async def record_usage(
        self,
        user_id: str,
        metric: UsageMetric,
        amount: float = 1.0,
        dimensions: Optional[Dict[str, str]] = None,
        session_id: Optional[str] = None,
    ) -> Tuple[float, Optional[UsageAlert]]:
        """
        Record usage and check for alerts.

        Returns:
            (new_total, optional_alert)
        """
        # Get user's limits
        subscription = await self.store.get_subscription(user_id)
        tier = subscription.tier if subscription else SubscriptionTier.FREE
        limits = get_tier_limits(tier)

        # Increment in real-time tracker
        new_total = await self.tracker.increment(
            user_id,
            metric,
            amount,
            dimensions=dimensions,
            period=UsagePeriod.MONTHLY,
        )

        # Also record for historical storage
        record = UsageRecord(
            user_id=user_id,
            metric=metric,
            value=amount,
            timestamp=datetime.utcnow(),
            period=UsagePeriod.DAILY,
            expert_type=dimensions.get("expert_type") if dimensions else None,
            memory_type=dimensions.get("memory_type") if dimensions else None,
            session_id=session_id,
        )
        await self.store.save_record(record)

        # Check for alerts
        alert = await self._check_for_alert(user_id, metric, new_total, limits, tier)

        return new_total, alert

    async def batch_record_usage(
        self,
        user_id: str,
        metrics: Dict[UsageMetric, float],
    ) -> Dict[UsageMetric, float]:
        """
        Record multiple metrics at once.

        More efficient than individual calls.
        """
        results = await self.tracker.batch_increment(
            user_id,
            metrics,
            UsagePeriod.MONTHLY,
        )

        # Record for historical storage
        records = [
            UsageRecord(
                user_id=user_id,
                metric=metric,
                value=amount,
                timestamp=datetime.utcnow(),
            )
            for metric, amount in metrics.items()
        ]
        await self.store.save_records_batch(records)

        return results

    async def get_current_usage(
        self,
        user_id: str,
        period: UsagePeriod = UsagePeriod.MONTHLY,
    ) -> UsageSummary:
        """
        Get current usage summary for a user.

        Combines real-time data with subscription info.
        """
        # Get all metrics from real-time tracker
        metrics = await self.tracker.get_all_metrics(user_id, period)

        # Get subscription
        subscription = await self.store.get_subscription(user_id)
        tier = subscription.tier if subscription else SubscriptionTier.FREE
        limits = get_tier_limits(tier)

        # Build summary
        now = datetime.utcnow()
        if period == UsagePeriod.MONTHLY:
            period_str = now.strftime("%Y-%m")
            period_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            if now.month == 12:
                period_end = now.replace(year=now.year + 1, month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
            else:
                period_end = now.replace(month=now.month + 1, day=1, hour=0, minute=0, second=0, microsecond=0)
        else:
            period_str = now.strftime("%Y-%m-%d")
            period_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            period_end = period_start + timedelta(days=1)

        # Build metric summaries
        usage = {}
        for metric, value in metrics.items():
            limit = self._get_limit_for_metric(metric, limits, period)
            unlimited = limit is None

            percentage = 0.0
            if limit and limit > 0:
                percentage = (value / limit) * 100

            usage[metric.value] = UsageMetricSummary(
                used=value,
                limit=limit,
                unlimited=unlimited,
                percentage=percentage,
                soft_limit_reached=percentage >= limits.soft_limit_threshold * 100,
                hard_limit_reached=percentage >= 100,
            )

        # Calculate billing info
        billing_end = None
        days_remaining = 0
        if subscription and subscription.current_period_end:
            billing_end = subscription.current_period_end
            days_remaining = (billing_end.date() - now.date()).days

        return UsageSummary(
            user_id=user_id,
            tier=tier,
            period=period_str,
            period_start=period_start,
            period_end=period_end,
            usage=usage,
            billing_cycle_end=billing_end,
            days_remaining=max(0, days_remaining),
            team_id=subscription.team_id if subscription else None,
        )

    async def get_limits(self, user_id: str) -> Dict[str, Any]:
        """
        Get all limits for a user.
        """
        subscription = await self.store.get_subscription(user_id)
        tier = subscription.tier if subscription else SubscriptionTier.FREE
        limits = get_tier_limits(tier)

        return {
            "tier": tier.value,
            "limits": limits.to_dict(),
            "features": {
                "priority_processing": limits.priority_processing,
                "custom_expert_training": limits.custom_expert_training,
                "dedicated_support": limits.dedicated_support,
                "shared_memory_pool": limits.shared_memory_pool,
                "admin_controls": limits.admin_controls,
            },
        }

    async def get_usage_breakdown(
        self,
        user_id: str,
        metric: UsageMetric,
        dimension: str,
        period: UsagePeriod = UsagePeriod.MONTHLY,
    ) -> Dict[str, float]:
        """
        Get usage breakdown by dimension.

        E.g., expert invocations by expert type.
        """
        return await self.tracker.get_breakdown(user_id, metric, dimension, period)

    async def get_usage_history(
        self,
        user_id: str,
        metric: UsageMetric,
        days: int = 30,
    ) -> List[Dict[str, Any]]:
        """
        Get historical usage data for charts.
        """
        history = await self.tracker.get_history(user_id, metric, days)

        return [
            {"date": date_str, "value": value}
            for date_str, value in history
        ]

    # =========================================================================
    # Subscription Management
    # =========================================================================

    async def get_subscription(self, user_id: str) -> Optional[Subscription]:
        """Get user's subscription."""
        return await self.store.get_subscription(user_id)

    async def create_subscription(
        self,
        user_id: str,
        tier: SubscriptionTier,
        billing_cycle: BillingCycle = BillingCycle.MONTHLY,
        team_id: Optional[str] = None,
        is_trial: bool = False,
        trial_days: int = 14,
    ) -> Subscription:
        """
        Create or update a subscription.
        """
        now = datetime.utcnow()

        # Calculate period end
        if billing_cycle == BillingCycle.MONTHLY:
            if now.month == 12:
                period_end = now.replace(year=now.year + 1, month=1, day=now.day)
            else:
                period_end = now.replace(month=now.month + 1)
        else:  # Yearly
            period_end = now.replace(year=now.year + 1)

        subscription = Subscription(
            user_id=user_id,
            tier=tier,
            billing_cycle=billing_cycle,
            started_at=now,
            current_period_start=now,
            current_period_end=period_end,
            team_id=team_id,
            is_active=True,
            is_trial=is_trial,
            trial_ends_at=now + timedelta(days=trial_days) if is_trial else None,
        )

        await self.store.save_subscription(subscription)

        logger.info(
            "Subscription created",
            user_id=user_id,
            tier=tier.value,
            is_trial=is_trial,
        )

        return subscription

    async def upgrade_subscription(
        self,
        user_id: str,
        new_tier: SubscriptionTier,
    ) -> Subscription:
        """
        Upgrade user's subscription tier.
        """
        subscription = await self.store.get_subscription(user_id)

        if not subscription:
            # Create new subscription
            return await self.create_subscription(user_id, new_tier)

        # Update tier
        subscription.tier = new_tier
        subscription.is_trial = False
        subscription.trial_ends_at = None

        await self.store.save_subscription(subscription)

        logger.info(
            "Subscription upgraded",
            user_id=user_id,
            new_tier=new_tier.value,
        )

        return subscription

    async def cancel_subscription(self, user_id: str) -> Optional[Subscription]:
        """
        Cancel a subscription (downgrade to free at period end).
        """
        subscription = await self.store.get_subscription(user_id)

        if not subscription:
            return None

        subscription.canceled_at = datetime.utcnow()

        await self.store.save_subscription(subscription)

        logger.info("Subscription canceled", user_id=user_id)

        return subscription

    # =========================================================================
    # Limit Checking
    # =========================================================================

    async def check_limit(
        self,
        user_id: str,
        metric: UsageMetric,
        period: UsagePeriod = UsagePeriod.MONTHLY,
    ) -> Tuple[bool, float, Optional[float], float]:
        """
        Check if user is within limits for a metric.

        Returns:
            (within_limit, current, limit, percentage)
        """
        subscription = await self.store.get_subscription(user_id)
        tier = subscription.tier if subscription else SubscriptionTier.FREE
        limits = get_tier_limits(tier)

        limit = self._get_limit_for_metric(metric, limits, period)

        within, current, percentage = await self.tracker.check_limit(
            user_id, metric, limit, period
        )

        return within, current, limit, percentage

    async def can_perform_action(
        self,
        user_id: str,
        action: str,
    ) -> Tuple[bool, str]:
        """
        Check if user can perform a specific action.

        Returns:
            (allowed, reason)
        """
        subscription = await self.store.get_subscription(user_id)
        tier = subscription.tier if subscription else SubscriptionTier.FREE
        limits = get_tier_limits(tier)

        # Map actions to checks
        if action == "send_message":
            within, current, limit, pct = await self.check_limit(
                user_id, UsageMetric.MESSAGES_SENT, UsagePeriod.MONTHLY
            )
            if not within:
                return False, f"Monthly message limit ({limit}) reached"

        elif action == "create_memory":
            within, current, limit, pct = await self.check_limit(
                user_id, UsageMetric.MEMORIES_TOTAL, UsagePeriod.ALL_TIME
            )
            if not within:
                return False, f"Maximum memories ({limit}) reached"

        elif action == "api_call":
            if limits.api_calls_per_month == 0:
                return False, "API access requires Pro tier or higher"

            within, current, limit, pct = await self.check_limit(
                user_id, UsageMetric.API_CALLS, UsagePeriod.MONTHLY
            )
            if not within:
                return False, f"Monthly API call limit ({limit}) reached"

        elif action.startswith("expert:"):
            expert_type = action.split(":")[1]
            if limits.allowed_experts is not None:
                if expert_type not in limits.allowed_experts:
                    return False, f"Expert '{expert_type}' not available in your plan"

        return True, "Action allowed"

    # =========================================================================
    # Analytics
    # =========================================================================

    async def get_usage_trends(
        self,
        user_id: str,
        metric: UsageMetric,
        days: int = 30,
    ) -> Dict[str, Any]:
        """
        Get usage trends with analysis.
        """
        history = await self.tracker.get_history(user_id, metric, days)

        if not history:
            return {
                "metric": metric.value,
                "trend": "no_data",
                "daily_average": 0,
                "total": 0,
                "peak": 0,
                "history": [],
            }

        values = [v for _, v in history]
        total = sum(values)
        daily_avg = total / len(values) if values else 0
        peak = max(values) if values else 0

        # Simple trend calculation
        if len(values) >= 7:
            recent = sum(values[-7:]) / 7
            older = sum(values[:-7]) / (len(values) - 7) if len(values) > 7 else recent

            if recent > older * 1.2:
                trend = "increasing"
            elif recent < older * 0.8:
                trend = "decreasing"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"

        return {
            "metric": metric.value,
            "trend": trend,
            "daily_average": round(daily_avg, 2),
            "total": total,
            "peak": peak,
            "history": [{"date": d, "value": v} for d, v in history],
        }

    async def forecast_usage(
        self,
        user_id: str,
        metric: UsageMetric,
    ) -> Dict[str, Any]:
        """
        Forecast usage for the remainder of the billing period.
        """
        # Get current usage
        current = await self.tracker.get_current(user_id, metric, UsagePeriod.MONTHLY)

        # Get subscription info
        subscription = await self.store.get_subscription(user_id)
        tier = subscription.tier if subscription else SubscriptionTier.FREE
        limits = get_tier_limits(tier)
        limit = self._get_limit_for_metric(metric, limits, UsagePeriod.MONTHLY)

        # Calculate days elapsed and remaining
        now = datetime.utcnow()
        month_start = now.replace(day=1)
        days_elapsed = (now - month_start).days + 1

        if now.month == 12:
            month_end = now.replace(year=now.year + 1, month=1, day=1)
        else:
            month_end = now.replace(month=now.month + 1, day=1)

        days_total = (month_end - month_start).days
        days_remaining = days_total - days_elapsed

        # Calculate average daily usage
        daily_avg = current / days_elapsed if days_elapsed > 0 else 0

        # Forecast end-of-month usage
        forecasted = current + (daily_avg * days_remaining)

        # Will exceed limit?
        will_exceed = False
        days_until_exceeded = None
        if limit:
            will_exceed = forecasted > limit
            if will_exceed and daily_avg > 0:
                remaining_quota = limit - current
                days_until_exceeded = int(remaining_quota / daily_avg)

        return {
            "metric": metric.value,
            "current": current,
            "limit": limit,
            "days_elapsed": days_elapsed,
            "days_remaining": days_remaining,
            "daily_average": round(daily_avg, 2),
            "forecasted_total": round(forecasted, 2),
            "will_exceed_limit": will_exceed,
            "days_until_exceeded": days_until_exceeded,
            "recommended_daily_limit": round((limit - current) / max(1, days_remaining), 2) if limit else None,
        }

    # =========================================================================
    # Alert Management
    # =========================================================================

    async def _check_for_alert(
        self,
        user_id: str,
        metric: UsageMetric,
        current: float,
        limits: TierLimits,
        tier: SubscriptionTier,
    ) -> Optional[UsageAlert]:
        """Check if an alert should be generated."""
        limit = self._get_limit_for_metric(metric, limits, UsagePeriod.MONTHLY)

        if not limit:
            return None

        percentage = (current / limit) * 100

        # Determine alert type and severity
        alert_type = None
        severity = None

        if percentage >= 100:
            alert_type = AlertType.LIMIT_REACHED
            severity = AlertSeverity.CRITICAL
        elif percentage >= 90:
            alert_type = AlertType.NEAR_LIMIT
            severity = AlertSeverity.WARNING
        elif percentage >= 80:
            alert_type = AlertType.APPROACHING_LIMIT
            severity = AlertSeverity.INFO

        if not alert_type:
            return None

        # Check cooldown
        alert_key = f"{user_id}:{metric.value}:{alert_type.value}"
        last_sent = self._sent_alerts.get(alert_key)

        if last_sent and datetime.utcnow() - last_sent < self._alert_cooldown:
            return None

        # Generate alert
        alert = UsageAlert(
            alert_id=str(uuid.uuid4()),
            user_id=user_id,
            alert_type=alert_type,
            severity=severity,
            metric=metric,
            current_value=current,
            limit_value=limit,
            percentage=percentage,
            title=self._generate_alert_title(metric, alert_type, percentage),
            message=self._generate_alert_message(metric, alert_type, current, limit, tier),
            action_url="/pricing" if tier == SubscriptionTier.FREE else "/usage",
            action_label="Upgrade Plan" if tier == SubscriptionTier.FREE else "View Usage",
        )

        # Record alert sent
        self._sent_alerts[alert_key] = datetime.utcnow()

        logger.info(
            "Usage alert generated",
            user_id=user_id,
            alert_type=alert_type.value,
            metric=metric.value,
            percentage=percentage,
        )

        return alert

    def _generate_alert_title(
        self,
        metric: UsageMetric,
        alert_type: AlertType,
        percentage: float,
    ) -> str:
        """Generate alert title."""
        metric_name = metric.value.replace("_", " ").title()

        if alert_type == AlertType.LIMIT_REACHED:
            return f"{metric_name} Limit Reached"
        elif alert_type == AlertType.NEAR_LIMIT:
            return f"{metric_name} at {int(percentage)}% of Limit"
        else:
            return f"{metric_name} Approaching Limit"

    def _generate_alert_message(
        self,
        metric: UsageMetric,
        alert_type: AlertType,
        current: float,
        limit: float,
        tier: SubscriptionTier,
    ) -> str:
        """Generate alert message."""
        metric_name = metric.value.replace("_", " ")

        if alert_type == AlertType.LIMIT_REACHED:
            if tier == SubscriptionTier.FREE:
                return f"You've used all {int(limit)} {metric_name} for this month. Upgrade to Pro for unlimited access."
            else:
                return f"You've reached your {metric_name} limit of {int(limit)}. Contact support for limit increase."
        elif alert_type == AlertType.NEAR_LIMIT:
            remaining = int(limit - current)
            return f"You have {remaining} {metric_name} remaining this month. Consider managing your usage or upgrading."
        else:
            return f"You've used {int(current)} of {int(limit)} {metric_name} this month."

    async def get_active_alerts(self, user_id: str) -> List[UsageAlert]:
        """Get active (unacknowledged) alerts for a user."""
        # In production, this would query the alerts table
        # For now, check current usage and generate alerts
        alerts = []

        subscription = await self.store.get_subscription(user_id)
        tier = subscription.tier if subscription else SubscriptionTier.FREE
        limits = get_tier_limits(tier)

        # Check key metrics
        for metric in [
            UsageMetric.MESSAGES_SENT,
            UsageMetric.API_CALLS,
            UsageMetric.MEMORIES_TOTAL,
        ]:
            current = await self.tracker.get_current(user_id, metric, UsagePeriod.MONTHLY)
            alert = await self._check_for_alert(user_id, metric, current, limits, tier)
            if alert:
                alerts.append(alert)

        return alerts

    async def acknowledge_alert(self, user_id: str, alert_id: str) -> bool:
        """Mark an alert as acknowledged."""
        # In production, update the alerts table
        logger.info("Alert acknowledged", user_id=user_id, alert_id=alert_id)
        return True

    # =========================================================================
    # Helpers
    # =========================================================================

    def _get_limit_for_metric(
        self,
        metric: UsageMetric,
        limits: TierLimits,
        period: UsagePeriod,
    ) -> Optional[float]:
        """Get the limit for a specific metric and period."""
        mapping = {
            (UsageMetric.MESSAGES_SENT, UsagePeriod.DAILY): limits.messages_per_day,
            (UsageMetric.MESSAGES_SENT, UsagePeriod.MONTHLY): limits.messages_per_month,
            (UsageMetric.TOKENS_INPUT, UsagePeriod.DAILY): limits.tokens_per_day,
            (UsageMetric.TOKENS_OUTPUT, UsagePeriod.DAILY): limits.tokens_per_day,
            (UsageMetric.TOKENS_TOTAL, UsagePeriod.DAILY): limits.tokens_per_day,
            (UsageMetric.TOKENS_TOTAL, UsagePeriod.MONTHLY): limits.tokens_per_month,
            (UsageMetric.API_CALLS, UsagePeriod.DAILY): limits.api_calls_per_day,
            (UsageMetric.API_CALLS, UsagePeriod.MONTHLY): limits.api_calls_per_month,
            (UsageMetric.MEMORIES_TOTAL, UsagePeriod.MONTHLY): limits.memories_max,
            (UsageMetric.MEMORIES_TOTAL, UsagePeriod.ALL_TIME): limits.memories_max,
            (UsageMetric.STORAGE_USED_MB, UsagePeriod.MONTHLY): limits.storage_mb,
            (UsageMetric.DOCUMENTS_COUNT, UsagePeriod.MONTHLY): limits.documents_max,
            (UsageMetric.ATTACHMENTS_COUNT, UsagePeriod.MONTHLY): limits.attachments_max,
            (UsageMetric.EXPERT_INVOCATIONS, UsagePeriod.DAILY): limits.expert_invocations_per_day,
        }
        return mapping.get((metric, period))

    # =========================================================================
    # Context Manager
    # =========================================================================

    async def __aenter__(self) -> "UsageService":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        pass
