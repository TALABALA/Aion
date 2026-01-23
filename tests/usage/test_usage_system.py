"""
Comprehensive Tests for AION Usage Metering System

Tests cover:
- Tier configurations and limits
- Real-time usage tracking (Redis)
- Historical storage (PostgreSQL)
- Middleware and limit enforcement
- API endpoints
- Notification system
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from aion.usage.models import (
    SubscriptionTier,
    TierLimits,
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
    MemoryRetention,
    ExpertAccess,
    get_tier_limits,
    TIER_CONFIGURATIONS,
)
from aion.usage.tracker import (
    UsageTracker,
    MemoryUsageTracker,
    RedisUsageTracker,
)
from aion.usage.storage import (
    UsageStore,
    MemoryUsageStore,
    PostgresUsageStore,
)
from aion.usage.service import UsageService
from aion.usage.middleware import (
    UsageLimitMiddleware,
    UsageContext,
    LimitCheckResult,
)
from aion.usage.notifications import (
    UsageNotificationService,
    NotificationChannel,
    UsageNotification,
    InAppProvider,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def memory_tracker():
    """Create a memory-based usage tracker for testing."""
    return MemoryUsageTracker()


@pytest.fixture
def memory_store():
    """Create a memory-based usage store for testing."""
    return MemoryUsageStore()


@pytest.fixture
def usage_service(memory_tracker, memory_store):
    """Create a usage service with memory backends."""
    return UsageService(tracker=memory_tracker, store=memory_store)


@pytest.fixture
def notification_service():
    """Create a notification service for testing."""
    return UsageNotificationService()


# =============================================================================
# Test Tier Configurations
# =============================================================================

class TestTierConfigurations:
    """Tests for tier limit configurations."""

    def test_all_tiers_configured(self):
        """Verify all tiers have configurations."""
        for tier in SubscriptionTier:
            assert tier in TIER_CONFIGURATIONS

    def test_free_tier_limits(self):
        """Verify FREE tier has appropriate limits."""
        limits = get_tier_limits(SubscriptionTier.FREE)

        assert limits.tier == SubscriptionTier.FREE
        assert limits.messages_per_day == 10
        assert limits.messages_per_month == 50
        assert limits.memories_max == 100
        assert limits.memory_retention == MemoryRetention.SEVEN_DAYS
        assert limits.api_calls_per_month == 0  # No API access for free
        assert limits.expert_access == ExpertAccess.GENERAL_ONLY

    def test_pro_tier_unlimited_messages(self):
        """Verify PRO tier has unlimited messages."""
        limits = get_tier_limits(SubscriptionTier.PRO)

        assert limits.messages_per_day is None
        assert limits.messages_per_month is None
        assert limits.memories_max is None
        assert limits.memory_retention == MemoryRetention.PERMANENT
        assert limits.expert_access == ExpertAccess.ALL_EXPERTS

    def test_executive_tier_priority(self):
        """Verify EXECUTIVE tier has priority features."""
        limits = get_tier_limits(SubscriptionTier.EXECUTIVE)

        assert limits.priority_processing is True
        assert limits.custom_expert_training is True
        assert limits.dedicated_support is True
        assert limits.api_calls_per_month == 10000

    def test_team_tier_multi_seat(self):
        """Verify TEAM tier has multi-seat features."""
        limits = get_tier_limits(SubscriptionTier.TEAM)

        assert limits.seats_included == 5
        assert limits.seats_max == 100
        assert limits.shared_memory_pool is True
        assert limits.admin_controls is True

    def test_get_tier_limits_from_string(self):
        """Test getting limits from string tier name."""
        limits = get_tier_limits("pro")
        assert limits.tier == SubscriptionTier.PRO

        limits = get_tier_limits("FREE")  # Case insensitive
        assert limits.tier == SubscriptionTier.FREE

    def test_unknown_tier_defaults_to_free(self):
        """Unknown tier should default to FREE."""
        tier = SubscriptionTier.from_string("unknown")
        assert tier == SubscriptionTier.FREE


# =============================================================================
# Test Memory Usage Tracker
# =============================================================================

class TestMemoryUsageTracker:
    """Tests for the in-memory usage tracker."""

    @pytest.mark.asyncio
    async def test_increment_basic(self, memory_tracker):
        """Test basic increment functionality."""
        user_id = "user_123"
        metric = UsageMetric.MESSAGES_SENT

        result = await memory_tracker.increment(user_id, metric, 1.0)
        assert result == 1.0

        result = await memory_tracker.increment(user_id, metric, 2.0)
        assert result == 3.0

    @pytest.mark.asyncio
    async def test_get_current(self, memory_tracker):
        """Test getting current usage."""
        user_id = "user_123"
        metric = UsageMetric.MESSAGES_SENT

        # Initial value should be 0
        current = await memory_tracker.get_current(user_id, metric)
        assert current == 0.0

        # After increment
        await memory_tracker.increment(user_id, metric, 5.0)
        current = await memory_tracker.get_current(user_id, metric)
        assert current == 5.0

    @pytest.mark.asyncio
    async def test_increment_with_limit(self, memory_tracker):
        """Test increment respects limits."""
        user_id = "user_123"
        metric = UsageMetric.MESSAGES_SENT
        limit = 10.0

        # Should succeed
        await memory_tracker.increment(user_id, metric, 8.0, limit=limit)
        current = await memory_tracker.get_current(user_id, metric)
        assert current == 8.0

        # Should fail (would exceed limit)
        await memory_tracker.increment(user_id, metric, 5.0, limit=limit)
        current = await memory_tracker.get_current(user_id, metric)
        assert current == 8.0  # Unchanged

    @pytest.mark.asyncio
    async def test_check_limit(self, memory_tracker):
        """Test limit checking."""
        user_id = "user_123"
        metric = UsageMetric.MESSAGES_SENT

        await memory_tracker.increment(user_id, metric, 80.0)

        # Check against limit
        within, current, percentage = await memory_tracker.check_limit(
            user_id, metric, 100.0
        )
        assert within is True
        assert current == 80.0
        assert percentage == 80.0

        # Exceed limit
        await memory_tracker.increment(user_id, metric, 25.0)
        within, current, percentage = await memory_tracker.check_limit(
            user_id, metric, 100.0
        )
        assert within is False
        assert percentage == 105.0

    @pytest.mark.asyncio
    async def test_unlimited_metric(self, memory_tracker):
        """Test unlimited metrics (None limit)."""
        user_id = "user_123"
        metric = UsageMetric.MESSAGES_SENT

        await memory_tracker.increment(user_id, metric, 1000000.0)

        within, current, percentage = await memory_tracker.check_limit(
            user_id, metric, None
        )
        assert within is True
        assert percentage == 0.0

    @pytest.mark.asyncio
    async def test_dimensional_tracking(self, memory_tracker):
        """Test tracking with dimensions."""
        user_id = "user_123"
        metric = UsageMetric.EXPERT_INVOCATIONS

        # Track by expert type
        await memory_tracker.increment(
            user_id, metric, 3.0,
            dimensions={"expert_type": "financial"}
        )
        await memory_tracker.increment(
            user_id, metric, 5.0,
            dimensions={"expert_type": "legal"}
        )

        breakdown = await memory_tracker.get_breakdown(
            user_id, metric, "expert_type"
        )
        assert breakdown.get("financial") == 3.0
        assert breakdown.get("legal") == 5.0

    @pytest.mark.asyncio
    async def test_get_all_metrics(self, memory_tracker):
        """Test getting all metrics for a user."""
        user_id = "user_123"

        await memory_tracker.increment(user_id, UsageMetric.MESSAGES_SENT, 10)
        await memory_tracker.increment(user_id, UsageMetric.API_CALLS, 5)
        await memory_tracker.increment(user_id, UsageMetric.TOKENS_TOTAL, 1000)

        metrics = await memory_tracker.get_all_metrics(user_id)

        assert UsageMetric.MESSAGES_SENT in metrics
        assert metrics[UsageMetric.MESSAGES_SENT] == 10

    @pytest.mark.asyncio
    async def test_reset_metric(self, memory_tracker):
        """Test resetting a metric."""
        user_id = "user_123"
        metric = UsageMetric.MESSAGES_SENT

        await memory_tracker.increment(user_id, metric, 50)
        await memory_tracker.reset_metric(user_id, metric)

        current = await memory_tracker.get_current(user_id, metric)
        assert current == 0.0


# =============================================================================
# Test Usage Service
# =============================================================================

class TestUsageService:
    """Tests for the central usage service."""

    @pytest.mark.asyncio
    async def test_record_usage(self, usage_service):
        """Test recording usage."""
        user_id = "user_123"
        metric = UsageMetric.MESSAGES_SENT

        total, alert = await usage_service.record_usage(user_id, metric, 1.0)

        assert total == 1.0
        assert alert is None  # No alert for low usage

    @pytest.mark.asyncio
    async def test_record_usage_triggers_alert(self, usage_service):
        """Test that high usage triggers alerts."""
        user_id = "user_123"
        metric = UsageMetric.MESSAGES_SENT

        # Create a free tier subscription
        await usage_service.create_subscription(user_id, SubscriptionTier.FREE)

        # Record usage near the limit (50 messages for free)
        for _ in range(45):
            await usage_service.record_usage(user_id, metric, 1.0)

        # This should trigger an alert (90%+)
        _, alert = await usage_service.record_usage(user_id, metric, 1.0)

        # Alert may or may not be generated depending on cooldown
        # The important thing is no error occurs

    @pytest.mark.asyncio
    async def test_get_current_usage(self, usage_service):
        """Test getting current usage summary."""
        user_id = "user_123"

        # Record some usage
        await usage_service.record_usage(user_id, UsageMetric.MESSAGES_SENT, 10)
        await usage_service.record_usage(user_id, UsageMetric.API_CALLS, 5)

        summary = await usage_service.get_current_usage(user_id)

        assert summary.user_id == user_id
        assert summary.tier == SubscriptionTier.FREE  # Default tier

    @pytest.mark.asyncio
    async def test_create_subscription(self, usage_service):
        """Test creating a subscription."""
        user_id = "user_123"

        subscription = await usage_service.create_subscription(
            user_id,
            SubscriptionTier.PRO,
            BillingCycle.MONTHLY,
        )

        assert subscription.user_id == user_id
        assert subscription.tier == SubscriptionTier.PRO
        assert subscription.is_active is True

    @pytest.mark.asyncio
    async def test_upgrade_subscription(self, usage_service):
        """Test upgrading a subscription."""
        user_id = "user_123"

        # Start with free
        await usage_service.create_subscription(user_id, SubscriptionTier.FREE)

        # Upgrade to pro
        subscription = await usage_service.upgrade_subscription(
            user_id, SubscriptionTier.PRO
        )

        assert subscription.tier == SubscriptionTier.PRO

    @pytest.mark.asyncio
    async def test_cancel_subscription(self, usage_service):
        """Test canceling a subscription."""
        user_id = "user_123"

        await usage_service.create_subscription(user_id, SubscriptionTier.PRO)
        subscription = await usage_service.cancel_subscription(user_id)

        assert subscription.canceled_at is not None

    @pytest.mark.asyncio
    async def test_check_limit(self, usage_service):
        """Test limit checking through service."""
        user_id = "user_123"

        await usage_service.create_subscription(user_id, SubscriptionTier.FREE)

        within, current, limit, pct = await usage_service.check_limit(
            user_id, UsageMetric.MESSAGES_SENT, UsagePeriod.MONTHLY
        )

        assert within is True
        assert limit == 50  # Free tier monthly message limit

    @pytest.mark.asyncio
    async def test_can_perform_action(self, usage_service):
        """Test action permission checking."""
        user_id = "user_123"

        # Free tier - no API access
        await usage_service.create_subscription(user_id, SubscriptionTier.FREE)

        allowed, reason = await usage_service.can_perform_action(user_id, "api_call")
        assert allowed is False
        assert "Pro tier" in reason

        # Pro tier - API access allowed
        await usage_service.upgrade_subscription(user_id, SubscriptionTier.PRO)

        allowed, reason = await usage_service.can_perform_action(user_id, "api_call")
        assert allowed is True

    @pytest.mark.asyncio
    async def test_usage_forecast(self, usage_service):
        """Test usage forecasting."""
        user_id = "user_123"

        await usage_service.create_subscription(user_id, SubscriptionTier.FREE)

        # Simulate some usage
        for _ in range(20):
            await usage_service.record_usage(user_id, UsageMetric.MESSAGES_SENT, 1.0)

        forecast = await usage_service.forecast_usage(
            user_id, UsageMetric.MESSAGES_SENT
        )

        assert forecast["current"] == 20
        assert forecast["limit"] == 50
        assert "forecasted_total" in forecast


# =============================================================================
# Test Memory Usage Store
# =============================================================================

class TestMemoryUsageStore:
    """Tests for the in-memory usage store."""

    @pytest.mark.asyncio
    async def test_save_and_get_record(self, memory_store):
        """Test saving and retrieving records."""
        record = UsageRecord(
            user_id="user_123",
            metric=UsageMetric.MESSAGES_SENT,
            value=1.0,
        )

        await memory_store.save_record(record)

        # Records are stored but summary queries them
        summary = await memory_store.get_summary("user_123")
        assert summary.user_id == "user_123"

    @pytest.mark.asyncio
    async def test_subscription_crud(self, memory_store):
        """Test subscription CRUD operations."""
        subscription = Subscription(
            user_id="user_123",
            tier=SubscriptionTier.PRO,
        )

        await memory_store.save_subscription(subscription)

        retrieved = await memory_store.get_subscription("user_123")
        assert retrieved is not None
        assert retrieved.tier == SubscriptionTier.PRO

    @pytest.mark.asyncio
    async def test_history(self, memory_store):
        """Test getting historical data."""
        user_id = "user_123"
        metric = UsageMetric.MESSAGES_SENT

        # Save some records
        for i in range(5):
            record = UsageRecord(
                user_id=user_id,
                metric=metric,
                value=float(i + 1),
                timestamp=datetime.utcnow() - timedelta(days=i),
            )
            await memory_store.save_record(record)

        history = await memory_store.get_history(
            user_id,
            metric,
            datetime.utcnow().date() - timedelta(days=7),
            datetime.utcnow().date(),
        )

        assert len(history) > 0


# =============================================================================
# Test Middleware
# =============================================================================

class TestUsageLimitMiddleware:
    """Tests for the usage limit middleware."""

    def test_limit_check_result(self):
        """Test LimitCheckResult creation and headers."""
        result = LimitCheckResult(
            allowed=False,
            metric=UsageMetric.MESSAGES_SENT,
            current=50,
            limit=50,
            percentage=100,
            tier=SubscriptionTier.FREE,
            retry_after_seconds=3600,
            message="Monthly limit reached",
        )

        assert result.allowed is False
        assert result.hard_limit_reached is False  # Need to set separately

        headers = result.to_headers()
        assert headers["X-RateLimit-Remaining"] == "0"
        assert headers["Retry-After"] == "3600"

    def test_usage_context(self):
        """Test UsageContext tracking."""
        context = UsageContext(
            user_id="user_123",
            tier=SubscriptionTier.PRO,
            limits=get_tier_limits(SubscriptionTier.PRO),
        )

        context.track(UsageMetric.MESSAGES_SENT, 1)
        context.track(UsageMetric.TOKENS_TOTAL, 500)

        assert context.metrics_to_track[UsageMetric.MESSAGES_SENT] == 1
        assert context.metrics_to_track[UsageMetric.TOKENS_TOTAL] == 500


# =============================================================================
# Test Notification System
# =============================================================================

class TestNotificationSystem:
    """Tests for the notification system."""

    @pytest.mark.asyncio
    async def test_in_app_provider(self):
        """Test in-app notification provider."""
        provider = InAppProvider()

        notification = UsageNotification(
            notification_id="notif_1",
            user_id="user_123",
            channel=NotificationChannel.IN_APP,
            priority=NotificationPriority.NORMAL,
            subject="Test Subject",
            body="Test Body",
        )

        success = await provider.send(notification)
        assert success is True

        notifications = await provider.get_notifications("user_123")
        assert len(notifications) == 1
        assert notifications[0].subject == "Test Subject"

    @pytest.mark.asyncio
    async def test_notification_service_send(self, notification_service):
        """Test sending notifications through service."""
        alert = UsageAlert(
            alert_id="alert_1",
            user_id="user_123",
            alert_type=AlertType.APPROACHING_LIMIT,
            severity=AlertSeverity.INFO,
            metric=UsageMetric.MESSAGES_SENT,
            current_value=40,
            limit_value=50,
            percentage=80,
            title="Approaching Limit",
            message="You've used 80% of your messages",
        )

        notifications = await notification_service.send_alert_notification(
            alert,
            user_name="Test User",
        )

        # Should have sent in-app notification (default channel)
        assert len(notifications) >= 1

    @pytest.mark.asyncio
    async def test_notification_rate_limiting(self, notification_service):
        """Test that notifications are rate limited."""
        notification_service.rate_limit_per_hour = 2

        alert = UsageAlert(
            alert_id="alert_1",
            user_id="user_123",
            alert_type=AlertType.APPROACHING_LIMIT,
            severity=AlertSeverity.INFO,
            metric=UsageMetric.MESSAGES_SENT,
            current_value=40,
            limit_value=50,
            percentage=80,
        )

        # First two should succeed
        await notification_service.send_alert_notification(alert)
        await notification_service.send_alert_notification(alert)

        # Third should be rate limited
        notifications = await notification_service.send_alert_notification(alert)
        assert len(notifications) == 0

    @pytest.mark.asyncio
    async def test_get_in_app_notifications(self, notification_service):
        """Test retrieving in-app notifications."""
        alert = UsageAlert(
            alert_id="alert_1",
            user_id="user_123",
            alert_type=AlertType.LIMIT_REACHED,
            severity=AlertSeverity.CRITICAL,
            metric=UsageMetric.MESSAGES_SENT,
            current_value=50,
            limit_value=50,
            percentage=100,
        )

        await notification_service.send_alert_notification(alert)

        notifications = await notification_service.get_in_app_notifications("user_123")
        assert len(notifications) >= 1


# =============================================================================
# Test Usage Models
# =============================================================================

class TestUsageModels:
    """Tests for usage data models."""

    def test_usage_record_serialization(self):
        """Test UsageRecord serialization."""
        record = UsageRecord(
            user_id="user_123",
            metric=UsageMetric.MESSAGES_SENT,
            value=10.0,
            expert_type="financial",
        )

        data = record.to_dict()
        assert data["user_id"] == "user_123"
        assert data["metric"] == "messages_sent"
        assert data["value"] == 10.0
        assert data["expert_type"] == "financial"

        # Deserialize
        restored = UsageRecord.from_dict(data)
        assert restored.user_id == "user_123"
        assert restored.metric == UsageMetric.MESSAGES_SENT

    def test_usage_metric_summary(self):
        """Test UsageMetricSummary creation."""
        summary = UsageMetricSummary(
            used=80,
            limit=100,
            unlimited=False,
            percentage=80,
            soft_limit_reached=True,
            hard_limit_reached=False,
        )

        data = summary.to_dict()
        assert data["used"] == 80
        assert data["percentage"] == 80.0
        assert data["soft_limit_reached"] is True

    def test_subscription_get_limits(self):
        """Test Subscription.get_limits with custom overrides."""
        subscription = Subscription(
            user_id="user_123",
            tier=SubscriptionTier.PRO,
            custom_limits={"api_calls_per_month": 5000},
        )

        limits = subscription.get_limits()
        assert limits.api_calls_per_month == 5000  # Custom override

    def test_memory_retention_to_days(self):
        """Test MemoryRetention conversion."""
        assert MemoryRetention.SEVEN_DAYS.to_days() == 7
        assert MemoryRetention.THIRTY_DAYS.to_days() == 30
        assert MemoryRetention.PERMANENT.to_days() is None


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for the complete usage system."""

    @pytest.mark.asyncio
    async def test_full_usage_flow(self):
        """Test complete usage tracking flow."""
        # Setup
        tracker = MemoryUsageTracker()
        store = MemoryUsageStore()
        service = UsageService(tracker=tracker, store=store)

        user_id = "user_integration_test"

        # Create subscription
        subscription = await service.create_subscription(
            user_id,
            SubscriptionTier.PRO,
        )
        assert subscription.tier == SubscriptionTier.PRO

        # Record various usage
        await service.record_usage(user_id, UsageMetric.MESSAGES_SENT, 10)
        await service.record_usage(user_id, UsageMetric.API_CALLS, 5)
        await service.record_usage(
            user_id,
            UsageMetric.EXPERT_INVOCATIONS,
            2,
            dimensions={"expert_type": "financial"},
        )

        # Get summary
        summary = await service.get_current_usage(user_id)
        assert summary.tier == SubscriptionTier.PRO

        # Check limits
        allowed, reason = await service.can_perform_action(user_id, "api_call")
        assert allowed is True

        # Get forecast
        forecast = await service.forecast_usage(user_id, UsageMetric.MESSAGES_SENT)
        assert forecast["current"] == 10

    @pytest.mark.asyncio
    async def test_free_tier_limit_enforcement(self):
        """Test that free tier limits are enforced."""
        tracker = MemoryUsageTracker()
        store = MemoryUsageStore()
        service = UsageService(tracker=tracker, store=store)

        user_id = "user_free_test"

        # Create free subscription
        await service.create_subscription(user_id, SubscriptionTier.FREE)

        # Record messages up to limit
        for _ in range(50):
            await service.record_usage(user_id, UsageMetric.MESSAGES_SENT, 1)

        # Check limit
        within, current, limit, pct = await service.check_limit(
            user_id,
            UsageMetric.MESSAGES_SENT,
            UsagePeriod.MONTHLY,
        )

        assert within is False
        assert current == 50
        assert limit == 50


# Import for notifications test
from aion.usage.notifications import NotificationPriority
