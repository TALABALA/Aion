"""
AION Usage Metering Models

State-of-the-art data models for:
- Subscription tiers with comprehensive feature gating
- Usage metrics with multi-dimensional tracking
- Usage records with time-series aggregation
- Alert thresholds with configurable policies
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from decimal import Decimal
import json

from pydantic import BaseModel, Field, field_validator


# =============================================================================
# Subscription Tiers
# =============================================================================

class SubscriptionTier(str, Enum):
    """
    Subscription tiers with progressive feature sets.

    Implements industry-standard SaaS tiering with:
    - FREE: Acquisition tier with limited features
    - PRO: Individual power users
    - EXECUTIVE: Premium individuals with priority access
    - TEAM: Multi-seat collaborative tier
    - ENTERPRISE: Custom solutions (contact sales)
    """
    FREE = "free"
    PRO = "pro"
    EXECUTIVE = "executive"
    TEAM = "team"
    ENTERPRISE = "enterprise"

    @classmethod
    def from_string(cls, value: str) -> "SubscriptionTier":
        """Parse tier from string, case-insensitive."""
        try:
            return cls(value.lower())
        except ValueError:
            return cls.FREE


class MemoryRetention(str, Enum):
    """Memory retention policies by tier."""
    SEVEN_DAYS = "7d"
    THIRTY_DAYS = "30d"
    NINETY_DAYS = "90d"
    ONE_YEAR = "1y"
    PERMANENT = "permanent"

    def to_days(self) -> Optional[int]:
        """Convert to number of days, None for permanent."""
        mapping = {
            self.SEVEN_DAYS: 7,
            self.THIRTY_DAYS: 30,
            self.NINETY_DAYS: 90,
            self.ONE_YEAR: 365,
            self.PERMANENT: None,
        }
        return mapping.get(self)


class ExpertAccess(str, Enum):
    """Expert access levels by tier."""
    GENERAL_ONLY = "general_only"
    BASIC_EXPERTS = "basic_experts"  # 10 experts
    ALL_EXPERTS = "all_experts"       # 34 experts
    CUSTOM_EXPERTS = "custom_experts" # All + custom training


# =============================================================================
# Tier Limits Configuration
# =============================================================================

@dataclass
class TierLimits:
    """
    Comprehensive tier limit configuration.

    Implements SOTA usage limiting with:
    - Soft limits (warnings) at configurable thresholds
    - Hard limits with enforcement
    - Burst allowances for occasional spikes
    - Rollover policies for unused quota
    """
    tier: SubscriptionTier

    # Message limits
    messages_per_day: Optional[int] = None      # None = unlimited
    messages_per_month: Optional[int] = None

    # Token limits
    tokens_per_day: Optional[int] = None
    tokens_per_month: Optional[int] = None

    # Memory limits
    memories_max: Optional[int] = None
    memory_retention: MemoryRetention = MemoryRetention.PERMANENT
    memories_per_type: Optional[Dict[str, int]] = None  # Per-type limits

    # API limits
    api_calls_per_day: Optional[int] = None
    api_calls_per_month: Optional[int] = None
    api_rate_per_minute: int = 60

    # Storage limits
    storage_mb: Optional[int] = None
    documents_max: Optional[int] = None
    attachments_max: Optional[int] = None

    # Expert access
    expert_access: ExpertAccess = ExpertAccess.ALL_EXPERTS
    expert_invocations_per_day: Optional[int] = None
    allowed_experts: Optional[List[str]] = None  # None = all allowed

    # Team features (for TEAM tier)
    seats_included: int = 1
    seats_max: Optional[int] = None
    shared_memory_pool: bool = False
    admin_controls: bool = False

    # Priority and processing
    priority_processing: bool = False
    custom_expert_training: bool = False
    dedicated_support: bool = False
    sla_guarantee: Optional[str] = None

    # Soft limit thresholds (percentage)
    soft_limit_threshold: float = 0.8   # Warn at 80%
    hard_limit_threshold: float = 1.0   # Block at 100%

    # Grace period for overage (hours)
    grace_period_hours: int = 0

    # Burst allowance (percentage above limit allowed temporarily)
    burst_allowance: float = 0.0

    def is_unlimited(self, metric: str) -> bool:
        """Check if a metric is unlimited for this tier."""
        limit = getattr(self, metric, None)
        return limit is None

    def get_limit(self, metric: str) -> Optional[int]:
        """Get limit for a specific metric."""
        return getattr(self, metric, None)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "tier": self.tier.value,
            "messages_per_day": self.messages_per_day,
            "messages_per_month": self.messages_per_month,
            "tokens_per_day": self.tokens_per_day,
            "tokens_per_month": self.tokens_per_month,
            "memories_max": self.memories_max,
            "memory_retention": self.memory_retention.value,
            "api_calls_per_day": self.api_calls_per_day,
            "api_calls_per_month": self.api_calls_per_month,
            "api_rate_per_minute": self.api_rate_per_minute,
            "storage_mb": self.storage_mb,
            "documents_max": self.documents_max,
            "expert_access": self.expert_access.value,
            "expert_invocations_per_day": self.expert_invocations_per_day,
            "seats_included": self.seats_included,
            "priority_processing": self.priority_processing,
            "custom_expert_training": self.custom_expert_training,
        }


# =============================================================================
# Tier Configurations
# =============================================================================

TIER_CONFIGURATIONS: Dict[SubscriptionTier, TierLimits] = {
    SubscriptionTier.FREE: TierLimits(
        tier=SubscriptionTier.FREE,
        # Strict limits for free tier
        messages_per_day=10,
        messages_per_month=50,
        tokens_per_day=10000,
        tokens_per_month=100000,
        memories_max=100,
        memory_retention=MemoryRetention.SEVEN_DAYS,
        api_calls_per_day=0,  # No API access
        api_calls_per_month=0,
        api_rate_per_minute=10,
        storage_mb=50,
        documents_max=10,
        attachments_max=20,
        expert_access=ExpertAccess.GENERAL_ONLY,
        expert_invocations_per_day=10,
        allowed_experts=["general"],
        seats_included=1,
        soft_limit_threshold=0.8,
        grace_period_hours=0,
    ),

    SubscriptionTier.PRO: TierLimits(
        tier=SubscriptionTier.PRO,
        # Generous limits for pro users
        messages_per_day=None,  # Unlimited
        messages_per_month=None,
        tokens_per_day=None,
        tokens_per_month=None,
        memories_max=None,  # Unlimited
        memory_retention=MemoryRetention.PERMANENT,
        api_calls_per_day=100,
        api_calls_per_month=1000,
        api_rate_per_minute=60,
        storage_mb=5000,  # 5GB
        documents_max=500,
        attachments_max=1000,
        expert_access=ExpertAccess.ALL_EXPERTS,
        expert_invocations_per_day=None,  # Unlimited
        allowed_experts=None,  # All experts
        seats_included=1,
        soft_limit_threshold=0.8,
        grace_period_hours=24,
    ),

    SubscriptionTier.EXECUTIVE: TierLimits(
        tier=SubscriptionTier.EXECUTIVE,
        # Premium limits with priority
        messages_per_day=None,
        messages_per_month=None,
        tokens_per_day=None,
        tokens_per_month=None,
        memories_max=None,
        memory_retention=MemoryRetention.PERMANENT,
        api_calls_per_day=1000,
        api_calls_per_month=10000,
        api_rate_per_minute=120,
        storage_mb=50000,  # 50GB
        documents_max=5000,
        attachments_max=10000,
        expert_access=ExpertAccess.CUSTOM_EXPERTS,
        expert_invocations_per_day=None,
        allowed_experts=None,
        seats_included=1,
        priority_processing=True,
        custom_expert_training=True,
        dedicated_support=True,
        sla_guarantee="99.9%",
        soft_limit_threshold=0.9,
        grace_period_hours=72,
        burst_allowance=0.2,  # 20% burst allowed
    ),

    SubscriptionTier.TEAM: TierLimits(
        tier=SubscriptionTier.TEAM,
        # Multi-seat with shared resources
        messages_per_day=None,  # Per seat
        messages_per_month=None,
        tokens_per_day=None,
        tokens_per_month=None,
        memories_max=None,
        memory_retention=MemoryRetention.PERMANENT,
        api_calls_per_day=500,  # Per seat
        api_calls_per_month=5000,  # Per seat
        api_rate_per_minute=100,
        storage_mb=100000,  # 100GB shared
        documents_max=10000,
        attachments_max=50000,
        expert_access=ExpertAccess.ALL_EXPERTS,
        expert_invocations_per_day=None,
        allowed_experts=None,
        seats_included=5,
        seats_max=100,
        shared_memory_pool=True,
        admin_controls=True,
        priority_processing=True,
        soft_limit_threshold=0.8,
        grace_period_hours=48,
        burst_allowance=0.1,
    ),

    SubscriptionTier.ENTERPRISE: TierLimits(
        tier=SubscriptionTier.ENTERPRISE,
        # Fully customizable - defaults are very generous
        messages_per_day=None,
        messages_per_month=None,
        tokens_per_day=None,
        tokens_per_month=None,
        memories_max=None,
        memory_retention=MemoryRetention.PERMANENT,
        api_calls_per_day=None,
        api_calls_per_month=None,
        api_rate_per_minute=1000,
        storage_mb=None,  # Unlimited
        documents_max=None,
        attachments_max=None,
        expert_access=ExpertAccess.CUSTOM_EXPERTS,
        expert_invocations_per_day=None,
        allowed_experts=None,
        seats_included=10,
        seats_max=None,  # Unlimited
        shared_memory_pool=True,
        admin_controls=True,
        priority_processing=True,
        custom_expert_training=True,
        dedicated_support=True,
        sla_guarantee="99.99%",
        soft_limit_threshold=0.9,
        grace_period_hours=168,  # 1 week
        burst_allowance=0.5,
    ),
}


def get_tier_limits(tier: Union[SubscriptionTier, str]) -> TierLimits:
    """Get limits configuration for a tier."""
    if isinstance(tier, str):
        tier = SubscriptionTier.from_string(tier)
    return TIER_CONFIGURATIONS.get(tier, TIER_CONFIGURATIONS[SubscriptionTier.FREE])


# =============================================================================
# Usage Metrics
# =============================================================================

class UsageMetric(str, Enum):
    """
    Tracked usage metrics.

    Comprehensive metrics covering:
    - Consumption metrics (messages, tokens)
    - Resource metrics (memory, storage)
    - Access metrics (API calls, expert usage)
    """
    # Message metrics
    MESSAGES_SENT = "messages_sent"
    MESSAGES_RECEIVED = "messages_received"

    # Token metrics
    TOKENS_INPUT = "tokens_input"
    TOKENS_OUTPUT = "tokens_output"
    TOKENS_TOTAL = "tokens_total"

    # Memory metrics
    MEMORIES_TOTAL = "memories_total"
    MEMORIES_CREATED = "memories_created"
    MEMORIES_ACCESSED = "memories_accessed"
    MEMORIES_BY_TYPE = "memories_by_type"

    # API metrics
    API_CALLS = "api_calls"
    API_ERRORS = "api_errors"

    # Storage metrics
    STORAGE_USED_MB = "storage_used_mb"
    DOCUMENTS_COUNT = "documents_count"
    ATTACHMENTS_COUNT = "attachments_count"

    # Expert metrics
    EXPERT_INVOCATIONS = "expert_invocations"
    EXPERT_INVOCATIONS_BY_TYPE = "expert_invocations_by_type"

    # Session metrics
    SESSIONS_CREATED = "sessions_created"
    SESSION_DURATION_SECONDS = "session_duration_seconds"

    # Cost metrics (for internal tracking)
    ESTIMATED_COST_USD = "estimated_cost_usd"


class UsagePeriod(str, Enum):
    """Time periods for usage aggregation."""
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    YEARLY = "yearly"
    ALL_TIME = "all_time"

    def to_timedelta(self) -> Optional[timedelta]:
        """Convert to timedelta for calculations."""
        mapping = {
            self.HOURLY: timedelta(hours=1),
            self.DAILY: timedelta(days=1),
            self.WEEKLY: timedelta(weeks=1),
            self.MONTHLY: timedelta(days=30),
            self.YEARLY: timedelta(days=365),
            self.ALL_TIME: None,
        }
        return mapping.get(self)


# =============================================================================
# Usage Records
# =============================================================================

@dataclass
class UsageRecord:
    """
    Individual usage record for time-series storage.

    Designed for efficient aggregation with:
    - Hierarchical time keys
    - Pre-computed aggregates
    - Dimensional metadata
    """
    user_id: str
    metric: UsageMetric
    value: float
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Dimensional attributes for drill-down
    period: UsagePeriod = UsagePeriod.DAILY
    expert_type: Optional[str] = None
    memory_type: Optional[str] = None
    session_id: Optional[str] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "user_id": self.user_id,
            "metric": self.metric.value,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "period": self.period.value,
            "expert_type": self.expert_type,
            "memory_type": self.memory_type,
            "session_id": self.session_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UsageRecord":
        """Deserialize from dictionary."""
        return cls(
            user_id=data["user_id"],
            metric=UsageMetric(data["metric"]),
            value=data["value"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            period=UsagePeriod(data.get("period", "daily")),
            expert_type=data.get("expert_type"),
            memory_type=data.get("memory_type"),
            session_id=data.get("session_id"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class UsageSummary:
    """
    Aggregated usage summary for a user.

    Provides comprehensive view with:
    - Current period usage vs limits
    - Historical comparisons
    - Trend indicators
    """
    user_id: str
    tier: SubscriptionTier
    period: str  # e.g., "2025-01" for monthly
    period_start: datetime
    period_end: datetime

    # Usage by metric
    usage: Dict[str, "UsageMetricSummary"] = field(default_factory=dict)

    # Computed fields
    billing_cycle_end: Optional[datetime] = None
    days_remaining: int = 0

    # Team-specific
    team_id: Optional[str] = None
    seat_number: Optional[int] = None

    def to_response(self) -> Dict[str, Any]:
        """Convert to API response format."""
        return {
            "period": self.period,
            "usage": {
                name: summary.to_dict()
                for name, summary in self.usage.items()
            },
            "tier": self.tier.value,
            "billing_cycle_end": self.billing_cycle_end.isoformat() if self.billing_cycle_end else None,
            "days_remaining": self.days_remaining,
            "team_id": self.team_id,
        }


@dataclass
class UsageMetricSummary:
    """Summary for a single metric."""
    used: float
    limit: Optional[float]
    unlimited: bool = False

    # Thresholds
    soft_limit_reached: bool = False
    hard_limit_reached: bool = False

    # Percentage (0-100+)
    percentage: float = 0.0

    # Trend
    change_from_previous: Optional[float] = None  # Percentage change

    def to_dict(self) -> Dict[str, Any]:
        return {
            "used": self.used,
            "limit": self.limit,
            "unlimited": self.unlimited,
            "percentage": round(self.percentage, 1),
            "soft_limit_reached": self.soft_limit_reached,
            "hard_limit_reached": self.hard_limit_reached,
            "change_from_previous": self.change_from_previous,
        }


# =============================================================================
# Alerts
# =============================================================================

class AlertType(str, Enum):
    """Types of usage alerts."""
    APPROACHING_LIMIT = "approaching_limit"     # 80% threshold
    NEAR_LIMIT = "near_limit"                   # 90% threshold
    LIMIT_REACHED = "limit_reached"             # 100% threshold
    LIMIT_EXCEEDED = "limit_exceeded"           # Over 100%
    USAGE_SPIKE = "usage_spike"                 # Unusual increase
    RENEWAL_REMINDER = "renewal_reminder"       # Billing reminder
    TIER_UPGRADE_SUGGESTED = "tier_upgrade_suggested"
    RETENTION_EXPIRY = "retention_expiry"       # Memory about to expire


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class UsageAlert:
    """
    Usage alert for notification system.
    """
    alert_id: str
    user_id: str
    alert_type: AlertType
    severity: AlertSeverity
    metric: UsageMetric

    # Alert details
    current_value: float
    limit_value: Optional[float]
    percentage: float

    # Timing
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None

    # Message
    title: str = ""
    message: str = ""

    # Actions
    action_url: Optional[str] = None
    action_label: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "user_id": self.user_id,
            "alert_type": self.alert_type.value,
            "severity": self.severity.value,
            "metric": self.metric.value,
            "current_value": self.current_value,
            "limit_value": self.limit_value,
            "percentage": self.percentage,
            "created_at": self.created_at.isoformat(),
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "title": self.title,
            "message": self.message,
            "action_url": self.action_url,
            "action_label": self.action_label,
        }


# =============================================================================
# Subscription and Billing
# =============================================================================

class BillingCycle(str, Enum):
    """Billing cycle options."""
    MONTHLY = "monthly"
    YEARLY = "yearly"


@dataclass
class Subscription:
    """
    User subscription information.
    """
    user_id: str
    tier: SubscriptionTier
    billing_cycle: BillingCycle = BillingCycle.MONTHLY

    # Dates
    started_at: datetime = field(default_factory=datetime.utcnow)
    current_period_start: datetime = field(default_factory=datetime.utcnow)
    current_period_end: Optional[datetime] = None
    canceled_at: Optional[datetime] = None

    # Team
    team_id: Optional[str] = None
    is_team_admin: bool = False
    seat_count: int = 1

    # Custom limits (overrides tier defaults)
    custom_limits: Optional[Dict[str, Any]] = None

    # Status
    is_active: bool = True
    is_trial: bool = False
    trial_ends_at: Optional[datetime] = None

    def get_limits(self) -> TierLimits:
        """Get effective limits for this subscription."""
        base_limits = get_tier_limits(self.tier)

        if self.custom_limits:
            # Apply custom overrides
            for key, value in self.custom_limits.items():
                if hasattr(base_limits, key):
                    setattr(base_limits, key, value)

        return base_limits

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "tier": self.tier.value,
            "billing_cycle": self.billing_cycle.value,
            "started_at": self.started_at.isoformat(),
            "current_period_start": self.current_period_start.isoformat(),
            "current_period_end": self.current_period_end.isoformat() if self.current_period_end else None,
            "is_active": self.is_active,
            "is_trial": self.is_trial,
            "team_id": self.team_id,
            "seat_count": self.seat_count,
        }
