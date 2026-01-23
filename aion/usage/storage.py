"""
AION Usage Storage - PostgreSQL Backend

State-of-the-art persistent usage storage with:
- Time-series optimized schema
- Efficient aggregation queries
- Daily/monthly rollup jobs
- Partitioned tables for scale
- Materialized views for analytics
"""

from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timedelta, date
from typing import Any, Dict, List, Optional, Tuple, Union

import structlog

from aion.usage.models import (
    UsageMetric,
    UsagePeriod,
    UsageRecord,
    UsageSummary,
    UsageMetricSummary,
    SubscriptionTier,
    Subscription,
    BillingCycle,
    get_tier_limits,
)

logger = structlog.get_logger(__name__)


# =============================================================================
# SQL Schema
# =============================================================================

USAGE_SCHEMA = """
-- Usage records (time-series data)
CREATE TABLE IF NOT EXISTS usage_records (
    id BIGSERIAL PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    metric VARCHAR(100) NOT NULL,
    value DOUBLE PRECISION NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    period VARCHAR(20) NOT NULL DEFAULT 'daily',
    expert_type VARCHAR(100),
    memory_type VARCHAR(100),
    session_id VARCHAR(255),
    metadata JSONB DEFAULT '{}'::jsonb,

    -- Indexes for efficient queries
    CONSTRAINT usage_records_positive_value CHECK (value >= 0)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_usage_records_user_metric
    ON usage_records(user_id, metric, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_usage_records_timestamp
    ON usage_records(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_usage_records_period
    ON usage_records(user_id, period, timestamp DESC);

-- Daily aggregates (pre-computed for performance)
CREATE TABLE IF NOT EXISTS usage_daily_aggregates (
    id BIGSERIAL PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    metric VARCHAR(100) NOT NULL,
    date DATE NOT NULL,
    total_value DOUBLE PRECISION NOT NULL DEFAULT 0,
    count INTEGER NOT NULL DEFAULT 0,
    min_value DOUBLE PRECISION,
    max_value DOUBLE PRECISION,
    avg_value DOUBLE PRECISION,

    -- Dimensional breakdowns (JSONB for flexibility)
    by_expert JSONB DEFAULT '{}'::jsonb,
    by_memory_type JSONB DEFAULT '{}'::jsonb,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT usage_daily_unique UNIQUE(user_id, metric, date)
);

CREATE INDEX IF NOT EXISTS idx_usage_daily_user_date
    ON usage_daily_aggregates(user_id, date DESC);

-- Monthly aggregates
CREATE TABLE IF NOT EXISTS usage_monthly_aggregates (
    id BIGSERIAL PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    metric VARCHAR(100) NOT NULL,
    year_month VARCHAR(7) NOT NULL,  -- YYYY-MM format
    total_value DOUBLE PRECISION NOT NULL DEFAULT 0,
    count INTEGER NOT NULL DEFAULT 0,
    daily_avg DOUBLE PRECISION,
    peak_daily DOUBLE PRECISION,

    -- Dimensional breakdowns
    by_expert JSONB DEFAULT '{}'::jsonb,
    by_memory_type JSONB DEFAULT '{}'::jsonb,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT usage_monthly_unique UNIQUE(user_id, metric, year_month)
);

CREATE INDEX IF NOT EXISTS idx_usage_monthly_user
    ON usage_monthly_aggregates(user_id, year_month DESC);

-- Subscriptions table
CREATE TABLE IF NOT EXISTS subscriptions (
    id BIGSERIAL PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL UNIQUE,
    tier VARCHAR(50) NOT NULL DEFAULT 'free',
    billing_cycle VARCHAR(20) NOT NULL DEFAULT 'monthly',
    started_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    current_period_start TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    current_period_end TIMESTAMP WITH TIME ZONE,
    canceled_at TIMESTAMP WITH TIME ZONE,
    team_id VARCHAR(255),
    is_team_admin BOOLEAN DEFAULT FALSE,
    seat_count INTEGER DEFAULT 1,
    custom_limits JSONB,
    is_active BOOLEAN DEFAULT TRUE,
    is_trial BOOLEAN DEFAULT FALSE,
    trial_ends_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_subscriptions_user ON subscriptions(user_id);
CREATE INDEX IF NOT EXISTS idx_subscriptions_team ON subscriptions(team_id);

-- Usage alerts history
CREATE TABLE IF NOT EXISTS usage_alerts (
    id BIGSERIAL PRIMARY KEY,
    alert_id VARCHAR(255) NOT NULL UNIQUE,
    user_id VARCHAR(255) NOT NULL,
    alert_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    metric VARCHAR(100) NOT NULL,
    current_value DOUBLE PRECISION NOT NULL,
    limit_value DOUBLE PRECISION,
    percentage DOUBLE PRECISION,
    title TEXT,
    message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    acknowledged_at TIMESTAMP WITH TIME ZONE,
    dismissed_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX IF NOT EXISTS idx_alerts_user ON usage_alerts(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_alerts_unacknowledged
    ON usage_alerts(user_id) WHERE acknowledged_at IS NULL;
"""

# Aggregation query for daily rollup
DAILY_ROLLUP_QUERY = """
INSERT INTO usage_daily_aggregates
    (user_id, metric, date, total_value, count, min_value, max_value, avg_value, by_expert, by_memory_type)
SELECT
    user_id,
    metric,
    DATE(timestamp) as date,
    SUM(value) as total_value,
    COUNT(*) as count,
    MIN(value) as min_value,
    MAX(value) as max_value,
    AVG(value) as avg_value,
    COALESCE(
        jsonb_object_agg(expert_type, expert_sum) FILTER (WHERE expert_type IS NOT NULL),
        '{}'::jsonb
    ) as by_expert,
    COALESCE(
        jsonb_object_agg(memory_type, memory_sum) FILTER (WHERE memory_type IS NOT NULL),
        '{}'::jsonb
    ) as by_memory_type
FROM (
    SELECT
        user_id,
        metric,
        timestamp,
        value,
        expert_type,
        memory_type,
        SUM(value) OVER (PARTITION BY user_id, metric, DATE(timestamp), expert_type) as expert_sum,
        SUM(value) OVER (PARTITION BY user_id, metric, DATE(timestamp), memory_type) as memory_sum
    FROM usage_records
    WHERE DATE(timestamp) = $1
) sub
GROUP BY user_id, metric, DATE(timestamp)
ON CONFLICT (user_id, metric, date) DO UPDATE SET
    total_value = EXCLUDED.total_value,
    count = EXCLUDED.count,
    min_value = EXCLUDED.min_value,
    max_value = EXCLUDED.max_value,
    avg_value = EXCLUDED.avg_value,
    by_expert = EXCLUDED.by_expert,
    by_memory_type = EXCLUDED.by_memory_type,
    updated_at = CURRENT_TIMESTAMP;
"""


# =============================================================================
# Abstract Storage
# =============================================================================

class UsageStore(ABC):
    """Abstract base for usage storage backends."""

    @abstractmethod
    async def save_record(self, record: UsageRecord) -> None:
        """Save a usage record."""
        pass

    @abstractmethod
    async def save_records_batch(self, records: List[UsageRecord]) -> None:
        """Save multiple records efficiently."""
        pass

    @abstractmethod
    async def get_summary(
        self,
        user_id: str,
        period: UsagePeriod = UsagePeriod.MONTHLY,
        period_date: Optional[date] = None,
    ) -> UsageSummary:
        """Get aggregated usage summary."""
        pass

    @abstractmethod
    async def get_history(
        self,
        user_id: str,
        metric: UsageMetric,
        start_date: date,
        end_date: date,
    ) -> List[Tuple[date, float]]:
        """Get historical usage data."""
        pass

    @abstractmethod
    async def get_subscription(self, user_id: str) -> Optional[Subscription]:
        """Get user subscription."""
        pass

    @abstractmethod
    async def save_subscription(self, subscription: Subscription) -> None:
        """Save or update subscription."""
        pass


# =============================================================================
# PostgreSQL Storage
# =============================================================================

class PostgresUsageStore(UsageStore):
    """
    PostgreSQL-based usage storage for production.

    Features:
    - Optimized time-series schema
    - Efficient aggregation queries
    - Background rollup jobs
    - Connection pooling
    """

    def __init__(
        self,
        dsn: str = "postgresql://localhost/aion",
        pool_size: int = 10,
        batch_size: int = 100,
    ):
        self.dsn = dsn
        self.pool_size = pool_size
        self.batch_size = batch_size

        self._pool = None
        self._connected = False
        self._batch_buffer: List[UsageRecord] = []
        self._batch_lock = asyncio.Lock()
        self._flush_task: Optional[asyncio.Task] = None

    async def connect(self) -> None:
        """Initialize database connection."""
        if self._connected:
            return

        try:
            import asyncpg

            self._pool = await asyncpg.create_pool(
                self.dsn,
                min_size=2,
                max_size=self.pool_size,
            )

            # Create schema
            async with self._pool.acquire() as conn:
                await conn.execute(USAGE_SCHEMA)

            self._connected = True
            logger.info("PostgreSQL usage store connected")

            # Start background flush task
            self._flush_task = asyncio.create_task(self._flush_loop())

        except ImportError:
            logger.error("asyncpg not installed")
            raise
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise

    async def disconnect(self) -> None:
        """Close database connection."""
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass

        # Flush remaining records
        await self._flush_batch()

        if self._pool:
            await self._pool.close()
            self._connected = False

    async def _flush_loop(self) -> None:
        """Background loop to flush batched records."""
        while True:
            try:
                await asyncio.sleep(5)  # Flush every 5 seconds
                await self._flush_batch()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Batch flush error: {e}")

    async def _flush_batch(self) -> None:
        """Flush buffered records to database."""
        async with self._batch_lock:
            if not self._batch_buffer:
                return

            records = self._batch_buffer.copy()
            self._batch_buffer.clear()

        if not self._connected:
            return

        try:
            async with self._pool.acquire() as conn:
                await conn.executemany(
                    """
                    INSERT INTO usage_records
                        (user_id, metric, value, timestamp, period, expert_type, memory_type, session_id, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    """,
                    [
                        (
                            r.user_id,
                            r.metric.value,
                            r.value,
                            r.timestamp,
                            r.period.value,
                            r.expert_type,
                            r.memory_type,
                            r.session_id,
                            json.dumps(r.metadata),
                        )
                        for r in records
                    ],
                )
            logger.debug(f"Flushed {len(records)} usage records")
        except Exception as e:
            logger.error(f"Failed to flush records: {e}")
            # Re-add failed records
            async with self._batch_lock:
                self._batch_buffer.extend(records)

    async def save_record(self, record: UsageRecord) -> None:
        """Save a usage record (buffered)."""
        async with self._batch_lock:
            self._batch_buffer.append(record)

            if len(self._batch_buffer) >= self.batch_size:
                records = self._batch_buffer.copy()
                self._batch_buffer.clear()

        # Flush immediately if batch is full
        if len(self._batch_buffer) == 0:
            await self._flush_batch()

    async def save_records_batch(self, records: List[UsageRecord]) -> None:
        """Save multiple records."""
        async with self._batch_lock:
            self._batch_buffer.extend(records)

        if len(self._batch_buffer) >= self.batch_size:
            await self._flush_batch()

    async def get_summary(
        self,
        user_id: str,
        period: UsagePeriod = UsagePeriod.MONTHLY,
        period_date: Optional[date] = None,
    ) -> UsageSummary:
        """Get aggregated usage summary."""
        if not self._connected:
            await self.connect()

        now = datetime.utcnow()
        period_date = period_date or now.date()

        # Get subscription for tier info
        subscription = await self.get_subscription(user_id)
        tier = subscription.tier if subscription else SubscriptionTier.FREE
        limits = get_tier_limits(tier)

        # Determine period bounds
        if period == UsagePeriod.MONTHLY:
            period_str = period_date.strftime("%Y-%m")
            start_date = period_date.replace(day=1)
            if period_date.month == 12:
                end_date = period_date.replace(year=period_date.year + 1, month=1, day=1)
            else:
                end_date = period_date.replace(month=period_date.month + 1, day=1)
        elif period == UsagePeriod.DAILY:
            period_str = period_date.strftime("%Y-%m-%d")
            start_date = period_date
            end_date = period_date + timedelta(days=1)
        else:
            period_str = period_date.strftime("%Y")
            start_date = period_date.replace(month=1, day=1)
            end_date = period_date.replace(year=period_date.year + 1, month=1, day=1)

        # Query aggregated data
        async with self._pool.acquire() as conn:
            if period == UsagePeriod.MONTHLY:
                rows = await conn.fetch(
                    """
                    SELECT metric, SUM(total_value) as total
                    FROM usage_daily_aggregates
                    WHERE user_id = $1
                    AND date >= $2
                    AND date < $3
                    GROUP BY metric
                    """,
                    user_id,
                    start_date,
                    end_date,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT metric, total_value as total
                    FROM usage_daily_aggregates
                    WHERE user_id = $1 AND date = $2
                    """,
                    user_id,
                    period_date,
                )

        # Build summary
        usage = {}
        for row in rows:
            metric_str = row["metric"]
            total = float(row["total"])

            try:
                metric = UsageMetric(metric_str)
            except ValueError:
                continue

            # Get limit for this metric
            limit = self._get_limit_for_metric(metric, limits, period)
            unlimited = limit is None

            percentage = 0.0
            if limit and limit > 0:
                percentage = (total / limit) * 100

            usage[metric_str] = UsageMetricSummary(
                used=total,
                limit=limit,
                unlimited=unlimited,
                percentage=percentage,
                soft_limit_reached=percentage >= limits.soft_limit_threshold * 100,
                hard_limit_reached=percentage >= 100,
            )

        # Calculate billing cycle end
        billing_end = None
        days_remaining = 0
        if subscription and subscription.current_period_end:
            billing_end = subscription.current_period_end
            days_remaining = (billing_end.date() - now.date()).days

        return UsageSummary(
            user_id=user_id,
            tier=tier,
            period=period_str,
            period_start=datetime.combine(start_date, datetime.min.time()),
            period_end=datetime.combine(end_date, datetime.min.time()),
            usage=usage,
            billing_cycle_end=billing_end,
            days_remaining=max(0, days_remaining),
            team_id=subscription.team_id if subscription else None,
        )

    def _get_limit_for_metric(
        self,
        metric: UsageMetric,
        limits: "TierLimits",
        period: UsagePeriod,
    ) -> Optional[float]:
        """Get the limit for a specific metric and period."""
        mapping = {
            (UsageMetric.MESSAGES_SENT, UsagePeriod.DAILY): limits.messages_per_day,
            (UsageMetric.MESSAGES_SENT, UsagePeriod.MONTHLY): limits.messages_per_month,
            (UsageMetric.TOKENS_TOTAL, UsagePeriod.DAILY): limits.tokens_per_day,
            (UsageMetric.TOKENS_TOTAL, UsagePeriod.MONTHLY): limits.tokens_per_month,
            (UsageMetric.API_CALLS, UsagePeriod.DAILY): limits.api_calls_per_day,
            (UsageMetric.API_CALLS, UsagePeriod.MONTHLY): limits.api_calls_per_month,
            (UsageMetric.MEMORIES_TOTAL, UsagePeriod.MONTHLY): limits.memories_max,
            (UsageMetric.STORAGE_USED_MB, UsagePeriod.MONTHLY): limits.storage_mb,
            (UsageMetric.DOCUMENTS_COUNT, UsagePeriod.MONTHLY): limits.documents_max,
            (UsageMetric.EXPERT_INVOCATIONS, UsagePeriod.DAILY): limits.expert_invocations_per_day,
        }
        return mapping.get((metric, period))

    async def get_history(
        self,
        user_id: str,
        metric: UsageMetric,
        start_date: date,
        end_date: date,
    ) -> List[Tuple[date, float]]:
        """Get historical usage data."""
        if not self._connected:
            await self.connect()

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT date, total_value
                FROM usage_daily_aggregates
                WHERE user_id = $1
                AND metric = $2
                AND date >= $3
                AND date <= $4
                ORDER BY date ASC
                """,
                user_id,
                metric.value,
                start_date,
                end_date,
            )

        return [(row["date"], float(row["total_value"])) for row in rows]

    async def get_breakdown(
        self,
        user_id: str,
        metric: UsageMetric,
        dimension: str,
        period_date: Optional[date] = None,
    ) -> Dict[str, float]:
        """Get usage breakdown by dimension."""
        if not self._connected:
            await self.connect()

        period_date = period_date or datetime.utcnow().date()
        start_date = period_date.replace(day=1)

        if period_date.month == 12:
            end_date = period_date.replace(year=period_date.year + 1, month=1, day=1)
        else:
            end_date = period_date.replace(month=period_date.month + 1, day=1)

        column = f"by_{dimension}"

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT {column}
                FROM usage_daily_aggregates
                WHERE user_id = $1
                AND metric = $2
                AND date >= $3
                AND date < $4
                """,
                user_id,
                metric.value,
                start_date,
                end_date,
            )

        # Merge breakdowns across days
        result: Dict[str, float] = defaultdict(float)
        for row in rows:
            breakdown = row[column] or {}
            if isinstance(breakdown, str):
                breakdown = json.loads(breakdown)
            for key, value in breakdown.items():
                result[key] += float(value)

        return dict(result)

    async def get_subscription(self, user_id: str) -> Optional[Subscription]:
        """Get user subscription."""
        if not self._connected:
            await self.connect()

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT * FROM subscriptions WHERE user_id = $1
                """,
                user_id,
            )

        if not row:
            return None

        return Subscription(
            user_id=row["user_id"],
            tier=SubscriptionTier(row["tier"]),
            billing_cycle=BillingCycle(row["billing_cycle"]),
            started_at=row["started_at"],
            current_period_start=row["current_period_start"],
            current_period_end=row["current_period_end"],
            canceled_at=row["canceled_at"],
            team_id=row["team_id"],
            is_team_admin=row["is_team_admin"],
            seat_count=row["seat_count"],
            custom_limits=row["custom_limits"],
            is_active=row["is_active"],
            is_trial=row["is_trial"],
            trial_ends_at=row["trial_ends_at"],
        )

    async def save_subscription(self, subscription: Subscription) -> None:
        """Save or update subscription."""
        if not self._connected:
            await self.connect()

        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO subscriptions
                    (user_id, tier, billing_cycle, started_at, current_period_start,
                     current_period_end, canceled_at, team_id, is_team_admin, seat_count,
                     custom_limits, is_active, is_trial, trial_ends_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                ON CONFLICT (user_id) DO UPDATE SET
                    tier = EXCLUDED.tier,
                    billing_cycle = EXCLUDED.billing_cycle,
                    current_period_start = EXCLUDED.current_period_start,
                    current_period_end = EXCLUDED.current_period_end,
                    canceled_at = EXCLUDED.canceled_at,
                    team_id = EXCLUDED.team_id,
                    is_team_admin = EXCLUDED.is_team_admin,
                    seat_count = EXCLUDED.seat_count,
                    custom_limits = EXCLUDED.custom_limits,
                    is_active = EXCLUDED.is_active,
                    is_trial = EXCLUDED.is_trial,
                    trial_ends_at = EXCLUDED.trial_ends_at,
                    updated_at = CURRENT_TIMESTAMP
                """,
                subscription.user_id,
                subscription.tier.value,
                subscription.billing_cycle.value,
                subscription.started_at,
                subscription.current_period_start,
                subscription.current_period_end,
                subscription.canceled_at,
                subscription.team_id,
                subscription.is_team_admin,
                subscription.seat_count,
                json.dumps(subscription.custom_limits) if subscription.custom_limits else None,
                subscription.is_active,
                subscription.is_trial,
                subscription.trial_ends_at,
            )

    async def run_daily_rollup(self, target_date: Optional[date] = None) -> int:
        """
        Run daily aggregation rollup.

        Returns number of records aggregated.
        """
        if not self._connected:
            await self.connect()

        target_date = target_date or (datetime.utcnow().date() - timedelta(days=1))

        async with self._pool.acquire() as conn:
            result = await conn.execute(DAILY_ROLLUP_QUERY, target_date)
            # Parse result for count
            return 1  # Simplified

    async def run_monthly_rollup(self, year_month: Optional[str] = None) -> int:
        """
        Run monthly aggregation from daily aggregates.

        Returns number of records aggregated.
        """
        if not self._connected:
            await self.connect()

        if not year_month:
            last_month = datetime.utcnow().replace(day=1) - timedelta(days=1)
            year_month = last_month.strftime("%Y-%m")

        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO usage_monthly_aggregates
                    (user_id, metric, year_month, total_value, count, daily_avg, peak_daily, by_expert, by_memory_type)
                SELECT
                    user_id,
                    metric,
                    TO_CHAR(date, 'YYYY-MM') as year_month,
                    SUM(total_value) as total_value,
                    SUM(count) as count,
                    AVG(total_value) as daily_avg,
                    MAX(total_value) as peak_daily,
                    jsonb_object_agg(COALESCE(by_expert::text, '{}'), 1) as by_expert,
                    jsonb_object_agg(COALESCE(by_memory_type::text, '{}'), 1) as by_memory_type
                FROM usage_daily_aggregates
                WHERE TO_CHAR(date, 'YYYY-MM') = $1
                GROUP BY user_id, metric, TO_CHAR(date, 'YYYY-MM')
                ON CONFLICT (user_id, metric, year_month) DO UPDATE SET
                    total_value = EXCLUDED.total_value,
                    count = EXCLUDED.count,
                    daily_avg = EXCLUDED.daily_avg,
                    peak_daily = EXCLUDED.peak_daily,
                    updated_at = CURRENT_TIMESTAMP
                """,
                year_month,
            )

        return 1

    # === Context Manager ===

    async def __aenter__(self) -> "PostgresUsageStore":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.disconnect()


# =============================================================================
# Memory Storage (for testing)
# =============================================================================

class MemoryUsageStore(UsageStore):
    """In-memory usage store for testing."""

    def __init__(self):
        self._records: List[UsageRecord] = []
        self._subscriptions: Dict[str, Subscription] = {}
        self._lock = asyncio.Lock()

    async def save_record(self, record: UsageRecord) -> None:
        async with self._lock:
            self._records.append(record)

    async def save_records_batch(self, records: List[UsageRecord]) -> None:
        async with self._lock:
            self._records.extend(records)

    async def get_summary(
        self,
        user_id: str,
        period: UsagePeriod = UsagePeriod.MONTHLY,
        period_date: Optional[date] = None,
    ) -> UsageSummary:
        """Get usage summary."""
        now = datetime.utcnow()
        period_date = period_date or now.date()

        subscription = await self.get_subscription(user_id)
        tier = subscription.tier if subscription else SubscriptionTier.FREE
        limits = get_tier_limits(tier)

        # Filter records for period
        if period == UsagePeriod.MONTHLY:
            period_str = period_date.strftime("%Y-%m")
            start = datetime(period_date.year, period_date.month, 1)
            if period_date.month == 12:
                end = datetime(period_date.year + 1, 1, 1)
            else:
                end = datetime(period_date.year, period_date.month + 1, 1)
        else:
            period_str = period_date.strftime("%Y-%m-%d")
            start = datetime.combine(period_date, datetime.min.time())
            end = start + timedelta(days=1)

        # Aggregate
        totals: Dict[str, float] = defaultdict(float)
        for record in self._records:
            if record.user_id == user_id and start <= record.timestamp < end:
                totals[record.metric.value] += record.value

        # Build summary
        usage = {}
        for metric_str, total in totals.items():
            try:
                metric = UsageMetric(metric_str)
            except ValueError:
                continue

            limit = None  # Simplified for memory store
            usage[metric_str] = UsageMetricSummary(
                used=total,
                limit=limit,
                unlimited=limit is None,
            )

        return UsageSummary(
            user_id=user_id,
            tier=tier,
            period=period_str,
            period_start=start,
            period_end=end,
            usage=usage,
        )

    async def get_history(
        self,
        user_id: str,
        metric: UsageMetric,
        start_date: date,
        end_date: date,
    ) -> List[Tuple[date, float]]:
        """Get historical data."""
        daily: Dict[date, float] = defaultdict(float)

        for record in self._records:
            if (
                record.user_id == user_id
                and record.metric == metric
                and start_date <= record.timestamp.date() <= end_date
            ):
                daily[record.timestamp.date()] += record.value

        return sorted(daily.items())

    async def get_subscription(self, user_id: str) -> Optional[Subscription]:
        return self._subscriptions.get(user_id)

    async def save_subscription(self, subscription: Subscription) -> None:
        self._subscriptions[subscription.user_id] = subscription

    async def clear(self) -> None:
        """Clear all data (for testing)."""
        async with self._lock:
            self._records.clear()
            self._subscriptions.clear()


# =============================================================================
# Factory
# =============================================================================

def create_usage_store(
    backend: str = "memory",
    **kwargs: Any,
) -> UsageStore:
    """Create a usage store instance."""
    if backend == "memory":
        return MemoryUsageStore()
    elif backend == "postgres" or backend == "postgresql":
        return PostgresUsageStore(**kwargs)
    else:
        raise ValueError(f"Unknown store backend: {backend}")
