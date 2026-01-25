"""
SLO/SLI Management System

SOTA SLO management with:
- Service Level Indicators (SLIs) - measurable metrics
- Service Level Objectives (SLOs) - targets for SLIs
- Error Budgets - allowable failure margin
- Burn Rate Alerts - rate of error budget consumption
- Multi-window analysis
- Compliance reporting
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Core Types
# =============================================================================

class SLIType(Enum):
    """Types of Service Level Indicators."""
    AVAILABILITY = "availability"  # Uptime percentage
    LATENCY = "latency"  # Response time
    THROUGHPUT = "throughput"  # Requests per second
    ERROR_RATE = "error_rate"  # Percentage of failed requests
    SATURATION = "saturation"  # Resource utilization
    CORRECTNESS = "correctness"  # Data accuracy
    FRESHNESS = "freshness"  # Data staleness
    DURABILITY = "durability"  # Data retention


class SLOWindow(Enum):
    """Time windows for SLO calculation."""
    ROLLING_1H = "1h"
    ROLLING_6H = "6h"
    ROLLING_24H = "24h"
    ROLLING_7D = "7d"
    ROLLING_28D = "28d"
    ROLLING_30D = "30d"
    CALENDAR_MONTH = "calendar_month"
    CALENDAR_QUARTER = "calendar_quarter"


class SLOStatus(Enum):
    """Status of an SLO."""
    MET = "met"
    AT_RISK = "at_risk"  # Budget consumption > 50%
    BREACHED = "breached"  # Budget exhausted
    UNKNOWN = "unknown"


@dataclass
class SLIValue:
    """A single SLI measurement."""
    timestamp: datetime
    value: float
    good_count: int = 0
    total_count: int = 0
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class SLI:
    """
    Service Level Indicator definition.

    An SLI is a quantitative measure of service behavior.
    """
    name: str
    sli_type: SLIType
    description: str = ""

    # Metric configuration
    good_metric: str = ""  # Metric counting good events
    total_metric: str = ""  # Metric counting total events
    ratio_metric: str = ""  # Alternative: direct ratio metric

    # For latency SLIs
    latency_metric: str = ""
    latency_threshold_ms: float = 0.0

    # Bucketed latency (histogram)
    latency_bucket_metric: str = ""
    latency_bucket_le: str = ""  # Label for bucket boundary

    # Labels to group by
    group_by_labels: List[str] = field(default_factory=list)

    def calculate(
        self,
        good_count: int,
        total_count: int,
    ) -> float:
        """Calculate SLI value as a ratio."""
        if total_count == 0:
            return 1.0  # No data = assume good
        return good_count / total_count


@dataclass
class ErrorBudget:
    """
    Error budget for an SLO.

    The error budget is the inverse of the SLO target.
    For example, 99.9% availability allows 0.1% downtime.
    """
    total_budget: float  # In same units as SLI (e.g., percentage)
    consumed: float = 0.0
    remaining: float = 0.0
    consumption_rate: float = 0.0  # Budget consumed per hour
    estimated_exhaustion: Optional[datetime] = None

    @property
    def remaining_percentage(self) -> float:
        """Percentage of budget remaining."""
        if self.total_budget == 0:
            return 0.0
        return (self.remaining / self.total_budget) * 100

    @property
    def consumed_percentage(self) -> float:
        """Percentage of budget consumed."""
        return 100 - self.remaining_percentage

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_budget": self.total_budget,
            "consumed": self.consumed,
            "remaining": self.remaining,
            "remaining_percentage": self.remaining_percentage,
            "consumed_percentage": self.consumed_percentage,
            "consumption_rate_per_hour": self.consumption_rate,
            "estimated_exhaustion": (
                self.estimated_exhaustion.isoformat()
                if self.estimated_exhaustion else None
            ),
        }


@dataclass
class BurnRate:
    """
    Burn rate measurement for error budget.

    Burn rate = (current error rate) / (allowed error rate)
    A burn rate of 1 means consuming budget exactly at pace.
    A burn rate of 2 means consuming 2x faster than allowed.
    """
    short_window: float = 0.0  # e.g., 5 minute window
    long_window: float = 0.0  # e.g., 1 hour window
    short_window_duration: str = "5m"
    long_window_duration: str = "1h"

    # Alert thresholds (based on Google SRE recommendations)
    critical_threshold: float = 14.4  # 2% budget in 1 hour
    warning_threshold: float = 6.0  # 5% budget in 6 hours

    @property
    def is_critical(self) -> bool:
        return self.short_window >= self.critical_threshold

    @property
    def is_warning(self) -> bool:
        return self.short_window >= self.warning_threshold

    def to_dict(self) -> Dict[str, Any]:
        return {
            "short_window": self.short_window,
            "long_window": self.long_window,
            "short_window_duration": self.short_window_duration,
            "long_window_duration": self.long_window_duration,
            "is_critical": self.is_critical,
            "is_warning": self.is_warning,
        }


@dataclass
class SLOReport:
    """Report on SLO compliance."""
    slo_name: str
    window: SLOWindow
    start_time: datetime
    end_time: datetime
    target: float
    actual: float
    status: SLOStatus
    error_budget: ErrorBudget
    burn_rate: BurnRate
    good_count: int = 0
    total_count: int = 0
    downtime_minutes: float = 0.0
    incidents: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "slo_name": self.slo_name,
            "window": self.window.value,
            "period": {
                "start": self.start_time.isoformat(),
                "end": self.end_time.isoformat(),
            },
            "target": self.target,
            "actual": self.actual,
            "status": self.status.value,
            "good_count": self.good_count,
            "total_count": self.total_count,
            "downtime_minutes": self.downtime_minutes,
            "error_budget": self.error_budget.to_dict(),
            "burn_rate": self.burn_rate.to_dict(),
            "incidents": self.incidents,
        }


@dataclass
class SLO:
    """
    Service Level Objective definition.

    An SLO is a target value for an SLI over a time window.
    """
    name: str
    sli: SLI
    target: float  # e.g., 0.999 for 99.9%
    window: SLOWindow = SLOWindow.ROLLING_30D
    description: str = ""

    # Alerting configuration
    enable_burn_rate_alerts: bool = True
    page_critical: bool = True
    ticket_warning: bool = True

    # Ownership
    service: str = ""
    team: str = ""
    owner: str = ""

    # Metadata
    labels: Dict[str, str] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def error_budget_fraction(self) -> float:
        """The fraction of requests that can fail."""
        return 1.0 - self.target

    @property
    def window_duration(self) -> timedelta:
        """Get the window as a timedelta."""
        durations = {
            SLOWindow.ROLLING_1H: timedelta(hours=1),
            SLOWindow.ROLLING_6H: timedelta(hours=6),
            SLOWindow.ROLLING_24H: timedelta(hours=24),
            SLOWindow.ROLLING_7D: timedelta(days=7),
            SLOWindow.ROLLING_28D: timedelta(days=28),
            SLOWindow.ROLLING_30D: timedelta(days=30),
            SLOWindow.CALENDAR_MONTH: timedelta(days=30),
            SLOWindow.CALENDAR_QUARTER: timedelta(days=90),
        }
        return durations.get(self.window, timedelta(days=30))

    def calculate_error_budget(
        self,
        good_count: int,
        total_count: int,
    ) -> ErrorBudget:
        """Calculate the error budget for given counts."""
        if total_count == 0:
            return ErrorBudget(
                total_budget=0,
                consumed=0,
                remaining=0,
            )

        # Total allowed errors
        allowed_errors = total_count * self.error_budget_fraction
        actual_errors = total_count - good_count
        remaining_errors = max(0, allowed_errors - actual_errors)

        return ErrorBudget(
            total_budget=allowed_errors,
            consumed=actual_errors,
            remaining=remaining_errors,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "sli": self.sli.name,
            "target": self.target,
            "target_percentage": f"{self.target * 100:.3f}%",
            "window": self.window.value,
            "error_budget_fraction": self.error_budget_fraction,
            "service": self.service,
            "team": self.team,
            "owner": self.owner,
            "labels": self.labels,
        }


# =============================================================================
# Multi-Window Burn Rate Calculator
# =============================================================================

class MultiWindowBurnRateCalculator:
    """
    Calculates burn rates across multiple time windows.

    Based on Google SRE "Multi-Window, Multi-Burn-Rate Alerts" approach.
    """

    # Standard windows and thresholds (from Google SRE book)
    WINDOWS = [
        # (short_window, long_window, burn_rate, budget_consumed)
        ("5m", "1h", 14.4, 0.02),  # 2% budget in 1 hour - PAGE
        ("30m", "6h", 6.0, 0.05),  # 5% budget in 6 hours - PAGE
        ("2h", "1d", 3.0, 0.10),  # 10% budget in 1 day - TICKET
        ("6h", "3d", 1.0, 0.10),  # 10% budget in 3 days - TICKET
    ]

    def __init__(self, slo: SLO):
        self.slo = slo
        self._measurements: deque = deque(maxlen=10000)

    def add_measurement(self, good_count: int, total_count: int) -> None:
        """Add a measurement."""
        self._measurements.append({
            "timestamp": time.time(),
            "good": good_count,
            "total": total_count,
        })

    def calculate_burn_rate(
        self,
        good_count: int,
        total_count: int,
        window_seconds: float,
    ) -> float:
        """Calculate burn rate for a window."""
        if total_count == 0:
            return 0.0

        # Current error rate
        error_rate = (total_count - good_count) / total_count

        # Allowed error rate over window
        allowed_error_rate = self.slo.error_budget_fraction

        if allowed_error_rate == 0:
            return float('inf') if error_rate > 0 else 0.0

        # Burn rate = actual / allowed
        return error_rate / allowed_error_rate

    def get_current_burn_rates(self) -> List[Dict[str, Any]]:
        """Get burn rates for all configured windows."""
        results = []
        now = time.time()

        for short_window, long_window, threshold, budget in self.WINDOWS:
            # Parse window durations
            short_secs = self._parse_duration(short_window)
            long_secs = self._parse_duration(long_window)

            # Get measurements for windows
            short_good, short_total = self._sum_window(now - short_secs, now)
            long_good, long_total = self._sum_window(now - long_secs, now)

            short_burn = self.calculate_burn_rate(short_good, short_total, short_secs)
            long_burn = self.calculate_burn_rate(long_good, long_total, long_secs)

            results.append({
                "short_window": short_window,
                "long_window": long_window,
                "short_burn_rate": short_burn,
                "long_burn_rate": long_burn,
                "threshold": threshold,
                "budget_consumed": budget,
                "is_alerting": short_burn >= threshold and long_burn >= threshold,
            })

        return results

    def _sum_window(
        self,
        start: float,
        end: float,
    ) -> Tuple[int, int]:
        """Sum good and total counts in a time window."""
        good = 0
        total = 0

        for m in self._measurements:
            if start <= m["timestamp"] <= end:
                good += m["good"]
                total += m["total"]

        return good, total

    def _parse_duration(self, duration: str) -> float:
        """Parse duration string to seconds."""
        units = {
            "s": 1,
            "m": 60,
            "h": 3600,
            "d": 86400,
        }
        value = float(duration[:-1])
        unit = duration[-1]
        return value * units.get(unit, 1)


# =============================================================================
# SLO Manager
# =============================================================================

class SLOManager:
    """
    Central manager for SLOs.

    Handles:
    - SLO registration and configuration
    - SLI measurement collection
    - Error budget calculation
    - Burn rate alerting
    - Compliance reporting
    """

    def __init__(
        self,
        metrics_engine: Optional[Any] = None,
        alert_engine: Optional[Any] = None,
    ):
        self.metrics_engine = metrics_engine
        self.alert_engine = alert_engine

        self._slos: Dict[str, SLO] = {}
        self._sli_history: Dict[str, deque] = {}
        self._burn_rate_calculators: Dict[str, MultiWindowBurnRateCalculator] = {}

        self._running = False
        self._eval_task: Optional[asyncio.Task] = None
        self._eval_interval = 60.0  # seconds

    async def initialize(self) -> None:
        """Initialize the SLO manager."""
        self._running = True
        self._eval_task = asyncio.create_task(self._evaluation_loop())
        logger.info("SLO Manager initialized")

    async def shutdown(self) -> None:
        """Shutdown the SLO manager."""
        self._running = False
        if self._eval_task:
            self._eval_task.cancel()
            try:
                await self._eval_task
            except asyncio.CancelledError:
                pass
        logger.info("SLO Manager shutdown")

    def register_slo(self, slo: SLO) -> None:
        """Register an SLO."""
        self._slos[slo.name] = slo
        self._sli_history[slo.name] = deque(maxlen=10000)
        self._burn_rate_calculators[slo.name] = MultiWindowBurnRateCalculator(slo)

        logger.info(f"Registered SLO: {slo.name} (target: {slo.target * 100:.2f}%)")

    def unregister_slo(self, name: str) -> bool:
        """Unregister an SLO."""
        if name in self._slos:
            del self._slos[name]
            del self._sli_history[name]
            del self._burn_rate_calculators[name]
            return True
        return False

    def get_slo(self, name: str) -> Optional[SLO]:
        """Get an SLO by name."""
        return self._slos.get(name)

    def list_slos(self) -> List[SLO]:
        """List all SLOs."""
        return list(self._slos.values())

    def record_event(
        self,
        slo_name: str,
        is_good: bool,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Record a single event (good or bad)."""
        if slo_name not in self._slos:
            return

        measurement = SLIValue(
            timestamp=datetime.utcnow(),
            value=1.0 if is_good else 0.0,
            good_count=1 if is_good else 0,
            total_count=1,
            labels=labels or {},
        )

        self._sli_history[slo_name].append(measurement)

        # Update burn rate calculator
        calculator = self._burn_rate_calculators[slo_name]
        calculator.add_measurement(
            good_count=1 if is_good else 0,
            total_count=1,
        )

    def record_counts(
        self,
        slo_name: str,
        good_count: int,
        total_count: int,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Record aggregated counts."""
        if slo_name not in self._slos:
            return

        value = good_count / total_count if total_count > 0 else 1.0

        measurement = SLIValue(
            timestamp=datetime.utcnow(),
            value=value,
            good_count=good_count,
            total_count=total_count,
            labels=labels or {},
        )

        self._sli_history[slo_name].append(measurement)

        # Update burn rate calculator
        calculator = self._burn_rate_calculators[slo_name]
        calculator.add_measurement(good_count, total_count)

    def get_current_status(self, slo_name: str) -> Optional[SLOReport]:
        """Get current status for an SLO."""
        slo = self._slos.get(slo_name)
        if not slo:
            return None

        history = self._sli_history.get(slo_name, deque())
        calculator = self._burn_rate_calculators[slo_name]

        # Calculate window bounds
        now = datetime.utcnow()
        window_start = now - slo.window_duration

        # Aggregate measurements in window
        good_count = 0
        total_count = 0

        for measurement in history:
            if measurement.timestamp >= window_start:
                good_count += measurement.good_count
                total_count += measurement.total_count

        # Calculate SLI
        if total_count > 0:
            actual = good_count / total_count
        else:
            actual = 1.0

        # Calculate error budget
        error_budget = slo.calculate_error_budget(good_count, total_count)

        # Calculate burn rate
        burn_rates = calculator.get_current_burn_rates()
        primary_burn = burn_rates[0] if burn_rates else {}

        burn_rate = BurnRate(
            short_window=primary_burn.get("short_burn_rate", 0),
            long_window=primary_burn.get("long_burn_rate", 0),
        )

        # Estimate budget exhaustion
        if burn_rate.short_window > 1 and error_budget.remaining > 0:
            hours_to_exhaust = error_budget.remaining / (
                burn_rate.short_window * slo.error_budget_fraction * (total_count or 1)
            )
            error_budget.estimated_exhaustion = now + timedelta(hours=hours_to_exhaust)
            error_budget.consumption_rate = burn_rate.short_window * slo.error_budget_fraction

        # Determine status
        if actual >= slo.target:
            status = SLOStatus.MET
        elif error_budget.remaining_percentage < 50:
            status = SLOStatus.AT_RISK
        elif error_budget.remaining <= 0:
            status = SLOStatus.BREACHED
        else:
            status = SLOStatus.AT_RISK

        return SLOReport(
            slo_name=slo_name,
            window=slo.window,
            start_time=window_start,
            end_time=now,
            target=slo.target,
            actual=actual,
            status=status,
            good_count=good_count,
            total_count=total_count,
            error_budget=error_budget,
            burn_rate=burn_rate,
        )

    def get_all_statuses(self) -> List[SLOReport]:
        """Get status for all SLOs."""
        return [
            report for name in self._slos
            if (report := self.get_current_status(name)) is not None
        ]

    async def _evaluation_loop(self) -> None:
        """Periodically evaluate SLOs and trigger alerts."""
        while self._running:
            try:
                await self._evaluate_all()
            except Exception as e:
                logger.error(f"Error in SLO evaluation: {e}")

            await asyncio.sleep(self._eval_interval)

    async def _evaluate_all(self) -> None:
        """Evaluate all SLOs for alerting."""
        for name, slo in self._slos.items():
            if not slo.enable_burn_rate_alerts:
                continue

            calculator = self._burn_rate_calculators[name]
            burn_rates = calculator.get_current_burn_rates()

            for br in burn_rates:
                if br["is_alerting"]:
                    # Trigger alert
                    if self.alert_engine and br["budget_consumed"] <= 0.05:
                        # High burn rate - page
                        logger.warning(
                            f"SLO {name} high burn rate: {br['short_burn_rate']:.1f}x "
                            f"(threshold: {br['threshold']}x)"
                        )
                        # TODO: Trigger actual alert

    # =========================================================================
    # Compliance Reporting
    # =========================================================================

    def generate_report(
        self,
        slo_name: str,
        window: Optional[SLOWindow] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Optional[SLOReport]:
        """Generate a compliance report for an SLO."""
        slo = self._slos.get(slo_name)
        if not slo:
            return None

        # Use SLO window if not specified
        if window:
            duration = {
                SLOWindow.ROLLING_1H: timedelta(hours=1),
                SLOWindow.ROLLING_6H: timedelta(hours=6),
                SLOWindow.ROLLING_24H: timedelta(hours=24),
                SLOWindow.ROLLING_7D: timedelta(days=7),
                SLOWindow.ROLLING_28D: timedelta(days=28),
                SLOWindow.ROLLING_30D: timedelta(days=30),
            }.get(window, slo.window_duration)
        else:
            window = slo.window
            duration = slo.window_duration

        end = end_time or datetime.utcnow()
        start = start_time or (end - duration)

        # Get measurements in range
        history = self._sli_history.get(slo_name, deque())

        good_count = 0
        total_count = 0
        incidents = []

        prev_good = True
        incident_start = None

        for m in history:
            if start <= m.timestamp <= end:
                good_count += m.good_count
                total_count += m.total_count

                # Track incidents (periods of bad service)
                is_good = m.good_count == m.total_count
                if not is_good and prev_good:
                    incident_start = m.timestamp
                elif is_good and not prev_good and incident_start:
                    incidents.append({
                        "start": incident_start.isoformat(),
                        "end": m.timestamp.isoformat(),
                        "duration_minutes": (m.timestamp - incident_start).total_seconds() / 60,
                    })
                    incident_start = None
                prev_good = is_good

        actual = good_count / total_count if total_count > 0 else 1.0
        error_budget = slo.calculate_error_budget(good_count, total_count)

        # Calculate downtime
        downtime_minutes = sum(i.get("duration_minutes", 0) for i in incidents)

        status = SLOStatus.MET if actual >= slo.target else SLOStatus.BREACHED

        return SLOReport(
            slo_name=slo_name,
            window=window,
            start_time=start,
            end_time=end,
            target=slo.target,
            actual=actual,
            status=status,
            good_count=good_count,
            total_count=total_count,
            error_budget=error_budget,
            burn_rate=BurnRate(),
            downtime_minutes=downtime_minutes,
            incidents=incidents,
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get SLO manager statistics."""
        statuses = self.get_all_statuses()

        return {
            "total_slos": len(self._slos),
            "slos_met": sum(1 for s in statuses if s.status == SLOStatus.MET),
            "slos_at_risk": sum(1 for s in statuses if s.status == SLOStatus.AT_RISK),
            "slos_breached": sum(1 for s in statuses if s.status == SLOStatus.BREACHED),
            "slos": [s.to_dict() for s in statuses],
        }


# =============================================================================
# Predefined SLOs
# =============================================================================

def create_availability_slo(
    name: str,
    target: float = 0.999,
    service: str = "",
    **kwargs,
) -> SLO:
    """Create a standard availability SLO."""
    sli = SLI(
        name=f"{name}_availability",
        sli_type=SLIType.AVAILABILITY,
        good_metric=f"{service}_requests_success_total" if service else "requests_success_total",
        total_metric=f"{service}_requests_total" if service else "requests_total",
    )

    return SLO(
        name=name,
        sli=sli,
        target=target,
        service=service,
        **kwargs,
    )


def create_latency_slo(
    name: str,
    target: float = 0.99,
    latency_threshold_ms: float = 200.0,
    service: str = "",
    **kwargs,
) -> SLO:
    """Create a standard latency SLO."""
    sli = SLI(
        name=f"{name}_latency",
        sli_type=SLIType.LATENCY,
        latency_metric=f"{service}_request_duration_seconds" if service else "request_duration_seconds",
        latency_threshold_ms=latency_threshold_ms,
    )

    return SLO(
        name=name,
        sli=sli,
        target=target,
        service=service,
        **kwargs,
    )


def create_error_rate_slo(
    name: str,
    target: float = 0.999,  # 0.1% error rate max
    service: str = "",
    **kwargs,
) -> SLO:
    """Create a standard error rate SLO."""
    sli = SLI(
        name=f"{name}_error_rate",
        sli_type=SLIType.ERROR_RATE,
        good_metric=f"{service}_requests_success_total" if service else "requests_success_total",
        total_metric=f"{service}_requests_total" if service else "requests_total",
    )

    return SLO(
        name=name,
        sli=sli,
        target=target,
        service=service,
        **kwargs,
    )
