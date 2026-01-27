"""AION Simulation Metrics - Statistical metrics collection and analysis.

Provides:
- MetricsCollector: Collects, aggregates, and analyses simulation metrics
  with support for percentiles, confidence intervals, and time-series tracking.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class MetricSample:
    """A single metric sample."""

    value: float
    tick: int = 0
    timestamp: float = 0.0
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class MetricSummary:
    """Statistical summary of a metric."""

    name: str
    count: int = 0
    total: float = 0.0
    mean: float = 0.0
    variance: float = 0.0
    std_dev: float = 0.0
    min_val: float = float("inf")
    max_val: float = float("-inf")
    p50: float = 0.0
    p90: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    ci_lower: float = 0.0  # 95% confidence interval
    ci_upper: float = 0.0


class MetricsCollector:
    """Collects and analyses simulation metrics.

    SOTA features:
    - Time-series tracking per metric.
    - Online mean/variance (Welford's algorithm) for streaming computation.
    - Percentile calculation (p50, p90, p95, p99).
    - Confidence interval estimation.
    - Metric tagging for slicing.
    - Rate computation (per tick, per second).
    - Multi-run aggregation for A/B comparisons.
    """

    def __init__(self) -> None:
        self._samples: Dict[str, List[MetricSample]] = defaultdict(list)
        # Welford's online stats
        self._counts: Dict[str, int] = defaultdict(int)
        self._means: Dict[str, float] = defaultdict(float)
        self._m2s: Dict[str, float] = defaultdict(float)
        self._mins: Dict[str, float] = {}
        self._maxs: Dict[str, float] = {}

    # -- Recording --

    def record(
        self,
        name: str,
        value: float,
        tick: int = 0,
        timestamp: float = 0.0,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """Record a metric sample."""
        sample = MetricSample(
            value=value, tick=tick, timestamp=timestamp, tags=tags or {},
        )
        self._samples[name].append(sample)

        # Welford's online update
        n = self._counts[name] + 1
        self._counts[name] = n
        delta = value - self._means[name]
        self._means[name] += delta / n
        delta2 = value - self._means[name]
        self._m2s[name] += delta * delta2

        if name not in self._mins or value < self._mins[name]:
            self._mins[name] = value
        if name not in self._maxs or value > self._maxs[name]:
            self._maxs[name] = value

    def increment(self, name: str, delta: float = 1.0, tick: int = 0) -> None:
        """Increment a counter metric."""
        samples = self._samples[name]
        current = samples[-1].value if samples else 0.0
        self.record(name, current + delta, tick=tick)

    # -- Queries --

    def get_values(self, name: str) -> List[float]:
        return [s.value for s in self._samples.get(name, [])]

    def get_latest(self, name: str) -> Optional[float]:
        samples = self._samples.get(name)
        return samples[-1].value if samples else None

    def get_time_series(self, name: str) -> List[tuple]:
        """Get (tick, value) pairs."""
        return [(s.tick, s.value) for s in self._samples.get(name, [])]

    def summarize(self, name: str, confidence: float = 0.95) -> MetricSummary:
        """Compute full statistical summary for a metric."""
        values = self.get_values(name)
        if not values:
            return MetricSummary(name=name)

        n = len(values)
        mean = self._means.get(name, 0.0)
        m2 = self._m2s.get(name, 0.0)
        variance = m2 / n if n > 0 else 0.0
        std_dev = math.sqrt(variance)

        sorted_vals = sorted(values)

        # Confidence interval (using t-distribution approximation for small n)
        z = 1.96  # For 95% CI (normal approximation)
        if confidence == 0.99:
            z = 2.576
        elif confidence == 0.90:
            z = 1.645
        se = std_dev / math.sqrt(n) if n > 0 else 0.0

        return MetricSummary(
            name=name,
            count=n,
            total=sum(values),
            mean=mean,
            variance=variance,
            std_dev=std_dev,
            min_val=self._mins.get(name, 0.0),
            max_val=self._maxs.get(name, 0.0),
            p50=self._percentile(sorted_vals, 0.50),
            p90=self._percentile(sorted_vals, 0.90),
            p95=self._percentile(sorted_vals, 0.95),
            p99=self._percentile(sorted_vals, 0.99),
            ci_lower=mean - z * se,
            ci_upper=mean + z * se,
        )

    def summarize_all(self) -> Dict[str, MetricSummary]:
        return {name: self.summarize(name) for name in self._samples}

    def rate(self, name: str, window_ticks: int = 0) -> float:
        """Compute rate of change over last N ticks (or all)."""
        samples = self._samples.get(name, [])
        if len(samples) < 2:
            return 0.0

        if window_ticks > 0:
            samples = [s for s in samples if s.tick > samples[-1].tick - window_ticks]
            if len(samples) < 2:
                return 0.0

        tick_range = samples[-1].tick - samples[0].tick
        if tick_range == 0:
            return 0.0

        value_range = samples[-1].value - samples[0].value
        return value_range / tick_range

    # -- Multi-run Comparison --

    def compare(
        self,
        name: str,
        other: "MetricsCollector",
        significance: float = 0.05,
    ) -> Dict[str, Any]:
        """Compare a metric between two collectors (e.g., A/B test).

        Uses Welch's t-test approximation.
        """
        s1 = self.summarize(name)
        s2 = other.summarize(name)

        if s1.count < 2 or s2.count < 2:
            return {
                "metric": name,
                "comparable": False,
                "reason": "Insufficient samples",
            }

        # Welch's t-test
        se1 = s1.variance / s1.count
        se2 = s2.variance / s2.count
        se_diff = math.sqrt(se1 + se2) if (se1 + se2) > 0 else 1e-10
        t_stat = (s1.mean - s2.mean) / se_diff

        # Degrees of freedom (Welch-Satterthwaite)
        if se1 + se2 > 0:
            df = (se1 + se2) ** 2 / (
                se1 ** 2 / max(s1.count - 1, 1) + se2 ** 2 / max(s2.count - 1, 1)
            )
        else:
            df = max(s1.count + s2.count - 2, 1)

        # Approximate p-value using normal approximation (for large df)
        p_value = 2.0 * (1.0 - self._normal_cdf(abs(t_stat)))

        return {
            "metric": name,
            "comparable": True,
            "run_1": {"mean": s1.mean, "std": s1.std_dev, "n": s1.count},
            "run_2": {"mean": s2.mean, "std": s2.std_dev, "n": s2.count},
            "diff_mean": s1.mean - s2.mean,
            "diff_percent": (
                ((s1.mean - s2.mean) / s2.mean * 100) if s2.mean != 0 else 0.0
            ),
            "t_statistic": t_stat,
            "degrees_of_freedom": df,
            "p_value": p_value,
            "significant": p_value < significance,
        }

    # -- Utilities --

    @staticmethod
    def _percentile(sorted_values: List[float], p: float) -> float:
        if not sorted_values:
            return 0.0
        k = (len(sorted_values) - 1) * p
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return sorted_values[int(k)]
        return sorted_values[int(f)] * (c - k) + sorted_values[int(c)] * (k - f)

    @staticmethod
    def _normal_cdf(x: float) -> float:
        """Approximate standard normal CDF using the error function."""
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    def metric_names(self) -> List[str]:
        return list(self._samples.keys())

    def clear(self) -> None:
        self._samples.clear()
        self._counts.clear()
        self._means.clear()
        self._m2s.clear()
        self._mins.clear()
        self._maxs.clear()
