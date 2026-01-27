"""
AION Statistical Analysis

Provides statistical tests for experiment evaluation:
- Welch's t-test for comparing means
- Mann-Whitney U test for non-parametric comparison
- Sequential testing with alpha spending
- Effect size estimation (Cohen's d)
- Confidence intervals
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy import stats as sp_stats


@dataclass
class AnalysisResult:
    """Result of a statistical analysis."""
    p_value: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    power: float
    significant: bool
    test_name: str
    detail: str = ""


class StatisticalAnalyzer:
    """Statistical analysis utilities for A/B experiments."""

    @staticmethod
    def welch_t_test(
        control_samples: List[float],
        treatment_samples: List[float],
        significance_level: float = 0.05,
    ) -> AnalysisResult:
        """Welch's t-test (unequal variance)."""
        c = np.array(control_samples)
        t = np.array(treatment_samples)

        if len(c) < 2 or len(t) < 2:
            return AnalysisResult(
                p_value=1.0, effect_size=0.0,
                confidence_interval=(0.0, 0.0), power=0.0,
                significant=False, test_name="welch_t_test",
                detail="Insufficient samples",
            )

        t_stat, p_value = sp_stats.ttest_ind(t, c, equal_var=False)

        # Cohen's d
        pooled_std = math.sqrt(
            ((len(t) - 1) * np.var(t, ddof=1) + (len(c) - 1) * np.var(c, ddof=1))
            / (len(t) + len(c) - 2)
        )
        effect_size = (np.mean(t) - np.mean(c)) / pooled_std if pooled_std > 0 else 0.0

        # Confidence interval for difference in means
        se = math.sqrt(np.var(t, ddof=1) / len(t) + np.var(c, ddof=1) / len(c))
        diff = float(np.mean(t) - np.mean(c))
        z = sp_stats.norm.ppf(1 - significance_level / 2)
        ci = (diff - z * se, diff + z * se)

        # Approximate power
        ncp = abs(effect_size) * math.sqrt(len(t) * len(c) / (len(t) + len(c)))
        power = 1 - sp_stats.t.cdf(
            sp_stats.t.ppf(1 - significance_level / 2, df=len(t) + len(c) - 2),
            df=len(t) + len(c) - 2,
            loc=ncp,
        )

        return AnalysisResult(
            p_value=float(p_value),
            effect_size=float(effect_size),
            confidence_interval=ci,
            power=float(power),
            significant=p_value < significance_level,
            test_name="welch_t_test",
        )

    @staticmethod
    def mann_whitney_u(
        control_samples: List[float],
        treatment_samples: List[float],
        significance_level: float = 0.05,
    ) -> AnalysisResult:
        """Non-parametric Mann-Whitney U test."""
        c = np.array(control_samples)
        t = np.array(treatment_samples)

        if len(c) < 2 or len(t) < 2:
            return AnalysisResult(
                p_value=1.0, effect_size=0.0,
                confidence_interval=(0.0, 0.0), power=0.0,
                significant=False, test_name="mann_whitney_u",
                detail="Insufficient samples",
            )

        u_stat, p_value = sp_stats.mannwhitneyu(t, c, alternative="two-sided")
        # Rank-biserial correlation as effect size
        effect_size = 1 - (2 * u_stat) / (len(t) * len(c))

        return AnalysisResult(
            p_value=float(p_value),
            effect_size=float(effect_size),
            confidence_interval=(0.0, 0.0),
            power=0.0,
            significant=p_value < significance_level,
            test_name="mann_whitney_u",
        )


class SequentialTester:
    """
    Sequential testing with alpha-spending for early stopping.

    Implements O'Brien-Fleming and Pocock alpha-spending functions
    to control the family-wise error rate across multiple looks.
    """

    def __init__(
        self,
        total_alpha: float = 0.05,
        max_looks: int = 10,
        spending_function: str = "obrien_fleming",
    ):
        self.total_alpha = total_alpha
        self.max_looks = max_looks
        self.spending_function = spending_function
        self._looks_taken = 0

    def get_boundary(self, look: int) -> float:
        """Return the alpha boundary for the given look (1-indexed)."""
        info_fraction = look / self.max_looks

        if self.spending_function == "obrien_fleming":
            return self._obrien_fleming(info_fraction)
        elif self.spending_function == "pocock":
            return self._pocock(info_fraction)
        else:
            # Linear spending
            return self.total_alpha * info_fraction

    def should_stop(
        self,
        p_value: float,
    ) -> bool:
        """Check if we should stop at the current look."""
        self._looks_taken += 1
        boundary = self.get_boundary(self._looks_taken)
        return p_value < boundary

    def _obrien_fleming(self, info_fraction: float) -> float:
        """O'Brien-Fleming alpha spending: conservative early, liberal late."""
        if info_fraction <= 0:
            return 0.0
        z = sp_stats.norm.ppf(1 - self.total_alpha / 2) / math.sqrt(info_fraction)
        return float(2 * (1 - sp_stats.norm.cdf(z)))

    def _pocock(self, info_fraction: float) -> float:
        """Pocock alpha spending: uniform across looks."""
        if info_fraction <= 0:
            return 0.0
        return float(self.total_alpha * math.log(1 + (math.e - 1) * info_fraction))
