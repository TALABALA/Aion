"""
AION Thompson Sampling

Implements Thompson Sampling for multi-armed bandits (Beta-Bernoulli)
and contextual Thompson Sampling with Bayesian linear regression.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from aion.learning.types import ArmStatistics


class ThompsonSampling:
    """Thompson Sampling with Beta-Bernoulli model."""

    def __init__(self, name: str = "default"):
        self.name = name
        self.arms: Dict[str, ArmStatistics] = {}
        self.total_pulls = 0

    def add_arm(self, arm_id: str) -> None:
        if arm_id not in self.arms:
            self.arms[arm_id] = ArmStatistics(arm_id=arm_id)

    def remove_arm(self, arm_id: str) -> None:
        self.arms.pop(arm_id, None)

    def select(self, available_arms: Optional[List[str]] = None) -> str:
        """Select arm by sampling from posterior Beta distributions."""
        arms = available_arms or list(self.arms.keys())
        if not arms:
            raise ValueError("No arms available")

        for arm_id in arms:
            self.add_arm(arm_id)

        samples = {
            arm_id: np.random.beta(
                self.arms[arm_id].alpha,
                self.arms[arm_id].beta,
            )
            for arm_id in arms
        }
        return max(samples, key=samples.get)  # type: ignore[arg-type]

    def select_top_k(self, k: int, available_arms: Optional[List[str]] = None) -> List[str]:
        """Select top-k arms by posterior sampling."""
        arms = available_arms or list(self.arms.keys())
        for arm_id in arms:
            self.add_arm(arm_id)

        samples = {
            arm_id: np.random.beta(self.arms[arm_id].alpha, self.arms[arm_id].beta)
            for arm_id in arms
        }
        ranked = sorted(samples, key=samples.get, reverse=True)  # type: ignore[arg-type]
        return ranked[:k]

    def update(self, arm_id: str, reward: float) -> None:
        """Update arm posterior with observed reward."""
        self.add_arm(arm_id)
        self.arms[arm_id].update(float(np.clip(reward, 0, 1)))
        self.total_pulls += 1

    def batch_update(self, arm_id: str, rewards: List[float]) -> None:
        """Batch update from multiple observations."""
        for r in rewards:
            self.update(arm_id, r)

    def get_expected_rewards(self) -> Dict[str, float]:
        """Return posterior mean for each arm."""
        return {
            arm_id: stats.alpha / (stats.alpha + stats.beta)
            for arm_id, stats in self.arms.items()
        }

    def get_confidence_intervals(self, alpha: float = 0.05) -> Dict[str, tuple]:
        """Return credible intervals from the Beta posterior."""
        from scipy import stats as sp_stats

        intervals = {}
        for arm_id, arm in self.arms.items():
            dist = sp_stats.beta(arm.alpha, arm.beta)
            low, high = dist.ppf(alpha / 2), dist.ppf(1 - alpha / 2)
            intervals[arm_id] = (float(low), float(high))
        return intervals

    def get_stats(self) -> Dict[str, dict]:
        return {
            arm_id: {
                "pulls": s.pulls,
                "avg_reward": s.avg_reward,
                "alpha": s.alpha,
                "beta": s.beta,
                "expected": s.alpha / (s.alpha + s.beta),
            }
            for arm_id, s in self.arms.items()
        }


class ContextualThompsonSampling:
    """
    Contextual Thompson Sampling with Bayesian linear regression.

    Uses a multivariate normal posterior over weight vectors,
    one per arm, updated with observed (context, reward) pairs.
    """

    def __init__(
        self,
        name: str = "contextual",
        feature_dim: int = 10,
        prior_variance: float = 1.0,
        noise_variance: float = 0.1,
    ):
        self.name = name
        self.feature_dim = feature_dim
        self.prior_variance = prior_variance
        self.noise_variance = noise_variance
        self._arms: Dict[str, Dict] = {}

    def add_arm(self, arm_id: str) -> None:
        if arm_id not in self._arms:
            self._arms[arm_id] = {
                "B": (1.0 / self.prior_variance) * np.eye(self.feature_dim),
                "B_inv": self.prior_variance * np.eye(self.feature_dim),
                "mu": np.zeros(self.feature_dim),
                "f": np.zeros(self.feature_dim),
                "pulls": 0,
            }

    def _pad_context(self, context: np.ndarray) -> np.ndarray:
        ctx = np.array(context, dtype=np.float64).flatten()[:self.feature_dim]
        if len(ctx) < self.feature_dim:
            ctx = np.pad(ctx, (0, self.feature_dim - len(ctx)))
        return ctx

    def select(
        self,
        context: np.ndarray,
        available_arms: Optional[List[str]] = None,
    ) -> str:
        """Select arm by sampling weights from posterior.

        Uses pre-maintained B_inv for O(d^2) per arm instead of O(d^3).
        """
        arms = available_arms or list(self._arms.keys())
        if not arms:
            raise ValueError("No arms available")

        ctx = self._pad_context(context)
        samples = {}

        for arm_id in arms:
            self.add_arm(arm_id)
            arm = self._arms[arm_id]
            try:
                theta = np.random.multivariate_normal(arm["mu"], arm["B_inv"])
            except np.linalg.LinAlgError:
                theta = arm["mu"]
            samples[arm_id] = float(np.dot(ctx, theta))

        return max(samples, key=samples.get)  # type: ignore[arg-type]

    def update(self, arm_id: str, context: np.ndarray, reward: float) -> None:
        """Bayesian update of the linear model for this arm.

        Uses Sherman-Morrison formula to maintain B_inv in O(d^2)
        instead of recomputing via O(d^3) matrix inversion.
        """
        self.add_arm(arm_id)
        arm = self._arms[arm_id]
        ctx = self._pad_context(context)

        outer = np.outer(ctx, ctx) / self.noise_variance
        arm["B"] += outer
        arm["f"] += reward * ctx / self.noise_variance

        # Sherman-Morrison update for B_inv: (A + uv^T)^{-1}
        B_inv = arm["B_inv"]
        Bx = B_inv @ ctx
        denom = self.noise_variance + float(ctx @ Bx)
        arm["B_inv"] = B_inv - np.outer(Bx, Bx) / denom

        arm["mu"] = arm["B_inv"] @ arm["f"]
        arm["pulls"] += 1

    def get_expected_reward(self, arm_id: str, context: np.ndarray) -> float:
        """Return expected reward for arm given context."""
        self.add_arm(arm_id)
        ctx = self._pad_context(context)
        return float(np.dot(ctx, self._arms[arm_id]["mu"]))
