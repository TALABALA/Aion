"""
AION Contextual Bandit Algorithms

Implements LinUCB (Li et al., 2010) and Hybrid LinUCB for contextual
multi-armed bandits with disjoint or shared feature models.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


class LinUCB:
    """
    LinUCB with disjoint linear models (Li et al., 2010).

    Each arm maintains its own ridge regression model.
    UCB score = x^T theta_a + alpha * sqrt(x^T A_a^{-1} x)
    """

    def __init__(
        self,
        feature_dim: int = 10,
        alpha: float = 1.0,
        regularisation: float = 1.0,
        name: str = "linucb",
    ):
        self.name = name
        self.feature_dim = feature_dim
        self.alpha = alpha
        self.regularisation = regularisation
        self._arms: Dict[str, Dict] = {}

    def add_arm(self, arm_id: str) -> None:
        if arm_id not in self._arms:
            self._arms[arm_id] = {
                "A": self.regularisation * np.eye(self.feature_dim),
                "b": np.zeros(self.feature_dim),
                "A_inv": np.eye(self.feature_dim) / self.regularisation,
                "theta": np.zeros(self.feature_dim),
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
    ) -> Tuple[str, float]:
        """Select arm with highest UCB score."""
        arms = available_arms or list(self._arms.keys())
        if not arms:
            raise ValueError("No arms available")

        ctx = self._pad_context(context)
        scores = {}

        for arm_id in arms:
            self.add_arm(arm_id)
            arm = self._arms[arm_id]
            theta = arm["A_inv"] @ arm["b"]
            pred = float(ctx @ theta)
            uncertainty = float(np.sqrt(ctx @ arm["A_inv"] @ ctx))
            scores[arm_id] = pred + self.alpha * uncertainty

        best = max(scores, key=scores.get)  # type: ignore[arg-type]
        return best, scores[best]

    def update(self, arm_id: str, context: np.ndarray, reward: float) -> None:
        """Update the arm's linear model."""
        self.add_arm(arm_id)
        arm = self._arms[arm_id]
        ctx = self._pad_context(context)

        arm["A"] += np.outer(ctx, ctx)
        arm["b"] += reward * ctx
        # Sherman-Morrison update for A_inv
        A_inv = arm["A_inv"]
        numerator = A_inv @ np.outer(ctx, ctx) @ A_inv
        denominator = 1.0 + float(ctx @ A_inv @ ctx)
        arm["A_inv"] = A_inv - numerator / denominator
        arm["theta"] = arm["A_inv"] @ arm["b"]
        arm["pulls"] += 1

    def get_expected_reward(self, arm_id: str, context: np.ndarray) -> float:
        self.add_arm(arm_id)
        ctx = self._pad_context(context)
        return float(ctx @ self._arms[arm_id]["theta"])


class HybridLinUCB:
    """
    Hybrid LinUCB with shared and arm-specific features.

    Combines a shared linear model (capturing global patterns) with
    per-arm linear models (capturing arm-specific effects).
    """

    def __init__(
        self,
        shared_dim: int = 5,
        arm_dim: int = 5,
        alpha: float = 1.0,
        regularisation: float = 1.0,
        name: str = "hybrid_linucb",
    ):
        self.name = name
        self.shared_dim = shared_dim
        self.arm_dim = arm_dim
        self.alpha = alpha
        self.regularisation = regularisation

        # Shared model
        self._A0 = regularisation * np.eye(shared_dim)
        self._b0 = np.zeros(shared_dim)
        self._A0_inv = np.eye(shared_dim) / regularisation

        # Per-arm models
        self._arms: Dict[str, Dict] = {}

    def add_arm(self, arm_id: str) -> None:
        if arm_id not in self._arms:
            self._arms[arm_id] = {
                "A": self.regularisation * np.eye(self.arm_dim),
                "B": np.zeros((self.arm_dim, self.shared_dim)),
                "b": np.zeros(self.arm_dim),
                "A_inv": np.eye(self.arm_dim) / self.regularisation,
                "pulls": 0,
            }

    def select(
        self,
        shared_context: np.ndarray,
        arm_contexts: Dict[str, np.ndarray],
        available_arms: Optional[List[str]] = None,
    ) -> Tuple[str, float]:
        """Select arm with highest hybrid UCB score."""
        arms = available_arms or list(arm_contexts.keys())
        if not arms:
            raise ValueError("No arms available")

        z = np.array(shared_context, dtype=np.float64).flatten()[:self.shared_dim]
        if len(z) < self.shared_dim:
            z = np.pad(z, (0, self.shared_dim - len(z)))

        beta_hat = self._A0_inv @ self._b0
        scores = {}

        for arm_id in arms:
            self.add_arm(arm_id)
            arm = self._arms[arm_id]
            x = np.array(arm_contexts.get(arm_id, np.zeros(self.arm_dim)), dtype=np.float64).flatten()[:self.arm_dim]
            if len(x) < self.arm_dim:
                x = np.pad(x, (0, self.arm_dim - len(x)))

            theta_hat = arm["A_inv"] @ (arm["b"] - arm["B"] @ beta_hat)
            pred = float(z @ beta_hat + x @ theta_hat)

            # UCB exploration term
            s_shared = float(z @ self._A0_inv @ z)
            s_arm = float(x @ arm["A_inv"] @ x)
            s_cross = float(-2.0 * z @ self._A0_inv @ arm["B"].T @ arm["A_inv"] @ x)
            s_mixed = float(x @ arm["A_inv"] @ arm["B"] @ self._A0_inv @ arm["B"].T @ arm["A_inv"] @ x)
            uncertainty = np.sqrt(abs(s_shared + s_arm + s_cross + s_mixed))

            scores[arm_id] = pred + self.alpha * uncertainty

        best = max(scores, key=scores.get)  # type: ignore[arg-type]
        return best, scores[best]

    def update(
        self,
        arm_id: str,
        shared_context: np.ndarray,
        arm_context: np.ndarray,
        reward: float,
    ) -> None:
        """Update both shared and arm-specific models."""
        self.add_arm(arm_id)
        arm = self._arms[arm_id]

        z = np.array(shared_context, dtype=np.float64).flatten()[:self.shared_dim]
        if len(z) < self.shared_dim:
            z = np.pad(z, (0, self.shared_dim - len(z)))

        x = np.array(arm_context, dtype=np.float64).flatten()[:self.arm_dim]
        if len(x) < self.arm_dim:
            x = np.pad(x, (0, self.arm_dim - len(x)))

        # Update shared model
        self._A0 += arm["B"].T @ arm["A_inv"] @ arm["B"]
        self._b0 += arm["B"].T @ arm["A_inv"] @ arm["b"]

        # Update arm model
        arm["A"] += np.outer(x, x)
        arm["B"] += np.outer(x, z)
        arm["b"] += reward * x

        # Update A_inv via Sherman-Morrison
        A_inv = arm["A_inv"]
        numerator = A_inv @ np.outer(x, x) @ A_inv
        denominator = 1.0 + float(x @ A_inv @ x)
        arm["A_inv"] = A_inv - numerator / denominator

        # Update shared model with new arm info
        self._A0 += np.outer(z, z) - arm["B"].T @ arm["A_inv"] @ arm["B"]
        self._b0 += reward * z - arm["B"].T @ arm["A_inv"] @ arm["b"]

        try:
            self._A0_inv = np.linalg.inv(self._A0)
        except np.linalg.LinAlgError:
            pass

        arm["pulls"] += 1
