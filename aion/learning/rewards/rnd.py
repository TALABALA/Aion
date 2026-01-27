"""
AION Random Network Distillation (RND) Curiosity

Implements RND (Burda et al., 2018) for intrinsic motivation.

Architecture:
    - f_target(s): Fixed random MLP (never trained)
    - f_predictor(s): Trainable MLP that learns to match f_target

Intrinsic reward = ||f_predictor(s) - f_target(s)||²

Novel states → high prediction error → high intrinsic reward.
As the predictor trains on visited states, familiar states produce
low error, driving the agent toward unexplored regions.

This replaces the count-based curiosity (1/√N) with a learned
novelty signal that generalises across continuous state spaces.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import structlog

from aion.learning.nn import MLP
from aion.learning.config import RNDConfig
from aion.learning.types import StateRepresentation
from aion.learning.rewards.shaping import RewardShaper

logger = structlog.get_logger(__name__)


class RNDCuriosityShaper(RewardShaper):
    """
    Random Network Distillation curiosity-driven intrinsic reward.

    Burda et al. (2018): "Exploration by Random Network Distillation"

    Two neural networks with identical architecture:
        f_target:    Fixed random weights (never updated)
        f_predictor: Trained to predict f_target's output

    Intrinsic reward = ||f_predictor(s) - f_target(s)||²

    This provides a learned novelty measure that:
    - Generalises across continuous state spaces (unlike count-based)
    - Requires no explicit state enumeration or hashing
    - Naturally decays as states become familiar
    - Is differentiable and integrates cleanly with gradient-based training
    """

    def __init__(self, config: Optional[RNDConfig] = None):
        self._config = config or RNDConfig()
        self._initialized = False

        self._target: Optional[MLP] = None
        self._predictor: Optional[MLP] = None

        # Running normalisation of intrinsic rewards (Welford)
        self._reward_mean = 0.0
        self._reward_var = 1.0
        self._reward_count = 0

    def _ensure_initialized(self, feature_dim: int) -> None:
        if self._initialized:
            return
        self._target = MLP(feature_dim, self._config.hidden_dim, self._config.embedding_dim)
        self._predictor = MLP(feature_dim, self._config.hidden_dim, self._config.embedding_dim)
        self._initialized = True

    # -----------------------------------------------------------------
    # RewardShaper interface
    # -----------------------------------------------------------------

    def shape(
        self,
        state: StateRepresentation,
        next_state: Optional[StateRepresentation],
        raw_reward: float,
        gamma: float = 0.99,
    ) -> float:
        """Add RND intrinsic reward to the raw extrinsic reward."""
        target_state = next_state if next_state is not None else state
        features = target_state.to_vector()
        intrinsic = self.compute_intrinsic_reward(features)
        return raw_reward + intrinsic

    # -----------------------------------------------------------------
    # Core RND operations
    # -----------------------------------------------------------------

    def compute_intrinsic_reward(self, state_features: np.ndarray) -> float:
        """Compute intrinsic reward for a single state.

        Returns scaled, optionally normalised prediction error.
        """
        features = np.atleast_1d(state_features).astype(np.float64)
        self._ensure_initialized(len(features))

        target_out = self._target.forward(features).flatten()    # type: ignore[union-attr]
        pred_out = self._predictor.forward(features).flatten()   # type: ignore[union-attr]

        # MSE prediction error
        error = float(np.mean((pred_out - target_out) ** 2))

        if self._config.normalise_intrinsic:
            error = self._normalise(error)

        return self._config.intrinsic_reward_scale * error

    def compute_intrinsic_batch(self, features: np.ndarray) -> np.ndarray:
        """Compute intrinsic rewards for a batch of states."""
        features = np.atleast_2d(features).astype(np.float64)
        self._ensure_initialized(features.shape[-1])

        target_out = self._target.forward(features)    # type: ignore[union-attr]
        pred_out = self._predictor.forward(features)   # type: ignore[union-attr]

        errors = np.mean((pred_out - target_out) ** 2, axis=1)

        if self._config.normalise_intrinsic:
            errors = np.array([self._normalise(e) for e in errors])

        return self._config.intrinsic_reward_scale * errors

    def train_predictor(self, state_features: np.ndarray) -> Dict[str, float]:
        """Train predictor to match target on observed states.

        This is called during the policy optimizer's training loop.
        As the predictor improves on visited states, intrinsic reward
        for those states decreases — driving exploration toward novelty.

        Args:
            state_features: (batch, dim) features from sampled experiences

        Returns:
            Training metrics.
        """
        features = np.atleast_2d(state_features).astype(np.float64)
        self._ensure_initialized(features.shape[-1])

        # Forward through both networks
        target_out = self._target.forward(features)      # type: ignore[union-attr]
        pred_out = self._predictor.forward(features)     # type: ignore[union-attr]

        # MSE loss gradient: d/d_pred of ||pred - target||² = 2·(pred - target)
        d_out = 2.0 * (pred_out - target_out) / features.shape[0]

        grads = self._predictor.backward(d_out, clip=self._config.gradient_clip)  # type: ignore[union-attr]
        self._predictor.apply_gradients(grads, self._config.learning_rate)       # type: ignore[union-attr]

        loss = float(np.mean((pred_out - target_out) ** 2))
        return {"rnd_loss": loss}

    # -----------------------------------------------------------------
    # Running normalisation
    # -----------------------------------------------------------------

    def _normalise(self, error: float) -> float:
        """Normalise intrinsic reward using running statistics."""
        self._reward_count += 1
        delta = error - self._reward_mean
        self._reward_mean += delta / self._reward_count
        delta2 = error - self._reward_mean
        self._reward_var += delta * delta2
        std = max(np.sqrt(self._reward_var / max(self._reward_count, 1)), 1e-8)
        return (error - self._reward_mean) / std

    # -----------------------------------------------------------------
    # Stats
    # -----------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        return {
            "initialized": self._initialized,
            "intrinsic_reward_mean": self._reward_mean,
            "intrinsic_reward_std": max(np.sqrt(self._reward_var / max(self._reward_count, 1)), 1e-8),
            "total_observations": self._reward_count,
        }
