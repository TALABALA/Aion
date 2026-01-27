"""
AION State Value Function with Target Networks

Provides V(s) estimation via a two-layer MLP trained with TD(0),
a Polyak-averaged target network for stable bootstrapping, and
Generalized Advantage Estimation (GAE, Schulman et al. 2015).

This is the "critic" in the actor-critic architecture that all
three domain policies (tool, planning, agent) share.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from aion.learning.nn import MLP
from aion.learning.config import ValueFunctionConfig


class TargetNetwork:
    """Polyak-averaged target network for stable TD bootstrapping.

    Maintains a slowly-updated copy of a source MLP:
        θ_target ← τ·θ_source + (1 − τ)·θ_target

    This prevents the instability caused by bootstrapping from a
    rapidly-changing value function (Mnih et al., 2015).
    """

    def __init__(self, source: MLP, tau: float = 0.005):
        self._target = source.copy()
        self._tau = tau

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Evaluate target network (no gradient tracking needed)."""
        return self._target.forward(x)

    def soft_update(self, source: MLP) -> None:
        """Polyak averaging: θ_target ← τ·θ + (1-τ)·θ_target."""
        src = source.get_params()
        tgt = self._target.get_params()
        blended = {}
        for key in src:
            blended[key] = self._tau * src[key] + (1.0 - self._tau) * tgt[key]
        self._target.set_params(blended)

    def hard_update(self, source: MLP) -> None:
        """Copy source weights directly."""
        self._target.set_params(source.get_params())

    def get_params(self) -> Dict[str, np.ndarray]:
        return self._target.get_params()


class StateValueFunction:
    """State value function V(s) with TD learning and target network.

    Architecture:
        - Online network: MLP(feature_dim → hidden → 1) trained with TD(0)
        - Target network: Polyak-averaged copy for stable bootstrap targets
        - GAE: Computes advantages for multi-step trajectories

    TD(0) update:
        δ = r + γ·V_target(s') − V(s)
        V(s) ← V(s) + α·δ·∇V(s)

    GAE (Schulman et al., 2015):
        δ_t = r_t + γ·V(s_{t+1}) − V(s_t)
        Â_t = Σ_{l=0}^{T-t-1} (γλ)^l · δ_{t+l}
    """

    def __init__(
        self,
        feature_dim: int = 7,
        config: Optional[ValueFunctionConfig] = None,
    ):
        self.config = config or ValueFunctionConfig()
        self.feature_dim = feature_dim
        self._initialized = False

        # Lazily initialised on first call (to auto-detect feature dim)
        self._network: Optional[MLP] = None
        self._target: Optional[TargetNetwork] = None
        self._update_count = 0

    def _ensure_initialized(self, dim: int) -> None:
        if self._initialized:
            return
        self.feature_dim = dim
        self._network = MLP(dim, self.config.hidden_dim, 1)
        self._target = TargetNetwork(self._network, self.config.tau)
        self._initialized = True

    # -----------------------------------------------------------------
    # Inference
    # -----------------------------------------------------------------

    def predict(self, state_features: np.ndarray) -> float:
        """Predict V(s) using the online network."""
        self._ensure_initialized(len(state_features))
        out = self._network.forward(state_features)  # type: ignore[union-attr]
        return float(out.flatten()[0])

    def predict_batch(self, features: np.ndarray) -> np.ndarray:
        """Predict V(s) for a batch. features shape: (batch, dim)."""
        self._ensure_initialized(features.shape[-1])
        return self._network.forward(features).flatten()  # type: ignore[union-attr]

    def predict_target(self, state_features: np.ndarray) -> float:
        """Predict V(s) using the target network (for stable bootstrap)."""
        self._ensure_initialized(len(state_features))
        out = self._target.forward(state_features)  # type: ignore[union-attr]
        return float(out.flatten()[0])

    def predict_target_batch(self, features: np.ndarray) -> np.ndarray:
        """Predict V(s) for a batch using the target network."""
        self._ensure_initialized(features.shape[-1])
        return self._target.forward(features).flatten()  # type: ignore[union-attr]

    # -----------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------

    def update(
        self,
        states: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
        importance_weights: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """TD(0) update with importance sampling weights.

        Args:
            states: (batch, dim) current state features
            rewards: (batch,) immediate rewards
            next_states: (batch, dim) next state features
            dones: (batch,) 1.0 if terminal, 0.0 otherwise
            importance_weights: (batch,) IS weights from prioritized replay

        Returns:
            Metrics dict with loss and td_errors.
        """
        states = np.atleast_2d(states)
        next_states = np.atleast_2d(next_states)
        self._ensure_initialized(states.shape[-1])

        gamma = self.config.gamma
        if importance_weights is None:
            importance_weights = np.ones(len(rewards))

        # V(s) from online network
        v_current = self._network.forward(states).flatten()  # type: ignore[union-attr]

        # V(s') from target network (stable bootstrap)
        v_next_target = self._target.forward(next_states).flatten()  # type: ignore[union-attr]

        # TD target: r + γ·V_target(s') · (1 - done)
        td_targets = rewards + gamma * v_next_target * (1.0 - dones)

        # TD errors
        td_errors = td_targets - v_current

        # Loss gradient: d/dV(s) of 0.5·w·δ² = -w·δ
        d_out = -(importance_weights * td_errors).reshape(-1, 1)

        grads = self._network.backward(d_out, clip=self.config.gradient_clip)  # type: ignore[union-attr]
        self._network.apply_gradients(grads, self.config.learning_rate)  # type: ignore[union-attr]

        # Soft-update target network
        self._target.soft_update(self._network)  # type: ignore[arg-type]
        self._update_count += 1

        return {
            "value_loss": float(np.mean(td_errors ** 2)),
            "mean_value": float(np.mean(v_current)),
            "td_errors": td_errors.tolist(),
        }

    # -----------------------------------------------------------------
    # GAE (Generalized Advantage Estimation)
    # -----------------------------------------------------------------

    def compute_gae(
        self,
        states: List[np.ndarray],
        rewards: List[float],
        next_states: List[np.ndarray],
        dones: List[bool],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute GAE advantages and value targets.

        GAE(γ, λ) from Schulman et al. (2015):
            δ_t = r_t + γ·V(s_{t+1})·(1-d_t) − V(s_t)
            Â_t = Σ_{l=0}^{T-t-1} (γλ)^l · δ_{t+l}

        Args:
            states: list of state feature vectors
            rewards: list of rewards
            next_states: list of next-state feature vectors
            dones: list of terminal flags

        Returns:
            (advantages, value_targets) as numpy arrays
        """
        T = len(rewards)
        if T == 0:
            return np.array([]), np.array([])

        gamma = self.config.gamma
        lam = self.config.gae_lambda

        # Batch-predict values
        s_arr = np.array(states)
        ns_arr = np.array(next_states)
        self._ensure_initialized(s_arr.shape[-1])

        values = self.predict_batch(s_arr)
        next_values = self.predict_target_batch(ns_arr)

        # TD errors
        dones_arr = np.array(dones, dtype=np.float64)
        rewards_arr = np.array(rewards, dtype=np.float64)
        deltas = rewards_arr + gamma * next_values * (1.0 - dones_arr) - values

        # GAE backward sweep
        advantages = np.zeros(T, dtype=np.float64)
        gae = 0.0
        for t in reversed(range(T)):
            gae = deltas[t] + gamma * lam * (1.0 - dones_arr[t]) * gae
            advantages[t] = gae

        # Value targets = advantages + V(s)
        value_targets = advantages + values

        return advantages, value_targets

    # -----------------------------------------------------------------
    # Serialisation
    # -----------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        return {
            "initialized": self._initialized,
            "feature_dim": self.feature_dim,
            "hidden_dim": self.config.hidden_dim,
            "update_count": self._update_count,
            "tau": self.config.tau,
        }

    def get_params(self) -> Optional[Dict[str, np.ndarray]]:
        if self._network:
            return self._network.get_params()
        return None

    def set_params(self, params: Dict[str, np.ndarray]) -> None:
        if self._network:
            self._network.set_params(params)
            self._target.hard_update(self._network)  # type: ignore[union-attr]
