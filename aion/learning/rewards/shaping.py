"""
AION Reward Shaping

Implements potential-based reward shaping (PBRS) to accelerate learning
while preserving optimal policy invariance (Ng et al., 1999).
Also provides curiosity-driven intrinsic rewards.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional

import numpy as np
import structlog

from aion.learning.types import RewardSignal, RewardSource, RewardType, StateRepresentation

logger = structlog.get_logger(__name__)


class RewardShaper(ABC):
    """Base class for reward shaping functions."""

    @abstractmethod
    def shape(
        self,
        state: StateRepresentation,
        next_state: Optional[StateRepresentation],
        raw_reward: float,
        gamma: float = 0.99,
    ) -> float:
        """Return the shaped reward."""
        ...


class PotentialBasedShaping(RewardShaper):
    """
    Potential-based reward shaping: F(s, s') = gamma * Phi(s') - Phi(s).

    Guarantees that the optimal policy is unchanged while providing
    denser learning signal.
    """

    def __init__(
        self,
        potential_fn: Optional[Callable[[StateRepresentation], float]] = None,
    ):
        self._potential_fn = potential_fn or self._default_potential

    def shape(
        self,
        state: StateRepresentation,
        next_state: Optional[StateRepresentation],
        raw_reward: float,
        gamma: float = 0.99,
    ) -> float:
        phi_s = self._potential_fn(state)
        phi_s_prime = self._potential_fn(next_state) if next_state else 0.0
        shaping_bonus = gamma * phi_s_prime - phi_s
        return raw_reward + shaping_bonus

    @staticmethod
    def _default_potential(state: StateRepresentation) -> float:
        """Default potential: reward recent positive outcomes."""
        if not state.recent_rewards:
            return 0.0
        return float(np.mean(state.recent_rewards))


class CuriosityRewardShaper(RewardShaper):
    """
    Curiosity-driven intrinsic reward based on prediction error.

    Rewards the agent for visiting states that are hard to predict,
    encouraging exploration of novel regions of the state space.
    """

    def __init__(self, prediction_error_scale: float = 0.1):
        self._scale = prediction_error_scale
        self._state_visit_counts: Dict[str, int] = {}
        self._max_visits = 0

    def shape(
        self,
        state: StateRepresentation,
        next_state: Optional[StateRepresentation],
        raw_reward: float,
        gamma: float = 0.99,
    ) -> float:
        state_key = self._state_key(state)
        count = self._state_visit_counts.get(state_key, 0) + 1
        self._state_visit_counts[state_key] = count
        self._max_visits = max(self._max_visits, count)
        # Intrinsic reward: inverse square-root of visit count
        intrinsic = self._scale / np.sqrt(count)
        return raw_reward + intrinsic

    @staticmethod
    def _state_key(state: StateRepresentation) -> str:
        return f"{state.query_type}:{int(state.query_complexity * 10)}:{state.turn_count}"


class CompositeRewardShaper(RewardShaper):
    """Composes multiple reward shapers with configurable weights."""

    def __init__(self) -> None:
        self._shapers: list[tuple[RewardShaper, float]] = []

    def add(self, shaper: RewardShaper, weight: float = 1.0) -> "CompositeRewardShaper":
        self._shapers.append((shaper, weight))
        return self

    def shape(
        self,
        state: StateRepresentation,
        next_state: Optional[StateRepresentation],
        raw_reward: float,
        gamma: float = 0.99,
    ) -> float:
        shaped = raw_reward
        for shaper, weight in self._shapers:
            bonus = shaper.shape(state, next_state, 0.0, gamma)  # only bonus
            shaped += weight * bonus
        return shaped
