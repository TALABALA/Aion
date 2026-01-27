"""
AION Base Policy Interface

Abstract base class for all learnable actor policies in the
actor-critic RL loop. Policies map states to action distributions.
The shared StateValueFunction (critic) is managed separately by
the PolicyOptimizer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from aion.learning.types import Experience, PolicyConfig, StateRepresentation


class BasePolicy(ABC):
    """Abstract actor policy that maps states to action distributions."""

    def __init__(self, config: PolicyConfig):
        self.config = config
        self._training = False
        self._update_count = 0

    @abstractmethod
    async def select_action(
        self,
        state: StateRepresentation,
        available_actions: List[str],
    ) -> Tuple[str, float]:
        """Select an action and return (choice, confidence)."""
        ...

    @abstractmethod
    async def update(
        self,
        experiences: List[Experience],
        weights: np.ndarray,
        advantages: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Update policy from a batch of experiences.

        Args:
            experiences: batch of experience tuples
            weights: importance sampling weights from prioritized replay
            advantages: A(s,a) computed by the value function (GAE or TD).
                       When provided, policies use advantage-based gradients
                       instead of raw rewards.

        Returns:
            Metrics dict with loss, td_errors, etc.
        """
        ...

    def should_explore(self) -> bool:
        return np.random.random() < self.config.exploration_rate

    def decay_exploration(self) -> None:
        self.config.exploration_rate = max(
            self.config.min_exploration,
            self.config.exploration_rate * self.config.exploration_decay,
        )

    def get_state(self) -> Dict[str, Any]:
        """Serialise policy state for persistence."""
        return {
            "name": self.config.name,
            "version": self.config.version,
            "exploration_rate": self.config.exploration_rate,
            "update_count": self._update_count,
        }
