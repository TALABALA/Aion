"""
AION Base Policy Interface

Abstract base class for all learnable policies in the RL loop.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

import numpy as np

from aion.learning.types import Experience, PolicyConfig, StateRepresentation


class BasePolicy(ABC):
    """Abstract policy that maps states to action distributions."""

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
    ) -> Dict[str, Any]:
        """Update policy from a batch of experiences. Return metrics."""
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
