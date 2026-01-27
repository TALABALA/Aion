"""
AION Experience Buffer

Prioritised experience replay buffer with sum-tree sampling (O(log N)),
importance-sampling weight correction, and n-step return support.
Based on Schaul et al., 2015.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import structlog

from aion.learning.config import BufferConfig
from aion.learning.types import Experience
from aion.learning.experience.sampling import ImportanceWeightCalculator, PrioritySampler

logger = structlog.get_logger(__name__)


class ExperienceBuffer:
    """Experience replay buffer with proportional priority sampling via sum-tree."""

    def __init__(self, config: Optional[BufferConfig] = None):
        self.config = config or BufferConfig()
        self._experiences: List[Optional[Experience]] = [None] * self.config.max_size
        self._sampler = PrioritySampler(self.config.max_size)
        self._write_idx = 0
        self._size = 0
        self._max_priority = 1.0
        self._total_added = 0
        self._current_beta = self.config.priority_beta

    # ------------------------------------------------------------------
    # Insertion
    # ------------------------------------------------------------------

    def add(self, experience: Experience) -> None:
        """Add experience with max priority (optimistic initialisation)."""
        experience.priority = self._max_priority
        idx = self._write_idx
        self._experiences[idx] = experience
        self._sampler.add(self._max_priority)
        self._write_idx = (self._write_idx + 1) % self.config.max_size
        self._size = min(self._size + 1, self.config.max_size)
        self._total_added += 1

    def add_batch(self, experiences: List[Experience]) -> None:
        for exp in experiences:
            self.add(exp)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(
        self,
        batch_size: int,
        beta: Optional[float] = None,
    ) -> Tuple[List[Experience], np.ndarray, List[int]]:
        """Sample a batch with priority-based or uniform sampling."""
        if self._size < self.config.min_size_for_sampling:
            return [], np.array([], dtype=np.float32), []

        batch_size = min(batch_size, self._size)
        beta = beta if beta is not None else self._current_beta

        if self.config.use_priority:
            return self._priority_sample(batch_size, beta)
        return self._uniform_sample(batch_size)

    def _uniform_sample(
        self, batch_size: int
    ) -> Tuple[List[Experience], np.ndarray, List[int]]:
        indices = random.sample(range(self._size), batch_size)
        experiences = [self._experiences[i] for i in indices]  # type: ignore[misc]
        weights = np.ones(batch_size, dtype=np.float32)
        return experiences, weights, indices

    def _priority_sample(
        self, batch_size: int, beta: float
    ) -> Tuple[List[Experience], np.ndarray, List[int]]:
        # O(log N) sampling via sum-tree
        indices, priorities = self._sampler.sample(batch_size)

        # Clamp indices to valid range
        indices = [min(max(i, 0), self._size - 1) for i in indices]

        # Compute probabilities from priorities
        total = self._sampler.total
        if total == 0:
            return self._uniform_sample(batch_size)

        probs = priorities / total
        weights = ImportanceWeightCalculator.compute(probs, self._size, beta)

        # Anneal beta towards 1
        self._current_beta = min(
            1.0, self._current_beta + self.config.priority_beta_increment
        )

        experiences = [self._experiences[i] for i in indices]  # type: ignore[misc]
        return experiences, weights, indices

    # ------------------------------------------------------------------
    # Priority updates
    # ------------------------------------------------------------------

    def update_priorities(self, indices: List[int], td_errors: np.ndarray) -> None:
        """Update priorities based on TD errors."""
        for idx, td_error in zip(indices, td_errors):
            if 0 <= idx < self._size:
                priority = (abs(td_error) + self.config.priority_epsilon) ** self.config.priority_alpha
                self._sampler.update(idx, priority)
                self._experiences[idx].td_error = float(td_error)  # type: ignore[union-attr]
                self._max_priority = max(self._max_priority, priority)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_by_interaction(self, interaction_id: str) -> List[Experience]:
        return [
            e for e in self._experiences[:self._size]
            if e is not None and e.interaction_id == interaction_id
        ]

    def get_recent(self, n: int = 100) -> List[Experience]:
        """Return the n most recently added experiences."""
        if self._total_added <= self.config.max_size:
            start = max(0, self._size - n)
            return [e for e in self._experiences[start:self._size] if e is not None]
        # Buffer has wrapped — recent items are behind write_idx
        result: List[Experience] = []
        for i in range(n):
            idx = (self._write_idx - 1 - i) % self.config.max_size
            exp = self._experiences[idx]
            if exp is not None:
                result.append(exp)
            if len(result) >= n:
                break
        return list(reversed(result))

    def __len__(self) -> int:
        return self._size

    def is_ready(self) -> bool:
        return self._size >= self.config.min_size_for_sampling

    def get_stats(self) -> Dict[str, Any]:
        valid = [e for e in self._experiences[:self._size] if e is not None]
        rewards = [e.cumulative_reward for e in valid] if valid else [0.0]
        return {
            "size": self._size,
            "capacity": self.config.max_size,
            "total_added": self._total_added,
            "max_priority": self._max_priority,
            "current_beta": self._current_beta,
            "ready": self.is_ready(),
            "avg_reward": float(np.mean(rewards)),
            "reward_std": float(np.std(rewards)),
        }
