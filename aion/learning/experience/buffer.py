"""
AION Experience Buffer

Prioritised experience replay buffer with sum-tree sampling,
importance-sampling weight correction, and n-step return support.
"""

from __future__ import annotations

import random
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import structlog

from aion.learning.config import BufferConfig
from aion.learning.types import Experience
from aion.learning.experience.sampling import ImportanceWeightCalculator

logger = structlog.get_logger(__name__)


class ExperienceBuffer:
    """Experience replay buffer with proportional priority sampling."""

    def __init__(self, config: Optional[BufferConfig] = None):
        self.config = config or BufferConfig()
        self._experiences: deque[Experience] = deque(maxlen=self.config.max_size)
        self._priorities: List[float] = []
        self._max_priority = 1.0
        self._total_added = 0
        self._current_beta = self.config.priority_beta

    # ------------------------------------------------------------------
    # Insertion
    # ------------------------------------------------------------------

    def add(self, experience: Experience) -> None:
        """Add experience with max priority (optimistic initialisation)."""
        experience.priority = self._max_priority
        self._experiences.append(experience)
        if len(self._priorities) >= self.config.max_size:
            self._priorities.pop(0)
        self._priorities.append(self._max_priority)
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
        if len(self._experiences) < self.config.min_size_for_sampling:
            return [], np.array([], dtype=np.float32), []

        batch_size = min(batch_size, len(self._experiences))
        beta = beta if beta is not None else self._current_beta

        if self.config.use_priority:
            return self._priority_sample(batch_size, beta)
        return self._uniform_sample(batch_size)

    def _uniform_sample(
        self, batch_size: int
    ) -> Tuple[List[Experience], np.ndarray, List[int]]:
        indices = random.sample(range(len(self._experiences)), batch_size)
        experiences = [self._experiences[i] for i in indices]
        weights = np.ones(batch_size, dtype=np.float32)
        return experiences, weights, indices

    def _priority_sample(
        self, batch_size: int, beta: float
    ) -> Tuple[List[Experience], np.ndarray, List[int]]:
        n = len(self._experiences)
        priorities = np.array(self._priorities[:n], dtype=np.float64)
        priorities = priorities ** self.config.priority_alpha

        total = priorities.sum()
        if total == 0:
            return self._uniform_sample(batch_size)

        probs = priorities / total

        # Stratified sampling for reduced variance
        indices = self._stratified_sample(probs, batch_size)

        weights = ImportanceWeightCalculator.compute(probs[indices], n, beta)

        # Anneal beta towards 1
        self._current_beta = min(
            1.0, self._current_beta + self.config.priority_beta_increment
        )

        experiences = [self._experiences[i] for i in indices]
        return experiences, weights, indices.tolist()

    @staticmethod
    def _stratified_sample(probs: np.ndarray, batch_size: int) -> np.ndarray:
        """Stratified sampling: divide CDF into equal segments."""
        cdf = np.cumsum(probs)
        segment = 1.0 / batch_size
        indices = []
        for i in range(batch_size):
            u = np.random.uniform(segment * i, segment * (i + 1))
            idx = int(np.searchsorted(cdf, u))
            idx = min(idx, len(probs) - 1)
            indices.append(idx)
        return np.array(indices, dtype=np.int64)

    # ------------------------------------------------------------------
    # Priority updates
    # ------------------------------------------------------------------

    def update_priorities(self, indices: List[int], td_errors: np.ndarray) -> None:
        """Update priorities based on TD errors."""
        for idx, td_error in zip(indices, td_errors):
            if 0 <= idx < len(self._priorities):
                priority = (abs(td_error) + self.config.priority_epsilon) ** self.config.priority_alpha
                self._priorities[idx] = priority
                self._experiences[idx].td_error = float(td_error)
                self._max_priority = max(self._max_priority, priority)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_by_interaction(self, interaction_id: str) -> List[Experience]:
        return [e for e in self._experiences if e.interaction_id == interaction_id]

    def get_recent(self, n: int = 100) -> List[Experience]:
        return list(self._experiences)[-n:]

    def __len__(self) -> int:
        return len(self._experiences)

    def is_ready(self) -> bool:
        return len(self._experiences) >= self.config.min_size_for_sampling

    def get_stats(self) -> Dict[str, Any]:
        rewards = [e.cumulative_reward for e in self._experiences] if self._experiences else [0.0]
        return {
            "size": len(self._experiences),
            "capacity": self.config.max_size,
            "total_added": self._total_added,
            "max_priority": self._max_priority,
            "current_beta": self._current_beta,
            "ready": self.is_ready(),
            "avg_reward": float(np.mean(rewards)),
            "reward_std": float(np.std(rewards)),
        }
