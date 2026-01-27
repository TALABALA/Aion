"""
AION Sampling Strategies

Implements prioritised experience replay sampling with importance
sampling weights (Schaul et al., 2015) and stratified sampling.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


class PrioritySampler:
    """
    Sum-tree based priority sampler for O(log N) prioritised sampling.

    Uses a sum tree data structure for efficient sampling proportional
    to priorities, with importance sampling weight correction.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._tree = np.zeros(2 * capacity)  # sum tree
        self._min_tree = np.full(2 * capacity, float("inf"))  # min tree
        self._max_priority = 1.0
        self._size = 0
        self._write_idx = 0

    def add(self, priority: float) -> int:
        """Add a new priority, returning the index."""
        idx = self._write_idx
        tree_idx = idx + self.capacity
        self._update(tree_idx, priority)
        self._write_idx = (self._write_idx + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)
        self._max_priority = max(self._max_priority, priority)
        return idx

    def update(self, idx: int, priority: float) -> None:
        """Update priority at a given index."""
        tree_idx = idx + self.capacity
        self._update(tree_idx, priority)
        self._max_priority = max(self._max_priority, priority)

    def sample(self, batch_size: int) -> Tuple[List[int], np.ndarray]:
        """Sample indices proportional to priorities."""
        indices = []
        total = self._tree[1]
        segment = total / batch_size

        for i in range(batch_size):
            low = segment * i
            high = segment * (i + 1)
            s = np.random.uniform(low, high)
            idx = self._retrieve(s)
            indices.append(idx - self.capacity)

        # Deduplicate
        indices = list(set(indices))
        while len(indices) < batch_size and self._size > batch_size:
            s = np.random.uniform(0, total)
            idx = self._retrieve(s) - self.capacity
            if idx not in indices:
                indices.append(idx)

        priorities = np.array([self._tree[i + self.capacity] for i in indices])
        return indices, priorities

    def _update(self, tree_idx: int, priority: float) -> None:
        self._tree[tree_idx] = priority
        self._min_tree[tree_idx] = priority
        while tree_idx > 1:
            tree_idx //= 2
            left = tree_idx * 2
            right = left + 1
            self._tree[tree_idx] = self._tree[left] + self._tree[right]
            self._min_tree[tree_idx] = min(self._min_tree[left], self._min_tree[right])

    def _retrieve(self, value: float) -> int:
        idx = 1
        while idx < self.capacity:
            left = 2 * idx
            right = left + 1
            if value <= self._tree[left]:
                idx = left
            else:
                value -= self._tree[left]
                idx = right
        return idx

    @property
    def total(self) -> float:
        return self._tree[1]

    @property
    def min_priority(self) -> float:
        return self._min_tree[1] if self._size > 0 else 0.0


class ImportanceWeightCalculator:
    """Calculates importance sampling weights for bias correction."""

    @staticmethod
    def compute(
        probabilities: np.ndarray,
        buffer_size: int,
        beta: float,
    ) -> np.ndarray:
        """
        Compute importance sampling weights: w_i = (N * P(i))^{-beta}.

        Normalised so max weight = 1.
        """
        weights = (buffer_size * probabilities) ** (-beta)
        weights /= weights.max()
        return weights.astype(np.float32)
