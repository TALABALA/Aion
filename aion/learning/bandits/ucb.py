"""
AION Upper Confidence Bound (UCB) Algorithms

Implements UCB1 and Sliding-Window UCB for non-stationary environments.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

from aion.learning.types import ArmStatistics


class UCB1:
    """
    UCB1 algorithm (Auer et al., 2002).

    Selects the arm maximising: avg_reward + c * sqrt(ln(t) / n_i)
    """

    def __init__(self, confidence: float = 2.0, name: str = "ucb1"):
        self.name = name
        self.confidence = confidence
        self.arms: Dict[str, ArmStatistics] = {}
        self.total_pulls = 0

    def add_arm(self, arm_id: str) -> None:
        if arm_id not in self.arms:
            self.arms[arm_id] = ArmStatistics(arm_id=arm_id)

    def select(self, available_arms: Optional[List[str]] = None) -> str:
        arms = available_arms or list(self.arms.keys())
        if not arms:
            raise ValueError("No arms available")

        for arm_id in arms:
            self.add_arm(arm_id)

        # Pull each arm at least once
        for arm_id in arms:
            if self.arms[arm_id].pulls == 0:
                return arm_id

        scores = {}
        for arm_id in arms:
            arm = self.arms[arm_id]
            exploration_bonus = self.confidence * math.sqrt(
                math.log(self.total_pulls + 1) / arm.pulls
            )
            scores[arm_id] = arm.avg_reward + exploration_bonus

        return max(scores, key=scores.get)  # type: ignore[arg-type]

    def update(self, arm_id: str, reward: float) -> None:
        self.add_arm(arm_id)
        self.arms[arm_id].update(reward)
        self.total_pulls += 1

    def get_ucb_scores(self) -> Dict[str, Tuple[float, float, float]]:
        """Return (avg, bonus, ucb) for each arm."""
        result = {}
        for arm_id, arm in self.arms.items():
            if arm.pulls == 0:
                result[arm_id] = (0.0, float("inf"), float("inf"))
            else:
                bonus = self.confidence * math.sqrt(
                    math.log(self.total_pulls + 1) / arm.pulls
                )
                result[arm_id] = (arm.avg_reward, bonus, arm.avg_reward + bonus)
        return result


class SlidingWindowUCB:
    """
    Sliding-Window UCB for non-stationary environments.

    Only considers the most recent `window_size` observations per arm,
    allowing adaptation to changing reward distributions.
    """

    def __init__(
        self,
        window_size: int = 500,
        confidence: float = 2.0,
        name: str = "sw_ucb",
    ):
        self.name = name
        self.window_size = window_size
        self.confidence = confidence
        self._arm_windows: Dict[str, deque] = {}
        self.total_pulls = 0

    def add_arm(self, arm_id: str) -> None:
        if arm_id not in self._arm_windows:
            self._arm_windows[arm_id] = deque(maxlen=self.window_size)

    def select(self, available_arms: Optional[List[str]] = None) -> str:
        arms = available_arms or list(self._arm_windows.keys())
        if not arms:
            raise ValueError("No arms available")

        for arm_id in arms:
            self.add_arm(arm_id)

        for arm_id in arms:
            if len(self._arm_windows[arm_id]) == 0:
                return arm_id

        scores = {}
        for arm_id in arms:
            window = self._arm_windows[arm_id]
            n = len(window)
            avg = float(np.mean(list(window)))
            bonus = self.confidence * math.sqrt(math.log(self.total_pulls + 1) / n)
            scores[arm_id] = avg + bonus

        return max(scores, key=scores.get)  # type: ignore[arg-type]

    def update(self, arm_id: str, reward: float) -> None:
        self.add_arm(arm_id)
        self._arm_windows[arm_id].append(reward)
        self.total_pulls += 1

    def get_stats(self) -> Dict[str, dict]:
        return {
            arm_id: {
                "window_size": len(w),
                "avg_reward": float(np.mean(list(w))) if w else 0.0,
                "std_reward": float(np.std(list(w))) if w else 0.0,
            }
            for arm_id, w in self._arm_windows.items()
        }
