"""
AION Explicit Feedback Processing

Handles user-provided feedback signals: thumbs up/down, star ratings,
free-text corrections, and preference comparisons.
"""

from __future__ import annotations

from typing import Any, Dict

from aion.learning.types import RewardSignal, RewardSource, RewardType


class ExplicitFeedbackProcessor:
    """Processes explicit user feedback into reward signals."""

    def __init__(
        self,
        positive_value: float = 1.0,
        negative_value: float = -1.0,
        rating_scale: tuple[int, int] = (1, 5),
        weight: float = 1.0,
    ):
        self._positive_value = positive_value
        self._negative_value = negative_value
        self._rating_min, self._rating_max = rating_scale
        self._weight = weight

    async def process_thumbs_up(
        self, value: Any, metadata: Dict[str, Any]
    ) -> RewardSignal:
        return RewardSignal(
            source=RewardSource.EXPLICIT_POSITIVE,
            reward_type=RewardType.IMMEDIATE,
            value=self._positive_value * self._weight,
            confidence=1.0,
        )

    async def process_thumbs_down(
        self, value: Any, metadata: Dict[str, Any]
    ) -> RewardSignal:
        return RewardSignal(
            source=RewardSource.EXPLICIT_NEGATIVE,
            reward_type=RewardType.IMMEDIATE,
            value=self._negative_value * self._weight,
            confidence=1.0,
        )

    async def process_rating(
        self, value: Any, metadata: Dict[str, Any]
    ) -> RewardSignal:
        rating = float(value)
        midpoint = (self._rating_max + self._rating_min) / 2.0
        half_range = (self._rating_max - self._rating_min) / 2.0
        normalised = (rating - midpoint) / half_range  # Maps to [-1, 1]
        return RewardSignal(
            source=RewardSource.EXPLICIT_RATING,
            reward_type=RewardType.IMMEDIATE,
            value=normalised * self._weight,
            confidence=1.0,
        )

    async def process_correction(
        self, value: Any, metadata: Dict[str, Any]
    ) -> RewardSignal:
        """A correction implies the response was wrong but the user engaged."""
        severity = metadata.get("severity", 0.5)
        return RewardSignal(
            source=RewardSource.EXPLICIT_CORRECTION,
            reward_type=RewardType.IMMEDIATE,
            value=-severity * self._weight,
            confidence=0.9,
            metadata={"correction": str(value)[:500]},
        )
