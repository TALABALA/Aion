"""
AION Implicit Signal Extraction

Extracts reward signals from implicit user behaviour: task completion,
abandonment, retries, copy events, dwell time, and edit distance.
"""

from __future__ import annotations

from typing import Any, Dict

from aion.learning.types import RewardSignal, RewardSource, RewardType


class ImplicitSignalExtractor:
    """Extracts reward signals from implicit behavioural cues."""

    def __init__(self, weight: float = 0.5):
        self._weight = weight

    async def process_completion(
        self, value: Any, metadata: Dict[str, Any]
    ) -> RewardSignal:
        return RewardSignal(
            source=RewardSource.IMPLICIT_COMPLETION,
            reward_type=RewardType.DELAYED,
            value=0.3 * self._weight,
            confidence=0.7,
        )

    async def process_abandonment(
        self, value: Any, metadata: Dict[str, Any]
    ) -> RewardSignal:
        return RewardSignal(
            source=RewardSource.IMPLICIT_ABANDONMENT,
            reward_type=RewardType.DELAYED,
            value=-0.5 * self._weight,
            confidence=0.5,
        )

    async def process_retry(
        self, value: Any, metadata: Dict[str, Any]
    ) -> RewardSignal:
        retry_count = int(metadata.get("retry_count", 1))
        penalty = min(-0.3 * retry_count, -1.0)
        return RewardSignal(
            source=RewardSource.IMPLICIT_RETRY,
            reward_type=RewardType.IMMEDIATE,
            value=penalty * self._weight,
            confidence=0.6,
        )

    async def process_copy(
        self, value: Any, metadata: Dict[str, Any]
    ) -> RewardSignal:
        return RewardSignal(
            source=RewardSource.IMPLICIT_COPY,
            reward_type=RewardType.IMMEDIATE,
            value=0.4 * self._weight,
            confidence=0.7,
        )

    async def process_dwell_time(
        self, value: Any, metadata: Dict[str, Any]
    ) -> RewardSignal:
        """Longer dwell time on a response suggests engagement."""
        seconds = float(value)
        if seconds < 2.0:
            # Too short — likely skipped
            normalised = -0.2
        elif seconds < 30.0:
            # Reasonable reading time
            normalised = 0.3
        elif seconds < 120.0:
            # Deep engagement
            normalised = 0.5
        else:
            # Possibly confused / stuck
            normalised = -0.1
        return RewardSignal(
            source=RewardSource.IMPLICIT_DWELL_TIME,
            reward_type=RewardType.DELAYED,
            value=normalised * self._weight,
            confidence=0.4,
        )

    async def process_edit_distance(
        self, value: Any, metadata: Dict[str, Any]
    ) -> RewardSignal:
        """Low edit distance from response to user's final version = good."""
        distance_ratio = float(value)  # 0 = identical, 1 = completely different
        normalised = 1.0 - 2.0 * distance_ratio  # Maps [0,1] -> [1,-1]
        return RewardSignal(
            source=RewardSource.IMPLICIT_EDIT_DISTANCE,
            reward_type=RewardType.DELAYED,
            value=normalised * self._weight,
            confidence=0.6,
        )
