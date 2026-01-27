"""
AION Reward Signal Processing

Defines the signal processing pipeline: each RewardSource maps to a
SignalProcessor that converts raw observations into normalised reward
values in [-1, 1].
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Coroutine, Dict, Optional

import structlog

from aion.learning.types import RewardSignal, RewardSource, RewardType

logger = structlog.get_logger(__name__)


class SignalProcessor(ABC):
    """Base class for reward signal processors."""

    source: RewardSource

    @abstractmethod
    async def process(
        self,
        value: Any,
        metadata: Dict[str, Any],
    ) -> RewardSignal:
        """Convert a raw observation into a RewardSignal."""
        ...


class SignalRegistry:
    """Registry mapping RewardSource -> SignalProcessor."""

    def __init__(self) -> None:
        self._processors: Dict[RewardSource, SignalProcessor] = {}

    def register(self, source: RewardSource, processor: SignalProcessor) -> None:
        self._processors[source] = processor
        logger.debug("signal_processor_registered", source=source.value)

    def get(self, source: RewardSource) -> Optional[SignalProcessor]:
        return self._processors.get(source)

    def register_fn(
        self,
        source: RewardSource,
        fn: Callable[..., Coroutine],
        reward_type: RewardType = RewardType.IMMEDIATE,
    ) -> None:
        """Register a simple async function as a processor."""

        class _FnProcessor(SignalProcessor):
            source_attr = source

            async def process(self, value: Any, metadata: Dict[str, Any]) -> RewardSignal:
                return await fn(value, metadata)

        proc = _FnProcessor()
        proc.source = source
        self.register(source, proc)

    @property
    def sources(self):
        return list(self._processors.keys())
