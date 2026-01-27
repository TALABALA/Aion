"""
AION Reward Collector

Collects and aggregates reward signals from explicit feedback, implicit
behavioural signals, outcome metrics, and intrinsic motivation.  Performs
online normalisation and temporal discounting before delivering the
aggregate reward to the experience buffer.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

import numpy as np
import structlog

from aion.learning.config import RewardConfig
from aion.learning.types import RewardSignal, RewardSource, RewardType
from aion.learning.rewards.explicit import ExplicitFeedbackProcessor
from aion.learning.rewards.implicit import ImplicitSignalExtractor
from aion.learning.rewards.signals import SignalRegistry

if TYPE_CHECKING:
    from aion.core.kernel import AIONKernel

logger = structlog.get_logger(__name__)


class RewardCollector:
    """Collects reward signals from multiple sources and aggregates them."""

    def __init__(
        self,
        kernel: "AIONKernel",
        config: Optional[RewardConfig] = None,
    ):
        self.kernel = kernel
        self._config = config or RewardConfig()
        self._registry = SignalRegistry()
        self._pending: Dict[str, List[RewardSignal]] = defaultdict(list)

        # Online normalisation state (Welford's)
        self._reward_count = 0
        self._reward_mean = 0.0
        self._reward_m2 = 0.0
        self._reward_std = 1.0

        # Build default processors
        self._explicit = ExplicitFeedbackProcessor(weight=self._config.explicit_weight)
        self._implicit = ImplicitSignalExtractor(weight=self._config.implicit_weight)
        self._register_default_processors()

    # ------------------------------------------------------------------
    # Processor registration
    # ------------------------------------------------------------------

    def _register_default_processors(self) -> None:
        """Wire up all default signal processors."""
        r = self._registry

        # Explicit
        r.register_fn(RewardSource.EXPLICIT_POSITIVE, self._explicit.process_thumbs_up)
        r.register_fn(RewardSource.EXPLICIT_NEGATIVE, self._explicit.process_thumbs_down)
        r.register_fn(RewardSource.EXPLICIT_RATING, self._explicit.process_rating)
        r.register_fn(RewardSource.EXPLICIT_CORRECTION, self._explicit.process_correction)

        # Implicit
        r.register_fn(RewardSource.IMPLICIT_COMPLETION, self._implicit.process_completion)
        r.register_fn(RewardSource.IMPLICIT_ABANDONMENT, self._implicit.process_abandonment)
        r.register_fn(RewardSource.IMPLICIT_RETRY, self._implicit.process_retry)
        r.register_fn(RewardSource.IMPLICIT_COPY, self._implicit.process_copy)
        r.register_fn(RewardSource.IMPLICIT_DWELL_TIME, self._implicit.process_dwell_time)
        r.register_fn(RewardSource.IMPLICIT_EDIT_DISTANCE, self._implicit.process_edit_distance)

        # Outcome
        r.register_fn(RewardSource.OUTCOME_SUCCESS, self._process_success)
        r.register_fn(RewardSource.OUTCOME_FAILURE, self._process_failure)
        r.register_fn(RewardSource.OUTCOME_PARTIAL, self._process_partial)

        # Metrics
        r.register_fn(RewardSource.METRIC_LATENCY, self._process_latency)
        r.register_fn(RewardSource.METRIC_QUALITY, self._process_quality)
        r.register_fn(RewardSource.METRIC_COST, self._process_cost)
        r.register_fn(RewardSource.METRIC_SAFETY, self._process_safety)

    # ------------------------------------------------------------------
    # Public collection API
    # ------------------------------------------------------------------

    async def collect(
        self,
        source: RewardSource,
        interaction_id: str,
        value: Any,
        action_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> RewardSignal:
        """Collect a single reward signal."""
        processor = self._registry.get(source)
        if not processor:
            logger.warning("no_processor_for_source", source=source.value)
            return RewardSignal(source=source, value=0.0, interaction_id=interaction_id)

        signal = await processor.process(value, metadata or {})
        signal.source = source
        signal.interaction_id = interaction_id
        signal.action_id = action_id
        if metadata:
            signal.metadata.update(metadata)

        self._pending[interaction_id].append(signal)
        self._update_normalisation(signal.value)
        return signal

    async def collect_explicit_feedback(
        self,
        interaction_id: str,
        feedback_type: str,
        value: Any,
        action_id: Optional[str] = None,
    ) -> RewardSignal:
        """Convenience: collect explicit user feedback."""
        source_map = {
            "thumbs_up": RewardSource.EXPLICIT_POSITIVE,
            "thumbs_down": RewardSource.EXPLICIT_NEGATIVE,
            "rating": RewardSource.EXPLICIT_RATING,
            "correction": RewardSource.EXPLICIT_CORRECTION,
        }
        source = source_map.get(feedback_type, RewardSource.EXPLICIT_POSITIVE)
        return await self.collect(source, interaction_id, value, action_id)

    async def collect_outcome(
        self,
        interaction_id: str,
        success: bool,
        partial: bool = False,
        metrics: Optional[Dict[str, float]] = None,
    ) -> List[RewardSignal]:
        """Collect outcome signal plus optional metric signals."""
        signals: List[RewardSignal] = []

        if partial:
            source = RewardSource.OUTCOME_PARTIAL
        elif success:
            source = RewardSource.OUTCOME_SUCCESS
        else:
            source = RewardSource.OUTCOME_FAILURE

        signals.append(await self.collect(source, interaction_id, {"success": success}))

        if metrics:
            metric_sources = {
                "latency": RewardSource.METRIC_LATENCY,
                "quality": RewardSource.METRIC_QUALITY,
                "cost": RewardSource.METRIC_COST,
                "safety": RewardSource.METRIC_SAFETY,
            }
            for name, val in metrics.items():
                ms = metric_sources.get(name)
                if ms:
                    signals.append(await self.collect(ms, interaction_id, val))

        return signals

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    async def aggregate_rewards(
        self,
        interaction_id: str,
        gamma: float = 0.99,
    ) -> float:
        """Aggregate all pending signals for an interaction."""
        signals = self._pending.get(interaction_id, [])
        if not signals:
            return 0.0
        total = sum(s.discounted_value(gamma) for s in signals)
        normalised = self._normalise_reward(total)
        return float(np.clip(normalised, -self._config.reward_clip, self._config.reward_clip))

    def get_pending_rewards(self, interaction_id: str) -> List[RewardSignal]:
        return list(self._pending.get(interaction_id, []))

    def clear_pending(self, interaction_id: str) -> None:
        self._pending.pop(interaction_id, None)

    # ------------------------------------------------------------------
    # Outcome / metric processors
    # ------------------------------------------------------------------

    async def _process_success(self, value: Any, metadata: Dict) -> RewardSignal:
        return RewardSignal(
            reward_type=RewardType.DELAYED,
            value=1.0 * self._config.outcome_weight,
            confidence=0.9,
        )

    async def _process_failure(self, value: Any, metadata: Dict) -> RewardSignal:
        return RewardSignal(
            reward_type=RewardType.DELAYED,
            value=-1.0 * self._config.outcome_weight,
            confidence=0.9,
        )

    async def _process_partial(self, value: Any, metadata: Dict) -> RewardSignal:
        return RewardSignal(
            reward_type=RewardType.DELAYED,
            value=0.3 * self._config.outcome_weight,
            confidence=0.7,
        )

    async def _process_latency(self, value: Any, metadata: Dict) -> RewardSignal:
        latency_ms = float(value)
        # Piecewise linear: <1s = +0.5, 1-5s linear, >5s = -0.5
        if latency_ms < 1000:
            normalised = 0.5
        elif latency_ms > 5000:
            normalised = -0.5
        else:
            normalised = 0.5 - (latency_ms - 1000) / 4000.0
        return RewardSignal(
            reward_type=RewardType.IMMEDIATE,
            value=normalised * self._config.metric_weight,
            confidence=1.0,
        )

    async def _process_quality(self, value: Any, metadata: Dict) -> RewardSignal:
        quality = float(value)
        normalised = (quality - 0.5) * 2.0  # [0,1] -> [-1,1]
        return RewardSignal(
            reward_type=RewardType.IMMEDIATE,
            value=normalised * self._config.metric_weight,
            confidence=0.8,
        )

    async def _process_cost(self, value: Any, metadata: Dict) -> RewardSignal:
        cost = float(value)
        # Lower cost = higher reward; normalise against expected cost
        expected = metadata.get("expected_cost", 1.0)
        ratio = cost / max(expected, 0.001)
        normalised = 1.0 - min(ratio, 2.0)  # ratio=0 -> +1, ratio=2 -> -1
        return RewardSignal(
            reward_type=RewardType.IMMEDIATE,
            value=normalised * self._config.metric_weight,
            confidence=0.7,
        )

    async def _process_safety(self, value: Any, metadata: Dict) -> RewardSignal:
        safety_score = float(value)  # 0 = unsafe, 1 = safe
        normalised = (safety_score - 0.5) * 2.0
        return RewardSignal(
            reward_type=RewardType.IMMEDIATE,
            value=normalised * self._config.metric_weight,
            confidence=0.95,
        )

    # ------------------------------------------------------------------
    # Online normalisation
    # ------------------------------------------------------------------

    def _update_normalisation(self, reward: float) -> None:
        self._reward_count += 1
        if self._config.use_ema_normalisation:
            # Exponentially-weighted moving average — adapts to
            # non-stationary reward distributions (user preferences shift)
            alpha = 1.0 - self._config.ema_decay
            if self._reward_count == 1:
                self._reward_mean = reward
                self._reward_m2 = 0.0
            else:
                delta = reward - self._reward_mean
                self._reward_mean += alpha * delta
                self._reward_m2 = (
                    self._config.ema_decay * (self._reward_m2 + alpha * delta * delta)
                )
            if self._reward_count > 10:
                self._reward_std = max(np.sqrt(self._reward_m2), 0.01)
        else:
            # Welford's algorithm — equal weight to all history (stationary)
            delta = reward - self._reward_mean
            self._reward_mean += delta / self._reward_count
            delta2 = reward - self._reward_mean
            self._reward_m2 += delta * delta2
            if self._reward_count > 10:
                self._reward_std = max(np.sqrt(self._reward_m2 / self._reward_count), 0.01)

    def _normalise_reward(self, reward: float) -> float:
        if self._reward_std > 0:
            return (reward - self._reward_mean) / self._reward_std
        return reward

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        return {
            "pending_interactions": len(self._pending),
            "total_signals": sum(len(v) for v in self._pending.values()),
            "reward_mean": self._reward_mean,
            "reward_std": self._reward_std,
            "registered_sources": len(self._registry.sources),
        }
