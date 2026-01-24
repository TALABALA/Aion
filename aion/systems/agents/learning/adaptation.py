"""
Continuous Adaptation System

Implements continuous learning and adaptation for agents,
enabling real-time adjustment of behavior based on feedback.
"""

import asyncio
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional

import structlog

logger = structlog.get_logger()


class AdaptationStrategy(Enum):
    """Strategies for adaptation."""

    REACTIVE = "reactive"  # Immediate response to feedback
    PROACTIVE = "proactive"  # Anticipate needed changes
    CONSERVATIVE = "conservative"  # Slow, careful changes
    AGGRESSIVE = "aggressive"  # Fast, bold changes
    CONTEXTUAL = "contextual"  # Adapt based on context


@dataclass
class PerformanceMetric:
    """A single performance metric."""

    name: str
    value: float
    target: float
    weight: float = 1.0
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def gap(self) -> float:
        """Gap between current and target."""
        return self.target - self.value

    @property
    def relative_performance(self) -> float:
        """Performance relative to target (1.0 = at target)."""
        if self.target == 0:
            return 1.0 if self.value == 0 else 0.0
        return self.value / self.target


@dataclass
class AdaptationAction:
    """An adaptation action to take."""

    id: str
    description: str
    parameter: str
    old_value: Any
    new_value: Any
    reason: str
    confidence: float
    applied: bool = False
    timestamp: datetime = field(default_factory=datetime.now)


class PerformanceTracker:
    """Tracks performance metrics over time."""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self._metrics: dict[str, deque[PerformanceMetric]] = {}
        self._targets: dict[str, float] = {}
        self._weights: dict[str, float] = {}

    def record(
        self,
        name: str,
        value: float,
        target: Optional[float] = None,
        weight: float = 1.0,
    ) -> PerformanceMetric:
        """Record a metric value."""
        if name not in self._metrics:
            self._metrics[name] = deque(maxlen=self.window_size)

        if target is not None:
            self._targets[name] = target
        if weight != 1.0:
            self._weights[name] = weight

        metric = PerformanceMetric(
            name=name,
            value=value,
            target=self._targets.get(name, 1.0),
            weight=self._weights.get(name, 1.0),
        )

        self._metrics[name].append(metric)
        return metric

    def get_current(self, name: str) -> Optional[PerformanceMetric]:
        """Get most recent metric value."""
        if name in self._metrics and self._metrics[name]:
            return self._metrics[name][-1]
        return None

    def get_average(self, name: str, last_n: Optional[int] = None) -> float:
        """Get average metric value."""
        if name not in self._metrics:
            return 0.0

        metrics = list(self._metrics[name])
        if last_n:
            metrics = metrics[-last_n:]

        if not metrics:
            return 0.0

        return sum(m.value for m in metrics) / len(metrics)

    def get_trend(self, name: str, last_n: int = 10) -> float:
        """Get trend of metric (positive = improving)."""
        if name not in self._metrics:
            return 0.0

        metrics = list(self._metrics[name])[-last_n:]
        if len(metrics) < 2:
            return 0.0

        # Simple linear trend
        first_half = metrics[:len(metrics) // 2]
        second_half = metrics[len(metrics) // 2:]

        avg_first = sum(m.value for m in first_half) / len(first_half)
        avg_second = sum(m.value for m in second_half) / len(second_half)

        return avg_second - avg_first

    def get_overall_score(self) -> float:
        """Get weighted overall performance score."""
        if not self._metrics:
            return 0.0

        total_weight = 0.0
        weighted_sum = 0.0

        for name, metrics in self._metrics.items():
            if not metrics:
                continue

            metric = metrics[-1]
            weight = self._weights.get(name, 1.0)

            weighted_sum += metric.relative_performance * weight
            total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def get_underperforming(self, threshold: float = 0.8) -> list[str]:
        """Get metrics performing below threshold of target."""
        underperforming = []

        for name, metrics in self._metrics.items():
            if metrics:
                metric = metrics[-1]
                if metric.relative_performance < threshold:
                    underperforming.append(name)

        return underperforming

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary summary."""
        return {
            name: {
                "current": metrics[-1].value if metrics else None,
                "target": self._targets.get(name),
                "average": self.get_average(name),
                "trend": self.get_trend(name),
            }
            for name, metrics in self._metrics.items()
        }


class ContinuousAdapter:
    """
    Continuous adaptation system.

    Features:
    - Real-time performance monitoring
    - Automatic parameter tuning
    - Strategy-based adaptation
    - Rollback support
    - Adaptation history
    """

    def __init__(
        self,
        agent_id: str,
        strategy: AdaptationStrategy = AdaptationStrategy.CONTEXTUAL,
        adaptation_rate: float = 0.1,
        min_samples: int = 10,
    ):
        self.agent_id = agent_id
        self.strategy = strategy
        self.adaptation_rate = adaptation_rate
        self.min_samples = min_samples

        # Tracking
        self.performance_tracker = PerformanceTracker()
        self._parameters: dict[str, Any] = {}
        self._parameter_ranges: dict[str, tuple[float, float]] = {}

        # Adaptation history
        self._actions: list[AdaptationAction] = []
        self._action_counter = 0

        # State
        self._last_adaptation: Optional[datetime] = None
        self._adaptation_count = 0

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize adapter."""
        self._initialized = True
        logger.info("continuous_adapter_initialized", agent_id=self.agent_id)

    async def shutdown(self) -> None:
        """Shutdown adapter."""
        self._initialized = False
        logger.info("continuous_adapter_shutdown")

    def set_parameter(
        self,
        name: str,
        value: Any,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
    ) -> None:
        """Set a tunable parameter."""
        self._parameters[name] = value
        if min_val is not None and max_val is not None:
            self._parameter_ranges[name] = (min_val, max_val)

    def get_parameter(self, name: str) -> Any:
        """Get current parameter value."""
        return self._parameters.get(name)

    def record_performance(
        self,
        metric_name: str,
        value: float,
        target: Optional[float] = None,
    ) -> None:
        """Record a performance metric."""
        self.performance_tracker.record(metric_name, value, target)

    async def adapt(self) -> list[AdaptationAction]:
        """Run adaptation cycle."""
        # Check if we have enough samples
        overall_score = self.performance_tracker.get_overall_score()
        underperforming = self.performance_tracker.get_underperforming()

        if not underperforming and overall_score >= 0.9:
            return []  # Performing well, no adaptation needed

        actions = []

        # Generate adaptation actions based on strategy
        if self.strategy == AdaptationStrategy.REACTIVE:
            actions = await self._reactive_adapt(underperforming)
        elif self.strategy == AdaptationStrategy.PROACTIVE:
            actions = await self._proactive_adapt()
        elif self.strategy == AdaptationStrategy.CONSERVATIVE:
            actions = await self._conservative_adapt(underperforming)
        elif self.strategy == AdaptationStrategy.AGGRESSIVE:
            actions = await self._aggressive_adapt(underperforming)
        elif self.strategy == AdaptationStrategy.CONTEXTUAL:
            actions = await self._contextual_adapt(underperforming, overall_score)

        # Apply actions
        for action in actions:
            self._apply_action(action)

        self._last_adaptation = datetime.now()
        self._adaptation_count += 1

        logger.info(
            "adaptation_complete",
            actions=len(actions),
            overall_score=overall_score,
        )

        return actions

    async def _reactive_adapt(self, underperforming: list[str]) -> list[AdaptationAction]:
        """Reactive adaptation - immediate response to issues."""
        actions = []

        for metric_name in underperforming:
            # Find related parameters
            for param_name, value in self._parameters.items():
                if not isinstance(value, (int, float)):
                    continue

                # Try adjusting parameter
                action = self._create_adjustment(
                    param_name, value,
                    direction=1,  # Increase
                    magnitude=self.adaptation_rate,
                    reason=f"Reactive adjustment for {metric_name}",
                )
                if action:
                    actions.append(action)
                    break

        return actions[:3]  # Limit actions

    async def _proactive_adapt(self) -> list[AdaptationAction]:
        """Proactive adaptation - anticipate needed changes."""
        actions = []

        # Look for negative trends
        for name in self.performance_tracker._metrics.keys():
            trend = self.performance_tracker.get_trend(name)

            if trend < -0.1:  # Declining performance
                # Find adjustable parameters
                for param_name, value in self._parameters.items():
                    if isinstance(value, (int, float)):
                        action = self._create_adjustment(
                            param_name, value,
                            direction=1,
                            magnitude=self.adaptation_rate * 0.5,
                            reason=f"Proactive adjustment for declining {name}",
                        )
                        if action:
                            actions.append(action)
                            break

        return actions[:2]

    async def _conservative_adapt(self, underperforming: list[str]) -> list[AdaptationAction]:
        """Conservative adaptation - slow, careful changes."""
        if not underperforming:
            return []

        # Only adapt most critical issue
        metric_name = underperforming[0]

        for param_name, value in self._parameters.items():
            if isinstance(value, (int, float)):
                action = self._create_adjustment(
                    param_name, value,
                    direction=1,
                    magnitude=self.adaptation_rate * 0.3,  # Smaller changes
                    reason=f"Conservative adjustment for {metric_name}",
                )
                if action:
                    return [action]

        return []

    async def _aggressive_adapt(self, underperforming: list[str]) -> list[AdaptationAction]:
        """Aggressive adaptation - fast, bold changes."""
        actions = []

        for metric_name in underperforming:
            for param_name, value in self._parameters.items():
                if isinstance(value, (int, float)):
                    action = self._create_adjustment(
                        param_name, value,
                        direction=1,
                        magnitude=self.adaptation_rate * 2,  # Larger changes
                        reason=f"Aggressive adjustment for {metric_name}",
                    )
                    if action:
                        actions.append(action)

        return actions[:5]

    async def _contextual_adapt(
        self,
        underperforming: list[str],
        overall_score: float,
    ) -> list[AdaptationAction]:
        """Contextual adaptation - based on current situation."""
        # Choose strategy based on context
        if overall_score < 0.5:
            return await self._aggressive_adapt(underperforming)
        elif overall_score < 0.7:
            return await self._reactive_adapt(underperforming)
        elif overall_score < 0.9:
            return await self._conservative_adapt(underperforming)
        else:
            return await self._proactive_adapt()

    def _create_adjustment(
        self,
        param_name: str,
        current_value: float,
        direction: int,
        magnitude: float,
        reason: str,
    ) -> Optional[AdaptationAction]:
        """Create an adjustment action."""
        if param_name not in self._parameter_ranges:
            return None

        min_val, max_val = self._parameter_ranges[param_name]

        # Calculate new value
        change = (max_val - min_val) * magnitude * direction
        new_value = current_value + change

        # Clamp to range
        new_value = max(min_val, min(max_val, new_value))

        if new_value == current_value:
            return None

        self._action_counter += 1

        return AdaptationAction(
            id=f"adapt-{self._action_counter}",
            description=f"Adjust {param_name}",
            parameter=param_name,
            old_value=current_value,
            new_value=new_value,
            reason=reason,
            confidence=0.7,
        )

    def _apply_action(self, action: AdaptationAction) -> None:
        """Apply an adaptation action."""
        self._parameters[action.parameter] = action.new_value
        action.applied = True
        self._actions.append(action)

        logger.debug(
            "adaptation_applied",
            parameter=action.parameter,
            old=action.old_value,
            new=action.new_value,
        )

    def rollback(self, action_id: str) -> bool:
        """Rollback an adaptation action."""
        for action in self._actions:
            if action.id == action_id and action.applied:
                self._parameters[action.parameter] = action.old_value
                action.applied = False
                logger.info("adaptation_rolled_back", action_id=action_id)
                return True
        return False

    def get_stats(self) -> dict[str, Any]:
        """Get adaptation statistics."""
        return {
            "agent_id": self.agent_id,
            "strategy": self.strategy.value,
            "adaptation_count": self._adaptation_count,
            "parameters": dict(self._parameters),
            "overall_score": self.performance_tracker.get_overall_score(),
            "performance": self.performance_tracker.to_dict(),
            "recent_actions": [
                {
                    "id": a.id,
                    "parameter": a.parameter,
                    "change": f"{a.old_value} -> {a.new_value}",
                    "reason": a.reason,
                }
                for a in self._actions[-5:]
            ],
        }
