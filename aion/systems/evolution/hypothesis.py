"""
AION Hypothesis Generator

Generates and tests hypotheses for system improvement:
- Parameter optimization hypotheses
- Behavior modification hypotheses
- Performance prediction models
"""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class HypothesisType(str, Enum):
    """Types of hypotheses."""
    PARAMETER = "parameter"      # Adjust a parameter value
    THRESHOLD = "threshold"      # Modify a threshold
    STRATEGY = "strategy"        # Change a strategy/algorithm
    COMPOSITION = "composition"  # Tool/method composition
    SCHEDULING = "scheduling"    # Timing/scheduling changes


class HypothesisStatus(str, Enum):
    """Status of a hypothesis."""
    PENDING = "pending"
    TESTING = "testing"
    VALIDATED = "validated"
    REJECTED = "rejected"
    APPLIED = "applied"
    ROLLED_BACK = "rolled_back"


@dataclass
class Hypothesis:
    """A hypothesis for system improvement."""
    id: str
    type: HypothesisType
    name: str
    description: str
    target: str  # What to modify
    current_value: Any
    proposed_value: Any
    expected_improvement: float  # Expected improvement percentage
    confidence: float  # Confidence in the hypothesis (0-1)
    status: HypothesisStatus = HypothesisStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    tested_at: Optional[datetime] = None
    result_improvement: Optional[float] = None
    test_samples: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "type": self.type.value,
            "name": self.name,
            "description": self.description,
            "target": self.target,
            "current_value": self.current_value,
            "proposed_value": self.proposed_value,
            "expected_improvement": self.expected_improvement,
            "confidence": self.confidence,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "result_improvement": self.result_improvement,
        }


@dataclass
class PerformanceMetric:
    """A performance metric for tracking."""
    name: str
    current_value: float
    target_value: float
    weight: float = 1.0  # Importance weight
    is_higher_better: bool = True


class HypothesisGenerator:
    """
    Generates hypotheses for system improvement.

    Uses historical performance data and optimization strategies
    to propose changes that could improve system performance.
    """

    def __init__(
        self,
        max_change_percent: float = 0.1,
        min_confidence: float = 0.3,
    ):
        self.max_change_percent = max_change_percent
        self.min_confidence = min_confidence

        # Performance history
        self._metrics_history: dict[str, list[float]] = {}
        self._parameter_history: dict[str, list[tuple[Any, float]]] = {}

        # Hypothesis tracking
        self._hypotheses: dict[str, Hypothesis] = {}
        self._successful_hypotheses: list[str] = []
        self._failed_hypotheses: list[str] = []

    def record_metric(self, name: str, value: float) -> None:
        """Record a performance metric value."""
        if name not in self._metrics_history:
            self._metrics_history[name] = []
        self._metrics_history[name].append(value)

        # Keep only recent history
        if len(self._metrics_history[name]) > 1000:
            self._metrics_history[name] = self._metrics_history[name][-1000:]

    def record_parameter_performance(
        self,
        param_name: str,
        param_value: Any,
        performance: float,
    ) -> None:
        """Record performance for a parameter value."""
        if param_name not in self._parameter_history:
            self._parameter_history[param_name] = []
        self._parameter_history[param_name].append((param_value, performance))

    def generate_parameter_hypothesis(
        self,
        param_name: str,
        current_value: float,
        value_range: tuple[float, float],
    ) -> Optional[Hypothesis]:
        """
        Generate a hypothesis for parameter optimization.

        Args:
            param_name: Name of the parameter
            current_value: Current parameter value
            value_range: Valid range (min, max)

        Returns:
            Hypothesis or None if no improvement predicted
        """
        min_val, max_val = value_range

        # Calculate proposed change
        max_delta = (max_val - min_val) * self.max_change_percent

        # Use historical data if available
        if param_name in self._parameter_history:
            history = self._parameter_history[param_name]
            if len(history) >= 5:
                # Find best performing values
                sorted_history = sorted(history, key=lambda x: x[1], reverse=True)
                best_value = sorted_history[0][0]

                # Move towards best value
                if best_value != current_value:
                    direction = 1 if best_value > current_value else -1
                    proposed_value = current_value + direction * max_delta
                    proposed_value = max(min_val, min(max_val, proposed_value))

                    confidence = min(0.8, len(history) / 20)  # More data = more confidence

                    return Hypothesis(
                        id=str(uuid.uuid4()),
                        type=HypothesisType.PARAMETER,
                        name=f"Optimize {param_name}",
                        description=f"Adjust {param_name} towards historically better value",
                        target=param_name,
                        current_value=current_value,
                        proposed_value=proposed_value,
                        expected_improvement=0.05,  # Conservative estimate
                        confidence=confidence,
                    )

        # Random exploration
        direction = random.choice([-1, 1])
        proposed_value = current_value + direction * max_delta * random.random()
        proposed_value = max(min_val, min(max_val, proposed_value))

        if abs(proposed_value - current_value) < 0.001:
            return None

        return Hypothesis(
            id=str(uuid.uuid4()),
            type=HypothesisType.PARAMETER,
            name=f"Explore {param_name}",
            description=f"Random exploration of {param_name} parameter",
            target=param_name,
            current_value=current_value,
            proposed_value=proposed_value,
            expected_improvement=0.02,  # Conservative
            confidence=self.min_confidence,
        )

    def generate_threshold_hypothesis(
        self,
        threshold_name: str,
        current_value: float,
        performance_data: list[tuple[float, float]],  # (threshold, performance)
    ) -> Optional[Hypothesis]:
        """
        Generate a hypothesis for threshold optimization.

        Args:
            threshold_name: Name of the threshold
            current_value: Current threshold value
            performance_data: Historical (threshold, performance) pairs

        Returns:
            Hypothesis or None
        """
        if len(performance_data) < 3:
            return None

        # Simple gradient estimation
        sorted_data = sorted(performance_data, key=lambda x: x[0])
        thresholds = [d[0] for d in sorted_data]
        performances = [d[1] for d in sorted_data]

        # Find local gradient at current value
        idx = min(range(len(thresholds)), key=lambda i: abs(thresholds[i] - current_value))

        if idx > 0 and idx < len(thresholds) - 1:
            gradient = (performances[idx + 1] - performances[idx - 1]) / (
                thresholds[idx + 1] - thresholds[idx - 1]
            )
        elif idx == 0:
            gradient = (performances[1] - performances[0]) / (thresholds[1] - thresholds[0])
        else:
            gradient = (performances[-1] - performances[-2]) / (thresholds[-1] - thresholds[-2])

        if abs(gradient) < 0.001:
            return None

        # Propose moving in gradient direction
        step = (max(thresholds) - min(thresholds)) * self.max_change_percent
        if gradient > 0:
            proposed_value = current_value + step
        else:
            proposed_value = current_value - step

        proposed_value = max(min(thresholds), min(max(thresholds), proposed_value))

        return Hypothesis(
            id=str(uuid.uuid4()),
            type=HypothesisType.THRESHOLD,
            name=f"Optimize {threshold_name}",
            description=f"Gradient-based optimization of {threshold_name}",
            target=threshold_name,
            current_value=current_value,
            proposed_value=proposed_value,
            expected_improvement=abs(gradient) * step,
            confidence=0.5,
        )

    def generate_batch(
        self,
        parameters: dict[str, tuple[float, tuple[float, float]]],  # name -> (current, range)
        count: int = 5,
    ) -> list[Hypothesis]:
        """
        Generate a batch of hypotheses.

        Args:
            parameters: Dict mapping param name to (current_value, (min, max))
            count: Number of hypotheses to generate

        Returns:
            List of Hypothesis
        """
        hypotheses = []

        for param_name, (current, value_range) in parameters.items():
            h = self.generate_parameter_hypothesis(param_name, current, value_range)
            if h:
                hypotheses.append(h)
                self._hypotheses[h.id] = h

            if len(hypotheses) >= count:
                break

        return hypotheses[:count]

    def evaluate_hypothesis(
        self,
        hypothesis_id: str,
        baseline_performance: float,
        test_performance: float,
        sample_count: int,
    ) -> bool:
        """
        Evaluate a hypothesis based on test results.

        Args:
            hypothesis_id: Hypothesis to evaluate
            baseline_performance: Performance with original value
            test_performance: Performance with proposed value
            sample_count: Number of samples in test

        Returns:
            True if hypothesis is validated
        """
        if hypothesis_id not in self._hypotheses:
            return False

        h = self._hypotheses[hypothesis_id]
        h.tested_at = datetime.now()
        h.test_samples = sample_count

        improvement = (test_performance - baseline_performance) / max(baseline_performance, 0.001)
        h.result_improvement = improvement

        # Statistical significance check (simplified)
        min_improvement = 0.01  # 1% minimum improvement
        min_samples = 10

        if sample_count >= min_samples and improvement >= min_improvement:
            h.status = HypothesisStatus.VALIDATED
            self._successful_hypotheses.append(hypothesis_id)
            logger.info(
                "Hypothesis validated",
                hypothesis_id=hypothesis_id,
                improvement=improvement,
            )
            return True
        else:
            h.status = HypothesisStatus.REJECTED
            self._failed_hypotheses.append(hypothesis_id)
            logger.info(
                "Hypothesis rejected",
                hypothesis_id=hypothesis_id,
                improvement=improvement,
            )
            return False

    def get_hypothesis(self, hypothesis_id: str) -> Optional[Hypothesis]:
        """Get a hypothesis by ID."""
        return self._hypotheses.get(hypothesis_id)

    def get_pending(self) -> list[Hypothesis]:
        """Get all pending hypotheses."""
        return [
            h for h in self._hypotheses.values()
            if h.status == HypothesisStatus.PENDING
        ]

    def get_validated(self) -> list[Hypothesis]:
        """Get all validated hypotheses."""
        return [
            h for h in self._hypotheses.values()
            if h.status == HypothesisStatus.VALIDATED
        ]

    def get_stats(self) -> dict[str, Any]:
        """Get hypothesis generation statistics."""
        return {
            "total_generated": len(self._hypotheses),
            "pending": len([h for h in self._hypotheses.values() if h.status == HypothesisStatus.PENDING]),
            "validated": len(self._successful_hypotheses),
            "rejected": len(self._failed_hypotheses),
            "success_rate": (
                len(self._successful_hypotheses) / max(
                    len(self._successful_hypotheses) + len(self._failed_hypotheses), 1
                )
            ),
        }
