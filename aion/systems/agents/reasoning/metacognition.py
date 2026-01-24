"""
Metacognitive Monitoring System

Implements metacognition - thinking about thinking - for agents,
enabling self-awareness of cognitive processes, confidence calibration,
and adaptive strategy selection.
"""

import asyncio
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional, Callable

import structlog

logger = structlog.get_logger()


class CognitiveLoad(Enum):
    """Levels of cognitive load."""

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    OVERLOADED = "overloaded"


class TaskDifficulty(Enum):
    """Estimated task difficulty."""

    TRIVIAL = "trivial"
    EASY = "easy"
    MODERATE = "moderate"
    HARD = "hard"
    VERY_HARD = "very_hard"


class StrategyType(Enum):
    """Cognitive strategies."""

    DIRECT = "direct"  # Answer directly
    DECOMPOSE = "decompose"  # Break into sub-problems
    ANALOGIZE = "analogize"  # Use analogy
    SEARCH = "search"  # Search for information
    REASON = "reason"  # Step-by-step reasoning
    VERIFY = "verify"  # Verify and check
    REFLECT = "reflect"  # Self-reflect
    DELEGATE = "delegate"  # Delegate to specialist


@dataclass
class ConfidenceEstimate:
    """A calibrated confidence estimate."""

    value: float  # 0-1 confidence
    calibration_factor: float  # Adjustment based on past performance
    uncertainty_type: str  # "epistemic" or "aleatory"
    reasoning: str  # Why this confidence level
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def calibrated_value(self) -> float:
        """Get calibrated confidence."""
        return min(1.0, max(0.0, self.value * self.calibration_factor))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "value": self.value,
            "calibration_factor": self.calibration_factor,
            "calibrated_value": self.calibrated_value,
            "uncertainty_type": self.uncertainty_type,
            "reasoning": self.reasoning,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class CognitiveState:
    """Current cognitive state of the agent."""

    cognitive_load: CognitiveLoad = CognitiveLoad.LOW
    attention_focus: Optional[str] = None
    active_goals: list[str] = field(default_factory=list)
    working_memory_usage: float = 0.0  # 0-1
    processing_depth: int = 1  # 1-5, current depth of processing
    current_strategy: StrategyType = StrategyType.DIRECT
    confidence_level: float = 0.5
    uncertainty_sources: list[str] = field(default_factory=list)
    time_pressure: float = 0.0  # 0-1
    fatigue_level: float = 0.0  # 0-1, simulated cognitive fatigue

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "cognitive_load": self.cognitive_load.value,
            "attention_focus": self.attention_focus,
            "active_goals": self.active_goals,
            "working_memory_usage": self.working_memory_usage,
            "processing_depth": self.processing_depth,
            "current_strategy": self.current_strategy.value,
            "confidence_level": self.confidence_level,
            "uncertainty_sources": self.uncertainty_sources,
            "time_pressure": self.time_pressure,
            "fatigue_level": self.fatigue_level,
        }


@dataclass
class PerformanceRecord:
    """Record of past performance for calibration."""

    task_type: str
    predicted_confidence: float
    actual_success: bool
    difficulty: TaskDifficulty
    strategy_used: StrategyType
    time_taken: float  # seconds
    timestamp: datetime = field(default_factory=datetime.now)


class MetacognitiveMonitor:
    """
    Metacognitive monitoring system.

    Features:
    - Cognitive state tracking
    - Confidence calibration
    - Strategy selection
    - Performance monitoring
    - Adaptive processing depth
    - Uncertainty quantification
    """

    def __init__(
        self,
        calibration_window: int = 100,  # Number of past tasks to use for calibration
        fatigue_recovery_rate: float = 0.1,
        max_cognitive_load: float = 0.9,
    ):
        self.calibration_window = calibration_window
        self.fatigue_recovery_rate = fatigue_recovery_rate
        self.max_cognitive_load = max_cognitive_load

        # State
        self._cognitive_state = CognitiveState()
        self._performance_history: deque[PerformanceRecord] = deque(maxlen=calibration_window)

        # Calibration
        self._calibration_factors: dict[str, float] = {}  # task_type -> calibration
        self._strategy_success_rates: dict[StrategyType, float] = {
            s: 0.5 for s in StrategyType
        }

        # Monitoring
        self._task_start_time: Optional[datetime] = None
        self._introspection_count = 0

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize metacognitive monitor."""
        self._initialized = True
        logger.info("metacognitive_monitor_initialized")

    async def shutdown(self) -> None:
        """Shutdown metacognitive monitor."""
        self._initialized = False
        logger.info("metacognitive_monitor_shutdown")

    def get_state(self) -> CognitiveState:
        """Get current cognitive state."""
        return self._cognitive_state

    def update_state(
        self,
        attention_focus: Optional[str] = None,
        active_goals: Optional[list[str]] = None,
        working_memory_usage: Optional[float] = None,
        time_pressure: Optional[float] = None,
    ) -> CognitiveState:
        """Update cognitive state."""
        state = self._cognitive_state

        if attention_focus is not None:
            state.attention_focus = attention_focus

        if active_goals is not None:
            state.active_goals = active_goals

        if working_memory_usage is not None:
            state.working_memory_usage = working_memory_usage
            # Update cognitive load based on memory usage
            if working_memory_usage < 0.3:
                state.cognitive_load = CognitiveLoad.LOW
            elif working_memory_usage < 0.6:
                state.cognitive_load = CognitiveLoad.MODERATE
            elif working_memory_usage < 0.8:
                state.cognitive_load = CognitiveLoad.HIGH
            else:
                state.cognitive_load = CognitiveLoad.OVERLOADED

        if time_pressure is not None:
            state.time_pressure = time_pressure

        return state

    def estimate_confidence(
        self,
        task_type: str,
        raw_confidence: float,
        reasoning: str = "",
    ) -> ConfidenceEstimate:
        """
        Generate a calibrated confidence estimate.

        Uses past performance to calibrate raw confidence.
        """
        # Get calibration factor
        calibration = self._calibration_factors.get(task_type, 1.0)

        # Determine uncertainty type
        if "don't know" in reasoning.lower() or "uncertain" in reasoning.lower():
            uncertainty_type = "epistemic"  # Lack of knowledge
        else:
            uncertainty_type = "aleatory"  # Inherent randomness

        # Adjust for cognitive state
        state_factor = 1.0
        if self._cognitive_state.cognitive_load == CognitiveLoad.OVERLOADED:
            state_factor = 0.8
        if self._cognitive_state.fatigue_level > 0.5:
            state_factor *= 0.9

        estimate = ConfidenceEstimate(
            value=raw_confidence,
            calibration_factor=calibration * state_factor,
            uncertainty_type=uncertainty_type,
            reasoning=reasoning,
        )

        self._cognitive_state.confidence_level = estimate.calibrated_value

        return estimate

    def assess_difficulty(
        self,
        task_description: str,
        task_type: Optional[str] = None,
    ) -> tuple[TaskDifficulty, float]:
        """
        Assess the difficulty of a task.

        Returns:
            Tuple of (difficulty, confidence in assessment)
        """
        # Heuristic difficulty assessment
        description_lower = task_description.lower()

        # Check for difficulty indicators
        difficulty_score = 0.5

        # Length-based
        if len(task_description) > 500:
            difficulty_score += 0.1

        # Keyword-based
        hard_keywords = ["complex", "advanced", "difficult", "challenging", "multiple", "analyze"]
        easy_keywords = ["simple", "basic", "quick", "just", "only", "single"]

        for keyword in hard_keywords:
            if keyword in description_lower:
                difficulty_score += 0.1

        for keyword in easy_keywords:
            if keyword in description_lower:
                difficulty_score -= 0.1

        # Check past performance on similar tasks
        if task_type:
            recent_similar = [
                r for r in self._performance_history
                if r.task_type == task_type
            ]
            if recent_similar:
                success_rate = sum(1 for r in recent_similar if r.actual_success) / len(recent_similar)
                difficulty_score = 1.0 - success_rate  # Higher failure = harder

        # Map to difficulty level
        difficulty_score = max(0.0, min(1.0, difficulty_score))

        if difficulty_score < 0.2:
            difficulty = TaskDifficulty.TRIVIAL
        elif difficulty_score < 0.4:
            difficulty = TaskDifficulty.EASY
        elif difficulty_score < 0.6:
            difficulty = TaskDifficulty.MODERATE
        elif difficulty_score < 0.8:
            difficulty = TaskDifficulty.HARD
        else:
            difficulty = TaskDifficulty.VERY_HARD

        # Confidence in assessment
        confidence = 0.7 if len(self._performance_history) > 10 else 0.5

        return difficulty, confidence

    def select_strategy(
        self,
        task_description: str,
        difficulty: TaskDifficulty,
        available_strategies: Optional[list[StrategyType]] = None,
    ) -> StrategyType:
        """
        Select the best strategy for a task.

        Uses past performance and task characteristics.
        """
        available = available_strategies or list(StrategyType)

        # Strategy selection based on difficulty
        strategy_weights = {s: self._strategy_success_rates.get(s, 0.5) for s in available}

        # Adjust weights based on difficulty
        if difficulty in (TaskDifficulty.TRIVIAL, TaskDifficulty.EASY):
            strategy_weights[StrategyType.DIRECT] = strategy_weights.get(StrategyType.DIRECT, 0.5) + 0.3

        if difficulty in (TaskDifficulty.HARD, TaskDifficulty.VERY_HARD):
            strategy_weights[StrategyType.DECOMPOSE] = strategy_weights.get(StrategyType.DECOMPOSE, 0.5) + 0.3
            strategy_weights[StrategyType.REASON] = strategy_weights.get(StrategyType.REASON, 0.5) + 0.2

        # Adjust for cognitive load
        if self._cognitive_state.cognitive_load == CognitiveLoad.OVERLOADED:
            strategy_weights[StrategyType.DELEGATE] = strategy_weights.get(StrategyType.DELEGATE, 0.5) + 0.4
            strategy_weights[StrategyType.DECOMPOSE] = strategy_weights.get(StrategyType.DECOMPOSE, 0.5) + 0.2

        # Adjust for time pressure
        if self._cognitive_state.time_pressure > 0.7:
            strategy_weights[StrategyType.DIRECT] = strategy_weights.get(StrategyType.DIRECT, 0.5) + 0.3
            strategy_weights[StrategyType.REASON] = strategy_weights.get(StrategyType.REASON, 0.5) - 0.2

        # Select best strategy
        best_strategy = max(available, key=lambda s: strategy_weights.get(s, 0))

        self._cognitive_state.current_strategy = best_strategy

        return best_strategy

    def determine_processing_depth(
        self,
        difficulty: TaskDifficulty,
        importance: float = 0.5,
    ) -> int:
        """
        Determine how deep to process based on difficulty and importance.

        Returns:
            Processing depth 1-5
        """
        # Base depth from difficulty
        difficulty_depths = {
            TaskDifficulty.TRIVIAL: 1,
            TaskDifficulty.EASY: 2,
            TaskDifficulty.MODERATE: 3,
            TaskDifficulty.HARD: 4,
            TaskDifficulty.VERY_HARD: 5,
        }

        base_depth = difficulty_depths.get(difficulty, 3)

        # Adjust for importance
        if importance > 0.7:
            base_depth = min(5, base_depth + 1)
        if importance < 0.3:
            base_depth = max(1, base_depth - 1)

        # Adjust for cognitive load
        if self._cognitive_state.cognitive_load == CognitiveLoad.OVERLOADED:
            base_depth = max(1, base_depth - 1)

        # Adjust for time pressure
        if self._cognitive_state.time_pressure > 0.8:
            base_depth = max(1, base_depth - 1)

        self._cognitive_state.processing_depth = base_depth

        return base_depth

    def start_task(self, task_description: str) -> None:
        """Record start of a task for timing."""
        self._task_start_time = datetime.now()
        self._cognitive_state.attention_focus = task_description[:100]

        # Increase fatigue slightly
        self._cognitive_state.fatigue_level = min(
            1.0, self._cognitive_state.fatigue_level + 0.05
        )

    def end_task(
        self,
        task_type: str,
        predicted_confidence: float,
        actual_success: bool,
        strategy_used: StrategyType,
        difficulty: TaskDifficulty,
    ) -> None:
        """
        Record end of task and update calibration.
        """
        time_taken = 0.0
        if self._task_start_time:
            time_taken = (datetime.now() - self._task_start_time).total_seconds()

        record = PerformanceRecord(
            task_type=task_type,
            predicted_confidence=predicted_confidence,
            actual_success=actual_success,
            difficulty=difficulty,
            strategy_used=strategy_used,
            time_taken=time_taken,
        )

        self._performance_history.append(record)

        # Update calibration
        self._update_calibration(task_type)

        # Update strategy success rate
        self._update_strategy_success(strategy_used, actual_success)

        # Reset task state
        self._task_start_time = None
        self._cognitive_state.attention_focus = None

    def _update_calibration(self, task_type: str) -> None:
        """Update confidence calibration for a task type."""
        relevant_records = [
            r for r in self._performance_history
            if r.task_type == task_type
        ]

        if len(relevant_records) < 3:
            return

        # Compare predicted confidence to actual success rate
        avg_confidence = sum(r.predicted_confidence for r in relevant_records) / len(relevant_records)
        actual_success_rate = sum(1 for r in relevant_records if r.actual_success) / len(relevant_records)

        # Calibration factor adjusts confidence to match reality
        if avg_confidence > 0:
            self._calibration_factors[task_type] = actual_success_rate / avg_confidence
        else:
            self._calibration_factors[task_type] = 1.0

    def _update_strategy_success(self, strategy: StrategyType, success: bool) -> None:
        """Update strategy success rate."""
        current_rate = self._strategy_success_rates.get(strategy, 0.5)

        # Exponential moving average
        alpha = 0.1
        new_rate = alpha * (1.0 if success else 0.0) + (1 - alpha) * current_rate

        self._strategy_success_rates[strategy] = new_rate

    def recover_fatigue(self, amount: Optional[float] = None) -> None:
        """Recover from cognitive fatigue."""
        recovery = amount if amount is not None else self.fatigue_recovery_rate
        self._cognitive_state.fatigue_level = max(
            0.0, self._cognitive_state.fatigue_level - recovery
        )

    def introspect(self) -> dict[str, Any]:
        """
        Perform introspection - thinking about current cognitive state.

        Returns comprehensive self-assessment.
        """
        self._introspection_count += 1

        state = self._cognitive_state

        # Assess current effectiveness
        recent_records = list(self._performance_history)[-10:]
        if recent_records:
            recent_success_rate = sum(1 for r in recent_records if r.actual_success) / len(recent_records)
        else:
            recent_success_rate = 0.5

        # Identify issues
        issues = []
        if state.cognitive_load == CognitiveLoad.OVERLOADED:
            issues.append("Cognitive overload - consider delegating or simplifying")
        if state.fatigue_level > 0.7:
            issues.append("High fatigue - quality may be degraded")
        if state.time_pressure > 0.8:
            issues.append("High time pressure - may need to sacrifice depth for speed")
        if state.working_memory_usage > 0.8:
            issues.append("Working memory near capacity - chunk or offload information")

        # Recommendations
        recommendations = []
        if state.cognitive_load in (CognitiveLoad.HIGH, CognitiveLoad.OVERLOADED):
            recommendations.append("Break task into smaller pieces")
            recommendations.append("Consider using external memory aids")
        if recent_success_rate < 0.5:
            recommendations.append("Review recent failures for patterns")
            recommendations.append("Consider more deliberate reasoning strategies")

        return {
            "cognitive_state": state.to_dict(),
            "recent_success_rate": recent_success_rate,
            "issues": issues,
            "recommendations": recommendations,
            "calibration_factors": dict(self._calibration_factors),
            "strategy_success_rates": {s.value: r for s, r in self._strategy_success_rates.items()},
            "introspection_count": self._introspection_count,
        }

    def should_stop_and_reflect(self) -> tuple[bool, str]:
        """
        Determine if the agent should stop and reflect.

        Returns:
            Tuple of (should_reflect, reason)
        """
        state = self._cognitive_state

        if state.cognitive_load == CognitiveLoad.OVERLOADED:
            return True, "Cognitive overload detected"

        if state.fatigue_level > 0.8:
            return True, "High fatigue level"

        if state.confidence_level < 0.3:
            return True, "Low confidence - need to reassess approach"

        # Check recent performance
        recent = list(self._performance_history)[-5:]
        if len(recent) >= 5:
            if sum(1 for r in recent if r.actual_success) == 0:
                return True, "Recent failures suggest strategy change needed"

        return False, ""

    def get_stats(self) -> dict[str, Any]:
        """Get metacognitive statistics."""
        return {
            "cognitive_state": self._cognitive_state.to_dict(),
            "performance_history_size": len(self._performance_history),
            "introspection_count": self._introspection_count,
            "calibration_factors": dict(self._calibration_factors),
            "strategy_success_rates": {s.value: r for s, r in self._strategy_success_rates.items()},
        }
