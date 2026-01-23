"""
AION Goal Prioritizer

Intelligent goal prioritization:
- Multi-factor priority scoring
- Dynamic priority adjustment
- Resource-aware prioritization
- Deadline handling
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Optional

import structlog

from aion.systems.goals.types import (
    Goal,
    GoalStatus,
    GoalPriority,
    GoalType,
    ValuePrinciple,
)

logger = structlog.get_logger(__name__)


class GoalPrioritizer:
    """
    Prioritizes goals based on multiple factors.

    Features:
    - Multi-factor scoring
    - Value alignment weighting
    - Deadline urgency
    - Resource efficiency
    - Dynamic reprioritization
    """

    def __init__(
        self,
        values: Optional[list[ValuePrinciple]] = None,
        deadline_weight: float = 1.5,
        value_weight: float = 1.2,
        effort_weight: float = 0.8,
        age_weight: float = 0.3,
    ):
        self._values = values or []
        self._deadline_weight = deadline_weight
        self._value_weight = value_weight
        self._effort_weight = effort_weight
        self._age_weight = age_weight

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the prioritizer."""
        if self._initialized:
            return

        logger.info("Initializing Goal Prioritizer")
        self._initialized = True

    async def shutdown(self) -> None:
        """Shutdown the prioritizer."""
        self._initialized = False

    def calculate_priority_score(self, goal: Goal) -> float:
        """
        Calculate a comprehensive priority score for a goal.

        Higher score = higher priority.

        Returns:
            Priority score (0-100+)
        """
        score = 0.0

        # Base priority score (CRITICAL=50, HIGH=40, MEDIUM=30, LOW=20, BACKGROUND=10)
        base_scores = {
            GoalPriority.CRITICAL: 50,
            GoalPriority.HIGH: 40,
            GoalPriority.MEDIUM: 30,
            GoalPriority.LOW: 20,
            GoalPriority.BACKGROUND: 10,
        }
        score += base_scores.get(goal.priority, 30)

        # Deadline urgency
        score += self._calculate_deadline_score(goal) * self._deadline_weight

        # Expected value
        score += goal.expected_value * 20 * self._value_weight

        # Value alignment
        score += self._calculate_value_alignment_score(goal) * 10

        # Age factor (older goals get slight boost)
        age_days = (datetime.now() - goal.created_at).days
        score += min(age_days * 0.5, 5) * self._age_weight

        # Effort efficiency (higher value per effort = higher priority)
        effort = self._estimate_effort(goal)
        if effort > 0:
            efficiency = goal.expected_value / effort
            score += efficiency * 10 * self._effort_weight

        # Progress momentum (goals with progress get slight boost)
        if goal.metrics.progress_percent > 0:
            score += min(goal.metrics.progress_percent * 0.1, 5)

        # Dependency satisfaction (ready goals get boost)
        if self._all_dependencies_met(goal):
            score += 5

        return score

    def _calculate_deadline_score(self, goal: Goal) -> float:
        """Calculate urgency score based on deadline."""
        if not goal.deadline:
            return 0.0

        time_remaining = goal.time_until_deadline()
        if not time_remaining:
            return 0.0

        hours_remaining = time_remaining.total_seconds() / 3600

        if hours_remaining < 0:
            # Overdue - highest urgency
            return 30
        elif hours_remaining < 1:
            return 25
        elif hours_remaining < 24:
            return 20
        elif hours_remaining < 72:
            return 15
        elif hours_remaining < 168:  # 1 week
            return 10
        elif hours_remaining < 720:  # 1 month
            return 5
        else:
            return 0

    def _calculate_value_alignment_score(self, goal: Goal) -> float:
        """Calculate how well goal aligns with values."""
        if not self._values:
            return 0.5  # Neutral if no values defined

        # Check goal tags and context for value alignment
        alignment_score = 0.0
        matched_values = 0

        for value in self._values:
            if not value.active:
                continue

            # Simple matching based on value name in goal context
            if value.name.lower() in goal.description.lower():
                alignment_score += value.prioritization_weight
                matched_values += 1

            # Check tags
            for tag in goal.tags:
                if value.name.lower() in tag.lower():
                    alignment_score += value.prioritization_weight * 0.5
                    matched_values += 1
                    break

        if matched_values > 0:
            return alignment_score / matched_values

        return 0.5  # Neutral if no matches

    def _estimate_effort(self, goal: Goal) -> float:
        """Estimate effort for a goal (0-1 scale)."""
        # Use existing metrics if available
        if goal.metrics.time_spent_seconds > 0 and goal.metrics.progress_percent > 0:
            # Extrapolate from current progress
            total_estimated = (
                goal.metrics.time_spent_seconds * 100 / goal.metrics.progress_percent
            )
            return min(total_estimated / 3600 / 8, 1.0)  # Normalize to 8-hour day

        # Estimate based on goal type and complexity
        base_effort = {
            GoalType.ACHIEVEMENT: 0.5,
            GoalType.MAINTENANCE: 0.3,
            GoalType.IMPROVEMENT: 0.6,
            GoalType.LEARNING: 0.4,
            GoalType.CREATION: 0.7,
            GoalType.EXPLORATION: 0.4,
            GoalType.OPTIMIZATION: 0.5,
        }

        effort = base_effort.get(goal.goal_type, 0.5)

        # Adjust for success criteria count
        criteria_count = len(goal.success_criteria)
        if criteria_count > 5:
            effort += 0.2
        elif criteria_count > 3:
            effort += 0.1

        return min(effort, 1.0)

    def _all_dependencies_met(self, goal: Goal) -> bool:
        """Check if all goal dependencies are met."""
        for constraint in goal.constraints:
            if constraint.depends_on_goals:
                # Would need registry access for full check
                return False
        return True

    async def prioritize_goals(
        self,
        goals: list[Goal],
        limit: Optional[int] = None,
    ) -> list[Goal]:
        """
        Sort goals by priority.

        Args:
            goals: Goals to prioritize
            limit: Maximum number to return

        Returns:
            Goals sorted by priority (highest first)
        """
        # Calculate scores
        scored_goals = []
        for goal in goals:
            score = self.calculate_priority_score(goal)
            scored_goals.append((score, goal))

        # Sort by score (descending)
        scored_goals.sort(key=lambda x: x[0], reverse=True)

        result = [g for _, g in scored_goals]

        if limit:
            result = result[:limit]

        return result

    async def suggest_priority_adjustment(
        self,
        goal: Goal,
        context: dict[str, Any],
    ) -> Optional[GoalPriority]:
        """
        Suggest a priority adjustment for a goal.

        Returns:
            Suggested new priority, or None if no change needed
        """
        current_score = self.calculate_priority_score(goal)

        # Determine what priority the score suggests
        if current_score >= 60:
            suggested = GoalPriority.CRITICAL
        elif current_score >= 45:
            suggested = GoalPriority.HIGH
        elif current_score >= 30:
            suggested = GoalPriority.MEDIUM
        elif current_score >= 15:
            suggested = GoalPriority.LOW
        else:
            suggested = GoalPriority.BACKGROUND

        # Only suggest if different from current
        if suggested != goal.priority:
            # Check if the difference is significant
            priority_diff = abs(suggested.value - goal.priority.value)
            if priority_diff >= 2:  # Significant difference
                return suggested

        return None

    async def rebalance_priorities(
        self,
        goals: list[Goal],
        max_critical: int = 2,
        max_high: int = 5,
    ) -> list[tuple[Goal, GoalPriority]]:
        """
        Rebalance priorities to avoid too many high-priority goals.

        Returns:
            List of (goal, new_priority) tuples for goals that should change
        """
        changes: list[tuple[Goal, GoalPriority]] = []

        # Count current priorities
        critical_count = sum(1 for g in goals if g.priority == GoalPriority.CRITICAL)
        high_count = sum(1 for g in goals if g.priority == GoalPriority.HIGH)

        # If within limits, no rebalancing needed
        if critical_count <= max_critical and high_count <= max_high:
            return changes

        # Sort by calculated score
        scored = [(self.calculate_priority_score(g), g) for g in goals]
        scored.sort(key=lambda x: x[0], reverse=True)

        # Reassign priorities based on ranking
        new_critical = 0
        new_high = 0

        for score, goal in scored:
            if new_critical < max_critical and score >= 55:
                if goal.priority != GoalPriority.CRITICAL:
                    changes.append((goal, GoalPriority.CRITICAL))
                new_critical += 1
            elif new_high < max_high and score >= 40:
                if goal.priority != GoalPriority.HIGH:
                    changes.append((goal, GoalPriority.HIGH))
                new_high += 1
            elif score >= 25:
                if goal.priority not in (GoalPriority.MEDIUM, GoalPriority.LOW):
                    changes.append((goal, GoalPriority.MEDIUM))
            else:
                if goal.priority not in (GoalPriority.LOW, GoalPriority.BACKGROUND):
                    changes.append((goal, GoalPriority.LOW))

        return changes

    def set_values(self, values: list[ValuePrinciple]) -> None:
        """Update value principles."""
        self._values = values

    def set_weights(
        self,
        deadline: Optional[float] = None,
        value: Optional[float] = None,
        effort: Optional[float] = None,
        age: Optional[float] = None,
    ) -> None:
        """Update scoring weights."""
        if deadline is not None:
            self._deadline_weight = deadline
        if value is not None:
            self._value_weight = value
        if effort is not None:
            self._effort_weight = effort
        if age is not None:
            self._age_weight = age

    def get_priority_breakdown(self, goal: Goal) -> dict[str, float]:
        """Get detailed breakdown of priority score components."""
        return {
            "base_priority": {
                GoalPriority.CRITICAL: 50,
                GoalPriority.HIGH: 40,
                GoalPriority.MEDIUM: 30,
                GoalPriority.LOW: 20,
                GoalPriority.BACKGROUND: 10,
            }.get(goal.priority, 30),
            "deadline_urgency": self._calculate_deadline_score(goal)
            * self._deadline_weight,
            "expected_value": goal.expected_value * 20 * self._value_weight,
            "value_alignment": self._calculate_value_alignment_score(goal) * 10,
            "age_factor": min((datetime.now() - goal.created_at).days * 0.5, 5)
            * self._age_weight,
            "effort_efficiency": self._calculate_efficiency_score(goal),
            "progress_momentum": min(goal.metrics.progress_percent * 0.1, 5),
            "dependency_bonus": 5 if self._all_dependencies_met(goal) else 0,
            "total": self.calculate_priority_score(goal),
        }

    def _calculate_efficiency_score(self, goal: Goal) -> float:
        """Calculate effort efficiency score."""
        effort = self._estimate_effort(goal)
        if effort > 0:
            efficiency = goal.expected_value / effort
            return efficiency * 10 * self._effort_weight
        return 0.0
