"""
AION Goal Decomposer

Breaks down complex goals into manageable subgoals:
- Hierarchical decomposition
- Dependency analysis
- Effort estimation
- Parallelization identification
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

import structlog

from aion.systems.goals.types import (
    Goal,
    GoalType,
    GoalSource,
    GoalConstraint,
)

logger = structlog.get_logger(__name__)


class GoalDecomposer:
    """
    Decomposes complex goals into subgoals.

    Features:
    - Automatic complexity detection
    - LLM-powered decomposition
    - Dependency graph generation
    - Parallel execution identification
    """

    def __init__(
        self,
        reasoner: Optional[Any] = None,  # GoalReasoner
        max_depth: int = 3,
        max_subgoals_per_level: int = 5,
    ):
        self._reasoner = reasoner
        self._max_depth = max_depth
        self._max_subgoals_per_level = max_subgoals_per_level

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the decomposer."""
        if self._initialized:
            return

        logger.info("Initializing Goal Decomposer")
        self._initialized = True

    async def shutdown(self) -> None:
        """Shutdown the decomposer."""
        self._initialized = False

    async def should_decompose(self, goal: Goal) -> tuple[bool, str]:
        """
        Determine if a goal should be decomposed.

        Returns:
            Tuple of (should_decompose, reason)
        """
        # Already has subgoals
        if goal.subgoal_ids:
            return False, "Goal already has subgoals"

        # Simple goals don't need decomposition
        if goal.goal_type == GoalType.MAINTENANCE:
            return False, "Maintenance goals typically don't need decomposition"

        # Check complexity indicators
        complexity_score = self._estimate_complexity(goal)

        if complexity_score > 0.7:
            return True, f"High complexity score: {complexity_score:.2f}"

        if complexity_score > 0.5 and len(goal.success_criteria) > 3:
            return True, "Multiple success criteria suggest decomposition"

        if complexity_score > 0.4:
            # Use LLM to decide
            if self._reasoner:
                recommendation = await self._get_decomposition_recommendation(goal)
                return recommendation.get("should_decompose", False), recommendation.get(
                    "reason", "LLM recommendation"
                )

        return False, "Goal appears simple enough"

    def _estimate_complexity(self, goal: Goal) -> float:
        """Estimate goal complexity (0-1)."""
        score = 0.0

        # Description length
        if len(goal.description) > 500:
            score += 0.2
        elif len(goal.description) > 200:
            score += 0.1

        # Number of success criteria
        criteria_count = len(goal.success_criteria)
        if criteria_count > 5:
            score += 0.3
        elif criteria_count > 3:
            score += 0.2
        elif criteria_count > 1:
            score += 0.1

        # Goal type
        if goal.goal_type in (GoalType.CREATION, GoalType.IMPROVEMENT):
            score += 0.15

        # Constraints
        if len(goal.constraints) > 2:
            score += 0.1

        # Expected value vs effort ratio
        # Higher effort suggests more complexity
        if goal.expected_value < 0.5:
            score += 0.1

        return min(1.0, score)

    async def _get_decomposition_recommendation(
        self, goal: Goal
    ) -> dict[str, Any]:
        """Get LLM recommendation on decomposition."""
        if not self._reasoner or not self._reasoner.llm:
            return {"should_decompose": False, "reason": "LLM not available"}

        try:
            system_prompt = """You are analyzing whether a goal needs to be broken down into subgoals.
Consider: complexity, number of distinct steps, dependencies, and whether parts can be done in parallel."""

            user_prompt = f"""
Should this goal be decomposed into subgoals?

Goal: {goal.title}
Description: {goal.description}
Success Criteria: {goal.success_criteria}
Type: {goal.goal_type.value}

Respond in JSON format:
{{
    "should_decompose": true/false,
    "reason": "...",
    "estimated_subgoals": 0,
    "complexity_assessment": "low/medium/high"
}}
"""

            response = await self._reasoner.llm.complete(
                messages=[{"role": "user", "content": user_prompt}],
                system=system_prompt,
            )

            return self._reasoner._extract_json(
                self._reasoner._get_text_from_response(response)
            )

        except Exception as e:
            logger.error(f"Error getting decomposition recommendation: {e}")
            return {"should_decompose": False, "reason": str(e)}

    async def decompose(
        self,
        goal: Goal,
        depth: int = 0,
    ) -> list[Goal]:
        """
        Decompose a goal into subgoals.

        Args:
            goal: The goal to decompose
            depth: Current decomposition depth

        Returns:
            List of subgoals
        """
        if depth >= self._max_depth:
            logger.warning(
                f"Max decomposition depth reached for goal", goal_id=goal.id[:8]
            )
            return []

        # Use reasoner for LLM-powered decomposition
        if self._reasoner:
            subgoals = await self._reasoner.decompose_goal(
                goal, max_subgoals=self._max_subgoals_per_level
            )

            if subgoals:
                logger.info(
                    f"Decomposed goal into {len(subgoals)} subgoals",
                    goal_id=goal.id[:8],
                )
                return subgoals

        # Fallback: rule-based decomposition
        return self._rule_based_decompose(goal)

    def _rule_based_decompose(self, goal: Goal) -> list[Goal]:
        """Rule-based decomposition when LLM is not available."""
        subgoals = []

        # Create a subgoal for each success criterion
        for i, criterion in enumerate(goal.success_criteria):
            subgoal = Goal(
                title=f"Achieve: {criterion[:50]}",
                description=f"Work toward criterion: {criterion}",
                success_criteria=[criterion],
                goal_type=goal.goal_type,
                source=GoalSource.DERIVED,
                priority=goal.priority,
                parent_goal_id=goal.id,
                context={"criterion_index": i},
            )
            subgoals.append(subgoal)

        return subgoals[:self._max_subgoals_per_level]

    async def decompose_recursive(
        self,
        goal: Goal,
        min_complexity: float = 0.5,
    ) -> list[Goal]:
        """
        Recursively decompose a goal until all subgoals are simple.

        Args:
            goal: The goal to decompose
            min_complexity: Stop decomposing below this complexity

        Returns:
            Flat list of all leaf subgoals
        """
        all_subgoals = []

        async def _decompose_recursive(g: Goal, depth: int) -> None:
            if depth >= self._max_depth:
                all_subgoals.append(g)
                return

            complexity = self._estimate_complexity(g)
            if complexity < min_complexity:
                all_subgoals.append(g)
                return

            subgoals = await self.decompose(g, depth=depth)
            if not subgoals:
                all_subgoals.append(g)
                return

            for sg in subgoals:
                await _decompose_recursive(sg, depth + 1)

        await _decompose_recursive(goal, 0)

        return all_subgoals

    def identify_parallel_subgoals(self, subgoals: list[Goal]) -> list[list[Goal]]:
        """
        Identify which subgoals can be executed in parallel.

        Returns:
            List of parallel execution groups
        """
        # Build dependency graph
        dep_graph: dict[str, set[str]] = {}
        for sg in subgoals:
            deps = set()
            for constraint in sg.constraints:
                deps.update(constraint.depends_on_goals)
            dep_graph[sg.id] = deps

        # Find independent groups using topological sorting
        groups: list[list[Goal]] = []
        remaining = {sg.id: sg for sg in subgoals}
        completed: set[str] = set()

        while remaining:
            # Find goals with no unmet dependencies
            ready = []
            for goal_id, goal in remaining.items():
                deps = dep_graph.get(goal_id, set())
                if deps.issubset(completed):
                    ready.append(goal)

            if not ready:
                # Circular dependency or error - add all remaining
                groups.append(list(remaining.values()))
                break

            groups.append(ready)

            # Mark as completed
            for goal in ready:
                completed.add(goal.id)
                del remaining[goal.id]

        return groups

    async def estimate_total_effort(self, goal: Goal) -> dict[str, Any]:
        """
        Estimate total effort for a goal including decomposition.

        Returns:
            Effort estimation with breakdown
        """
        estimation = {
            "goal_id": goal.id,
            "complexity": self._estimate_complexity(goal),
            "estimated_hours": 0.0,
            "subgoal_breakdown": [],
        }

        should_decompose, _ = await self.should_decompose(goal)

        if should_decompose:
            subgoals = await self.decompose(goal)
            for sg in subgoals:
                sg_complexity = self._estimate_complexity(sg)
                sg_hours = self._complexity_to_hours(sg_complexity)
                estimation["subgoal_breakdown"].append(
                    {
                        "title": sg.title,
                        "complexity": sg_complexity,
                        "estimated_hours": sg_hours,
                    }
                )
                estimation["estimated_hours"] += sg_hours
        else:
            estimation["estimated_hours"] = self._complexity_to_hours(
                estimation["complexity"]
            )

        return estimation

    def _complexity_to_hours(self, complexity: float) -> float:
        """Convert complexity score to estimated hours."""
        # Simple mapping
        if complexity < 0.2:
            return 0.5
        elif complexity < 0.4:
            return 1.0
        elif complexity < 0.6:
            return 2.0
        elif complexity < 0.8:
            return 4.0
        else:
            return 8.0
