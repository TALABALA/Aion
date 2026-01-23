"""
AION Goal Reasoner

LLM-powered cognitive engine for goal-related reasoning:
- Goal generation from context
- Goal evaluation and refinement
- Decomposition strategies
- Conflict resolution
- Progress assessment
"""

from __future__ import annotations

import json
from typing import Any, Optional

import structlog

from aion.systems.goals.types import (
    Goal,
    GoalProposal,
    GoalType,
    GoalSource,
    GoalPriority,
    GoalConstraint,
    Objective,
    ValuePrinciple,
)

logger = structlog.get_logger(__name__)


class GoalReasoner:
    """
    LLM-powered goal reasoning engine.

    Uses Claude to:
    - Generate goals from context and values
    - Evaluate goal feasibility and value
    - Decompose complex goals into subgoals
    - Resolve conflicts between goals
    - Assess progress and adjust strategies
    """

    def __init__(
        self,
        llm_provider: Optional[Any] = None,
        values: Optional[list[ValuePrinciple]] = None,
        default_model: str = "claude-sonnet-4-20250514",
    ):
        self._llm = llm_provider
        self.values = values or self._get_default_values()
        self.default_model = default_model

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the reasoner."""
        if self._initialized:
            return

        # Initialize LLM provider if not provided
        if self._llm is None:
            try:
                from aion.conversation.llm.claude import ClaudeProvider

                self._llm = ClaudeProvider(default_model=self.default_model)
                await self._llm.initialize()
            except Exception as e:
                logger.warning(f"Could not initialize LLM provider: {e}")

        self._initialized = True
        logger.info("Goal Reasoner initialized")

    async def shutdown(self) -> None:
        """Shutdown the reasoner."""
        if self._llm:
            await self._llm.shutdown()
        self._initialized = False

    @property
    def llm(self) -> Any:
        """Get the LLM provider."""
        return self._llm

    async def generate_goals(
        self,
        context: dict[str, Any],
        objectives: list[Objective],
        existing_goals: list[Goal],
        max_goals: int = 3,
    ) -> list[GoalProposal]:
        """
        Generate new goal proposals based on context.

        Args:
            context: Current system state, user preferences, etc.
            objectives: Active high-level objectives
            existing_goals: Currently active goals
            max_goals: Maximum number of goals to propose

        Returns:
            List of goal proposals
        """
        if not self._llm:
            logger.warning("LLM not available for goal generation")
            return []

        # Build prompt
        system_prompt = self._build_goal_generation_prompt()

        user_prompt = f"""
Based on the current context and objectives, propose up to {max_goals} new goals for AION to pursue.

## Current Context
{json.dumps(context, indent=2, default=str)}

## Active Objectives
{self._format_objectives(objectives)}

## Existing Goals (avoid duplicates)
{self._format_goals(existing_goals)}

## Core Values to Consider
{self._format_values()}

For each proposed goal, provide:
1. Title (concise, action-oriented)
2. Description (detailed explanation)
3. Success criteria (measurable outcomes)
4. Goal type (achievement, maintenance, improvement, learning, creation, exploration, optimization)
5. Priority (critical, high, medium, low, background)
6. Reasoning (why this goal is valuable)
7. Expected effort (low, medium, high)
8. Dependencies (any prerequisites)

Respond in JSON format:
{{
    "proposals": [
        {{
            "title": "...",
            "description": "...",
            "success_criteria": ["...", "..."],
            "goal_type": "achievement",
            "priority": "medium",
            "reasoning": "...",
            "expected_effort": "medium",
            "dependencies": [],
            "expected_value": 0.8,
            "confidence": 0.7
        }}
    ]
}}
"""

        try:
            response = await self._llm.complete(
                messages=[{"role": "user", "content": user_prompt}],
                system=system_prompt,
            )

            # Parse response
            response_text = self._get_text_from_response(response)
            data = self._extract_json(response_text)

            proposals = []
            for p in data.get("proposals", []):
                goal = Goal(
                    title=p.get("title", ""),
                    description=p.get("description", ""),
                    success_criteria=p.get("success_criteria", []),
                    goal_type=GoalType(p.get("goal_type", "achievement")),
                    source=GoalSource.SYSTEM,
                    priority=self._parse_priority(p.get("priority", "medium")),
                    expected_value=p.get("expected_value", 0.5),
                )

                proposals.append(
                    GoalProposal(
                        goal=goal,
                        reasoning=p.get("reasoning", ""),
                        expected_value=p.get("expected_value", 0.5),
                        expected_effort=self._parse_effort(
                            p.get("expected_effort", "medium")
                        ),
                        confidence=p.get("confidence", 0.5),
                    )
                )

            logger.info(f"Generated {len(proposals)} goal proposals")
            return proposals

        except Exception as e:
            logger.error(f"Failed to generate goals: {e}")
            return []

    async def evaluate_goal(
        self,
        goal: Goal,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Evaluate a goal's feasibility and value.

        Returns:
            Dict with feasibility, value, risks, recommendations
        """
        if not self._llm:
            return {
                "feasibility_score": 0.5,
                "value_score": 0.5,
                "should_pursue": True,
                "reasoning": "LLM not available for evaluation",
            }

        system_prompt = """You are evaluating whether a goal is worth pursuing.
Consider: feasibility, value, risks, resource requirements, and alignment with values.
Be realistic and thorough in your assessment."""

        user_prompt = f"""
Evaluate this goal:

## Goal
Title: {goal.title}
Description: {goal.description}
Type: {goal.goal_type.value}
Priority: {goal.priority.name}
Success Criteria: {goal.success_criteria}

## Context
{json.dumps(context, indent=2, default=str)}

Provide your evaluation in JSON format:
{{
    "feasibility_score": 0.0-1.0,
    "value_score": 0.0-1.0,
    "alignment_score": 0.0-1.0,
    "risks": ["risk1", "risk2"],
    "requirements": ["req1", "req2"],
    "estimated_duration_hours": 0,
    "recommended_approach": "...",
    "should_pursue": true/false,
    "reasoning": "..."
}}
"""

        try:
            response = await self._llm.complete(
                messages=[{"role": "user", "content": user_prompt}],
                system=system_prompt,
            )

            response_text = self._get_text_from_response(response)
            return self._extract_json(response_text)

        except Exception as e:
            logger.error(f"Failed to evaluate goal: {e}")
            return {
                "feasibility_score": 0.5,
                "value_score": 0.5,
                "should_pursue": True,
                "reasoning": f"Could not evaluate: {e}",
            }

    async def decompose_goal(
        self,
        goal: Goal,
        max_subgoals: int = 5,
    ) -> list[Goal]:
        """
        Decompose a complex goal into smaller subgoals.

        Args:
            goal: The goal to decompose
            max_subgoals: Maximum number of subgoals

        Returns:
            List of subgoals
        """
        if not self._llm:
            logger.warning("LLM not available for goal decomposition")
            return []

        system_prompt = """You are breaking down a complex goal into smaller, actionable subgoals.
Each subgoal should be:
- Specific and measurable
- Independently achievable
- Contributing to the parent goal
- Properly sequenced (dependencies clear)"""

        user_prompt = f"""
Break down this goal into {max_subgoals} or fewer subgoals:

## Goal
Title: {goal.title}
Description: {goal.description}
Success Criteria: {goal.success_criteria}

Provide subgoals in JSON format:
{{
    "subgoals": [
        {{
            "title": "...",
            "description": "...",
            "success_criteria": ["..."],
            "sequence_order": 1,
            "depends_on": [],
            "estimated_effort": "low/medium/high"
        }}
    ],
    "decomposition_reasoning": "..."
}}
"""

        try:
            response = await self._llm.complete(
                messages=[{"role": "user", "content": user_prompt}],
                system=system_prompt,
            )

            response_text = self._get_text_from_response(response)
            data = self._extract_json(response_text)

            subgoals = []
            subgoal_ids = []

            for i, sg in enumerate(data.get("subgoals", [])):
                subgoal = Goal(
                    title=sg.get("title", f"Subgoal {i + 1}"),
                    description=sg.get("description", ""),
                    success_criteria=sg.get("success_criteria", []),
                    goal_type=goal.goal_type,
                    source=GoalSource.DERIVED,
                    priority=goal.priority,
                    parent_goal_id=goal.id,
                    context={"sequence_order": sg.get("sequence_order", i + 1)},
                )
                subgoals.append(subgoal)
                subgoal_ids.append(subgoal.id)

            # Set up dependencies between subgoals
            for i, sg in enumerate(data.get("subgoals", [])):
                depends_on = sg.get("depends_on", [])
                if depends_on and i < len(subgoals):
                    for dep_idx in depends_on:
                        if dep_idx < len(subgoal_ids):
                            subgoals[i].constraints.append(
                                GoalConstraint(
                                    type="dependency",
                                    description=f"Depends on subgoal {dep_idx + 1}",
                                    depends_on_goals=[subgoal_ids[dep_idx]],
                                )
                            )

            logger.info(
                f"Decomposed goal into {len(subgoals)} subgoals", goal_id=goal.id[:8]
            )
            return subgoals

        except Exception as e:
            logger.error(f"Failed to decompose goal: {e}")
            return []

    async def resolve_conflict(
        self,
        goals: list[Goal],
        conflict_type: str,
    ) -> dict[str, Any]:
        """
        Resolve conflicts between goals.

        Args:
            goals: Conflicting goals
            conflict_type: Type of conflict (resource, priority, dependency, etc.)

        Returns:
            Resolution recommendation
        """
        if not self._llm:
            return {
                "resolution_type": "prioritize",
                "reasoning": "LLM not available for conflict resolution",
            }

        system_prompt = """You are resolving conflicts between goals.
Consider: priority, dependencies, resource constraints, and overall value.
Provide a clear recommendation for how to proceed."""

        user_prompt = f"""
Resolve this conflict between goals:

## Conflict Type: {conflict_type}

## Conflicting Goals
{self._format_goals(goals)}

Provide your resolution in JSON format:
{{
    "resolution_type": "prioritize/merge/sequence/abandon",
    "recommended_order": ["goal_id1", "goal_id2"],
    "goals_to_pause": ["goal_id"],
    "goals_to_abandon": ["goal_id"],
    "reasoning": "...",
    "alternative_approaches": ["..."]
}}
"""

        try:
            response = await self._llm.complete(
                messages=[{"role": "user", "content": user_prompt}],
                system=system_prompt,
            )

            response_text = self._get_text_from_response(response)
            return self._extract_json(response_text)

        except Exception as e:
            logger.error(f"Failed to resolve conflict: {e}")
            return {
                "resolution_type": "prioritize",
                "reasoning": f"Could not resolve: {e}",
            }

    async def assess_progress(
        self,
        goal: Goal,
        events: list[dict],
        artifacts: list[dict],
    ) -> dict[str, Any]:
        """
        Assess progress toward a goal.

        Returns:
            Progress assessment with recommendations
        """
        if not self._llm:
            return {
                "progress_percent": goal.metrics.progress_percent,
                "status_assessment": "unknown",
            }

        system_prompt = """You are assessing progress toward a goal.
Be realistic about progress and provide actionable recommendations."""

        user_prompt = f"""
Assess progress on this goal:

## Goal
Title: {goal.title}
Description: {goal.description}
Success Criteria: {goal.success_criteria}
Current Progress: {goal.metrics.progress_percent}%
Time Spent: {goal.metrics.time_spent_seconds / 3600:.1f} hours
Tool Calls: {goal.metrics.tool_calls}

## Recent Events
{json.dumps(events[-10:] if events else [], indent=2, default=str)}

## Artifacts Created
{json.dumps(artifacts, indent=2, default=str)}

Provide your assessment in JSON format:
{{
    "progress_percent": 0-100,
    "status_assessment": "on_track/behind/blocked/ahead",
    "completed_criteria": ["..."],
    "remaining_work": ["..."],
    "blockers": ["..."],
    "recommendations": ["..."],
    "estimated_remaining_hours": 0,
    "confidence": 0.0-1.0
}}
"""

        try:
            response = await self._llm.complete(
                messages=[{"role": "user", "content": user_prompt}],
                system=system_prompt,
            )

            response_text = self._get_text_from_response(response)
            return self._extract_json(response_text)

        except Exception as e:
            logger.error(f"Failed to assess progress: {e}")
            return {
                "progress_percent": goal.metrics.progress_percent,
                "status_assessment": "unknown",
                "error": str(e),
            }

    async def suggest_next_action(
        self,
        goal: Goal,
        context: dict[str, Any],
        available_tools: list[str],
    ) -> dict[str, Any]:
        """
        Suggest the next action to take for a goal.

        Returns:
            Suggested action with reasoning
        """
        if not self._llm:
            return {
                "action": "continue",
                "reasoning": "LLM not available",
            }

        system_prompt = """You are suggesting the next action to take toward achieving a goal.
Consider the current state, available tools, and success criteria."""

        user_prompt = f"""
Suggest the next action for this goal:

## Goal
Title: {goal.title}
Description: {goal.description}
Success Criteria: {goal.success_criteria}
Current Progress: {goal.metrics.progress_percent}%

## Context
{json.dumps(context, indent=2, default=str)}

## Available Tools
{', '.join(available_tools)}

Provide your suggestion in JSON format:
{{
    "action_type": "tool_use/reasoning/wait/complete/abort",
    "tool_name": "...",
    "tool_parameters": {{}},
    "reasoning": "...",
    "expected_outcome": "...",
    "fallback_action": "..."
}}
"""

        try:
            response = await self._llm.complete(
                messages=[{"role": "user", "content": user_prompt}],
                system=system_prompt,
            )

            response_text = self._get_text_from_response(response)
            return self._extract_json(response_text)

        except Exception as e:
            logger.error(f"Failed to suggest action: {e}")
            return {
                "action_type": "continue",
                "reasoning": f"Error: {e}",
            }

    def _build_goal_generation_prompt(self) -> str:
        """Build the system prompt for goal generation."""
        return """You are AION's goal generation engine. Your role is to propose meaningful, achievable goals that align with AION's values and serve the user's interests.

When generating goals, consider:
1. VALUE: Does this goal provide real value?
2. FEASIBILITY: Can AION actually achieve this?
3. ALIGNMENT: Does it align with core values?
4. NOVELTY: Is it distinct from existing goals?
5. TIMING: Is now the right time for this goal?

Generate goals that are:
- Specific and measurable
- Achievable with available capabilities
- Relevant to current context
- Time-bound when appropriate

Avoid goals that are:
- Too vague or abstract
- Beyond current capabilities
- Duplicates of existing goals
- Misaligned with values

Always respond with valid JSON."""

    def _format_objectives(self, objectives: list[Objective]) -> str:
        """Format objectives for prompts."""
        if not objectives:
            return "No active objectives"

        lines = []
        for obj in objectives:
            lines.append(f"- {obj.name}: {obj.description}")
        return "\n".join(lines)

    def _format_goals(self, goals: list[Goal]) -> str:
        """Format goals for prompts."""
        if not goals:
            return "No existing goals"

        lines = []
        for goal in goals:
            lines.append(
                f"- [{goal.id[:8]}] {goal.title} ({goal.status.value}, {goal.priority.name})"
            )
        return "\n".join(lines)

    def _format_values(self) -> str:
        """Format values for prompts."""
        lines = []
        for value in self.values:
            lines.append(f"- {value.name}: {value.description}")
        return "\n".join(lines)

    def _parse_priority(self, priority_str: str) -> GoalPriority:
        """Parse priority string to enum."""
        mapping = {
            "critical": GoalPriority.CRITICAL,
            "high": GoalPriority.HIGH,
            "medium": GoalPriority.MEDIUM,
            "low": GoalPriority.LOW,
            "background": GoalPriority.BACKGROUND,
        }
        return mapping.get(priority_str.lower(), GoalPriority.MEDIUM)

    def _parse_effort(self, effort_str: str) -> float:
        """Parse effort string to float."""
        mapping = {"low": 0.3, "medium": 0.5, "high": 0.8}
        return mapping.get(effort_str.lower(), 0.5)

    def _get_text_from_response(self, response: Any) -> str:
        """Extract text from LLM response."""
        if hasattr(response, "content"):
            for block in response.content:
                if hasattr(block, "text"):
                    return block.text
        return str(response)

    def _extract_json(self, text: str) -> dict[str, Any]:
        """Extract JSON from text response."""
        try:
            # Try to find JSON block
            json_start = text.find("{")
            json_end = text.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                return json.loads(text[json_start:json_end])
        except json.JSONDecodeError:
            pass

        # Try parsing entire text
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {}

    def _get_default_values(self) -> list[ValuePrinciple]:
        """Get default value principles."""
        return [
            ValuePrinciple(
                name="Helpfulness",
                description="Prioritize actions that genuinely help the user achieve their goals",
                priority=1,
                goal_generation_prompt="Consider how this goal helps the user",
            ),
            ValuePrinciple(
                name="Safety",
                description="Avoid actions that could cause harm or have dangerous side effects",
                priority=1,
                goal_generation_prompt="Ensure goals do not cause harm",
                constraints=["no_dangerous_actions", "require_approval_for_risky"],
            ),
            ValuePrinciple(
                name="Honesty",
                description="Be truthful and transparent about capabilities and limitations",
                priority=2,
                goal_generation_prompt="Set realistic, achievable goals",
            ),
            ValuePrinciple(
                name="Autonomy Respect",
                description="Respect user autonomy and get approval for significant actions",
                priority=2,
                goal_generation_prompt="Consider user preferences and consent",
            ),
            ValuePrinciple(
                name="Efficiency",
                description="Use resources wisely and avoid waste",
                priority=3,
                goal_generation_prompt="Optimize for resource efficiency",
            ),
            ValuePrinciple(
                name="Learning",
                description="Continuously improve through experience and feedback",
                priority=3,
                goal_generation_prompt="Include goals for self-improvement",
            ),
        ]

    def set_values(self, values: list[ValuePrinciple]) -> None:
        """Update the value principles."""
        self.values = values

    def add_value(self, value: ValuePrinciple) -> None:
        """Add a value principle."""
        self.values.append(value)
