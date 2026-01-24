"""
AION Reviewer Agent

Specialist agent for quality review and feedback.
"""

from __future__ import annotations

from typing import Any, Optional

import structlog

from aion.systems.agents.types import AgentRole, AgentStatus, TeamTask
from aion.systems.agents.archetypes.base import BaseSpecialist

logger = structlog.get_logger(__name__)


class ReviewerAgent(BaseSpecialist):
    """
    Reviewer specialist agent.

    Capabilities:
    - Quality assessment
    - Feedback generation
    - Verification against criteria
    - Improvement suggestions
    """

    @property
    def role(self) -> AgentRole:
        return AgentRole.REVIEWER

    async def process_task(
        self,
        task: TeamTask,
        context: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Process a review task."""
        logger.info(
            "Processing review task",
            agent_id=self.id[:8],
            task=task.title[:50],
        )

        self.update_status(AgentStatus.BUSY)

        try:
            # Get the work to review
            work = context.get("work", task.description) if context else task.description
            criteria = task.success_criteria or []

            # Phase 1: Understand requirements
            requirements = await self._understand_requirements(task, criteria)

            # Phase 2: Evaluate work
            evaluation = await self._evaluate_work(task, work, requirements)

            # Phase 3: Identify issues
            issues = await self._identify_issues(task, work, evaluation)

            # Phase 4: Generate recommendations
            recommendations = await self._generate_recommendations(task, issues)

            # Phase 5: Final verdict
            verdict = await self._final_verdict(task, evaluation, issues)

            result = {
                "success": True,
                "requirements": requirements,
                "evaluation": evaluation,
                "issues": issues,
                "recommendations": recommendations,
                "verdict": verdict,
            }

            self.instance.tasks_completed += 1
            return result

        except Exception as e:
            logger.error("Review task failed", error=str(e))
            self.instance.tasks_failed += 1
            return {"success": False, "error": str(e)}

        finally:
            self.update_status(AgentStatus.IDLE)

    async def _understand_requirements(
        self,
        task: TeamTask,
        criteria: list[str],
    ) -> str:
        """Understand review requirements."""
        criteria_str = "\n".join(f"- {c}" for c in criteria) if criteria else "- General quality"

        prompt = f"""Understand the review requirements:

## Task: {task.title}

## Objective:
{task.objective}

## Success Criteria:
{criteria_str}

List:
1. What needs to be verified
2. Quality standards to apply
3. Key areas of focus"""

        return await self.think(prompt)

    async def _evaluate_work(
        self,
        task: TeamTask,
        work: str,
        requirements: str,
    ) -> str:
        """Evaluate the work against requirements."""
        prompt = f"""Evaluate this work against requirements:

## Work to Review:
{work}

## Requirements:
{requirements}

Provide detailed evaluation:
1. How well does it meet each requirement?
2. Overall quality assessment
3. Strengths identified
4. Areas of concern"""

        return await self.think(prompt)

    async def _identify_issues(
        self,
        task: TeamTask,
        work: str,
        evaluation: str,
    ) -> str:
        """Identify specific issues."""
        prompt = f"""Identify specific issues in this work:

## Work:
{work}

## Initial Evaluation:
{evaluation}

List issues found:
1. Critical issues (must fix)
2. Major issues (should fix)
3. Minor issues (nice to fix)
4. Suggestions (optional improvements)

Be specific and provide examples."""

        return await self.think(prompt)

    async def _generate_recommendations(
        self,
        task: TeamTask,
        issues: str,
    ) -> str:
        """Generate improvement recommendations."""
        prompt = f"""Generate recommendations to address these issues:

## Issues Found:
{issues}

For each issue, provide:
1. Specific recommendation
2. How to implement it
3. Expected improvement
4. Priority level"""

        return await self.think(prompt)

    async def _final_verdict(
        self,
        task: TeamTask,
        evaluation: str,
        issues: str,
    ) -> str:
        """Provide final verdict."""
        criteria_str = "\n".join(f"- {c}" for c in task.success_criteria) if task.success_criteria else "- General quality"

        prompt = f"""Provide final review verdict:

## Success Criteria:
{criteria_str}

## Evaluation:
{evaluation}

## Issues:
{issues}

Provide:
1. APPROVE / NEEDS CHANGES / REJECT
2. Summary of decision rationale
3. Required changes (if any)
4. Confidence in verdict"""

        return await self.think(prompt)

    async def review(
        self,
        work: str,
        criteria: list[str],
    ) -> str:
        """Quick review of work against criteria."""
        criteria_str = "\n".join(f"- {c}" for c in criteria)

        prompt = f"""Review this work:

{work}

Against criteria:
{criteria_str}

Provide: verdict, key issues, recommendations."""

        return await self.think(prompt)
