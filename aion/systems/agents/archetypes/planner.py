"""
AION Planner Agent

Specialist agent for strategic planning and task decomposition.
"""

from __future__ import annotations

from typing import Any, Optional

import structlog

from aion.systems.agents.types import AgentRole, AgentStatus, TeamTask
from aion.systems.agents.archetypes.base import BaseSpecialist

logger = structlog.get_logger(__name__)


class PlannerAgent(BaseSpecialist):
    """
    Planner specialist agent.

    Capabilities:
    - Task decomposition
    - Strategic planning
    - Dependency identification
    - Resource allocation
    - Risk assessment
    """

    @property
    def role(self) -> AgentRole:
        return AgentRole.PLANNER

    async def process_task(
        self,
        task: TeamTask,
        context: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Process a planning task."""
        logger.info(
            "Processing planning task",
            agent_id=self.id[:8],
            task=task.title[:50],
        )

        self.update_status(AgentStatus.BUSY)

        try:
            # Phase 1: Analyze the objective
            analysis = await self._analyze_objective(task)

            # Phase 2: Decompose into subtasks
            subtasks = await self._decompose_task(task, analysis)

            # Phase 3: Identify dependencies
            dependencies = await self._identify_dependencies(subtasks)

            # Phase 4: Assess risks
            risks = await self._assess_risks(task, subtasks)

            # Phase 5: Create plan
            plan = await self._create_plan(task, subtasks, dependencies, risks)

            result = {
                "success": True,
                "analysis": analysis,
                "subtasks": subtasks,
                "dependencies": dependencies,
                "risks": risks,
                "plan": plan,
            }

            self.instance.tasks_completed += 1
            return result

        except Exception as e:
            logger.error("Planning task failed", error=str(e))
            self.instance.tasks_failed += 1
            return {"success": False, "error": str(e)}

        finally:
            self.update_status(AgentStatus.IDLE)

    async def _analyze_objective(self, task: TeamTask) -> str:
        """Analyze the task objective."""
        prompt = f"""Analyze this objective:

## Task: {task.title}

## Objective:
{task.objective}

## Description:
{task.description}

## Success Criteria:
{chr(10).join(f"- {c}" for c in task.success_criteria) if task.success_criteria else "- Complete successfully"}

Analyze:
1. What exactly needs to be achieved
2. Scope and boundaries
3. Key constraints
4. Critical success factors"""

        return await self.think(prompt)

    async def _decompose_task(
        self,
        task: TeamTask,
        analysis: str,
    ) -> str:
        """Decompose task into subtasks."""
        prompt = f"""Decompose this task into subtasks:

## Task: {task.title}

## Analysis:
{analysis}

Create a breakdown:
1. List all subtasks needed
2. Estimated complexity for each
3. Required skills/roles
4. Expected deliverables

Make subtasks actionable and specific."""

        return await self.think(prompt)

    async def _identify_dependencies(self, subtasks: str) -> str:
        """Identify dependencies between subtasks."""
        prompt = f"""Identify dependencies between these subtasks:

{subtasks}

Provide:
1. Which tasks depend on others
2. Which can run in parallel
3. Critical path
4. Potential bottlenecks"""

        return await self.think(prompt)

    async def _assess_risks(
        self,
        task: TeamTask,
        subtasks: str,
    ) -> str:
        """Assess risks in the plan."""
        prompt = f"""Assess risks for this plan:

## Task: {task.title}

## Subtasks:
{subtasks}

Identify:
1. Potential risks
2. Likelihood and impact
3. Mitigation strategies
4. Contingency plans"""

        return await self.think(prompt)

    async def _create_plan(
        self,
        task: TeamTask,
        subtasks: str,
        dependencies: str,
        risks: str,
    ) -> str:
        """Create the final plan."""
        prompt = f"""Create an execution plan:

## Task: {task.title}

## Subtasks:
{subtasks}

## Dependencies:
{dependencies}

## Risks:
{risks}

Create a comprehensive plan with:
1. Ordered sequence of actions
2. Parallel work opportunities
3. Milestones and checkpoints
4. Risk mitigation integrated
5. Success criteria for each phase"""

        return await self.think(prompt)

    async def plan(
        self,
        objective: str,
        constraints: list[str] = None,
    ) -> str:
        """Create a plan for an objective."""
        constraints_str = "\n".join(f"- {c}" for c in constraints) if constraints else "None specified"

        prompt = f"""Create a plan for:

{objective}

Constraints:
{constraints_str}

Provide: subtasks, dependencies, timeline, risks."""

        return await self.think(prompt)
