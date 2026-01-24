"""
AION Executor Agent

Specialist agent for task execution and automation.
"""

from __future__ import annotations

from typing import Any, Optional

import structlog

from aion.systems.agents.types import AgentRole, AgentStatus, TeamTask
from aion.systems.agents.archetypes.base import BaseSpecialist

logger = structlog.get_logger(__name__)


class ExecutorAgent(BaseSpecialist):
    """
    Executor specialist agent.

    Capabilities:
    - Task execution
    - Process automation
    - Step-by-step completion
    - Progress reporting
    """

    @property
    def role(self) -> AgentRole:
        return AgentRole.EXECUTOR

    async def process_task(
        self,
        task: TeamTask,
        context: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Process an execution task."""
        logger.info(
            "Processing execution task",
            agent_id=self.id[:8],
            task=task.title[:50],
        )

        self.update_status(AgentStatus.BUSY)

        try:
            # Phase 1: Understand the task
            understanding = await self._understand_task(task, context)

            # Phase 2: Plan execution steps
            steps = await self._plan_steps(task, understanding)

            # Phase 3: Execute steps
            execution = await self._execute_steps(task, steps)

            # Phase 4: Verify completion
            verification = await self._verify_completion(task, execution)

            result = {
                "success": verification.get("complete", False),
                "understanding": understanding,
                "steps": steps,
                "execution": execution,
                "verification": verification,
            }

            self.instance.tasks_completed += 1
            return result

        except Exception as e:
            logger.error("Execution task failed", error=str(e))
            self.instance.tasks_failed += 1
            return {"success": False, "error": str(e)}

        finally:
            self.update_status(AgentStatus.IDLE)

    async def _understand_task(
        self,
        task: TeamTask,
        context: Optional[dict[str, Any]],
    ) -> str:
        """Understand what needs to be executed."""
        context_str = ""
        if context:
            context_str = "\n".join(f"- {k}: {v}" for k, v in context.items())

        prompt = f"""Understand this execution task:

## Task: {task.title}

## Description:
{task.description}

## Objective:
{task.objective}

## Context:
{context_str if context_str else "No additional context"}

Clarify:
1. What exactly needs to be done
2. Expected inputs and outputs
3. Constraints to follow
4. Definition of done"""

        return await self.think(prompt)

    async def _plan_steps(
        self,
        task: TeamTask,
        understanding: str,
    ) -> str:
        """Plan execution steps."""
        prompt = f"""Plan execution steps:

## Task: {task.title}

## Understanding:
{understanding}

Create step-by-step plan:
1. Ordered list of actions
2. What each step produces
3. How to verify each step
4. Handling of potential issues"""

        return await self.think(prompt)

    async def _execute_steps(
        self,
        task: TeamTask,
        steps: str,
    ) -> str:
        """Execute the planned steps."""
        prompt = f"""Execute these steps:

## Task: {task.title}

## Steps:
{steps}

For each step:
1. Perform the action
2. Record the result
3. Note any issues
4. Confirm completion

Report progress and results."""

        return await self.think(prompt)

    async def _verify_completion(
        self,
        task: TeamTask,
        execution: str,
    ) -> dict[str, Any]:
        """Verify task completion."""
        criteria_str = "\n".join(f"- {c}" for c in task.success_criteria) if task.success_criteria else "- Task completed successfully"

        prompt = f"""Verify task completion:

## Task: {task.title}

## Success Criteria:
{criteria_str}

## Execution Results:
{execution}

Verify:
1. Are all success criteria met?
2. Is the task fully complete?
3. Any remaining issues?
4. Final status: COMPLETE / INCOMPLETE

Provide clear YES/NO for completion."""

        result = await self.think(prompt)

        # Parse completion status
        complete = any(w in result.lower() for w in ["complete", "yes", "done", "success"])

        return {
            "complete": complete,
            "verification": result,
        }

    async def execute(
        self,
        instructions: str,
    ) -> str:
        """Execute given instructions."""
        prompt = f"""Execute these instructions:

{instructions}

Follow the instructions precisely and report:
1. Actions taken
2. Results produced
3. Any issues encountered
4. Completion status"""

        return await self.think(prompt)
