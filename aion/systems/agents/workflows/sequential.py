"""
AION Sequential Workflow

Agents work one after another, each building on previous work.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import structlog

from aion.systems.agents.types import (
    Team,
    TeamTask,
    WorkflowPattern,
    AgentStatus,
    MessageType,
)
from aion.systems.agents.workflows.base import WorkflowExecutor

if TYPE_CHECKING:
    from aion.systems.agents.pool import AgentPool
    from aion.systems.agents.messaging import MessageBus

logger = structlog.get_logger(__name__)


class SequentialWorkflow(WorkflowExecutor):
    """
    Sequential workflow: agents work one at a time.

    Each agent receives the task and all previous contributions,
    then adds their own contribution. This creates a pipeline
    where each agent builds on previous work.

    Best for:
    - Research -> Analysis -> Writing tasks
    - Code -> Review -> Refactor chains
    - Tasks with clear dependencies
    """

    @property
    def pattern(self) -> WorkflowPattern:
        return WorkflowPattern.SEQUENTIAL

    async def execute(
        self,
        team: Team,
        task: TeamTask,
        pool: "AgentPool",
        bus: "MessageBus",
    ) -> dict[str, Any]:
        """Execute sequential workflow."""
        logger.info(
            "Starting sequential workflow",
            task=task.title[:50],
            agents=len(team.agent_ids),
        )

        execution = self.create_execution(team, task)

        # Track accumulated work
        accumulated_work = {
            "task": task.to_dict(),
            "contributions": [],
        }

        total_tokens = 0

        for i, agent_id in enumerate(team.agent_ids):
            agent = pool.get_agent(agent_id)
            if not agent:
                logger.warning("Agent not found", agent_id=agent_id[:8])
                continue

            # Create step
            step = self.add_step(
                execution,
                agent_id,
                f"Contribution {i+1}/{len(team.agent_ids)}",
                accumulated_work,
            )

            # Update agent status
            pool.update_status(agent_id, AgentStatus.BUSY, task.id)

            # Send task message
            await bus.send(
                sender_id="orchestrator",
                recipient_id=agent_id,
                message_type=MessageType.TASK,
                content={
                    "task": task.to_dict(),
                    "previous_work": accumulated_work,
                    "your_role": agent.profile.role.value,
                    "position": f"{i+1}/{len(team.agent_ids)}",
                },
                subject=f"Task: {task.title}",
                team_id=team.id,
            )

            # Execute agent's work
            prompt = self._build_prompt(task, accumulated_work, agent, i, len(team.agent_ids))
            result = await self.execute_agent_step(agent, prompt, pool=pool)

            if result.get("success"):
                contribution = {
                    "agent_id": agent_id,
                    "agent_role": agent.profile.role.value,
                    "output": result["output"],
                    "tokens": result.get("tokens", 0),
                }
                accumulated_work["contributions"].append(contribution)
                total_tokens += result.get("tokens", 0)

                self.complete_step(step, result["output"], result.get("tokens", 0))
            else:
                self.complete_step(step, None, 0, result.get("error"))
                logger.warning(
                    "Agent step failed",
                    agent_id=agent_id[:8],
                    error=result.get("error"),
                )

            # Update status
            pool.update_status(agent_id, AgentStatus.IDLE)

            # Update progress
            task.progress = (i + 1) / len(team.agent_ids)

        # Complete execution
        self.complete_execution(execution, True, accumulated_work)

        logger.info(
            "Sequential workflow complete",
            task=task.id[:8],
            contributions=len(accumulated_work["contributions"]),
            total_tokens=total_tokens,
        )

        return {
            "success": True,
            "output": accumulated_work,
            "workflow": "sequential",
            "steps": len(execution.steps),
            "total_tokens": total_tokens,
        }

    def _build_prompt(
        self,
        task: TeamTask,
        accumulated: dict,
        agent,
        position: int,
        total: int,
    ) -> str:
        """Build prompt for agent."""
        previous_work = ""
        if accumulated["contributions"]:
            previous_work = "\n\n## Previous Contributions:\n"
            for contrib in accumulated["contributions"]:
                previous_work += f"\n### {contrib['agent_role'].title()}:\n{contrib['output']}\n"

        prompt = f"""You are a {agent.profile.role.value} agent working on a team task.

## Task: {task.title}

## Objective:
{task.objective}

## Description:
{task.description}

## Success Criteria:
{chr(10).join(f"- {c}" for c in task.success_criteria) if task.success_criteria else "- Complete the task effectively"}
{previous_work}

## Your Role:
You are agent {position + 1} of {total}. As the {agent.profile.role.value}, contribute your expertise.
{"Build on the previous work above." if previous_work else "You are the first to contribute."}

Provide your contribution clearly and thoroughly."""

        return prompt
