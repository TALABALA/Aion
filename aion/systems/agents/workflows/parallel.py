"""
AION Parallel Workflow

Agents work concurrently and results are combined.
"""

from __future__ import annotations

import asyncio
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


class ParallelWorkflow(WorkflowExecutor):
    """
    Parallel workflow: all agents work simultaneously.

    Each agent receives the same task and works independently.
    Results are collected and merged at the end.

    Best for:
    - Independent research on different aspects
    - Multiple perspectives on same problem
    - Time-sensitive tasks
    """

    @property
    def pattern(self) -> WorkflowPattern:
        return WorkflowPattern.PARALLEL

    async def execute(
        self,
        team: Team,
        task: TeamTask,
        pool: "AgentPool",
        bus: "MessageBus",
    ) -> dict[str, Any]:
        """Execute parallel workflow."""
        logger.info(
            "Starting parallel workflow",
            task=task.title[:50],
            agents=len(team.agent_ids),
        )

        execution = self.create_execution(team, task)

        # Create tasks for each agent
        agent_tasks = []
        for agent_id in team.agent_ids:
            agent = pool.get_agent(agent_id)
            if agent:
                agent_tasks.append(
                    self._execute_agent(
                        agent,
                        task,
                        team,
                        pool,
                        bus,
                        execution,
                    )
                )

        # Execute all agents in parallel
        results = await asyncio.gather(*agent_tasks, return_exceptions=True)

        # Collect results
        contributions = []
        total_tokens = 0
        successful = 0

        for result in results:
            if isinstance(result, dict) and result.get("success"):
                contributions.append(result["contribution"])
                total_tokens += result.get("tokens", 0)
                successful += 1
            elif isinstance(result, Exception):
                logger.warning("Agent error", error=str(result))

        # Combine results
        combined_output = {
            "task": task.to_dict(),
            "contributions": contributions,
            "parallel_execution": True,
            "agents_succeeded": successful,
            "agents_total": len(team.agent_ids),
        }

        # Mark all agents as complete, update progress
        task.progress = 1.0

        # Complete execution
        self.complete_execution(execution, successful > 0, combined_output)

        logger.info(
            "Parallel workflow complete",
            task=task.id[:8],
            successful=successful,
            total=len(team.agent_ids),
            total_tokens=total_tokens,
        )

        return {
            "success": successful > 0,
            "output": combined_output,
            "workflow": "parallel",
            "steps": len(execution.steps),
            "total_tokens": total_tokens,
            "agents_succeeded": successful,
        }

    async def _execute_agent(
        self,
        agent,
        task: TeamTask,
        team: Team,
        pool: "AgentPool",
        bus: "MessageBus",
        execution,
    ) -> dict[str, Any]:
        """Execute a single agent's work."""
        step = self.add_step(
            execution,
            agent.id,
            f"Parallel work: {agent.profile.role.value}",
            task.to_dict(),
        )

        pool.update_status(agent.id, AgentStatus.BUSY, task.id)

        # Notify agent
        await bus.send(
            sender_id="orchestrator",
            recipient_id=agent.id,
            message_type=MessageType.TASK,
            content={
                "task": task.to_dict(),
                "workflow": "parallel",
                "your_role": agent.profile.role.value,
            },
            subject=f"Task: {task.title}",
            team_id=team.id,
        )

        try:
            prompt = self._build_prompt(task, agent)
            result = await self.execute_agent_step(agent, prompt, pool=pool)

            if result.get("success"):
                contribution = {
                    "agent_id": agent.id,
                    "agent_role": agent.profile.role.value,
                    "output": result["output"],
                }
                self.complete_step(step, result["output"], result.get("tokens", 0))

                return {
                    "success": True,
                    "contribution": contribution,
                    "tokens": result.get("tokens", 0),
                }
            else:
                self.complete_step(step, None, 0, result.get("error"))
                return {
                    "success": False,
                    "error": result.get("error"),
                }

        except Exception as e:
            self.complete_step(step, None, 0, str(e))
            return {"success": False, "error": str(e)}

        finally:
            pool.update_status(agent.id, AgentStatus.IDLE)

    def _build_prompt(self, task: TeamTask, agent) -> str:
        """Build prompt for agent."""
        return f"""You are a {agent.profile.role.value} agent working on a parallel team task.

## Task: {task.title}

## Objective:
{task.objective}

## Description:
{task.description}

## Success Criteria:
{chr(10).join(f"- {c}" for c in task.success_criteria) if task.success_criteria else "- Complete the task effectively"}

## Your Role:
As the {agent.profile.role.value}, provide your unique perspective and expertise on this task.
Your work will be combined with other agents' contributions.

Focus on what you do best and provide thorough, valuable output."""
