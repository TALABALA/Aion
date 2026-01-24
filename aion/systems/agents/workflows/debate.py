"""
AION Debate Workflow

Agents improve work through structured debate and refinement.
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


class DebateWorkflow(WorkflowExecutor):
    """
    Debate workflow: agents refine through discussion.

    Agents take turns critiquing and improving each other's work,
    leading to higher quality output through iterative refinement.

    Best for:
    - Complex decisions requiring multiple perspectives
    - Quality-critical outputs
    - Creative or analytical tasks
    """

    def __init__(self, max_rounds: int = 3):
        super().__init__()
        self.max_rounds = max_rounds

    @property
    def pattern(self) -> WorkflowPattern:
        return WorkflowPattern.DEBATE

    async def execute(
        self,
        team: Team,
        task: TeamTask,
        pool: "AgentPool",
        bus: "MessageBus",
    ) -> dict[str, Any]:
        """Execute debate workflow."""
        logger.info(
            "Starting debate workflow",
            task=task.title[:50],
            agents=len(team.agent_ids),
            rounds=self.max_rounds,
        )

        if len(team.agent_ids) < 2:
            return {"success": False, "error": "Debate requires at least 2 agents"}

        execution = self.create_execution(team, task)

        # Phase 1: Initial proposal from first agent
        proposer = pool.get_agent(team.agent_ids[0])
        pool.update_status(proposer.id, AgentStatus.BUSY, task.id)

        proposal_step = self.add_step(
            execution,
            proposer.id,
            "Create initial proposal",
            task.to_dict(),
        )

        current_work = await self._create_initial_proposal(proposer, task, pool)
        self.complete_step(proposal_step, current_work)
        pool.update_status(proposer.id, AgentStatus.IDLE)

        debate_history = [{
            "round": 0,
            "type": "proposal",
            "agent_id": proposer.id,
            "agent_role": proposer.profile.role.value,
            "content": current_work,
        }]

        # Phase 2: Debate rounds
        total_tokens = 0

        for round_num in range(1, self.max_rounds + 1):
            logger.info(f"Debate round {round_num}/{self.max_rounds}")

            # Each agent critiques and improves
            for i, agent_id in enumerate(team.agent_ids):
                agent = pool.get_agent(agent_id)
                if not agent:
                    continue

                pool.update_status(agent_id, AgentStatus.BUSY, task.id)

                # Step 1: Critique
                critique_step = self.add_step(
                    execution,
                    agent_id,
                    f"Round {round_num} critique",
                    {"current_work": current_work, "history": debate_history[-3:]},
                )

                critique = await self._get_critique(
                    agent, task, current_work, debate_history, pool
                )

                self.complete_step(critique_step, critique)

                debate_history.append({
                    "round": round_num,
                    "type": "critique",
                    "agent_id": agent_id,
                    "agent_role": agent.profile.role.value,
                    "content": critique.get("output", ""),
                })

                # Step 2: Improvement
                improve_step = self.add_step(
                    execution,
                    agent_id,
                    f"Round {round_num} improvement",
                    {"current_work": current_work, "critique": critique},
                )

                improvement = await self._get_improvement(
                    agent, task, current_work, critique.get("output", ""), pool
                )

                current_work = improvement.get("output", current_work)
                self.complete_step(improve_step, current_work)

                debate_history.append({
                    "round": round_num,
                    "type": "improvement",
                    "agent_id": agent_id,
                    "agent_role": agent.profile.role.value,
                    "content": current_work,
                })

                total_tokens += critique.get("tokens", 0) + improvement.get("tokens", 0)

                pool.update_status(agent_id, AgentStatus.IDLE)

                # Broadcast update
                await bus.broadcast(
                    sender_id=agent_id,
                    content={
                        "type": "debate_update",
                        "round": round_num,
                        "agent_role": agent.profile.role.value,
                        "action": "improvement",
                    },
                    subject=f"Debate round {round_num} update",
                    team_id=team.id,
                )

            # Update progress
            task.progress = round_num / self.max_rounds

        # Complete execution
        final_output = {
            "final_work": current_work,
            "debate_history": debate_history,
            "rounds": self.max_rounds,
            "participants": len(team.agent_ids),
        }

        self.complete_execution(execution, True, final_output)

        logger.info(
            "Debate workflow complete",
            task=task.id[:8],
            rounds=self.max_rounds,
            history_entries=len(debate_history),
            total_tokens=total_tokens,
        )

        return {
            "success": True,
            "output": final_output,
            "workflow": "debate",
            "rounds": self.max_rounds,
            "total_tokens": total_tokens,
        }

    async def _create_initial_proposal(
        self,
        agent,
        task: TeamTask,
        pool: "AgentPool",
    ) -> str:
        """Create initial proposal."""
        prompt = f"""Create an initial proposal for this task:

## Task: {task.title}

## Objective:
{task.objective}

## Description:
{task.description}

## Success Criteria:
{chr(10).join(f"- {c}" for c in task.success_criteria) if task.success_criteria else "- Complete effectively"}

Provide a complete first draft that addresses all criteria. This will be refined through debate."""

        result = await self.execute_agent_step(agent, prompt, pool=pool)
        return result.get("output", "")

    async def _get_critique(
        self,
        agent,
        task: TeamTask,
        current_work: str,
        history: list[dict],
        pool: "AgentPool",
    ) -> dict[str, Any]:
        """Get agent's critique of current work."""
        # Summarize recent history
        recent_history = ""
        for entry in history[-3:]:
            recent_history += f"\n[{entry['type'].upper()}] {entry['agent_role']}: {entry['content'][:500]}...\n"

        prompt = f"""Critically review this work:

## Task: {task.title}

## Success Criteria:
{chr(10).join(f"- {c}" for c in task.success_criteria) if task.success_criteria else "- Complete effectively"}

## Current Work:
{current_work}

## Recent Debate History:
{recent_history if recent_history else "This is the first review."}

Provide constructive critique:
1. **Strengths**: What works well?
2. **Weaknesses**: What could be improved?
3. **Gaps**: What's missing?
4. **Suggestions**: Specific actionable improvements

Be thorough but constructive."""

        return await self.execute_agent_step(
            agent,
            prompt,
            f"{agent.profile.system_prompt}\n\nBe constructive but thorough in your critique.",
            pool=pool,
        )

    async def _get_improvement(
        self,
        agent,
        task: TeamTask,
        current_work: str,
        critique: str,
        pool: "AgentPool",
    ) -> dict[str, Any]:
        """Get improved version based on critique."""
        prompt = f"""Improve this work based on the critique:

## Task: {task.title}

## Success Criteria:
{chr(10).join(f"- {c}" for c in task.success_criteria) if task.success_criteria else "- Complete effectively"}

## Current Work:
{current_work}

## Critique to Address:
{critique}

Create an improved version that:
1. Addresses all valid criticism
2. Maintains what works well
3. Fills identified gaps
4. Implements suggested improvements

Provide the complete improved version."""

        return await self.execute_agent_step(agent, prompt, pool=pool)
