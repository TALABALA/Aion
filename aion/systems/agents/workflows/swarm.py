"""
AION Swarm Workflow

Emergent coordination through decentralized agent communication.
"""

from __future__ import annotations

import asyncio
import random
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


class SwarmWorkflow(WorkflowExecutor):
    """
    Swarm workflow: emergent coordination through communication.

    Agents work semi-independently, sharing discoveries and
    building on each other's work through message passing.
    The collective output emerges from their interactions.

    Best for:
    - Exploratory tasks
    - Creative brainstorming
    - Complex problems with multiple solution paths
    """

    def __init__(
        self,
        max_iterations: int = 5,
        convergence_threshold: float = 0.8,
    ):
        super().__init__()
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold

    @property
    def pattern(self) -> WorkflowPattern:
        return WorkflowPattern.SWARM

    async def execute(
        self,
        team: Team,
        task: TeamTask,
        pool: "AgentPool",
        bus: "MessageBus",
    ) -> dict[str, Any]:
        """Execute swarm workflow."""
        logger.info(
            "Starting swarm workflow",
            task=task.title[:50],
            agents=len(team.agent_ids),
            max_iterations=self.max_iterations,
        )

        if len(team.agent_ids) < 2:
            # Single agent - just execute directly
            return await self._single_agent_execution(
                pool.get_agent(team.agent_ids[0]),
                task,
                pool,
            )

        execution = self.create_execution(team, task)

        # Initialize agent states
        agent_states: dict[str, dict] = {}
        for agent_id in team.agent_ids:
            agent_states[agent_id] = {
                "discoveries": [],
                "contributions": [],
                "insights": [],
            }

        # Shared knowledge pool
        shared_knowledge: list[dict] = []

        total_tokens = 0

        # Swarm iterations
        for iteration in range(self.max_iterations):
            logger.info(f"Swarm iteration {iteration + 1}/{self.max_iterations}")

            # Each agent works and shares
            iteration_results = await self._run_iteration(
                iteration,
                team,
                task,
                agent_states,
                shared_knowledge,
                pool,
                bus,
                execution,
            )

            total_tokens += iteration_results.get("tokens", 0)

            # Update shared knowledge
            for result in iteration_results.get("contributions", []):
                shared_knowledge.append(result)

            # Update progress
            task.progress = (iteration + 1) / self.max_iterations

            # Check for convergence
            if self._check_convergence(agent_states, iteration):
                logger.info("Swarm converged early", iteration=iteration + 1)
                break

        # Synthesize final output
        final_output = await self._synthesize_swarm_output(
            team,
            task,
            shared_knowledge,
            pool,
            execution,
        )

        total_tokens += final_output.get("tokens", 0)

        # Complete execution
        result = {
            "synthesis": final_output.get("output"),
            "shared_knowledge": shared_knowledge,
            "iterations": iteration + 1,
            "agent_contributions": {
                aid: state["contributions"]
                for aid, state in agent_states.items()
            },
        }

        self.complete_execution(execution, True, result)

        logger.info(
            "Swarm workflow complete",
            task=task.id[:8],
            iterations=iteration + 1,
            knowledge_items=len(shared_knowledge),
            total_tokens=total_tokens,
        )

        return {
            "success": True,
            "output": result,
            "workflow": "swarm",
            "iterations": iteration + 1,
            "total_tokens": total_tokens,
        }

    async def _single_agent_execution(
        self,
        agent,
        task: TeamTask,
        pool: "AgentPool",
    ) -> dict[str, Any]:
        """Handle single agent case."""
        prompt = f"""Complete this task:

## Task: {task.title}

## Objective:
{task.objective}

## Description:
{task.description}

Provide a complete solution."""

        result = await self.execute_agent_step(agent, prompt, pool=pool)

        return {
            "success": result.get("success", False),
            "output": {"synthesis": result.get("output")},
            "workflow": "swarm",
            "iterations": 1,
            "total_tokens": result.get("tokens", 0),
        }

    async def _run_iteration(
        self,
        iteration: int,
        team: Team,
        task: TeamTask,
        agent_states: dict[str, dict],
        shared_knowledge: list[dict],
        pool: "AgentPool",
        bus: "MessageBus",
        execution,
    ) -> dict[str, Any]:
        """Run one swarm iteration."""
        async def agent_work(agent_id: str) -> dict:
            agent = pool.get_agent(agent_id)
            if not agent:
                return {"agent_id": agent_id, "error": "Agent not found"}

            step = self.add_step(
                execution,
                agent_id,
                f"Iteration {iteration + 1} work",
                {"iteration": iteration, "knowledge_count": len(shared_knowledge)},
            )

            pool.update_status(agent_id, AgentStatus.BUSY, task.id)

            try:
                # Build prompt with shared knowledge
                prompt = self._build_swarm_prompt(
                    task,
                    agent,
                    agent_states[agent_id],
                    shared_knowledge,
                    iteration,
                )

                result = await self.execute_agent_step(agent, prompt, pool=pool)

                contribution = {
                    "agent_id": agent_id,
                    "agent_role": agent.profile.role.value,
                    "iteration": iteration,
                    "content": result.get("output", ""),
                }

                agent_states[agent_id]["contributions"].append(contribution)
                self.complete_step(step, result.get("output"), result.get("tokens", 0))

                # Share with team
                await bus.broadcast(
                    sender_id=agent_id,
                    content={
                        "type": "swarm_contribution",
                        "iteration": iteration,
                        "contribution": contribution,
                    },
                    subject=f"Swarm contribution (iteration {iteration + 1})",
                    team_id=team.id,
                )

                return {
                    "success": True,
                    "contribution": contribution,
                    "tokens": result.get("tokens", 0),
                }

            except Exception as e:
                self.complete_step(step, None, 0, str(e))
                return {"agent_id": agent_id, "error": str(e), "success": False}

            finally:
                pool.update_status(agent_id, AgentStatus.IDLE)

        # Run all agents in parallel
        results = await asyncio.gather(*[
            agent_work(aid) for aid in team.agent_ids
        ])

        contributions = [
            r["contribution"] for r in results
            if r.get("success") and "contribution" in r
        ]

        total_tokens = sum(r.get("tokens", 0) for r in results)

        return {
            "contributions": contributions,
            "tokens": total_tokens,
        }

    def _build_swarm_prompt(
        self,
        task: TeamTask,
        agent,
        agent_state: dict,
        shared_knowledge: list[dict],
        iteration: int,
    ) -> str:
        """Build prompt for swarm agent."""
        # Summarize shared knowledge
        knowledge_summary = ""
        if shared_knowledge:
            recent = shared_knowledge[-5:]  # Last 5 items
            for k in recent:
                knowledge_summary += f"\n- [{k['agent_role']}]: {k['content'][:200]}...\n"

        # Summarize own contributions
        own_work = ""
        if agent_state["contributions"]:
            own_work = f"\n\nYour previous contributions:\n"
            for c in agent_state["contributions"][-2:]:
                own_work += f"- {c['content'][:200]}...\n"

        return f"""You are part of a swarm working on a task. Share discoveries and build on others' work.

## Task: {task.title}

## Objective:
{task.objective}

## Description:
{task.description}

## Iteration: {iteration + 1}

## Team's Shared Knowledge:
{knowledge_summary if knowledge_summary else "No shared knowledge yet - you can start!"}
{own_work}

## Your Role: {agent.profile.role.value}

Instructions:
1. Review the shared knowledge from your teammates
2. Build on promising ideas or explore new directions
3. Share any insights, discoveries, or progress
4. Collaborate by acknowledging and extending others' work

What do you contribute this iteration? Be specific and constructive."""

    def _check_convergence(
        self,
        agent_states: dict[str, dict],
        iteration: int,
    ) -> bool:
        """Check if the swarm has converged."""
        if iteration < 2:
            return False

        # Check if contributions are becoming similar
        # (simplified convergence check)
        all_contributions = []
        for state in agent_states.values():
            if state["contributions"]:
                all_contributions.append(state["contributions"][-1]["content"])

        if len(all_contributions) < 2:
            return False

        # Simple heuristic: if contributions mention similar concepts
        # In practice, you'd use embedding similarity
        common_words = set()
        for contrib in all_contributions:
            words = set(contrib.lower().split())
            if not common_words:
                common_words = words
            else:
                common_words &= words

        # If many common words, agents are converging
        return len(common_words) > 20

    async def _synthesize_swarm_output(
        self,
        team: Team,
        task: TeamTask,
        shared_knowledge: list[dict],
        pool: "AgentPool",
        execution,
    ) -> dict[str, Any]:
        """Synthesize final output from swarm knowledge."""
        # Use team leader or first agent
        synthesizer_id = team.leader_id or team.agent_ids[0]
        synthesizer = pool.get_agent(synthesizer_id)

        step = self.add_step(
            execution,
            synthesizer_id,
            "Synthesize swarm output",
            {"knowledge_count": len(shared_knowledge)},
        )

        pool.update_status(synthesizer_id, AgentStatus.BUSY, task.id)

        try:
            # Format all knowledge
            knowledge_text = ""
            for k in shared_knowledge:
                knowledge_text += f"\n### [{k['agent_role']}] (Iteration {k['iteration'] + 1}):\n{k['content']}\n"

            prompt = f"""Synthesize the swarm's collective work into a final output.

## Task: {task.title}

## Objective:
{task.objective}

## Success Criteria:
{chr(10).join(f"- {c}" for c in task.success_criteria) if task.success_criteria else "- Complete effectively"}

## Swarm's Collective Knowledge:
{knowledge_text}

Create a comprehensive final output that:
1. Integrates the best ideas from all contributions
2. Resolves any conflicts between different approaches
3. Addresses all success criteria
4. Represents the swarm's collective intelligence

Provide the complete synthesized output."""

            result = await self.execute_agent_step(synthesizer, prompt, pool=pool)
            self.complete_step(step, result.get("output"), result.get("tokens", 0))

            return {
                "output": result.get("output"),
                "tokens": result.get("tokens", 0),
            }

        finally:
            pool.update_status(synthesizer_id, AgentStatus.IDLE)
