"""
AION Hierarchical Workflow

Manager agent delegates to worker agents and synthesizes results.
"""

from __future__ import annotations

import asyncio
import json
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


class HierarchicalWorkflow(WorkflowExecutor):
    """
    Hierarchical workflow: manager delegates to workers.

    The team leader (manager) breaks down the task,
    assigns subtasks to workers, and synthesizes results.

    Best for:
    - Complex tasks requiring decomposition
    - Tasks with multiple independent components
    - Projects needing coordination
    """

    @property
    def pattern(self) -> WorkflowPattern:
        return WorkflowPattern.HIERARCHICAL

    async def execute(
        self,
        team: Team,
        task: TeamTask,
        pool: "AgentPool",
        bus: "MessageBus",
    ) -> dict[str, Any]:
        """Execute hierarchical workflow."""
        logger.info(
            "Starting hierarchical workflow",
            task=task.title[:50],
            agents=len(team.agent_ids),
        )

        execution = self.create_execution(team, task)

        # Get manager (team leader)
        manager_id = team.leader_id or team.agent_ids[0]
        manager = pool.get_agent(manager_id)

        if not manager:
            return {"success": False, "error": "No manager available"}

        # Workers are everyone else
        worker_ids = [aid for aid in team.agent_ids if aid != manager_id]

        if not worker_ids:
            # Single agent - just execute directly
            return await self._single_agent_execution(manager, task, pool, execution)

        # Phase 1: Manager decomposes task
        logger.info("Phase 1: Manager decomposing task")
        pool.update_status(manager_id, AgentStatus.BUSY, task.id)

        decompose_step = self.add_step(
            execution,
            manager_id,
            "Decompose task into subtasks",
            task.to_dict(),
        )

        subtasks = await self._manager_decompose(manager, task, worker_ids, pool)
        self.complete_step(decompose_step, subtasks)

        pool.update_status(manager_id, AgentStatus.WAITING)
        task.progress = 0.2

        # Phase 2: Workers execute subtasks
        logger.info("Phase 2: Workers executing subtasks", subtasks=len(subtasks))

        worker_results = await self._execute_workers(
            subtasks,
            worker_ids,
            task,
            team,
            pool,
            bus,
            execution,
        )

        task.progress = 0.7

        # Phase 3: Manager synthesizes results
        logger.info("Phase 3: Manager synthesizing results")
        pool.update_status(manager_id, AgentStatus.BUSY, task.id)

        synthesize_step = self.add_step(
            execution,
            manager_id,
            "Synthesize worker results",
            worker_results,
        )

        final_result = await self._manager_synthesize(
            manager, task, worker_results, pool
        )

        self.complete_step(synthesize_step, final_result)
        pool.update_status(manager_id, AgentStatus.IDLE)
        task.progress = 1.0

        # Complete execution
        total_tokens = sum(s.tokens_used for s in execution.steps)
        self.complete_execution(execution, True, final_result)

        logger.info(
            "Hierarchical workflow complete",
            task=task.id[:8],
            subtasks=len(subtasks),
            total_tokens=total_tokens,
        )

        return {
            "success": True,
            "output": final_result,
            "workflow": "hierarchical",
            "subtasks_completed": len(worker_results),
            "total_tokens": total_tokens,
        }

    async def _single_agent_execution(
        self,
        agent,
        task: TeamTask,
        pool: "AgentPool",
        execution,
    ) -> dict[str, Any]:
        """Handle single agent case."""
        step = self.add_step(execution, agent.id, "Execute task", task.to_dict())

        prompt = f"""Complete this task:

## Task: {task.title}

## Objective:
{task.objective}

## Description:
{task.description}

## Success Criteria:
{chr(10).join(f"- {c}" for c in task.success_criteria) if task.success_criteria else "- Complete the task effectively"}

Provide a complete solution."""

        result = await self.execute_agent_step(agent, prompt, pool=pool)
        self.complete_step(step, result.get("output"), result.get("tokens", 0))
        self.complete_execution(execution, result.get("success", False), result.get("output"))

        return {
            "success": result.get("success", False),
            "output": result.get("output"),
            "workflow": "hierarchical",
            "total_tokens": result.get("tokens", 0),
        }

    async def _manager_decompose(
        self,
        manager,
        task: TeamTask,
        worker_ids: list[str],
        pool: "AgentPool",
    ) -> list[dict]:
        """Manager decomposes task into subtasks."""
        workers_info = []
        for wid in worker_ids:
            worker = pool.get_agent(wid)
            if worker:
                workers_info.append({
                    "id": wid,
                    "role": worker.profile.role.value,
                    "capabilities": [c.name for c in worker.profile.capabilities],
                })

        prompt = f"""You are the team manager. Break down this task into subtasks for your team.

## Task: {task.title}

## Objective:
{task.objective}

## Description:
{task.description}

## Success Criteria:
{chr(10).join(f"- {c}" for c in task.success_criteria) if task.success_criteria else "- Complete the task"}

## Available Workers:
{json.dumps(workers_info, indent=2)}

Create subtasks that leverage each worker's strengths. Respond with a JSON array:
```json
[
    {{"worker_id": "...", "subtask": "...", "instructions": "..."}}
]
```

Assign one subtask per worker. Make subtasks specific and actionable."""

        result = await self.execute_agent_step(
            manager,
            prompt,
            "You are a skilled project manager who delegates effectively.",
            pool=pool,
        )

        # Parse subtasks from response
        try:
            text = result.get("output", "")
            # Find JSON in response
            start = text.find('[')
            end = text.rfind(']') + 1
            if start >= 0 and end > start:
                subtasks = json.loads(text[start:end])
                return subtasks
        except (json.JSONDecodeError, Exception) as e:
            logger.warning("Failed to parse subtasks", error=str(e))

        # Fallback: distribute task evenly
        return [
            {"worker_id": wid, "subtask": task.title, "instructions": task.description}
            for wid in worker_ids
        ]

    async def _execute_workers(
        self,
        subtasks: list[dict],
        worker_ids: list[str],
        task: TeamTask,
        team: Team,
        pool: "AgentPool",
        bus: "MessageBus",
        execution,
    ) -> list[dict]:
        """Execute worker subtasks in parallel."""

        async def execute_worker(subtask: dict) -> dict:
            worker_id = subtask.get("worker_id")
            worker = pool.get_agent(worker_id)

            if not worker:
                return {"worker_id": worker_id, "error": "Worker not found"}

            step = self.add_step(
                execution,
                worker_id,
                f"Execute subtask: {subtask.get('subtask', '')[:30]}",
                subtask,
            )

            pool.update_status(worker_id, AgentStatus.BUSY, task.id)

            try:
                # Notify worker
                await bus.send(
                    sender_id="manager",
                    recipient_id=worker_id,
                    message_type=MessageType.TASK,
                    content=subtask,
                    subject=f"Subtask: {subtask.get('subtask', '')}",
                    team_id=team.id,
                )

                result = await self._execute_worker_subtask(worker, subtask, pool)

                self.complete_step(step, result, result.get("tokens", 0))

                return {
                    "worker_id": worker_id,
                    "subtask": subtask.get("subtask"),
                    "result": result.get("output"),
                    "success": result.get("success", False),
                }

            except Exception as e:
                self.complete_step(step, None, 0, str(e))
                return {
                    "worker_id": worker_id,
                    "subtask": subtask.get("subtask"),
                    "error": str(e),
                    "success": False,
                }

            finally:
                pool.update_status(worker_id, AgentStatus.IDLE)

        # Run all workers in parallel
        results = await asyncio.gather(*[execute_worker(st) for st in subtasks])
        return list(results)

    async def _execute_worker_subtask(
        self,
        worker,
        subtask: dict,
        pool: "AgentPool",
    ) -> dict[str, Any]:
        """Execute a single worker's subtask."""
        prompt = f"""Complete this subtask assigned to you:

## Subtask: {subtask.get('subtask')}

## Instructions:
{subtask.get('instructions')}

As a {worker.profile.role.value}, provide your work output clearly and thoroughly."""

        return await self.execute_agent_step(worker, prompt, pool=pool)

    async def _manager_synthesize(
        self,
        manager,
        task: TeamTask,
        worker_results: list[dict],
        pool: "AgentPool",
    ) -> dict:
        """Manager synthesizes worker results."""
        # Format worker results
        results_text = ""
        for wr in worker_results:
            if wr.get("success"):
                results_text += f"\n### Worker: {wr.get('subtask', 'Task')}\n"
                results_text += f"{wr.get('result', 'No output')}\n"
            else:
                results_text += f"\n### Worker (FAILED): {wr.get('subtask', 'Task')}\n"
                results_text += f"Error: {wr.get('error', 'Unknown error')}\n"

        prompt = f"""As team manager, synthesize your workers' results into a final deliverable.

## Original Task: {task.title}

## Objective:
{task.objective}

## Success Criteria:
{chr(10).join(f"- {c}" for c in task.success_criteria) if task.success_criteria else "- Complete the task"}

## Worker Results:
{results_text}

Create the final integrated output that:
1. Combines all successful worker contributions
2. Addresses any gaps from failed subtasks
3. Achieves the original objective
4. Meets all success criteria"""

        result = await self.execute_agent_step(
            manager,
            prompt,
            "You are a skilled manager who synthesizes team work into cohesive deliverables.",
            pool=pool,
        )

        return {
            "synthesis": result.get("output"),
            "worker_contributions": worker_results,
            "success": result.get("success", False),
        }
