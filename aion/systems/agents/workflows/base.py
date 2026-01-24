"""
AION Workflow Base

Abstract base class for multi-agent workflow executors.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Optional, TYPE_CHECKING

import structlog

from aion.systems.agents.types import (
    Team,
    TeamTask,
    WorkflowPattern,
    WorkflowStep,
    WorkflowExecution,
    AgentStatus,
)

if TYPE_CHECKING:
    from aion.systems.agents.pool import AgentPool
    from aion.systems.agents.messaging import MessageBus

logger = structlog.get_logger(__name__)


class WorkflowExecutor(ABC):
    """
    Abstract base for workflow executors.

    Workflow patterns define how agents collaborate:
    - Sequential: One agent after another, building on previous work
    - Parallel: All agents work simultaneously
    - Hierarchical: Manager delegates to workers
    - Debate: Agents refine through discussion
    - Swarm: Emergent coordination through communication
    """

    def __init__(self):
        self._executions: dict[str, WorkflowExecution] = {}

    @property
    @abstractmethod
    def pattern(self) -> WorkflowPattern:
        """Get the workflow pattern this executor implements."""
        pass

    @abstractmethod
    async def execute(
        self,
        team: Team,
        task: TeamTask,
        pool: "AgentPool",
        bus: "MessageBus",
    ) -> dict[str, Any]:
        """
        Execute the workflow.

        Args:
            team: The team executing the workflow
            task: The task to execute
            pool: Agent pool for accessing agents
            bus: Message bus for communication

        Returns:
            Workflow result with success status and output
        """
        pass

    def create_execution(
        self,
        team: Team,
        task: TeamTask,
    ) -> WorkflowExecution:
        """Create a new workflow execution record."""
        execution = WorkflowExecution(
            workflow_pattern=self.pattern,
            team_id=team.id,
            task_id=task.id,
            status="active",
            started_at=datetime.now(),
        )
        self._executions[execution.id] = execution
        return execution

    def complete_execution(
        self,
        execution: WorkflowExecution,
        success: bool,
        output: Any = None,
        error: Optional[str] = None,
    ) -> None:
        """Mark execution as complete."""
        execution.status = "completed" if success else "failed"
        execution.completed_at = datetime.now()
        execution.final_output = output
        execution.error = error

        if execution.started_at:
            execution.total_duration_seconds = (
                execution.completed_at - execution.started_at
            ).total_seconds()

        execution.total_tokens_used = sum(s.tokens_used for s in execution.steps)

    def add_step(
        self,
        execution: WorkflowExecution,
        agent_id: str,
        action: str,
        input_data: Any = None,
    ) -> WorkflowStep:
        """Add a step to the execution."""
        step = WorkflowStep(
            agent_id=agent_id,
            action=action,
            input_data=input_data,
            status="active",
            started_at=datetime.now(),
        )
        execution.steps.append(step)
        execution.current_step_index = len(execution.steps) - 1
        return step

    def complete_step(
        self,
        step: WorkflowStep,
        output: Any,
        tokens_used: int = 0,
        error: Optional[str] = None,
    ) -> None:
        """Mark a step as complete."""
        step.completed_at = datetime.now()
        step.output_data = output
        step.tokens_used = tokens_used

        if error:
            step.status = "failed"
            step.error = error
        else:
            step.status = "completed"

        if step.started_at:
            step.duration_seconds = (
                step.completed_at - step.started_at
            ).total_seconds()

    async def execute_agent_step(
        self,
        agent,
        prompt: str,
        system_prompt: Optional[str] = None,
        pool: Optional["AgentPool"] = None,
    ) -> dict[str, Any]:
        """
        Execute a single agent step using LLM.

        Args:
            agent: Agent instance
            prompt: The prompt for the agent
            system_prompt: Override system prompt
            pool: Agent pool for status updates

        Returns:
            Agent response with output and token usage
        """
        from aion.conversation.llm.claude import ClaudeProvider

        # Update status
        if pool:
            pool.update_status(agent.id, AgentStatus.BUSY)

        try:
            llm = ClaudeProvider()
            await llm.initialize()

            response = await llm.complete(
                messages=[{"role": "user", "content": prompt}],
                system=system_prompt or agent.profile.system_prompt,
            )

            output = ""
            for block in response.content:
                if hasattr(block, "text"):
                    output += block.text

            tokens = response.input_tokens + response.output_tokens

            return {
                "success": True,
                "output": output,
                "tokens": tokens,
            }

        except Exception as e:
            logger.error("Agent step error", agent_id=agent.id[:8], error=str(e))
            return {
                "success": False,
                "output": None,
                "error": str(e),
                "tokens": 0,
            }

        finally:
            if pool:
                pool.update_status(agent.id, AgentStatus.IDLE)


def get_workflow_executor(pattern: WorkflowPattern) -> WorkflowExecutor:
    """
    Get the appropriate workflow executor for a pattern.

    Args:
        pattern: Workflow pattern

    Returns:
        Workflow executor instance
    """
    from aion.systems.agents.workflows.sequential import SequentialWorkflow
    from aion.systems.agents.workflows.parallel import ParallelWorkflow
    from aion.systems.agents.workflows.hierarchical import HierarchicalWorkflow
    from aion.systems.agents.workflows.debate import DebateWorkflow
    from aion.systems.agents.workflows.swarm import SwarmWorkflow

    executors = {
        WorkflowPattern.SEQUENTIAL: SequentialWorkflow,
        WorkflowPattern.PARALLEL: ParallelWorkflow,
        WorkflowPattern.HIERARCHICAL: HierarchicalWorkflow,
        WorkflowPattern.DEBATE: DebateWorkflow,
        WorkflowPattern.SWARM: SwarmWorkflow,
        WorkflowPattern.PIPELINE: SequentialWorkflow,  # Alias
        WorkflowPattern.CONSENSUS: DebateWorkflow,  # Alias
    }

    executor_class = executors.get(pattern, SequentialWorkflow)
    return executor_class()
