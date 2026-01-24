"""
AION Task Delegator

Intelligent task routing and delegation to appropriate agents.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional, TYPE_CHECKING

import structlog

from aion.systems.agents.types import (
    TeamTask,
    AgentRole,
    AgentStatus,
    WorkflowPattern,
    TaskPriority,
)

if TYPE_CHECKING:
    from aion.systems.agents.pool import AgentPool
    from aion.systems.agents.team import TeamManager

logger = structlog.get_logger(__name__)


# Keywords for task type detection
TASK_KEYWORDS = {
    "research": ["research", "find", "search", "investigate", "lookup", "gather", "explore"],
    "code": ["code", "implement", "build", "develop", "program", "script", "function", "class"],
    "debug": ["debug", "fix", "troubleshoot", "diagnose", "error", "bug", "issue"],
    "analyze": ["analyze", "examine", "study", "evaluate", "assess", "measure", "data"],
    "write": ["write", "draft", "compose", "document", "content", "article", "text"],
    "review": ["review", "check", "verify", "validate", "test", "quality", "audit"],
    "plan": ["plan", "strategy", "organize", "design", "architect", "structure"],
    "execute": ["execute", "run", "perform", "do", "complete", "finish"],
}

# Role priority for task types
ROLE_PRIORITY = {
    "research": [AgentRole.RESEARCHER, AgentRole.ANALYST, AgentRole.GENERALIST],
    "code": [AgentRole.CODER, AgentRole.EXECUTOR, AgentRole.GENERALIST],
    "debug": [AgentRole.CODER, AgentRole.ANALYST, AgentRole.GENERALIST],
    "analyze": [AgentRole.ANALYST, AgentRole.RESEARCHER, AgentRole.GENERALIST],
    "write": [AgentRole.WRITER, AgentRole.RESEARCHER, AgentRole.GENERALIST],
    "review": [AgentRole.REVIEWER, AgentRole.ANALYST, AgentRole.GENERALIST],
    "plan": [AgentRole.PLANNER, AgentRole.ANALYST, AgentRole.GENERALIST],
    "execute": [AgentRole.EXECUTOR, AgentRole.CODER, AgentRole.GENERALIST],
}


class TaskDelegator:
    """
    Intelligent task delegation and routing.

    Features:
    - Task type detection from description
    - Capability-based agent matching
    - Load balancing across agents
    - Workflow pattern recommendation
    - Priority-based scheduling
    """

    def __init__(
        self,
        agent_pool: "AgentPool",
        team_manager: "TeamManager",
    ):
        self.pool = agent_pool
        self.teams = team_manager

        # Delegation history for learning
        self._delegation_history: list[dict] = []

        # Statistics
        self._delegations: int = 0
        self._successful_delegations: int = 0

    def detect_task_type(self, task: TeamTask) -> str:
        """
        Detect the primary type of a task.

        Args:
            task: Task to analyze

        Returns:
            Detected task type
        """
        text = f"{task.title} {task.description} {task.objective}".lower()

        # Count keyword matches for each type
        type_scores: dict[str, int] = {}

        for task_type, keywords in TASK_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text)
            if score > 0:
                type_scores[task_type] = score

        if not type_scores:
            return "general"

        # Return type with highest score
        return max(type_scores, key=type_scores.get)

    def recommend_roles(
        self,
        task: TeamTask,
        max_roles: int = 5,
    ) -> list[AgentRole]:
        """
        Recommend roles for a task.

        Args:
            task: Task to analyze
            max_roles: Maximum roles to recommend

        Returns:
            List of recommended roles
        """
        task_type = self.detect_task_type(task)
        text = f"{task.title} {task.description} {task.objective}".lower()

        roles: list[AgentRole] = []
        seen: set[AgentRole] = set()

        # Get primary roles for detected type
        primary_roles = ROLE_PRIORITY.get(task_type, [AgentRole.GENERALIST])
        for role in primary_roles:
            if role not in seen and len(roles) < max_roles:
                roles.append(role)
                seen.add(role)

        # Check for additional types
        for other_type, keywords in TASK_KEYWORDS.items():
            if other_type != task_type and any(kw in text for kw in keywords):
                for role in ROLE_PRIORITY.get(other_type, []):
                    if role not in seen and len(roles) < max_roles:
                        roles.append(role)
                        seen.add(role)

        # Add planner for complex tasks
        if len(roles) > 2 and AgentRole.PLANNER not in seen:
            roles.insert(0, AgentRole.PLANNER)
            if len(roles) > max_roles:
                roles = roles[:max_roles]

        return roles if roles else [AgentRole.GENERALIST]

    def recommend_workflow(
        self,
        task: TeamTask,
        roles: list[AgentRole],
    ) -> WorkflowPattern:
        """
        Recommend a workflow pattern for a task.

        Args:
            task: Task to analyze
            roles: Assigned roles

        Returns:
            Recommended workflow pattern
        """
        text = f"{task.title} {task.description} {task.objective}".lower()

        # Check for specific patterns
        if any(w in text for w in ["debate", "discuss", "compare", "contrast", "argue"]):
            return WorkflowPattern.DEBATE

        if any(w in text for w in ["parallel", "concurrent", "simultaneous", "together"]):
            return WorkflowPattern.PARALLEL

        if any(w in text for w in ["consensus", "agree", "vote", "decide"]):
            return WorkflowPattern.CONSENSUS

        if AgentRole.PLANNER in roles or len(roles) > 3:
            return WorkflowPattern.HIERARCHICAL

        if any(w in text for w in ["swarm", "emergent", "collective"]):
            return WorkflowPattern.SWARM

        # Default based on role count
        if len(roles) <= 2:
            return WorkflowPattern.SEQUENTIAL
        else:
            return WorkflowPattern.HIERARCHICAL

    def find_best_agent(
        self,
        task: TeamTask,
        exclude_ids: Optional[list[str]] = None,
    ) -> Optional[str]:
        """
        Find the best agent for a task.

        Args:
            task: Task to delegate
            exclude_ids: Agents to exclude

        Returns:
            Best agent ID or None
        """
        task_type = self.detect_task_type(task)
        roles = ROLE_PRIORITY.get(task_type, [AgentRole.GENERALIST])

        # Try each role in priority order
        for role in roles:
            agents = self.pool.find_agents_by_role(role, prefer_idle=True)

            for agent in agents:
                if exclude_ids and agent.id in exclude_ids:
                    continue

                if agent.status == AgentStatus.IDLE:
                    return agent.id

        # Fall back to any capable agent
        agent = self.pool.find_capable_agent(
            task_type,
            prefer_idle=True,
            exclude_ids=exclude_ids,
        )

        return agent.id if agent else None

    async def delegate_task(
        self,
        task: TeamTask,
        create_team: bool = True,
        spawn_if_needed: bool = True,
    ) -> dict[str, Any]:
        """
        Delegate a task to appropriate agent(s).

        Args:
            task: Task to delegate
            create_team: Create a team if needed
            spawn_if_needed: Spawn agents if needed

        Returns:
            Delegation result with team/agent info
        """
        self._delegations += 1

        # Analyze task
        task_type = self.detect_task_type(task)
        roles = self.recommend_roles(task)
        workflow = self.recommend_workflow(task, roles)

        logger.info(
            "Delegating task",
            task_id=task.id[:8],
            task_type=task_type,
            roles=[r.value for r in roles],
            workflow=workflow.value,
        )

        if create_team and len(roles) > 1:
            # Create a team
            task.workflow_pattern = workflow
            team = await self.teams.create_team_for_task(
                task,
                spawn_if_needed=spawn_if_needed,
            )

            self._successful_delegations += 1

            # Record delegation
            self._record_delegation(task, team.id, None, roles, workflow)

            return {
                "success": True,
                "type": "team",
                "team_id": team.id,
                "agent_ids": team.agent_ids,
                "workflow": workflow.value,
                "roles": [r.value for r in roles],
            }

        else:
            # Find single agent
            agent_id = self.find_best_agent(task)

            if not agent_id and spawn_if_needed:
                # Spawn an agent
                try:
                    agent = await self.pool.spawn_agent(role=roles[0])
                    agent_id = agent.id
                except RuntimeError:
                    pass

            if agent_id:
                self._successful_delegations += 1
                task.assigned_agent_ids = [agent_id]

                # Record delegation
                self._record_delegation(task, None, agent_id, roles[:1], workflow)

                return {
                    "success": True,
                    "type": "agent",
                    "agent_id": agent_id,
                    "roles": [roles[0].value],
                }

            return {
                "success": False,
                "error": "No suitable agent available",
            }

    def _record_delegation(
        self,
        task: TeamTask,
        team_id: Optional[str],
        agent_id: Optional[str],
        roles: list[AgentRole],
        workflow: WorkflowPattern,
    ) -> None:
        """Record delegation for analysis."""
        self._delegation_history.append({
            "task_id": task.id,
            "task_type": self.detect_task_type(task),
            "team_id": team_id,
            "agent_id": agent_id,
            "roles": [r.value for r in roles],
            "workflow": workflow.value,
            "timestamp": datetime.now().isoformat(),
        })

        # Keep history bounded
        if len(self._delegation_history) > 1000:
            self._delegation_history = self._delegation_history[-500:]

    def estimate_complexity(self, task: TeamTask) -> dict[str, Any]:
        """
        Estimate task complexity.

        Args:
            task: Task to analyze

        Returns:
            Complexity assessment
        """
        text = f"{task.title} {task.description} {task.objective}".lower()

        # Complexity indicators
        indicators = {
            "multi_step": any(w in text for w in ["steps", "phases", "stages", "first then"]),
            "research_needed": any(w in text for w in ["research", "investigate", "find out"]),
            "code_involved": any(w in text for w in ["code", "implement", "build"]),
            "analysis_needed": any(w in text for w in ["analyze", "evaluate", "compare"]),
            "review_required": any(w in text for w in ["review", "verify", "quality"]),
            "collaboration": any(w in text for w in ["collaborate", "together", "team"]),
        }

        complexity_score = sum(indicators.values())

        if complexity_score >= 4:
            level = "high"
            recommended_agents = 3
        elif complexity_score >= 2:
            level = "medium"
            recommended_agents = 2
        else:
            level = "low"
            recommended_agents = 1

        return {
            "level": level,
            "score": complexity_score,
            "indicators": indicators,
            "recommended_agents": recommended_agents,
            "recommended_workflow": self.recommend_workflow(
                task, self.recommend_roles(task)
            ).value,
        }

    def get_stats(self) -> dict[str, Any]:
        """Get delegator statistics."""
        return {
            "total_delegations": self._delegations,
            "successful_delegations": self._successful_delegations,
            "success_rate": (
                self._successful_delegations / self._delegations
                if self._delegations > 0 else 0
            ),
            "history_size": len(self._delegation_history),
        }
