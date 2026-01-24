"""
AION Team Manager

Forms and coordinates agent teams for collaborative task execution.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Optional, TYPE_CHECKING

import structlog

from aion.systems.agents.types import (
    Team,
    TeamTask,
    TeamStatus,
    AgentRole,
    AgentStatus,
    WorkflowPattern,
    Message,
    MessageType,
)

if TYPE_CHECKING:
    from aion.systems.agents.pool import AgentPool
    from aion.systems.agents.messaging import MessageBus

logger = structlog.get_logger(__name__)


# Task type to role mapping
TASK_ROLE_MAPPING = {
    "research": AgentRole.RESEARCHER,
    "search": AgentRole.RESEARCHER,
    "find": AgentRole.RESEARCHER,
    "gather": AgentRole.RESEARCHER,
    "investigate": AgentRole.RESEARCHER,
    "code": AgentRole.CODER,
    "implement": AgentRole.CODER,
    "build": AgentRole.CODER,
    "develop": AgentRole.CODER,
    "program": AgentRole.CODER,
    "debug": AgentRole.CODER,
    "fix": AgentRole.CODER,
    "analyze": AgentRole.ANALYST,
    "data": AgentRole.ANALYST,
    "statistics": AgentRole.ANALYST,
    "metrics": AgentRole.ANALYST,
    "report": AgentRole.ANALYST,
    "visualize": AgentRole.ANALYST,
    "write": AgentRole.WRITER,
    "document": AgentRole.WRITER,
    "content": AgentRole.WRITER,
    "article": AgentRole.WRITER,
    "draft": AgentRole.WRITER,
    "review": AgentRole.REVIEWER,
    "check": AgentRole.REVIEWER,
    "verify": AgentRole.REVIEWER,
    "validate": AgentRole.REVIEWER,
    "quality": AgentRole.REVIEWER,
    "plan": AgentRole.PLANNER,
    "strategy": AgentRole.PLANNER,
    "organize": AgentRole.PLANNER,
    "coordinate": AgentRole.PLANNER,
    "execute": AgentRole.EXECUTOR,
    "run": AgentRole.EXECUTOR,
    "perform": AgentRole.EXECUTOR,
}


class TeamManager:
    """
    Manages agent teams and their workflows.

    Features:
    - Team formation based on task requirements
    - Automatic role-based agent selection
    - Workflow execution coordination
    - Team lifecycle management
    - Shared context management
    """

    def __init__(
        self,
        agent_pool: "AgentPool",
        message_bus: "MessageBus",
    ):
        self.pool = agent_pool
        self.bus = message_bus

        # Active teams
        self._teams: dict[str, Team] = {}

        # Task tracking
        self._tasks: dict[str, TeamTask] = {}

        # Statistics
        self._total_teams_created: int = 0
        self._total_teams_disbanded: int = 0

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the team manager."""
        if self._initialized:
            return

        logger.info("Initializing Team Manager")
        self._initialized = True

    async def shutdown(self) -> None:
        """Shutdown and disband all teams."""
        logger.info("Shutting down Team Manager", active_teams=len(self._teams))

        for team_id in list(self._teams.keys()):
            await self.disband_team(team_id)

        self._initialized = False

    # === Team Formation ===

    async def create_team(
        self,
        name: str,
        purpose: str,
        roles: list[AgentRole],
        workflow: WorkflowPattern = WorkflowPattern.SEQUENTIAL,
        leader_role: Optional[AgentRole] = None,
        spawn_if_needed: bool = True,
    ) -> Team:
        """
        Create a new team with specified roles.

        Args:
            name: Team name
            purpose: Team's purpose/objective
            roles: Roles needed for the team
            workflow: Workflow pattern to use
            leader_role: Role that leads the team
            spawn_if_needed: Spawn new agents if needed

        Returns:
            Created team
        """
        team = Team(
            name=name,
            purpose=purpose,
            workflow_pattern=workflow,
        )

        # Assign or spawn agents for each role
        for role in roles:
            agent = await self._get_or_spawn_agent(role, team.id, spawn_if_needed)

            if agent:
                team.agent_ids.append(agent.id)
                agent.current_team_id = team.id

                # Register with message bus
                self.bus.register_team_member(team.id, agent.id)

                # Set leader
                if leader_role and role == leader_role:
                    team.leader_id = agent.id

        # If no leader specified, use first agent
        if not team.leader_id and team.agent_ids:
            team.leader_id = team.agent_ids[0]

        team.status = TeamStatus.ACTIVE
        self._teams[team.id] = team
        self._total_teams_created += 1

        logger.info(
            "Created team",
            team_id=team.id[:8],
            name=name,
            agents=len(team.agent_ids),
            workflow=workflow.value,
        )

        return team

    async def _get_or_spawn_agent(
        self,
        role: AgentRole,
        team_id: str,
        spawn_if_needed: bool,
    ):
        """Get an existing agent or spawn a new one."""
        # Try to find an idle agent with this role
        agents = self.pool.find_agents_by_role(role, prefer_idle=True)

        for agent in agents:
            if agent.status == AgentStatus.IDLE and not agent.current_team_id:
                return agent

        # Spawn new agent if needed and allowed
        if spawn_if_needed:
            try:
                return await self.pool.spawn_agent(role=role)
            except RuntimeError:
                logger.warning("Cannot spawn agent", role=role.value)
                return None

        return None

    async def create_team_for_task(
        self,
        task: TeamTask,
        spawn_if_needed: bool = True,
    ) -> Team:
        """
        Automatically create an optimal team for a task.

        Analyzes the task and selects appropriate roles.

        Args:
            task: The task to create a team for
            spawn_if_needed: Spawn new agents if needed

        Returns:
            Created team
        """
        # Determine required roles based on task
        roles = self._analyze_task_requirements(task)

        # Determine leader role
        leader_role = AgentRole.PLANNER if AgentRole.PLANNER in roles else None

        # Create team
        team = await self.create_team(
            name=f"Team: {task.title[:30]}",
            purpose=task.objective,
            roles=roles,
            workflow=task.workflow_pattern,
            leader_role=leader_role,
            spawn_if_needed=spawn_if_needed,
        )

        # Link task and team
        task.assigned_team_id = team.id
        task.assigned_agent_ids = team.agent_ids.copy()
        team.task_id = task.id
        self._tasks[task.id] = task

        return team

    def _analyze_task_requirements(self, task: TeamTask) -> list[AgentRole]:
        """Analyze task to determine required roles."""
        roles: list[AgentRole] = []
        seen_roles: set[AgentRole] = set()

        # Combine task text for analysis
        task_text = f"{task.title} {task.description} {task.objective}".lower()

        # Check for keywords that indicate needed roles
        for keyword, role in TASK_ROLE_MAPPING.items():
            if keyword in task_text and role not in seen_roles:
                roles.append(role)
                seen_roles.add(role)

        # Always have at least one role
        if not roles:
            roles.append(AgentRole.GENERALIST)

        # Add planner for complex tasks (more than 2 roles)
        if len(roles) > 2 and AgentRole.PLANNER not in seen_roles:
            roles.insert(0, AgentRole.PLANNER)

        # Add reviewer for tasks mentioning quality
        if any(w in task_text for w in ["quality", "correct", "accurate", "reliable"]):
            if AgentRole.REVIEWER not in seen_roles:
                roles.append(AgentRole.REVIEWER)

        return roles

    # === Team Operations ===

    async def assign_task(
        self,
        team_id: str,
        task: TeamTask,
    ) -> bool:
        """
        Assign a task to an existing team.

        Args:
            team_id: Team ID
            task: Task to assign

        Returns:
            True if assigned successfully
        """
        team = self._teams.get(team_id)
        if not team:
            logger.warning("Team not found", team_id=team_id[:8])
            return False

        if team.status != TeamStatus.ACTIVE:
            logger.warning("Team not active", team_id=team_id[:8], status=team.status.value)
            return False

        # Link task and team
        task.assigned_team_id = team_id
        task.assigned_agent_ids = team.agent_ids.copy()
        team.task_id = task.id
        self._tasks[task.id] = task

        # Notify team via broadcast
        await self.bus.broadcast(
            sender_id="orchestrator",
            content={
                "type": "task_assigned",
                "task": task.to_dict(),
            },
            subject=f"New task: {task.title}",
            team_id=team_id,
        )

        logger.info(
            "Task assigned to team",
            task_id=task.id[:8],
            team_id=team_id[:8],
        )

        return True

    async def execute_task(
        self,
        team_id: str,
        task: TeamTask,
    ) -> dict[str, Any]:
        """
        Execute a task with a team.

        Args:
            team_id: Team ID
            task: Task to execute

        Returns:
            Task execution result
        """
        team = self._teams.get(team_id)
        if not team:
            return {"success": False, "error": f"Team not found: {team_id}"}

        # Update task status
        task.mark_started()

        # Get workflow executor
        from aion.systems.agents.workflows.base import get_workflow_executor
        executor = get_workflow_executor(team.workflow_pattern)

        try:
            # Execute workflow
            result = await executor.execute(
                team=team,
                task=task,
                pool=self.pool,
                bus=self.bus,
            )

            # Update task based on result
            if result.get("success"):
                task.mark_completed(result.get("output"))
                team.tasks_completed += 1
            else:
                task.mark_failed(result.get("error", "Unknown error"))
                team.tasks_failed += 1

            return result

        except Exception as e:
            logger.error("Task execution error", error=str(e), task_id=task.id[:8])
            task.mark_failed(str(e))
            team.tasks_failed += 1
            return {"success": False, "error": str(e)}

    async def add_agent_to_team(
        self,
        team_id: str,
        agent_id: str,
    ) -> bool:
        """Add an agent to an existing team."""
        team = self._teams.get(team_id)
        agent = self.pool.get_agent(agent_id)

        if not team or not agent:
            return False

        if agent_id not in team.agent_ids:
            team.add_agent(agent_id)
            agent.current_team_id = team_id
            self.bus.register_team_member(team_id, agent_id)

            # Notify team
            await self.bus.broadcast(
                sender_id="orchestrator",
                content={
                    "type": "agent_joined",
                    "agent_id": agent_id,
                    "agent_role": agent.profile.role.value,
                },
                subject="Agent joined team",
                team_id=team_id,
            )

        return True

    async def remove_agent_from_team(
        self,
        team_id: str,
        agent_id: str,
    ) -> bool:
        """Remove an agent from a team."""
        team = self._teams.get(team_id)
        agent = self.pool.get_agent(agent_id)

        if not team:
            return False

        if agent_id in team.agent_ids:
            team.remove_agent(agent_id)
            self.bus.unregister_team_member(team_id, agent_id)

            if agent:
                agent.current_team_id = None
                self.pool.update_status(agent_id, AgentStatus.IDLE)

            # Notify team
            await self.bus.broadcast(
                sender_id="orchestrator",
                content={
                    "type": "agent_left",
                    "agent_id": agent_id,
                },
                subject="Agent left team",
                team_id=team_id,
            )

        return True

    async def disband_team(self, team_id: str) -> bool:
        """
        Disband a team and release agents.

        Args:
            team_id: Team ID

        Returns:
            True if disbanded
        """
        team = self._teams.pop(team_id, None)
        if not team:
            return False

        # Release agents
        for agent_id in team.agent_ids:
            agent = self.pool.get_agent(agent_id)
            if agent:
                agent.current_team_id = None
                self.pool.update_status(agent_id, AgentStatus.IDLE)

        # Clean up message bus
        self.bus.disband_team(team_id)

        # Update team status
        team.status = TeamStatus.DISBANDED
        team.disbanded_at = datetime.now()

        self._total_teams_disbanded += 1

        logger.info(
            "Disbanded team",
            team_id=team_id[:8],
            name=team.name,
            tasks_completed=team.tasks_completed,
        )

        return True

    # === Team Context ===

    def update_shared_context(
        self,
        team_id: str,
        key: str,
        value: Any,
    ) -> bool:
        """Update team's shared context."""
        team = self._teams.get(team_id)
        if not team:
            return False

        team.shared_context[key] = value
        return True

    def get_shared_context(
        self,
        team_id: str,
        key: Optional[str] = None,
    ) -> Any:
        """Get team's shared context."""
        team = self._teams.get(team_id)
        if not team:
            return None

        if key:
            return team.shared_context.get(key)
        return team.shared_context.copy()

    # === Team Access ===

    def get_team(self, team_id: str) -> Optional[Team]:
        """Get a team by ID."""
        return self._teams.get(team_id)

    def get_task(self, task_id: str) -> Optional[TeamTask]:
        """Get a task by ID."""
        return self._tasks.get(task_id)

    def list_teams(
        self,
        status: Optional[TeamStatus] = None,
    ) -> list[Team]:
        """List teams with optional filter."""
        teams = list(self._teams.values())

        if status is not None:
            teams = [t for t in teams if t.status == status]

        return teams

    def get_team_for_agent(self, agent_id: str) -> Optional[Team]:
        """Get the team an agent belongs to."""
        agent = self.pool.get_agent(agent_id)
        if agent and agent.current_team_id:
            return self._teams.get(agent.current_team_id)
        return None

    # === Statistics ===

    def get_stats(self) -> dict[str, Any]:
        """Get team manager statistics."""
        status_counts: dict[str, int] = {}
        for status in TeamStatus:
            status_counts[status.value] = len(self.list_teams(status=status))

        total_tasks_completed = sum(t.tasks_completed for t in self._teams.values())
        total_tasks_failed = sum(t.tasks_failed for t in self._teams.values())

        return {
            "total_teams": len(self._teams),
            "by_status": status_counts,
            "total_created": self._total_teams_created,
            "total_disbanded": self._total_teams_disbanded,
            "total_tasks": len(self._tasks),
            "tasks_completed": total_tasks_completed,
            "tasks_failed": total_tasks_failed,
        }
