"""
AION Multi-Agent Orchestrator

Central coordinator for multi-agent operations.
Provides high-level API for team formation, task execution, and coordination.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Optional

import structlog

from aion.systems.agents.types import (
    Team,
    TeamTask,
    TeamStatus,
    AgentRole,
    AgentStatus,
    WorkflowPattern,
    AgentProfile,
    OrchestratorStats,
)
from aion.systems.agents.pool import AgentPool
from aion.systems.agents.messaging import MessageBus
from aion.systems.agents.team import TeamManager
from aion.systems.agents.delegation import TaskDelegator
from aion.systems.agents.consensus import ConsensusEngine

logger = structlog.get_logger(__name__)


class MultiAgentOrchestrator:
    """
    Central orchestrator for multi-agent operations.

    Provides high-level API for:
    - Creating and managing agent teams
    - Executing multi-agent tasks
    - Coordinating agent communication
    - Consensus decisions
    - Common task patterns (research, code, analyze, etc.)
    """

    def __init__(
        self,
        max_agents: int = 20,
        max_teams: int = 10,
    ):
        self.max_agents = max_agents
        self.max_teams = max_teams

        # Core components
        self.pool = AgentPool(max_agents=max_agents)
        self.bus = MessageBus()
        self.teams = TeamManager(self.pool, self.bus)
        self.delegator = TaskDelegator(self.pool, self.teams)
        self.consensus = ConsensusEngine(self.bus)

        # Task tracking
        self._active_tasks: dict[str, TeamTask] = {}

        # Statistics
        self._started_at: Optional[datetime] = None
        self._total_tasks_executed: int = 0
        self._total_tasks_succeeded: int = 0

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the orchestrator and all components."""
        if self._initialized:
            return

        logger.info("Initializing Multi-Agent Orchestrator")

        await self.pool.initialize()
        await self.bus.initialize()
        await self.teams.initialize()

        self._started_at = datetime.now()
        self._initialized = True

        logger.info("Multi-Agent Orchestrator initialized")

    async def shutdown(self) -> None:
        """Shutdown the orchestrator and all components."""
        logger.info("Shutting down Multi-Agent Orchestrator")

        # Disband all teams
        for team in self.teams.list_teams():
            await self.teams.disband_team(team.id)

        # Shutdown components
        await self.bus.shutdown()
        await self.pool.shutdown()

        self._initialized = False
        logger.info("Multi-Agent Orchestrator shutdown complete")

    # === High-Level Task Execution ===

    async def execute_task(
        self,
        title: str,
        description: str,
        objective: str,
        success_criteria: Optional[list[str]] = None,
        workflow: WorkflowPattern = WorkflowPattern.SEQUENTIAL,
        roles: Optional[list[AgentRole]] = None,
        max_iterations: int = 10,
        timeout: Optional[float] = None,
    ) -> dict[str, Any]:
        """
        Execute a task with an automatically formed team.

        Args:
            title: Task title
            description: Task description
            objective: What success looks like
            success_criteria: Measurable criteria
            workflow: Workflow pattern to use
            roles: Specific roles to include (auto-detect if None)
            max_iterations: Maximum workflow iterations
            timeout: Execution timeout in seconds

        Returns:
            Task result with output and metadata
        """
        if not self._initialized:
            await self.initialize()

        logger.info(
            "Executing task",
            title=title[:50],
            workflow=workflow.value,
        )

        # Create task
        task = TeamTask(
            title=title,
            description=description,
            objective=objective,
            success_criteria=success_criteria or [],
            workflow_pattern=workflow,
            max_iterations=max_iterations,
        )

        self._active_tasks[task.id] = task
        self._total_tasks_executed += 1

        try:
            # Create team
            if roles:
                team = await self.teams.create_team(
                    name=f"Team: {title[:30]}",
                    purpose=objective,
                    roles=roles,
                    workflow=workflow,
                )
            else:
                team = await self.teams.create_team_for_task(task)

            # Execute with optional timeout
            if timeout:
                result = await asyncio.wait_for(
                    self.teams.execute_task(team.id, task),
                    timeout=timeout,
                )
            else:
                result = await self.teams.execute_task(team.id, task)

            if result.get("success"):
                self._total_tasks_succeeded += 1

            return {
                "success": result.get("success", False),
                "output": result.get("output"),
                "team_id": team.id,
                "task_id": task.id,
                "workflow": workflow.value,
                "agents_used": len(team.agent_ids),
                "error": result.get("error"),
            }

        except asyncio.TimeoutError:
            logger.warning("Task execution timeout", task_id=task.id[:8])
            return {
                "success": False,
                "error": "Execution timeout",
                "task_id": task.id,
            }

        except Exception as e:
            logger.error("Task execution error", error=str(e), task_id=task.id[:8])
            return {
                "success": False,
                "error": str(e),
                "task_id": task.id,
            }

        finally:
            # Cleanup
            self._active_tasks.pop(task.id, None)

            # Disband team
            if "team" in dir() and team:
                await self.teams.disband_team(team.id)

    # === Convenience Methods ===

    async def research(
        self,
        topic: str,
        depth: str = "medium",
        questions: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """
        Research a topic using a research team.

        Args:
            topic: Topic to research
            depth: "shallow", "medium", or "deep"
            questions: Specific questions to answer

        Returns:
            Research findings
        """
        roles = [AgentRole.RESEARCHER]

        if depth in ("medium", "deep"):
            roles.append(AgentRole.ANALYST)

        if depth == "deep":
            roles.append(AgentRole.WRITER)
            roles.append(AgentRole.REVIEWER)

        questions_str = ""
        if questions:
            questions_str = "\n\nQuestions to answer:\n" + "\n".join(f"- {q}" for q in questions)

        return await self.execute_task(
            title=f"Research: {topic}",
            description=f"Conduct {depth} research on: {topic}{questions_str}",
            objective=f"Comprehensive understanding of {topic}",
            success_criteria=[
                "Key facts identified",
                "Multiple sources consulted",
                "Findings clearly organized",
                "Confidence levels noted",
            ],
            workflow=WorkflowPattern.SEQUENTIAL,
            roles=roles,
        )

    async def code_task(
        self,
        description: str,
        language: str = "python",
        include_tests: bool = True,
        include_review: bool = True,
    ) -> dict[str, Any]:
        """
        Complete a coding task with a dev team.

        Args:
            description: What to build
            language: Programming language
            include_tests: Include test writing
            include_review: Include code review

        Returns:
            Code and review results
        """
        roles = [AgentRole.CODER]

        if include_review:
            roles.append(AgentRole.REVIEWER)

        workflow = WorkflowPattern.DEBATE if include_review else WorkflowPattern.SEQUENTIAL

        criteria = [
            "Code is correct and functional",
            f"Follows {language} best practices",
            "Code is readable and maintainable",
        ]

        if include_tests:
            criteria.append("Tests included and passing")

        if include_review:
            criteria.append("Review feedback addressed")

        return await self.execute_task(
            title=f"Code: {description[:50]}",
            description=f"Implement in {language}: {description}",
            objective="Working, tested, reviewed code",
            success_criteria=criteria,
            workflow=workflow,
            roles=roles,
        )

    async def analyze_data(
        self,
        data_description: str,
        questions: list[str],
    ) -> dict[str, Any]:
        """
        Analyze data with an analysis team.

        Args:
            data_description: Description of the data
            questions: Questions to answer

        Returns:
            Analysis results
        """
        questions_str = "\n".join(f"- {q}" for q in questions)

        return await self.execute_task(
            title=f"Analyze: {data_description[:50]}",
            description=f"Data: {data_description}\n\nQuestions:\n{questions_str}",
            objective="Data-driven answers to all questions",
            success_criteria=[f"Answer: {q}" for q in questions],
            workflow=WorkflowPattern.HIERARCHICAL,
            roles=[AgentRole.PLANNER, AgentRole.ANALYST, AgentRole.WRITER],
        )

    async def write_content(
        self,
        topic: str,
        content_type: str = "article",
        audience: str = "general",
        include_research: bool = True,
        include_review: bool = True,
    ) -> dict[str, Any]:
        """
        Write content with a writing team.

        Args:
            topic: Topic to write about
            content_type: Type of content (article, doc, report, etc.)
            audience: Target audience
            include_research: Include research phase
            include_review: Include review phase

        Returns:
            Written content
        """
        roles = []

        if include_research:
            roles.append(AgentRole.RESEARCHER)

        roles.append(AgentRole.WRITER)

        if include_review:
            roles.append(AgentRole.REVIEWER)

        return await self.execute_task(
            title=f"Write {content_type}: {topic}",
            description=f"Create a {content_type} about {topic} for {audience} audience",
            objective=f"High-quality {content_type} that engages the target audience",
            success_criteria=[
                "Content is accurate and well-researched" if include_research else "Content is clear",
                "Writing is clear and engaging",
                f"Appropriate for {audience} audience",
                "Well-structured and organized",
            ],
            workflow=WorkflowPattern.SEQUENTIAL,
            roles=roles,
        )

    async def debate(
        self,
        topic: str,
        positions: Optional[list[str]] = None,
        rounds: int = 3,
    ) -> dict[str, Any]:
        """
        Have agents debate a topic.

        Args:
            topic: Topic to debate
            positions: Predefined positions (or generate)
            rounds: Number of debate rounds

        Returns:
            Debate transcript and conclusions
        """
        positions_str = ""
        if positions:
            positions_str = "\n\nPositions to consider:\n" + "\n".join(f"- {p}" for p in positions)

        return await self.execute_task(
            title=f"Debate: {topic}",
            description=f"Debate topic: {topic}{positions_str}",
            objective="Thorough exploration of different perspectives",
            success_criteria=[
                "Multiple perspectives presented",
                "Arguments well-reasoned",
                "Counterarguments addressed",
                "Synthesis or conclusion reached",
            ],
            workflow=WorkflowPattern.DEBATE,
            roles=[AgentRole.ANALYST, AgentRole.ANALYST, AgentRole.REVIEWER],
        )

    async def plan_project(
        self,
        project_description: str,
        constraints: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """
        Create a project plan.

        Args:
            project_description: What the project is
            constraints: Any constraints to consider

        Returns:
            Project plan
        """
        constraints_str = ""
        if constraints:
            constraints_str = "\n\nConstraints:\n" + "\n".join(f"- {c}" for c in constraints)

        return await self.execute_task(
            title=f"Plan: {project_description[:50]}",
            description=f"Project: {project_description}{constraints_str}",
            objective="Comprehensive, actionable project plan",
            success_criteria=[
                "Clear task breakdown",
                "Dependencies identified",
                "Risks assessed",
                "Milestones defined",
            ],
            workflow=WorkflowPattern.SEQUENTIAL,
            roles=[AgentRole.PLANNER, AgentRole.ANALYST],
        )

    # === Agent Management ===

    async def spawn_agent(
        self,
        role: AgentRole,
        name: Optional[str] = None,
    ) -> dict[str, Any]:
        """Spawn a new agent."""
        agent = await self.pool.spawn_agent(role=role, name_override=name)
        return agent.to_dict()

    async def terminate_agent(self, agent_id: str) -> bool:
        """Terminate an agent."""
        return await self.pool.terminate_agent(agent_id)

    def list_agents(
        self,
        status: Optional[AgentStatus] = None,
        role: Optional[AgentRole] = None,
    ) -> list[dict[str, Any]]:
        """List agents."""
        agents = self.pool.list_agents(status=status, role=role)
        return [a.to_dict() for a in agents]

    # === Team Management ===

    def list_teams(
        self,
        status: Optional[TeamStatus] = None,
    ) -> list[dict[str, Any]]:
        """List teams."""
        teams = self.teams.list_teams(status=status)
        return [t.to_dict() for t in teams]

    # === Statistics ===

    def get_stats(self) -> dict[str, Any]:
        """Get orchestrator statistics."""
        uptime = 0.0
        if self._started_at:
            uptime = (datetime.now() - self._started_at).total_seconds()

        return {
            "initialized": self._initialized,
            "uptime_seconds": uptime,
            "total_tasks_executed": self._total_tasks_executed,
            "total_tasks_succeeded": self._total_tasks_succeeded,
            "success_rate": (
                self._total_tasks_succeeded / self._total_tasks_executed
                if self._total_tasks_executed > 0 else 0
            ),
            "active_tasks": len(self._active_tasks),
            "pool": self.pool.get_stats(),
            "teams": self.teams.get_stats(),
            "messaging": self.bus.get_stats(),
            "consensus": self.consensus.get_stats(),
            "delegation": self.delegator.get_stats(),
        }

    def get_orchestrator_stats(self) -> OrchestratorStats:
        """Get typed statistics object."""
        pool_stats = self.pool.get_stats()
        team_stats = self.teams.get_stats()
        bus_stats = self.bus.get_stats()

        return OrchestratorStats(
            total_agents=pool_stats["total_agents"],
            active_agents=pool_stats["by_status"].get("busy", 0) + pool_stats["by_status"].get("waiting", 0),
            idle_agents=pool_stats["by_status"].get("idle", 0),
            agents_by_role=pool_stats["by_role"],
            agents_by_status=pool_stats["by_status"],
            total_teams=team_stats["total_teams"],
            active_teams=team_stats["by_status"].get("active", 0),
            teams_by_status=team_stats["by_status"],
            total_tasks=self._total_tasks_executed,
            tasks_completed=team_stats["tasks_completed"],
            tasks_failed=team_stats["tasks_failed"],
            tasks_pending=len(self._active_tasks),
            messages_sent=bus_stats["messages_sent"],
            messages_delivered=bus_stats["messages_delivered"],
            broadcasts=bus_stats["broadcasts"],
            total_tokens_used=pool_stats["total_tokens_used"],
            total_runtime_seconds=pool_stats["total_runtime_seconds"],
            uptime_seconds=(datetime.now() - self._started_at).total_seconds() if self._started_at else 0,
            started_at=self._started_at,
        )
