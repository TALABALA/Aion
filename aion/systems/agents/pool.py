"""
AION Agent Pool

Registry and lifecycle management for agent instances.
Provides centralized agent creation, tracking, discovery, and termination.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Callable, Optional

import structlog

from aion.systems.agents.types import (
    AgentProfile,
    AgentInstance,
    AgentRole,
    AgentStatus,
    AgentCapability,
)

logger = structlog.get_logger(__name__)


def _create_default_profiles() -> dict[AgentRole, AgentProfile]:
    """Create default agent profiles for each role."""
    return {
        AgentRole.RESEARCHER: AgentProfile(
            name="Researcher",
            role=AgentRole.RESEARCHER,
            system_prompt="""You are a research specialist agent. Your role is to:
- Gather information from available sources
- Synthesize findings into clear summaries
- Identify key insights and patterns
- Cite sources and note confidence levels
- Flag gaps in information

Be thorough, accurate, and cite your sources. When uncertain, say so.""",
            personality_traits=["curious", "thorough", "analytical"],
            communication_style="academic",
            capabilities=[
                AgentCapability(
                    name="information_gathering",
                    description="Search and collect information from various sources",
                    proficiency=0.9,
                    can_handle_tasks=["research", "search", "gather", "investigate", "find", "lookup"],
                    tools_required=["web_search", "memory_search"],
                ),
                AgentCapability(
                    name="synthesis",
                    description="Combine information from multiple sources",
                    proficiency=0.85,
                    can_handle_tasks=["synthesize", "summarize", "compile"],
                ),
            ],
            tools=["web_search", "web_fetch", "memory_search", "memory_store"],
        ),

        AgentRole.CODER: AgentProfile(
            name="Coder",
            role=AgentRole.CODER,
            system_prompt="""You are a coding specialist agent. Your role is to:
- Write clean, efficient, well-documented code
- Debug and fix issues systematically
- Review code for quality, security, and best practices
- Follow language/framework conventions
- Write tests when appropriate

Prioritize correctness, readability, and maintainability. Explain your approach.""",
            personality_traits=["precise", "methodical", "detail-oriented"],
            communication_style="technical",
            capabilities=[
                AgentCapability(
                    name="code_generation",
                    description="Write and modify code",
                    proficiency=0.9,
                    can_handle_tasks=["code", "implement", "build", "develop", "program", "create"],
                    tools_required=["code_execute", "file_write"],
                ),
                AgentCapability(
                    name="debugging",
                    description="Debug and fix code issues",
                    proficiency=0.85,
                    can_handle_tasks=["debug", "fix", "troubleshoot", "diagnose"],
                ),
                AgentCapability(
                    name="refactoring",
                    description="Improve code structure and quality",
                    proficiency=0.8,
                    can_handle_tasks=["refactor", "optimize", "improve", "clean"],
                ),
            ],
            tools=["code_execute", "file_read", "file_write", "bash"],
        ),

        AgentRole.ANALYST: AgentProfile(
            name="Analyst",
            role=AgentRole.ANALYST,
            system_prompt="""You are a data analysis specialist agent. Your role is to:
- Analyze data to find patterns and insights
- Create visualizations and reports
- Perform statistical analysis
- Draw evidence-based conclusions
- Communicate findings clearly

Be rigorous, objective, and data-driven. Quantify uncertainty when possible.""",
            personality_traits=["analytical", "objective", "precise"],
            communication_style="data-driven",
            capabilities=[
                AgentCapability(
                    name="data_analysis",
                    description="Analyze data and produce insights",
                    proficiency=0.9,
                    can_handle_tasks=["analyze", "examine", "study", "evaluate"],
                    tools_required=["code_execute"],
                ),
                AgentCapability(
                    name="visualization",
                    description="Create charts and visualizations",
                    proficiency=0.85,
                    can_handle_tasks=["visualize", "chart", "graph", "plot"],
                ),
                AgentCapability(
                    name="statistics",
                    description="Statistical analysis and modeling",
                    proficiency=0.85,
                    can_handle_tasks=["statistics", "model", "predict", "forecast"],
                ),
            ],
            tools=["code_execute", "file_read", "file_write"],
        ),

        AgentRole.WRITER: AgentProfile(
            name="Writer",
            role=AgentRole.WRITER,
            system_prompt="""You are a content creation specialist agent. Your role is to:
- Write clear, engaging content
- Adapt tone and style to the audience
- Structure content logically
- Edit and refine text
- Ensure accuracy and clarity

Prioritize clarity, engagement, and appropriateness for the target audience.""",
            personality_traits=["creative", "articulate", "empathetic"],
            communication_style="adaptable",
            capabilities=[
                AgentCapability(
                    name="content_creation",
                    description="Create written content",
                    proficiency=0.9,
                    can_handle_tasks=["write", "draft", "compose", "author", "create"],
                ),
                AgentCapability(
                    name="editing",
                    description="Edit and refine text",
                    proficiency=0.85,
                    can_handle_tasks=["edit", "revise", "polish", "proofread"],
                ),
                AgentCapability(
                    name="summarization",
                    description="Summarize and condense content",
                    proficiency=0.85,
                    can_handle_tasks=["summarize", "condense", "abstract", "brief"],
                ),
            ],
            tools=["file_write", "file_read", "web_search"],
        ),

        AgentRole.REVIEWER: AgentProfile(
            name="Reviewer",
            role=AgentRole.REVIEWER,
            system_prompt="""You are a quality review specialist agent. Your role is to:
- Review work for quality and correctness
- Identify issues, errors, and improvements
- Provide constructive feedback
- Verify against requirements
- Ensure consistency and standards

Be thorough, fair, and constructive. Explain your reasoning.""",
            personality_traits=["critical", "fair", "constructive"],
            communication_style="evaluative",
            capabilities=[
                AgentCapability(
                    name="quality_review",
                    description="Review and critique work",
                    proficiency=0.9,
                    can_handle_tasks=["review", "critique", "evaluate", "assess"],
                ),
                AgentCapability(
                    name="verification",
                    description="Verify correctness and compliance",
                    proficiency=0.85,
                    can_handle_tasks=["verify", "check", "validate", "test", "confirm"],
                ),
                AgentCapability(
                    name="feedback",
                    description="Provide constructive feedback",
                    proficiency=0.85,
                    can_handle_tasks=["feedback", "suggest", "recommend", "advise"],
                ),
            ],
            tools=["file_read"],
        ),

        AgentRole.PLANNER: AgentProfile(
            name="Planner",
            role=AgentRole.PLANNER,
            system_prompt="""You are a strategic planning specialist agent. Your role is to:
- Break down complex problems into manageable steps
- Create actionable plans
- Identify dependencies and risks
- Allocate resources effectively
- Adjust plans based on feedback

Be strategic, realistic, and adaptable. Consider constraints and alternatives.""",
            personality_traits=["strategic", "organized", "forward-thinking"],
            communication_style="structured",
            capabilities=[
                AgentCapability(
                    name="strategic_planning",
                    description="Create and manage plans",
                    proficiency=0.9,
                    can_handle_tasks=["plan", "strategize", "organize", "structure"],
                ),
                AgentCapability(
                    name="decomposition",
                    description="Break down complex tasks",
                    proficiency=0.9,
                    can_handle_tasks=["decompose", "break down", "divide", "split"],
                ),
                AgentCapability(
                    name="risk_assessment",
                    description="Identify and assess risks",
                    proficiency=0.8,
                    can_handle_tasks=["risk", "assess", "evaluate", "identify"],
                ),
            ],
            tools=["planning_create", "planning_execute"],
        ),

        AgentRole.EXECUTOR: AgentProfile(
            name="Executor",
            role=AgentRole.EXECUTOR,
            system_prompt="""You are a task execution specialist agent. Your role is to:
- Execute tasks efficiently and accurately
- Follow instructions precisely
- Report progress and issues
- Handle errors gracefully
- Complete work on time

Be efficient, reliable, and communicative. Flag issues early.""",
            personality_traits=["reliable", "efficient", "focused"],
            communication_style="concise",
            capabilities=[
                AgentCapability(
                    name="task_execution",
                    description="Execute assigned tasks",
                    proficiency=0.9,
                    can_handle_tasks=["execute", "perform", "do", "run", "complete"],
                ),
                AgentCapability(
                    name="automation",
                    description="Automate repetitive tasks",
                    proficiency=0.85,
                    can_handle_tasks=["automate", "script", "batch"],
                ),
            ],
            tools=["bash", "code_execute", "file_read", "file_write"],
        ),

        AgentRole.GENERALIST: AgentProfile(
            name="Generalist",
            role=AgentRole.GENERALIST,
            system_prompt="""You are a general-purpose agent. Your role is to:
- Handle a variety of tasks flexibly
- Adapt to different requirements
- Know when to seek specialist help
- Provide well-rounded assistance

Be versatile and adaptive. Ask for clarification when needed.""",
            personality_traits=["adaptable", "resourceful", "flexible"],
            communication_style="professional",
            capabilities=[
                AgentCapability(
                    name="general",
                    description="Handle general tasks",
                    proficiency=0.7,
                    can_handle_tasks=["help", "assist", "support", "general"],
                ),
            ],
            tools=["web_search", "file_read", "file_write", "code_execute"],
        ),
    }


class AgentPool:
    """
    Registry and lifecycle manager for agent instances.

    Features:
    - Agent creation from profiles
    - Instance tracking and status
    - Capability-based agent discovery
    - Statistics and monitoring
    """

    def __init__(
        self,
        max_agents: int = 20,
    ):
        self.max_agents = max_agents

        # Profiles and instances
        self._profiles: dict[str, AgentProfile] = {}
        self._instances: dict[str, AgentInstance] = {}

        # Role to profile mapping
        self._role_profiles: dict[AgentRole, str] = {}

        # Callbacks
        self._on_spawn_callbacks: list[Callable[[AgentInstance], None]] = []
        self._on_terminate_callbacks: list[Callable[[AgentInstance], None]] = []

        # Statistics
        self._total_spawned: int = 0
        self._total_terminated: int = 0

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the agent pool with default profiles."""
        if self._initialized:
            return

        logger.info("Initializing Agent Pool", max_agents=self.max_agents)

        # Register default profiles
        default_profiles = _create_default_profiles()
        for role, profile in default_profiles.items():
            self._profiles[profile.id] = profile
            self._role_profiles[role] = profile.id

        self._initialized = True
        logger.info("Agent Pool initialized", profiles_loaded=len(self._profiles))

    async def shutdown(self) -> None:
        """Shutdown all agents and clean up."""
        logger.info("Shutting down Agent Pool", active_agents=len(self._instances))

        # Terminate all agents
        for agent_id in list(self._instances.keys()):
            await self.terminate_agent(agent_id)

        self._initialized = False
        logger.info("Agent Pool shutdown complete")

    # === Profile Management ===

    def register_profile(self, profile: AgentProfile) -> None:
        """Register a new agent profile."""
        self._profiles[profile.id] = profile

        # Update role mapping if this is a role-specific profile
        if profile.role != AgentRole.CUSTOM:
            self._role_profiles[profile.role] = profile.id

        logger.debug("Registered profile", name=profile.name, role=profile.role.value)

    def get_profile(self, profile_id: str) -> Optional[AgentProfile]:
        """Get a profile by ID."""
        return self._profiles.get(profile_id)

    def get_profile_by_role(self, role: AgentRole) -> Optional[AgentProfile]:
        """Get a profile by role."""
        profile_id = self._role_profiles.get(role)
        if profile_id:
            return self._profiles.get(profile_id)
        return None

    def list_profiles(self) -> list[AgentProfile]:
        """List all registered profiles."""
        return list(self._profiles.values())

    # === Agent Lifecycle ===

    async def spawn_agent(
        self,
        profile_id: Optional[str] = None,
        role: Optional[AgentRole] = None,
        custom_profile: Optional[AgentProfile] = None,
        name_override: Optional[str] = None,
    ) -> AgentInstance:
        """
        Spawn a new agent instance.

        Args:
            profile_id: ID of profile to use
            role: Role to spawn (uses default profile for role)
            custom_profile: Custom profile to use
            name_override: Optional name override

        Returns:
            New agent instance

        Raises:
            RuntimeError: If max agents reached
            ValueError: If no valid profile specified
        """
        if len(self._instances) >= self.max_agents:
            raise RuntimeError(f"Maximum agents ({self.max_agents}) reached")

        # Determine profile to use
        profile: Optional[AgentProfile] = None

        if custom_profile:
            profile = custom_profile
            self.register_profile(profile)
        elif profile_id:
            profile = self._profiles.get(profile_id)
            if not profile:
                raise ValueError(f"Profile not found: {profile_id}")
        elif role:
            profile = self.get_profile_by_role(role)
            if not profile:
                raise ValueError(f"No profile for role: {role}")
        else:
            raise ValueError("Must specify profile_id, role, or custom_profile")

        # Create instance
        instance = AgentInstance(profile=profile)

        # Apply name override
        if name_override:
            instance.profile = AgentProfile(
                id=profile.id,
                name=name_override,
                role=profile.role,
                system_prompt=profile.system_prompt,
                personality_traits=profile.personality_traits.copy(),
                communication_style=profile.communication_style,
                capabilities=profile.capabilities.copy(),
                tools=profile.tools.copy(),
                max_concurrent_tasks=profile.max_concurrent_tasks,
                max_tokens_per_task=profile.max_tokens_per_task,
                timeout_seconds=profile.timeout_seconds,
                model=profile.model,
                temperature=profile.temperature,
                metadata=profile.metadata.copy(),
            )

        self._instances[instance.id] = instance
        self._total_spawned += 1

        # Notify callbacks
        for callback in self._on_spawn_callbacks:
            try:
                callback(instance)
            except Exception as e:
                logger.warning("Spawn callback error", error=str(e))

        logger.info(
            "Spawned agent",
            agent_id=instance.id[:8],
            name=instance.profile.name,
            role=instance.profile.role.value,
        )

        return instance

    async def terminate_agent(self, agent_id: str) -> bool:
        """
        Terminate an agent instance.

        Args:
            agent_id: ID of agent to terminate

        Returns:
            True if terminated, False if not found
        """
        instance = self._instances.pop(agent_id, None)
        if not instance:
            return False

        # Update status
        instance.status = AgentStatus.TERMINATED

        self._total_terminated += 1

        # Notify callbacks
        for callback in self._on_terminate_callbacks:
            try:
                callback(instance)
            except Exception as e:
                logger.warning("Terminate callback error", error=str(e))

        logger.info(
            "Terminated agent",
            agent_id=agent_id[:8],
            name=instance.profile.name,
            tasks_completed=instance.tasks_completed,
        )

        return True

    # === Instance Access ===

    def get_agent(self, agent_id: str) -> Optional[AgentInstance]:
        """Get an agent instance by ID."""
        return self._instances.get(agent_id)

    def list_agents(
        self,
        status: Optional[AgentStatus] = None,
        role: Optional[AgentRole] = None,
        team_id: Optional[str] = None,
    ) -> list[AgentInstance]:
        """
        List agent instances with optional filters.

        Args:
            status: Filter by status
            role: Filter by role
            team_id: Filter by team membership

        Returns:
            List of matching agents
        """
        agents = list(self._instances.values())

        if status is not None:
            agents = [a for a in agents if a.status == status]

        if role is not None:
            agents = [a for a in agents if a.profile.role == role]

        if team_id is not None:
            agents = [a for a in agents if a.current_team_id == team_id]

        return agents

    def get_idle_agents(self) -> list[AgentInstance]:
        """Get all idle agents."""
        return self.list_agents(status=AgentStatus.IDLE)

    def get_available_agents(self) -> list[AgentInstance]:
        """Get all agents that can accept tasks."""
        return [a for a in self._instances.values() if a.status.is_available()]

    # === Capability-Based Discovery ===

    def find_capable_agent(
        self,
        task_type: str,
        prefer_idle: bool = True,
        exclude_ids: Optional[list[str]] = None,
    ) -> Optional[AgentInstance]:
        """
        Find an agent capable of handling a task type.

        Args:
            task_type: Type of task to handle
            prefer_idle: Prefer idle agents over busy ones
            exclude_ids: Agent IDs to exclude

        Returns:
            Capable agent or None
        """
        exclude_ids = exclude_ids or []
        candidates = []

        for agent in self._instances.values():
            if agent.id in exclude_ids:
                continue

            if agent.profile.can_handle(task_type):
                proficiency = agent.profile.get_proficiency(task_type)
                candidates.append((agent, proficiency))

        if not candidates:
            return None

        # Sort by proficiency (descending)
        candidates.sort(key=lambda x: x[1], reverse=True)

        if prefer_idle:
            # Find best idle agent
            for agent, _ in candidates:
                if agent.status == AgentStatus.IDLE:
                    return agent

        # Return best overall
        return candidates[0][0]

    def find_capable_agents(
        self,
        task_type: str,
        count: int = 1,
        prefer_idle: bool = True,
        exclude_ids: Optional[list[str]] = None,
    ) -> list[AgentInstance]:
        """
        Find multiple agents capable of handling a task type.

        Args:
            task_type: Type of task to handle
            count: Number of agents to find
            prefer_idle: Prefer idle agents
            exclude_ids: Agent IDs to exclude

        Returns:
            List of capable agents
        """
        exclude_ids = exclude_ids or []
        candidates = []

        for agent in self._instances.values():
            if agent.id in exclude_ids:
                continue

            if agent.profile.can_handle(task_type):
                proficiency = agent.profile.get_proficiency(task_type)
                is_idle = agent.status == AgentStatus.IDLE
                candidates.append((agent, proficiency, is_idle))

        if not candidates:
            return []

        # Sort by idle status (if preferred) then proficiency
        if prefer_idle:
            candidates.sort(key=lambda x: (not x[2], -x[1]))
        else:
            candidates.sort(key=lambda x: -x[1])

        return [c[0] for c in candidates[:count]]

    def find_agents_by_role(
        self,
        role: AgentRole,
        prefer_idle: bool = True,
    ) -> list[AgentInstance]:
        """Find agents with a specific role."""
        agents = self.list_agents(role=role)

        if prefer_idle:
            # Sort idle first
            agents.sort(key=lambda a: a.status != AgentStatus.IDLE)

        return agents

    # === Status Updates ===

    def update_status(
        self,
        agent_id: str,
        status: AgentStatus,
        task_id: Optional[str] = None,
    ) -> bool:
        """
        Update an agent's status.

        Args:
            agent_id: Agent ID
            status: New status
            task_id: Associated task ID (if any)

        Returns:
            True if updated, False if agent not found
        """
        agent = self._instances.get(agent_id)
        if not agent:
            return False

        old_status = agent.status
        agent.status = status
        agent.current_task_id = task_id
        agent.update_activity()

        logger.debug(
            "Updated agent status",
            agent_id=agent_id[:8],
            old_status=old_status.value,
            new_status=status.value,
        )

        return True

    def update_team_assignment(
        self,
        agent_id: str,
        team_id: Optional[str],
    ) -> bool:
        """
        Update an agent's team assignment.

        Args:
            agent_id: Agent ID
            team_id: Team ID or None to unassign

        Returns:
            True if updated
        """
        agent = self._instances.get(agent_id)
        if not agent:
            return False

        agent.current_team_id = team_id
        agent.update_activity()
        return True

    def record_task_completion(
        self,
        agent_id: str,
        success: bool,
        tokens_used: int = 0,
        duration_seconds: float = 0.0,
        error: Optional[str] = None,
    ) -> bool:
        """
        Record task completion for an agent.

        Args:
            agent_id: Agent ID
            success: Whether task succeeded
            tokens_used: Tokens consumed
            duration_seconds: Time taken
            error: Error message if failed

        Returns:
            True if recorded
        """
        agent = self._instances.get(agent_id)
        if not agent:
            return False

        if success:
            agent.tasks_completed += 1
            agent.consecutive_errors = 0
        else:
            agent.tasks_failed += 1
            agent.consecutive_errors += 1
            agent.last_error = error

        agent.total_tokens_used += tokens_used
        agent.total_runtime_seconds += duration_seconds
        agent.update_activity()

        return True

    # === Callbacks ===

    def on_spawn(self, callback: Callable[[AgentInstance], None]) -> None:
        """Register a callback for agent spawn events."""
        self._on_spawn_callbacks.append(callback)

    def on_terminate(self, callback: Callable[[AgentInstance], None]) -> None:
        """Register a callback for agent termination events."""
        self._on_terminate_callbacks.append(callback)

    # === Statistics ===

    def get_stats(self) -> dict[str, Any]:
        """Get pool statistics."""
        status_counts: dict[str, int] = {}
        for status in AgentStatus:
            status_counts[status.value] = len(self.list_agents(status=status))

        role_counts: dict[str, int] = {}
        for role in AgentRole:
            role_counts[role.value] = len(self.list_agents(role=role))

        total_tokens = sum(a.total_tokens_used for a in self._instances.values())
        total_runtime = sum(a.total_runtime_seconds for a in self._instances.values())
        total_completed = sum(a.tasks_completed for a in self._instances.values())
        total_failed = sum(a.tasks_failed for a in self._instances.values())

        return {
            "total_agents": len(self._instances),
            "max_agents": self.max_agents,
            "by_status": status_counts,
            "by_role": role_counts,
            "total_profiles": len(self._profiles),
            "total_spawned": self._total_spawned,
            "total_terminated": self._total_terminated,
            "total_tokens_used": total_tokens,
            "total_runtime_seconds": total_runtime,
            "total_tasks_completed": total_completed,
            "total_tasks_failed": total_failed,
        }
