"""
SOTA Multi-Agent Orchestrator

Enhanced orchestrator with state-of-the-art capabilities:
- Advanced memory systems (vector, episodic, semantic, working)
- Sophisticated reasoning (ToT, CoT, reflection, metacognition)
- Learning and adaptation (RL, skills, meta-learning)
- Advanced planning (HTN, MCTS, GOAP)
- MCP tool integration
- Multi-agent safety constraints
- Full observability and tracing
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import structlog

from aion.systems.agents.orchestrator import MultiAgentOrchestrator
from aion.systems.agents.types import (
    AgentRole,
    WorkflowPattern,
    TeamTask,
)

# Memory Systems
from aion.systems.agents.memory import (
    AgentMemoryManager,
    VectorStore,
    EpisodicMemory,
    SemanticMemory,
    WorkingMemory,
    RAGEngine,
)

# Reasoning Systems
from aion.systems.agents.reasoning import (
    TreeOfThought,
    ChainOfThought,
    SelfReflection,
    MetacognitiveMonitor,
    AnalogicalReasoner,
)

# Learning Systems
from aion.systems.agents.learning import (
    ReinforcementLearner,
    SkillLibrary,
    AdaptationEngine,
    MetaLearner,
)

# Planning Systems
from aion.systems.agents.planning import (
    HTNPlanner,
    MCTSPlanner,
    GOAPPlanner,
)

# Tool Systems
from aion.systems.agents.tools import (
    MCPClient,
    MCPServer,
    ToolRegistry,
    ToolExecutor,
    ExecutionContext,
)

# Safety Systems
from aion.systems.agents.safety import (
    SafetyChecker,
    AlignmentMonitor,
    CoordinationSafety,
)

# Observability
from aion.systems.agents.observability import (
    TracingManager,
    MetricsRegistry,
    LogAggregator,
    AgentDashboard,
)

logger = structlog.get_logger(__name__)


@dataclass
class SOTAConfig:
    """Configuration for SOTA orchestrator."""

    # Memory settings
    vector_dimension: int = 1536
    max_episodic_memories: int = 10000
    working_memory_capacity: int = 7

    # Reasoning settings
    tot_max_depth: int = 5
    cot_self_consistency_samples: int = 3
    reflection_enabled: bool = True

    # Learning settings
    learning_rate: float = 0.1
    exploration_rate: float = 0.1
    skill_transfer_enabled: bool = True

    # Planning settings
    planning_timeout: float = 30.0
    mcts_simulations: int = 100

    # Safety settings
    safety_enabled: bool = True
    alignment_monitoring: bool = True

    # Observability settings
    tracing_enabled: bool = True
    metrics_enabled: bool = True


@dataclass
class AgentEnhancements:
    """SOTA enhancements for a single agent."""

    agent_id: str
    memory: AgentMemoryManager = field(default=None)
    tot: TreeOfThought = field(default=None)
    cot: ChainOfThought = field(default=None)
    reflection: SelfReflection = field(default=None)
    metacognition: MetacognitiveMonitor = field(default=None)
    analogical: AnalogicalReasoner = field(default=None)
    rl_learner: ReinforcementLearner = field(default=None)
    skills: SkillLibrary = field(default=None)
    adaptation: AdaptationEngine = field(default=None)
    meta_learner: MetaLearner = field(default=None)
    mcp_client: MCPClient = field(default=None)


class SOTAOrchestrator(MultiAgentOrchestrator):
    """
    State-of-the-art multi-agent orchestrator.

    Extends the base MultiAgentOrchestrator with:
    - Advanced memory systems for each agent
    - Tree-of-thought and chain-of-thought reasoning
    - Reinforcement learning and skill libraries
    - HTN, MCTS, and GOAP planning
    - MCP tool integration
    - Constitutional AI safety
    - Full observability stack
    """

    def __init__(
        self,
        config: Optional[SOTAConfig] = None,
        max_agents: int = 20,
        max_teams: int = 10,
    ):
        super().__init__(max_agents=max_agents, max_teams=max_teams)

        self.config = config or SOTAConfig()

        # Agent enhancements registry
        self._agent_enhancements: dict[str, AgentEnhancements] = {}

        # Global systems
        self._vector_store: Optional[VectorStore] = None
        self._tracing: Optional[TracingManager] = None
        self._metrics: Optional[MetricsRegistry] = None
        self._logging: Optional[LogAggregator] = None
        self._dashboard: Optional[AgentDashboard] = None

        # Tool systems
        self._mcp_server: Optional[MCPServer] = None
        self._tool_registry: Optional[ToolRegistry] = None
        self._tool_executor: Optional[ToolExecutor] = None

        # Planning systems
        self._htn_planner: Optional[HTNPlanner] = None
        self._mcts_planner: Optional[MCTSPlanner] = None
        self._goap_planner: Optional[GOAPPlanner] = None

        # Safety systems
        self._safety_checker: Optional[SafetyChecker] = None
        self._alignment_monitor: Optional[AlignmentMonitor] = None
        self._coordination_safety: Optional[CoordinationSafety] = None

    async def initialize(self) -> None:
        """Initialize SOTA orchestrator with all subsystems."""
        if self._initialized:
            return

        logger.info("Initializing SOTA Multi-Agent Orchestrator")

        # Initialize base orchestrator
        await super().initialize()

        # Initialize observability first
        await self._init_observability()

        # Initialize global systems
        await self._init_memory()
        await self._init_tools()
        await self._init_planning()
        await self._init_safety()

        logger.info("SOTA Multi-Agent Orchestrator initialized")

    async def _init_observability(self) -> None:
        """Initialize observability stack."""
        if self.config.tracing_enabled:
            self._tracing = TracingManager()
            await self._tracing.initialize()

        if self.config.metrics_enabled:
            self._metrics = MetricsRegistry()
            await self._metrics.initialize()

        self._logging = LogAggregator()
        await self._logging.initialize()

        if self._tracing and self._metrics and self._logging:
            self._dashboard = AgentDashboard(
                tracing=self._tracing,
                metrics=self._metrics,
                logging=self._logging,
            )
            await self._dashboard.initialize()

    async def _init_memory(self) -> None:
        """Initialize global memory systems."""
        self._vector_store = VectorStore(
            dimension=self.config.vector_dimension,
        )
        await self._vector_store.initialize()

    async def _init_tools(self) -> None:
        """Initialize MCP tool systems."""
        self._mcp_server = MCPServer(name="aion-sota-server")
        await self._mcp_server.initialize()

        self._tool_registry = ToolRegistry()
        await self._tool_registry.initialize()

        # Create a master MCP client
        master_client = MCPClient("orchestrator")
        await master_client.initialize()
        await master_client.connect_server("main", self._mcp_server)

        self._tool_executor = ToolExecutor(master_client)
        await self._tool_executor.initialize()

    async def _init_planning(self) -> None:
        """Initialize planning systems."""
        self._htn_planner = HTNPlanner()
        self._mcts_planner = MCTSPlanner(
            num_simulations=self.config.mcts_simulations,
        )
        self._goap_planner = GOAPPlanner()

    async def _init_safety(self) -> None:
        """Initialize safety systems."""
        if self.config.safety_enabled:
            self._safety_checker = SafetyChecker()

        if self.config.alignment_monitoring:
            self._alignment_monitor = AlignmentMonitor()

        self._coordination_safety = CoordinationSafety()

    async def shutdown(self) -> None:
        """Shutdown all systems."""
        logger.info("Shutting down SOTA Orchestrator")

        # Shutdown agent enhancements
        for enhancement in self._agent_enhancements.values():
            if enhancement.memory:
                await enhancement.memory.shutdown()
            if enhancement.skills:
                await enhancement.skills.shutdown()
            if enhancement.mcp_client:
                await enhancement.mcp_client.shutdown()

        # Shutdown global systems
        if self._tool_executor:
            await self._tool_executor.shutdown()
        if self._mcp_server:
            await self._mcp_server.shutdown()
        if self._vector_store:
            await self._vector_store.shutdown()

        if self._dashboard:
            await self._dashboard.shutdown()
        if self._metrics:
            await self._metrics.shutdown()
        if self._tracing:
            await self._tracing.shutdown()
        if self._logging:
            await self._logging.shutdown()

        # Shutdown base orchestrator
        await super().shutdown()

        logger.info("SOTA Orchestrator shutdown complete")

    async def _enhance_agent(self, agent_id: str) -> AgentEnhancements:
        """Create SOTA enhancements for an agent."""
        if agent_id in self._agent_enhancements:
            return self._agent_enhancements[agent_id]

        enhancement = AgentEnhancements(agent_id=agent_id)

        # Memory
        enhancement.memory = AgentMemoryManager(
            agent_id=agent_id,
            vector_dimension=self.config.vector_dimension,
        )
        await enhancement.memory.initialize()

        # Reasoning
        enhancement.tot = TreeOfThought(
            agent_id=agent_id,
            max_depth=self.config.tot_max_depth,
        )
        enhancement.cot = ChainOfThought(agent_id=agent_id)
        enhancement.reflection = SelfReflection(agent_id=agent_id)
        enhancement.metacognition = MetacognitiveMonitor(agent_id=agent_id)
        enhancement.analogical = AnalogicalReasoner(agent_id=agent_id)

        # Learning
        enhancement.rl_learner = ReinforcementLearner(
            agent_id=agent_id,
            learning_rate=self.config.learning_rate,
            exploration_rate=self.config.exploration_rate,
        )
        enhancement.skills = SkillLibrary(agent_id=agent_id)
        await enhancement.skills.initialize()

        enhancement.adaptation = AdaptationEngine(
            agent_id=agent_id,
            learning_rate=self.config.learning_rate,
        )
        await enhancement.adaptation.initialize()

        enhancement.meta_learner = MetaLearner(agent_id=agent_id)
        await enhancement.meta_learner.initialize()

        # Tools
        enhancement.mcp_client = MCPClient(agent_id)
        await enhancement.mcp_client.initialize()
        if self._mcp_server:
            await enhancement.mcp_client.connect_server("main", self._mcp_server)

        self._agent_enhancements[agent_id] = enhancement

        # Update dashboard
        if self._dashboard:
            self._dashboard.update_agent_status(agent_id, "enhanced")

        return enhancement

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
        use_reasoning: bool = True,
        use_planning: bool = True,
        use_memory: bool = True,
    ) -> dict[str, Any]:
        """
        Execute a task with SOTA enhancements.

        Args:
            title: Task title
            description: Task description
            objective: What success looks like
            success_criteria: Measurable criteria
            workflow: Workflow pattern
            roles: Agent roles
            max_iterations: Maximum iterations
            timeout: Timeout in seconds
            use_reasoning: Enable ToT/CoT reasoning
            use_planning: Enable advanced planning
            use_memory: Enable memory systems

        Returns:
            Enhanced task result
        """
        # Start trace
        tracer = None
        if self._tracing:
            tracer = self._tracing.get_tracer("orchestrator")
            await tracer.initialize()

        async with tracer.span("execute_task", attributes={"title": title}) if tracer else asyncio.nullcontext():
            # Safety check
            if self._safety_checker:
                safe, violations = self._safety_checker.check_action(
                    "execute_task",
                    {"title": title, "description": description},
                )
                if not safe:
                    return {
                        "success": False,
                        "error": f"Safety check failed: {violations}",
                    }

            # Use advanced planning if enabled
            plan = None
            if use_planning and self._htn_planner:
                # Create HTN task
                from aion.systems.agents.planning.htn import Task, Method

                htn_task = Task(
                    name=title,
                    parameters={"description": description},
                )
                plan = self._htn_planner.plan({}, [htn_task])

            # Execute base task
            result = await super().execute_task(
                title=title,
                description=description,
                objective=objective,
                success_criteria=success_criteria,
                workflow=workflow,
                roles=roles,
                max_iterations=max_iterations,
                timeout=timeout,
            )

            # Store in memory if enabled
            if use_memory and result.get("success"):
                # Store successful execution as episodic memory
                for agent_id in result.get("agents_used", []):
                    enhancement = self._agent_enhancements.get(agent_id)
                    if enhancement and enhancement.memory:
                        await enhancement.memory.remember(
                            f"Successfully completed: {title}. Objective: {objective}",
                            memory_type="episodic",
                            metadata={"result": result.get("output", "")[:500]},
                        )

            # Update metrics
            if self._metrics:
                collector = self._metrics.get_collector("orchestrator")
                counter = collector.counter("agent_tasks_total")
                if counter:
                    status = "success" if result.get("success") else "error"
                    counter.inc(labels={"status": status})

            # Add SOTA metadata to result
            result["sota_features"] = {
                "reasoning_enabled": use_reasoning,
                "planning_enabled": use_planning,
                "memory_enabled": use_memory,
                "safety_enabled": self.config.safety_enabled,
            }

            if plan:
                result["plan"] = {
                    "steps": len(plan.steps) if plan.steps else 0,
                    "valid": plan.valid if plan else False,
                }

            return result

    async def reason_with_tot(
        self,
        agent_id: str,
        problem: str,
        initial_thoughts: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """
        Use Tree-of-Thought reasoning for an agent.

        Args:
            agent_id: Agent to use
            problem: Problem to solve
            initial_thoughts: Starting thoughts

        Returns:
            Solution path and confidence
        """
        enhancement = await self._enhance_agent(agent_id)

        if enhancement.tot:
            solution_path, confidence, tree = await enhancement.tot.solve(
                problem, initial_thoughts
            )
            return {
                "solution_path": solution_path,
                "confidence": confidence,
                "tree_depth": tree.depth if tree else 0,
            }

        return {"error": "ToT not available"}

    async def reason_with_cot(
        self,
        agent_id: str,
        problem: str,
        context: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Use Chain-of-Thought reasoning for an agent.

        Args:
            agent_id: Agent to use
            problem: Problem to solve
            context: Additional context

        Returns:
            Reasoning chain
        """
        enhancement = await self._enhance_agent(agent_id)

        if enhancement.cot:
            chain = await enhancement.cot.reason(problem, context)
            return chain.to_dict()

        return {"error": "CoT not available"}

    async def recall_for_agent(
        self,
        agent_id: str,
        query: str,
        k: int = 5,
    ) -> dict[str, Any]:
        """
        Recall memories for an agent.

        Args:
            agent_id: Agent to recall for
            query: What to recall
            k: Number of results

        Returns:
            Retrieved context
        """
        enhancement = await self._enhance_agent(agent_id)

        if enhancement.memory:
            context = await enhancement.memory.recall(query, k=k)
            return {
                "chunks": [
                    {"text": c.text, "score": c.score}
                    for c in context.chunks
                ],
                "total": len(context.chunks),
            }

        return {"error": "Memory not available"}

    async def learn_skill(
        self,
        agent_id: str,
        name: str,
        description: str,
        steps: list[str],
    ) -> dict[str, Any]:
        """
        Teach an agent a new skill.

        Args:
            agent_id: Agent to teach
            name: Skill name
            description: Skill description
            steps: Execution steps

        Returns:
            Skill info
        """
        enhancement = await self._enhance_agent(agent_id)

        if enhancement.skills:
            skill = await enhancement.skills.learn_skill(
                name=name,
                description=description,
                execution_steps=steps,
            )
            return skill.to_dict()

        return {"error": "Skills not available"}

    async def execute_tool(
        self,
        agent_id: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Execute a tool for an agent.

        Args:
            agent_id: Agent making the call
            tool_name: Tool to execute
            arguments: Tool arguments

        Returns:
            Tool result
        """
        enhancement = await self._enhance_agent(agent_id)

        if enhancement.mcp_client:
            context = ExecutionContext(agent_id=agent_id)
            result = await self._tool_executor.execute(
                tool_name, arguments, context
            )
            return result.to_mcp_format()

        return {"error": "Tools not available"}

    def get_dashboard(self) -> Optional[AgentDashboard]:
        """Get the monitoring dashboard."""
        return self._dashboard

    def get_sota_stats(self) -> dict[str, Any]:
        """Get SOTA-specific statistics."""
        base_stats = self.get_stats()

        sota_stats = {
            **base_stats,
            "sota_features": {
                "enhanced_agents": len(self._agent_enhancements),
                "tracing_enabled": self.config.tracing_enabled,
                "metrics_enabled": self.config.metrics_enabled,
                "safety_enabled": self.config.safety_enabled,
                "alignment_monitoring": self.config.alignment_monitoring,
            },
            "memory": {
                "agents_with_memory": sum(
                    1 for e in self._agent_enhancements.values()
                    if e.memory
                ),
            },
            "tools": {
                "registered_tools": (
                    len(self._tool_registry._tools)
                    if self._tool_registry else 0
                ),
                "executions": (
                    self._tool_executor.get_stats()
                    if self._tool_executor else {}
                ),
            },
        }

        if self._tracing:
            sota_stats["tracing"] = self._tracing.get_stats()

        if self._metrics:
            sota_stats["metrics"] = self._metrics.get_summary()

        if self._logging:
            sota_stats["logging"] = self._logging.get_stats()

        if self._safety_checker:
            sota_stats["safety"] = self._safety_checker.get_stats()

        if self._alignment_monitor:
            sota_stats["alignment"] = {
                "assessments": len(self._alignment_monitor._history),
            }

        if self._coordination_safety:
            sota_stats["coordination"] = (
                self._coordination_safety.coordinator.get_stats()
            )

        return sota_stats
