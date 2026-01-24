"""
AION Multi-Agent Orchestration System

Enables multiple specialized agents to collaborate on complex tasks,
communicate with each other, and achieve emergent problem-solving
capabilities that exceed any single agent's abilities.

Main Components:
- MultiAgentOrchestrator: Central coordinator for multi-agent operations
- AgentPool: Registry and lifecycle management for agents
- TeamManager: Team formation and workflow coordination
- MessageBus: Inter-agent communication system
- TaskDelegator: Intelligent task routing
- ConsensusEngine: Voting and conflict resolution

Workflow Patterns:
- Sequential: Agents work one after another
- Parallel: Agents work concurrently
- Hierarchical: Manager delegates to workers
- Debate: Agents refine through discussion
- Swarm: Emergent coordination

Example Usage:
    ```python
    from aion.systems.agents import MultiAgentOrchestrator, AgentRole, WorkflowPattern

    orchestrator = MultiAgentOrchestrator()
    await orchestrator.initialize()

    # Execute a research task
    result = await orchestrator.research(
        topic="quantum computing applications",
        depth="deep",
    )

    # Execute a coding task
    result = await orchestrator.code_task(
        description="Create a REST API for user management",
        language="python",
        include_review=True,
    )

    # Execute a custom multi-agent task
    result = await orchestrator.execute_task(
        title="Market Analysis",
        description="Analyze market trends for AI startups",
        objective="Comprehensive market report",
        workflow=WorkflowPattern.HIERARCHICAL,
        roles=[AgentRole.RESEARCHER, AgentRole.ANALYST, AgentRole.WRITER],
    )

    await orchestrator.shutdown()
    ```
"""

# Core Types
from aion.systems.agents.types import (
    # Enums
    AgentRole,
    AgentStatus,
    MessageType,
    TeamStatus,
    WorkflowPattern,
    ConsensusMethod,
    TaskPriority,
    # Dataclasses
    AgentCapability,
    AgentProfile,
    AgentInstance,
    Message,
    TeamTask,
    Team,
    ConsensusVote,
    ConsensusResult,
    WorkflowStep,
    WorkflowExecution,
    OrchestratorStats,
)

# Core Components
from aion.systems.agents.orchestrator import MultiAgentOrchestrator
from aion.systems.agents.pool import AgentPool
from aion.systems.agents.messaging import MessageBus
from aion.systems.agents.team import TeamManager
from aion.systems.agents.delegation import TaskDelegator
from aion.systems.agents.consensus import ConsensusEngine
from aion.systems.agents.persistence import MultiAgentPersistence

# Workflows
from aion.systems.agents.workflows import (
    WorkflowExecutor,
    get_workflow_executor,
    SequentialWorkflow,
    ParallelWorkflow,
    HierarchicalWorkflow,
    DebateWorkflow,
    SwarmWorkflow,
)

# Archetypes
from aion.systems.agents.archetypes import (
    BaseSpecialist,
    ResearcherAgent,
    CoderAgent,
    AnalystAgent,
    WriterAgent,
    ReviewerAgent,
    PlannerAgent,
    ExecutorAgent,
)

__all__ = [
    # === Core Enums ===
    "AgentRole",
    "AgentStatus",
    "MessageType",
    "TeamStatus",
    "WorkflowPattern",
    "ConsensusMethod",
    "TaskPriority",

    # === Core Types ===
    "AgentCapability",
    "AgentProfile",
    "AgentInstance",
    "Message",
    "TeamTask",
    "Team",
    "ConsensusVote",
    "ConsensusResult",
    "WorkflowStep",
    "WorkflowExecution",
    "OrchestratorStats",

    # === Main Orchestrator ===
    "MultiAgentOrchestrator",

    # === Core Components ===
    "AgentPool",
    "MessageBus",
    "TeamManager",
    "TaskDelegator",
    "ConsensusEngine",
    "MultiAgentPersistence",

    # === Workflows ===
    "WorkflowExecutor",
    "get_workflow_executor",
    "SequentialWorkflow",
    "ParallelWorkflow",
    "HierarchicalWorkflow",
    "DebateWorkflow",
    "SwarmWorkflow",

    # === Archetypes ===
    "BaseSpecialist",
    "ResearcherAgent",
    "CoderAgent",
    "AnalystAgent",
    "WriterAgent",
    "ReviewerAgent",
    "PlannerAgent",
    "ExecutorAgent",
]
