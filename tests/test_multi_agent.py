"""
AION Multi-Agent Orchestration System Tests

Comprehensive test suite for the multi-agent orchestration system.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import shutil
from unittest.mock import AsyncMock, MagicMock, patch


class TestAgentTypes:
    """Test agent type definitions and dataclasses."""

    def test_agent_role_enum(self):
        """Test AgentRole enum values."""
        from aion.systems.agents.types import AgentRole

        assert AgentRole.RESEARCHER.value == "researcher"
        assert AgentRole.CODER.value == "coder"
        assert AgentRole.ANALYST.value == "analyst"
        assert AgentRole.WRITER.value == "writer"
        assert AgentRole.REVIEWER.value == "reviewer"
        assert AgentRole.PLANNER.value == "planner"
        assert AgentRole.EXECUTOR.value == "executor"
        assert AgentRole.GENERALIST.value == "generalist"

    def test_agent_status_enum(self):
        """Test AgentStatus enum values."""
        from aion.systems.agents.types import AgentStatus

        assert AgentStatus.IDLE.value == "idle"
        assert AgentStatus.BUSY.value == "busy"
        assert AgentStatus.WAITING.value == "waiting"
        assert AgentStatus.ERROR.value == "error"
        assert AgentStatus.TERMINATED.value == "terminated"

    def test_workflow_pattern_enum(self):
        """Test WorkflowPattern enum values."""
        from aion.systems.agents.types import WorkflowPattern

        assert WorkflowPattern.SEQUENTIAL.value == "sequential"
        assert WorkflowPattern.PARALLEL.value == "parallel"
        assert WorkflowPattern.HIERARCHICAL.value == "hierarchical"
        assert WorkflowPattern.DEBATE.value == "debate"
        assert WorkflowPattern.SWARM.value == "swarm"

    def test_agent_capability_creation(self):
        """Test AgentCapability dataclass."""
        from aion.systems.agents.types import AgentCapability

        capability = AgentCapability(
            name="web_search",
            description="Can search the web",
            proficiency=0.9,
        )

        assert capability.name == "web_search"
        assert capability.proficiency == 0.9
        assert capability.enabled

    def test_agent_profile_creation(self):
        """Test AgentProfile dataclass."""
        from aion.systems.agents.types import AgentProfile, AgentRole, AgentCapability

        profile = AgentProfile(
            role=AgentRole.RESEARCHER,
            name="Research Agent",
            description="Specialized in research",
            capabilities=[
                AgentCapability(name="search", description="Search capability"),
            ],
            system_prompt="You are a research agent.",
        )

        assert profile.role == AgentRole.RESEARCHER
        assert profile.name == "Research Agent"
        assert len(profile.capabilities) == 1
        assert profile.id is not None

    def test_agent_instance_creation(self):
        """Test AgentInstance dataclass."""
        from aion.systems.agents.types import (
            AgentInstance,
            AgentProfile,
            AgentRole,
            AgentStatus,
        )

        profile = AgentProfile(
            role=AgentRole.CODER,
            name="Code Agent",
            description="Coding specialist",
            system_prompt="You write code.",
        )

        instance = AgentInstance(profile=profile)

        assert instance.profile == profile
        assert instance.status == AgentStatus.IDLE
        assert instance.tasks_completed == 0
        assert instance.id is not None

    def test_agent_instance_serialization(self):
        """Test AgentInstance to_dict and from_dict."""
        from aion.systems.agents.types import (
            AgentInstance,
            AgentProfile,
            AgentRole,
        )

        profile = AgentProfile(
            role=AgentRole.ANALYST,
            name="Analyst Agent",
            description="Data analysis",
            system_prompt="You analyze data.",
        )

        original = AgentInstance(profile=profile)
        original.tasks_completed = 5

        data = original.to_dict()
        restored = AgentInstance.from_dict(data)

        assert restored.id == original.id
        assert restored.profile.role == original.profile.role
        assert restored.tasks_completed == 5

    def test_message_creation(self):
        """Test Message dataclass."""
        from aion.systems.agents.types import Message, MessageType

        message = Message(
            sender_id="agent-1",
            receiver_id="agent-2",
            message_type=MessageType.REQUEST,
            content={"task": "analyze data"},
        )

        assert message.sender_id == "agent-1"
        assert message.receiver_id == "agent-2"
        assert message.message_type == MessageType.REQUEST
        assert message.priority == 5
        assert message.id is not None

    def test_team_task_creation(self):
        """Test TeamTask dataclass."""
        from aion.systems.agents.types import TeamTask, WorkflowPattern

        task = TeamTask(
            title="Research Task",
            description="Research a topic",
            objective="Comprehensive report",
            success_criteria=["Accuracy", "Completeness"],
            workflow_pattern=WorkflowPattern.SEQUENTIAL,
        )

        assert task.title == "Research Task"
        assert len(task.success_criteria) == 2
        assert task.workflow_pattern == WorkflowPattern.SEQUENTIAL
        assert task.id is not None

    def test_team_creation(self):
        """Test Team dataclass."""
        from aion.systems.agents.types import Team, TeamStatus, WorkflowPattern

        team = Team(
            name="Research Team",
            purpose="Research tasks",
            agent_ids=["agent-1", "agent-2"],
            workflow=WorkflowPattern.PARALLEL,
        )

        assert team.name == "Research Team"
        assert len(team.agent_ids) == 2
        assert team.status == TeamStatus.FORMING
        assert team.id is not None

    def test_team_serialization(self):
        """Test Team to_dict and from_dict."""
        from aion.systems.agents.types import Team, WorkflowPattern

        original = Team(
            name="Test Team",
            purpose="Testing",
            agent_ids=["a1", "a2"],
            workflow=WorkflowPattern.DEBATE,
        )

        data = original.to_dict()
        restored = Team.from_dict(data)

        assert restored.id == original.id
        assert restored.name == original.name
        assert restored.workflow == original.workflow

    def test_consensus_vote(self):
        """Test ConsensusVote dataclass."""
        from aion.systems.agents.types import ConsensusVote

        vote = ConsensusVote(
            agent_id="agent-1",
            topic_id="topic-1",
            choice="option-a",
            confidence=0.85,
            reasoning="This is the best option.",
        )

        assert vote.agent_id == "agent-1"
        assert vote.choice == "option-a"
        assert vote.confidence == 0.85

    def test_workflow_step(self):
        """Test WorkflowStep dataclass."""
        from aion.systems.agents.types import WorkflowStep

        step = WorkflowStep(
            step_number=1,
            agent_id="agent-1",
            action="research",
            input_data={"topic": "AI"},
        )

        assert step.step_number == 1
        assert step.action == "research"
        assert step.status == "pending"


class TestAgentPool:
    """Test agent pool functionality."""

    @pytest.fixture
    def pool(self):
        """Create an agent pool for testing."""
        from aion.systems.agents.pool import AgentPool

        return AgentPool(max_agents=10)

    @pytest.mark.asyncio
    async def test_pool_initialization(self, pool):
        """Test pool initialization."""
        await pool.initialize()

        assert pool._initialized

        await pool.shutdown()

    @pytest.mark.asyncio
    async def test_spawn_agent(self, pool):
        """Test spawning an agent."""
        from aion.systems.agents.types import AgentRole, AgentStatus

        await pool.initialize()

        agent = await pool.spawn_agent(role=AgentRole.RESEARCHER)

        assert agent is not None
        assert agent.profile.role == AgentRole.RESEARCHER
        assert agent.status == AgentStatus.IDLE

        await pool.shutdown()

    @pytest.mark.asyncio
    async def test_spawn_agent_with_name(self, pool):
        """Test spawning an agent with custom name."""
        from aion.systems.agents.types import AgentRole

        await pool.initialize()

        agent = await pool.spawn_agent(
            role=AgentRole.CODER,
            name_override="Custom Coder",
        )

        assert agent.profile.name == "Custom Coder"

        await pool.shutdown()

    @pytest.mark.asyncio
    async def test_get_agent(self, pool):
        """Test getting an agent by ID."""
        from aion.systems.agents.types import AgentRole

        await pool.initialize()

        agent = await pool.spawn_agent(role=AgentRole.WRITER)
        retrieved = pool.get_agent(agent.id)

        assert retrieved is not None
        assert retrieved.id == agent.id

        await pool.shutdown()

    @pytest.mark.asyncio
    async def test_terminate_agent(self, pool):
        """Test terminating an agent."""
        from aion.systems.agents.types import AgentRole, AgentStatus

        await pool.initialize()

        agent = await pool.spawn_agent(role=AgentRole.ANALYST)
        success = await pool.terminate_agent(agent.id)

        assert success
        assert agent.status == AgentStatus.TERMINATED

        await pool.shutdown()

    @pytest.mark.asyncio
    async def test_list_agents_by_role(self, pool):
        """Test listing agents by role."""
        from aion.systems.agents.types import AgentRole

        await pool.initialize()

        await pool.spawn_agent(role=AgentRole.RESEARCHER)
        await pool.spawn_agent(role=AgentRole.RESEARCHER)
        await pool.spawn_agent(role=AgentRole.CODER)

        researchers = pool.list_agents(role=AgentRole.RESEARCHER)
        assert len(researchers) == 2

        coders = pool.list_agents(role=AgentRole.CODER)
        assert len(coders) == 1

        await pool.shutdown()

    @pytest.mark.asyncio
    async def test_list_agents_by_status(self, pool):
        """Test listing agents by status."""
        from aion.systems.agents.types import AgentRole, AgentStatus

        await pool.initialize()

        agent1 = await pool.spawn_agent(role=AgentRole.WRITER)
        agent2 = await pool.spawn_agent(role=AgentRole.REVIEWER)

        pool.update_status(agent1.id, AgentStatus.BUSY)

        idle_agents = pool.list_agents(status=AgentStatus.IDLE)
        assert len(idle_agents) == 1

        busy_agents = pool.list_agents(status=AgentStatus.BUSY)
        assert len(busy_agents) == 1

        await pool.shutdown()

    @pytest.mark.asyncio
    async def test_find_capable_agent(self, pool):
        """Test finding an agent by capability."""
        from aion.systems.agents.types import AgentRole

        await pool.initialize()

        await pool.spawn_agent(role=AgentRole.RESEARCHER)

        # Researchers should have research capability
        agent = pool.find_capable_agent("research")
        assert agent is not None
        assert agent.profile.role == AgentRole.RESEARCHER

        await pool.shutdown()

    @pytest.mark.asyncio
    async def test_pool_max_agents(self, pool):
        """Test max agents limit."""
        from aion.systems.agents.types import AgentRole

        await pool.initialize()

        # Spawn up to max
        for _ in range(pool.max_agents):
            await pool.spawn_agent(role=AgentRole.GENERALIST)

        # Try to spawn one more
        agent = await pool.spawn_agent(role=AgentRole.GENERALIST)
        assert agent is None  # Should fail

        await pool.shutdown()

    @pytest.mark.asyncio
    async def test_pool_stats(self, pool):
        """Test pool statistics."""
        from aion.systems.agents.types import AgentRole

        await pool.initialize()

        await pool.spawn_agent(role=AgentRole.RESEARCHER)
        await pool.spawn_agent(role=AgentRole.CODER)

        stats = pool.get_stats()

        assert stats["total_agents"] == 2
        assert stats["max_agents"] == 10
        assert "by_role" in stats
        assert "by_status" in stats

        await pool.shutdown()


class TestMessageBus:
    """Test message bus functionality."""

    @pytest.fixture
    def bus(self):
        """Create a message bus for testing."""
        from aion.systems.agents.messaging import MessageBus

        return MessageBus()

    @pytest.mark.asyncio
    async def test_bus_initialization(self, bus):
        """Test bus initialization."""
        await bus.initialize()

        assert bus._initialized

        await bus.shutdown()

    @pytest.mark.asyncio
    async def test_send_message(self, bus):
        """Test sending a message."""
        from aion.systems.agents.types import Message, MessageType

        await bus.initialize()

        message = Message(
            sender_id="agent-1",
            receiver_id="agent-2",
            message_type=MessageType.INFORM,
            content={"data": "test"},
        )

        success = await bus.send(message)
        assert success

        stats = bus.get_stats()
        assert stats["messages_sent"] == 1

        await bus.shutdown()

    @pytest.mark.asyncio
    async def test_receive_message(self, bus):
        """Test receiving a message."""
        from aion.systems.agents.types import Message, MessageType

        await bus.initialize()

        message = Message(
            sender_id="agent-1",
            receiver_id="agent-2",
            message_type=MessageType.INFORM,
            content={"data": "test"},
        )

        await bus.send(message)
        received = await bus.receive("agent-2", timeout=1.0)

        assert received is not None
        assert received.sender_id == "agent-1"
        assert received.content["data"] == "test"

        await bus.shutdown()

    @pytest.mark.asyncio
    async def test_broadcast(self, bus):
        """Test broadcasting a message."""
        from aion.systems.agents.types import Message, MessageType

        await bus.initialize()

        # Join some agents to a team
        bus.join_team("agent-1", "team-1")
        bus.join_team("agent-2", "team-1")
        bus.join_team("agent-3", "team-1")

        message = Message(
            sender_id="agent-1",
            receiver_id="team-1",
            message_type=MessageType.BROADCAST,
            content={"announcement": "hello"},
        )

        count = await bus.broadcast(message, "team-1")
        assert count == 2  # All except sender

        stats = bus.get_stats()
        assert stats["broadcasts"] == 1

        await bus.shutdown()

    @pytest.mark.asyncio
    async def test_subscribe_topic(self, bus):
        """Test topic subscription."""
        await bus.initialize()

        bus.subscribe("agent-1", "updates")
        bus.subscribe("agent-2", "updates")

        subs = bus.get_topic_subscribers("updates")
        assert len(subs) == 2
        assert "agent-1" in subs
        assert "agent-2" in subs

        await bus.shutdown()

    @pytest.mark.asyncio
    async def test_publish_topic(self, bus):
        """Test publishing to a topic."""
        from aion.systems.agents.types import Message, MessageType

        await bus.initialize()

        bus.subscribe("agent-1", "news")
        bus.subscribe("agent-2", "news")

        message = Message(
            sender_id="system",
            receiver_id="news",
            message_type=MessageType.INFORM,
            content={"news": "update"},
        )

        count = await bus.publish("news", message)
        assert count == 2

        await bus.shutdown()

    @pytest.mark.asyncio
    async def test_request_response(self, bus):
        """Test request-response pattern."""
        from aion.systems.agents.types import MessageType

        await bus.initialize()

        # Simulate a responder
        async def mock_responder():
            msg = await bus.receive("agent-2", timeout=2.0)
            if msg:
                response = await bus.create_response(
                    msg,
                    content={"answer": 42},
                )
                await bus.send(response)

        # Start responder task
        responder_task = asyncio.create_task(mock_responder())

        # Send request
        response = await bus.request(
            from_agent="agent-1",
            to_agent="agent-2",
            content={"question": "meaning of life"},
            timeout=3.0,
        )

        await responder_task

        assert response is not None
        assert response["answer"] == 42

        await bus.shutdown()

    @pytest.mark.asyncio
    async def test_message_history(self, bus):
        """Test message history retrieval."""
        from aion.systems.agents.types import Message, MessageType

        await bus.initialize()

        for i in range(5):
            message = Message(
                sender_id="agent-1",
                receiver_id="agent-2",
                message_type=MessageType.INFORM,
                content={"index": i},
            )
            await bus.send(message)

        history = bus.get_history("agent-2", limit=3)
        assert len(history) == 3

        await bus.shutdown()


class TestTeamManager:
    """Test team manager functionality."""

    @pytest.fixture
    def team_setup(self):
        """Create team manager with dependencies."""
        from aion.systems.agents.pool import AgentPool
        from aion.systems.agents.messaging import MessageBus
        from aion.systems.agents.team import TeamManager

        pool = AgentPool(max_agents=20)
        bus = MessageBus()
        team_manager = TeamManager(pool, bus)

        return pool, bus, team_manager

    @pytest.mark.asyncio
    async def test_team_creation(self, team_setup):
        """Test creating a team."""
        from aion.systems.agents.types import AgentRole, WorkflowPattern

        pool, bus, manager = team_setup

        await pool.initialize()
        await bus.initialize()
        await manager.initialize()

        team = await manager.create_team(
            name="Research Team",
            purpose="Research tasks",
            roles=[AgentRole.RESEARCHER, AgentRole.ANALYST],
            workflow=WorkflowPattern.SEQUENTIAL,
        )

        assert team is not None
        assert team.name == "Research Team"
        assert len(team.agent_ids) == 2

        await manager.shutdown()
        await bus.shutdown()
        await pool.shutdown()

    @pytest.mark.asyncio
    async def test_create_team_for_task(self, team_setup):
        """Test creating a team for a specific task."""
        from aion.systems.agents.types import TeamTask, WorkflowPattern

        pool, bus, manager = team_setup

        await pool.initialize()
        await bus.initialize()
        await manager.initialize()

        task = TeamTask(
            title="Research AI trends",
            description="Research current trends in artificial intelligence",
            objective="Comprehensive report",
            workflow_pattern=WorkflowPattern.SEQUENTIAL,
        )

        team = await manager.create_team_for_task(task)

        assert team is not None
        assert len(team.agent_ids) > 0

        await manager.shutdown()
        await bus.shutdown()
        await pool.shutdown()

    @pytest.mark.asyncio
    async def test_disband_team(self, team_setup):
        """Test disbanding a team."""
        from aion.systems.agents.types import AgentRole, WorkflowPattern, TeamStatus

        pool, bus, manager = team_setup

        await pool.initialize()
        await bus.initialize()
        await manager.initialize()

        team = await manager.create_team(
            name="Temp Team",
            purpose="Temporary",
            roles=[AgentRole.GENERALIST],
            workflow=WorkflowPattern.SEQUENTIAL,
        )

        await manager.disband_team(team.id)

        assert team.status == TeamStatus.DISBANDED

        await manager.shutdown()
        await bus.shutdown()
        await pool.shutdown()

    @pytest.mark.asyncio
    async def test_list_teams(self, team_setup):
        """Test listing teams."""
        from aion.systems.agents.types import AgentRole, WorkflowPattern

        pool, bus, manager = team_setup

        await pool.initialize()
        await bus.initialize()
        await manager.initialize()

        await manager.create_team(
            name="Team 1",
            purpose="Purpose 1",
            roles=[AgentRole.WRITER],
            workflow=WorkflowPattern.SEQUENTIAL,
        )

        await manager.create_team(
            name="Team 2",
            purpose="Purpose 2",
            roles=[AgentRole.CODER],
            workflow=WorkflowPattern.SEQUENTIAL,
        )

        teams = manager.list_teams()
        assert len(teams) == 2

        await manager.shutdown()
        await bus.shutdown()
        await pool.shutdown()

    @pytest.mark.asyncio
    async def test_team_stats(self, team_setup):
        """Test team statistics."""
        from aion.systems.agents.types import AgentRole, WorkflowPattern

        pool, bus, manager = team_setup

        await pool.initialize()
        await bus.initialize()
        await manager.initialize()

        await manager.create_team(
            name="Stats Team",
            purpose="Testing stats",
            roles=[AgentRole.REVIEWER],
            workflow=WorkflowPattern.SEQUENTIAL,
        )

        stats = manager.get_stats()

        assert stats["total_teams"] == 1
        assert "by_status" in stats

        await manager.shutdown()
        await bus.shutdown()
        await pool.shutdown()


class TestTaskDelegator:
    """Test task delegator functionality."""

    @pytest.fixture
    def delegator_setup(self):
        """Create task delegator with dependencies."""
        from aion.systems.agents.pool import AgentPool
        from aion.systems.agents.messaging import MessageBus
        from aion.systems.agents.team import TeamManager
        from aion.systems.agents.delegation import TaskDelegator

        pool = AgentPool(max_agents=20)
        bus = MessageBus()
        team_manager = TeamManager(pool, bus)
        delegator = TaskDelegator(pool, team_manager)

        return pool, bus, team_manager, delegator

    @pytest.mark.asyncio
    async def test_analyze_task(self, delegator_setup):
        """Test task analysis."""
        from aion.systems.agents.types import TeamTask

        pool, bus, manager, delegator = delegator_setup

        task = TeamTask(
            title="Research and analyze data",
            description="Research AI trends and analyze the market data",
            objective="Comprehensive analysis",
        )

        analysis = delegator.analyze_task(task)

        assert "task_type" in analysis
        assert "recommended_roles" in analysis
        assert "complexity" in analysis

    @pytest.mark.asyncio
    async def test_recommend_workflow(self, delegator_setup):
        """Test workflow recommendation."""
        from aion.systems.agents.types import TeamTask, WorkflowPattern

        pool, bus, manager, delegator = delegator_setup

        # Simple task should get sequential
        simple_task = TeamTask(
            title="Write report",
            description="Write a simple report",
            objective="Report complete",
        )

        workflow = delegator.recommend_workflow(simple_task)
        assert workflow in WorkflowPattern

    @pytest.mark.asyncio
    async def test_recommend_team_size(self, delegator_setup):
        """Test team size recommendation."""
        from aion.systems.agents.types import TeamTask

        pool, bus, manager, delegator = delegator_setup

        task = TeamTask(
            title="Complex project",
            description="A very complex project requiring extensive research, coding, analysis, and review",
            objective="Complete project",
            success_criteria=["Research done", "Code written", "Analysis complete", "Reviewed"],
        )

        size = delegator.recommend_team_size(task)
        assert size >= 2
        assert size <= 7

    @pytest.mark.asyncio
    async def test_delegator_stats(self, delegator_setup):
        """Test delegator statistics."""
        from aion.systems.agents.types import TeamTask

        pool, bus, manager, delegator = delegator_setup

        task = TeamTask(
            title="Test task",
            description="Testing delegator",
            objective="Test complete",
        )

        delegator.analyze_task(task)

        stats = delegator.get_stats()
        assert stats["tasks_analyzed"] == 1


class TestConsensusEngine:
    """Test consensus engine functionality."""

    @pytest.fixture
    def consensus_setup(self):
        """Create consensus engine with dependencies."""
        from aion.systems.agents.messaging import MessageBus
        from aion.systems.agents.consensus import ConsensusEngine

        bus = MessageBus()
        consensus = ConsensusEngine(bus)

        return bus, consensus

    @pytest.mark.asyncio
    async def test_majority_voting(self, consensus_setup):
        """Test majority voting."""
        from aion.systems.agents.types import ConsensusVote, ConsensusMethod

        bus, consensus = consensus_setup

        votes = [
            ConsensusVote(agent_id="a1", topic_id="t1", choice="A", confidence=0.8),
            ConsensusVote(agent_id="a2", topic_id="t1", choice="A", confidence=0.7),
            ConsensusVote(agent_id="a3", topic_id="t1", choice="B", confidence=0.9),
        ]

        result = consensus.resolve(
            topic_id="t1",
            votes=votes,
            method=ConsensusMethod.MAJORITY,
        )

        assert result.winner == "A"
        assert result.vote_count == 3

    @pytest.mark.asyncio
    async def test_weighted_voting(self, consensus_setup):
        """Test weighted voting by confidence."""
        from aion.systems.agents.types import ConsensusVote, ConsensusMethod

        bus, consensus = consensus_setup

        # B has higher total confidence weight
        votes = [
            ConsensusVote(agent_id="a1", topic_id="t1", choice="A", confidence=0.3),
            ConsensusVote(agent_id="a2", topic_id="t1", choice="A", confidence=0.3),
            ConsensusVote(agent_id="a3", topic_id="t1", choice="B", confidence=0.95),
        ]

        result = consensus.resolve(
            topic_id="t1",
            votes=votes,
            method=ConsensusMethod.WEIGHTED,
        )

        assert result.winner == "B"

    @pytest.mark.asyncio
    async def test_unanimous_voting(self, consensus_setup):
        """Test unanimous voting requirement."""
        from aion.systems.agents.types import ConsensusVote, ConsensusMethod

        bus, consensus = consensus_setup

        # Not unanimous
        votes = [
            ConsensusVote(agent_id="a1", topic_id="t1", choice="A", confidence=0.8),
            ConsensusVote(agent_id="a2", topic_id="t1", choice="A", confidence=0.7),
            ConsensusVote(agent_id="a3", topic_id="t1", choice="B", confidence=0.9),
        ]

        result = consensus.resolve(
            topic_id="t1",
            votes=votes,
            method=ConsensusMethod.UNANIMOUS,
        )

        assert not result.consensus_reached

    @pytest.mark.asyncio
    async def test_consensus_stats(self, consensus_setup):
        """Test consensus statistics."""
        from aion.systems.agents.types import ConsensusVote, ConsensusMethod

        bus, consensus = consensus_setup

        votes = [
            ConsensusVote(agent_id="a1", topic_id="t1", choice="A", confidence=0.8),
            ConsensusVote(agent_id="a2", topic_id="t1", choice="A", confidence=0.7),
        ]

        consensus.resolve(
            topic_id="t1",
            votes=votes,
            method=ConsensusMethod.MAJORITY,
        )

        stats = consensus.get_stats()
        assert stats["total_decisions"] == 1


class TestWorkflows:
    """Test workflow executors."""

    @pytest.fixture
    def workflow_setup(self):
        """Create workflow dependencies."""
        from aion.systems.agents.pool import AgentPool
        from aion.systems.agents.messaging import MessageBus

        pool = AgentPool(max_agents=20)
        bus = MessageBus()

        return pool, bus

    def test_get_workflow_executor(self, workflow_setup):
        """Test workflow executor factory."""
        from aion.systems.agents.types import WorkflowPattern
        from aion.systems.agents.workflows import (
            get_workflow_executor,
            SequentialWorkflow,
            ParallelWorkflow,
            HierarchicalWorkflow,
            DebateWorkflow,
            SwarmWorkflow,
        )

        pool, bus = workflow_setup

        sequential = get_workflow_executor(WorkflowPattern.SEQUENTIAL, pool, bus)
        assert isinstance(sequential, SequentialWorkflow)

        parallel = get_workflow_executor(WorkflowPattern.PARALLEL, pool, bus)
        assert isinstance(parallel, ParallelWorkflow)

        hierarchical = get_workflow_executor(WorkflowPattern.HIERARCHICAL, pool, bus)
        assert isinstance(hierarchical, HierarchicalWorkflow)

        debate = get_workflow_executor(WorkflowPattern.DEBATE, pool, bus)
        assert isinstance(debate, DebateWorkflow)

        swarm = get_workflow_executor(WorkflowPattern.SWARM, pool, bus)
        assert isinstance(swarm, SwarmWorkflow)

    def test_workflow_executor_properties(self, workflow_setup):
        """Test workflow executor properties."""
        from aion.systems.agents.types import WorkflowPattern
        from aion.systems.agents.workflows import SequentialWorkflow

        pool, bus = workflow_setup

        workflow = SequentialWorkflow(pool, bus)

        assert workflow.pattern == WorkflowPattern.SEQUENTIAL
        assert workflow.pool == pool
        assert workflow.bus == bus


class TestPersistence:
    """Test persistence layer."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory."""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path)

    @pytest.fixture
    def persistence(self, temp_dir):
        """Create persistence layer for testing."""
        from aion.systems.agents.persistence import MultiAgentPersistence

        return MultiAgentPersistence(storage_dir=temp_dir)

    def test_save_and_load_agent(self, persistence):
        """Test saving and loading agent state."""
        from aion.systems.agents.types import (
            AgentInstance,
            AgentProfile,
            AgentRole,
        )

        profile = AgentProfile(
            role=AgentRole.RESEARCHER,
            name="Test Agent",
            description="Testing",
            system_prompt="Test prompt",
        )

        agent = AgentInstance(profile=profile)
        agent.tasks_completed = 5

        # Save
        success = persistence.save_agent(agent)
        assert success

        # Load
        loaded = persistence.load_agent(agent.id)
        assert loaded is not None
        assert loaded.id == agent.id
        assert loaded.tasks_completed == 5

    def test_list_saved_agents(self, persistence):
        """Test listing saved agents."""
        from aion.systems.agents.types import (
            AgentInstance,
            AgentProfile,
            AgentRole,
        )

        for i in range(3):
            profile = AgentProfile(
                role=AgentRole.GENERALIST,
                name=f"Agent {i}",
                description="Testing",
                system_prompt="Test",
            )
            agent = AgentInstance(profile=profile)
            persistence.save_agent(agent)

        saved = persistence.list_saved_agents()
        assert len(saved) == 3

    def test_delete_agent(self, persistence):
        """Test deleting saved agent."""
        from aion.systems.agents.types import (
            AgentInstance,
            AgentProfile,
            AgentRole,
        )

        profile = AgentProfile(
            role=AgentRole.WRITER,
            name="To Delete",
            description="Will be deleted",
            system_prompt="Test",
        )
        agent = AgentInstance(profile=profile)

        persistence.save_agent(agent)
        assert persistence.load_agent(agent.id) is not None

        persistence.delete_agent(agent.id)
        assert persistence.load_agent(agent.id) is None

    def test_save_and_load_team(self, persistence):
        """Test saving and loading team state."""
        from aion.systems.agents.types import Team, WorkflowPattern

        team = Team(
            name="Test Team",
            purpose="Testing persistence",
            agent_ids=["a1", "a2"],
            workflow=WorkflowPattern.PARALLEL,
        )

        success = persistence.save_team(team)
        assert success

        loaded = persistence.load_team(team.id)
        assert loaded is not None
        assert loaded.name == "Test Team"
        assert len(loaded.agent_ids) == 2

    def test_save_and_load_task(self, persistence):
        """Test saving and loading task state."""
        from aion.systems.agents.types import TeamTask, WorkflowPattern

        task = TeamTask(
            title="Test Task",
            description="Testing task persistence",
            objective="Test objective",
            workflow_pattern=WorkflowPattern.SEQUENTIAL,
        )

        success = persistence.save_task(task)
        assert success

        loaded = persistence.load_task(task.id)
        assert loaded is not None
        assert loaded.title == "Test Task"

    def test_save_message_history(self, persistence):
        """Test saving and loading message history."""
        from aion.systems.agents.types import Message, MessageType

        messages = [
            Message(
                sender_id="a1",
                receiver_id="a2",
                message_type=MessageType.INFORM,
                content={"msg": f"Message {i}"},
            )
            for i in range(5)
        ]

        success = persistence.save_message_history(messages, "context-1")
        assert success

        loaded = persistence.load_message_history("context-1")
        assert len(loaded) == 5

    def test_save_and_load_snapshot(self, persistence):
        """Test saving and loading system snapshot."""
        from aion.systems.agents.types import (
            AgentInstance,
            AgentProfile,
            AgentRole,
            Team,
            TeamTask,
            WorkflowPattern,
        )

        agents = [
            AgentInstance(
                profile=AgentProfile(
                    role=AgentRole.RESEARCHER,
                    name="Agent 1",
                    description="Test",
                    system_prompt="Test",
                )
            )
        ]

        teams = [
            Team(
                name="Team 1",
                purpose="Test",
                agent_ids=["a1"],
                workflow=WorkflowPattern.SEQUENTIAL,
            )
        ]

        tasks = [
            TeamTask(
                title="Task 1",
                description="Test task",
                objective="Test",
            )
        ]

        snapshot_id = persistence.save_snapshot(agents, teams, tasks)
        assert snapshot_id is not None

        loaded = persistence.load_snapshot(snapshot_id)
        assert loaded is not None
        assert len(loaded["agents"]) == 1
        assert len(loaded["teams"]) == 1
        assert len(loaded["tasks"]) == 1

    def test_list_snapshots(self, persistence):
        """Test listing snapshots."""
        from aion.systems.agents.types import (
            AgentInstance,
            AgentProfile,
            AgentRole,
        )

        agents = [
            AgentInstance(
                profile=AgentProfile(
                    role=AgentRole.GENERALIST,
                    name="Test",
                    description="Test",
                    system_prompt="Test",
                )
            )
        ]

        persistence.save_snapshot(agents, [], [], snapshot_id="snap1")
        persistence.save_snapshot(agents, [], [], snapshot_id="snap2")

        snapshots = persistence.list_snapshots()
        assert len(snapshots) == 2
        assert "snap1" in snapshots
        assert "snap2" in snapshots

    def test_storage_stats(self, persistence):
        """Test storage statistics."""
        from aion.systems.agents.types import (
            AgentInstance,
            AgentProfile,
            AgentRole,
        )

        # Save some data
        for i in range(3):
            agent = AgentInstance(
                profile=AgentProfile(
                    role=AgentRole.ANALYST,
                    name=f"Agent {i}",
                    description="Test",
                    system_prompt="Test",
                )
            )
            persistence.save_agent(agent)

        stats = persistence.get_storage_stats()

        assert stats["agents_saved"] == 3
        assert stats["total_size_bytes"] > 0


class TestOrchestrator:
    """Test multi-agent orchestrator."""

    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator for testing."""
        from aion.systems.agents.orchestrator import MultiAgentOrchestrator

        return MultiAgentOrchestrator(max_agents=10, max_teams=5)

    @pytest.mark.asyncio
    async def test_orchestrator_initialization(self, orchestrator):
        """Test orchestrator initialization."""
        await orchestrator.initialize()

        assert orchestrator._initialized

        await orchestrator.shutdown()

    @pytest.mark.asyncio
    async def test_spawn_agent_via_orchestrator(self, orchestrator):
        """Test spawning agent through orchestrator."""
        from aion.systems.agents.types import AgentRole

        await orchestrator.initialize()

        agent_data = await orchestrator.spawn_agent(
            role=AgentRole.RESEARCHER,
            name="Test Researcher",
        )

        assert agent_data is not None
        assert agent_data["profile"]["role"] == "researcher"

        await orchestrator.shutdown()

    @pytest.mark.asyncio
    async def test_list_agents_via_orchestrator(self, orchestrator):
        """Test listing agents through orchestrator."""
        from aion.systems.agents.types import AgentRole

        await orchestrator.initialize()

        await orchestrator.spawn_agent(role=AgentRole.CODER)
        await orchestrator.spawn_agent(role=AgentRole.WRITER)

        agents = orchestrator.list_agents()
        assert len(agents) == 2

        await orchestrator.shutdown()

    @pytest.mark.asyncio
    async def test_terminate_agent_via_orchestrator(self, orchestrator):
        """Test terminating agent through orchestrator."""
        from aion.systems.agents.types import AgentRole

        await orchestrator.initialize()

        agent_data = await orchestrator.spawn_agent(role=AgentRole.ANALYST)
        agent_id = agent_data["id"]

        success = await orchestrator.terminate_agent(agent_id)
        assert success

        await orchestrator.shutdown()

    @pytest.mark.asyncio
    async def test_orchestrator_stats(self, orchestrator):
        """Test orchestrator statistics."""
        from aion.systems.agents.types import AgentRole

        await orchestrator.initialize()

        await orchestrator.spawn_agent(role=AgentRole.PLANNER)

        stats = orchestrator.get_stats()

        assert stats["initialized"]
        assert "pool" in stats
        assert "teams" in stats
        assert "messaging" in stats
        assert "consensus" in stats

        await orchestrator.shutdown()

    @pytest.mark.asyncio
    async def test_orchestrator_typed_stats(self, orchestrator):
        """Test orchestrator typed statistics object."""
        from aion.systems.agents.types import OrchestratorStats

        await orchestrator.initialize()

        stats = orchestrator.get_orchestrator_stats()

        assert isinstance(stats, OrchestratorStats)
        assert stats.total_agents >= 0
        assert stats.total_teams >= 0

        await orchestrator.shutdown()


class TestModuleExports:
    """Test module exports and imports."""

    def test_main_module_exports(self):
        """Test main module exports all expected classes."""
        from aion.systems.agents import (
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
            # Main Orchestrator
            MultiAgentOrchestrator,
            # Core Components
            AgentPool,
            MessageBus,
            TeamManager,
            TaskDelegator,
            ConsensusEngine,
            MultiAgentPersistence,
            # Workflows
            WorkflowExecutor,
            get_workflow_executor,
            SequentialWorkflow,
            ParallelWorkflow,
            HierarchicalWorkflow,
            DebateWorkflow,
            SwarmWorkflow,
            # Archetypes
            BaseSpecialist,
            ResearcherAgent,
            CoderAgent,
            AnalystAgent,
            WriterAgent,
            ReviewerAgent,
            PlannerAgent,
            ExecutorAgent,
        )

        # Just verify imports work
        assert AgentRole is not None
        assert MultiAgentOrchestrator is not None
        assert get_workflow_executor is not None

    def test_workflow_exports(self):
        """Test workflow module exports."""
        from aion.systems.agents.workflows import (
            WorkflowExecutor,
            get_workflow_executor,
            SequentialWorkflow,
            ParallelWorkflow,
            HierarchicalWorkflow,
            DebateWorkflow,
            SwarmWorkflow,
        )

        assert WorkflowExecutor is not None
        assert get_workflow_executor is not None

    def test_archetype_exports(self):
        """Test archetype module exports."""
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

        assert BaseSpecialist is not None
        assert ResearcherAgent is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
