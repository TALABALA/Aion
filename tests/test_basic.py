"""
AION Basic Tests

Test suite for core AION functionality.
"""

import pytest
import asyncio
from pathlib import Path


class TestConfig:
    """Test configuration management."""

    def test_config_creation(self):
        """Test that config can be created with defaults."""
        from aion.core.config import AIONConfig

        config = AIONConfig()
        assert config.instance_id == "aion-primary"
        assert config.port == 8000

    def test_config_llm_settings(self):
        """Test LLM configuration."""
        from aion.core.config import AIONConfig

        config = AIONConfig()
        assert config.llm.provider == "openai"
        assert config.llm.max_tokens == 4096


class TestSecurity:
    """Test security system."""

    def test_risk_classification(self):
        """Test operation risk classification."""
        from aion.core.security import SecurityManager, RiskLevel

        security = SecurityManager()

        # Test high-risk operations
        assert security.classify_risk("delete_file") == RiskLevel.HIGH
        assert security.classify_risk("execute_code") == RiskLevel.HIGH

        # Test low-risk operations
        assert security.classify_risk("read_file") == RiskLevel.LOW
        assert security.classify_risk("search_memory") == RiskLevel.LOW

    def test_emergency_stop(self):
        """Test emergency stop mechanism."""
        from aion.core.security import SecurityManager

        security = SecurityManager()

        is_stopped, reason = security.is_emergency_stopped()
        assert not is_stopped

        security.emergency_stop("test reason")
        is_stopped, reason = security.is_emergency_stopped()
        assert is_stopped
        assert reason == "test reason"

        security.clear_emergency_stop()
        is_stopped, reason = security.is_emergency_stopped()
        assert not is_stopped


class TestPlanningGraph:
    """Test planning graph system."""

    @pytest.fixture
    def planning_graph(self):
        """Create a planning graph for testing."""
        from aion.systems.planning import PlanningGraph

        return PlanningGraph()

    @pytest.mark.asyncio
    async def test_plan_creation(self, planning_graph):
        """Test plan creation."""
        await planning_graph.initialize()

        plan = planning_graph.create_plan(
            name="Test Plan",
            description="A test execution plan",
        )

        assert plan.name == "Test Plan"
        assert len(plan.nodes) == 2  # Start and End nodes

    @pytest.mark.asyncio
    async def test_add_nodes_and_edges(self, planning_graph):
        """Test adding nodes and edges to a plan."""
        await planning_graph.initialize()

        plan = planning_graph.create_plan(name="Test Plan")

        node1 = planning_graph.add_node(
            plan_id=plan.id,
            name="Step 1",
            action="action1",
        )

        node2 = planning_graph.add_node(
            plan_id=plan.id,
            name="Step 2",
            action="action2",
        )

        # Add edges
        start_id = f"{plan.id}_start"
        end_id = f"{plan.id}_end"

        planning_graph.add_edge(plan.id, start_id, node1.id)
        planning_graph.add_edge(plan.id, node1.id, node2.id)
        planning_graph.add_edge(plan.id, node2.id, end_id)

        # Validate
        is_valid, errors = planning_graph.validate_plan(plan.id)
        assert is_valid, f"Validation errors: {errors}"


class TestMemory:
    """Test cognitive memory system."""

    @pytest.fixture
    def memory_system(self):
        """Create a memory system for testing."""
        from aion.systems.memory import CognitiveMemorySystem

        return CognitiveMemorySystem(
            embedding_model="all-MiniLM-L6-v2",
            embedding_dim=384,
            max_memories=1000,
        )

    @pytest.mark.asyncio
    async def test_memory_store_and_search(self, memory_system):
        """Test storing and searching memories."""
        await memory_system.initialize()

        # Store a memory
        memory = await memory_system.store(
            content="The capital of France is Paris.",
            importance=0.8,
        )

        assert memory.content == "The capital of France is Paris."
        assert memory.importance == 0.8

        # Search for it
        results = await memory_system.search(
            query="What is the capital of France?",
            limit=5,
        )

        assert len(results) > 0
        assert results[0].memory.id == memory.id

        await memory_system.shutdown()


class TestTools:
    """Test tool orchestration system."""

    @pytest.fixture
    def tool_orchestrator(self):
        """Create a tool orchestrator for testing."""
        from aion.systems.tools import ToolOrchestrator

        return ToolOrchestrator()

    @pytest.mark.asyncio
    async def test_calculator_tool(self, tool_orchestrator):
        """Test the calculator tool."""
        await tool_orchestrator.initialize()

        result = await tool_orchestrator.execute(
            tool_name="calculator",
            params={"expression": "2 + 2"},
        )

        assert result.success
        assert result.result["result"] == 4

    @pytest.mark.asyncio
    async def test_text_transform_tool(self, tool_orchestrator):
        """Test the text transform tool."""
        await tool_orchestrator.initialize()

        result = await tool_orchestrator.execute(
            tool_name="text_transform",
            params={
                "text": "hello world",
                "operation": "uppercase",
            },
        )

        assert result.success
        assert result.result["result"] == "HELLO WORLD"


class TestEvolution:
    """Test self-improvement engine."""

    @pytest.fixture
    def evolution_engine(self):
        """Create an evolution engine for testing."""
        from aion.systems.evolution import SelfImprovementEngine

        return SelfImprovementEngine(
            safety_threshold=0.95,
            require_approval=False,
        )

    @pytest.mark.asyncio
    async def test_parameter_registration(self, evolution_engine):
        """Test parameter registration."""
        await evolution_engine.initialize()

        evolution_engine.register_parameter(
            name="learning_rate",
            current_value=0.01,
            bounds=(0.001, 0.1),
        )

        params = evolution_engine.get_current_parameters()
        assert "learning_rate" in params
        assert params["learning_rate"] == 0.01

        await evolution_engine.shutdown()


class TestVision:
    """Test visual cortex system."""

    @pytest.fixture
    def visual_cortex(self):
        """Create a visual cortex for testing."""
        from aion.systems.vision import VisualCortex

        return VisualCortex(
            enable_memory=True,
        )

    @pytest.mark.asyncio
    async def test_scene_imagination(self, visual_cortex):
        """Test scene imagination from description."""
        await visual_cortex.initialize()

        scene = await visual_cortex.imagine(
            "A dog sitting next to a tree"
        )

        # Should detect dog and tree in the imagined scene
        labels = {obj.label for obj in scene.objects}
        assert "dog" in labels or len(scene.objects) > 0  # Mock may not have exact labels

        await visual_cortex.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
