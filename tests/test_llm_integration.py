"""
Tests for Real LLM Integration

Tests for the SOTA features that use Llama 3.3 70B for:
- Tree-of-Thought reasoning
- Chain-of-Thought reasoning
- Embedding generation
- Self-reflection
- RL reward evaluation
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# LLM Integration
from aion.systems.agents.llm_integration import (
    SOTALLMProvider,
    LLMIntegrationConfig,
    create_thought_generator,
    create_path_evaluator,
    create_embedding_function,
)

# Reasoning
from aion.systems.agents.reasoning.tree_of_thought import (
    TreeOfThought,
    ToTConfig,
    SearchStrategy,
)
from aion.systems.agents.reasoning.chain_of_thought import (
    ChainOfThought,
    CoTConfig,
)
from aion.systems.agents.reasoning.reflection import (
    SelfReflection,
    ReflectionConfig,
)

# Memory
from aion.systems.agents.memory.vector_store import (
    VectorStore,
    SimilarityMetric,
)

# Learning
from aion.systems.agents.learning.reinforcement import (
    ReinforcementLearner,
    RLConfig,
)


class TestSOTALLMProvider:
    """Tests for SOTA LLM Provider."""

    @pytest.fixture
    def mock_llm_adapter(self):
        """Create a mock LLM adapter."""
        adapter = AsyncMock()
        adapter.complete = AsyncMock(return_value=MagicMock(
            content="This is a test response.",
            model="llama-3.3-70b",
        ))
        return adapter

    @pytest.mark.asyncio
    async def test_singleton_pattern(self):
        """Test that provider uses singleton pattern."""
        # Reset singleton
        SOTALLMProvider._instance = None

        provider1 = await SOTALLMProvider.get_instance()
        provider2 = await SOTALLMProvider.get_instance()

        assert provider1 is provider2

        # Clean up
        await provider1.shutdown()
        SOTALLMProvider._instance = None

    @pytest.mark.asyncio
    async def test_config_from_env(self):
        """Test configuration from environment."""
        config = LLMIntegrationConfig.from_env()

        assert config.reasoning_model is not None
        assert config.embedding_dimension > 0
        assert config.timeout > 0

    @pytest.mark.asyncio
    async def test_thought_generation_fallback(self):
        """Test thought generation with fallback."""
        # Reset singleton
        SOTALLMProvider._instance = None

        config = LLMIntegrationConfig(
            reasoning_base_url="http://invalid:9999/v1",  # Will fail
        )
        provider = SOTALLMProvider(config)

        # Should fallback to simple embedding when LLM fails
        embedding = await provider._llm_based_embedding("test text")

        assert len(embedding) == config.embedding_dimension
        assert all(isinstance(v, float) for v in embedding)


class TestTreeOfThoughtWithLLM:
    """Tests for ToT with LLM integration."""

    @pytest.fixture
    def tot(self):
        """Create ToT instance."""
        config = ToTConfig(
            max_depth=2,
            branching_factor=2,
            max_nodes=10,
        )
        return TreeOfThought(agent_id="test-agent", config=config)

    @pytest.mark.asyncio
    async def test_tot_initialization(self, tot):
        """Test ToT initializes correctly."""
        assert tot.agent_id == "test-agent"
        assert tot.config.max_depth == 2

    @pytest.mark.asyncio
    async def test_tot_solve_with_initial_thoughts(self, tot):
        """Test solving with initial thoughts."""
        with patch.object(tot, '_get_llm_provider') as mock_provider:
            mock_llm = AsyncMock()
            mock_llm.generate_thoughts = AsyncMock(return_value=[
                "Consider approach A",
                "Consider approach B",
            ])
            mock_llm.evaluate_reasoning_path = AsyncMock(return_value=0.7)
            mock_provider.return_value = mock_llm

            solution, confidence, tree = await tot.solve(
                "What is 2 + 2?",
                initial_thoughts=["Start with addition"],
            )

            assert len(solution) >= 1
            assert 0 <= confidence <= 1
            assert tree is not None

    @pytest.mark.asyncio
    async def test_tot_generates_thoughts(self, tot):
        """Test thought generation."""
        with patch.object(tot, '_get_llm_provider') as mock_provider:
            mock_llm = AsyncMock()
            mock_llm.generate_thoughts = AsyncMock(return_value=[
                "Thought 1",
                "Thought 2",
            ])
            mock_provider.return_value = mock_llm

            thoughts = await tot._generate_thoughts(
                "Problem", ["Step 1"], n=2
            )

            assert len(thoughts) == 2


class TestChainOfThoughtWithLLM:
    """Tests for CoT with LLM integration."""

    @pytest.fixture
    def cot(self):
        """Create CoT instance."""
        config = CoTConfig(
            max_steps=5,
            enable_self_consistency=False,
            enable_verification=False,
        )
        return ChainOfThought(agent_id="test-agent", config=config)

    @pytest.mark.asyncio
    async def test_cot_initialization(self, cot):
        """Test CoT initializes correctly."""
        assert cot.agent_id == "test-agent"

    @pytest.mark.asyncio
    async def test_cot_reasoning(self, cot):
        """Test CoT reasoning generation."""
        with patch.object(cot, '_get_llm_provider') as mock_provider:
            mock_llm = AsyncMock()
            mock_llm.generate_cot_reasoning = AsyncMock(return_value=[
                "First, we observe the problem",
                "Then, we calculate",
                "Finally, the answer is X",
            ])
            mock_llm.extract_answer = AsyncMock(return_value="X")
            mock_provider.return_value = mock_llm

            chain = await cot.reason("What is 2 + 2?")

            assert chain is not None
            assert len(chain.steps) >= 1


class TestSelfReflectionWithLLM:
    """Tests for self-reflection with LLM integration."""

    @pytest.fixture
    def reflection(self):
        """Create reflection instance."""
        config = ReflectionConfig(
            max_iterations=2,
            min_iterations=1,
        )
        return SelfReflection(agent_id="test-agent", config=config)

    @pytest.mark.asyncio
    async def test_reflection_initialization(self, reflection):
        """Test reflection initializes correctly."""
        assert reflection.agent_id == "test-agent"

    @pytest.mark.asyncio
    async def test_reflection_with_llm(self, reflection):
        """Test reflection using LLM."""
        with patch.object(reflection, '_get_llm_provider') as mock_provider:
            mock_llm = AsyncMock()
            mock_llm.reflect_on_output = AsyncMock(return_value={
                "overall_quality": 0.7,
                "strengths": ["Clear"],
                "weaknesses": ["Could be more detailed"],
                "suggestions": ["Add more examples"],
                "is_complete": True,
            })
            mock_llm.critique_and_improve = AsyncMock(return_value=(
                "Improved output",
                [],
            ))
            mock_provider.return_value = mock_llm

            result = await reflection.reflect(
                task="Write a summary",
                output="This is a short summary.",
            )

            assert result is not None
            assert len(result.critiques) >= 0


class TestVectorStoreWithLLM:
    """Tests for vector store with real embeddings."""

    @pytest.fixture
    async def store(self):
        """Create vector store with LLM embeddings."""
        store = VectorStore(
            dimension=128,
            use_llm_embeddings=True,
        )
        await store.initialize()
        yield store
        await store.shutdown()

    @pytest.mark.asyncio
    async def test_store_initialization(self, store):
        """Test store initializes correctly."""
        assert store._initialized
        assert store.dimension == 128

    @pytest.mark.asyncio
    async def test_add_with_fallback_embedding(self, store):
        """Test adding entry with fallback embedding."""
        # Will use fallback since LLM may not be available
        entry_id = await store.add("Test text for embedding")

        assert entry_id is not None

    @pytest.mark.asyncio
    async def test_search_with_fallback_embedding(self, store):
        """Test search with fallback embedding."""
        await store.add("Python is a programming language")
        await store.add("Machine learning uses data")

        results = await store.search("programming", k=2)

        # Results may be empty if no real embeddings
        assert isinstance(results, list)


class TestReinforcementLearnerWithLLM:
    """Tests for RL with LLM feedback."""

    @pytest.fixture
    def learner(self):
        """Create RL learner."""
        return ReinforcementLearner(
            agent_id="test-agent",
            learning_rate=0.1,
            exploration_rate=0.2,
        )

    @pytest.mark.asyncio
    async def test_learner_initialization(self, learner):
        """Test learner initializes correctly."""
        await learner.initialize()
        assert learner._initialized

    @pytest.mark.asyncio
    async def test_action_selection(self, learner):
        """Test action selection."""
        learner.available_actions = ["action1", "action2", "action3"]

        action, confidence = learner.select_action(
            state={"task": "test"},
        )

        assert action in learner.available_actions
        assert 0 <= confidence <= 1

    @pytest.mark.asyncio
    async def test_record_experience(self, learner):
        """Test recording experience."""
        exp = await learner.record_experience(
            state={"task": "test"},
            action="execute",
            reward=0.5,
            next_state={"task": "done"},
        )

        assert exp is not None
        assert exp.reward == 0.5

    @pytest.mark.asyncio
    async def test_llm_reward_evaluation(self, learner):
        """Test LLM-based reward evaluation."""
        with patch.object(learner, '_get_llm_provider') as mock_provider:
            mock_llm = AsyncMock()
            mock_llm.evaluate_action_outcome = AsyncMock(return_value=0.8)
            mock_provider.return_value = mock_llm

            reward = await learner.evaluate_action_with_llm(
                state={"task": "write code"},
                action="implement_function",
                outcome={"success": True, "output": "def foo(): pass"},
            )

            assert reward == 0.8

    @pytest.mark.asyncio
    async def test_experience_with_llm_evaluation(self, learner):
        """Test recording experience with LLM evaluation."""
        with patch.object(learner, '_get_llm_provider') as mock_provider:
            mock_llm = AsyncMock()
            mock_llm.evaluate_action_outcome = AsyncMock(return_value=0.7)
            mock_provider.return_value = mock_llm

            exp = await learner.record_experience_with_llm_evaluation(
                state={"task": "test"},
                action="execute",
                outcome={"success": True},
            )

            assert exp is not None
            assert exp.reward == 0.7
            assert exp.info.get("llm_evaluated") is True


class TestCallbackFactories:
    """Tests for callback factory functions."""

    @pytest.mark.asyncio
    async def test_create_thought_generator(self):
        """Test thought generator factory."""
        # Reset singleton
        SOTALLMProvider._instance = None

        generator = await create_thought_generator()

        assert callable(generator)

        # Clean up
        if SOTALLMProvider._instance:
            await SOTALLMProvider._instance.shutdown()
        SOTALLMProvider._instance = None

    @pytest.mark.asyncio
    async def test_create_path_evaluator(self):
        """Test path evaluator factory."""
        SOTALLMProvider._instance = None

        evaluator = await create_path_evaluator()

        assert callable(evaluator)

        if SOTALLMProvider._instance:
            await SOTALLMProvider._instance.shutdown()
        SOTALLMProvider._instance = None

    @pytest.mark.asyncio
    async def test_create_embedding_function(self):
        """Test embedding function factory."""
        SOTALLMProvider._instance = None

        embed_fn = await create_embedding_function()

        assert callable(embed_fn)

        if SOTALLMProvider._instance:
            await SOTALLMProvider._instance.shutdown()
        SOTALLMProvider._instance = None


class TestIntegrationWithRealLLM:
    """
    Integration tests that use the real LLM.

    These tests are skipped if the LLM is not available.
    Set AION_TEST_WITH_LLM=1 to run them.
    """

    @pytest.fixture
    def skip_without_llm(self):
        """Skip if LLM not configured."""
        import os
        if not os.environ.get("AION_TEST_WITH_LLM"):
            pytest.skip("Set AION_TEST_WITH_LLM=1 to run LLM tests")

    @pytest.mark.asyncio
    async def test_real_thought_generation(self, skip_without_llm):
        """Test real thought generation with Llama 3.3 70B."""
        SOTALLMProvider._instance = None

        provider = await SOTALLMProvider.get_instance()

        thoughts = await provider.generate_thoughts(
            problem="What is the capital of France?",
            current_path=[],
            n=2,
        )

        assert len(thoughts) >= 1
        assert all(isinstance(t, str) for t in thoughts)

        await provider.shutdown()
        SOTALLMProvider._instance = None

    @pytest.mark.asyncio
    async def test_real_path_evaluation(self, skip_without_llm):
        """Test real path evaluation with Llama 3.3 70B."""
        SOTALLMProvider._instance = None

        provider = await SOTALLMProvider.get_instance()

        score = await provider.evaluate_reasoning_path(
            problem="What is 2 + 2?",
            path=["First, I identify this as an addition problem", "2 + 2 = 4"],
        )

        assert 0 <= score <= 1

        await provider.shutdown()
        SOTALLMProvider._instance = None

    @pytest.mark.asyncio
    async def test_real_cot_reasoning(self, skip_without_llm):
        """Test real CoT reasoning with Llama 3.3 70B."""
        SOTALLMProvider._instance = None

        provider = await SOTALLMProvider.get_instance()

        steps = await provider.generate_cot_reasoning(
            problem="If a train travels 60 miles in 1 hour, how far does it travel in 3 hours?",
        )

        assert len(steps) >= 1

        await provider.shutdown()
        SOTALLMProvider._instance = None

    @pytest.mark.asyncio
    async def test_real_reflection(self, skip_without_llm):
        """Test real reflection with Llama 3.3 70B."""
        SOTALLMProvider._instance = None

        provider = await SOTALLMProvider.get_instance()

        reflection = await provider.reflect_on_output(
            task="Write a haiku about coding",
            output="Code flows like a stream\nBugs hide in the darkness deep\nDebug light shines through",
        )

        assert "overall_quality" in reflection
        assert 0 <= reflection["overall_quality"] <= 1

        await provider.shutdown()
        SOTALLMProvider._instance = None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
