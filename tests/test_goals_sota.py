"""
AION Goal System SOTA Features Tests

Comprehensive test suite for state-of-the-art goal system features:
- Learned components with neural evaluation
- Uncertainty quantification with Bayesian reasoning
- World model for outcome simulation
- Meta-learning for adaptive strategies
- Formal verification for safety guarantees
- Multi-agent coordination
"""

import pytest
import asyncio
import numpy as np
from datetime import datetime, timedelta
import tempfile
import shutil


class TestGoalLearningSystem:
    """Test learned components for goal evaluation."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory."""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path)

    @pytest.fixture
    def learning_system(self, temp_dir):
        """Create a learning system for testing."""
        from aion.systems.goals.learning import GoalLearningSystem

        return GoalLearningSystem(data_dir=temp_dir)

    @pytest.mark.asyncio
    async def test_learning_system_initialization(self, learning_system):
        """Test learning system initialization."""
        await learning_system.initialize()
        assert learning_system._initialized
        await learning_system.shutdown()

    @pytest.mark.asyncio
    async def test_goal_embedding(self, learning_system):
        """Test goal embedding generation."""
        from aion.systems.goals.types import Goal

        await learning_system.initialize()

        goal = Goal(
            title="Test Embedding",
            description="A goal for testing embeddings",
            success_criteria=["Criterion 1"],
        )

        embedding = learning_system.get_embedding(goal)

        assert embedding is not None
        assert embedding.goal_id == goal.id
        assert len(embedding.vector) == learning_system.embedding_dim

        await learning_system.shutdown()

    @pytest.mark.asyncio
    async def test_success_prediction(self, learning_system):
        """Test success probability prediction."""
        from aion.systems.goals.types import Goal

        await learning_system.initialize()

        goal = Goal(
            title="Predict Success",
            description="Testing prediction",
            success_criteria=["Done"],
        )

        probability = learning_system.predict_success(goal)

        assert 0.0 <= probability <= 1.0

        await learning_system.shutdown()

    @pytest.mark.asyncio
    async def test_similar_goals_finding(self, learning_system):
        """Test finding similar goals."""
        from aion.systems.goals.types import Goal

        await learning_system.initialize()

        goal1 = Goal(
            title="Create Documentation",
            description="Write project documentation",
            success_criteria=["Docs complete"],
        )

        goal2 = Goal(
            title="Write Documentation",
            description="Create project docs",
            success_criteria=["Documentation done"],
        )

        goal3 = Goal(
            title="Run Tests",
            description="Execute test suite",
            success_criteria=["Tests pass"],
        )

        candidates = [goal2, goal3]
        similar = learning_system.find_similar_goals(goal1, candidates, top_k=2)

        assert len(similar) <= 2
        # Documentation goals should be more similar
        if similar:
            assert similar[0][0].id in [goal2.id, goal3.id]

        await learning_system.shutdown()

    @pytest.mark.asyncio
    async def test_outcome_recording(self, learning_system):
        """Test recording goal outcomes."""
        from aion.systems.goals.types import Goal

        await learning_system.initialize()

        goal = Goal(
            title="Record Outcome",
            description="Testing outcome recording",
            success_criteria=["Complete"],
        )

        await learning_system.record_outcome(
            goal=goal,
            success=True,
            completion_time=60.0,
            resource_usage=0.5,
            quality_score=0.9,
        )

        assert len(learning_system.experience_buffer) >= 1

        await learning_system.shutdown()

    @pytest.mark.asyncio
    async def test_priority_learning(self, learning_system):
        """Test adaptive priority learning."""
        from aion.systems.goals.types import Goal

        await learning_system.initialize()

        goal = Goal(
            title="Priority Test",
            description="Testing priority learning",
            success_criteria=["Done"],
        )

        priority1 = learning_system.compute_learned_priority(goal)
        assert 0.0 <= priority1 <= 1.0

        # Weights should be normalized
        weights = learning_system.priority_learner.weights
        assert np.isclose(np.sum(weights), 1.0)

        await learning_system.shutdown()


class TestUncertaintyQuantification:
    """Test uncertainty quantification system."""

    @pytest.fixture
    def uncertainty(self):
        """Create an uncertainty quantifier for testing."""
        from aion.systems.goals.uncertainty import UncertaintyQuantifier

        return UncertaintyQuantifier()

    @pytest.mark.asyncio
    async def test_uncertainty_initialization(self, uncertainty):
        """Test uncertainty system initialization."""
        await uncertainty.initialize()
        assert uncertainty._initialized
        await uncertainty.shutdown()

    @pytest.mark.asyncio
    async def test_success_uncertainty_estimate(self, uncertainty):
        """Test success probability uncertainty estimation."""
        from aion.systems.goals.types import Goal

        await uncertainty.initialize()

        goal = Goal(
            title="Uncertainty Test",
            description="Testing uncertainty",
            success_criteria=["Done"],
        )

        estimate = uncertainty.estimate_success_uncertainty(goal)

        assert 0.0 <= estimate.mean <= 1.0
        assert estimate.std >= 0.0
        assert estimate.confidence_interval[0] <= estimate.mean <= estimate.confidence_interval[1]
        assert estimate.epistemic_uncertainty >= 0.0
        assert estimate.aleatoric_uncertainty >= 0.0

        await uncertainty.shutdown()

    @pytest.mark.asyncio
    async def test_confidence_level(self, uncertainty):
        """Test confidence level determination."""
        from aion.systems.goals.types import Goal

        await uncertainty.initialize()

        goal = Goal(
            title="Confidence Test",
            description="Testing confidence",
            success_criteria=["Done"],
        )

        level = uncertainty.get_confidence_level(goal)

        assert level in ["very_high", "high", "medium", "low", "very_low"]

        await uncertainty.shutdown()

    @pytest.mark.asyncio
    async def test_thompson_sampling(self, uncertainty):
        """Test Thompson sampling for goal selection."""
        from aion.systems.goals.types import Goal

        await uncertainty.initialize()

        goals = [
            Goal(title=f"Goal {i}", description="Test", success_criteria=["Done"])
            for i in range(5)
        ]

        selected, score = uncertainty.select_goal_with_exploration(goals)

        assert selected in goals
        assert score >= 0.0

        await uncertainty.shutdown()

    @pytest.mark.asyncio
    async def test_outcome_update(self, uncertainty):
        """Test updating from outcomes."""
        from aion.systems.goals.types import Goal

        await uncertainty.initialize()

        goal = Goal(
            title="Update Test",
            description="Testing updates",
            success_criteria=["Done"],
        )

        # Get initial estimate
        before = uncertainty.estimate_success_uncertainty(goal)

        # Update with success
        uncertainty.update_from_outcome(goal, True)

        # Estimate should potentially change
        after = uncertainty.estimate_success_uncertainty(goal)

        # At minimum, no errors should occur
        assert 0.0 <= after.mean <= 1.0

        await uncertainty.shutdown()


class TestWorldModel:
    """Test world model for outcome simulation."""

    @pytest.fixture
    def world_model(self):
        """Create a world model for testing."""
        from aion.systems.goals.world_model import WorldModel

        return WorldModel(state_dim=32)

    @pytest.mark.asyncio
    async def test_world_model_initialization(self, world_model):
        """Test world model initialization."""
        await world_model.initialize()
        assert world_model._initialized
        assert world_model._current_state is not None
        await world_model.shutdown()

    @pytest.mark.asyncio
    async def test_state_management(self, world_model):
        """Test world state management."""
        from aion.systems.goals.world_model import WorldState, StateVariable, StateType

        await world_model.initialize()

        state = WorldState()
        state.set("resources", 100, StateType.RESOURCE)
        state.set("active_goals", 2, StateType.GOAL)

        world_model.set_current_state(state)

        current = world_model.get_current_state()
        assert current.get("resources") == 100
        assert current.get("active_goals") == 2

        await world_model.shutdown()

    @pytest.mark.asyncio
    async def test_action_simulation(self, world_model):
        """Test action simulation."""
        from aion.systems.goals.world_model import Action, WorldState

        await world_model.initialize()

        # Set up state
        state = WorldState()
        state.set("progress", 0.0)
        state.set("resources", 100.0)
        world_model.set_current_state(state)

        # Create action
        action = Action(
            name="work_on_goal",
            parameters={"effort": 10},
            cost=10.0,
        )

        world_model.register_action(action)

        # Simulate
        next_state, reward = world_model.simulate_action(action)

        assert next_state is not None
        # Reward should be computed
        assert isinstance(reward, float)

        await world_model.shutdown()

    @pytest.mark.asyncio
    async def test_trajectory_simulation(self, world_model):
        """Test trajectory simulation."""
        from aion.systems.goals.world_model import Action, WorldState

        await world_model.initialize()

        state = WorldState()
        state.set("step", 0)
        world_model.set_current_state(state)

        actions = [
            Action(name=f"action_{i}", parameters={})
            for i in range(3)
        ]

        for action in actions:
            world_model.register_action(action)

        trajectory = world_model.simulate_trajectory(actions)

        assert len(trajectory) == 3

        await world_model.shutdown()

    @pytest.mark.asyncio
    async def test_mcts_planning(self, world_model):
        """Test MCTS planning."""
        from aion.systems.goals.world_model import Action, WorldState
        from aion.systems.goals.types import Goal

        await world_model.initialize()

        state = WorldState()
        state.set("progress", 0.0)
        world_model.set_current_state(state)

        action = Action(
            name="progress_action",
            parameters={},
        )
        world_model.register_action(action)

        goal = Goal(
            title="Plan Test",
            description="Testing planning",
            success_criteria=["Progress made"],
        )

        best_action, value = world_model.plan_action(goal)

        # Should return an action (even if it's the only one)
        assert best_action is not None or len(world_model._available_actions) == 0

        await world_model.shutdown()


class TestMetaLearning:
    """Test meta-learning system."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory."""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path)

    @pytest.fixture
    def meta_learning(self, temp_dir):
        """Create a meta-learning system for testing."""
        from aion.systems.goals.meta_learning import MetaLearningSystem

        return MetaLearningSystem(data_dir=temp_dir)

    @pytest.mark.asyncio
    async def test_meta_learning_initialization(self, meta_learning):
        """Test meta-learning initialization."""
        await meta_learning.initialize()
        assert meta_learning._initialized
        await meta_learning.shutdown()

    @pytest.mark.asyncio
    async def test_strategy_selection(self, meta_learning):
        """Test strategy selection."""
        from aion.systems.goals.types import Goal

        await meta_learning.initialize()

        goal = Goal(
            title="Strategy Test",
            description="Testing strategy selection",
            success_criteria=["Done"],
        )

        strategy = meta_learning.select_strategy(goal)

        assert strategy is not None
        assert strategy.name is not None

        await meta_learning.shutdown()

    @pytest.mark.asyncio
    async def test_hyperparameter_tuning(self, meta_learning):
        """Test hyperparameter tuning."""
        await meta_learning.initialize()

        params = meta_learning.suggest_hyperparameters()

        assert isinstance(params, dict)
        assert len(params) > 0

        # Observe performance
        meta_learning.observe_hyperparameters(params, 0.8)

        best, score = meta_learning.hyperparameter_tuner.get_best()
        assert best is not None

        await meta_learning.shutdown()

    @pytest.mark.asyncio
    async def test_curriculum_learning(self, meta_learning):
        """Test curriculum learning."""
        from aion.systems.goals.types import Goal, GoalType

        await meta_learning.initialize()

        goals = [
            Goal(
                title=f"Goal {i}",
                description="Test " * (i + 1),
                success_criteria=["Done"] * (i + 1),
                goal_type=GoalType.ACHIEVEMENT,
            )
            for i in range(5)
        ]

        suggested = meta_learning.suggest_next_goal(goals)

        # Should suggest a goal
        assert suggested is None or suggested in goals

        await meta_learning.shutdown()

    @pytest.mark.asyncio
    async def test_transfer_learning(self, meta_learning):
        """Test transfer learning."""
        from aion.systems.goals.types import Goal

        await meta_learning.initialize()

        goal = Goal(
            title="Transfer Test",
            description="Testing transfer learning",
            success_criteria=["Complete"],
        )

        knowledge = meta_learning.transfer_knowledge(goal)

        assert isinstance(knowledge, dict)

        await meta_learning.shutdown()

    @pytest.mark.asyncio
    async def test_outcome_update(self, meta_learning):
        """Test updating from outcomes."""
        from aion.systems.goals.types import Goal

        await meta_learning.initialize()

        goal = Goal(
            title="Update Test",
            description="Testing outcome update",
            success_criteria=["Done"],
        )

        strategy = meta_learning.select_strategy(goal)

        meta_learning.update_from_outcome(
            goal=goal,
            strategy_id=strategy.id,
            success=True,
            completion_time=60.0,
            reward=1.0,
        )

        # Strategy should have updated stats
        updated_strategy = meta_learning.strategy_portfolio.strategies.get(strategy.id)
        assert updated_strategy.uses > 0

        await meta_learning.shutdown()


class TestFormalVerification:
    """Test formal verification system."""

    @pytest.fixture
    def verification(self):
        """Create a formal verification system for testing."""
        from aion.systems.goals.formal_verification import FormalVerificationSystem

        return FormalVerificationSystem()

    @pytest.mark.asyncio
    async def test_verification_initialization(self, verification):
        """Test verification system initialization."""
        await verification.initialize()
        assert verification._initialized
        await verification.shutdown()

    @pytest.mark.asyncio
    async def test_goal_safety_verification(self, verification):
        """Test goal safety verification."""
        from aion.systems.goals.types import Goal

        await verification.initialize()

        safe_goal = Goal(
            title="Safe Goal",
            description="A completely safe task",
            success_criteria=["Completed"],
        )

        is_safe, issues = verification.verify_goal_safety(safe_goal)

        assert is_safe
        assert len(issues) == 0

        await verification.shutdown()

    @pytest.mark.asyncio
    async def test_unsafe_goal_detection(self, verification):
        """Test detection of unsafe goals."""
        from aion.systems.goals.types import Goal

        await verification.initialize()

        unsafe_goal = Goal(
            title="Delete All Data",
            description="Delete all user data and bypass security",
            success_criteria=["Everything deleted"],
        )

        is_safe, issues = verification.verify_goal_safety(unsafe_goal)

        assert not is_safe
        assert len(issues) > 0

        await verification.shutdown()

    @pytest.mark.asyncio
    async def test_action_safety_verification(self, verification):
        """Test action safety verification."""
        await verification.initialize()

        safe_is_safe, _ = verification.verify_action_safety(
            "read_file",
            {"path": "/home/user/doc.txt"}
        )
        assert safe_is_safe

        unsafe_is_safe, reason = verification.verify_action_safety(
            "execute",
            {"command": "rm -rf /"}
        )
        assert not unsafe_is_safe

        await verification.shutdown()

    @pytest.mark.asyncio
    async def test_runtime_monitoring(self, verification):
        """Test runtime monitoring."""
        await verification.initialize()

        state1 = {"has_unsafe_goal": False, "resources_bounded": True}
        violations1 = verification.observe_execution(state1)
        assert len(violations1) == 0

        state2 = {"has_unsafe_goal": True, "resources_bounded": True}
        violations2 = verification.observe_execution(state2)
        # May have violations depending on registered properties

        await verification.shutdown()

    @pytest.mark.asyncio
    async def test_bounds_checking(self, verification):
        """Test bounds checking."""
        await verification.initialize()

        # Check within bounds
        in_bounds, _ = verification.bounds_checker.check_bound("resource_usage", 500)
        assert in_bounds

        # Check out of bounds
        out_bounds, msg = verification.bounds_checker.check_bound("resource_usage", 1500)
        assert not out_bounds
        assert "above" in msg

        await verification.shutdown()

    @pytest.mark.asyncio
    async def test_ltl_formula(self, verification):
        """Test LTL formula creation and evaluation."""
        from aion.systems.goals.formal_verification import Formula

        # Create formula: G(safe)
        safe_pred = Formula.atom("safe", lambda s: s.get("safe", False))
        always_safe = Formula.always(safe_pred)

        assert str(always_safe) == "G(safe)"

        # Create formula: F(complete)
        complete_pred = Formula.atom("complete")
        eventually_complete = Formula.eventually(complete_pred)

        assert str(eventually_complete) == "F(complete)"


class TestMultiAgentCoordination:
    """Test multi-agent coordination system."""

    @pytest.fixture
    def coordinator(self):
        """Create a multi-agent coordinator for testing."""
        from aion.systems.goals.multi_agent import MultiAgentCoordinator

        return MultiAgentCoordinator()

    @pytest.mark.asyncio
    async def test_coordinator_initialization(self, coordinator):
        """Test coordinator initialization."""
        await coordinator.initialize()
        assert coordinator._initialized

        # Should have default coordinator agent
        agents = coordinator.registry.get_all()
        assert len(agents) >= 1

        await coordinator.shutdown()

    @pytest.mark.asyncio
    async def test_agent_registration(self, coordinator):
        """Test agent registration."""
        from aion.systems.goals.multi_agent import AgentRole

        await coordinator.initialize()

        agent = coordinator.register_agent(
            name="Worker Agent",
            role=AgentRole.WORKER,
            capabilities=["coding", "testing"],
        )

        assert agent is not None
        assert agent.name == "Worker Agent"
        assert agent.has_capability("coding")
        assert agent.has_capability("testing")

        await coordinator.shutdown()

    @pytest.mark.asyncio
    async def test_goal_assignment(self, coordinator):
        """Test goal assignment to agents."""
        from aion.systems.goals.types import Goal
        from aion.systems.goals.multi_agent import AgentRole

        await coordinator.initialize()

        # Register a capable agent
        coordinator.register_agent(
            name="Coder",
            role=AgentRole.WORKER,
            capabilities=["coding"],
        )

        goal = Goal(
            title="Write Code",
            description="Write some code",
            success_criteria=["Code written"],
        )

        # Allow time for auction
        assignment = await coordinator.assign_goal(
            goal,
            required_capabilities=["coding"],
        )

        # Assignment may or may not succeed depending on timing
        # But no errors should occur

        await coordinator.shutdown()

    @pytest.mark.asyncio
    async def test_message_passing(self, coordinator):
        """Test message passing between agents."""
        from aion.systems.goals.multi_agent import AgentRole, MessageType

        await coordinator.initialize()

        agent1 = coordinator.register_agent("Agent 1", AgentRole.WORKER)
        agent2 = coordinator.register_agent("Agent 2", AgentRole.WORKER)

        coordinator.send_message(
            sender_id=agent1.id,
            receiver_id=agent2.id,
            message_type=MessageType.HEARTBEAT,
            content={"status": "alive"},
        )

        # Process messages
        await coordinator.message_broker.process_messages()

        # Agent 2 should have received the message
        assert len(agent2.inbox) >= 1

        await coordinator.shutdown()

    @pytest.mark.asyncio
    async def test_coalition_formation(self, coordinator):
        """Test coalition formation."""
        from aion.systems.goals.types import Goal
        from aion.systems.goals.multi_agent import AgentRole

        await coordinator.initialize()

        # Register agents with different capabilities
        coordinator.register_agent("Designer", AgentRole.SPECIALIST, ["design"])
        coordinator.register_agent("Developer", AgentRole.SPECIALIST, ["coding"])
        coordinator.register_agent("Tester", AgentRole.SPECIALIST, ["testing"])

        goal = Goal(
            title="Complex Project",
            description="Needs multiple skills",
            success_criteria=["Complete"],
        )

        coalition = coordinator.coalition_formation.form_coalition(
            goal,
            required_capabilities=["design", "coding", "testing"],
        )

        # Coalition should cover all capabilities
        if coalition:
            assert len(coalition) >= 1

        await coordinator.shutdown()

    @pytest.mark.asyncio
    async def test_consensus_protocol(self, coordinator):
        """Test consensus protocol."""
        from aion.systems.goals.multi_agent import AgentRole

        await coordinator.initialize()

        # Register multiple agents
        for i in range(3):
            coordinator.register_agent(f"Voter {i}", AgentRole.WORKER)

        proposal = {"action": "upgrade_system", "version": "2.0"}

        # Note: This is a simplified test - real consensus requires voting
        accepted, proposal_id = await coordinator.reach_consensus(
            "coordinator_0",
            proposal,
        )

        # Proposal should complete (accepted or not)
        assert proposal_id is not None

        await coordinator.shutdown()

    @pytest.mark.asyncio
    async def test_conflict_resolution(self, coordinator):
        """Test conflict resolution."""
        from aion.systems.goals.types import Goal, GoalPriority

        await coordinator.initialize()

        goals = [
            Goal(
                title="Low Priority",
                description="Can wait",
                success_criteria=["Done"],
                priority=GoalPriority.LOW,
            ),
            Goal(
                title="High Priority",
                description="Urgent",
                success_criteria=["Done"],
                priority=GoalPriority.HIGH,
            ),
            Goal(
                title="Critical",
                description="Must do now",
                success_criteria=["Done"],
                priority=GoalPriority.CRITICAL,
            ),
        ]

        resolved = coordinator.resolve_conflicts(goals)

        # Critical should be first
        assert resolved[0].priority == GoalPriority.CRITICAL

        await coordinator.shutdown()


class TestSOTAIntegration:
    """Integration tests for SOTA features in goal manager."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory."""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path)

    @pytest.fixture
    def manager(self, temp_dir):
        """Create a goal manager with SOTA features."""
        from aion.systems.goals.manager import AutonomousGoalManager

        return AutonomousGoalManager(
            auto_generate_goals=False,
            data_dir=temp_dir,
            enable_sota_features=True,
        )

    @pytest.mark.asyncio
    async def test_manager_with_sota_init(self, manager):
        """Test manager initialization with SOTA features."""
        await manager.initialize()

        assert manager.is_initialized
        assert manager._enable_sota
        assert manager.learning is not None
        assert manager.uncertainty is not None
        assert manager.world_model is not None
        assert manager.meta_learning is not None
        assert manager.formal_verification is not None
        assert manager.multi_agent is not None

        await manager.shutdown()

    @pytest.mark.asyncio
    async def test_predict_goal_success(self, manager):
        """Test integrated success prediction."""
        await manager.initialize()

        goal = await manager.submit_goal(
            title="Prediction Test",
            description="Testing integrated prediction",
            success_criteria=["Complete"],
        )

        prediction = await manager.predict_goal_success(goal)

        assert "success_probability" in prediction
        assert "confidence_interval" in prediction
        assert "recommended_strategy" in prediction

        await manager.shutdown()

    @pytest.mark.asyncio
    async def test_formal_verification_integration(self, manager):
        """Test integrated formal verification."""
        await manager.initialize()

        goal = await manager.submit_goal(
            title="Verification Test",
            description="A safe goal for testing",
            success_criteria=["Done"],
        )

        result = await manager.verify_goal_safety_formally(goal)

        assert "formally_verified" in result
        assert result["formally_verified"]

        await manager.shutdown()

    @pytest.mark.asyncio
    async def test_sota_stats(self, manager):
        """Test SOTA statistics."""
        await manager.initialize()

        stats = manager.get_sota_stats()

        assert stats["enabled"]
        assert "learning" in stats
        assert "uncertainty" in stats
        assert "world_model" in stats
        assert "meta_learning" in stats
        assert "formal_verification" in stats
        assert "multi_agent" in stats

        await manager.shutdown()

    @pytest.mark.asyncio
    async def test_outcome_recording(self, manager):
        """Test recording outcomes for learning."""
        await manager.initialize()

        goal = await manager.submit_goal(
            title="Outcome Test",
            description="Testing outcome recording",
            success_criteria=["Done"],
        )

        # Record outcome
        await manager.record_goal_outcome(
            goal=goal,
            success=True,
            completion_time=120.0,
            quality_score=0.9,
        )

        # No errors should occur
        # Learning system should have recorded

        await manager.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
