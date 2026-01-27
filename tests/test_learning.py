"""
Comprehensive tests for the AION Reinforcement Learning Loop.

Tests cover:
- Core types and state representation
- Reward collection (explicit, implicit, outcome)
- Experience buffer with priority sampling
- Policy optimizer and tool/planning/agent policies
- Bandit algorithms (Thompson Sampling, UCB1, LinUCB)
- A/B testing framework with statistical analysis
- Main RL loop coordinator lifecycle
- N-step transition building
- Reward shaping (potential-based + curiosity)
- Integration modules
- Persistence
"""

# Isolate aion.learning imports from the full kernel dependency chain
import tests.conftest_learning  # noqa: F401  (patches sys.modules)

import asyncio
import math
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------
from aion.learning.types import (
    Action,
    ActionType,
    ArmStatistics,
    Experience,
    Experiment,
    ExperimentStatus,
    ExperimentVariant,
    PolicyConfig,
    RewardSignal,
    RewardSource,
    RewardType,
    StateRepresentation,
)


class TestRewardSignal:
    def test_default_values(self):
        sig = RewardSignal()
        assert sig.value == 0.0
        assert sig.confidence == 1.0
        assert sig.source == RewardSource.IMPLICIT_COMPLETION

    def test_discounted_value_no_delay(self):
        sig = RewardSignal(value=1.0, confidence=1.0, delay_seconds=0.0)
        assert sig.discounted_value(gamma=0.99) == 1.0

    def test_discounted_value_with_delay(self):
        sig = RewardSignal(value=1.0, confidence=1.0, delay_seconds=60.0)
        dv = sig.discounted_value(gamma=0.99)
        assert dv == pytest.approx(0.99, abs=0.01)

    def test_discounted_value_with_low_confidence(self):
        sig = RewardSignal(value=1.0, confidence=0.5, delay_seconds=0.0)
        assert sig.discounted_value() == 0.5


class TestStateRepresentation:
    def test_to_vector_default(self):
        state = StateRepresentation()
        vec = state.to_vector()
        assert isinstance(vec, np.ndarray)
        assert vec.dtype == np.float32
        assert len(vec) >= 7

    def test_to_vector_with_rewards(self):
        state = StateRepresentation(recent_rewards=[0.5, 1.0, -0.5])
        vec = state.to_vector()
        # Mean of [0.5, 1.0, -0.5] ~ 0.333
        assert vec[4] == pytest.approx(np.mean([0.5, 1.0, -0.5]), abs=0.01)

    def test_to_vector_with_context_features(self):
        state = StateRepresentation(context_features={"a": 0.1, "b": 0.2})
        vec = state.to_vector()
        assert len(vec) == 9  # 7 base + 2 context


class TestAction:
    def test_default(self):
        action = Action()
        assert action.action_type == ActionType.TOOL_SELECTION
        assert action.choice == ""
        assert action.exploration is False


class TestExperience:
    def test_compute_cumulative_reward(self):
        signals = [
            RewardSignal(value=0.5, confidence=1.0, delay_seconds=0),
            RewardSignal(value=0.3, confidence=0.8, delay_seconds=0),
        ]
        exp = Experience(reward=1.0, rewards=signals)
        total = exp.compute_cumulative_reward(gamma=0.99)
        expected = 1.0 + 0.5 + 0.3 * 0.8
        assert total == pytest.approx(expected, abs=0.01)


class TestArmStatistics:
    def test_update(self):
        arm = ArmStatistics(arm_id="test")
        arm.update(1.0)
        assert arm.pulls == 1
        assert arm.avg_reward == 1.0
        arm.update(0.0)
        assert arm.pulls == 2
        assert arm.avg_reward == 0.5

    def test_variance(self):
        arm = ArmStatistics(arm_id="test")
        for r in [1.0, 0.0, 1.0, 0.0]:
            arm.update(r)
        assert arm.variance > 0


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
from aion.learning.config import LearningConfig, BufferConfig, RewardConfig


class TestLearningConfig:
    def test_defaults(self):
        cfg = LearningConfig()
        assert cfg.enabled is True
        assert cfg.training_enabled is True
        assert cfg.buffer.max_size == 100_000

    def test_to_dict(self):
        cfg = LearningConfig()
        d = cfg.to_dict()
        assert "reward" in d
        assert "buffer" in d

    def test_from_dict(self):
        cfg = LearningConfig.from_dict({"enabled": False, "buffer": {"max_size": 5000}})
        assert cfg.enabled is False
        assert cfg.buffer.max_size == 5000


# ---------------------------------------------------------------------------
# Reward collection
# ---------------------------------------------------------------------------
from aion.learning.rewards.collector import RewardCollector
from aion.learning.rewards.explicit import ExplicitFeedbackProcessor
from aion.learning.rewards.implicit import ImplicitSignalExtractor
from aion.learning.rewards.shaping import (
    PotentialBasedShaping,
    CuriosityRewardShaper,
    CompositeRewardShaper,
)


class TestExplicitFeedbackProcessor:
    @pytest.mark.asyncio
    async def test_thumbs_up(self):
        proc = ExplicitFeedbackProcessor()
        sig = await proc.process_thumbs_up(None, {})
        assert sig.value == 1.0

    @pytest.mark.asyncio
    async def test_thumbs_down(self):
        proc = ExplicitFeedbackProcessor()
        sig = await proc.process_thumbs_down(None, {})
        assert sig.value == -1.0

    @pytest.mark.asyncio
    async def test_rating(self):
        proc = ExplicitFeedbackProcessor()
        sig = await proc.process_rating(5, {})
        assert sig.value == 1.0
        sig2 = await proc.process_rating(1, {})
        assert sig2.value == -1.0
        sig3 = await proc.process_rating(3, {})
        assert sig3.value == pytest.approx(0.0, abs=0.01)


class TestImplicitSignalExtractor:
    @pytest.mark.asyncio
    async def test_completion(self):
        ext = ImplicitSignalExtractor()
        sig = await ext.process_completion(None, {})
        assert sig.value > 0

    @pytest.mark.asyncio
    async def test_abandonment(self):
        ext = ImplicitSignalExtractor()
        sig = await ext.process_abandonment(None, {})
        assert sig.value < 0

    @pytest.mark.asyncio
    async def test_dwell_time(self):
        ext = ImplicitSignalExtractor()
        # Short dwell = negative
        sig = await ext.process_dwell_time(1.0, {})
        assert sig.value < 0
        # Medium dwell = positive
        sig2 = await ext.process_dwell_time(10.0, {})
        assert sig2.value > 0


class TestRewardShaping:
    def test_potential_based(self):
        shaper = PotentialBasedShaping()
        s1 = StateRepresentation(recent_rewards=[0.0])
        s2 = StateRepresentation(recent_rewards=[1.0])
        shaped = shaper.shape(s1, s2, raw_reward=0.5, gamma=0.99)
        # Should be > raw_reward since s2 has higher potential
        assert shaped > 0.5

    def test_curiosity_shaper(self):
        shaper = CuriosityRewardShaper(prediction_error_scale=0.5)
        state = StateRepresentation(query_type="code", query_complexity=0.5, turn_count=1)
        r1 = shaper.shape(state, None, 0.0)
        # First visit: high intrinsic reward
        assert r1 > 0
        # Second visit: lower
        r2 = shaper.shape(state, None, 0.0)
        assert r2 < r1

    def test_composite_shaper(self):
        shaper = CompositeRewardShaper()
        shaper.add(PotentialBasedShaping(), weight=1.0)
        s = StateRepresentation()
        result = shaper.shape(s, s, 1.0)
        assert isinstance(result, float)


class TestRewardCollector:
    @pytest.fixture
    def collector(self):
        kernel = MagicMock()
        return RewardCollector(kernel)

    @pytest.mark.asyncio
    async def test_collect_explicit(self, collector):
        sig = await collector.collect_explicit_feedback("int-1", "thumbs_up", None)
        assert sig.value > 0
        assert sig.interaction_id == "int-1"

    @pytest.mark.asyncio
    async def test_collect_outcome(self, collector):
        signals = await collector.collect_outcome("int-1", success=True, metrics={"latency": 500})
        assert len(signals) >= 1
        assert signals[0].value > 0

    @pytest.mark.asyncio
    async def test_aggregate(self, collector):
        await collector.collect_explicit_feedback("int-1", "thumbs_up", None)
        await collector.collect_explicit_feedback("int-1", "rating", 4)
        total = await collector.aggregate_rewards("int-1")
        assert isinstance(total, float)

    @pytest.mark.asyncio
    async def test_clear_pending(self, collector):
        await collector.collect_explicit_feedback("int-1", "thumbs_up", None)
        collector.clear_pending("int-1")
        assert collector.get_pending_rewards("int-1") == []


# ---------------------------------------------------------------------------
# Experience Buffer
# ---------------------------------------------------------------------------
from aion.learning.experience.buffer import ExperienceBuffer
from aion.learning.experience.transition import TransitionBuilder, NStepTransitionBuilder


class TestExperienceBuffer:
    def test_add_and_len(self):
        buf = ExperienceBuffer(BufferConfig(max_size=100, min_size_for_sampling=2))
        for i in range(10):
            buf.add(Experience(reward=float(i)))
        assert len(buf) == 10

    def test_sample_not_ready(self):
        buf = ExperienceBuffer(BufferConfig(min_size_for_sampling=100))
        buf.add(Experience(reward=1.0))
        exps, w, idx = buf.sample(5)
        assert exps == []

    def test_sample_uniform(self):
        buf = ExperienceBuffer(BufferConfig(max_size=100, min_size_for_sampling=5, use_priority=False))
        for i in range(20):
            buf.add(Experience(reward=float(i)))
        exps, weights, indices = buf.sample(5)
        assert len(exps) == 5
        assert np.all(weights == 1.0)

    def test_sample_priority(self):
        buf = ExperienceBuffer(BufferConfig(max_size=100, min_size_for_sampling=5, use_priority=True))
        for i in range(20):
            buf.add(Experience(reward=float(i)))
        exps, weights, indices = buf.sample(5)
        assert len(exps) == 5
        assert len(weights) == 5

    def test_update_priorities(self):
        buf = ExperienceBuffer(BufferConfig(max_size=100, min_size_for_sampling=2))
        for _ in range(5):
            buf.add(Experience())
        buf.update_priorities([0, 1, 2], np.array([0.1, 0.5, 1.0]))
        # Verify via the sum-tree: higher TD error → higher priority
        p0 = buf._sampler._tree[0 + buf._sampler.capacity]
        p2 = buf._sampler._tree[2 + buf._sampler.capacity]
        assert p2 > p0

    def test_get_stats(self):
        buf = ExperienceBuffer()
        stats = buf.get_stats()
        assert "size" in stats
        assert "ready" in stats

    def test_get_by_interaction(self):
        buf = ExperienceBuffer(BufferConfig(max_size=100, min_size_for_sampling=1))
        buf.add(Experience(interaction_id="a"))
        buf.add(Experience(interaction_id="b"))
        buf.add(Experience(interaction_id="a"))
        assert len(buf.get_by_interaction("a")) == 2


class TestNStepTransitionBuilder:
    def test_single_step(self):
        builder = NStepTransitionBuilder(n=1, gamma=0.99)
        exp = Experience(reward=1.0)
        result = builder.add(exp)
        assert result is not None
        assert result.reward == 1.0

    def test_three_step(self):
        builder = NStepTransitionBuilder(n=3, gamma=0.99)
        e1 = Experience(reward=1.0)
        assert builder.add(e1) is None
        e2 = Experience(reward=2.0)
        assert builder.add(e2) is None
        e3 = Experience(reward=3.0, done=True)
        result = builder.add(e3)
        assert result is not None
        expected = 1.0 + 0.99 * 2.0 + 0.99**2 * 3.0
        assert result.reward == pytest.approx(expected, abs=0.01)

    def test_flush(self):
        builder = NStepTransitionBuilder(n=3, gamma=0.99)
        builder.add(Experience(reward=1.0))
        builder.add(Experience(reward=2.0))
        results = builder.flush()
        assert len(results) == 2


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------
from aion.learning.policies.tool_policy import ToolSelectionPolicy
from aion.learning.policies.planning_policy import PlanningStrategyPolicy
from aion.learning.policies.agent_policy import AgentBehaviorPolicy
from aion.learning.policies.optimizer import PolicyOptimizer


class TestToolSelectionPolicy:
    @pytest.mark.asyncio
    async def test_select_action(self):
        policy = ToolSelectionPolicy(PolicyConfig(name="test"))
        state = StateRepresentation(query_complexity=0.5)
        choice, conf = await policy.select_action(state, ["tool_a", "tool_b", "tool_c"])
        assert choice in ["tool_a", "tool_b", "tool_c"]

    @pytest.mark.asyncio
    async def test_update(self):
        policy = ToolSelectionPolicy(PolicyConfig(name="test", learning_rate=0.1))
        exp = Experience(
            state=StateRepresentation(query_complexity=0.5),
            action=Action(choice="tool_a"),
            cumulative_reward=1.0,
        )
        metrics = await policy.update([exp], np.array([1.0]))
        assert "loss" in metrics
        assert "td_errors" in metrics
        assert len(metrics["td_errors"]) == 1

    @pytest.mark.asyncio
    async def test_tool_rankings(self):
        policy = ToolSelectionPolicy(PolicyConfig(name="test", learning_rate=0.5))
        state = StateRepresentation(query_complexity=0.5)
        # Train with positive reward for tool_a
        for _ in range(10):
            exp = Experience(
                state=state,
                action=Action(choice="tool_a"),
                cumulative_reward=1.0,
            )
            await policy.update([exp], np.array([1.0]))
        # Train with negative reward for tool_b
        for _ in range(10):
            exp = Experience(
                state=state,
                action=Action(choice="tool_b"),
                cumulative_reward=-1.0,
            )
            await policy.update([exp], np.array([1.0]))

        rankings = policy.get_tool_rankings(state)
        assert len(rankings) == 2
        assert rankings[0][0] == "tool_a"


class TestPlanningStrategyPolicy:
    @pytest.mark.asyncio
    async def test_select_and_update(self):
        policy = PlanningStrategyPolicy(PolicyConfig(name="test"))
        state = StateRepresentation()
        choice, _ = await policy.select_action(state, ["greedy", "mcts", "hierarchical"])
        assert choice in ["greedy", "mcts", "hierarchical"]

        exp = Experience(
            state=state,
            action=Action(choice="mcts"),
            cumulative_reward=0.8,
        )
        metrics = await policy.update([exp], np.array([1.0]))
        assert "loss" in metrics


class TestAgentBehaviorPolicy:
    @pytest.mark.asyncio
    async def test_select_and_update(self):
        policy = AgentBehaviorPolicy(PolicyConfig(name="test", entropy_coefficient=0.02))
        state = StateRepresentation()
        choice, _ = await policy.select_action(state, ["analytical", "creative", "cautious"])
        assert choice in ["analytical", "creative", "cautious"]

        exp = Experience(
            state=state,
            action=Action(choice="analytical"),
            cumulative_reward=0.9,
        )
        metrics = await policy.update([exp], np.array([1.0]))
        assert "loss" in metrics


class TestPolicyOptimizer:
    @pytest.mark.asyncio
    async def test_select_action_no_policy(self):
        kernel = MagicMock()
        buf = ExperienceBuffer(BufferConfig(min_size_for_sampling=1))
        opt = PolicyOptimizer(kernel, buf)
        state = StateRepresentation()
        action = await opt.select_action(ActionType.TOOL_SELECTION, state, ["a", "b"])
        assert action.choice in ["a", "b"]
        assert action.exploration is True

    @pytest.mark.asyncio
    async def test_select_action_with_policy(self):
        kernel = MagicMock()
        buf = ExperienceBuffer(BufferConfig(min_size_for_sampling=1))
        opt = PolicyOptimizer(kernel, buf)
        policy = ToolSelectionPolicy(PolicyConfig(name="test", exploration_rate=0.0))
        opt.register_policy(ActionType.TOOL_SELECTION, policy)
        state = StateRepresentation()
        action = await opt.select_action(ActionType.TOOL_SELECTION, state, ["a", "b"])
        assert action.choice in ["a", "b"]


# ---------------------------------------------------------------------------
# Bandits
# ---------------------------------------------------------------------------
from aion.learning.bandits.thompson import ThompsonSampling, ContextualThompsonSampling
from aion.learning.bandits.ucb import UCB1, SlidingWindowUCB
from aion.learning.bandits.contextual import LinUCB, HybridLinUCB


class TestThompsonSampling:
    def test_select_cold_start(self):
        ts = ThompsonSampling()
        arm = ts.select(["a", "b", "c"])
        assert arm in ["a", "b", "c"]

    def test_select_converges(self):
        ts = ThompsonSampling()
        # Arm "good" always gets reward 1, "bad" always 0
        for _ in range(100):
            ts.update("good", 1.0)
            ts.update("bad", 0.0)
        expected = ts.get_expected_rewards()
        assert expected["good"] > expected["bad"]

    def test_select_top_k(self):
        ts = ThompsonSampling()
        for _ in range(50):
            ts.update("a", 0.9)
            ts.update("b", 0.5)
            ts.update("c", 0.1)
        top = ts.select_top_k(2, ["a", "b", "c"])
        assert len(top) == 2


class TestContextualThompsonSampling:
    def test_select_and_update(self):
        cts = ContextualThompsonSampling(feature_dim=3)
        ctx = np.array([1.0, 0.5, 0.2])
        arm = cts.select(ctx, ["x", "y"])
        assert arm in ["x", "y"]
        cts.update("x", ctx, 1.0)
        er = cts.get_expected_reward("x", ctx)
        assert isinstance(er, float)


class TestUCB1:
    def test_select_pulls_all_first(self):
        ucb = UCB1()
        ucb.add_arm("a")
        ucb.add_arm("b")
        ucb.add_arm("c")
        # Should pull unpulled arms first
        first = ucb.select(["a", "b", "c"])
        ucb.update(first, 0.5)
        second = ucb.select(["a", "b", "c"])
        assert second != first  # Should pull a different one

    def test_converges(self):
        ucb = UCB1(confidence=1.0)
        for _ in range(200):
            ucb.update("good", 1.0)
            ucb.update("bad", 0.0)
        scores = ucb.get_ucb_scores()
        assert scores["good"][0] > scores["bad"][0]


class TestSlidingWindowUCB:
    def test_non_stationary(self):
        sw = SlidingWindowUCB(window_size=20)
        # Initially "a" is good
        for _ in range(30):
            sw.update("a", 1.0)
            sw.update("b", 0.0)
        # Then "b" becomes good
        for _ in range(30):
            sw.update("a", 0.0)
            sw.update("b", 1.0)
        stats = sw.get_stats()
        assert stats["b"]["avg_reward"] > stats["a"]["avg_reward"]


class TestLinUCB:
    def test_select_and_update(self):
        linucb = LinUCB(feature_dim=3, alpha=1.0)
        ctx = np.array([1.0, 0.5, 0.2])
        arm, score = linucb.select(ctx, ["a", "b"])
        assert arm in ["a", "b"]
        linucb.update("a", ctx, 1.0)
        er = linucb.get_expected_reward("a", ctx)
        assert isinstance(er, float)


class TestHybridLinUCB:
    def test_select_and_update(self):
        hybrid = HybridLinUCB(shared_dim=3, arm_dim=2)
        shared_ctx = np.array([1.0, 0.5, 0.2])
        arm_ctxs = {"a": np.array([0.1, 0.2]), "b": np.array([0.3, 0.4])}
        arm, score = hybrid.select(shared_ctx, arm_ctxs, ["a", "b"])
        assert arm in ["a", "b"]
        hybrid.update("a", shared_ctx, arm_ctxs["a"], 1.0)


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------
from aion.learning.experiments.framework import ABTestingFramework
from aion.learning.experiments.experiment import ExperimentManager
from aion.learning.experiments.analysis import StatisticalAnalyzer, SequentialTester


class TestStatisticalAnalyzer:
    def test_welch_t_test_significant(self):
        np.random.seed(42)
        control = list(np.random.normal(0.0, 1.0, 200))
        treatment = list(np.random.normal(0.5, 1.0, 200))
        result = StatisticalAnalyzer.welch_t_test(control, treatment)
        assert result.significant == True  # noqa: E712 (np.bool_ vs bool)
        assert result.effect_size > 0

    def test_welch_t_test_not_significant(self):
        np.random.seed(42)
        control = list(np.random.normal(0.0, 1.0, 200))
        treatment = list(np.random.normal(0.0, 1.0, 200))
        result = StatisticalAnalyzer.welch_t_test(control, treatment)
        # May or may not be significant by chance, but effect size should be small
        assert abs(result.effect_size) < 0.5

    def test_mann_whitney(self):
        control = [1.0, 2.0, 3.0, 4.0, 5.0]
        treatment = [6.0, 7.0, 8.0, 9.0, 10.0]
        result = StatisticalAnalyzer.mann_whitney_u(control, treatment)
        assert result.significant == True  # noqa: E712


class TestSequentialTester:
    def test_obrien_fleming(self):
        tester = SequentialTester(total_alpha=0.05, max_looks=5)
        # Early looks should have very tight boundaries
        b1 = tester.get_boundary(1)
        b5 = tester.get_boundary(5)
        assert b1 < b5  # O'Brien-Fleming: stricter early

    def test_should_stop(self):
        tester = SequentialTester(total_alpha=0.05, max_looks=5)
        # O'Brien-Fleming at look 1/5 has a very tight boundary (~1e-5),
        # so use an extremely small p-value to trigger early stopping.
        assert tester.should_stop(1e-8) is True


class TestExperimentManager:
    def test_lifecycle(self):
        mgr = ExperimentManager()
        exp = mgr.create("test", ActionType.TOOL_SELECTION, {}, {})
        assert exp.status == ExperimentStatus.DRAFT
        assert mgr.start(exp.id) is True
        assert exp.status == ExperimentStatus.RUNNING
        assert mgr.pause(exp.id) is True
        assert exp.status == ExperimentStatus.PAUSED
        assert mgr.resume(exp.id) is True
        assert exp.status == ExperimentStatus.RUNNING
        assert mgr.complete(exp.id) is True
        assert exp.status == ExperimentStatus.COMPLETED


class TestABTestingFramework:
    @pytest.fixture
    def framework(self):
        kernel = MagicMock()
        return ABTestingFramework(kernel)

    @pytest.mark.asyncio
    async def test_create_and_start(self, framework):
        exp = await framework.create_experiment(
            "test", ActionType.TOOL_SELECTION, {"lr": 0.01}, {"lr": 0.1}
        )
        assert exp.status == ExperimentStatus.DRAFT
        ok = await framework.start_experiment(exp.id)
        assert ok is True

    @pytest.mark.asyncio
    async def test_variant_assignment(self, framework):
        exp = await framework.create_experiment(
            "test", ActionType.TOOL_SELECTION, {}, {}, traffic_split=0.5,
        )
        await framework.start_experiment(exp.id)
        v1 = framework.get_variant(exp.id, "user_1")
        v2 = framework.get_variant(exp.id, "user_1")
        assert v1.id == v2.id  # Same user = same variant

    @pytest.mark.asyncio
    async def test_record_result(self, framework):
        exp = await framework.create_experiment(
            "test", ActionType.TOOL_SELECTION, {}, {},
        )
        await framework.start_experiment(exp.id)
        await framework.record_result(exp.id, exp.control.id, 1.0)
        assert exp.control.sample_count == 1
        assert exp.control.avg_reward == 1.0


# ---------------------------------------------------------------------------
# Main RL Loop
# ---------------------------------------------------------------------------
from aion.learning.loop import ReinforcementLearningLoop


class TestReinforcementLearningLoop:
    @pytest.fixture
    def rl_loop(self):
        kernel = MagicMock()
        return ReinforcementLearningLoop(kernel)

    @pytest.mark.asyncio
    async def test_initialize_and_shutdown(self, rl_loop):
        await rl_loop.initialize()
        assert rl_loop._initialized is True
        stats = rl_loop.get_stats()
        assert stats["initialized"] is True
        await rl_loop.shutdown()
        assert rl_loop._initialized is False

    @pytest.mark.asyncio
    async def test_interaction_lifecycle(self, rl_loop):
        await rl_loop.initialize()

        state = StateRepresentation(query_complexity=0.5)
        await rl_loop.start_interaction("int-1", state)
        assert "int-1" in rl_loop._current_interactions

        action = await rl_loop.select_action(
            ActionType.TOOL_SELECTION, state, ["tool_a", "tool_b"], "int-1"
        )
        assert action.choice in ["tool_a", "tool_b"]

        await rl_loop.collect_feedback("int-1", "thumbs_up", None)
        await rl_loop.collect_outcome("int-1", success=True, metrics={"latency": 500})

        total_reward = await rl_loop.end_interaction("int-1", state)
        assert isinstance(total_reward, float)
        assert "int-1" not in rl_loop._current_interactions

        await rl_loop.shutdown()

    @pytest.mark.asyncio
    async def test_select_tool(self, rl_loop):
        await rl_loop.initialize()
        state = StateRepresentation()
        tool = await rl_loop.select_tool(state, ["hammer", "screwdriver"])
        assert tool in ["hammer", "screwdriver"]
        await rl_loop.shutdown()

    @pytest.mark.asyncio
    async def test_select_strategy(self, rl_loop):
        await rl_loop.initialize()
        state = StateRepresentation()
        strategy = await rl_loop.select_strategy(state, ["greedy", "mcts"])
        assert strategy in ["greedy", "mcts"]
        await rl_loop.shutdown()

    @pytest.mark.asyncio
    async def test_get_stats(self, rl_loop):
        await rl_loop.initialize()
        stats = rl_loop.get_stats()
        assert "experience_buffer" in stats
        assert "policy_optimizer" in stats
        assert "bandits" in stats
        assert "experiments" in stats
        await rl_loop.shutdown()

    @pytest.mark.asyncio
    async def test_disabled_loop(self, rl_loop):
        rl_loop._config.enabled = False
        await rl_loop.initialize()
        state = StateRepresentation()
        action = await rl_loop.select_action(
            ActionType.TOOL_SELECTION, state, ["a", "b"]
        )
        assert action.choice in ["a", "b"]
        await rl_loop.shutdown()


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------
from aion.learning.integration.evolution import EvolutionIntegration
from aion.learning.integration.tools import ToolLearningIntegration


class TestEvolutionIntegration:
    @pytest.mark.asyncio
    async def test_get_improvement_candidates(self):
        kernel = MagicMock()
        rl = ReinforcementLearningLoop(kernel)
        await rl.initialize()
        evo = EvolutionIntegration(rl)
        candidates = await evo.get_improvement_candidates()
        assert isinstance(candidates, list)
        await rl.shutdown()


class TestToolLearningIntegration:
    @pytest.mark.asyncio
    async def test_select_best_tool(self):
        kernel = MagicMock()
        rl = ReinforcementLearningLoop(kernel)
        await rl.initialize()
        tools = ToolLearningIntegration(rl)
        tool = await tools.select_best_tool("test query", ["a", "b"])
        assert tool in ["a", "b"]
        await rl.shutdown()

    @pytest.mark.asyncio
    async def test_record_tool_outcome(self):
        kernel = MagicMock()
        rl = ReinforcementLearningLoop(kernel)
        await rl.initialize()
        tools = ToolLearningIntegration(rl)

        state = StateRepresentation()
        await rl.start_interaction("int-1", state)
        await tools.record_tool_outcome("int-1", "tool_a", success=True, latency_ms=500)
        await rl.end_interaction("int-1")

        perf = tools.get_tool_performance()
        assert isinstance(perf, dict)
        await rl.shutdown()


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
from aion.learning.persistence.repository import LearningStateRepository


class TestLearningStateRepository:
    @pytest.mark.asyncio
    async def test_save_checkpoint(self, tmp_path):
        kernel = MagicMock()
        rl = ReinforcementLearningLoop(kernel)
        await rl.initialize()

        from aion.learning.config import PersistenceConfig
        config = PersistenceConfig(checkpoint_dir=str(tmp_path / "checkpoints"))
        repo = LearningStateRepository(rl, config)

        path = await repo.save_checkpoint("test_checkpoint")
        assert "test_checkpoint" in path

        checkpoints = await repo.list_checkpoints()
        assert len(checkpoints) == 1

        loaded = await repo.load_checkpoint(path)
        assert loaded is True
        await rl.shutdown()


# ---------------------------------------------------------------------------
# End-to-end integration test
# ---------------------------------------------------------------------------

class TestEndToEndLearningLoop:
    @pytest.mark.asyncio
    async def test_full_learning_cycle(self):
        """Simulate a complete learning cycle with multiple interactions."""
        kernel = MagicMock()
        rl = ReinforcementLearningLoop(kernel, LearningConfig(
            buffer=BufferConfig(min_size_for_sampling=5),
        ))
        await rl.initialize()

        # Simulate 20 interactions
        for i in range(20):
            state = StateRepresentation(
                query_complexity=float(i % 5) / 5.0,
                turn_count=i,
            )
            interaction_id = f"int-{i}"
            await rl.start_interaction(interaction_id, state)

            tool = await rl.select_tool(state, ["search", "calculate", "fetch"], interaction_id)
            await rl.collect_feedback(interaction_id, "thumbs_up" if i % 3 != 0 else "thumbs_down", None)
            await rl.collect_outcome(interaction_id, success=(i % 4 != 0))

            await rl.end_interaction(interaction_id, state)

        # Check that the buffer has data
        stats = rl.get_stats()
        assert stats["experience_buffer"]["total_added"] > 0

        # Check bandits have been updated
        bandit_stats = rl.get_bandit_stats("tools")
        assert len(bandit_stats) > 0

        await rl.shutdown()


# ---------------------------------------------------------------------------
# SOTA Components: MLP, TargetNetwork, StateValueFunction, RND
# ---------------------------------------------------------------------------
from aion.learning.nn import MLP
from aion.learning.policies.value_function import (
    TargetNetwork,
    StateValueFunction,
)
from aion.learning.config import ValueFunctionConfig, RNDConfig
from aion.learning.rewards.rnd import RNDCuriosityShaper


class TestMLP:
    def test_forward_single(self):
        mlp = MLP(4, 8, 1)
        x = np.array([1.0, 2.0, 3.0, 4.0])
        out = mlp.forward(x)
        assert out.shape == (1, 1)

    def test_forward_batch(self):
        mlp = MLP(4, 8, 2)
        x = np.random.randn(5, 4)
        out = mlp.forward(x)
        assert out.shape == (5, 2)

    def test_backward_and_apply(self):
        mlp = MLP(3, 6, 1)
        x = np.random.randn(4, 3)
        out = mlp.forward(x)
        d_out = np.ones_like(out)
        grads = mlp.backward(d_out, clip=1.0)
        assert set(grads.keys()) == {"W1", "b1", "W2", "b2"}
        assert grads["W1"].shape == mlp.W1.shape
        assert grads["b1"].shape == mlp.b1.shape
        assert grads["W2"].shape == mlp.W2.shape
        assert grads["b2"].shape == mlp.b2.shape
        mlp.apply_gradients(grads, lr=0.01)

    def test_gradient_clipping(self):
        mlp = MLP(3, 6, 1)
        x = np.random.randn(1, 3) * 100
        mlp.forward(x)
        d_out = np.array([[1000.0]])
        grads = mlp.backward(d_out, clip=0.5)
        total_norm = np.sqrt(sum(np.sum(g ** 2) for g in grads.values()))
        assert total_norm <= 0.5 + 1e-6

    def test_copy(self):
        mlp = MLP(3, 6, 1)
        clone = mlp.copy()
        np.testing.assert_array_equal(clone.W1, mlp.W1)
        np.testing.assert_array_equal(clone.b1, mlp.b1)
        clone.W1[0, 0] = 999.0
        assert mlp.W1[0, 0] != 999.0

    def test_get_set_params(self):
        mlp = MLP(3, 6, 1)
        params = mlp.get_params()
        mlp2 = MLP(3, 6, 1)
        mlp2.set_params(params)
        np.testing.assert_array_equal(mlp2.W1, mlp.W1)

    def test_learning_reduces_loss(self):
        """Train MLP to fit y = mean(x) and verify loss decreases."""
        np.random.seed(42)
        mlp = MLP(3, 16, 1)
        X = np.random.randn(32, 3)
        y = X.mean(axis=1, keepdims=True)

        losses = []
        for _ in range(50):
            out = mlp.forward(X)
            loss = float(np.mean((out - y) ** 2))
            losses.append(loss)
            d_out = 2 * (out - y) / len(X)
            grads = mlp.backward(d_out, clip=5.0)
            mlp.apply_gradients(grads, lr=0.01)

        assert losses[-1] < losses[0]


class TestTargetNetwork:
    def test_soft_update(self):
        source = MLP(3, 6, 1)
        target = TargetNetwork(source, tau=0.5)
        src_params = source.get_params()
        tgt_params = target.get_params()
        np.testing.assert_array_almost_equal(src_params["W1"], tgt_params["W1"])

        source.W1 += 10.0
        target.soft_update(source)
        tgt_params_after = target.get_params()
        expected_W1 = 0.5 * source.W1 + 0.5 * src_params["W1"]
        np.testing.assert_array_almost_equal(tgt_params_after["W1"], expected_W1)

    def test_hard_update(self):
        source = MLP(3, 6, 1)
        target = TargetNetwork(source, tau=0.01)
        source.W1 += 100.0
        target.hard_update(source)
        np.testing.assert_array_equal(target.get_params()["W1"], source.W1)

    def test_forward(self):
        source = MLP(3, 6, 1)
        target = TargetNetwork(source, tau=0.01)
        x = np.array([1.0, 2.0, 3.0])
        out = target.forward(x)
        assert out.shape == (1, 1)


class TestStateValueFunction:
    def test_predict(self):
        vf = StateValueFunction(feature_dim=5)
        features = np.random.randn(5)
        v = vf.predict(features)
        assert isinstance(v, float)

    def test_predict_batch(self):
        vf = StateValueFunction(feature_dim=5)
        features = np.random.randn(10, 5)
        values = vf.predict_batch(features)
        assert values.shape == (10,)

    def test_predict_target(self):
        vf = StateValueFunction(feature_dim=5)
        features = np.random.randn(5)
        v = vf.predict_target(features)
        assert isinstance(v, float)

    def test_update_returns_metrics(self):
        vf = StateValueFunction(feature_dim=4)
        states = np.random.randn(8, 4)
        rewards = np.random.randn(8)
        next_states = np.random.randn(8, 4)
        dones = np.zeros(8)
        metrics = vf.update(states, rewards, next_states, dones)
        assert "value_loss" in metrics
        assert "mean_value" in metrics
        assert "td_errors" in metrics
        assert len(metrics["td_errors"]) == 8

    def test_update_with_importance_weights(self):
        vf = StateValueFunction(feature_dim=4)
        states = np.random.randn(8, 4)
        rewards = np.random.randn(8)
        next_states = np.random.randn(8, 4)
        dones = np.zeros(8)
        weights = np.ones(8) * 0.5
        metrics = vf.update(states, rewards, next_states, dones, importance_weights=weights)
        assert "value_loss" in metrics

    def test_td_learning_improves_predictions(self):
        """Verify TD(0) updates move V(s) toward the observed returns."""
        np.random.seed(42)
        vf = StateValueFunction(
            feature_dim=3,
            config=ValueFunctionConfig(learning_rate=0.01, hidden_dim=16),
        )
        state = np.array([1.0, 0.5, 0.2])
        for _ in range(100):
            vf.update(
                states=state.reshape(1, -1),
                rewards=np.array([1.0]),
                next_states=state.reshape(1, -1),
                dones=np.array([1.0]),
            )
        v = vf.predict(state)
        assert abs(v - 1.0) < 0.5

    def test_compute_gae_empty(self):
        vf = StateValueFunction(feature_dim=3)
        adv, targets = vf.compute_gae([], [], [], [])
        assert len(adv) == 0
        assert len(targets) == 0

    def test_compute_gae_single_step(self):
        vf = StateValueFunction(feature_dim=3)
        s = [np.array([1.0, 0.5, 0.2])]
        ns = [np.array([0.5, 0.3, 0.1])]
        adv, targets = vf.compute_gae(s, [1.0], ns, [False])
        assert len(adv) == 1
        assert len(targets) == 1

    def test_compute_gae_multi_step(self):
        np.random.seed(42)
        vf = StateValueFunction(
            feature_dim=3,
            config=ValueFunctionConfig(gamma=0.99, gae_lambda=0.95),
        )
        T = 5
        states = [np.random.randn(3) for _ in range(T)]
        next_states = states[1:] + [np.random.randn(3)]
        rewards = [1.0, 0.5, 0.0, -0.5, 1.0]
        dones = [False, False, False, False, True]

        adv, targets = vf.compute_gae(states, rewards, next_states, dones)
        assert adv.shape == (5,)
        assert targets.shape == (5,)
        values = vf.predict_batch(np.array(states))
        np.testing.assert_array_almost_equal(targets, adv + values)

    def test_get_stats(self):
        vf = StateValueFunction(feature_dim=5)
        stats = vf.get_stats()
        assert stats["initialized"] is False
        vf.predict(np.random.randn(5))
        stats = vf.get_stats()
        assert stats["initialized"] is True
        assert stats["feature_dim"] == 5

    def test_get_set_params(self):
        vf = StateValueFunction(feature_dim=3)
        vf.predict(np.random.randn(3))
        params = vf.get_params()
        assert params is not None
        vf2 = StateValueFunction(feature_dim=3)
        vf2.predict(np.random.randn(3))
        vf2.set_params(params)
        x = np.random.randn(3)
        assert vf.predict(x) == vf2.predict(x)


class TestRNDCuriosityShaper:
    def test_intrinsic_reward(self):
        rnd = RNDCuriosityShaper(RNDConfig(normalise_intrinsic=False))
        features = np.random.randn(5)
        reward = rnd.compute_intrinsic_reward(features)
        assert isinstance(reward, float)
        assert reward >= 0.0

    def test_intrinsic_batch(self):
        rnd = RNDCuriosityShaper(RNDConfig(normalise_intrinsic=False))
        features = np.random.randn(10, 5)
        rewards = rnd.compute_intrinsic_batch(features)
        assert rewards.shape == (10,)

    def test_train_predictor_reduces_error(self):
        """Training the predictor should reduce prediction error on the same data."""
        np.random.seed(42)
        rnd = RNDCuriosityShaper(RNDConfig(
            normalise_intrinsic=False,
            learning_rate=0.01,
            hidden_dim=32,
        ))
        features = np.random.randn(16, 5)

        initial_loss = rnd.train_predictor(features)["rnd_loss"]
        for _ in range(50):
            rnd.train_predictor(features)
        final_loss = rnd.train_predictor(features)["rnd_loss"]
        assert final_loss < initial_loss

    def test_novel_states_have_higher_reward(self):
        """States never seen during training should have higher intrinsic reward."""
        np.random.seed(42)
        rnd = RNDCuriosityShaper(RNDConfig(
            normalise_intrinsic=False,
            learning_rate=0.01,
        ))
        familiar = np.random.randn(20, 5)
        novel = np.random.randn(5) + 10.0

        for _ in range(100):
            rnd.train_predictor(familiar)

        familiar_reward = rnd.compute_intrinsic_reward(familiar[0])
        novel_reward = rnd.compute_intrinsic_reward(novel)
        assert novel_reward > familiar_reward

    def test_shape_interface(self):
        """RNDCuriosityShaper implements the RewardShaper interface."""
        rnd = RNDCuriosityShaper()
        state = StateRepresentation(query_complexity=0.5, turn_count=1)
        shaped = rnd.shape(state, state, raw_reward=1.0)
        assert isinstance(shaped, float)
        assert shaped >= 1.0

    def test_normalisation(self):
        """With normalisation, intrinsic rewards should not explode."""
        rnd = RNDCuriosityShaper(RNDConfig(normalise_intrinsic=True))
        rewards = []
        for _ in range(100):
            features = np.random.randn(5) * 10
            r = rnd.compute_intrinsic_reward(features)
            rewards.append(r)
        assert np.std(rewards) < 10.0

    def test_get_stats(self):
        rnd = RNDCuriosityShaper()
        stats = rnd.get_stats()
        assert "initialized" in stats
        assert stats["initialized"] is False
        rnd.compute_intrinsic_reward(np.random.randn(5))
        stats = rnd.get_stats()
        assert stats["initialized"] is True


class TestActorCriticIntegration:
    """Test the full actor-critic pipeline through the PolicyOptimizer."""

    @pytest.mark.asyncio
    async def test_optimizer_has_value_function_and_rnd(self):
        kernel = MagicMock()
        buf = ExperienceBuffer(BufferConfig(min_size_for_sampling=1))
        opt = PolicyOptimizer(kernel, buf)
        assert hasattr(opt, "value_function")
        assert hasattr(opt, "rnd")
        assert isinstance(opt.value_function, StateValueFunction)
        assert isinstance(opt.rnd, RNDCuriosityShaper)

    @pytest.mark.asyncio
    async def test_optimizer_stats_include_critic_and_rnd(self):
        kernel = MagicMock()
        buf = ExperienceBuffer(BufferConfig(min_size_for_sampling=1))
        opt = PolicyOptimizer(kernel, buf)
        stats = opt.get_stats()
        assert "value_function" in stats
        assert "rnd" in stats
        assert "total_critic_loss" in stats
        assert "total_rnd_loss" in stats

    @pytest.mark.asyncio
    async def test_advantage_based_policy_update(self):
        """Verify policies accept and use advantage parameter."""
        policy = ToolSelectionPolicy(PolicyConfig(name="test", learning_rate=0.1))
        exp = Experience(
            state=StateRepresentation(query_complexity=0.5),
            action=Action(choice="tool_a"),
            cumulative_reward=1.0,
        )
        advantages = np.array([2.5])
        metrics = await policy.update([exp], np.array([1.0]), advantages=advantages)
        assert "loss" in metrics
