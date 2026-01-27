"""
AION Reinforcement Learning Loop - Main Coordinator

Orchestrates the complete RL pipeline:
  1. State observation
  2. Policy-based action selection (with exploration)
  3. Reward collection (explicit + implicit + outcome)
  4. Experience storage with priority replay
  5. Background policy optimization
  6. A/B testing integration
  7. Reward shaping (potential-based + curiosity)
  8. Bandit arm tracking for fast online learning

This module is the single integration point that ties rewards, buffers,
policies, bandits, experiments, and persistence together.
"""

from __future__ import annotations

import asyncio
import random
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import structlog

from aion.learning.config import LearningConfig
from aion.learning.types import (
    Action,
    ActionType,
    Experience,
    PolicyConfig,
    RewardSignal,
    StateRepresentation,
)
from aion.learning.rewards.collector import RewardCollector
from aion.learning.rewards.shaping import CompositeRewardShaper, PotentialBasedShaping
from aion.learning.rewards.rnd import RNDCuriosityShaper
from aion.learning.experience.buffer import ExperienceBuffer
from aion.learning.experience.transition import NStepTransitionBuilder
from aion.learning.policies.optimizer import PolicyOptimizer
from aion.learning.policies.tool_policy import ToolSelectionPolicy
from aion.learning.policies.planning_policy import PlanningStrategyPolicy
from aion.learning.policies.agent_policy import AgentBehaviorPolicy
from aion.learning.bandits.thompson import ThompsonSampling
from aion.learning.experiments.framework import ABTestingFramework

if TYPE_CHECKING:
    from aion.core.kernel import AIONKernel

logger = structlog.get_logger(__name__)


class ReinforcementLearningLoop:
    """
    Main RL Loop coordinator.

    Integrates reward collection, experience replay, policy optimization,
    bandit algorithms, A/B testing, and reward shaping into a coherent
    system-wide learning loop.
    """

    def __init__(
        self,
        kernel: "AIONKernel",
        config: Optional[LearningConfig] = None,
    ):
        self.kernel = kernel
        self._config = config or LearningConfig()

        # Sub-systems
        self.reward_collector = RewardCollector(kernel, self._config.reward)
        self.experience_buffer = ExperienceBuffer(self._config.buffer)
        self.policy_optimizer = PolicyOptimizer(
            kernel,
            self.experience_buffer,
            self._config.policy_optimizer,
            value_config=self._config.value_function,
            rnd_config=self._config.rnd,
        )
        self.ab_testing = ABTestingFramework(kernel, self._config.experiment)

        # Bandit instances for fast online learning
        self._bandits: Dict[str, ThompsonSampling] = {}

        # Reward shaping: PBRS (Ng et al., 1999) + RND curiosity (Burda et al., 2018)
        # The RND shaper is shared with the optimizer so its predictor
        # is trained during the background training loop.
        self._rnd_shaper = self.policy_optimizer.rnd
        self._reward_shaper = CompositeRewardShaper()
        self._reward_shaper.add(PotentialBasedShaping(), weight=0.3)
        self._reward_shaper.add(self._rnd_shaper, weight=0.1)

        # N-step transition builders per interaction
        self._nstep_builders: Dict[str, NStepTransitionBuilder] = {}

        # Active interaction tracking
        self._current_interactions: Dict[str, Dict[str, Any]] = {}

        self._initialized = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Initialize all learning subsystems and register default policies."""
        if self._initialized:
            return

        # Register policies for each action type
        tool_policy = ToolSelectionPolicy(PolicyConfig(
            name="tool_selection", version="1.0.0",
            learning_rate=self._config.policy_optimizer.default_learning_rate,
            exploration_rate=self._config.policy_optimizer.default_exploration_rate,
            exploration_decay=self._config.policy_optimizer.exploration_decay,
            min_exploration=self._config.policy_optimizer.min_exploration,
        ))
        self.policy_optimizer.register_policy(ActionType.TOOL_SELECTION, tool_policy)

        planning_policy = PlanningStrategyPolicy(PolicyConfig(
            name="planning_strategy", version="1.0.0",
            learning_rate=self._config.policy_optimizer.default_learning_rate,
            exploration_rate=0.15,  # Slightly more exploration for strategies
        ))
        self.policy_optimizer.register_policy(ActionType.PLANNING_STRATEGY, planning_policy)

        agent_policy = AgentBehaviorPolicy(PolicyConfig(
            name="agent_behavior", version="1.0.0",
            learning_rate=self._config.policy_optimizer.default_learning_rate,
            exploration_rate=0.12,
            entropy_coefficient=0.02,
        ))
        self.policy_optimizer.register_policy(ActionType.AGENT_ASSIGNMENT, agent_policy)

        # Initialize bandits
        self._bandits["tools"] = ThompsonSampling("tools")
        self._bandits["strategies"] = ThompsonSampling("strategies")
        self._bandits["agents"] = ThompsonSampling("agents")

        # Start background training if enabled
        if self._config.training_enabled:
            await self.policy_optimizer.start_training()

        self._initialized = True
        logger.info("rl_loop_initialized")

    async def shutdown(self) -> None:
        """Gracefully shut down the learning loop."""
        await self.policy_optimizer.stop_training()
        # Flush any remaining n-step transitions
        for builder in self._nstep_builders.values():
            for exp in builder.flush():
                self.experience_buffer.add(exp)
        self._initialized = False
        logger.info("rl_loop_shutdown")

    # ------------------------------------------------------------------
    # Interaction lifecycle
    # ------------------------------------------------------------------

    async def start_interaction(
        self,
        interaction_id: str,
        initial_state: StateRepresentation,
    ) -> None:
        """Begin tracking an interaction."""
        self._current_interactions[interaction_id] = {
            "state": initial_state,
            "actions": [],
            "experiences": [],
            "start_time": datetime.now(),
            "step_index": 0,
        }
        self._nstep_builders[interaction_id] = NStepTransitionBuilder(
            n=self._config.buffer.n_step_returns,
            gamma=self._config.reward.discount_gamma,
        )

    async def record_action(
        self,
        interaction_id: str,
        action: Action,
        state: Optional[StateRepresentation] = None,
    ) -> None:
        """Record an action taken during an interaction."""
        if interaction_id not in self._current_interactions:
            return

        interaction = self._current_interactions[interaction_id]
        current_state = state or interaction["state"]

        experience = Experience(
            state=current_state,
            action=action,
            interaction_id=interaction_id,
            step_index=interaction["step_index"],
        )
        interaction["actions"].append(action)
        interaction["experiences"].append(experience)
        interaction["step_index"] += 1

    async def end_interaction(
        self,
        interaction_id: str,
        final_state: Optional[StateRepresentation] = None,
    ) -> float:
        """End an interaction, compute rewards, store experiences."""
        if interaction_id not in self._current_interactions:
            return 0.0

        interaction = self._current_interactions[interaction_id]
        total_reward = await self.reward_collector.aggregate_rewards(interaction_id)
        rewards = self.reward_collector.get_pending_rewards(interaction_id)

        for exp in interaction["experiences"]:
            exp.rewards = rewards
            exp.reward = total_reward / max(len(interaction["experiences"]), 1)
            exp.next_state = final_state
            exp.done = True

            # Apply reward shaping
            exp.reward = self._reward_shaper.shape(
                exp.state, exp.next_state, exp.reward,
                gamma=self._config.reward.discount_gamma,
            )
            exp.compute_cumulative_reward(self._config.reward.discount_gamma)

            # N-step transition building
            builder = self._nstep_builders.get(interaction_id)
            if builder:
                nstep_exp = builder.add(exp)
                if nstep_exp:
                    self.experience_buffer.add(nstep_exp)
            else:
                self.experience_buffer.add(exp)

        # Flush remaining n-step transitions
        builder = self._nstep_builders.pop(interaction_id, None)
        if builder:
            for nstep_exp in builder.flush():
                self.experience_buffer.add(nstep_exp)

        # Update bandits
        for action in interaction["actions"]:
            bandit_key = {
                ActionType.TOOL_SELECTION: "tools",
                ActionType.PLANNING_STRATEGY: "strategies",
                ActionType.AGENT_ASSIGNMENT: "agents",
            }.get(action.action_type)
            if bandit_key and bandit_key in self._bandits:
                normalised = max(0.0, min(1.0, (total_reward + 1) / 2))
                self._bandits[bandit_key].update(action.choice, normalised)

        self.reward_collector.clear_pending(interaction_id)
        del self._current_interactions[interaction_id]
        return total_reward

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    async def select_action(
        self,
        action_type: ActionType,
        state: StateRepresentation,
        available_actions: List[str],
        interaction_id: Optional[str] = None,
    ) -> Action:
        """Select an action using the appropriate policy."""
        if not self._config.enabled or not available_actions:
            choice = random.choice(available_actions) if available_actions else ""
            return Action(
                action_type=action_type,
                choice=choice,
                alternatives=list(available_actions),
            )

        # Check A/B experiments
        experiments = self.ab_testing.get_active_experiments(action_type)
        exp_meta: Dict[str, str] = {}
        if experiments and state.user_id:
            exp = experiments[0]
            variant = self.ab_testing.get_variant(exp.id, state.user_id)
            if variant:
                exp_meta = {"experiment_id": exp.id, "variant_id": variant.id}

        # Select via policy
        action = await self.policy_optimizer.select_action(
            action_type, state, available_actions,
        )
        action.parameters.update(exp_meta)

        # Record if in active interaction
        if interaction_id:
            await self.record_action(interaction_id, action, state)

        return action

    async def select_tool(
        self,
        state: StateRepresentation,
        available_tools: List[str],
        interaction_id: Optional[str] = None,
    ) -> str:
        """Convenience: select a tool."""
        action = await self.select_action(
            ActionType.TOOL_SELECTION, state, available_tools, interaction_id,
        )
        return action.choice

    async def select_strategy(
        self,
        state: StateRepresentation,
        available_strategies: List[str],
        interaction_id: Optional[str] = None,
    ) -> str:
        """Convenience: select a planning strategy."""
        action = await self.select_action(
            ActionType.PLANNING_STRATEGY, state, available_strategies, interaction_id,
        )
        return action.choice

    # ------------------------------------------------------------------
    # Feedback collection
    # ------------------------------------------------------------------

    async def collect_feedback(
        self,
        interaction_id: str,
        feedback_type: str,
        value: Any,
        action_id: Optional[str] = None,
    ) -> RewardSignal:
        """Collect explicit user feedback."""
        signal = await self.reward_collector.collect_explicit_feedback(
            interaction_id, feedback_type, value, action_id,
        )

        # Forward to A/B testing
        if interaction_id in self._current_interactions:
            for action in self._current_interactions[interaction_id]["actions"]:
                if "experiment_id" in action.parameters:
                    await self.ab_testing.record_result(
                        action.parameters["experiment_id"],
                        action.parameters["variant_id"],
                        signal.value,
                    )
        return signal

    async def collect_outcome(
        self,
        interaction_id: str,
        success: bool,
        partial: bool = False,
        metrics: Optional[Dict[str, float]] = None,
    ) -> List[RewardSignal]:
        """Collect outcome signals."""
        return await self.reward_collector.collect_outcome(
            interaction_id, success, partial, metrics,
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_tool_rankings(self, state: StateRepresentation) -> List[tuple]:
        """Get tool rankings from the tool selection policy."""
        policy = self.policy_optimizer.get_policy(ActionType.TOOL_SELECTION)
        if policy and hasattr(policy, "get_tool_rankings"):
            return policy.get_tool_rankings(state)
        return []

    def get_bandit_stats(self, bandit_name: str) -> Dict[str, Any]:
        bandit = self._bandits.get(bandit_name)
        if bandit:
            return bandit.get_stats()
        return {}

    def get_stats(self) -> Dict[str, Any]:
        return {
            "enabled": self._config.enabled,
            "initialized": self._initialized,
            "active_interactions": len(self._current_interactions),
            "experience_buffer": self.experience_buffer.get_stats(),
            "policy_optimizer": self.policy_optimizer.get_stats(),
            "reward_collector": self.reward_collector.get_stats(),
            "experiments": self.ab_testing.get_stats(),
            "bandits": {
                name: b.get_stats() for name, b in self._bandits.items()
            },
        }
