"""
Reinforcement Learning from Feedback

Implementation of reinforcement learning for agents including
policy optimization, value estimation, and experience replay.
"""

import asyncio
import math
import random
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional, Callable, Awaitable

import structlog

logger = structlog.get_logger()


class RewardType(Enum):
    """Types of reward signals."""

    TASK_COMPLETION = "task_completion"
    USER_FEEDBACK = "user_feedback"
    SELF_EVALUATION = "self_evaluation"
    OUTCOME_BASED = "outcome_based"
    PROCESS_BASED = "process_based"


@dataclass
class Experience:
    """A single experience tuple (state, action, reward, next_state)."""

    id: str
    state: dict[str, Any]
    action: str
    action_params: dict[str, Any] = field(default_factory=dict)
    reward: float = 0.0
    reward_type: RewardType = RewardType.OUTCOME_BASED
    next_state: Optional[dict[str, Any]] = None
    done: bool = False
    info: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    priority: float = 1.0  # For prioritized replay

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "state": self.state,
            "action": self.action,
            "action_params": self.action_params,
            "reward": self.reward,
            "reward_type": self.reward_type.value,
            "next_state": self.next_state,
            "done": self.done,
            "info": self.info,
            "timestamp": self.timestamp.isoformat(),
            "priority": self.priority,
        }


@dataclass
class Policy:
    """A policy for action selection."""

    id: str
    name: str
    action_preferences: dict[str, float] = field(default_factory=dict)
    context_rules: list[dict[str, Any]] = field(default_factory=list)
    exploration_rate: float = 0.1
    temperature: float = 1.0
    version: int = 1
    performance_score: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def select_action(
        self,
        available_actions: list[str],
        state: Optional[dict[str, Any]] = None,
    ) -> str:
        """Select an action using the policy."""
        # Epsilon-greedy exploration
        if random.random() < self.exploration_rate:
            return random.choice(available_actions)

        # Get action values
        action_values = {
            action: self.action_preferences.get(action, 0.0)
            for action in available_actions
        }

        # Apply context rules
        if state:
            for rule in self.context_rules:
                if self._matches_rule(state, rule.get("condition", {})):
                    action = rule.get("action")
                    if action in action_values:
                        action_values[action] += rule.get("bonus", 0.5)

        # Softmax selection
        if self.temperature > 0:
            probs = self._softmax(action_values, self.temperature)
            return random.choices(list(probs.keys()), weights=list(probs.values()))[0]

        # Greedy selection
        return max(action_values, key=action_values.get)

    def _matches_rule(self, state: dict[str, Any], condition: dict[str, Any]) -> bool:
        """Check if state matches a rule condition."""
        for key, value in condition.items():
            if state.get(key) != value:
                return False
        return True

    def _softmax(self, values: dict[str, float], temperature: float) -> dict[str, float]:
        """Apply softmax to action values."""
        max_val = max(values.values()) if values else 0
        exp_values = {k: math.exp((v - max_val) / temperature) for k, v in values.items()}
        total = sum(exp_values.values())
        return {k: v / total for k, v in exp_values.items()}

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "action_preferences": self.action_preferences,
            "context_rules": self.context_rules,
            "exploration_rate": self.exploration_rate,
            "temperature": self.temperature,
            "version": self.version,
            "performance_score": self.performance_score,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass
class ValueFunction:
    """Value function for state evaluation."""

    id: str
    state_values: dict[str, float] = field(default_factory=dict)
    feature_weights: dict[str, float] = field(default_factory=dict)
    learning_rate: float = 0.1
    discount_factor: float = 0.99

    def evaluate(self, state: dict[str, Any]) -> float:
        """Evaluate a state's value."""
        # Check for cached value
        state_key = str(sorted(state.items()))
        if state_key in self.state_values:
            return self.state_values[state_key]

        # Linear function approximation
        value = 0.0
        for feature, weight in self.feature_weights.items():
            if feature in state:
                feature_value = state[feature]
                if isinstance(feature_value, (int, float)):
                    value += weight * feature_value
                elif isinstance(feature_value, bool):
                    value += weight * (1.0 if feature_value else 0.0)

        return value

    def update(
        self,
        state: dict[str, Any],
        target: float,
    ) -> float:
        """Update value function with TD target."""
        state_key = str(sorted(state.items()))
        current_value = self.evaluate(state)

        # TD update
        td_error = target - current_value
        new_value = current_value + self.learning_rate * td_error

        self.state_values[state_key] = new_value

        # Update feature weights
        for feature in state:
            if feature in self.feature_weights:
                feature_value = state.get(feature, 0)
                if isinstance(feature_value, (int, float)):
                    self.feature_weights[feature] += self.learning_rate * td_error * feature_value

        return td_error

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "state_values_count": len(self.state_values),
            "feature_weights": self.feature_weights,
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor,
        }


@dataclass
class RLConfig:
    """Configuration for reinforcement learning."""

    # Replay buffer
    buffer_size: int = 10000
    batch_size: int = 32
    prioritized_replay: bool = True
    priority_alpha: float = 0.6
    priority_beta: float = 0.4

    # Learning
    learning_rate: float = 0.01
    discount_factor: float = 0.99
    exploration_initial: float = 1.0
    exploration_final: float = 0.1
    exploration_decay: float = 0.995

    # Policy
    policy_update_frequency: int = 10
    target_update_frequency: int = 100

    # Rewards
    reward_scaling: float = 1.0
    reward_clipping: Optional[tuple[float, float]] = (-1.0, 1.0)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "buffer_size": self.buffer_size,
            "batch_size": self.batch_size,
            "prioritized_replay": self.prioritized_replay,
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor,
            "exploration_initial": self.exploration_initial,
            "exploration_final": self.exploration_final,
        }


class ReinforcementLearner:
    """
    Reinforcement learning system for agents.

    Features:
    - Experience replay with prioritization
    - Policy gradient updates
    - Value function learning
    - Exploration-exploitation balance
    - Reward shaping with LLM feedback
    - Multi-objective rewards
    - LLM-based action outcome evaluation
    """

    def __init__(
        self,
        agent_id: str,
        config: Optional[RLConfig] = None,
        available_actions: Optional[list[str]] = None,
        learning_rate: float = 0.1,
        exploration_rate: float = 0.1,
    ):
        self.agent_id = agent_id
        self.config = config or RLConfig(
            learning_rate=learning_rate,
            exploration_initial=exploration_rate,
        )
        self.available_actions = available_actions or []

        # LLM provider for reward evaluation
        self._llm_provider = None

        # Experience replay buffer
        self._replay_buffer: deque[Experience] = deque(maxlen=self.config.buffer_size)
        self._priority_sum_tree: list[float] = []

        # Policy and value function
        self._policy = Policy(
            id=f"policy-{agent_id}",
            name=f"Policy for {agent_id}",
            exploration_rate=self.config.exploration_initial,
        )

        self._value_function = ValueFunction(
            id=f"value-{agent_id}",
            learning_rate=self.config.learning_rate,
            discount_factor=self.config.discount_factor,
        )

        # Tracking
        self._experience_counter = 0
        self._update_counter = 0
        self._total_reward = 0.0
        self._episode_rewards: list[float] = []

        self._initialized = False

    async def _get_llm_provider(self):
        """Get or create LLM provider for reward evaluation."""
        if self._llm_provider is None:
            try:
                from aion.systems.agents.llm_integration import SOTALLMProvider
                self._llm_provider = await SOTALLMProvider.get_instance()
            except Exception as e:
                logger.warning("llm_provider_init_failed", error=str(e))
        return self._llm_provider

    async def initialize(self) -> None:
        """Initialize reinforcement learner."""
        self._initialized = True
        # Pre-initialize LLM provider
        await self._get_llm_provider()
        logger.info("reinforcement_learner_initialized", agent_id=self.agent_id)

    async def shutdown(self) -> None:
        """Shutdown reinforcement learner."""
        self._initialized = False
        logger.info("reinforcement_learner_shutdown", agent_id=self.agent_id)

    async def record_experience(
        self,
        state: dict[str, Any],
        action: str,
        reward: float,
        next_state: Optional[dict[str, Any]] = None,
        done: bool = False,
        reward_type: RewardType = RewardType.OUTCOME_BASED,
        action_params: Optional[dict[str, Any]] = None,
        info: Optional[dict[str, Any]] = None,
    ) -> Experience:
        """Record a new experience."""
        self._experience_counter += 1

        # Apply reward processing
        processed_reward = self._process_reward(reward)

        experience = Experience(
            id=f"exp-{self._experience_counter}",
            state=state,
            action=action,
            action_params=action_params or {},
            reward=processed_reward,
            reward_type=reward_type,
            next_state=next_state,
            done=done,
            info=info or {},
        )

        # Calculate priority for prioritized replay
        if self.config.prioritized_replay:
            td_error = self._calculate_td_error(experience)
            experience.priority = (abs(td_error) + 0.01) ** self.config.priority_alpha

        self._replay_buffer.append(experience)
        self._total_reward += processed_reward

        logger.debug(
            "experience_recorded",
            action=action,
            reward=processed_reward,
            done=done,
        )

        # Trigger learning if buffer is sufficient
        if len(self._replay_buffer) >= self.config.batch_size:
            if self._experience_counter % self.config.policy_update_frequency == 0:
                await self._update_policy()

        return experience

    async def record_feedback(
        self,
        experience_id: str,
        feedback: float,  # -1 to 1
        feedback_type: str = "user",
    ) -> None:
        """Record external feedback for an experience."""
        # Find experience and update reward
        for exp in self._replay_buffer:
            if exp.id == experience_id:
                # Blend feedback with original reward
                exp.reward = exp.reward * 0.7 + feedback * 0.3
                exp.reward_type = RewardType.USER_FEEDBACK
                exp.info["feedback"] = feedback
                exp.info["feedback_type"] = feedback_type

                # Recalculate priority
                if self.config.prioritized_replay:
                    td_error = self._calculate_td_error(exp)
                    exp.priority = (abs(td_error) + 0.01) ** self.config.priority_alpha

                break

    async def evaluate_action_with_llm(
        self,
        state: dict[str, Any],
        action: str,
        outcome: dict[str, Any],
    ) -> float:
        """
        Evaluate an action's outcome using Llama 3.3 70B.

        This provides intelligent reward shaping based on LLM understanding
        of the task context and action quality.

        Args:
            state: State before action
            action: Action taken
            outcome: Result of action

        Returns:
            Reward signal from -1.0 to 1.0
        """
        try:
            llm_provider = await self._get_llm_provider()
            if llm_provider:
                reward = await llm_provider.evaluate_action_outcome(state, action, outcome)
                return reward
            return 0.0
        except Exception as e:
            logger.warning("llm_reward_evaluation_error", error=str(e))
            return 0.0

    async def record_experience_with_llm_evaluation(
        self,
        state: dict[str, Any],
        action: str,
        outcome: dict[str, Any],
        next_state: Optional[dict[str, Any]] = None,
        done: bool = False,
    ) -> Experience:
        """
        Record an experience with LLM-based reward evaluation.

        Automatically evaluates the action outcome using Llama 3.3 70B
        to generate intelligent reward signals.

        Args:
            state: State before action
            action: Action taken
            outcome: Result/output of the action
            next_state: State after action
            done: Whether episode ended

        Returns:
            Recorded experience with LLM-evaluated reward
        """
        # Evaluate reward using LLM
        reward = await self.evaluate_action_with_llm(state, action, outcome)

        # Record with evaluated reward
        return await self.record_experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            reward_type=RewardType.SELF_EVALUATION,
            info={"llm_evaluated": True, "outcome": str(outcome)[:200]},
        )

    def select_action(
        self,
        state: dict[str, Any],
        available_actions: Optional[list[str]] = None,
    ) -> tuple[str, float]:
        """
        Select an action using the current policy.

        Returns:
            Tuple of (action, confidence)
        """
        actions = available_actions or self.available_actions

        if not actions:
            raise ValueError("No actions available")

        action = self._policy.select_action(actions, state)

        # Estimate confidence from value function
        state_value = self._value_function.evaluate(state)
        confidence = min(1.0, max(0.0, (state_value + 1) / 2))  # Normalize to [0, 1]

        return action, confidence

    async def _update_policy(self) -> None:
        """Update policy from experiences."""
        if len(self._replay_buffer) < self.config.batch_size:
            return

        # Sample batch
        batch = self._sample_batch()

        # Update value function
        for exp in batch:
            if exp.next_state and not exp.done:
                next_value = self._value_function.evaluate(exp.next_state)
                target = exp.reward + self.config.discount_factor * next_value
            else:
                target = exp.reward

            td_error = self._value_function.update(exp.state, target)

            # Update priority
            if self.config.prioritized_replay:
                exp.priority = (abs(td_error) + 0.01) ** self.config.priority_alpha

        # Update policy preferences
        for exp in batch:
            advantage = exp.reward - self._value_function.evaluate(exp.state)

            # Update action preference
            current_pref = self._policy.action_preferences.get(exp.action, 0.0)
            self._policy.action_preferences[exp.action] = (
                current_pref + self.config.learning_rate * advantage
            )

        # Decay exploration
        self._policy.exploration_rate = max(
            self.config.exploration_final,
            self._policy.exploration_rate * self.config.exploration_decay,
        )

        self._update_counter += 1
        self._policy.updated_at = datetime.now()
        self._policy.version += 1

        logger.debug(
            "policy_updated",
            version=self._policy.version,
            exploration=self._policy.exploration_rate,
        )

    def _sample_batch(self) -> list[Experience]:
        """Sample a batch of experiences."""
        if self.config.prioritized_replay:
            return self._prioritized_sample()
        else:
            return random.sample(list(self._replay_buffer), self.config.batch_size)

    def _prioritized_sample(self) -> list[Experience]:
        """Sample with prioritization."""
        experiences = list(self._replay_buffer)
        priorities = [exp.priority for exp in experiences]
        total_priority = sum(priorities)

        if total_priority == 0:
            return random.sample(experiences, min(self.config.batch_size, len(experiences)))

        probs = [p / total_priority for p in priorities]
        indices = random.choices(
            range(len(experiences)),
            weights=probs,
            k=min(self.config.batch_size, len(experiences)),
        )

        return [experiences[i] for i in indices]

    def _calculate_td_error(self, experience: Experience) -> float:
        """Calculate TD error for an experience."""
        current_value = self._value_function.evaluate(experience.state)

        if experience.next_state and not experience.done:
            next_value = self._value_function.evaluate(experience.next_state)
            target = experience.reward + self.config.discount_factor * next_value
        else:
            target = experience.reward

        return target - current_value

    def _process_reward(self, reward: float) -> float:
        """Process and scale reward."""
        # Scale
        reward = reward * self.config.reward_scaling

        # Clip if configured
        if self.config.reward_clipping:
            min_r, max_r = self.config.reward_clipping
            reward = max(min_r, min(max_r, reward))

        return reward

    def add_context_rule(
        self,
        condition: dict[str, Any],
        action: str,
        bonus: float = 0.5,
    ) -> None:
        """Add a context-based rule to the policy."""
        self._policy.context_rules.append({
            "condition": condition,
            "action": action,
            "bonus": bonus,
        })

    def end_episode(self) -> float:
        """End current episode and return total reward."""
        episode_reward = self._total_reward
        self._episode_rewards.append(episode_reward)
        self._total_reward = 0.0

        # Update policy performance
        if len(self._episode_rewards) >= 10:
            self._policy.performance_score = sum(self._episode_rewards[-10:]) / 10

        return episode_reward

    def get_policy(self) -> Policy:
        """Get the current policy."""
        return self._policy

    def get_value_function(self) -> ValueFunction:
        """Get the current value function."""
        return self._value_function

    def get_stats(self) -> dict[str, Any]:
        """Get learning statistics."""
        return {
            "agent_id": self.agent_id,
            "total_experiences": self._experience_counter,
            "buffer_size": len(self._replay_buffer),
            "policy_updates": self._update_counter,
            "policy_version": self._policy.version,
            "exploration_rate": self._policy.exploration_rate,
            "avg_episode_reward": sum(self._episode_rewards[-10:]) / max(1, len(self._episode_rewards[-10:])),
            "policy_performance": self._policy.performance_score,
            "action_preferences": dict(list(self._policy.action_preferences.items())[:10]),
        }
