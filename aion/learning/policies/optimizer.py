"""
AION Policy Optimizer (Actor-Critic)

Coordinates training of the full actor-critic pipeline:
  1. Sample batch from prioritized experience buffer
  2. Compute TD advantages via shared StateValueFunction (critic)
  3. Update critic (value function) with TD(0) loss
  4. Update actor policies with advantage-based policy gradients
  5. Update experience buffer priorities from TD errors
  6. Train RND predictor on visited states (intrinsic motivation)
  7. Polyak-average the target network (automatic in value function)
  8. Decay exploration rates

Runs an async background training loop that does not block the
main interaction pipeline.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np
import structlog

from aion.learning.config import PolicyOptimizerConfig, ValueFunctionConfig, RNDConfig
from aion.learning.types import Action, ActionType, Experience, StateRepresentation
from aion.learning.experience.buffer import ExperienceBuffer
from aion.learning.policies.base import BasePolicy
from aion.learning.policies.value_function import StateValueFunction
from aion.learning.rewards.rnd import RNDCuriosityShaper

if TYPE_CHECKING:
    from aion.core.kernel import AIONKernel

logger = structlog.get_logger(__name__)


class PolicyOptimizer:
    """Actor-critic optimizer with shared value function, target network, and RND."""

    def __init__(
        self,
        kernel: "AIONKernel",
        buffer: ExperienceBuffer,
        config: Optional[PolicyOptimizerConfig] = None,
        value_config: Optional[ValueFunctionConfig] = None,
        rnd_config: Optional[RNDConfig] = None,
    ):
        self.kernel = kernel
        self.buffer = buffer
        self._config = config or PolicyOptimizerConfig()
        self._policies: Dict[ActionType, BasePolicy] = {}

        # Shared critic: V(s) with target network
        self.value_function = StateValueFunction(config=value_config or ValueFunctionConfig())

        # RND curiosity module
        self.rnd = RNDCuriosityShaper(config=rnd_config)

        self._training = False
        self._training_task: Optional[asyncio.Task] = None
        self._stats: Dict[str, Any] = {
            "training_steps": 0,
            "total_updates": 0,
            "total_actor_loss": 0.0,
            "total_critic_loss": 0.0,
            "total_rnd_loss": 0.0,
        }

    # ------------------------------------------------------------------
    # Policy management
    # ------------------------------------------------------------------

    def register_policy(self, action_type: ActionType, policy: BasePolicy) -> None:
        self._policies[action_type] = policy
        logger.info("policy_registered", action_type=action_type.value, policy=policy.config.name)

    def get_policy(self, action_type: ActionType) -> Optional[BasePolicy]:
        return self._policies.get(action_type)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    async def select_action(
        self,
        action_type: ActionType,
        state: StateRepresentation,
        available_actions: List[str],
    ) -> Action:
        """Select an action using the registered policy (or random fallback)."""
        policy = self._policies.get(action_type)

        if not policy or not available_actions:
            choice = np.random.choice(available_actions) if available_actions else ""
            return Action(
                action_type=action_type,
                choice=choice,
                alternatives=list(available_actions),
                exploration=True,
            )

        exploring = policy.should_explore()
        if exploring:
            choice = str(np.random.choice(available_actions))
            confidence = 0.0
        else:
            choice, confidence = await policy.select_action(state, available_actions)

        return Action(
            action_type=action_type,
            choice=choice,
            alternatives=list(available_actions),
            confidence=confidence,
            exploration=exploring,
            policy_version=policy.config.version,
            state_id=state.id,
        )

    # ------------------------------------------------------------------
    # Background training loop
    # ------------------------------------------------------------------

    async def start_training(self) -> None:
        if self._training:
            return
        self._training = True
        self._training_task = asyncio.create_task(self._training_loop())
        logger.info("actor_critic_training_started")

    async def stop_training(self) -> None:
        self._training = False
        if self._training_task:
            self._training_task.cancel()
            try:
                await self._training_task
            except asyncio.CancelledError:
                pass
        logger.info("actor_critic_training_stopped")

    async def _training_loop(self) -> None:
        while self._training:
            try:
                if self.buffer.is_ready():
                    await self._train_step()
                await asyncio.sleep(self._config.training_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("training_loop_error")
                await asyncio.sleep(self._config.training_interval_seconds)

    async def _train_step(self) -> None:
        """Run a single actor-critic training step.

        Pipeline:
          1. Sample prioritized batch
          2. Update critic (value function) with TD(0) — produces TD errors
          3. Compute normalised advantages from TD errors
          4. Update actor policies with advantage-based gradients
          5. Train RND predictor on observed states
          6. Update buffer priorities from critic TD errors
        """
        for _ in range(self._config.updates_per_interval):
            experiences, weights, indices = self.buffer.sample(
                self._config.batch_size
            )
            if not experiences:
                break

            # ----------------------------------------------------------
            # Step 1: Extract state features into arrays
            # ----------------------------------------------------------
            states = np.array([e.state.to_vector() for e in experiences])
            rewards = np.array([e.cumulative_reward for e in experiences])
            next_states = np.array([
                e.next_state.to_vector() if e.next_state else e.state.to_vector()
                for e in experiences
            ])
            dones = np.array([1.0 if e.done else 0.0 for e in experiences])

            # ----------------------------------------------------------
            # Step 2: Update critic (value function) with TD(0)
            # ----------------------------------------------------------
            critic_metrics = self.value_function.update(
                states, rewards, next_states, dones, weights,
            )
            td_errors = np.array(critic_metrics["td_errors"])
            self._stats["total_critic_loss"] += critic_metrics["value_loss"]

            # ----------------------------------------------------------
            # Step 3: Normalise advantages for stable actor gradients
            # ----------------------------------------------------------
            advantages = td_errors.copy()
            adv_std = advantages.std()
            if adv_std > 1e-8:
                advantages = (advantages - advantages.mean()) / adv_std

            # ----------------------------------------------------------
            # Step 4: Update actor policies with advantages
            # ----------------------------------------------------------
            by_type: Dict[ActionType, List[tuple]] = {}
            for idx_in_batch, (exp, weight, buf_idx) in enumerate(
                zip(experiences, weights, indices)
            ):
                at = exp.action.action_type
                by_type.setdefault(at, []).append((exp, weight, buf_idx, idx_in_batch))

            for action_type, items in by_type.items():
                policy = self._policies.get(action_type)
                if not policy:
                    continue

                exps = [i[0] for i in items]
                ws = np.array([i[1] for i in items], dtype=np.float32)
                batch_indices = [i[3] for i in items]
                advs = advantages[batch_indices]

                metrics = await policy.update(exps, ws, advantages=advs)
                self._stats["total_actor_loss"] += metrics.get("loss", 0.0)

            # ----------------------------------------------------------
            # Step 5: Train RND predictor on observed states
            # ----------------------------------------------------------
            rnd_metrics = self.rnd.train_predictor(states)
            self._stats["total_rnd_loss"] += rnd_metrics.get("rnd_loss", 0.0)

            # ----------------------------------------------------------
            # Step 6: Update buffer priorities from critic TD errors
            # ----------------------------------------------------------
            self.buffer.update_priorities(indices, td_errors)
            self._stats["training_steps"] += 1

        # Decay exploration for all actor policies
        for policy in self._policies.values():
            policy.decay_exploration()

        self._stats["total_updates"] += 1

    # ------------------------------------------------------------------
    # Manual training trigger
    # ------------------------------------------------------------------

    async def train_on_batch(
        self,
        experiences: List[Experience],
    ) -> Dict[str, Any]:
        """Manually train on a specific batch (useful for integration)."""
        all_metrics: Dict[str, Any] = {}
        by_type: Dict[ActionType, List[Experience]] = {}
        for exp in experiences:
            by_type.setdefault(exp.action.action_type, []).append(exp)

        for action_type, exps in by_type.items():
            policy = self._policies.get(action_type)
            if policy:
                weights = np.ones(len(exps), dtype=np.float32)
                metrics = await policy.update(exps, weights)
                all_metrics[action_type.value] = metrics

        return all_metrics

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        return {
            **self._stats,
            "value_function": self.value_function.get_stats(),
            "rnd": self.rnd.get_stats(),
            "policies": {
                at.value: {
                    "exploration_rate": p.config.exploration_rate,
                    "version": p.config.version,
                    "updates": p._update_count,
                }
                for at, p in self._policies.items()
            },
        }
