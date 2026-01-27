"""
AION Policy Optimizer

Coordinates training of multiple policies from the shared experience
buffer.  Runs an async background training loop that samples batches,
dispatches to per-ActionType policies, and updates priorities.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np
import structlog

from aion.learning.config import PolicyOptimizerConfig
from aion.learning.types import Action, ActionType, Experience, StateRepresentation
from aion.learning.experience.buffer import ExperienceBuffer
from aion.learning.policies.base import BasePolicy

if TYPE_CHECKING:
    from aion.core.kernel import AIONKernel

logger = structlog.get_logger(__name__)


class PolicyOptimizer:
    """Optimises multiple policies based on shared experience."""

    def __init__(
        self,
        kernel: "AIONKernel",
        buffer: ExperienceBuffer,
        config: Optional[PolicyOptimizerConfig] = None,
    ):
        self.kernel = kernel
        self.buffer = buffer
        self._config = config or PolicyOptimizerConfig()
        self._policies: Dict[ActionType, BasePolicy] = {}
        self._training = False
        self._training_task: Optional[asyncio.Task] = None
        self._stats: Dict[str, Any] = {
            "training_steps": 0,
            "total_updates": 0,
            "total_loss": 0.0,
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
        logger.info("policy_training_started")

    async def stop_training(self) -> None:
        self._training = False
        if self._training_task:
            self._training_task.cancel()
            try:
                await self._training_task
            except asyncio.CancelledError:
                pass
        logger.info("policy_training_stopped")

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
        """Run a single training step (multiple mini-batch updates)."""
        for _ in range(self._config.updates_per_interval):
            experiences, weights, indices = self.buffer.sample(
                self._config.batch_size
            )
            if not experiences:
                break

            # Group experiences by action type
            by_type: Dict[ActionType, List[tuple]] = {}
            for exp, weight, idx in zip(experiences, weights, indices):
                at = exp.action.action_type
                by_type.setdefault(at, []).append((exp, weight, idx))

            td_errors = np.zeros(len(indices))

            for action_type, items in by_type.items():
                policy = self._policies.get(action_type)
                if not policy:
                    continue

                exps = [i[0] for i in items]
                ws = np.array([i[1] for i in items], dtype=np.float32)
                metrics = await policy.update(exps, ws)
                self._stats["total_loss"] += metrics.get("loss", 0.0)

                if "td_errors" in metrics:
                    for (_, _, idx), td_err in zip(items, metrics["td_errors"]):
                        pos = indices.index(idx) if idx in indices else -1
                        if pos >= 0:
                            td_errors[pos] = td_err

            self.buffer.update_priorities(indices, td_errors)
            self._stats["training_steps"] += 1

        # Decay exploration for all policies
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
            "policies": {
                at.value: {
                    "exploration_rate": p.config.exploration_rate,
                    "version": p.config.version,
                    "updates": p._update_count,
                }
                for at, p in self._policies.items()
            },
        }
