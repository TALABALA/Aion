"""
AION Planning Strategy Policy (Actor-Critic)

Learns to select the best planning strategy (e.g. greedy, MCTS,
hierarchical, reactive) based on query characteristics.

Uses advantage-based policy gradients with UCB exploration bonus.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from aion.learning.types import Experience, PolicyConfig, StateRepresentation
from aion.learning.policies.base import BasePolicy


class PlanningStrategyPolicy(BasePolicy):
    """Actor policy for planning strategy selection."""

    def __init__(self, config: PolicyConfig):
        super().__init__(config)
        self._strategy_values: Dict[str, float] = {}
        self._strategy_counts: Dict[str, int] = {}
        self._weights: Dict[str, np.ndarray] = {}
        self._bias: Dict[str, float] = {}

    async def select_action(
        self,
        state: StateRepresentation,
        available_actions: List[str],
    ) -> Tuple[str, float]:
        if not available_actions:
            return "", 0.0

        features = state.to_vector()
        scores: Dict[str, float] = {}

        for strategy in available_actions:
            if strategy in self._weights:
                scores[strategy] = float(np.dot(self._weights[strategy], features)) + self._bias.get(strategy, 0.0)
            else:
                scores[strategy] = self._strategy_values.get(strategy, 0.0)

        # UCB-style bonus for under-explored strategies
        total_count = sum(self._strategy_counts.values()) + 1
        for s in available_actions:
            count = self._strategy_counts.get(s, 0) + 1
            ucb_bonus = np.sqrt(2 * np.log(total_count) / count)
            scores[s] += 0.1 * ucb_bonus

        vals = np.array(list(scores.values()))
        exp_vals = np.exp(vals - np.max(vals))
        probs = exp_vals / exp_vals.sum()

        idx = int(np.random.choice(len(available_actions), p=probs))
        return available_actions[idx], float(probs[idx])

    async def update(
        self,
        experiences: List[Experience],
        weights: np.ndarray,
        advantages: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Advantage-based policy gradient update."""
        td_errors: List[float] = []
        total_loss = 0.0

        for i, (exp, w) in enumerate(zip(experiences, weights)):
            strategy = exp.action.choice
            reward = exp.cumulative_reward
            features = exp.state.to_vector()

            if strategy not in self._weights:
                self._weights[strategy] = np.zeros(len(features), dtype=np.float32)
                self._bias[strategy] = 0.0

            # Use advantage if available, else fall back to TD residual
            if advantages is not None and i < len(advantages):
                advantage = float(advantages[i])
            else:
                predicted = float(np.dot(self._weights[strategy], features)) + self._bias[strategy]
                advantage = reward - predicted

            td_errors.append(advantage)

            lr = self.config.learning_rate * float(w)
            grad = advantage * features
            grad_norm = np.linalg.norm(grad)
            if grad_norm > self.config.gradient_clip:
                grad = grad * (self.config.gradient_clip / grad_norm)

            self._weights[strategy] += lr * grad
            self._bias[strategy] += lr * advantage
            total_loss += advantage ** 2

            # Running average
            self._strategy_counts[strategy] = self._strategy_counts.get(strategy, 0) + 1
            n = self._strategy_counts[strategy]
            old_avg = self._strategy_values.get(strategy, 0.0)
            self._strategy_values[strategy] = old_avg + (reward - old_avg) / n

        self._update_count += 1
        return {
            "loss": total_loss / max(len(experiences), 1),
            "td_errors": td_errors,
        }
