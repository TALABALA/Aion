"""
AION Agent Behavior Policy

Learns to select the best agent archetype or behavioral mode for
a given task, optimising for task success and user satisfaction.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from aion.learning.types import Experience, PolicyConfig, StateRepresentation
from aion.learning.policies.base import BasePolicy


class AgentBehaviorPolicy(BasePolicy):
    """Policy for agent behavior / archetype selection."""

    def __init__(self, config: PolicyConfig):
        super().__init__(config)
        self._behavior_values: Dict[str, float] = {}
        self._behavior_counts: Dict[str, int] = {}
        self._weights: Dict[str, np.ndarray] = {}
        self._bias: Dict[str, float] = {}
        self._reward_history: Dict[str, List[float]] = {}

    async def select_action(
        self,
        state: StateRepresentation,
        available_actions: List[str],
    ) -> Tuple[str, float]:
        if not available_actions:
            return "", 0.0

        features = state.to_vector()
        scores: Dict[str, float] = {}

        for behavior in available_actions:
            if behavior in self._weights:
                scores[behavior] = (
                    float(np.dot(self._weights[behavior], features))
                    + self._bias.get(behavior, 0.0)
                )
            else:
                scores[behavior] = self._behavior_values.get(behavior, 0.0)

        vals = np.array(list(scores.values()))
        exp_vals = np.exp(vals - np.max(vals))
        probs = exp_vals / exp_vals.sum()

        # Entropy-regularised selection
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        probs_with_entropy = probs + self.config.entropy_coefficient * (1.0 / len(probs) - probs)
        probs_with_entropy = np.clip(probs_with_entropy, 0, None)
        probs_with_entropy /= probs_with_entropy.sum()

        best_idx = int(np.argmax(probs_with_entropy))
        return available_actions[best_idx], float(probs_with_entropy[best_idx])

    async def update(
        self,
        experiences: List[Experience],
        weights: np.ndarray,
    ) -> Dict[str, Any]:
        td_errors: List[float] = []
        total_loss = 0.0

        for exp, w in zip(experiences, weights):
            behavior = exp.action.choice
            reward = exp.cumulative_reward
            features = exp.state.to_vector()

            if behavior not in self._weights:
                self._weights[behavior] = np.zeros(len(features), dtype=np.float32)
                self._bias[behavior] = 0.0

            predicted = float(np.dot(self._weights[behavior], features)) + self._bias[behavior]
            td_error = reward - predicted
            td_errors.append(td_error)

            lr = self.config.learning_rate * float(w)
            grad = td_error * features
            grad_norm = np.linalg.norm(grad)
            if grad_norm > self.config.gradient_clip:
                grad = grad * (self.config.gradient_clip / grad_norm)

            self._weights[behavior] += lr * grad
            self._bias[behavior] += lr * td_error
            total_loss += td_error ** 2

            # Track per-behavior reward history
            self._reward_history.setdefault(behavior, []).append(reward)
            if len(self._reward_history[behavior]) > 200:
                self._reward_history[behavior] = self._reward_history[behavior][-100:]

            self._behavior_counts[behavior] = self._behavior_counts.get(behavior, 0) + 1
            n = self._behavior_counts[behavior]
            old = self._behavior_values.get(behavior, 0.0)
            self._behavior_values[behavior] = old + (reward - old) / n

        self._update_count += 1
        return {
            "loss": total_loss / max(len(experiences), 1),
            "td_errors": td_errors,
        }

    def get_behavior_stats(self) -> Dict[str, Dict[str, float]]:
        """Return per-behavior statistics."""
        stats = {}
        for b, vals in self._reward_history.items():
            stats[b] = {
                "avg_reward": float(np.mean(vals)) if vals else 0.0,
                "count": self._behavior_counts.get(b, 0),
                "recent_trend": float(np.mean(vals[-20:])) - float(np.mean(vals[:20])) if len(vals) > 20 else 0.0,
            }
        return stats
