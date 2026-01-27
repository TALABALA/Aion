"""
AION Tool Selection Policy

Learns to select the best tool for a given context using contextual
bandit-style linear models with softmax action selection.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from aion.learning.types import Experience, PolicyConfig, StateRepresentation
from aion.learning.policies.base import BasePolicy


class ToolSelectionPolicy(BasePolicy):
    """Policy for tool selection using contextual linear models."""

    def __init__(self, config: PolicyConfig):
        super().__init__(config)
        self._weights: Dict[str, np.ndarray] = {}
        self._bias: Dict[str, float] = {}
        self._tool_stats: Dict[str, Dict[str, float]] = {}
        self._recent_rewards: List[float] = []

    async def select_action(
        self,
        state: StateRepresentation,
        available_actions: List[str],
    ) -> Tuple[str, float]:
        if not available_actions:
            return "", 0.0

        scores: Dict[str, float] = {}
        features = state.to_vector()

        for tool in available_actions:
            if tool in self._weights:
                score = float(np.dot(self._weights[tool], features)) + self._bias.get(tool, 0.0)
            else:
                stats = self._tool_stats.get(tool, {})
                score = stats.get("avg_reward", 0.0)
            scores[tool] = score

        # Softmax with temperature
        vals = np.array(list(scores.values()))
        temp = max(0.01, self.config.exploration_rate)
        exp_vals = np.exp((vals - np.max(vals)) / temp)
        probs = exp_vals / exp_vals.sum()

        best_idx = int(np.argmax(probs))
        return available_actions[best_idx], float(probs[best_idx])

    async def update(
        self,
        experiences: List[Experience],
        weights: np.ndarray,
    ) -> Dict[str, Any]:
        td_errors: List[float] = []
        total_loss = 0.0

        for exp, w in zip(experiences, weights):
            tool = exp.action.choice
            reward = exp.cumulative_reward
            features = exp.state.to_vector()

            # Lazy initialisation of per-tool weights
            if tool not in self._weights:
                self._weights[tool] = np.zeros(len(features), dtype=np.float32)
                self._bias[tool] = 0.0

            predicted = float(np.dot(self._weights[tool], features)) + self._bias[tool]
            td_error = reward - predicted
            td_errors.append(td_error)

            # Gradient step with importance weight
            lr = self.config.learning_rate * float(w)

            # Gradient clipping
            grad = td_error * features
            grad_norm = np.linalg.norm(grad)
            if grad_norm > self.config.gradient_clip:
                grad = grad * (self.config.gradient_clip / grad_norm)

            self._weights[tool] += lr * grad
            self._bias[tool] += lr * td_error
            total_loss += td_error ** 2

            # Update running statistics
            self._tool_stats.setdefault(tool, {"total_reward": 0.0, "count": 0, "avg_reward": 0.0})
            stats = self._tool_stats[tool]
            stats["total_reward"] += reward
            stats["count"] += 1
            stats["avg_reward"] = stats["total_reward"] / stats["count"]

        self._recent_rewards.extend([e.cumulative_reward for e in experiences])
        self._recent_rewards = self._recent_rewards[-200:]
        self._update_count += 1

        return {
            "loss": total_loss / max(len(experiences), 1),
            "td_errors": td_errors,
            "avg_reward": float(np.mean(self._recent_rewards)) if self._recent_rewards else 0.0,
        }

    def get_tool_rankings(self, state: StateRepresentation) -> List[Tuple[str, float]]:
        """Return tools ranked by predicted value for the given state."""
        features = state.to_vector()
        scores = []
        for tool, w in self._weights.items():
            s = float(np.dot(w, features)) + self._bias.get(tool, 0.0)
            scores.append((tool, s))
        return sorted(scores, key=lambda x: x[1], reverse=True)

    def get_state(self) -> Dict[str, Any]:
        base = super().get_state()
        base["tool_stats"] = dict(self._tool_stats)
        base["num_tools_learned"] = len(self._weights)
        return base
