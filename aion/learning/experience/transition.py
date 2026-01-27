"""
AION Transition Builders

Constructs experience transitions including multi-step (n-step) returns
for variance reduction and faster credit assignment.
"""

from __future__ import annotations

from collections import deque
from typing import List, Optional

from aion.learning.types import Action, Experience, RewardSignal, StateRepresentation


class TransitionBuilder:
    """Builds single-step experience transitions."""

    @staticmethod
    def build(
        state: StateRepresentation,
        action: Action,
        reward: float,
        next_state: Optional[StateRepresentation] = None,
        done: bool = False,
        interaction_id: str = "",
        rewards: Optional[List[RewardSignal]] = None,
    ) -> Experience:
        exp = Experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            interaction_id=interaction_id,
            rewards=rewards or [],
        )
        exp.compute_cumulative_reward()
        return exp


class NStepTransitionBuilder:
    """
    Builds n-step return transitions for faster credit assignment.

    Accumulates rewards over n steps:  R = r_t + gamma*r_{t+1} + ... + gamma^{n-1}*r_{t+n-1}
    and pairs with the state n steps ahead.
    """

    def __init__(self, n: int = 3, gamma: float = 0.99):
        self.n = n
        self.gamma = gamma
        self._buffer: deque[Experience] = deque(maxlen=n)

    def add(self, experience: Experience) -> Optional[Experience]:
        """Add a transition; returns a completed n-step transition when ready."""
        self._buffer.append(experience)

        if len(self._buffer) < self.n and not experience.done:
            return None

        return self._build_nstep()

    def flush(self) -> List[Experience]:
        """Flush remaining transitions at episode end."""
        results = []
        while self._buffer:
            results.append(self._build_nstep())
        return results

    def _build_nstep(self) -> Experience:
        """Construct the n-step transition from the buffer."""
        first = self._buffer[0]
        n_step_reward = 0.0
        for i, exp in enumerate(self._buffer):
            n_step_reward += (self.gamma ** i) * exp.reward

        last = self._buffer[-1]
        nstep_exp = Experience(
            state=first.state,
            action=first.action,
            reward=n_step_reward,
            next_state=last.next_state,
            done=last.done,
            interaction_id=first.interaction_id,
            rewards=first.rewards,
            episode_id=first.episode_id,
            step_index=first.step_index,
        )
        nstep_exp.cumulative_reward = n_step_reward
        self._buffer.popleft()
        return nstep_exp
