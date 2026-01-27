"""
AION Tool Learning Integration

Bridges the RL loop with the tool orchestration system, providing
learned tool selection and performance feedback.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import structlog

from aion.learning.types import ActionType, RewardSource, StateRepresentation

if TYPE_CHECKING:
    from aion.learning.loop import ReinforcementLearningLoop

logger = structlog.get_logger(__name__)


class ToolLearningIntegration:
    """Integrates RL-based tool selection with the tool orchestrator."""

    def __init__(self, rl_loop: "ReinforcementLearningLoop"):
        self._rl = rl_loop

    async def select_best_tool(
        self,
        query: str,
        available_tools: List[str],
        context: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        interaction_id: Optional[str] = None,
    ) -> str:
        """Select the best tool using learned preferences."""
        state = StateRepresentation(
            query_type=context.get("query_type", "general") if context else "general",
            query_complexity=context.get("complexity", 0.5) if context else 0.5,
            available_tools=available_tools,
            user_id=user_id,
        )
        return await self._rl.select_tool(state, available_tools, interaction_id)

    async def record_tool_outcome(
        self,
        interaction_id: str,
        tool_name: str,
        success: bool,
        latency_ms: float = 0.0,
        quality_score: float = 0.0,
    ) -> None:
        """Record the outcome of a tool invocation."""
        metrics: Dict[str, float] = {}
        if latency_ms > 0:
            metrics["latency"] = latency_ms
        if quality_score > 0:
            metrics["quality"] = quality_score

        await self._rl.collect_outcome(
            interaction_id,
            success=success,
            metrics=metrics,
        )

        # Also update the tool bandit directly for fast adaptation
        bandit = self._rl._bandits.get("tools")
        if bandit:
            reward = 1.0 if success else 0.0
            bandit.update(tool_name, reward)

    def get_tool_performance(self) -> Dict[str, Dict[str, float]]:
        """Get per-tool performance statistics from bandits."""
        bandit = self._rl._bandits.get("tools")
        if not bandit:
            return {}

        expected = bandit.get_expected_rewards()
        stats = bandit.get_stats()
        result = {}
        for arm_id, data in stats.items():
            if isinstance(data, dict):
                result[arm_id] = {
                    "expected_reward": expected.get(arm_id, 0.5),
                    "pulls": data.get("pulls", 0),
                    "avg_reward": data.get("avg_reward", 0.0),
                }
        return result

    def get_tool_recommendations(
        self,
        state: StateRepresentation,
        top_k: int = 5,
    ) -> List[tuple]:
        """Get top-k tool recommendations for the given state."""
        rankings = self._rl.get_tool_rankings(state)
        return rankings[:top_k]
