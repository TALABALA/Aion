"""
AION Evolution Integration

Bridges the RL learning loop with the self-improvement engine,
feeding learning insights back into the evolution system.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

import structlog

from aion.learning.types import ActionType, StateRepresentation

if TYPE_CHECKING:
    from aion.learning.loop import ReinforcementLearningLoop

logger = structlog.get_logger(__name__)


class EvolutionIntegration:
    """Integrates RL insights with the self-improvement engine."""

    def __init__(self, rl_loop: "ReinforcementLearningLoop"):
        self._rl = rl_loop

    async def get_improvement_candidates(self) -> List[Dict[str, Any]]:
        """
        Identify areas where the system is underperforming and
        suggest improvements for the evolution engine.
        """
        candidates = []
        stats = self._rl.get_stats()
        buffer_stats = stats.get("experience_buffer", {})

        # Check if average reward is declining
        avg_reward = buffer_stats.get("avg_reward", 0.0)
        if avg_reward < -0.2:
            candidates.append({
                "area": "overall_performance",
                "severity": "high",
                "avg_reward": avg_reward,
                "suggestion": "Overall reward is negative; review recent policy changes",
            })

        # Check per-policy performance
        policy_stats = stats.get("policy_optimizer", {}).get("policies", {})
        for action_type, p_stats in policy_stats.items():
            exploration = p_stats.get("exploration_rate", 0)
            if exploration <= 0.02:
                candidates.append({
                    "area": f"policy_{action_type}",
                    "severity": "medium",
                    "suggestion": f"Policy {action_type} has very low exploration; may be stuck in local optimum",
                })

        # Check bandit underperformers
        for bandit_name, bandit_stats in stats.get("bandits", {}).items():
            for arm_id, arm_data in bandit_stats.items():
                if isinstance(arm_data, dict) and arm_data.get("pulls", 0) > 50:
                    expected = arm_data.get("expected", 0.5)
                    if expected < 0.3:
                        candidates.append({
                            "area": f"bandit_{bandit_name}",
                            "arm": arm_id,
                            "severity": "low",
                            "expected_reward": expected,
                            "suggestion": f"Arm {arm_id} in {bandit_name} consistently underperforms",
                        })

        return candidates

    async def apply_evolution_result(
        self,
        action_type_str: str,
        new_config: Dict[str, Any],
    ) -> bool:
        """
        Apply a configuration change suggested by the evolution engine.

        Creates an A/B experiment to validate the change before
        committing to it.
        """
        try:
            action_type = ActionType(action_type_str)
        except ValueError:
            logger.warning("invalid_action_type", action_type=action_type_str)
            return False

        # Create an experiment to validate the change
        current_policy = self._rl.policy_optimizer.get_policy(action_type)
        if not current_policy:
            return False

        control_config = {
            "learning_rate": current_policy.config.learning_rate,
            "exploration_rate": current_policy.config.exploration_rate,
        }

        experiment = await self._rl.ab_testing.create_experiment(
            name=f"evolution_{action_type_str}",
            action_type=action_type,
            control_config=control_config,
            treatment_config=new_config,
            hypothesis=f"Evolution-suggested config improves {action_type_str}",
            traffic_split=0.2,  # Conservative: only 20% traffic
        )
        await self._rl.ab_testing.start_experiment(experiment.id)
        logger.info("evolution_experiment_created", experiment_id=experiment.id)
        return True
