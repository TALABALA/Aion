"""
AION Learning State Persistence

Saves and restores the state of all learning subsystems:
policies, bandits, experience buffer, and experiments.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING

import numpy as np
import structlog

from aion.learning.config import PersistenceConfig

if TYPE_CHECKING:
    from aion.learning.loop import ReinforcementLearningLoop

logger = structlog.get_logger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


class LearningStateRepository:
    """Persists and restores learning loop state."""

    def __init__(
        self,
        rl_loop: "ReinforcementLearningLoop",
        config: Optional[PersistenceConfig] = None,
    ):
        self._rl = rl_loop
        self._config = config or PersistenceConfig()
        self._checkpoint_dir = Path(self._config.checkpoint_dir)

    async def save_checkpoint(self, name: Optional[str] = None) -> str:
        """Save a checkpoint of all learning state."""
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_name = name or f"checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        checkpoint_path = self._checkpoint_dir / f"{checkpoint_name}.json"

        state: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "name": checkpoint_name,
        }

        # Save policy states
        if self._config.save_policies:
            policies = {}
            for at, policy in self._rl.policy_optimizer._policies.items():
                policies[at.value] = policy.get_state()
            state["policies"] = policies

        # Save bandit states
        if self._config.save_bandits:
            bandits = {}
            for name_key, bandit in self._rl._bandits.items():
                bandits[name_key] = bandit.get_stats()
            state["bandits"] = bandits

        # Save experiment states
        if self._config.save_experiments:
            state["experiments"] = self._rl.ab_testing.get_stats()

        # Save buffer stats (not full buffer — too large)
        state["buffer_stats"] = self._rl.experience_buffer.get_stats()

        # Save RL loop stats
        state["rl_stats"] = self._rl.get_stats()

        with open(checkpoint_path, "w") as f:
            json.dump(state, f, cls=NumpyEncoder, indent=2)

        # Prune old checkpoints
        await self._prune_checkpoints()

        logger.info("checkpoint_saved", path=str(checkpoint_path))
        return str(checkpoint_path)

    async def load_checkpoint(self, path: str) -> bool:
        """Load a checkpoint (partial restore — statistics and configs)."""
        try:
            with open(path) as f:
                state = json.load(f)

            # Restore policy exploration rates
            if "policies" in state:
                for at_str, policy_state in state["policies"].items():
                    from aion.learning.types import ActionType
                    try:
                        at = ActionType(at_str)
                        policy = self._rl.policy_optimizer.get_policy(at)
                        if policy and "exploration_rate" in policy_state:
                            policy.config.exploration_rate = policy_state["exploration_rate"]
                    except (ValueError, KeyError):
                        pass

            logger.info("checkpoint_loaded", path=path)
            return True

        except Exception as e:
            logger.error("checkpoint_load_failed", path=path, error=str(e))
            return False

    async def list_checkpoints(self) -> list:
        """List available checkpoints."""
        if not self._checkpoint_dir.exists():
            return []
        checkpoints = sorted(self._checkpoint_dir.glob("*.json"), key=os.path.getmtime, reverse=True)
        return [str(p) for p in checkpoints]

    async def _prune_checkpoints(self) -> None:
        """Remove old checkpoints beyond the max count."""
        checkpoints = await self.list_checkpoints()
        if len(checkpoints) > self._config.max_checkpoints:
            for old in checkpoints[self._config.max_checkpoints:]:
                try:
                    os.remove(old)
                except OSError:
                    pass
