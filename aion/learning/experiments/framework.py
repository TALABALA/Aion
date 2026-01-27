"""
AION A/B Testing Framework

Integrates experiment management with statistical analysis and
sequential testing for efficient policy experiments.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np
import structlog

from aion.learning.config import ExperimentConfig
from aion.learning.types import (
    ActionType,
    Experiment,
    ExperimentStatus,
    ExperimentVariant,
)
from aion.learning.experiments.experiment import ExperimentManager
from aion.learning.experiments.analysis import StatisticalAnalyzer, SequentialTester

if TYPE_CHECKING:
    from aion.core.kernel import AIONKernel

logger = structlog.get_logger(__name__)


class ABTestingFramework:
    """A/B testing framework for policy experiments."""

    def __init__(
        self,
        kernel: "AIONKernel",
        config: Optional[ExperimentConfig] = None,
    ):
        self.kernel = kernel
        self._config = config or ExperimentConfig()
        self._manager = ExperimentManager()
        self._analyzer = StatisticalAnalyzer()
        self._assignments: Dict[str, Dict[str, str]] = {}
        self._sequential_testers: Dict[str, SequentialTester] = {}

    # ------------------------------------------------------------------
    # Experiment lifecycle
    # ------------------------------------------------------------------

    async def create_experiment(
        self,
        name: str,
        action_type: ActionType,
        control_config: Dict[str, Any],
        treatment_config: Dict[str, Any],
        hypothesis: str = "",
        traffic_split: float = 0.5,
    ) -> Experiment:
        exp = self._manager.create(
            name=name,
            action_type=action_type,
            control_config=control_config,
            treatment_config=treatment_config,
            hypothesis=hypothesis,
            traffic_split=traffic_split,
        )
        if self._config.sequential_testing_enabled:
            self._sequential_testers[exp.id] = SequentialTester(
                total_alpha=self._config.significance_level,
                spending_function=self._config.alpha_spending_function,
            )
        return exp

    async def start_experiment(self, experiment_id: str) -> bool:
        return self._manager.start(experiment_id)

    async def stop_experiment(self, experiment_id: str) -> bool:
        exp = self._manager.get(experiment_id)
        if not exp:
            return False
        self._manager.complete(experiment_id)
        await self._analyze_experiment(exp)
        return True

    # ------------------------------------------------------------------
    # Variant assignment (deterministic hashing)
    # ------------------------------------------------------------------

    def get_variant(self, experiment_id: str, user_id: str) -> Optional[ExperimentVariant]:
        exp = self._manager.get(experiment_id)
        if not exp or exp.status != ExperimentStatus.RUNNING:
            return None

        cache = self._assignments.setdefault(experiment_id, {})
        if user_id in cache:
            vid = cache[user_id]
            return exp.control if vid == exp.control.id else exp.treatment

        # Deterministic bucketing
        hash_input = f"{experiment_id}:{user_id}".encode()
        bucket = (int(hashlib.sha256(hash_input).hexdigest(), 16) % 10000) / 10000.0
        variant = exp.treatment if bucket < exp.treatment.traffic_percentage else exp.control
        cache[user_id] = variant.id
        return variant

    # ------------------------------------------------------------------
    # Result recording
    # ------------------------------------------------------------------

    def get_active_experiments(self, action_type: Optional[ActionType] = None) -> List[Experiment]:
        return self._manager.get_active(action_type)

    async def record_result(
        self,
        experiment_id: str,
        variant_id: str,
        reward: float,
    ) -> None:
        exp = self._manager.get(experiment_id)
        if not exp or exp.status != ExperimentStatus.RUNNING:
            return

        variant = (
            exp.control if variant_id == exp.control.id
            else exp.treatment if variant_id == exp.treatment.id
            else None
        )
        if not variant:
            return

        variant.record(reward)

        # Check auto-stop
        if self._config.auto_stop_enabled:
            await self._check_auto_stop(exp)

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    async def _analyze_experiment(self, exp: Experiment) -> None:
        c, t = exp.control, exp.treatment
        if len(c.reward_samples) < 2 or len(t.reward_samples) < 2:
            return

        result = self._analyzer.welch_t_test(
            c.reward_samples, t.reward_samples, self._config.significance_level,
        )
        exp.p_value = result.p_value
        exp.effect_size = result.effect_size
        exp.confidence_interval = result.confidence_interval

        if result.significant:
            exp.winner = t.id if t.avg_reward > c.avg_reward else c.id

    async def _check_auto_stop(self, exp: Experiment) -> None:
        min_n = self._config.min_samples_per_variant
        c, t = exp.control, exp.treatment
        if c.sample_count < min_n or t.sample_count < min_n:
            return

        if len(c.reward_samples) < 2 or len(t.reward_samples) < 2:
            return

        result = self._analyzer.welch_t_test(
            c.reward_samples, t.reward_samples, self._config.significance_level,
        )

        # Sequential testing boundary check
        tester = self._sequential_testers.get(exp.id)
        if tester:
            if tester.should_stop(result.p_value):
                exp.p_value = result.p_value
                exp.effect_size = result.effect_size
                exp.winner = t.id if t.avg_reward > c.avg_reward else c.id
                await self.stop_experiment(exp.id)
        elif (
            result.significant
            and abs(result.effect_size) > self._config.min_effect_size
        ):
            await self.stop_experiment(exp.id)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        all_exps = self._manager.list_all()
        return {
            "total_experiments": len(all_exps),
            "active": len(self._manager.get_active()),
            "completed": sum(1 for e in all_exps if e.status == ExperimentStatus.COMPLETED),
            "experiments": {
                e.id: {
                    "name": e.name,
                    "status": e.status.value,
                    "control_samples": e.control.sample_count,
                    "treatment_samples": e.treatment.sample_count,
                    "p_value": e.p_value,
                    "winner": e.winner,
                }
                for e in all_exps
            },
        }
