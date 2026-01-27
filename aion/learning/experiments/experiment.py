"""
AION Experiment Manager

Manages the lifecycle of experiments: creation, assignment, recording,
and analysis.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import structlog

from aion.learning.types import (
    ActionType,
    Experiment,
    ExperimentStatus,
    ExperimentVariant,
)

logger = structlog.get_logger(__name__)


class ExperimentManager:
    """Manages the full lifecycle of experiments."""

    def __init__(self) -> None:
        self._experiments: Dict[str, Experiment] = {}

    def create(
        self,
        name: str,
        action_type: ActionType,
        control_config: Dict[str, Any],
        treatment_config: Dict[str, Any],
        hypothesis: str = "",
        traffic_split: float = 0.5,
        min_samples: int = 1000,
    ) -> Experiment:
        exp = Experiment(
            name=name,
            action_type=action_type,
            hypothesis=hypothesis,
            min_samples=min_samples,
            control=ExperimentVariant(
                name="control",
                policy_config=control_config,
                traffic_percentage=1 - traffic_split,
            ),
            treatment=ExperimentVariant(
                name="treatment",
                policy_config=treatment_config,
                traffic_percentage=traffic_split,
            ),
        )
        self._experiments[exp.id] = exp
        logger.info("experiment_created", name=name, id=exp.id)
        return exp

    def start(self, experiment_id: str) -> bool:
        exp = self._experiments.get(experiment_id)
        if not exp or exp.status != ExperimentStatus.DRAFT:
            return False
        exp.status = ExperimentStatus.RUNNING
        exp.started_at = datetime.now()
        logger.info("experiment_started", id=experiment_id)
        return True

    def pause(self, experiment_id: str) -> bool:
        exp = self._experiments.get(experiment_id)
        if not exp or exp.status != ExperimentStatus.RUNNING:
            return False
        exp.status = ExperimentStatus.PAUSED
        return True

    def resume(self, experiment_id: str) -> bool:
        exp = self._experiments.get(experiment_id)
        if not exp or exp.status != ExperimentStatus.PAUSED:
            return False
        exp.status = ExperimentStatus.RUNNING
        return True

    def complete(self, experiment_id: str) -> bool:
        exp = self._experiments.get(experiment_id)
        if not exp:
            return False
        exp.status = ExperimentStatus.COMPLETED
        exp.completed_at = datetime.now()
        logger.info("experiment_completed", id=experiment_id)
        return True

    def cancel(self, experiment_id: str) -> bool:
        exp = self._experiments.get(experiment_id)
        if not exp:
            return False
        exp.status = ExperimentStatus.CANCELLED
        exp.completed_at = datetime.now()
        return True

    def get(self, experiment_id: str) -> Optional[Experiment]:
        return self._experiments.get(experiment_id)

    def get_active(self, action_type: Optional[ActionType] = None) -> List[Experiment]:
        exps = [e for e in self._experiments.values() if e.status == ExperimentStatus.RUNNING]
        if action_type:
            exps = [e for e in exps if e.action_type == action_type]
        return exps

    def list_all(self) -> List[Experiment]:
        return list(self._experiments.values())
