"""
AION Self-Improvement Engine

Autonomous evolution and optimization system with:
- Continuous performance monitoring
- Hypothesis-driven optimization
- Safe parameter adjustment
- Rollback on degradation
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import structlog

from aion.systems.evolution.hypothesis import (
    HypothesisGenerator,
    Hypothesis,
    HypothesisStatus,
    HypothesisType,
)
from aion.systems.evolution.optimizer import ParameterOptimizer, OptimizationBounds

logger = structlog.get_logger(__name__)


class EvolutionPhase(str, Enum):
    """Phase of the evolution process."""
    MONITORING = "monitoring"      # Collecting performance data
    HYPOTHESIS = "hypothesis"      # Generating improvement hypotheses
    TESTING = "testing"           # Testing hypotheses
    APPLYING = "applying"         # Applying validated changes
    ROLLBACK = "rollback"         # Rolling back failed changes


@dataclass
class PerformanceSnapshot:
    """A snapshot of system performance."""
    timestamp: datetime
    metrics: dict[str, float]
    parameters: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvolutionCheckpoint:
    """A checkpoint for rollback."""
    id: str
    timestamp: datetime
    parameters: dict[str, Any]
    performance: dict[str, float]
    applied_hypotheses: list[str]


class SafetyGuard:
    """
    Safety guard for self-improvement.

    Ensures that system performance stays above minimum thresholds
    and prevents dangerous modifications.
    """

    def __init__(
        self,
        min_performance: float = 0.9,  # Minimum relative performance
        max_degradation: float = 0.05,  # Maximum allowed degradation
        critical_metrics: Optional[list[str]] = None,
    ):
        self.min_performance = min_performance
        self.max_degradation = max_degradation
        self.critical_metrics = critical_metrics or []

        self._baseline_performance: dict[str, float] = {}
        self._blocked_parameters: set[str] = set()

    def set_baseline(self, metrics: dict[str, float]) -> None:
        """Set the baseline performance metrics."""
        self._baseline_performance = metrics.copy()
        logger.info("Safety baseline set", metrics=metrics)

    def check_performance(
        self,
        current_metrics: dict[str, float],
    ) -> tuple[bool, Optional[str]]:
        """
        Check if current performance is acceptable.

        Returns:
            Tuple of (is_safe, reason if not safe)
        """
        if not self._baseline_performance:
            return True, None

        for metric, baseline in self._baseline_performance.items():
            current = current_metrics.get(metric, 0)

            if baseline <= 0:
                continue

            relative = current / baseline

            # Check minimum performance
            if relative < self.min_performance:
                reason = f"Metric '{metric}' below minimum ({relative:.2%} of baseline)"
                return False, reason

            # Check critical metrics more strictly
            if metric in self.critical_metrics and relative < 0.95:
                reason = f"Critical metric '{metric}' degraded ({relative:.2%} of baseline)"
                return False, reason

        return True, None

    def check_change_safety(
        self,
        parameter: str,
        current_value: Any,
        proposed_value: Any,
    ) -> tuple[bool, Optional[str]]:
        """
        Check if a parameter change is safe.

        Returns:
            Tuple of (is_safe, reason if not safe)
        """
        if parameter in self._blocked_parameters:
            return False, f"Parameter '{parameter}' is blocked"

        # Check relative change magnitude
        if isinstance(current_value, (int, float)) and isinstance(proposed_value, (int, float)):
            if current_value != 0:
                change = abs(proposed_value - current_value) / abs(current_value)
                if change > 0.5:  # More than 50% change
                    return False, f"Change magnitude too large ({change:.0%})"

        return True, None

    def block_parameter(self, parameter: str) -> None:
        """Block a parameter from being modified."""
        self._blocked_parameters.add(parameter)

    def unblock_parameter(self, parameter: str) -> None:
        """Unblock a parameter."""
        self._blocked_parameters.discard(parameter)


class SelfImprovementEngine:
    """
    AION Self-Improvement Engine

    Continuously monitors system performance and makes
    data-driven improvements while maintaining safety bounds.
    """

    def __init__(
        self,
        safety_threshold: float = 0.95,
        improvement_interval: int = 3600,  # seconds
        min_samples: int = 100,
        require_approval: bool = True,
        checkpoint_dir: Optional[Path] = None,
    ):
        self.safety_threshold = safety_threshold
        self.improvement_interval = improvement_interval
        self.min_samples = min_samples
        self.require_approval = require_approval
        self.checkpoint_dir = checkpoint_dir or Path("./checkpoints/evolution")

        # Components
        self._hypothesis_generator = HypothesisGenerator(
            max_change_percent=0.1,
            min_confidence=0.3,
        )
        self._safety_guard = SafetyGuard(
            min_performance=safety_threshold,
        )

        # State
        self._phase = EvolutionPhase.MONITORING
        self._current_parameters: dict[str, Any] = {}
        self._parameter_bounds: dict[str, tuple[float, float]] = {}
        self._performance_history: list[PerformanceSnapshot] = []
        self._checkpoints: list[EvolutionCheckpoint] = []
        self._pending_approvals: list[Hypothesis] = []

        # Callbacks
        self._parameter_setter: Optional[Callable[[str, Any], bool]] = None
        self._performance_getter: Optional[Callable[[], dict[str, float]]] = None
        self._approval_callback: Optional[Callable[[Hypothesis], asyncio.Future]] = None

        # Background task
        self._improvement_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

        # Statistics
        self._stats = {
            "improvements_applied": 0,
            "rollbacks_performed": 0,
            "hypotheses_tested": 0,
            "hypotheses_validated": 0,
            "total_improvement": 0.0,
        }

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the self-improvement engine."""
        if self._initialized:
            return

        logger.info("Initializing Self-Improvement Engine")

        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self._initialized = True
        logger.info("Self-Improvement Engine initialized")

    async def shutdown(self) -> None:
        """Shutdown the engine."""
        logger.info("Shutting down Self-Improvement Engine")

        if self._improvement_task:
            self._stop_event.set()
            self._improvement_task.cancel()
            try:
                await self._improvement_task
            except asyncio.CancelledError:
                pass

        self._initialized = False

    def register_parameter(
        self,
        name: str,
        current_value: Any,
        bounds: Optional[tuple[float, float]] = None,
        setter: Optional[Callable[[Any], bool]] = None,
    ) -> None:
        """
        Register a parameter for optimization.

        Args:
            name: Parameter name
            current_value: Current value
            bounds: (min, max) bounds for numeric parameters
            setter: Optional custom setter function
        """
        self._current_parameters[name] = current_value
        if bounds:
            self._parameter_bounds[name] = bounds

        logger.debug("Parameter registered", name=name, value=current_value)

    def set_parameter_setter(
        self,
        setter: Callable[[str, Any], bool],
    ) -> None:
        """Set the callback for applying parameter changes."""
        self._parameter_setter = setter

    def set_performance_getter(
        self,
        getter: Callable[[], dict[str, float]],
    ) -> None:
        """Set the callback for getting current performance."""
        self._performance_getter = getter

    def set_approval_callback(
        self,
        callback: Callable[[Hypothesis], asyncio.Future],
    ) -> None:
        """Set the callback for requesting approvals."""
        self._approval_callback = callback

    def record_performance(
        self,
        metrics: dict[str, float],
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Record current performance metrics.

        Args:
            metrics: Performance metrics
            metadata: Additional metadata
        """
        snapshot = PerformanceSnapshot(
            timestamp=datetime.now(),
            metrics=metrics,
            parameters=self._current_parameters.copy(),
            metadata=metadata or {},
        )

        self._performance_history.append(snapshot)

        # Update hypothesis generator
        for metric, value in metrics.items():
            self._hypothesis_generator.record_metric(metric, value)

        # Limit history size
        if len(self._performance_history) > 10000:
            self._performance_history = self._performance_history[-10000:]

    async def start_improvement_loop(self) -> None:
        """Start the background improvement loop."""
        if self._improvement_task is not None:
            return

        self._stop_event.clear()
        self._improvement_task = asyncio.create_task(self._improvement_loop())
        logger.info("Started improvement loop")

    async def stop_improvement_loop(self) -> None:
        """Stop the improvement loop."""
        if self._improvement_task:
            self._stop_event.set()
            self._improvement_task.cancel()
            try:
                await self._improvement_task
            except asyncio.CancelledError:
                pass
            self._improvement_task = None
            logger.info("Stopped improvement loop")

    async def _improvement_loop(self) -> None:
        """Background improvement loop."""
        while not self._stop_event.is_set():
            try:
                await asyncio.sleep(self.improvement_interval)

                if self._stop_event.is_set():
                    break

                await self._run_improvement_cycle()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Improvement cycle error", error=str(e))

    async def _run_improvement_cycle(self) -> None:
        """Run a single improvement cycle."""
        logger.info("Starting improvement cycle")

        # Check if we have enough data
        if len(self._performance_history) < self.min_samples:
            logger.info(
                "Not enough samples for improvement",
                current=len(self._performance_history),
                required=self.min_samples,
            )
            return

        # Phase 1: Generate hypotheses
        self._phase = EvolutionPhase.HYPOTHESIS

        params_with_bounds = {
            name: (self._current_parameters[name], bounds)
            for name, bounds in self._parameter_bounds.items()
            if name in self._current_parameters
        }

        hypotheses = self._hypothesis_generator.generate_batch(
            params_with_bounds,
            count=5,
        )

        if not hypotheses:
            logger.info("No hypotheses generated")
            return

        # Phase 2: Test hypotheses
        self._phase = EvolutionPhase.TESTING

        for hypothesis in hypotheses:
            self._stats["hypotheses_tested"] += 1

            # Get baseline performance
            if self._performance_getter:
                baseline = self._performance_getter()
            else:
                baseline = self._get_recent_performance()

            # Apply proposed change temporarily
            success = await self._apply_parameter(
                hypothesis.target,
                hypothesis.proposed_value,
            )

            if not success:
                hypothesis.status = HypothesisStatus.REJECTED
                continue

            # Collect test samples
            test_metrics = []
            for _ in range(min(10, self.min_samples // 10)):
                await asyncio.sleep(0.1)  # Wait for new samples
                if self._performance_getter:
                    test_metrics.append(self._performance_getter())

            # Revert change
            await self._apply_parameter(
                hypothesis.target,
                hypothesis.current_value,
            )

            # Evaluate hypothesis
            if test_metrics:
                test_avg = {
                    k: np.mean([m.get(k, 0) for m in test_metrics])
                    for k in baseline
                }
                baseline_avg = np.mean(list(baseline.values()))
                test_avg_val = np.mean(list(test_avg.values()))

                validated = self._hypothesis_generator.evaluate_hypothesis(
                    hypothesis.id,
                    baseline_avg,
                    test_avg_val,
                    len(test_metrics),
                )

                if validated:
                    self._stats["hypotheses_validated"] += 1

                    # Request approval if required
                    if self.require_approval:
                        self._pending_approvals.append(hypothesis)
                    else:
                        await self._apply_validated_hypothesis(hypothesis)

    async def _apply_parameter(
        self,
        name: str,
        value: Any,
    ) -> bool:
        """Apply a parameter change."""
        # Safety check
        current = self._current_parameters.get(name)
        is_safe, reason = self._safety_guard.check_change_safety(name, current, value)

        if not is_safe:
            logger.warning("Parameter change blocked by safety guard", reason=reason)
            return False

        # Apply change
        if self._parameter_setter:
            success = self._parameter_setter(name, value)
        else:
            success = True

        if success:
            self._current_parameters[name] = value
            logger.debug("Parameter applied", name=name, value=value)

        return success

    async def _apply_validated_hypothesis(self, hypothesis: Hypothesis) -> None:
        """Apply a validated hypothesis."""
        self._phase = EvolutionPhase.APPLYING

        # Create checkpoint
        checkpoint = self._create_checkpoint([hypothesis.id])

        # Apply change
        success = await self._apply_parameter(
            hypothesis.target,
            hypothesis.proposed_value,
        )

        if success:
            hypothesis.status = HypothesisStatus.APPLIED
            self._stats["improvements_applied"] += 1

            if hypothesis.result_improvement:
                self._stats["total_improvement"] += hypothesis.result_improvement

            logger.info(
                "Hypothesis applied",
                hypothesis=hypothesis.name,
                improvement=hypothesis.result_improvement,
            )
        else:
            # Rollback
            await self._rollback_to_checkpoint(checkpoint.id)

    def _create_checkpoint(
        self,
        hypothesis_ids: list[str],
    ) -> EvolutionCheckpoint:
        """Create a checkpoint."""
        import uuid

        checkpoint = EvolutionCheckpoint(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            parameters=self._current_parameters.copy(),
            performance=self._get_recent_performance(),
            applied_hypotheses=hypothesis_ids,
        )

        self._checkpoints.append(checkpoint)

        # Save to disk
        self._save_checkpoint(checkpoint)

        return checkpoint

    async def _rollback_to_checkpoint(self, checkpoint_id: str) -> bool:
        """Rollback to a checkpoint."""
        self._phase = EvolutionPhase.ROLLBACK

        checkpoint = next(
            (c for c in self._checkpoints if c.id == checkpoint_id),
            None,
        )

        if not checkpoint:
            logger.error("Checkpoint not found", checkpoint_id=checkpoint_id)
            return False

        # Restore parameters
        for name, value in checkpoint.parameters.items():
            await self._apply_parameter(name, value)

        self._stats["rollbacks_performed"] += 1

        logger.warning(
            "Rolled back to checkpoint",
            checkpoint_id=checkpoint_id,
            timestamp=checkpoint.timestamp.isoformat(),
        )

        return True

    def _get_recent_performance(self) -> dict[str, float]:
        """Get average of recent performance metrics."""
        if not self._performance_history:
            return {}

        recent = self._performance_history[-10:]
        all_metrics = set()

        for snapshot in recent:
            all_metrics.update(snapshot.metrics.keys())

        return {
            metric: np.mean([
                s.metrics.get(metric, 0) for s in recent
            ])
            for metric in all_metrics
        }

    def _save_checkpoint(self, checkpoint: EvolutionCheckpoint) -> None:
        """Save checkpoint to disk."""
        path = self.checkpoint_dir / f"{checkpoint.id}.json"
        data = {
            "id": checkpoint.id,
            "timestamp": checkpoint.timestamp.isoformat(),
            "parameters": checkpoint.parameters,
            "performance": checkpoint.performance,
            "applied_hypotheses": checkpoint.applied_hypotheses,
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def approve_hypothesis(self, hypothesis_id: str) -> bool:
        """Approve a pending hypothesis."""
        for h in self._pending_approvals:
            if h.id == hypothesis_id:
                self._pending_approvals.remove(h)
                asyncio.create_task(self._apply_validated_hypothesis(h))
                return True
        return False

    def reject_hypothesis(self, hypothesis_id: str) -> bool:
        """Reject a pending hypothesis."""
        for h in self._pending_approvals:
            if h.id == hypothesis_id:
                self._pending_approvals.remove(h)
                h.status = HypothesisStatus.REJECTED
                return True
        return False

    def get_pending_approvals(self) -> list[dict[str, Any]]:
        """Get list of pending approval requests."""
        return [h.to_dict() for h in self._pending_approvals]

    def get_stats(self) -> dict[str, Any]:
        """Get engine statistics."""
        return {
            **self._stats,
            "phase": self._phase.value,
            "registered_parameters": len(self._current_parameters),
            "performance_samples": len(self._performance_history),
            "checkpoints": len(self._checkpoints),
            "pending_approvals": len(self._pending_approvals),
            "hypothesis_stats": self._hypothesis_generator.get_stats(),
        }

    def get_current_parameters(self) -> dict[str, Any]:
        """Get current parameter values."""
        return self._current_parameters.copy()

    def emergency_rollback(self) -> bool:
        """Perform emergency rollback to earliest checkpoint."""
        if not self._checkpoints:
            logger.warning("No checkpoints available for emergency rollback")
            return False

        earliest = self._checkpoints[0]
        asyncio.create_task(self._rollback_to_checkpoint(earliest.id))
        return True
