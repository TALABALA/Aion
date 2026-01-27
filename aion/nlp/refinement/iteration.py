"""
AION Iteration Manager - Manage iterative refinement cycles.

Tracks and orchestrates the feedback-refine-validate cycle
for progressive system improvement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import structlog

from aion.nlp.types import GeneratedCode, ProgrammingSession, ValidationResult

logger = structlog.get_logger(__name__)


@dataclass
class IterationSnapshot:
    """Snapshot of a single iteration."""

    iteration: int
    timestamp: datetime = field(default_factory=datetime.now)
    feedback: str = ""
    spec_snapshot: Optional[Dict[str, Any]] = None
    code_snapshot: Optional[str] = None
    validation_snapshot: Optional[Dict[str, Any]] = None
    changes_made: List[str] = field(default_factory=list)


class IterationManager:
    """
    Manages the iterative refinement cycle.

    Tracks:
    - Iteration history with snapshots
    - Convergence metrics
    - Quality progression
    """

    def __init__(self, max_iterations: int = 50):
        self._max_iterations = max_iterations
        self._histories: Dict[str, List[IterationSnapshot]] = {}

    def start_iteration(self, session: ProgrammingSession, feedback: str) -> int:
        """Start a new iteration and snapshot current state."""
        session.iterations += 1
        iteration = session.iterations

        if iteration > self._max_iterations:
            logger.warning(
                "Max iterations reached",
                session_id=session.id,
                max=self._max_iterations,
            )

        # Create snapshot
        snapshot = IterationSnapshot(
            iteration=iteration,
            feedback=feedback,
        )

        if session.current_spec and hasattr(session.current_spec, "to_dict"):
            snapshot.spec_snapshot = session.current_spec.to_dict()
        if session.current_code:
            snapshot.code_snapshot = session.current_code.code[:1000]

        # Store
        if session.id not in self._histories:
            self._histories[session.id] = []
        self._histories[session.id].append(snapshot)

        return iteration

    def record_result(
        self,
        session: ProgrammingSession,
        validation: Optional[ValidationResult],
        changes: List[str],
    ) -> None:
        """Record the result of an iteration."""
        history = self._histories.get(session.id, [])
        if history:
            latest = history[-1]
            latest.changes_made = changes
            if validation:
                latest.validation_snapshot = {
                    "status": validation.status.value,
                    "errors": len(validation.errors),
                    "warnings": len(validation.warnings),
                    "safety_score": validation.safety_score,
                }

    def get_history(self, session_id: str) -> List[IterationSnapshot]:
        """Get iteration history for a session."""
        return self._histories.get(session_id, [])

    def get_quality_trend(self, session_id: str) -> List[Dict[str, Any]]:
        """Get quality metrics over iterations."""
        history = self._histories.get(session_id, [])
        trend: List[Dict[str, Any]] = []
        for snapshot in history:
            if snapshot.validation_snapshot:
                trend.append({
                    "iteration": snapshot.iteration,
                    "errors": snapshot.validation_snapshot.get("errors", 0),
                    "warnings": snapshot.validation_snapshot.get("warnings", 0),
                    "safety": snapshot.validation_snapshot.get("safety_score", 1.0),
                })
        return trend

    def is_converging(self, session_id: str) -> bool:
        """Check if iterations are converging (errors decreasing)."""
        trend = self.get_quality_trend(session_id)
        if len(trend) < 2:
            return True

        recent = trend[-3:]
        error_counts = [t["errors"] for t in recent]
        return error_counts[-1] <= error_counts[0]

    def cleanup(self, session_id: str) -> None:
        """Clean up iteration history for a session."""
        self._histories.pop(session_id, None)
