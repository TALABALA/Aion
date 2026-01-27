"""AION Constraint Solver - Abstract constraint system for simulation worlds.

Provides:
- ConstraintSolver: Evaluates and enforces world constraints (invariants,
  resource limits, temporal ordering, mutual exclusion).
- Constraint propagation for multi-constraint satisfaction.
- Violation tracking and automated correction.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import structlog

from aion.simulation.types import (
    Constraint,
    ConstraintType,
    EventType,
    SimulationEvent,
    WorldState,
)

logger = structlog.get_logger(__name__)


@dataclass
class ConstraintViolation:
    """Record of a constraint violation."""

    constraint_id: str
    constraint_name: str
    tick: int
    simulation_time: float
    message: str
    severity: str  # error, warning
    corrected: bool = False
    context: Dict[str, Any] = field(default_factory=dict)


class ConstraintSolver:
    """Evaluates and enforces world constraints.

    Features:
    - Multiple constraint types (invariants, preconditions, resource limits, etc.).
    - Priority-ordered evaluation.
    - Automatic correction when possible.
    - Violation history and analysis.
    - Constraint propagation (iterative satisfaction).
    """

    def __init__(self, max_propagation_rounds: int = 10) -> None:
        self._constraints: List[Constraint] = []
        self._violations: List[ConstraintViolation] = []
        self._max_propagation = max_propagation_rounds

        # Pre/post condition hooks
        self._preconditions: Dict[str, List[Constraint]] = defaultdict(list)
        self._postconditions: Dict[str, List[Constraint]] = defaultdict(list)

    def add_constraint(self, constraint: Constraint) -> None:
        """Add a constraint."""
        self._constraints.append(constraint)
        self._constraints.sort(key=lambda c: c.priority, reverse=True)

    def add_invariant(
        self,
        name: str,
        condition: Callable[[WorldState], bool],
        on_violation: str = "error",
        correction: Optional[Callable[[WorldState], None]] = None,
        priority: int = 0,
    ) -> Constraint:
        """Add an invariant constraint (must always hold)."""
        c = Constraint(
            name=name,
            type=ConstraintType.INVARIANT,
            condition=condition,
            on_violation=on_violation,
            correction=correction,
            priority=priority,
        )
        self.add_constraint(c)
        return c

    def add_resource_limit(
        self,
        name: str,
        resource_key: str,
        max_value: float,
        correction: Optional[Callable[[WorldState], None]] = None,
    ) -> Constraint:
        """Add a resource limit constraint."""
        def check(state: WorldState) -> bool:
            return state.metrics.get(resource_key, 0.0) <= max_value

        c = Constraint(
            name=name,
            type=ConstraintType.RESOURCE_LIMIT,
            condition=check,
            on_violation="correct" if correction else "error",
            correction=correction,
            priority=10,
        )
        self.add_constraint(c)
        return c

    def add_precondition(
        self,
        action: str,
        name: str,
        condition: Callable[[WorldState], bool],
    ) -> Constraint:
        """Add a precondition for a specific action."""
        c = Constraint(
            name=name,
            type=ConstraintType.PRECONDITION,
            condition=condition,
        )
        self._preconditions[action].append(c)
        return c

    def add_postcondition(
        self,
        action: str,
        name: str,
        condition: Callable[[WorldState], bool],
    ) -> Constraint:
        """Add a postcondition for a specific action."""
        c = Constraint(
            name=name,
            type=ConstraintType.POSTCONDITION,
            condition=condition,
        )
        self._postconditions[action].append(c)
        return c

    # -- Evaluation --

    async def check_all(self, state: WorldState) -> List[ConstraintViolation]:
        """Check all constraints against current state.

        Returns:
            List of violations found.
        """
        violations: List[ConstraintViolation] = []

        for constraint in self._constraints:
            if constraint.condition is None:
                continue
            try:
                satisfied = constraint.condition(state)
            except Exception as exc:
                violations.append(ConstraintViolation(
                    constraint_id=constraint.id,
                    constraint_name=constraint.name,
                    tick=state.tick,
                    simulation_time=state.simulation_time,
                    message=f"Constraint evaluation error: {exc}",
                    severity="error",
                ))
                continue

            if not satisfied:
                violation = ConstraintViolation(
                    constraint_id=constraint.id,
                    constraint_name=constraint.name,
                    tick=state.tick,
                    simulation_time=state.simulation_time,
                    message=f"Constraint '{constraint.name}' violated",
                    severity="error" if constraint.on_violation == "error" else "warning",
                )

                # Attempt correction
                if constraint.on_violation in ("correct", "rollback") and constraint.correction:
                    try:
                        constraint.correction(state)
                        violation.corrected = True
                    except Exception as exc:
                        violation.message += f" (correction failed: {exc})"

                constraint.violations += 1
                constraint.last_violation_tick = state.tick
                violations.append(violation)

        self._violations.extend(violations)
        return violations

    async def check_preconditions(
        self,
        action: str,
        state: WorldState,
    ) -> Tuple[bool, List[ConstraintViolation]]:
        """Check preconditions for an action.

        Returns:
            (all_satisfied, violations)
        """
        violations: List[ConstraintViolation] = []
        all_ok = True

        for constraint in self._preconditions.get(action, []):
            if constraint.condition is None:
                continue
            try:
                if not constraint.condition(state):
                    all_ok = False
                    violations.append(ConstraintViolation(
                        constraint_id=constraint.id,
                        constraint_name=constraint.name,
                        tick=state.tick,
                        simulation_time=state.simulation_time,
                        message=f"Precondition '{constraint.name}' not met for action '{action}'",
                        severity="error",
                    ))
            except Exception as exc:
                all_ok = False
                violations.append(ConstraintViolation(
                    constraint_id=constraint.id,
                    constraint_name=constraint.name,
                    tick=state.tick,
                    simulation_time=state.simulation_time,
                    message=f"Precondition error: {exc}",
                    severity="error",
                ))

        self._violations.extend(violations)
        return all_ok, violations

    async def check_postconditions(
        self,
        action: str,
        state: WorldState,
    ) -> List[ConstraintViolation]:
        """Check postconditions after an action."""
        violations: List[ConstraintViolation] = []

        for constraint in self._postconditions.get(action, []):
            if constraint.condition is None:
                continue
            try:
                if not constraint.condition(state):
                    violations.append(ConstraintViolation(
                        constraint_id=constraint.id,
                        constraint_name=constraint.name,
                        tick=state.tick,
                        simulation_time=state.simulation_time,
                        message=f"Postcondition '{constraint.name}' not met after action '{action}'",
                        severity="warning",
                    ))
            except Exception as exc:
                violations.append(ConstraintViolation(
                    constraint_id=constraint.id,
                    constraint_name=constraint.name,
                    tick=state.tick,
                    simulation_time=state.simulation_time,
                    message=f"Postcondition error: {exc}",
                    severity="error",
                ))

        self._violations.extend(violations)
        return violations

    async def propagate(self, state: WorldState) -> int:
        """Run constraint propagation (iterative satisfaction).

        Repeatedly checks and corrects constraints until stable
        or max iterations reached.

        Returns:
            Number of corrections applied.
        """
        total_corrections = 0

        for _ in range(self._max_propagation):
            violations = await self.check_all(state)
            corrections = sum(1 for v in violations if v.corrected)
            total_corrections += corrections
            if corrections == 0:
                break

        return total_corrections

    # -- Violation Analysis --

    @property
    def violations(self) -> List[ConstraintViolation]:
        return list(self._violations)

    @property
    def violation_count(self) -> int:
        return len(self._violations)

    def violations_for_constraint(self, name: str) -> List[ConstraintViolation]:
        return [v for v in self._violations if v.constraint_name == name]

    def violations_at_tick(self, tick: int) -> List[ConstraintViolation]:
        return [v for v in self._violations if v.tick == tick]

    def violation_summary(self) -> Dict[str, Any]:
        """Aggregate violation statistics."""
        by_constraint: Dict[str, int] = defaultdict(int)
        by_severity: Dict[str, int] = defaultdict(int)
        corrected_count = 0

        for v in self._violations:
            by_constraint[v.constraint_name] += 1
            by_severity[v.severity] += 1
            if v.corrected:
                corrected_count += 1

        return {
            "total": len(self._violations),
            "by_constraint": dict(by_constraint),
            "by_severity": dict(by_severity),
            "corrected": corrected_count,
            "uncorrected": len(self._violations) - corrected_count,
        }

    def clear_violations(self) -> None:
        self._violations.clear()

    def clear(self) -> None:
        self._constraints.clear()
        self._preconditions.clear()
        self._postconditions.clear()
        self._violations.clear()
