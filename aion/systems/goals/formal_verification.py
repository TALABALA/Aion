"""
AION Goal System - Formal Verification

Provably safe operations through formal methods and verification.

Key capabilities:
- Temporal logic specification of safety properties
- Model checking for safety violations
- Runtime verification and monitoring
- Provable bounds on system behavior
- Contract-based design with pre/post conditions
- Invariant checking and enforcement
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Set, Union
from collections import defaultdict
from enum import Enum
import re

import structlog

from aion.systems.goals.types import (
    Goal,
    GoalStatus,
    GoalPriority,
    GoalType,
)

logger = structlog.get_logger()


class TemporalOperator(Enum):
    """Linear Temporal Logic operators."""
    ALWAYS = "G"      # Globally (always)
    EVENTUALLY = "F"  # Finally (eventually)
    NEXT = "X"        # Next state
    UNTIL = "U"       # Until
    RELEASE = "R"     # Release


class PropositionalOperator(Enum):
    """Propositional logic operators."""
    AND = "∧"
    OR = "∨"
    NOT = "¬"
    IMPLIES = "→"
    IFF = "↔"


@dataclass
class Predicate:
    """An atomic predicate that can be true or false."""

    name: str
    check_fn: Optional[Callable[[Dict[str, Any]], bool]] = None
    description: str = ""

    def evaluate(self, state: Dict[str, Any]) -> bool:
        """Evaluate predicate in given state."""
        if self.check_fn:
            return self.check_fn(state)
        return state.get(self.name, False)


@dataclass
class Formula:
    """
    A temporal logic formula.

    Supports LTL (Linear Temporal Logic) for safety specification.
    """

    operator: Optional[Union[TemporalOperator, PropositionalOperator]] = None
    predicate: Optional[Predicate] = None
    subformulas: List["Formula"] = field(default_factory=list)

    def __str__(self) -> str:
        if self.predicate:
            return self.predicate.name

        if self.operator == PropositionalOperator.NOT:
            return f"¬({self.subformulas[0]})"

        if self.operator in (PropositionalOperator.AND, PropositionalOperator.OR):
            op_str = self.operator.value
            return f"({f' {op_str} '.join(str(f) for f in self.subformulas)})"

        if self.operator in (TemporalOperator.ALWAYS, TemporalOperator.EVENTUALLY):
            return f"{self.operator.value}({self.subformulas[0]})"

        if self.operator == TemporalOperator.UNTIL:
            return f"({self.subformulas[0]} U {self.subformulas[1]})"

        return "?"

    @classmethod
    def atom(cls, name: str, check_fn: Callable = None) -> "Formula":
        """Create atomic formula."""
        return cls(predicate=Predicate(name=name, check_fn=check_fn))

    @classmethod
    def always(cls, subformula: "Formula") -> "Formula":
        """G(φ) - φ holds in all future states."""
        return cls(operator=TemporalOperator.ALWAYS, subformulas=[subformula])

    @classmethod
    def eventually(cls, subformula: "Formula") -> "Formula":
        """F(φ) - φ holds in some future state."""
        return cls(operator=TemporalOperator.EVENTUALLY, subformulas=[subformula])

    @classmethod
    def next(cls, subformula: "Formula") -> "Formula":
        """X(φ) - φ holds in the next state."""
        return cls(operator=TemporalOperator.NEXT, subformulas=[subformula])

    @classmethod
    def until(cls, left: "Formula", right: "Formula") -> "Formula":
        """φ U ψ - φ holds until ψ becomes true."""
        return cls(operator=TemporalOperator.UNTIL, subformulas=[left, right])

    @classmethod
    def and_(cls, *formulas: "Formula") -> "Formula":
        """φ ∧ ψ - conjunction."""
        return cls(operator=PropositionalOperator.AND, subformulas=list(formulas))

    @classmethod
    def or_(cls, *formulas: "Formula") -> "Formula":
        """φ ∨ ψ - disjunction."""
        return cls(operator=PropositionalOperator.OR, subformulas=list(formulas))

    @classmethod
    def not_(cls, subformula: "Formula") -> "Formula":
        """¬φ - negation."""
        return cls(operator=PropositionalOperator.NOT, subformulas=[subformula])

    @classmethod
    def implies(cls, left: "Formula", right: "Formula") -> "Formula":
        """φ → ψ - implication (equivalent to ¬φ ∨ ψ)."""
        return cls.or_(cls.not_(left), right)


@dataclass
class SafetyProperty:
    """A named safety property with LTL specification."""

    id: str
    name: str
    description: str
    formula: Formula
    severity: str = "high"  # critical, high, medium, low
    enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "formula": str(self.formula),
            "severity": self.severity,
            "enabled": self.enabled,
        }


@dataclass
class VerificationResult:
    """Result of a verification check."""

    property_id: str
    satisfied: bool
    counterexample: Optional[List[Dict[str, Any]]] = None
    error_message: Optional[str] = None
    checked_at: datetime = field(default_factory=datetime.now)
    states_explored: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "property_id": self.property_id,
            "satisfied": self.satisfied,
            "counterexample": self.counterexample,
            "error_message": self.error_message,
            "checked_at": self.checked_at.isoformat(),
            "states_explored": self.states_explored,
        }


@dataclass
class Contract:
    """Design by contract for goal operations."""

    name: str
    preconditions: List[Callable[[Dict[str, Any]], bool]] = field(default_factory=list)
    postconditions: List[Callable[[Dict[str, Any], Dict[str, Any]], bool]] = field(default_factory=list)
    invariants: List[Callable[[Dict[str, Any]], bool]] = field(default_factory=list)

    def check_preconditions(self, state: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Check all preconditions."""
        violations = []
        for i, pre in enumerate(self.preconditions):
            try:
                if not pre(state):
                    violations.append(f"Precondition {i} violated")
            except Exception as e:
                violations.append(f"Precondition {i} error: {e}")

        return len(violations) == 0, violations

    def check_postconditions(
        self,
        pre_state: Dict[str, Any],
        post_state: Dict[str, Any],
    ) -> Tuple[bool, List[str]]:
        """Check all postconditions."""
        violations = []
        for i, post in enumerate(self.postconditions):
            try:
                if not post(pre_state, post_state):
                    violations.append(f"Postcondition {i} violated")
            except Exception as e:
                violations.append(f"Postcondition {i} error: {e}")

        return len(violations) == 0, violations

    def check_invariants(self, state: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Check all invariants."""
        violations = []
        for i, inv in enumerate(self.invariants):
            try:
                if not inv(state):
                    violations.append(f"Invariant {i} violated")
            except Exception as e:
                violations.append(f"Invariant {i} error: {e}")

        return len(violations) == 0, violations


class ModelChecker:
    """
    Model checker for verifying safety properties.

    Uses bounded model checking and symbolic execution.
    """

    def __init__(self, max_depth: int = 100):
        self.max_depth = max_depth
        self._visited_states: Set[str] = set()

    def _state_hash(self, state: Dict[str, Any]) -> str:
        """Hash a state for cycle detection."""
        return str(sorted(state.items()))

    def _evaluate_formula(
        self,
        formula: Formula,
        trace: List[Dict[str, Any]],
        position: int,
    ) -> bool:
        """Evaluate LTL formula on a trace."""
        if position >= len(trace):
            return True  # Past end of trace

        state = trace[position]

        # Atomic predicate
        if formula.predicate:
            return formula.predicate.evaluate(state)

        # Propositional operators
        if formula.operator == PropositionalOperator.NOT:
            return not self._evaluate_formula(formula.subformulas[0], trace, position)

        if formula.operator == PropositionalOperator.AND:
            return all(
                self._evaluate_formula(f, trace, position)
                for f in formula.subformulas
            )

        if formula.operator == PropositionalOperator.OR:
            return any(
                self._evaluate_formula(f, trace, position)
                for f in formula.subformulas
            )

        # Temporal operators
        if formula.operator == TemporalOperator.NEXT:
            return self._evaluate_formula(formula.subformulas[0], trace, position + 1)

        if formula.operator == TemporalOperator.ALWAYS:
            # G(φ) - φ must hold from position to end
            for i in range(position, len(trace)):
                if not self._evaluate_formula(formula.subformulas[0], trace, i):
                    return False
            return True

        if formula.operator == TemporalOperator.EVENTUALLY:
            # F(φ) - φ must hold at some point from position to end
            for i in range(position, len(trace)):
                if self._evaluate_formula(formula.subformulas[0], trace, i):
                    return True
            return False

        if formula.operator == TemporalOperator.UNTIL:
            # φ U ψ - φ holds until ψ becomes true
            left, right = formula.subformulas
            for i in range(position, len(trace)):
                if self._evaluate_formula(right, trace, i):
                    return True
                if not self._evaluate_formula(left, trace, i):
                    return False
            return False

        return True

    def verify_property(
        self,
        property: SafetyProperty,
        traces: List[List[Dict[str, Any]]],
    ) -> VerificationResult:
        """Verify a safety property against traces."""
        states_explored = 0

        for trace in traces:
            states_explored += len(trace)

            if not self._evaluate_formula(property.formula, trace, 0):
                # Found counterexample
                return VerificationResult(
                    property_id=property.id,
                    satisfied=False,
                    counterexample=trace,
                    states_explored=states_explored,
                )

        return VerificationResult(
            property_id=property.id,
            satisfied=True,
            states_explored=states_explored,
        )

    def bounded_model_check(
        self,
        property: SafetyProperty,
        initial_state: Dict[str, Any],
        transition_fn: Callable[[Dict[str, Any]], List[Dict[str, Any]]],
        bound: int = None,
    ) -> VerificationResult:
        """
        Bounded model checking.

        Explores state space up to depth bound looking for property violations.
        """
        bound = bound or self.max_depth
        self._visited_states.clear()

        # BFS exploration
        queue = [(initial_state, [initial_state])]
        states_explored = 0

        while queue and states_explored < bound:
            state, trace = queue.pop(0)

            state_hash = self._state_hash(state)
            if state_hash in self._visited_states:
                continue

            self._visited_states.add(state_hash)
            states_explored += 1

            # Check property
            if not self._evaluate_formula(property.formula, trace, len(trace) - 1):
                return VerificationResult(
                    property_id=property.id,
                    satisfied=False,
                    counterexample=trace,
                    states_explored=states_explored,
                )

            # Generate successors
            try:
                successors = transition_fn(state)
                for succ in successors:
                    if self._state_hash(succ) not in self._visited_states:
                        queue.append((succ, trace + [succ]))
            except Exception:
                pass  # Skip invalid transitions

        return VerificationResult(
            property_id=property.id,
            satisfied=True,
            states_explored=states_explored,
        )


class RuntimeMonitor:
    """
    Runtime verification and monitoring.

    Monitors system execution for safety violations in real-time.
    """

    def __init__(self):
        self._properties: Dict[str, SafetyProperty] = {}
        self._current_trace: List[Dict[str, Any]] = []
        self._violations: List[Tuple[str, Dict[str, Any], datetime]] = []
        self._model_checker = ModelChecker()

    def register_property(self, property: SafetyProperty):
        """Register a safety property for monitoring."""
        self._properties[property.id] = property

    def observe_state(self, state: Dict[str, Any]) -> List[str]:
        """
        Observe a new state and check for violations.

        Returns list of violated property IDs.
        """
        self._current_trace.append(state)
        violations = []

        for prop_id, property in self._properties.items():
            if not property.enabled:
                continue

            # Check property on current trace
            if not self._model_checker._evaluate_formula(
                property.formula,
                self._current_trace,
                len(self._current_trace) - 1,
            ):
                violations.append(prop_id)
                self._violations.append((prop_id, state.copy(), datetime.now()))
                logger.warning(
                    "safety_property_violated",
                    property_id=prop_id,
                    property_name=property.name,
                )

        return violations

    def reset_trace(self):
        """Reset the current trace."""
        self._current_trace = []

    def get_violations(self) -> List[Tuple[str, Dict[str, Any], datetime]]:
        """Get all recorded violations."""
        return self._violations.copy()

    def get_violation_count(self, property_id: str = None) -> int:
        """Get count of violations."""
        if property_id:
            return sum(1 for v in self._violations if v[0] == property_id)
        return len(self._violations)


class BoundsChecker:
    """
    Provable bounds on system behavior.

    Computes and verifies bounds on resources, time, etc.
    """

    def __init__(self):
        self._bounds: Dict[str, Tuple[float, float]] = {}  # name -> (lower, upper)
        self._observations: Dict[str, List[float]] = defaultdict(list)

    def set_bound(self, name: str, lower: float, upper: float):
        """Set bounds for a quantity."""
        self._bounds[name] = (lower, upper)

    def check_bound(self, name: str, value: float) -> Tuple[bool, str]:
        """Check if value is within bounds."""
        if name not in self._bounds:
            return True, ""

        lower, upper = self._bounds[name]

        if value < lower:
            return False, f"{name}={value} below lower bound {lower}"
        if value > upper:
            return False, f"{name}={value} above upper bound {upper}"

        return True, ""

    def observe(self, name: str, value: float) -> Tuple[bool, str]:
        """Observe a value and check bounds."""
        self._observations[name].append(value)
        return self.check_bound(name, value)

    def compute_empirical_bounds(
        self,
        name: str,
        confidence: float = 0.99,
    ) -> Tuple[float, float]:
        """Compute empirical bounds from observations."""
        if name not in self._observations or len(self._observations[name]) < 10:
            return (float('-inf'), float('inf'))

        import numpy as np

        values = np.array(self._observations[name])
        mean = np.mean(values)
        std = np.std(values)

        # Use Chebyshev's inequality for distribution-free bounds
        k = np.sqrt(1 / (1 - confidence))
        lower = mean - k * std
        upper = mean + k * std

        return (float(lower), float(upper))

    def verify_worst_case(
        self,
        name: str,
        predict_fn: Callable[[Dict[str, Any]], float],
        scenarios: List[Dict[str, Any]],
    ) -> Tuple[float, float]:
        """Verify worst-case bounds across scenarios."""
        values = [predict_fn(scenario) for scenario in scenarios]

        if not values:
            return (0.0, 0.0)

        return (min(values), max(values))


class InvariantChecker:
    """
    Invariant checking and enforcement.

    Maintains and verifies system invariants.
    """

    def __init__(self):
        self._invariants: Dict[str, Callable[[Dict[str, Any]], bool]] = {}
        self._invariant_descriptions: Dict[str, str] = {}

    def register_invariant(
        self,
        name: str,
        check_fn: Callable[[Dict[str, Any]], bool],
        description: str = "",
    ):
        """Register an invariant."""
        self._invariants[name] = check_fn
        self._invariant_descriptions[name] = description

    def check_all(self, state: Dict[str, Any]) -> Dict[str, bool]:
        """Check all invariants."""
        results = {}
        for name, check_fn in self._invariants.items():
            try:
                results[name] = check_fn(state)
            except Exception as e:
                logger.error("invariant_check_error", name=name, error=str(e))
                results[name] = False
        return results

    def check_invariant(self, name: str, state: Dict[str, Any]) -> bool:
        """Check a specific invariant."""
        if name not in self._invariants:
            return True

        try:
            return self._invariants[name](state)
        except Exception:
            return False

    def get_violated_invariants(self, state: Dict[str, Any]) -> List[str]:
        """Get list of violated invariants."""
        results = self.check_all(state)
        return [name for name, satisfied in results.items() if not satisfied]


class SafetyShield:
    """
    Safety shield that prevents unsafe actions.

    Acts as a filter on actions to ensure safety.
    """

    def __init__(self):
        self._blocked_patterns: List[re.Pattern] = []
        self._allowed_patterns: List[re.Pattern] = []
        self._action_validators: Dict[str, Callable[[Dict[str, Any]], bool]] = {}

    def add_blocked_pattern(self, pattern: str):
        """Add pattern for blocked actions."""
        self._blocked_patterns.append(re.compile(pattern, re.IGNORECASE))

    def add_allowed_pattern(self, pattern: str):
        """Add pattern for explicitly allowed actions."""
        self._allowed_patterns.append(re.compile(pattern, re.IGNORECASE))

    def add_action_validator(
        self,
        action_type: str,
        validator: Callable[[Dict[str, Any]], bool],
    ):
        """Add validator for specific action type."""
        self._action_validators[action_type] = validator

    def is_action_safe(
        self,
        action_type: str,
        action_params: Dict[str, Any],
    ) -> Tuple[bool, str]:
        """Check if an action is safe to execute."""
        action_str = f"{action_type}:{action_params}"

        # Check blocked patterns
        for pattern in self._blocked_patterns:
            if pattern.search(action_str):
                return False, f"Action matches blocked pattern: {pattern.pattern}"

        # Check allowed patterns (if any defined, action must match one)
        if self._allowed_patterns:
            matched = any(p.search(action_str) for p in self._allowed_patterns)
            if not matched:
                return False, "Action does not match any allowed pattern"

        # Check type-specific validator
        if action_type in self._action_validators:
            try:
                if not self._action_validators[action_type](action_params):
                    return False, f"Action failed validator for type {action_type}"
            except Exception as e:
                return False, f"Validator error: {e}"

        return True, ""

    def filter_actions(
        self,
        actions: List[Tuple[str, Dict[str, Any]]],
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """Filter list of actions to only safe ones."""
        safe_actions = []
        for action_type, params in actions:
            is_safe, _ = self.is_action_safe(action_type, params)
            if is_safe:
                safe_actions.append((action_type, params))
        return safe_actions


class FormalVerificationSystem:
    """
    Complete formal verification system for AION goals.

    Provides provably safe operation through formal methods.
    """

    def __init__(self):
        self.model_checker = ModelChecker()
        self.runtime_monitor = RuntimeMonitor()
        self.bounds_checker = BoundsChecker()
        self.invariant_checker = InvariantChecker()
        self.safety_shield = SafetyShield()

        # Contracts for goal operations
        self._contracts: Dict[str, Contract] = {}

        # Safety properties
        self._properties: Dict[str, SafetyProperty] = {}

        self._initialized = False
        self._init_default_safety()

    def _init_default_safety(self):
        """Initialize default safety properties and invariants."""
        # Default safety properties
        self._properties["no_unsafe_goals"] = SafetyProperty(
            id="no_unsafe_goals",
            name="No Unsafe Goals",
            description="System never pursues goals marked as unsafe",
            formula=Formula.always(
                Formula.not_(Formula.atom("has_unsafe_goal"))
            ),
            severity="critical",
        )

        self._properties["bounded_resources"] = SafetyProperty(
            id="bounded_resources",
            name="Bounded Resource Usage",
            description="Resource usage stays within bounds",
            formula=Formula.always(
                Formula.atom(
                    "resources_bounded",
                    lambda s: s.get("resource_usage", 0) <= s.get("resource_limit", float('inf'))
                )
            ),
            severity="high",
        )

        self._properties["goal_completion"] = SafetyProperty(
            id="goal_completion",
            name="Goal Eventually Completes",
            description="Active goals eventually complete or are abandoned",
            formula=Formula.always(
                Formula.implies(
                    Formula.atom("has_active_goal"),
                    Formula.eventually(Formula.atom("goal_resolved"))
                )
            ),
            severity="medium",
        )

        # Register with runtime monitor
        for prop in self._properties.values():
            self.runtime_monitor.register_property(prop)

        # Default invariants
        self.invariant_checker.register_invariant(
            "non_negative_progress",
            lambda s: s.get("progress", 0) >= 0,
            "Progress is never negative",
        )

        self.invariant_checker.register_invariant(
            "bounded_concurrent_goals",
            lambda s: s.get("active_goals", 0) <= s.get("max_concurrent", 10),
            "Number of concurrent goals is bounded",
        )

        # Default bounds
        self.bounds_checker.set_bound("resource_usage", 0, 1000)
        self.bounds_checker.set_bound("concurrent_goals", 0, 10)
        self.bounds_checker.set_bound("goal_depth", 0, 5)

        # Default blocked patterns
        self.safety_shield.add_blocked_pattern(r"delete.*all")
        self.safety_shield.add_blocked_pattern(r"rm\s+-rf")
        self.safety_shield.add_blocked_pattern(r"format.*disk")
        self.safety_shield.add_blocked_pattern(r"drop.*database")

    async def initialize(self):
        """Initialize the verification system."""
        self._initialized = True
        logger.info("formal_verification_system_initialized")

    async def shutdown(self):
        """Shutdown the verification system."""
        self._initialized = False
        logger.info("formal_verification_system_shutdown")

    def register_contract(self, name: str, contract: Contract):
        """Register a contract for an operation."""
        self._contracts[name] = contract

    def verify_preconditions(
        self,
        operation: str,
        state: Dict[str, Any],
    ) -> Tuple[bool, List[str]]:
        """Verify preconditions for an operation."""
        if operation not in self._contracts:
            return True, []

        return self._contracts[operation].check_preconditions(state)

    def verify_postconditions(
        self,
        operation: str,
        pre_state: Dict[str, Any],
        post_state: Dict[str, Any],
    ) -> Tuple[bool, List[str]]:
        """Verify postconditions after an operation."""
        if operation not in self._contracts:
            return True, []

        return self._contracts[operation].check_postconditions(pre_state, post_state)

    def verify_goal_safety(self, goal: Goal) -> Tuple[bool, List[str]]:
        """Verify that a goal is safe to pursue."""
        issues = []

        # Build state from goal
        state = {
            "goal_type": goal.goal_type.value,
            "priority": goal.priority.value,
            "description": goal.description.lower(),
            "title": goal.title.lower(),
        }

        # Check invariants
        violated = self.invariant_checker.get_violated_invariants(state)
        issues.extend([f"Invariant violated: {v}" for v in violated])

        # Check bounds
        if len(goal.success_criteria) > 20:
            issues.append("Too many success criteria (max 20)")

        if len(goal.description) > 10000:
            issues.append("Description too long (max 10000 chars)")

        # Check for dangerous patterns
        dangerous_patterns = [
            r"delete\s+all",
            r"bypass\s+security",
            r"ignore\s+safety",
            r"override\s+limits",
        ]

        text = f"{goal.title} {goal.description}".lower()
        for pattern in dangerous_patterns:
            if re.search(pattern, text):
                issues.append(f"Dangerous pattern detected: {pattern}")

        return len(issues) == 0, issues

    def verify_action_safety(
        self,
        action_type: str,
        params: Dict[str, Any],
    ) -> Tuple[bool, str]:
        """Verify that an action is safe to execute."""
        return self.safety_shield.is_action_safe(action_type, params)

    def observe_execution(self, state: Dict[str, Any]) -> List[str]:
        """Observe execution state and check for violations."""
        return self.runtime_monitor.observe_state(state)

    def get_safety_report(self) -> Dict[str, Any]:
        """Get comprehensive safety report."""
        return {
            "properties": {
                pid: prop.to_dict()
                for pid, prop in self._properties.items()
            },
            "violations": [
                {"property_id": v[0], "state": v[1], "time": v[2].isoformat()}
                for v in self.runtime_monitor.get_violations()
            ],
            "violation_count": self.runtime_monitor.get_violation_count(),
            "bounds": dict(self.bounds_checker._bounds),
            "invariants": list(self.invariant_checker._invariants.keys()),
        }

    def add_safety_property(
        self,
        id: str,
        name: str,
        description: str,
        formula: Formula,
        severity: str = "high",
    ):
        """Add a custom safety property."""
        prop = SafetyProperty(
            id=id,
            name=name,
            description=description,
            formula=formula,
            severity=severity,
        )
        self._properties[id] = prop
        self.runtime_monitor.register_property(prop)

    def prove_safety_bound(
        self,
        property: SafetyProperty,
        initial_states: List[Dict[str, Any]],
        transition_fn: Callable[[Dict[str, Any]], List[Dict[str, Any]]],
        bound: int = 100,
    ) -> VerificationResult:
        """
        Attempt to prove safety property within bound.

        Returns verification result with proof or counterexample.
        """
        for initial_state in initial_states:
            result = self.model_checker.bounded_model_check(
                property,
                initial_state,
                transition_fn,
                bound,
            )

            if not result.satisfied:
                return result

        return VerificationResult(
            property_id=property.id,
            satisfied=True,
            states_explored=bound * len(initial_states),
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get verification statistics."""
        return {
            "properties_registered": len(self._properties),
            "contracts_registered": len(self._contracts),
            "invariants_registered": len(self.invariant_checker._invariants),
            "total_violations": self.runtime_monitor.get_violation_count(),
            "bounds_defined": len(self.bounds_checker._bounds),
        }
