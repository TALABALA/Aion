"""
Safety Constraints System

Implements safety constraints for agent actions and outputs.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional, Callable
import re
import structlog

logger = structlog.get_logger()


class ConstraintType(Enum):
    """Types of safety constraints."""

    FORBIDDEN_ACTION = "forbidden_action"
    REQUIRED_CONDITION = "required_condition"
    OUTPUT_FILTER = "output_filter"
    RATE_LIMIT = "rate_limit"
    RESOURCE_LIMIT = "resource_limit"
    SCOPE_LIMIT = "scope_limit"


class Severity(Enum):
    """Severity levels for violations."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SafetyConstraint:
    """A safety constraint."""

    id: str
    name: str
    description: str
    constraint_type: ConstraintType
    check_fn: Optional[Callable[[dict], bool]] = None
    pattern: Optional[str] = None  # Regex pattern for text checks
    parameters: dict[str, Any] = field(default_factory=dict)
    severity: Severity = Severity.MEDIUM
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.now)

    def check(self, context: dict[str, Any]) -> bool:
        """Check if constraint is satisfied (True = safe)."""
        if not self.enabled:
            return True

        if self.check_fn:
            return self.check_fn(context)

        if self.pattern:
            text = context.get("text", "") or context.get("action", "")
            if re.search(self.pattern, text, re.IGNORECASE):
                return False

        return True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "constraint_type": self.constraint_type.value,
            "severity": self.severity.value,
            "enabled": self.enabled,
        }


@dataclass
class SafetyViolation:
    """A safety constraint violation."""

    id: str
    constraint_id: str
    constraint_name: str
    severity: Severity
    context: dict[str, Any]
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    resolved: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "constraint_id": self.constraint_id,
            "constraint_name": self.constraint_name,
            "severity": self.severity.value,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "resolved": self.resolved,
        }


class SafetyChecker:
    """
    Safety checker for agent actions and outputs.

    Features:
    - Constraint management
    - Action pre-checking
    - Output filtering
    - Violation logging
    - Constitutional AI constraints
    """

    def __init__(self):
        self._constraints: dict[str, SafetyConstraint] = {}
        self._violations: list[SafetyViolation] = []
        self._violation_counter = 0

        # Add default constitutional constraints
        self._add_default_constraints()

    def _add_default_constraints(self) -> None:
        """Add default safety constraints."""
        defaults = [
            SafetyConstraint(
                id="const-harmful",
                name="No Harmful Content",
                description="Prevent generation of harmful content",
                constraint_type=ConstraintType.OUTPUT_FILTER,
                pattern=r"(kill|harm|destroy|attack|weapon)",
                severity=Severity.HIGH,
            ),
            SafetyConstraint(
                id="const-deception",
                name="No Deception",
                description="Prevent deceptive outputs",
                constraint_type=ConstraintType.OUTPUT_FILTER,
                check_fn=lambda ctx: not ctx.get("is_deceptive", False),
                severity=Severity.HIGH,
            ),
            SafetyConstraint(
                id="const-privacy",
                name="Protect Privacy",
                description="Prevent exposure of private data",
                constraint_type=ConstraintType.OUTPUT_FILTER,
                pattern=r"(ssn|social security|password|credit card)",
                severity=Severity.CRITICAL,
            ),
            SafetyConstraint(
                id="const-scope",
                name="Stay In Scope",
                description="Actions must be within defined scope",
                constraint_type=ConstraintType.SCOPE_LIMIT,
                check_fn=lambda ctx: ctx.get("in_scope", True),
                severity=Severity.MEDIUM,
            ),
        ]

        for constraint in defaults:
            self._constraints[constraint.id] = constraint

    def add_constraint(self, constraint: SafetyConstraint) -> None:
        """Add a safety constraint."""
        self._constraints[constraint.id] = constraint
        logger.info("constraint_added", id=constraint.id, name=constraint.name)

    def remove_constraint(self, constraint_id: str) -> bool:
        """Remove a constraint."""
        if constraint_id in self._constraints:
            del self._constraints[constraint_id]
            return True
        return False

    def enable_constraint(self, constraint_id: str) -> bool:
        """Enable a constraint."""
        if constraint_id in self._constraints:
            self._constraints[constraint_id].enabled = True
            return True
        return False

    def disable_constraint(self, constraint_id: str) -> bool:
        """Disable a constraint."""
        if constraint_id in self._constraints:
            self._constraints[constraint_id].enabled = False
            return True
        return False

    def check_action(self, action: str, context: dict[str, Any]) -> tuple[bool, list[SafetyViolation]]:
        """
        Check if an action is safe.

        Returns:
            Tuple of (is_safe, list of violations)
        """
        violations = []
        check_context = {"action": action, **context}

        for constraint in self._constraints.values():
            if constraint.constraint_type == ConstraintType.FORBIDDEN_ACTION:
                if not constraint.check(check_context):
                    violation = self._create_violation(
                        constraint,
                        check_context,
                        f"Action '{action}' violates constraint: {constraint.name}",
                    )
                    violations.append(violation)

        is_safe = len(violations) == 0

        if not is_safe:
            logger.warning(
                "action_safety_violations",
                action=action,
                violations=len(violations),
            )

        return is_safe, violations

    def check_output(self, text: str, context: dict[str, Any]) -> tuple[bool, list[SafetyViolation], str]:
        """
        Check if output text is safe.

        Returns:
            Tuple of (is_safe, violations, filtered_text)
        """
        violations = []
        check_context = {"text": text, **context}
        filtered_text = text

        for constraint in self._constraints.values():
            if constraint.constraint_type == ConstraintType.OUTPUT_FILTER:
                if not constraint.check(check_context):
                    violation = self._create_violation(
                        constraint,
                        check_context,
                        f"Output violates constraint: {constraint.name}",
                    )
                    violations.append(violation)

                    # Filter problematic content
                    if constraint.pattern:
                        filtered_text = re.sub(
                            constraint.pattern,
                            "[FILTERED]",
                            filtered_text,
                            flags=re.IGNORECASE,
                        )

        is_safe = len(violations) == 0

        return is_safe, violations, filtered_text

    def _create_violation(
        self,
        constraint: SafetyConstraint,
        context: dict[str, Any],
        message: str,
    ) -> SafetyViolation:
        """Create a violation record."""
        self._violation_counter += 1

        violation = SafetyViolation(
            id=f"violation-{self._violation_counter}",
            constraint_id=constraint.id,
            constraint_name=constraint.name,
            severity=constraint.severity,
            context=context,
            message=message,
        )

        self._violations.append(violation)

        logger.warning(
            "safety_violation",
            violation_id=violation.id,
            constraint=constraint.name,
            severity=constraint.severity.value,
        )

        return violation

    def get_violations(
        self,
        severity: Optional[Severity] = None,
        limit: int = 100,
    ) -> list[SafetyViolation]:
        """Get recent violations."""
        violations = self._violations

        if severity:
            violations = [v for v in violations if v.severity == severity]

        return violations[-limit:]

    def get_stats(self) -> dict[str, Any]:
        """Get safety statistics."""
        severity_counts = {}
        for v in self._violations:
            sev = v.severity.value
            severity_counts[sev] = severity_counts.get(sev, 0) + 1

        return {
            "total_constraints": len(self._constraints),
            "enabled_constraints": sum(1 for c in self._constraints.values() if c.enabled),
            "total_violations": len(self._violations),
            "violations_by_severity": severity_counts,
            "constraints": [c.to_dict() for c in self._constraints.values()],
        }
