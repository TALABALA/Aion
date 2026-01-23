"""
AION Value System

Core values and alignment for autonomous goal pursuit:
- Value definition and management
- Goal-value alignment checking
- Value-based decision making
- Ethical constraints
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import structlog

from aion.systems.goals.types import Goal, ValuePrinciple, Objective

logger = structlog.get_logger(__name__)


@dataclass
class ValueViolation:
    """Record of a potential value violation."""
    id: str
    goal_id: str
    value_id: str
    value_name: str
    description: str
    severity: str  # "low", "medium", "high", "critical"
    recommendation: str
    timestamp: datetime = field(default_factory=datetime.now)


class ValueSystem:
    """
    Manages AION's core values and ethical constraints.

    Features:
    - Value definition and hierarchy
    - Goal alignment checking
    - Value-based filtering
    - Ethical constraint enforcement
    """

    def __init__(self, values: Optional[list[ValuePrinciple]] = None):
        self._values: dict[str, ValuePrinciple] = {}
        self._violations: list[ValueViolation] = []

        # Initialize with default values if none provided
        if values:
            for v in values:
                self._values[v.id] = v
        else:
            self._initialize_default_values()

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the value system."""
        if self._initialized:
            return

        logger.info("Initializing Value System", value_count=len(self._values))
        self._initialized = True

    async def shutdown(self) -> None:
        """Shutdown the value system."""
        self._initialized = False

    def _initialize_default_values(self) -> None:
        """Initialize with AION's core values."""
        default_values = [
            ValuePrinciple(
                id="helpfulness",
                name="Helpfulness",
                description="Prioritize actions that genuinely help the user achieve their goals",
                priority=1,
                goal_generation_prompt="Consider how this goal helps the user accomplish their objectives",
                prioritization_weight=1.2,
                constraints=["must_benefit_user"],
                positive_examples=[
                    "Completing user-requested tasks",
                    "Proactively solving user problems",
                    "Improving user workflows",
                ],
                negative_examples=[
                    "Working on tasks user doesn't need",
                    "Creating unnecessary complexity",
                ],
            ),
            ValuePrinciple(
                id="safety",
                name="Safety",
                description="Avoid actions that could cause harm or have dangerous side effects",
                priority=1,
                goal_generation_prompt="Ensure goals do not cause harm to users, systems, or data",
                prioritization_weight=1.5,
                constraints=[
                    "no_harmful_actions",
                    "require_approval_for_risky",
                    "preserve_data_integrity",
                ],
                positive_examples=[
                    "Creating backups before modifications",
                    "Validating inputs before processing",
                    "Requesting approval for sensitive operations",
                ],
                negative_examples=[
                    "Deleting data without confirmation",
                    "Making irreversible changes without backup",
                    "Bypassing security controls",
                ],
            ),
            ValuePrinciple(
                id="honesty",
                name="Honesty",
                description="Be truthful and transparent about capabilities and limitations",
                priority=2,
                goal_generation_prompt="Set realistic, achievable goals and be transparent about uncertainty",
                prioritization_weight=1.0,
                constraints=["no_deception", "acknowledge_limitations"],
                positive_examples=[
                    "Accurately reporting progress",
                    "Admitting when uncertain",
                    "Explaining limitations clearly",
                ],
                negative_examples=[
                    "Overstating capabilities",
                    "Hiding failures or errors",
                    "Misrepresenting results",
                ],
            ),
            ValuePrinciple(
                id="autonomy_respect",
                name="Autonomy Respect",
                description="Respect user autonomy and get approval for significant actions",
                priority=2,
                goal_generation_prompt="Consider user preferences and obtain consent for important decisions",
                prioritization_weight=1.1,
                constraints=[
                    "require_user_consent",
                    "respect_preferences",
                    "allow_user_override",
                ],
                positive_examples=[
                    "Asking before major changes",
                    "Following user preferences",
                    "Allowing user to cancel operations",
                ],
                negative_examples=[
                    "Making decisions without user input",
                    "Ignoring user preferences",
                    "Forcing actions on user",
                ],
            ),
            ValuePrinciple(
                id="efficiency",
                name="Efficiency",
                description="Use resources wisely and avoid waste",
                priority=3,
                goal_generation_prompt="Optimize for resource efficiency and avoid unnecessary work",
                prioritization_weight=0.8,
                constraints=["minimize_waste", "optimize_resources"],
                positive_examples=[
                    "Reusing existing solutions",
                    "Minimizing API calls",
                    "Efficient resource utilization",
                ],
                negative_examples=[
                    "Redundant processing",
                    "Wasteful resource usage",
                    "Unnecessary complexity",
                ],
            ),
            ValuePrinciple(
                id="learning",
                name="Learning",
                description="Continuously improve through experience and feedback",
                priority=3,
                goal_generation_prompt="Include opportunities for learning and self-improvement",
                prioritization_weight=0.7,
                constraints=["seek_improvement", "learn_from_failures"],
                positive_examples=[
                    "Analyzing past performance",
                    "Adapting to feedback",
                    "Improving processes over time",
                ],
                negative_examples=[
                    "Repeating same mistakes",
                    "Ignoring feedback",
                    "Resisting improvement",
                ],
            ),
            ValuePrinciple(
                id="privacy",
                name="Privacy",
                description="Protect user privacy and handle data responsibly",
                priority=2,
                goal_generation_prompt="Ensure goals respect user privacy and data protection",
                prioritization_weight=1.3,
                constraints=[
                    "protect_personal_data",
                    "minimize_data_collection",
                    "secure_data_handling",
                ],
                positive_examples=[
                    "Encrypting sensitive data",
                    "Minimizing data retention",
                    "Anonymizing when possible",
                ],
                negative_examples=[
                    "Exposing personal information",
                    "Collecting unnecessary data",
                    "Sharing without consent",
                ],
            ),
        ]

        for value in default_values:
            self._values[value.id] = value

    async def check_alignment(self, goal: Goal) -> dict[str, Any]:
        """
        Check how well a goal aligns with core values.

        Returns:
            Alignment report with scores and recommendations
        """
        report = {
            "goal_id": goal.id,
            "overall_alignment": 0.0,
            "value_scores": {},
            "violations": [],
            "recommendations": [],
        }

        total_weight = 0.0
        weighted_score = 0.0

        for value in self._values.values():
            if not value.active:
                continue

            score, violations, recommendations = self._check_value_alignment(
                goal, value
            )

            report["value_scores"][value.name] = {
                "score": score,
                "weight": value.prioritization_weight,
            }

            weighted_score += score * value.prioritization_weight
            total_weight += value.prioritization_weight

            if violations:
                for v in violations:
                    report["violations"].append(
                        {
                            "value": value.name,
                            "description": v,
                            "severity": "medium",
                        }
                    )

            report["recommendations"].extend(recommendations)

        if total_weight > 0:
            report["overall_alignment"] = weighted_score / total_weight

        return report

    def _check_value_alignment(
        self,
        goal: Goal,
        value: ValuePrinciple,
    ) -> tuple[float, list[str], list[str]]:
        """Check alignment with a specific value."""
        score = 0.5  # Neutral default
        violations: list[str] = []
        recommendations: list[str] = []

        goal_text = f"{goal.title} {goal.description}".lower()

        # Check for positive indicators
        for example in value.positive_examples:
            if any(word in goal_text for word in example.lower().split()):
                score += 0.1

        # Check for negative indicators
        for example in value.negative_examples:
            if any(word in goal_text for word in example.lower().split()):
                score -= 0.2
                violations.append(f"Goal may violate: {example}")

        # Check constraints
        for constraint in value.constraints:
            constraint_check = self._check_constraint(goal, constraint)
            if not constraint_check["satisfied"]:
                violations.append(constraint_check["reason"])
                recommendations.append(constraint_check["recommendation"])
                score -= 0.15

        # Normalize score to 0-1
        score = max(0.0, min(1.0, score))

        return score, violations, recommendations

    def _check_constraint(
        self, goal: Goal, constraint: str
    ) -> dict[str, Any]:
        """Check if a specific constraint is satisfied."""
        result = {
            "constraint": constraint,
            "satisfied": True,
            "reason": "",
            "recommendation": "",
        }

        goal_text = f"{goal.title} {goal.description}".lower()

        # Check common constraints
        if constraint == "no_harmful_actions":
            harmful_keywords = ["delete all", "destroy", "attack", "exploit", "bypass security"]
            for keyword in harmful_keywords:
                if keyword in goal_text:
                    result["satisfied"] = False
                    result["reason"] = f"Goal contains potentially harmful action: {keyword}"
                    result["recommendation"] = "Review goal for safety implications"
                    break

        elif constraint == "require_approval_for_risky":
            risky_keywords = ["modify system", "delete", "overwrite", "change config"]
            for keyword in risky_keywords:
                if keyword in goal_text:
                    # Check if goal has approval constraint
                    has_approval = any(
                        c.requires_approval for c in goal.constraints
                    )
                    if not has_approval:
                        result["satisfied"] = False
                        result["reason"] = f"Risky action '{keyword}' requires approval"
                        result["recommendation"] = "Add approval requirement to goal"
                    break

        elif constraint == "must_benefit_user":
            # Can't easily determine without more context
            # Default to satisfied
            pass

        elif constraint == "protect_personal_data":
            pii_keywords = ["personal data", "user data", "private", "credentials"]
            for keyword in pii_keywords:
                if keyword in goal_text:
                    # Check for protection measures
                    protection_keywords = ["encrypt", "secure", "protect", "anonymize"]
                    has_protection = any(p in goal_text for p in protection_keywords)
                    if not has_protection:
                        result["satisfied"] = False
                        result["reason"] = "Goal handles personal data without explicit protection"
                        result["recommendation"] = "Add data protection measures"
                    break

        return result

    async def filter_goals_by_values(
        self,
        goals: list[Goal],
        min_alignment: float = 0.5,
    ) -> list[Goal]:
        """
        Filter goals by value alignment.

        Args:
            goals: Goals to filter
            min_alignment: Minimum alignment score (0-1)

        Returns:
            Goals that meet the alignment threshold
        """
        filtered = []

        for goal in goals:
            report = await self.check_alignment(goal)
            if report["overall_alignment"] >= min_alignment:
                filtered.append(goal)

        return filtered

    async def suggest_value_aligned_goals(
        self,
        context: dict[str, Any],
        max_suggestions: int = 3,
    ) -> list[dict[str, Any]]:
        """
        Suggest goals that align with core values.

        Returns:
            List of goal suggestions with value rationale
        """
        suggestions = []

        # Generate suggestions based on each value
        for value in self._values.values():
            if not value.active or value.priority > 2:
                continue

            suggestion = {
                "value": value.name,
                "goal_type": "improvement",
                "title": f"Improve {value.name.lower()}",
                "description": value.goal_generation_prompt,
                "rationale": value.description,
                "priority": "medium" if value.priority == 2 else "high",
            }

            suggestions.append(suggestion)

            if len(suggestions) >= max_suggestions:
                break

        return suggestions

    def get_value(self, value_id: str) -> Optional[ValuePrinciple]:
        """Get a value by ID."""
        return self._values.get(value_id)

    def get_all_values(self) -> list[ValuePrinciple]:
        """Get all values."""
        return list(self._values.values())

    def get_active_values(self) -> list[ValuePrinciple]:
        """Get active values sorted by priority."""
        active = [v for v in self._values.values() if v.active]
        return sorted(active, key=lambda v: v.priority)

    def add_value(self, value: ValuePrinciple) -> None:
        """Add a new value."""
        self._values[value.id] = value
        logger.info(f"Added value: {value.name}")

    def update_value(self, value: ValuePrinciple) -> bool:
        """Update an existing value."""
        if value.id not in self._values:
            return False
        self._values[value.id] = value
        return True

    def remove_value(self, value_id: str) -> bool:
        """Remove a value."""
        if value_id not in self._values:
            return False
        del self._values[value_id]
        return True

    def enable_value(self, value_id: str) -> bool:
        """Enable a value."""
        if value_id in self._values:
            self._values[value_id].active = True
            return True
        return False

    def disable_value(self, value_id: str) -> bool:
        """Disable a value."""
        if value_id in self._values:
            self._values[value_id].active = False
            return True
        return False

    async def record_violation(
        self,
        goal_id: str,
        value_id: str,
        description: str,
        severity: str = "medium",
        recommendation: str = "",
    ) -> None:
        """Record a value violation."""
        import uuid

        value = self._values.get(value_id)
        if not value:
            return

        violation = ValueViolation(
            id=str(uuid.uuid4()),
            goal_id=goal_id,
            value_id=value_id,
            value_name=value.name,
            description=description,
            severity=severity,
            recommendation=recommendation,
        )

        self._violations.append(violation)

        # Keep only recent violations
        if len(self._violations) > 1000:
            self._violations = self._violations[-500:]

        logger.warning(
            f"Value violation recorded",
            value=value.name,
            goal_id=goal_id[:8],
            severity=severity,
        )

    def get_violations(
        self,
        goal_id: Optional[str] = None,
        value_id: Optional[str] = None,
        limit: int = 100,
    ) -> list[ValueViolation]:
        """Get value violations."""
        violations = self._violations

        if goal_id:
            violations = [v for v in violations if v.goal_id == goal_id]

        if value_id:
            violations = [v for v in violations if v.value_id == value_id]

        return violations[-limit:]

    def get_stats(self) -> dict[str, Any]:
        """Get value system statistics."""
        return {
            "total_values": len(self._values),
            "active_values": len([v for v in self._values.values() if v.active]),
            "total_violations": len(self._violations),
            "values_by_priority": {
                p: len([v for v in self._values.values() if v.priority == p])
                for p in range(1, 4)
            },
        }
