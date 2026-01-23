"""
AION Goal Safety Boundaries

Enforces safety constraints on autonomous goal pursuit:
- Action constraints
- Resource limits
- Human approval gates
- Rollback capabilities
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Optional

import structlog

from aion.systems.goals.types import Goal, GoalConstraint

logger = structlog.get_logger(__name__)


class RuleAction(str, Enum):
    """Action to take when a rule is triggered."""
    BLOCK = "block"
    WARN = "warn"
    REQUIRE_APPROVAL = "require_approval"
    LOG = "log"
    THROTTLE = "throttle"


@dataclass
class SafetyRule:
    """A safety rule that constrains goal execution."""
    id: str
    name: str
    description: str

    # Rule type
    rule_type: str  # "action", "resource", "approval", "time", "content"

    # Conditions
    condition: str  # Expression or description of when rule applies

    # Actions
    action: RuleAction  # What to do when rule triggers

    # Parameters
    parameters: dict[str, Any] = field(default_factory=dict)

    # Status
    enabled: bool = True

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    triggered_count: int = 0
    last_triggered: Optional[datetime] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "rule_type": self.rule_type,
            "condition": self.condition,
            "action": self.action.value,
            "parameters": self.parameters,
            "enabled": self.enabled,
            "triggered_count": self.triggered_count,
        }


@dataclass
class ApprovalRequest:
    """A request for human approval."""
    id: str
    goal_id: str
    action_description: str
    reason: str

    # Status
    status: str = "pending"  # "pending", "approved", "denied", "expired"

    # Timing
    requested_at: datetime = field(default_factory=datetime.now)
    responded_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    response_by: Optional[str] = None

    # Auto-approve settings
    auto_approve_after: Optional[timedelta] = None

    # Additional context
    context: dict[str, Any] = field(default_factory=dict)
    risk_level: str = "medium"  # "low", "medium", "high", "critical"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "goal_id": self.goal_id,
            "action_description": self.action_description,
            "reason": self.reason,
            "status": self.status,
            "requested_at": self.requested_at.isoformat(),
            "responded_at": self.responded_at.isoformat() if self.responded_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "response_by": self.response_by,
            "risk_level": self.risk_level,
        }

    def is_expired(self) -> bool:
        """Check if the request has expired."""
        if self.expires_at and datetime.now() > self.expires_at:
            return True
        return False

    def can_auto_approve(self) -> bool:
        """Check if the request can be auto-approved."""
        if not self.auto_approve_after:
            return False
        elapsed = datetime.now() - self.requested_at
        return elapsed >= self.auto_approve_after


@dataclass
class SafetyViolation:
    """Record of a safety violation."""
    id: str
    goal_id: str
    rule_id: str
    rule_name: str
    description: str
    severity: str  # "low", "medium", "high", "critical"
    action_taken: str
    timestamp: datetime = field(default_factory=datetime.now)
    details: dict[str, Any] = field(default_factory=dict)


class SafetyBoundary:
    """
    Enforces safety boundaries on goal execution.

    Features:
    - Pre-execution checks
    - Resource monitoring
    - Human approval gates
    - Action logging
    - Emergency stop
    """

    def __init__(
        self,
        default_token_limit: int = 1000000,
        default_cost_limit: float = 10.0,
        default_duration_limit_hours: float = 24.0,
    ):
        self._rules: dict[str, SafetyRule] = {}
        self._approval_queue: dict[str, ApprovalRequest] = {}
        self._violations: list[SafetyViolation] = []
        self._blocked_actions: list[dict] = []

        # Emergency stop flag
        self._emergency_stop = False
        self._emergency_stop_reason: Optional[str] = None

        # Resource limits
        self._default_token_limit = default_token_limit
        self._default_cost_limit = default_cost_limit
        self._default_duration_limit = default_duration_limit_hours

        # Callbacks
        self._approval_callbacks: list[Callable[[ApprovalRequest], None]] = []
        self._violation_callbacks: list[Callable[[SafetyViolation], None]] = []

        # Lock for thread safety
        self._lock = asyncio.Lock()

        # Initialize default rules
        self._initialize_default_rules()

    def _initialize_default_rules(self) -> None:
        """Set up default safety rules."""
        default_rules = [
            SafetyRule(
                id="no_dangerous_actions",
                name="Block Dangerous Actions",
                description="Prevent actions that could cause harm",
                rule_type="action",
                condition="action.category == 'dangerous'",
                action=RuleAction.BLOCK,
                parameters={
                    "dangerous_categories": [
                        "delete_all",
                        "modify_system",
                        "execute_unsafe",
                        "network_attack",
                        "data_exfiltration",
                    ]
                },
            ),
            SafetyRule(
                id="resource_limits",
                name="Enforce Resource Limits",
                description="Ensure goals don't exceed resource limits",
                rule_type="resource",
                condition="resource_usage > limit",
                action=RuleAction.BLOCK,
                parameters={
                    "max_tokens_per_goal": 1000000,
                    "max_cost_per_goal": 10.0,
                    "max_duration_hours": 24,
                    "max_tool_calls_per_goal": 1000,
                },
            ),
            SafetyRule(
                id="external_action_approval",
                name="Require Approval for External Actions",
                description="Get approval before actions affecting external systems",
                rule_type="approval",
                condition="action.affects_external",
                action=RuleAction.REQUIRE_APPROVAL,
                parameters={
                    "external_actions": [
                        "send_email",
                        "api_call",
                        "file_upload",
                        "database_write",
                    ]
                },
            ),
            SafetyRule(
                id="high_impact_approval",
                name="Require Approval for High Impact",
                description="Get approval for high-impact actions",
                rule_type="approval",
                condition="action.impact_level == 'high'",
                action=RuleAction.REQUIRE_APPROVAL,
            ),
            SafetyRule(
                id="rate_limiting",
                name="Rate Limit Actions",
                description="Prevent excessive action rates",
                rule_type="resource",
                condition="action_rate > threshold",
                action=RuleAction.THROTTLE,
                parameters={
                    "max_actions_per_minute": 60,
                    "max_api_calls_per_minute": 30,
                },
            ),
            SafetyRule(
                id="content_safety",
                name="Content Safety Check",
                description="Check content for safety violations",
                rule_type="content",
                condition="content.contains_unsafe",
                action=RuleAction.BLOCK,
                parameters={
                    "check_pii": True,
                    "check_harmful": True,
                    "check_secrets": True,
                },
            ),
        ]

        for rule in default_rules:
            self._rules[rule.id] = rule

    async def check_goal_safety(self, goal: Goal) -> tuple[bool, list[str]]:
        """
        Check if a goal is safe to pursue.

        Returns:
            Tuple of (is_safe, list of concerns)
        """
        concerns = []

        if self._emergency_stop:
            return False, [f"Emergency stop is active: {self._emergency_stop_reason}"]

        # Check against all enabled rules
        for rule in self._rules.values():
            if not rule.enabled:
                continue

            violation = await self._check_rule(rule, goal)
            if violation:
                rule.triggered_count += 1
                rule.last_triggered = datetime.now()

                if rule.action == RuleAction.BLOCK:
                    concerns.append(f"BLOCKED: {rule.description} - {violation}")
                    await self._record_violation(
                        goal.id, rule, violation, "blocked", "high"
                    )
                elif rule.action == RuleAction.WARN:
                    concerns.append(f"WARNING: {rule.description} - {violation}")
                    await self._record_violation(
                        goal.id, rule, violation, "warned", "medium"
                    )
                elif rule.action == RuleAction.REQUIRE_APPROVAL:
                    concerns.append(f"APPROVAL REQUIRED: {rule.description}")

        is_safe = not any("BLOCKED" in c for c in concerns)

        return is_safe, concerns

    async def check_action_safety(
        self,
        goal: Goal,
        action: dict[str, Any],
    ) -> tuple[bool, Optional[str]]:
        """
        Check if a specific action is safe to execute.

        Returns:
            Tuple of (is_safe, reason_if_blocked)
        """
        if self._emergency_stop:
            return False, f"Emergency stop is active: {self._emergency_stop_reason}"

        # Check action-specific rules
        for rule in self._rules.values():
            if not rule.enabled or rule.rule_type not in ("action", "content"):
                continue

            violation = self._check_action_rule(rule, action)
            if violation:
                rule.triggered_count += 1
                rule.last_triggered = datetime.now()

                if rule.action == RuleAction.BLOCK:
                    self._blocked_actions.append(
                        {
                            "goal_id": goal.id,
                            "action": action,
                            "rule": rule.name,
                            "timestamp": datetime.now(),
                        }
                    )
                    await self._record_violation(
                        goal.id, rule, violation, "blocked", "high"
                    )
                    return False, f"{rule.name}: {violation}"

                elif rule.action == RuleAction.REQUIRE_APPROVAL:
                    # Create approval request
                    request_id = await self.request_approval(
                        goal.id,
                        str(action),
                        rule.description,
                        risk_level="medium",
                    )
                    return False, f"Requires approval (request_id: {request_id})"

        return True, None

    async def _check_rule(self, rule: SafetyRule, goal: Goal) -> Optional[str]:
        """Check if a goal violates a rule."""
        if rule.rule_type == "resource":
            params = rule.parameters

            # Check token limit
            max_tokens = params.get(
                "max_tokens_per_goal", self._default_token_limit
            )
            for constraint in goal.constraints:
                if constraint.max_tokens and constraint.max_tokens > max_tokens:
                    return f"Token limit exceeds maximum ({constraint.max_tokens} > {max_tokens})"

            # Check cost limit
            max_cost = params.get("max_cost_per_goal", self._default_cost_limit)
            for constraint in goal.constraints:
                if constraint.max_cost_dollars and constraint.max_cost_dollars > max_cost:
                    return f"Cost limit exceeds maximum (${constraint.max_cost_dollars} > ${max_cost})"

            # Check duration limit
            max_hours = params.get(
                "max_duration_hours", self._default_duration_limit
            )
            for constraint in goal.constraints:
                if constraint.max_duration_hours and constraint.max_duration_hours > max_hours:
                    return f"Duration limit exceeds maximum ({constraint.max_duration_hours}h > {max_hours}h)"

        elif rule.rule_type == "approval":
            # Check if goal requires approval
            for constraint in goal.constraints:
                if constraint.requires_approval and not constraint.approved_at:
                    return f"Requires approval: {constraint.description}"

        return None

    def _check_action_rule(self, rule: SafetyRule, action: dict) -> Optional[str]:
        """Check if an action violates a rule."""
        action_type = action.get("type", "")
        action_name = action.get("name", "")

        if rule.rule_type == "action":
            # Check dangerous action types
            dangerous = rule.parameters.get("dangerous_categories", [])
            if action_type in dangerous or action_name in dangerous:
                return f"Action type '{action_type or action_name}' is blocked"

        elif rule.rule_type == "content":
            # Check content safety
            content = action.get("content", "")
            params = rule.parameters

            if params.get("check_secrets"):
                # Simple check for common secret patterns
                secret_patterns = ["password", "api_key", "secret", "token"]
                for pattern in secret_patterns:
                    if pattern.lower() in content.lower():
                        return f"Content may contain secrets ({pattern})"

        return None

    async def _record_violation(
        self,
        goal_id: str,
        rule: SafetyRule,
        description: str,
        action_taken: str,
        severity: str,
    ) -> None:
        """Record a safety violation."""
        import uuid

        violation = SafetyViolation(
            id=str(uuid.uuid4()),
            goal_id=goal_id,
            rule_id=rule.id,
            rule_name=rule.name,
            description=description,
            severity=severity,
            action_taken=action_taken,
        )

        async with self._lock:
            self._violations.append(violation)

            # Keep only recent violations
            if len(self._violations) > 1000:
                self._violations = self._violations[-500:]

        # Notify callbacks
        for callback in self._violation_callbacks:
            try:
                callback(violation)
            except Exception as e:
                logger.error(f"Violation callback error: {e}")

        logger.warning(
            "Safety violation",
            goal_id=goal_id[:8],
            rule=rule.name,
            severity=severity,
            action=action_taken,
        )

    async def request_approval(
        self,
        goal_id: str,
        action_description: str,
        reason: str,
        risk_level: str = "medium",
        auto_approve_minutes: Optional[int] = None,
        expires_minutes: Optional[int] = 60,
    ) -> str:
        """Create an approval request."""
        import uuid

        request_id = str(uuid.uuid4())

        auto_approve_after = None
        if auto_approve_minutes:
            auto_approve_after = timedelta(minutes=auto_approve_minutes)

        expires_at = None
        if expires_minutes:
            expires_at = datetime.now() + timedelta(minutes=expires_minutes)

        request = ApprovalRequest(
            id=request_id,
            goal_id=goal_id,
            action_description=action_description,
            reason=reason,
            risk_level=risk_level,
            auto_approve_after=auto_approve_after,
            expires_at=expires_at,
        )

        async with self._lock:
            self._approval_queue[request_id] = request

        # Notify callbacks
        for callback in self._approval_callbacks:
            try:
                callback(request)
            except Exception as e:
                logger.error(f"Approval callback error: {e}")

        logger.info(
            f"Approval requested",
            request_id=request_id[:8],
            goal_id=goal_id[:8],
            risk_level=risk_level,
        )

        return request_id

    async def get_approval_status(self, request_id: str) -> Optional[str]:
        """Get the status of an approval request."""
        request = self._approval_queue.get(request_id)
        if not request:
            return None

        # Check for auto-approval
        if request.status == "pending" and request.can_auto_approve():
            request.status = "approved"
            request.responded_at = datetime.now()
            request.response_by = "auto"
            return "approved"

        # Check for expiration
        if request.status == "pending" and request.is_expired():
            request.status = "expired"
            return "expired"

        return request.status

    async def approve(self, request_id: str, approver: str) -> bool:
        """Approve a pending request."""
        async with self._lock:
            request = self._approval_queue.get(request_id)
            if not request or request.status != "pending":
                return False

            request.status = "approved"
            request.responded_at = datetime.now()
            request.response_by = approver

        logger.info(
            f"Request approved",
            request_id=request_id[:8],
            approver=approver,
        )

        return True

    async def deny(self, request_id: str, approver: str, reason: str = "") -> bool:
        """Deny a pending request."""
        async with self._lock:
            request = self._approval_queue.get(request_id)
            if not request or request.status != "pending":
                return False

            request.status = "denied"
            request.responded_at = datetime.now()
            request.response_by = approver
            request.context["denial_reason"] = reason

        logger.info(
            f"Request denied",
            request_id=request_id[:8],
            approver=approver,
            reason=reason,
        )

        return True

    def emergency_stop(self, reason: str = "Manual emergency stop") -> None:
        """Activate emergency stop."""
        self._emergency_stop = True
        self._emergency_stop_reason = reason
        logger.critical("EMERGENCY STOP ACTIVATED", reason=reason)

    def clear_emergency_stop(self) -> None:
        """Clear emergency stop."""
        self._emergency_stop = False
        self._emergency_stop_reason = None
        logger.info("Emergency stop cleared")

    def is_emergency_stopped(self) -> bool:
        """Check if emergency stop is active."""
        return self._emergency_stop

    def add_rule(self, rule: SafetyRule) -> None:
        """Add a safety rule."""
        self._rules[rule.id] = rule
        logger.info(f"Safety rule added: {rule.name}")

    def remove_rule(self, rule_id: str) -> bool:
        """Remove a safety rule."""
        if rule_id in self._rules:
            del self._rules[rule_id]
            logger.info(f"Safety rule removed: {rule_id}")
            return True
        return False

    def enable_rule(self, rule_id: str) -> bool:
        """Enable a safety rule."""
        if rule_id in self._rules:
            self._rules[rule_id].enabled = True
            return True
        return False

    def disable_rule(self, rule_id: str) -> bool:
        """Disable a safety rule."""
        if rule_id in self._rules:
            self._rules[rule_id].enabled = False
            return True
        return False

    def get_rules(self) -> list[SafetyRule]:
        """Get all safety rules."""
        return list(self._rules.values())

    def get_pending_approvals(self) -> list[ApprovalRequest]:
        """Get all pending approval requests."""
        return [
            r for r in self._approval_queue.values()
            if r.status == "pending" and not r.is_expired()
        ]

    def get_blocked_actions(self, limit: int = 100) -> list[dict]:
        """Get recently blocked actions."""
        return self._blocked_actions[-limit:]

    def get_violations(
        self,
        goal_id: Optional[str] = None,
        severity: Optional[str] = None,
        limit: int = 100,
    ) -> list[SafetyViolation]:
        """Get safety violations."""
        violations = self._violations

        if goal_id:
            violations = [v for v in violations if v.goal_id == goal_id]

        if severity:
            violations = [v for v in violations if v.severity == severity]

        return violations[-limit:]

    def get_stats(self) -> dict[str, Any]:
        """Get safety statistics."""
        rule_stats = {}
        for rule in self._rules.values():
            rule_stats[rule.id] = {
                "name": rule.name,
                "enabled": rule.enabled,
                "triggered_count": rule.triggered_count,
            }

        return {
            "emergency_stop": self._emergency_stop,
            "emergency_stop_reason": self._emergency_stop_reason,
            "total_rules": len(self._rules),
            "enabled_rules": len([r for r in self._rules.values() if r.enabled]),
            "pending_approvals": len(self.get_pending_approvals()),
            "total_violations": len(self._violations),
            "blocked_actions": len(self._blocked_actions),
            "rules": rule_stats,
        }

    def on_approval_request(
        self, callback: Callable[[ApprovalRequest], None]
    ) -> None:
        """Register callback for approval requests."""
        self._approval_callbacks.append(callback)

    def on_violation(
        self, callback: Callable[[SafetyViolation], None]
    ) -> None:
        """Register callback for violations."""
        self._violation_callbacks.append(callback)
