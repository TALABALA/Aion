"""
Context-Aware Access Control for Zero Trust.

Implements dynamic, context-aware access decisions based on:
- User identity and role
- Device trust level
- Network context
- Resource sensitivity
- Real-time risk assessment
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

import structlog

from aion.security.zero_trust.verification import VerificationResult
from aion.security.zero_trust.device_trust import DeviceTrustScore

logger = structlog.get_logger()


class AccessDecisionType(str, Enum):
    """Types of access decisions."""
    ALLOW = "allow"
    DENY = "deny"
    ALLOW_WITH_CONDITIONS = "allow_with_conditions"
    REQUIRE_STEP_UP = "require_step_up"
    REQUIRE_APPROVAL = "require_approval"


class AccessCondition(str, Enum):
    """Conditions that can be applied to access."""
    MFA_REQUIRED = "mfa_required"
    TRUSTED_DEVICE_REQUIRED = "trusted_device_required"
    COMPLIANT_DEVICE_REQUIRED = "compliant_device_required"
    CORPORATE_NETWORK_REQUIRED = "corporate_network_required"
    RECENT_AUTH_REQUIRED = "recent_auth_required"
    READ_ONLY = "read_only"
    TIME_LIMITED = "time_limited"
    MONITORED = "monitored"
    APPROVAL_REQUIRED = "approval_required"


@dataclass
class AccessDecision:
    """Result of an access decision."""
    decision: AccessDecisionType
    allowed: bool
    conditions: list[AccessCondition]
    reason: str
    resource: str
    action: str
    context_factors: dict[str, Any]
    expires_at: Optional[float] = None
    audit_id: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision.value,
            "allowed": self.allowed,
            "conditions": [c.value for c in self.conditions],
            "reason": self.reason,
            "resource": self.resource,
            "action": self.action,
            "expires_at": self.expires_at,
        }


@dataclass
class AccessPolicy:
    """Policy defining access requirements for a resource."""
    name: str
    resource_pattern: str  # Glob pattern for matching resources
    actions: list[str]  # Allowed actions (read, write, delete, etc.)

    # Requirements
    min_trust_score: float = 0.0
    min_device_trust: float = 0.0
    require_mfa: bool = False
    require_trusted_device: bool = False
    require_compliant_device: bool = False
    require_corporate_network: bool = False
    max_auth_age_seconds: int = 86400
    max_risk_score: float = 70.0

    # Conditions to apply when allowing
    apply_conditions: list[AccessCondition] = field(default_factory=list)

    # Time restrictions
    allowed_hours: Optional[list[int]] = None  # Hours 0-23
    allowed_days: Optional[list[int]] = None   # Days 0-6 (Mon-Sun)

    # Network restrictions
    allowed_ip_ranges: Optional[list[str]] = None
    blocked_ip_ranges: Optional[list[str]] = None
    allowed_countries: Optional[list[str]] = None
    blocked_countries: Optional[list[str]] = None

    # User restrictions
    allowed_roles: Optional[list[str]] = None
    allowed_groups: Optional[list[str]] = None

    # Sensitivity
    sensitivity_level: str = "normal"  # low, normal, high, critical

    # Enabled
    enabled: bool = True


@dataclass
class AccessContext:
    """Context for access decision."""
    # User info
    user_id: str
    roles: list[str]
    groups: list[str]

    # Request info
    resource: str
    action: str
    ip_address: str

    # Verification results
    verification_result: Optional[VerificationResult] = None
    device_trust_score: Optional[DeviceTrustScore] = None

    # Risk score
    risk_score: float = 0.0

    # Session info
    auth_method: str = ""
    mfa_verified: bool = False
    auth_time: float = 0.0

    # Network info
    is_corporate_network: bool = False
    country_code: Optional[str] = None

    # Additional context
    metadata: dict[str, Any] = field(default_factory=dict)


class ContextAwareAccessController:
    """
    Context-aware access controller for Zero Trust.

    Makes dynamic access decisions based on multiple context factors
    including identity, device, network, and real-time risk.
    """

    def __init__(
        self,
        default_deny: bool = True,
        audit_callback: Optional[Callable[[AccessDecision], None]] = None,
    ) -> None:
        self.default_deny = default_deny
        self.audit_callback = audit_callback
        self._policies: list[AccessPolicy] = []
        self._logger = logger.bind(component="access_controller")

    def add_policy(self, policy: AccessPolicy) -> None:
        """Add an access policy."""
        self._policies.append(policy)
        # Sort by specificity (longer patterns first)
        self._policies.sort(key=lambda p: len(p.resource_pattern), reverse=True)

    def remove_policy(self, name: str) -> bool:
        """Remove a policy by name."""
        initial_count = len(self._policies)
        self._policies = [p for p in self._policies if p.name != name]
        return len(self._policies) < initial_count

    def check_access(self, context: AccessContext) -> AccessDecision:
        """
        Check if access should be allowed.

        Args:
            context: Access context with all relevant information

        Returns:
            AccessDecision with the result
        """
        # Find matching policies
        matching_policies = self._find_matching_policies(context.resource, context.action)

        if not matching_policies:
            if self.default_deny:
                return self._create_decision(
                    AccessDecisionType.DENY,
                    context,
                    "No matching policy found",
                )
            else:
                return self._create_decision(
                    AccessDecisionType.ALLOW,
                    context,
                    "Default allow - no policy matched",
                )

        # Evaluate each matching policy
        for policy in matching_policies:
            if not policy.enabled:
                continue

            decision = self._evaluate_policy(policy, context)
            if decision.decision != AccessDecisionType.DENY:
                # First non-deny decision wins
                self._audit_decision(decision)
                return decision

        # All policies denied
        decision = self._create_decision(
            AccessDecisionType.DENY,
            context,
            "Access denied by policy",
        )
        self._audit_decision(decision)
        return decision

    def _find_matching_policies(
        self,
        resource: str,
        action: str,
    ) -> list[AccessPolicy]:
        """Find policies matching the resource and action."""
        matching: list[AccessPolicy] = []

        for policy in self._policies:
            if self._pattern_matches(policy.resource_pattern, resource):
                if action in policy.actions or "*" in policy.actions:
                    matching.append(policy)

        return matching

    def _pattern_matches(self, pattern: str, resource: str) -> bool:
        """Check if pattern matches resource (simple glob)."""
        if pattern == "*":
            return True

        if pattern.endswith("/*"):
            prefix = pattern[:-2]
            return resource.startswith(prefix)

        if pattern.endswith("/**"):
            prefix = pattern[:-3]
            return resource.startswith(prefix)

        return pattern == resource

    def _evaluate_policy(
        self,
        policy: AccessPolicy,
        context: AccessContext,
    ) -> AccessDecision:
        """Evaluate a single policy against context."""
        conditions: list[AccessCondition] = list(policy.apply_conditions)
        reasons: list[str] = []

        # Check verification result
        if context.verification_result:
            if not context.verification_result.verified:
                return self._create_decision(
                    AccessDecisionType.DENY,
                    context,
                    "Verification failed",
                    policy,
                )

            if context.verification_result.overall_confidence < policy.min_trust_score:
                if context.verification_result.requires_step_up:
                    return self._create_decision(
                        AccessDecisionType.REQUIRE_STEP_UP,
                        context,
                        "Insufficient trust score, step-up required",
                        policy,
                    )
                reasons.append("Low trust score")

        # Check device trust
        if context.device_trust_score:
            if context.device_trust_score.trust_score < policy.min_device_trust:
                reasons.append("Insufficient device trust")

            if policy.require_trusted_device and not context.device_trust_score.is_trusted:
                return self._create_decision(
                    AccessDecisionType.DENY,
                    context,
                    "Trusted device required",
                    policy,
                )

            if policy.require_compliant_device and not context.device_trust_score.is_compliant:
                return self._create_decision(
                    AccessDecisionType.DENY,
                    context,
                    "Compliant device required",
                    policy,
                )

        # Check MFA
        if policy.require_mfa and not context.mfa_verified:
            conditions.append(AccessCondition.MFA_REQUIRED)
            return self._create_decision(
                AccessDecisionType.REQUIRE_STEP_UP,
                context,
                "MFA required",
                policy,
                conditions,
            )

        # Check auth age
        auth_age = time.time() - context.auth_time
        if auth_age > policy.max_auth_age_seconds:
            conditions.append(AccessCondition.RECENT_AUTH_REQUIRED)
            return self._create_decision(
                AccessDecisionType.REQUIRE_STEP_UP,
                context,
                "Recent authentication required",
                policy,
                conditions,
            )

        # Check risk score
        if context.risk_score > policy.max_risk_score:
            return self._create_decision(
                AccessDecisionType.DENY,
                context,
                f"Risk score too high: {context.risk_score}",
                policy,
            )

        # Check network requirements
        if policy.require_corporate_network and not context.is_corporate_network:
            return self._create_decision(
                AccessDecisionType.DENY,
                context,
                "Corporate network required",
                policy,
            )

        # Check allowed IP ranges
        if policy.allowed_ip_ranges:
            if not self._ip_in_ranges(context.ip_address, policy.allowed_ip_ranges):
                return self._create_decision(
                    AccessDecisionType.DENY,
                    context,
                    "IP address not in allowed range",
                    policy,
                )

        # Check blocked IP ranges
        if policy.blocked_ip_ranges:
            if self._ip_in_ranges(context.ip_address, policy.blocked_ip_ranges):
                return self._create_decision(
                    AccessDecisionType.DENY,
                    context,
                    "IP address blocked",
                    policy,
                )

        # Check country restrictions
        if policy.allowed_countries and context.country_code:
            if context.country_code not in policy.allowed_countries:
                return self._create_decision(
                    AccessDecisionType.DENY,
                    context,
                    f"Country not allowed: {context.country_code}",
                    policy,
                )

        if policy.blocked_countries and context.country_code:
            if context.country_code in policy.blocked_countries:
                return self._create_decision(
                    AccessDecisionType.DENY,
                    context,
                    f"Country blocked: {context.country_code}",
                    policy,
                )

        # Check time restrictions
        if policy.allowed_hours or policy.allowed_days:
            current_time = time.localtime()
            if policy.allowed_hours and current_time.tm_hour not in policy.allowed_hours:
                return self._create_decision(
                    AccessDecisionType.DENY,
                    context,
                    "Access not allowed at this time",
                    policy,
                )
            if policy.allowed_days and current_time.tm_wday not in policy.allowed_days:
                return self._create_decision(
                    AccessDecisionType.DENY,
                    context,
                    "Access not allowed on this day",
                    policy,
                )

        # Check role restrictions
        if policy.allowed_roles:
            if not any(role in policy.allowed_roles for role in context.roles):
                return self._create_decision(
                    AccessDecisionType.DENY,
                    context,
                    "Role not authorized",
                    policy,
                )

        # Check group restrictions
        if policy.allowed_groups:
            if not any(group in policy.allowed_groups for group in context.groups):
                return self._create_decision(
                    AccessDecisionType.DENY,
                    context,
                    "Group not authorized",
                    policy,
                )

        # All checks passed
        if conditions:
            return self._create_decision(
                AccessDecisionType.ALLOW_WITH_CONDITIONS,
                context,
                "Access allowed with conditions",
                policy,
                conditions,
            )

        return self._create_decision(
            AccessDecisionType.ALLOW,
            context,
            "Access allowed",
            policy,
        )

    def _ip_in_ranges(self, ip: str, ranges: list[str]) -> bool:
        """Check if IP is in any of the specified ranges."""
        import ipaddress

        try:
            ip_addr = ipaddress.ip_address(ip)
            for range_str in ranges:
                try:
                    network = ipaddress.ip_network(range_str, strict=False)
                    if ip_addr in network:
                        return True
                except ValueError:
                    continue
        except ValueError:
            return False

        return False

    def _create_decision(
        self,
        decision_type: AccessDecisionType,
        context: AccessContext,
        reason: str,
        policy: Optional[AccessPolicy] = None,
        conditions: Optional[list[AccessCondition]] = None,
    ) -> AccessDecision:
        """Create an access decision."""
        allowed = decision_type in (
            AccessDecisionType.ALLOW,
            AccessDecisionType.ALLOW_WITH_CONDITIONS,
        )

        expires_at = None
        if allowed and policy and policy.sensitivity_level in ("high", "critical"):
            # Time-limited access for sensitive resources
            expires_at = time.time() + 3600  # 1 hour

        return AccessDecision(
            decision=decision_type,
            allowed=allowed,
            conditions=conditions or [],
            reason=reason,
            resource=context.resource,
            action=context.action,
            context_factors={
                "user_id": context.user_id,
                "roles": context.roles,
                "ip_address": context.ip_address,
                "risk_score": context.risk_score,
                "mfa_verified": context.mfa_verified,
                "policy": policy.name if policy else None,
            },
            expires_at=expires_at,
            audit_id=f"access-{int(time.time() * 1000)}",
        )

    def _audit_decision(self, decision: AccessDecision) -> None:
        """Audit an access decision."""
        if self.audit_callback:
            try:
                self.audit_callback(decision)
            except Exception as e:
                self._logger.error("Audit callback failed", error=str(e))

        self._logger.info(
            "Access decision made",
            decision=decision.decision.value,
            resource=decision.resource,
            action=decision.action,
            allowed=decision.allowed,
        )
