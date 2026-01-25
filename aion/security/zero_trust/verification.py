"""
Zero Trust Continuous Verification.

Implements continuous verification of identity, device, and context
throughout the session lifecycle, not just at authentication time.

Key principles:
- Verify every request
- Re-authenticate on context changes
- Monitor for anomalies
- Enforce adaptive policies
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

import structlog

from aion.security.adaptive.fingerprint import DeviceFingerprint
from aion.security.adaptive.risk import RiskAssessment, RiskContext, RiskEngine, RiskLevel

logger = structlog.get_logger()


class VerificationOutcome(str, Enum):
    """Outcome of a verification check."""
    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"
    REQUIRE_STEP_UP = "require_step_up"
    REQUIRE_REAUTHENTICATION = "require_reauthentication"


class VerificationType(str, Enum):
    """Types of verification checks."""
    IDENTITY = "identity"
    DEVICE = "device"
    NETWORK = "network"
    CONTEXT = "context"
    BEHAVIOR = "behavior"
    POLICY = "policy"


class ContextChangeType(str, Enum):
    """Types of context changes that trigger re-verification."""
    IP_CHANGE = "ip_change"
    DEVICE_CHANGE = "device_change"
    LOCATION_CHANGE = "location_change"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    SENSITIVE_OPERATION = "sensitive_operation"
    SESSION_TIMEOUT = "session_timeout"
    ANOMALY_DETECTED = "anomaly_detected"


@dataclass
class VerificationCheck:
    """Individual verification check result."""
    check_type: VerificationType
    outcome: VerificationOutcome
    confidence: float  # 0.0 to 1.0
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    requires_action: Optional[str] = None


@dataclass
class VerificationResult:
    """Complete verification result."""
    verified: bool
    checks: list[VerificationCheck]
    overall_confidence: float
    requires_step_up: bool
    requires_reauthentication: bool
    context_changes: list[ContextChangeType]
    restrictions: dict[str, Any]
    valid_until: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def passed_checks(self) -> list[VerificationCheck]:
        return [c for c in self.checks if c.outcome == VerificationOutcome.PASS]

    @property
    def failed_checks(self) -> list[VerificationCheck]:
        return [c for c in self.checks if c.outcome == VerificationOutcome.FAIL]

    def to_dict(self) -> dict[str, Any]:
        return {
            "verified": self.verified,
            "overall_confidence": self.overall_confidence,
            "requires_step_up": self.requires_step_up,
            "requires_reauthentication": self.requires_reauthentication,
            "context_changes": [c.value for c in self.context_changes],
            "restrictions": self.restrictions,
            "valid_until": self.valid_until,
            "checks": [
                {
                    "type": c.check_type.value,
                    "outcome": c.outcome.value,
                    "confidence": c.confidence,
                }
                for c in self.checks
            ],
        }


@dataclass
class VerificationContext:
    """Context for verification."""
    # Session info
    session_id: str
    user_id: str
    tenant_id: Optional[str] = None

    # Request info
    ip_address: str = ""
    user_agent: str = ""
    request_path: str = ""
    request_method: str = ""

    # Device info
    device_fingerprint: Optional[DeviceFingerprint] = None

    # Location info
    geo_location: Optional[dict[str, Any]] = None

    # Session state
    authenticated_at: float = 0.0
    last_activity_at: float = 0.0
    activity_count: int = 0
    auth_methods_used: list[str] = field(default_factory=list)
    mfa_verified: bool = False

    # Previous context (for change detection)
    previous_ip: Optional[str] = None
    previous_device_id: Optional[str] = None
    previous_location: Optional[dict[str, Any]] = None

    # Operation info
    operation: str = ""
    operation_sensitivity: str = "normal"
    requested_resources: list[str] = field(default_factory=list)
    requested_permissions: list[str] = field(default_factory=list)

    # Additional signals
    signals: dict[str, Any] = field(default_factory=dict)


@dataclass
class VerificationPolicy:
    """Policy for verification requirements."""
    name: str

    # Time-based requirements
    max_session_age: int = 86400  # 24 hours
    max_idle_time: int = 3600  # 1 hour
    re_verify_interval: int = 900  # 15 minutes

    # Context change triggers
    re_verify_on_ip_change: bool = True
    re_verify_on_device_change: bool = True
    re_verify_on_location_change: bool = True

    # Sensitivity requirements
    require_mfa_for_sensitive: bool = True
    require_recent_auth_for_critical: bool = True
    max_auth_age_for_critical: int = 300  # 5 minutes

    # Confidence thresholds
    min_confidence_threshold: float = 0.7
    require_step_up_threshold: float = 0.5

    # Enabled checks
    enabled_checks: list[VerificationType] = field(default_factory=lambda: [
        VerificationType.IDENTITY,
        VerificationType.DEVICE,
        VerificationType.NETWORK,
        VerificationType.CONTEXT,
        VerificationType.BEHAVIOR,
        VerificationType.POLICY,
    ])


class ContinuousVerifier:
    """
    Continuous verification engine.

    Implements Zero Trust by continuously verifying identity, device,
    and context throughout the session lifecycle.
    """

    def __init__(
        self,
        risk_engine: RiskEngine,
        default_policy: Optional[VerificationPolicy] = None,
        alert_callback: Optional[Callable[[str, VerificationResult], None]] = None,
    ) -> None:
        self.risk_engine = risk_engine
        self.default_policy = default_policy or VerificationPolicy(name="default")
        self.alert_callback = alert_callback
        self._policies: dict[str, VerificationPolicy] = {"default": self.default_policy}
        self._session_state: dict[str, SessionState] = {}
        self._logger = logger.bind(component="continuous_verifier")

    def add_policy(self, policy: VerificationPolicy) -> None:
        """Add a verification policy."""
        self._policies[policy.name] = policy

    async def verify(
        self,
        context: VerificationContext,
        policy_name: Optional[str] = None,
    ) -> VerificationResult:
        """
        Perform continuous verification.

        Args:
            context: The verification context
            policy_name: Name of policy to use (default if not specified)

        Returns:
            VerificationResult with all check outcomes
        """
        policy = self._policies.get(policy_name or "default", self.default_policy)
        checks: list[VerificationCheck] = []
        context_changes: list[ContextChangeType] = []

        # Get or create session state
        session_state = self._get_or_create_session_state(context)

        # Detect context changes
        context_changes = self._detect_context_changes(context, session_state, policy)

        # Run enabled verification checks
        if VerificationType.IDENTITY in policy.enabled_checks:
            checks.append(await self._verify_identity(context, session_state, policy))

        if VerificationType.DEVICE in policy.enabled_checks:
            checks.append(await self._verify_device(context, session_state, policy))

        if VerificationType.NETWORK in policy.enabled_checks:
            checks.append(await self._verify_network(context, session_state, policy))

        if VerificationType.CONTEXT in policy.enabled_checks:
            checks.append(await self._verify_context(context, session_state, policy, context_changes))

        if VerificationType.BEHAVIOR in policy.enabled_checks:
            checks.append(await self._verify_behavior(context, session_state, policy))

        if VerificationType.POLICY in policy.enabled_checks:
            checks.append(await self._verify_policy(context, session_state, policy))

        # Calculate overall result
        result = self._calculate_result(checks, context_changes, policy)

        # Update session state
        self._update_session_state(context, session_state, result)

        # Alert on significant issues
        if result.requires_reauthentication or any(
            c.outcome == VerificationOutcome.FAIL for c in checks
        ):
            if self.alert_callback:
                try:
                    self.alert_callback(context.session_id, result)
                except Exception as e:
                    self._logger.error("Alert callback failed", error=str(e))

        self._logger.info(
            "Verification completed",
            session_id=context.session_id,
            verified=result.verified,
            confidence=result.overall_confidence,
            context_changes=[c.value for c in context_changes],
        )

        return result

    def _get_or_create_session_state(self, context: VerificationContext) -> "SessionState":
        """Get or create session state."""
        if context.session_id not in self._session_state:
            self._session_state[context.session_id] = SessionState(
                session_id=context.session_id,
                user_id=context.user_id,
                created_at=time.time(),
                initial_ip=context.ip_address,
                initial_device_id=context.device_fingerprint.fingerprint_id if context.device_fingerprint else None,
            )
        return self._session_state[context.session_id]

    def _detect_context_changes(
        self,
        context: VerificationContext,
        session_state: "SessionState",
        policy: VerificationPolicy,
    ) -> list[ContextChangeType]:
        """Detect context changes since last verification."""
        changes: list[ContextChangeType] = []

        # IP change
        if context.ip_address and session_state.last_ip:
            if context.ip_address != session_state.last_ip:
                changes.append(ContextChangeType.IP_CHANGE)

        # Device change
        if context.device_fingerprint:
            current_device_id = context.device_fingerprint.fingerprint_id
            if session_state.last_device_id and current_device_id != session_state.last_device_id:
                changes.append(ContextChangeType.DEVICE_CHANGE)

        # Location change
        if context.geo_location and session_state.last_location:
            current_country = context.geo_location.get("country_code")
            last_country = session_state.last_location.get("country_code")
            if current_country and last_country and current_country != last_country:
                changes.append(ContextChangeType.LOCATION_CHANGE)

        # Privilege escalation
        if context.operation_sensitivity in ("high", "critical"):
            changes.append(ContextChangeType.PRIVILEGE_ESCALATION)

        # Sensitive operation
        if context.operation_sensitivity == "critical":
            changes.append(ContextChangeType.SENSITIVE_OPERATION)

        # Session timeout approaching
        now = time.time()
        if now - context.last_activity_at > policy.max_idle_time * 0.9:
            changes.append(ContextChangeType.SESSION_TIMEOUT)

        return changes

    async def _verify_identity(
        self,
        context: VerificationContext,
        session_state: "SessionState",
        policy: VerificationPolicy,
    ) -> VerificationCheck:
        """Verify identity is still valid."""
        now = time.time()
        confidence = 1.0
        outcome = VerificationOutcome.PASS
        requires_action = None
        details: dict[str, Any] = {}

        # Check authentication age
        auth_age = now - context.authenticated_at
        if auth_age > policy.max_session_age:
            confidence = 0.0
            outcome = VerificationOutcome.REQUIRE_REAUTHENTICATION
            requires_action = "Session expired, re-authentication required"
        elif auth_age > policy.max_session_age * 0.9:
            confidence = 0.3
            outcome = VerificationOutcome.WARN
            details["warning"] = "Session nearing expiration"

        # Check idle time
        idle_time = now - context.last_activity_at
        if idle_time > policy.max_idle_time:
            confidence = 0.0
            outcome = VerificationOutcome.REQUIRE_REAUTHENTICATION
            requires_action = "Session idle timeout, re-authentication required"

        # Check MFA for sensitive operations
        if context.operation_sensitivity in ("high", "critical"):
            if not context.mfa_verified:
                if outcome == VerificationOutcome.PASS:
                    outcome = VerificationOutcome.REQUIRE_STEP_UP
                    requires_action = "MFA required for this operation"
                    confidence *= 0.5

        # Check recent auth for critical operations
        if context.operation_sensitivity == "critical":
            if auth_age > policy.max_auth_age_for_critical:
                if outcome == VerificationOutcome.PASS:
                    outcome = VerificationOutcome.REQUIRE_REAUTHENTICATION
                    requires_action = "Recent authentication required for critical operations"
                    confidence = 0.2

        return VerificationCheck(
            check_type=VerificationType.IDENTITY,
            outcome=outcome,
            confidence=confidence,
            details=details,
            requires_action=requires_action,
        )

    async def _verify_device(
        self,
        context: VerificationContext,
        session_state: "SessionState",
        policy: VerificationPolicy,
    ) -> VerificationCheck:
        """Verify device is trusted."""
        confidence = 0.8
        outcome = VerificationOutcome.PASS
        requires_action = None
        details: dict[str, Any] = {}

        if not context.device_fingerprint:
            # No device fingerprint - reduced trust
            confidence = 0.5
            outcome = VerificationOutcome.WARN
            details["warning"] = "Device fingerprint not available"
            return VerificationCheck(
                check_type=VerificationType.DEVICE,
                outcome=outcome,
                confidence=confidence,
                details=details,
            )

        fp = context.device_fingerprint

        # Check for suspicious indicators
        if fp.is_suspicious:
            confidence *= 0.5
            if len(fp.risk_indicators) >= 3:
                outcome = VerificationOutcome.FAIL
                details["risk_indicators"] = fp.risk_indicators
            else:
                outcome = VerificationOutcome.WARN
                details["risk_indicators"] = fp.risk_indicators

        # Check if device changed mid-session
        if session_state.initial_device_id:
            if fp.fingerprint_id != session_state.initial_device_id:
                confidence *= 0.3
                if policy.re_verify_on_device_change:
                    outcome = VerificationOutcome.REQUIRE_STEP_UP
                    requires_action = "Device changed, step-up authentication required"
                    details["device_change"] = True

        # Check device trust level
        trust_scores = {
            "unknown": 0.5,
            "low": 0.6,
            "medium": 0.8,
            "high": 0.95,
            "trusted": 1.0,
        }
        confidence *= trust_scores.get(fp.trust_level.value, 0.5)

        return VerificationCheck(
            check_type=VerificationType.DEVICE,
            outcome=outcome,
            confidence=confidence,
            details=details,
            requires_action=requires_action,
        )

    async def _verify_network(
        self,
        context: VerificationContext,
        session_state: "SessionState",
        policy: VerificationPolicy,
    ) -> VerificationCheck:
        """Verify network context."""
        confidence = 0.9
        outcome = VerificationOutcome.PASS
        requires_action = None
        details: dict[str, Any] = {}

        # Check for IP change
        if session_state.initial_ip and context.ip_address != session_state.initial_ip:
            confidence *= 0.6
            details["ip_changed"] = True

            if policy.re_verify_on_ip_change:
                outcome = VerificationOutcome.REQUIRE_STEP_UP
                requires_action = "IP address changed, verification required"

        # Check geo location
        if context.geo_location:
            geo = context.geo_location

            if geo.get("is_vpn"):
                confidence *= 0.7
                details["vpn_detected"] = True

            if geo.get("is_tor"):
                confidence *= 0.3
                details["tor_detected"] = True
                outcome = VerificationOutcome.WARN

            if geo.get("is_datacenter"):
                confidence *= 0.8
                details["datacenter_ip"] = True

        return VerificationCheck(
            check_type=VerificationType.NETWORK,
            outcome=outcome,
            confidence=confidence,
            details=details,
            requires_action=requires_action,
        )

    async def _verify_context(
        self,
        context: VerificationContext,
        session_state: "SessionState",
        policy: VerificationPolicy,
        context_changes: list[ContextChangeType],
    ) -> VerificationCheck:
        """Verify overall context consistency."""
        confidence = 1.0
        outcome = VerificationOutcome.PASS
        requires_action = None
        details: dict[str, Any] = {"changes": [c.value for c in context_changes]}

        # Penalize for each context change
        change_penalties = {
            ContextChangeType.IP_CHANGE: 0.15,
            ContextChangeType.DEVICE_CHANGE: 0.25,
            ContextChangeType.LOCATION_CHANGE: 0.20,
            ContextChangeType.PRIVILEGE_ESCALATION: 0.10,
            ContextChangeType.SENSITIVE_OPERATION: 0.05,
            ContextChangeType.SESSION_TIMEOUT: 0.30,
            ContextChangeType.ANOMALY_DETECTED: 0.40,
        }

        for change in context_changes:
            confidence -= change_penalties.get(change, 0.1)

        confidence = max(0.0, confidence)

        # Determine outcome based on confidence
        if confidence < 0.3:
            outcome = VerificationOutcome.REQUIRE_REAUTHENTICATION
            requires_action = "Significant context changes detected"
        elif confidence < 0.5:
            outcome = VerificationOutcome.REQUIRE_STEP_UP
            requires_action = "Context changes require additional verification"
        elif confidence < 0.7:
            outcome = VerificationOutcome.WARN

        return VerificationCheck(
            check_type=VerificationType.CONTEXT,
            outcome=outcome,
            confidence=confidence,
            details=details,
            requires_action=requires_action,
        )

    async def _verify_behavior(
        self,
        context: VerificationContext,
        session_state: "SessionState",
        policy: VerificationPolicy,
    ) -> VerificationCheck:
        """Verify behavioral patterns using risk engine."""
        # Build risk context
        risk_context = RiskContext(
            ip_address=context.ip_address,
            user_agent=context.user_agent,
            request_path=context.request_path,
            request_method=context.request_method,
            device_fingerprint=context.device_fingerprint,
            session_id=context.session_id,
            session_age_seconds=time.time() - context.authenticated_at,
            session_activity_count=context.activity_count,
            user_id=context.user_id,
            operation_type=context.operation,
            operation_sensitivity=context.operation_sensitivity,
            signals=context.signals,
        )

        # Run risk assessment
        risk_assessment = self.risk_engine.assess(risk_context)

        # Map risk level to verification outcome
        outcome_map = {
            RiskLevel.MINIMAL: VerificationOutcome.PASS,
            RiskLevel.LOW: VerificationOutcome.PASS,
            RiskLevel.MEDIUM: VerificationOutcome.WARN,
            RiskLevel.HIGH: VerificationOutcome.REQUIRE_STEP_UP,
            RiskLevel.CRITICAL: VerificationOutcome.FAIL,
        }

        outcome = outcome_map[risk_assessment.score.level]
        confidence = 1.0 - (risk_assessment.score.total / 100.0)

        requires_action = None
        if risk_assessment.requires_mfa:
            requires_action = "Risk assessment requires additional verification"

        return VerificationCheck(
            check_type=VerificationType.BEHAVIOR,
            outcome=outcome,
            confidence=confidence,
            details={
                "risk_score": risk_assessment.score.total,
                "risk_level": risk_assessment.score.level.value,
                "top_factors": [f.name for f in risk_assessment.score.top_factors[:3]],
            },
            requires_action=requires_action,
        )

    async def _verify_policy(
        self,
        context: VerificationContext,
        session_state: "SessionState",
        policy: VerificationPolicy,
    ) -> VerificationCheck:
        """Verify against policy requirements."""
        confidence = 1.0
        outcome = VerificationOutcome.PASS
        requires_action = None
        details: dict[str, Any] = {}
        violations: list[str] = []

        now = time.time()

        # Check re-verification interval
        if session_state.last_verification_at:
            time_since_verification = now - session_state.last_verification_at
            if time_since_verification > policy.re_verify_interval:
                confidence *= 0.8
                details["re_verification_needed"] = True

        # Check required auth methods for sensitive operations
        if context.operation_sensitivity in ("high", "critical"):
            if policy.require_mfa_for_sensitive and not context.mfa_verified:
                violations.append("MFA required for sensitive operation")
                confidence *= 0.5

        if violations:
            details["policy_violations"] = violations
            if len(violations) > 1:
                outcome = VerificationOutcome.FAIL
            else:
                outcome = VerificationOutcome.REQUIRE_STEP_UP
            requires_action = "; ".join(violations)

        return VerificationCheck(
            check_type=VerificationType.POLICY,
            outcome=outcome,
            confidence=confidence,
            details=details,
            requires_action=requires_action,
        )

    def _calculate_result(
        self,
        checks: list[VerificationCheck],
        context_changes: list[ContextChangeType],
        policy: VerificationPolicy,
    ) -> VerificationResult:
        """Calculate overall verification result."""
        # Calculate overall confidence (weighted average)
        if checks:
            total_confidence = sum(c.confidence for c in checks) / len(checks)
        else:
            total_confidence = 0.0

        # Determine if verified
        failed_checks = [c for c in checks if c.outcome == VerificationOutcome.FAIL]
        verified = len(failed_checks) == 0 and total_confidence >= policy.min_confidence_threshold

        # Check for step-up requirements
        requires_step_up = any(
            c.outcome == VerificationOutcome.REQUIRE_STEP_UP
            for c in checks
        ) or total_confidence < policy.require_step_up_threshold

        # Check for reauthentication requirements
        requires_reauthentication = any(
            c.outcome == VerificationOutcome.REQUIRE_REAUTHENTICATION
            for c in checks
        )

        # Build restrictions
        restrictions: dict[str, Any] = {}
        if not verified or requires_step_up:
            restrictions["sensitive_operations_blocked"] = True
        if total_confidence < 0.5:
            restrictions["reduced_session_duration"] = True
            restrictions["max_session_duration"] = 1800  # 30 minutes

        # Calculate validity duration based on confidence
        base_validity = 900  # 15 minutes
        validity_multiplier = max(0.5, total_confidence)
        validity_duration = int(base_validity * validity_multiplier)

        return VerificationResult(
            verified=verified,
            checks=checks,
            overall_confidence=total_confidence,
            requires_step_up=requires_step_up,
            requires_reauthentication=requires_reauthentication,
            context_changes=context_changes,
            restrictions=restrictions,
            valid_until=time.time() + validity_duration,
        )

    def _update_session_state(
        self,
        context: VerificationContext,
        session_state: "SessionState",
        result: VerificationResult,
    ) -> None:
        """Update session state after verification."""
        now = time.time()

        session_state.last_verification_at = now
        session_state.last_ip = context.ip_address
        session_state.last_device_id = (
            context.device_fingerprint.fingerprint_id
            if context.device_fingerprint else None
        )
        session_state.last_location = context.geo_location
        session_state.verification_count += 1
        session_state.last_confidence = result.overall_confidence

        if not result.verified:
            session_state.failed_verification_count += 1

    def get_session_state(self, session_id: str) -> Optional["SessionState"]:
        """Get session state for monitoring."""
        return self._session_state.get(session_id)

    def invalidate_session(self, session_id: str) -> bool:
        """Invalidate a session."""
        if session_id in self._session_state:
            del self._session_state[session_id]
            return True
        return False


@dataclass
class SessionState:
    """Internal session state for continuous verification."""
    session_id: str
    user_id: str
    created_at: float

    # Initial context (baseline)
    initial_ip: Optional[str] = None
    initial_device_id: Optional[str] = None

    # Last known context
    last_ip: Optional[str] = None
    last_device_id: Optional[str] = None
    last_location: Optional[dict[str, Any]] = None

    # Verification history
    last_verification_at: Optional[float] = None
    verification_count: int = 0
    failed_verification_count: int = 0
    last_confidence: float = 1.0

    # Risk tracking
    accumulated_risk_score: float = 0.0
    context_change_count: int = 0


class ZeroTrustEnforcer:
    """
    Enforcement layer for Zero Trust policies.

    Integrates with continuous verification to enforce
    access decisions in real-time.
    """

    def __init__(
        self,
        verifier: ContinuousVerifier,
        block_on_failure: bool = True,
        allow_degraded_mode: bool = False,
    ) -> None:
        self.verifier = verifier
        self.block_on_failure = block_on_failure
        self.allow_degraded_mode = allow_degraded_mode
        self._logger = logger.bind(component="zero_trust_enforcer")

    async def enforce(
        self,
        context: VerificationContext,
        policy_name: Optional[str] = None,
    ) -> tuple[bool, Optional[str], dict[str, Any]]:
        """
        Enforce Zero Trust for a request.

        Returns:
            Tuple of (allowed, action_required, metadata)
        """
        try:
            result = await self.verifier.verify(context, policy_name)

            if result.requires_reauthentication:
                return (False, "reauthenticate", result.to_dict())

            if result.requires_step_up:
                return (False, "step_up", result.to_dict())

            if not result.verified:
                if self.block_on_failure:
                    return (False, "blocked", result.to_dict())
                elif self.allow_degraded_mode:
                    return (True, "degraded", {
                        **result.to_dict(),
                        "degraded_mode": True,
                    })
                else:
                    return (False, "blocked", result.to_dict())

            return (True, None, result.to_dict())

        except Exception as e:
            self._logger.error("Enforcement error", error=str(e))

            if self.allow_degraded_mode:
                return (True, "error_degraded", {"error": str(e)})
            return (False, "error", {"error": str(e)})
