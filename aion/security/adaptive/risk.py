"""
Risk Scoring Engine.

Implements adaptive authentication through real-time risk assessment.
Evaluates multiple risk factors to determine the appropriate authentication
level and security response.

Risk factors include:
- Device trust
- Location anomalies
- Behavioral patterns
- Session characteristics
- Historical data
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

import structlog

from aion.security.adaptive.fingerprint import (
    DeviceFingerprint,
    DeviceTrustLevel,
    GeoLocation,
)

logger = structlog.get_logger()


class RiskLevel(str, Enum):
    """Risk level classifications."""
    MINIMAL = "minimal"      # Score 0-20: Normal access
    LOW = "low"              # Score 21-40: Normal access with logging
    MEDIUM = "medium"        # Score 41-60: Additional verification recommended
    HIGH = "high"            # Score 61-80: MFA required
    CRITICAL = "critical"    # Score 81-100: Block or manual review


class RiskCategory(str, Enum):
    """Categories of risk factors."""
    DEVICE = "device"
    LOCATION = "location"
    BEHAVIOR = "behavior"
    SESSION = "session"
    ACCOUNT = "account"
    NETWORK = "network"
    TEMPORAL = "temporal"


class RiskAction(str, Enum):
    """Recommended actions based on risk."""
    ALLOW = "allow"
    ALLOW_WITH_LOGGING = "allow_with_logging"
    REQUIRE_MFA = "require_mfa"
    REQUIRE_STEP_UP = "require_step_up"
    REQUIRE_CAPTCHA = "require_captcha"
    DELAY = "delay"
    BLOCK = "block"
    MANUAL_REVIEW = "manual_review"


@dataclass
class RiskFactor:
    """
    Individual risk factor contributing to overall score.
    """
    name: str
    category: RiskCategory
    score: float  # 0.0 to 1.0
    weight: float  # Importance multiplier
    description: str
    evidence: dict[str, Any] = field(default_factory=dict)
    mitigations: list[str] = field(default_factory=list)

    @property
    def weighted_score(self) -> float:
        return self.score * self.weight


@dataclass
class RiskScore:
    """
    Composite risk score with factor breakdown.
    """
    total: float  # 0-100 scale
    level: RiskLevel
    factors: list[RiskFactor]
    timestamp: float = field(default_factory=time.time)

    @property
    def category_scores(self) -> dict[RiskCategory, float]:
        """Get scores broken down by category."""
        scores: dict[RiskCategory, float] = {}
        for factor in self.factors:
            if factor.category not in scores:
                scores[factor.category] = 0.0
            scores[factor.category] += factor.weighted_score
        return scores

    @property
    def top_factors(self) -> list[RiskFactor]:
        """Get the highest contributing factors."""
        return sorted(self.factors, key=lambda f: f.weighted_score, reverse=True)[:5]


@dataclass
class RiskAssessment:
    """
    Complete risk assessment result.
    """
    score: RiskScore
    actions: list[RiskAction]
    requires_mfa: bool
    block_access: bool
    session_restrictions: dict[str, Any]
    audit_metadata: dict[str, Any]
    expires_at: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": {
                "total": self.score.total,
                "level": self.score.level.value,
                "factors": [
                    {
                        "name": f.name,
                        "category": f.category.value,
                        "score": f.score,
                        "weight": f.weight,
                        "description": f.description,
                    }
                    for f in self.score.factors
                ],
            },
            "actions": [a.value for a in self.actions],
            "requires_mfa": self.requires_mfa,
            "block_access": self.block_access,
            "session_restrictions": self.session_restrictions,
            "expires_at": self.expires_at,
        }


@dataclass
class UserRiskProfile:
    """
    User's historical risk profile.
    """
    user_id: str
    baseline_score: float = 25.0
    known_locations: list[GeoLocation] = field(default_factory=list)
    known_devices: list[str] = field(default_factory=list)
    typical_hours: list[int] = field(default_factory=list)  # Hours of day (0-23)
    typical_days: list[int] = field(default_factory=list)   # Days of week (0-6)
    failed_auth_count: int = 0
    successful_auth_count: int = 0
    last_assessment: Optional[RiskScore] = None
    last_activity_at: Optional[float] = None
    account_age_days: int = 0
    mfa_enabled: bool = False
    email_verified: bool = False
    phone_verified: bool = False
    flags: set[str] = field(default_factory=set)

    @property
    def trust_score(self) -> float:
        """Calculate user trust score (0-100)."""
        score = 50.0  # Base

        # Account verification
        if self.email_verified:
            score += 10
        if self.phone_verified:
            score += 10
        if self.mfa_enabled:
            score += 15

        # Account age
        if self.account_age_days > 365:
            score += 10
        elif self.account_age_days > 90:
            score += 5
        elif self.account_age_days < 7:
            score -= 10

        # Auth history
        if self.successful_auth_count > 100:
            score += 10
        elif self.successful_auth_count > 10:
            score += 5

        if self.failed_auth_count > 10:
            score -= 15
        elif self.failed_auth_count > 5:
            score -= 5

        # Flags
        if "suspicious_activity" in self.flags:
            score -= 20
        if "account_takeover_attempt" in self.flags:
            score -= 30
        if "verified_identity" in self.flags:
            score += 15

        return max(0, min(100, score))


@dataclass
class RiskContext:
    """
    Context for risk assessment.
    """
    # Request context
    ip_address: str
    user_agent: str
    request_path: str
    request_method: str
    headers: dict[str, str] = field(default_factory=dict)

    # Device context
    device_fingerprint: Optional[DeviceFingerprint] = None

    # Location context
    geo_location: Optional[GeoLocation] = None

    # Session context
    session_id: Optional[str] = None
    session_age_seconds: Optional[float] = None
    session_activity_count: int = 0

    # User context
    user_id: Optional[str] = None
    user_profile: Optional[UserRiskProfile] = None

    # Operation context
    operation_type: str = "authentication"
    operation_sensitivity: str = "normal"  # low, normal, high, critical

    # Temporal context
    timestamp: float = field(default_factory=time.time)

    # Additional signals
    signals: dict[str, Any] = field(default_factory=dict)


class RiskEngine:
    """
    Risk assessment engine.

    Evaluates multiple risk factors to produce a risk score and
    recommended security actions.
    """

    # Risk thresholds
    THRESHOLDS = {
        RiskLevel.MINIMAL: (0, 20),
        RiskLevel.LOW: (21, 40),
        RiskLevel.MEDIUM: (41, 60),
        RiskLevel.HIGH: (61, 80),
        RiskLevel.CRITICAL: (81, 100),
    }

    # Default factor weights
    DEFAULT_WEIGHTS = {
        "device_unknown": 0.15,
        "device_suspicious": 0.20,
        "device_trust_low": 0.10,
        "location_new": 0.12,
        "location_impossible_travel": 0.25,
        "location_high_risk": 0.18,
        "location_vpn": 0.08,
        "location_tor": 0.20,
        "behavior_unusual_time": 0.08,
        "behavior_rapid_requests": 0.12,
        "behavior_pattern_anomaly": 0.15,
        "session_new": 0.05,
        "session_long_lived": 0.08,
        "account_new": 0.10,
        "account_unverified": 0.12,
        "account_failed_attempts": 0.15,
        "network_datacenter": 0.10,
        "network_proxy": 0.08,
        "network_outdated_tls": 0.05,
        "operation_sensitive": 0.15,
    }

    def __init__(
        self,
        weights: Optional[dict[str, float]] = None,
        custom_evaluators: Optional[list[Callable[[RiskContext], Optional[RiskFactor]]]] = None,
        assessment_ttl: int = 300,
    ) -> None:
        self.weights = {**self.DEFAULT_WEIGHTS, **(weights or {})}
        self.custom_evaluators = custom_evaluators or []
        self.assessment_ttl = assessment_ttl
        self._user_profiles: dict[str, UserRiskProfile] = {}
        self._location_history: dict[str, list[tuple[GeoLocation, float]]] = {}
        self._logger = logger.bind(component="risk_engine")

    def assess(self, context: RiskContext) -> RiskAssessment:
        """
        Perform risk assessment for the given context.

        Returns a complete risk assessment with score, actions, and restrictions.
        """
        factors: list[RiskFactor] = []

        # Evaluate all risk categories
        factors.extend(self._evaluate_device_risk(context))
        factors.extend(self._evaluate_location_risk(context))
        factors.extend(self._evaluate_behavior_risk(context))
        factors.extend(self._evaluate_session_risk(context))
        factors.extend(self._evaluate_account_risk(context))
        factors.extend(self._evaluate_network_risk(context))
        factors.extend(self._evaluate_temporal_risk(context))
        factors.extend(self._evaluate_operation_risk(context))

        # Run custom evaluators
        for evaluator in self.custom_evaluators:
            try:
                factor = evaluator(context)
                if factor:
                    factors.append(factor)
            except Exception as e:
                self._logger.error("Custom evaluator failed", error=str(e))

        # Calculate total score
        total_weighted = sum(f.weighted_score for f in factors)
        max_possible = sum(f.weight for f in factors) if factors else 1.0
        normalized_score = (total_weighted / max_possible) * 100 if max_possible > 0 else 0

        # Apply baseline adjustment from user profile
        if context.user_profile:
            trust_adjustment = (100 - context.user_profile.trust_score) / 100
            normalized_score = normalized_score * (0.7 + 0.3 * trust_adjustment)

        # Determine risk level
        final_score = min(100, max(0, normalized_score))
        level = self._score_to_level(final_score)

        risk_score = RiskScore(
            total=final_score,
            level=level,
            factors=factors,
        )

        # Determine actions
        actions = self._determine_actions(risk_score, context)

        # Build assessment
        assessment = RiskAssessment(
            score=risk_score,
            actions=actions,
            requires_mfa=RiskAction.REQUIRE_MFA in actions or RiskAction.REQUIRE_STEP_UP in actions,
            block_access=RiskAction.BLOCK in actions,
            session_restrictions=self._determine_session_restrictions(risk_score, context),
            audit_metadata={
                "ip_address": context.ip_address,
                "user_agent": context.user_agent,
                "fingerprint_id": context.device_fingerprint.fingerprint_id if context.device_fingerprint else None,
                "factors_count": len(factors),
                "top_factor": factors[0].name if factors else None,
            },
            expires_at=time.time() + self.assessment_ttl,
        )

        # Update history
        self._update_history(context, assessment)

        self._logger.info(
            "Risk assessment completed",
            score=final_score,
            level=level.value,
            factors_count=len(factors),
            actions=[a.value for a in actions],
        )

        return assessment

    def _evaluate_device_risk(self, context: RiskContext) -> list[RiskFactor]:
        """Evaluate device-related risk factors."""
        factors: list[RiskFactor] = []

        if not context.device_fingerprint:
            factors.append(RiskFactor(
                name="device_unknown",
                category=RiskCategory.DEVICE,
                score=0.6,
                weight=self.weights["device_unknown"],
                description="Device fingerprint not available",
            ))
            return factors

        fp = context.device_fingerprint

        # Check device trust level
        if fp.trust_level == DeviceTrustLevel.UNKNOWN:
            factors.append(RiskFactor(
                name="device_trust_unknown",
                category=RiskCategory.DEVICE,
                score=0.5,
                weight=self.weights["device_trust_low"],
                description="Unknown device with no trust history",
            ))
        elif fp.trust_level == DeviceTrustLevel.LOW:
            factors.append(RiskFactor(
                name="device_trust_low",
                category=RiskCategory.DEVICE,
                score=0.7,
                weight=self.weights["device_trust_low"],
                description="Device has low trust level",
            ))

        # Check risk indicators
        if fp.is_suspicious:
            suspicious_score = min(1.0, len(fp.risk_indicators) * 0.2)
            factors.append(RiskFactor(
                name="device_suspicious",
                category=RiskCategory.DEVICE,
                score=suspicious_score,
                weight=self.weights["device_suspicious"],
                description=f"Suspicious device indicators: {', '.join(fp.risk_indicators)}",
                evidence={"indicators": fp.risk_indicators},
            ))

        # Check if device is known to user
        if context.user_profile and fp.fingerprint_id:
            if fp.fingerprint_id not in context.user_profile.known_devices:
                factors.append(RiskFactor(
                    name="device_new_for_user",
                    category=RiskCategory.DEVICE,
                    score=0.4,
                    weight=self.weights["device_unknown"],
                    description="First time seeing this device for user",
                ))

        # Bot detection
        if fp.user_agent.is_bot:
            factors.append(RiskFactor(
                name="device_bot",
                category=RiskCategory.DEVICE,
                score=0.9,
                weight=0.25,
                description=f"Bot detected: {fp.user_agent.bot_name}",
            ))

        return factors

    def _evaluate_location_risk(self, context: RiskContext) -> list[RiskFactor]:
        """Evaluate location-related risk factors."""
        factors: list[RiskFactor] = []

        if not context.geo_location:
            return factors

        geo = context.geo_location

        # VPN detection
        if geo.is_vpn:
            factors.append(RiskFactor(
                name="location_vpn",
                category=RiskCategory.LOCATION,
                score=0.4,
                weight=self.weights["location_vpn"],
                description="VPN detected",
                evidence={"is_vpn": True},
            ))

        # Tor detection
        if geo.is_tor:
            factors.append(RiskFactor(
                name="location_tor",
                category=RiskCategory.LOCATION,
                score=0.8,
                weight=self.weights["location_tor"],
                description="Tor exit node detected",
                evidence={"is_tor": True},
            ))

        # Proxy detection
        if geo.is_proxy:
            factors.append(RiskFactor(
                name="location_proxy",
                category=RiskCategory.LOCATION,
                score=0.5,
                weight=self.weights["location_vpn"],
                description="Proxy detected",
            ))

        # High-risk countries (example list)
        high_risk_countries = {"XX", "YY", "ZZ"}  # Replace with actual list
        if geo.country_code in high_risk_countries:
            factors.append(RiskFactor(
                name="location_high_risk",
                category=RiskCategory.LOCATION,
                score=0.7,
                weight=self.weights["location_high_risk"],
                description=f"High-risk location: {geo.country_name}",
            ))

        # Check for impossible travel
        if context.user_id:
            impossible_travel = self._check_impossible_travel(context.user_id, geo, context.timestamp)
            if impossible_travel:
                factors.append(RiskFactor(
                    name="location_impossible_travel",
                    category=RiskCategory.LOCATION,
                    score=0.95,
                    weight=self.weights["location_impossible_travel"],
                    description=f"Impossible travel detected: {impossible_travel['description']}",
                    evidence=impossible_travel,
                ))

        # New location for user
        if context.user_profile:
            is_known_location = any(
                loc.country_code == geo.country_code and loc.region_code == geo.region_code
                for loc in context.user_profile.known_locations
            )
            if not is_known_location:
                factors.append(RiskFactor(
                    name="location_new",
                    category=RiskCategory.LOCATION,
                    score=0.5,
                    weight=self.weights["location_new"],
                    description=f"New location: {geo.city}, {geo.country_name}",
                ))

        return factors

    def _evaluate_behavior_risk(self, context: RiskContext) -> list[RiskFactor]:
        """Evaluate behavioral risk factors."""
        factors: list[RiskFactor] = []

        # Check activity timing
        if context.user_profile:
            current_hour = time.localtime(context.timestamp).tm_hour
            current_day = time.localtime(context.timestamp).tm_wday

            if context.user_profile.typical_hours:
                if current_hour not in context.user_profile.typical_hours:
                    factors.append(RiskFactor(
                        name="behavior_unusual_time",
                        category=RiskCategory.BEHAVIOR,
                        score=0.4,
                        weight=self.weights["behavior_unusual_time"],
                        description=f"Unusual activity time: {current_hour}:00",
                        evidence={"hour": current_hour, "typical": context.user_profile.typical_hours},
                    ))

            if context.user_profile.typical_days:
                if current_day not in context.user_profile.typical_days:
                    factors.append(RiskFactor(
                        name="behavior_unusual_day",
                        category=RiskCategory.BEHAVIOR,
                        score=0.3,
                        weight=self.weights["behavior_unusual_time"],
                        description=f"Unusual activity day",
                    ))

        # Check rapid request patterns
        if context.signals.get("requests_per_minute", 0) > 60:
            factors.append(RiskFactor(
                name="behavior_rapid_requests",
                category=RiskCategory.BEHAVIOR,
                score=0.7,
                weight=self.weights["behavior_rapid_requests"],
                description="Rapid request pattern detected",
                evidence={"rpm": context.signals.get("requests_per_minute")},
            ))

        return factors

    def _evaluate_session_risk(self, context: RiskContext) -> list[RiskFactor]:
        """Evaluate session-related risk factors."""
        factors: list[RiskFactor] = []

        if context.session_age_seconds is None:
            factors.append(RiskFactor(
                name="session_new",
                category=RiskCategory.SESSION,
                score=0.3,
                weight=self.weights["session_new"],
                description="New session",
            ))
        elif context.session_age_seconds > 86400:  # 24 hours
            factors.append(RiskFactor(
                name="session_long_lived",
                category=RiskCategory.SESSION,
                score=0.4,
                weight=self.weights["session_long_lived"],
                description="Long-lived session",
                evidence={"age_hours": context.session_age_seconds / 3600},
            ))

        return factors

    def _evaluate_account_risk(self, context: RiskContext) -> list[RiskFactor]:
        """Evaluate account-related risk factors."""
        factors: list[RiskFactor] = []

        if not context.user_profile:
            return factors

        profile = context.user_profile

        # New account
        if profile.account_age_days < 7:
            factors.append(RiskFactor(
                name="account_new",
                category=RiskCategory.ACCOUNT,
                score=0.5,
                weight=self.weights["account_new"],
                description=f"New account ({profile.account_age_days} days old)",
            ))

        # Unverified account
        if not profile.email_verified or not profile.mfa_enabled:
            score = 0.0
            descriptions = []
            if not profile.email_verified:
                score += 0.3
                descriptions.append("email not verified")
            if not profile.mfa_enabled:
                score += 0.2
                descriptions.append("MFA not enabled")

            factors.append(RiskFactor(
                name="account_unverified",
                category=RiskCategory.ACCOUNT,
                score=score,
                weight=self.weights["account_unverified"],
                description=f"Account security: {', '.join(descriptions)}",
                mitigations=["Enable MFA", "Verify email"],
            ))

        # Recent failed attempts
        if profile.failed_auth_count > 0:
            score = min(1.0, profile.failed_auth_count * 0.15)
            factors.append(RiskFactor(
                name="account_failed_attempts",
                category=RiskCategory.ACCOUNT,
                score=score,
                weight=self.weights["account_failed_attempts"],
                description=f"{profile.failed_auth_count} recent failed authentication attempts",
            ))

        # Account flags
        if "suspicious_activity" in profile.flags:
            factors.append(RiskFactor(
                name="account_flagged_suspicious",
                category=RiskCategory.ACCOUNT,
                score=0.8,
                weight=0.20,
                description="Account flagged for suspicious activity",
            ))

        return factors

    def _evaluate_network_risk(self, context: RiskContext) -> list[RiskFactor]:
        """Evaluate network-related risk factors."""
        factors: list[RiskFactor] = []

        if context.geo_location and context.geo_location.is_datacenter:
            factors.append(RiskFactor(
                name="network_datacenter",
                category=RiskCategory.NETWORK,
                score=0.5,
                weight=self.weights["network_datacenter"],
                description="Request from datacenter IP",
            ))

        # Check TLS version from device fingerprint
        if context.device_fingerprint:
            network = context.device_fingerprint.network
            if network.tls_version and network.tls_version < "TLSv1.2":
                factors.append(RiskFactor(
                    name="network_outdated_tls",
                    category=RiskCategory.NETWORK,
                    score=0.4,
                    weight=self.weights["network_outdated_tls"],
                    description=f"Outdated TLS version: {network.tls_version}",
                ))

        return factors

    def _evaluate_temporal_risk(self, context: RiskContext) -> list[RiskFactor]:
        """Evaluate time-based risk factors."""
        factors: list[RiskFactor] = []

        # Off-hours for business applications
        current_hour = time.localtime(context.timestamp).tm_hour
        if context.signals.get("business_hours_only"):
            if current_hour < 6 or current_hour > 22:
                factors.append(RiskFactor(
                    name="temporal_off_hours",
                    category=RiskCategory.TEMPORAL,
                    score=0.5,
                    weight=0.10,
                    description="Access outside business hours",
                ))

        return factors

    def _evaluate_operation_risk(self, context: RiskContext) -> list[RiskFactor]:
        """Evaluate operation-specific risk factors."""
        factors: list[RiskFactor] = []

        sensitivity_scores = {
            "low": 0.1,
            "normal": 0.2,
            "high": 0.5,
            "critical": 0.8,
        }

        if context.operation_sensitivity in ("high", "critical"):
            factors.append(RiskFactor(
                name="operation_sensitive",
                category=RiskCategory.BEHAVIOR,
                score=sensitivity_scores.get(context.operation_sensitivity, 0.2),
                weight=self.weights["operation_sensitive"],
                description=f"Sensitive operation: {context.operation_type}",
            ))

        return factors

    def _check_impossible_travel(
        self,
        user_id: str,
        current_geo: GeoLocation,
        current_time: float,
    ) -> Optional[dict[str, Any]]:
        """
        Check for impossible travel based on location history.

        Returns evidence dict if impossible travel detected, None otherwise.
        """
        if user_id not in self._location_history:
            return None

        history = self._location_history[user_id]
        if not history:
            return None

        # Get most recent location
        last_geo, last_time = history[-1]

        if not last_geo.latitude or not current_geo.latitude:
            return None

        # Calculate distance
        distance_km = self._haversine_distance(
            last_geo.latitude, last_geo.longitude,
            current_geo.latitude, current_geo.longitude,
        )

        # Calculate time difference
        time_diff_hours = (current_time - last_time) / 3600

        if time_diff_hours <= 0:
            return None

        # Calculate required speed
        required_speed = distance_km / time_diff_hours

        # Max realistic travel speed (commercial jet ~900 km/h with buffer)
        max_speed = 1000

        if required_speed > max_speed:
            return {
                "from_location": f"{last_geo.city}, {last_geo.country_name}",
                "to_location": f"{current_geo.city}, {current_geo.country_name}",
                "distance_km": round(distance_km, 2),
                "time_diff_hours": round(time_diff_hours, 2),
                "required_speed_kmh": round(required_speed, 2),
                "description": f"Traveled {round(distance_km)}km in {round(time_diff_hours, 1)}h (requires {round(required_speed)}km/h)",
            }

        return None

    def _haversine_distance(
        self,
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float,
    ) -> float:
        """Calculate the great circle distance between two points."""
        R = 6371  # Earth's radius in kilometers

        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)

        a = (
            math.sin(delta_lat / 2) ** 2 +
            math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c

    def _score_to_level(self, score: float) -> RiskLevel:
        """Convert numeric score to risk level."""
        for level, (min_score, max_score) in self.THRESHOLDS.items():
            if min_score <= score <= max_score:
                return level
        return RiskLevel.CRITICAL

    def _determine_actions(
        self,
        risk_score: RiskScore,
        context: RiskContext,
    ) -> list[RiskAction]:
        """Determine recommended actions based on risk score."""
        actions: list[RiskAction] = []

        if risk_score.level == RiskLevel.MINIMAL:
            actions.append(RiskAction.ALLOW)

        elif risk_score.level == RiskLevel.LOW:
            actions.append(RiskAction.ALLOW_WITH_LOGGING)

        elif risk_score.level == RiskLevel.MEDIUM:
            actions.append(RiskAction.ALLOW_WITH_LOGGING)
            if context.operation_sensitivity in ("high", "critical"):
                actions.append(RiskAction.REQUIRE_MFA)
            else:
                actions.append(RiskAction.REQUIRE_CAPTCHA)

        elif risk_score.level == RiskLevel.HIGH:
            actions.append(RiskAction.REQUIRE_MFA)
            if context.operation_sensitivity == "critical":
                actions.append(RiskAction.REQUIRE_STEP_UP)

        elif risk_score.level == RiskLevel.CRITICAL:
            # Check specific factors
            impossible_travel = any(
                f.name == "location_impossible_travel"
                for f in risk_score.factors
            )
            bot_detected = any(f.name == "device_bot" for f in risk_score.factors)

            if impossible_travel or bot_detected:
                actions.append(RiskAction.BLOCK)
            else:
                actions.append(RiskAction.MANUAL_REVIEW)
                actions.append(RiskAction.DELAY)

        return actions

    def _determine_session_restrictions(
        self,
        risk_score: RiskScore,
        context: RiskContext,
    ) -> dict[str, Any]:
        """Determine session restrictions based on risk."""
        restrictions: dict[str, Any] = {}

        if risk_score.level in (RiskLevel.HIGH, RiskLevel.CRITICAL):
            restrictions["max_session_duration"] = 1800  # 30 minutes
            restrictions["require_reauthentication"] = True
            restrictions["disable_remember_me"] = True
            restrictions["restrict_sensitive_operations"] = True

        elif risk_score.level == RiskLevel.MEDIUM:
            restrictions["max_session_duration"] = 3600  # 1 hour
            restrictions["require_mfa_for_sensitive"] = True

        return restrictions

    def _update_history(
        self,
        context: RiskContext,
        assessment: RiskAssessment,
    ) -> None:
        """Update location and assessment history."""
        if context.user_id and context.geo_location:
            if context.user_id not in self._location_history:
                self._location_history[context.user_id] = []

            self._location_history[context.user_id].append(
                (context.geo_location, context.timestamp)
            )

            # Keep only last 10 locations
            self._location_history[context.user_id] = \
                self._location_history[context.user_id][-10:]

        # Update user profile
        if context.user_profile:
            context.user_profile.last_assessment = assessment.score
            context.user_profile.last_activity_at = context.timestamp

    def register_user_profile(self, profile: UserRiskProfile) -> None:
        """Register or update a user risk profile."""
        self._user_profiles[profile.user_id] = profile

    def get_user_profile(self, user_id: str) -> Optional[UserRiskProfile]:
        """Get a user's risk profile."""
        return self._user_profiles.get(user_id)

    def add_custom_evaluator(
        self,
        evaluator: Callable[[RiskContext], Optional[RiskFactor]],
    ) -> None:
        """Add a custom risk factor evaluator."""
        self.custom_evaluators.append(evaluator)


class AdaptiveAuthenticator:
    """
    Adaptive authentication controller.

    Uses risk assessment to dynamically adjust authentication requirements.
    """

    def __init__(self, risk_engine: RiskEngine) -> None:
        self.risk_engine = risk_engine
        self._logger = logger.bind(component="adaptive_auth")

    async def evaluate_authentication_request(
        self,
        context: RiskContext,
    ) -> dict[str, Any]:
        """
        Evaluate an authentication request and return requirements.

        Returns dict with:
        - allowed: bool
        - requirements: list of required auth methods
        - challenges: any additional challenges needed
        - restrictions: session restrictions to apply
        """
        assessment = self.risk_engine.assess(context)

        response: dict[str, Any] = {
            "allowed": not assessment.block_access,
            "risk_score": assessment.score.total,
            "risk_level": assessment.score.level.value,
            "requirements": [],
            "challenges": [],
            "restrictions": assessment.session_restrictions,
        }

        if assessment.block_access:
            response["blocked"] = True
            response["block_reason"] = "Risk assessment threshold exceeded"
            return response

        # Determine authentication requirements
        if RiskAction.REQUIRE_MFA in assessment.actions:
            response["requirements"].append("mfa")

        if RiskAction.REQUIRE_STEP_UP in assessment.actions:
            response["requirements"].append("step_up_auth")

        if RiskAction.REQUIRE_CAPTCHA in assessment.actions:
            response["challenges"].append("captcha")

        if RiskAction.DELAY in assessment.actions:
            response["delay_seconds"] = 5

        if RiskAction.MANUAL_REVIEW in assessment.actions:
            response["pending_review"] = True
            response["requirements"].append("manual_approval")

        return response

    async def post_authentication_action(
        self,
        context: RiskContext,
        auth_success: bool,
    ) -> None:
        """
        Handle post-authentication actions.

        Updates user profile and triggers any necessary alerts.
        """
        if not context.user_profile:
            return

        profile = context.user_profile

        if auth_success:
            profile.successful_auth_count += 1
            profile.failed_auth_count = 0  # Reset on success

            # Add device to known devices
            if context.device_fingerprint:
                if context.device_fingerprint.fingerprint_id not in profile.known_devices:
                    profile.known_devices.append(context.device_fingerprint.fingerprint_id)
                    # Keep only last N devices
                    profile.known_devices = profile.known_devices[-10:]

            # Add location to known locations
            if context.geo_location:
                if not any(
                    loc.country_code == context.geo_location.country_code and
                    loc.region_code == context.geo_location.region_code
                    for loc in profile.known_locations
                ):
                    profile.known_locations.append(context.geo_location)
                    profile.known_locations = profile.known_locations[-10:]

            # Update typical times
            current_hour = time.localtime(context.timestamp).tm_hour
            current_day = time.localtime(context.timestamp).tm_wday

            if current_hour not in profile.typical_hours:
                profile.typical_hours.append(current_hour)
                profile.typical_hours = sorted(profile.typical_hours)[-8:]

            if current_day not in profile.typical_days:
                profile.typical_days.append(current_day)

        else:
            profile.failed_auth_count += 1

            # Flag for review if too many failures
            if profile.failed_auth_count >= 5:
                profile.flags.add("suspicious_activity")
                self._logger.warning(
                    "Account flagged for suspicious activity",
                    user_id=profile.user_id,
                    failed_attempts=profile.failed_auth_count,
                )
