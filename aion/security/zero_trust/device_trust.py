"""
Device Trust Evaluation for Zero Trust.

Implements device trust scoring based on:
- Device identity verification
- Security posture assessment
- Compliance checking
- Certificate validation
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import structlog

from aion.security.adaptive.fingerprint import DeviceFingerprint, DeviceTrustLevel

logger = structlog.get_logger()


class ComplianceStatus(str, Enum):
    """Device compliance status."""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    UNKNOWN = "unknown"
    PARTIALLY_COMPLIANT = "partially_compliant"


class SecurityPosture(str, Enum):
    """Device security posture."""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    CRITICAL = "critical"


@dataclass
class DeviceComplianceCheck:
    """Individual compliance check result."""
    check_name: str
    passed: bool
    required: bool
    details: str = ""
    remediation: Optional[str] = None


@dataclass
class DeviceTrustScore:
    """Complete device trust assessment."""
    device_id: str
    trust_score: float  # 0.0 to 100.0
    trust_level: DeviceTrustLevel
    security_posture: SecurityPosture
    compliance_status: ComplianceStatus
    compliance_checks: list[DeviceComplianceCheck]
    risk_factors: list[str]
    assessed_at: float = field(default_factory=time.time)
    valid_until: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_trusted(self) -> bool:
        return self.trust_level in (DeviceTrustLevel.HIGH, DeviceTrustLevel.TRUSTED)

    @property
    def is_compliant(self) -> bool:
        return self.compliance_status == ComplianceStatus.COMPLIANT

    def to_dict(self) -> dict[str, Any]:
        return {
            "device_id": self.device_id,
            "trust_score": self.trust_score,
            "trust_level": self.trust_level.value,
            "security_posture": self.security_posture.value,
            "compliance_status": self.compliance_status.value,
            "risk_factors": self.risk_factors,
            "is_trusted": self.is_trusted,
            "is_compliant": self.is_compliant,
        }


@dataclass
class DeviceTrustPolicy:
    """Policy for device trust evaluation."""
    name: str

    # Minimum requirements
    min_trust_score: float = 60.0
    require_compliance: bool = True
    require_certificate: bool = False

    # Enabled checks
    check_os_version: bool = True
    check_encryption: bool = True
    check_antivirus: bool = True
    check_firewall: bool = True
    check_patches: bool = True
    check_jailbreak: bool = True
    check_screen_lock: bool = True

    # Scoring weights
    weights: dict[str, float] = field(default_factory=lambda: {
        "identity_verified": 0.20,
        "certificate_valid": 0.15,
        "encryption_enabled": 0.15,
        "os_up_to_date": 0.10,
        "security_software": 0.10,
        "firewall_enabled": 0.08,
        "screen_lock_enabled": 0.07,
        "not_jailbroken": 0.10,
        "no_suspicious_indicators": 0.05,
    })


class DeviceTrustEvaluator:
    """
    Device trust evaluation engine.

    Assesses device trustworthiness for Zero Trust access decisions.
    """

    def __init__(
        self,
        default_policy: Optional[DeviceTrustPolicy] = None,
        cache_ttl: int = 300,
    ) -> None:
        self.default_policy = default_policy or DeviceTrustPolicy(name="default")
        self.cache_ttl = cache_ttl
        self._policies: dict[str, DeviceTrustPolicy] = {"default": self.default_policy}
        self._cache: dict[str, DeviceTrustScore] = {}
        self._known_devices: dict[str, dict[str, Any]] = {}
        self._logger = logger.bind(component="device_trust")

    def add_policy(self, policy: DeviceTrustPolicy) -> None:
        """Add a device trust policy."""
        self._policies[policy.name] = policy

    def evaluate(
        self,
        fingerprint: DeviceFingerprint,
        device_info: Optional[dict[str, Any]] = None,
        policy_name: Optional[str] = None,
    ) -> DeviceTrustScore:
        """
        Evaluate device trust.

        Args:
            fingerprint: Device fingerprint
            device_info: Additional device information (MDM, certificates, etc.)
            policy_name: Policy to use for evaluation

        Returns:
            DeviceTrustScore with complete assessment
        """
        # Check cache
        cache_key = fingerprint.fingerprint_id
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            if time.time() < cached.valid_until:
                return cached

        policy = self._policies.get(policy_name or "default", self.default_policy)
        device_info = device_info or {}

        # Run compliance checks
        compliance_checks = self._run_compliance_checks(fingerprint, device_info, policy)

        # Calculate scores
        trust_score = self._calculate_trust_score(fingerprint, device_info, compliance_checks, policy)
        trust_level = self._score_to_trust_level(trust_score)
        security_posture = self._assess_security_posture(compliance_checks)
        compliance_status = self._assess_compliance(compliance_checks)
        risk_factors = self._identify_risk_factors(fingerprint, device_info, compliance_checks)

        result = DeviceTrustScore(
            device_id=fingerprint.fingerprint_id,
            trust_score=trust_score,
            trust_level=trust_level,
            security_posture=security_posture,
            compliance_status=compliance_status,
            compliance_checks=compliance_checks,
            risk_factors=risk_factors,
            valid_until=time.time() + self.cache_ttl,
        )

        # Cache result
        self._cache[cache_key] = result

        self._logger.info(
            "Device trust evaluated",
            device_id=fingerprint.fingerprint_id,
            trust_score=trust_score,
            trust_level=trust_level.value,
        )

        return result

    def _run_compliance_checks(
        self,
        fingerprint: DeviceFingerprint,
        device_info: dict[str, Any],
        policy: DeviceTrustPolicy,
    ) -> list[DeviceComplianceCheck]:
        """Run all compliance checks."""
        checks: list[DeviceComplianceCheck] = []

        # OS version check
        if policy.check_os_version:
            checks.append(self._check_os_version(fingerprint, device_info))

        # Encryption check
        if policy.check_encryption:
            checks.append(self._check_encryption(device_info))

        # Antivirus check
        if policy.check_antivirus:
            checks.append(self._check_antivirus(device_info))

        # Firewall check
        if policy.check_firewall:
            checks.append(self._check_firewall(device_info))

        # Patch level check
        if policy.check_patches:
            checks.append(self._check_patches(device_info))

        # Jailbreak/root check
        if policy.check_jailbreak:
            checks.append(self._check_jailbreak(fingerprint, device_info))

        # Screen lock check
        if policy.check_screen_lock:
            checks.append(self._check_screen_lock(device_info))

        # Certificate check
        if policy.require_certificate:
            checks.append(self._check_certificate(device_info))

        return checks

    def _check_os_version(
        self,
        fingerprint: DeviceFingerprint,
        device_info: dict[str, Any],
    ) -> DeviceComplianceCheck:
        """Check OS version is supported."""
        # Get OS info from fingerprint or device_info
        os_version = device_info.get("os_version") or fingerprint.user_agent.os_version

        # Define minimum supported versions (simplified)
        min_versions = {
            "windows": "10",
            "macos": "11",
            "android": "10",
            "ios": "14",
            "linux": "5",
        }

        os_family = fingerprint.user_agent.os_family.value.lower()
        min_version = min_versions.get(os_family)

        if not os_version or not min_version:
            return DeviceComplianceCheck(
                check_name="os_version",
                passed=False,
                required=True,
                details="Unable to determine OS version",
            )

        try:
            # Simple version comparison
            current = int(os_version.split(".")[0])
            minimum = int(min_version.split(".")[0])
            passed = current >= minimum
        except (ValueError, IndexError):
            passed = False

        return DeviceComplianceCheck(
            check_name="os_version",
            passed=passed,
            required=True,
            details=f"OS: {os_family} {os_version}",
            remediation="Update to a supported OS version" if not passed else None,
        )

    def _check_encryption(self, device_info: dict[str, Any]) -> DeviceComplianceCheck:
        """Check if device encryption is enabled."""
        encryption_enabled = device_info.get("disk_encryption_enabled")

        if encryption_enabled is None:
            return DeviceComplianceCheck(
                check_name="encryption",
                passed=False,
                required=True,
                details="Encryption status unknown",
            )

        return DeviceComplianceCheck(
            check_name="encryption",
            passed=encryption_enabled,
            required=True,
            details="Disk encryption enabled" if encryption_enabled else "Disk encryption disabled",
            remediation="Enable full disk encryption" if not encryption_enabled else None,
        )

    def _check_antivirus(self, device_info: dict[str, Any]) -> DeviceComplianceCheck:
        """Check if antivirus is installed and up to date."""
        av_installed = device_info.get("antivirus_installed", False)
        av_up_to_date = device_info.get("antivirus_up_to_date", False)

        passed = av_installed and av_up_to_date

        if not av_installed:
            details = "No antivirus detected"
            remediation = "Install approved antivirus software"
        elif not av_up_to_date:
            details = "Antivirus definitions out of date"
            remediation = "Update antivirus definitions"
        else:
            details = "Antivirus installed and up to date"
            remediation = None

        return DeviceComplianceCheck(
            check_name="antivirus",
            passed=passed,
            required=False,
            details=details,
            remediation=remediation,
        )

    def _check_firewall(self, device_info: dict[str, Any]) -> DeviceComplianceCheck:
        """Check if firewall is enabled."""
        firewall_enabled = device_info.get("firewall_enabled")

        if firewall_enabled is None:
            return DeviceComplianceCheck(
                check_name="firewall",
                passed=False,
                required=False,
                details="Firewall status unknown",
            )

        return DeviceComplianceCheck(
            check_name="firewall",
            passed=firewall_enabled,
            required=False,
            details="Firewall enabled" if firewall_enabled else "Firewall disabled",
            remediation="Enable firewall" if not firewall_enabled else None,
        )

    def _check_patches(self, device_info: dict[str, Any]) -> DeviceComplianceCheck:
        """Check if security patches are up to date."""
        last_patch_date = device_info.get("last_security_patch")

        if not last_patch_date:
            return DeviceComplianceCheck(
                check_name="patches",
                passed=False,
                required=True,
                details="Patch status unknown",
            )

        # Check if patched within last 30 days
        try:
            patch_age_days = (time.time() - last_patch_date) / 86400
            passed = patch_age_days <= 30
        except (TypeError, ValueError):
            passed = False
            patch_age_days = -1

        return DeviceComplianceCheck(
            check_name="patches",
            passed=passed,
            required=True,
            details=f"Last patched {int(patch_age_days)} days ago" if patch_age_days >= 0 else "Unknown",
            remediation="Install latest security patches" if not passed else None,
        )

    def _check_jailbreak(
        self,
        fingerprint: DeviceFingerprint,
        device_info: dict[str, Any],
    ) -> DeviceComplianceCheck:
        """Check if device is jailbroken/rooted."""
        is_jailbroken = device_info.get("is_jailbroken", False)
        is_rooted = device_info.get("is_rooted", False)

        # Also check fingerprint indicators
        if fingerprint.is_suspicious:
            jailbreak_indicators = [
                i for i in fingerprint.risk_indicators
                if "jailbreak" in i.lower() or "root" in i.lower()
            ]
            if jailbreak_indicators:
                is_jailbroken = True

        passed = not (is_jailbroken or is_rooted)

        return DeviceComplianceCheck(
            check_name="jailbreak_detection",
            passed=passed,
            required=True,
            details="Device not jailbroken/rooted" if passed else "Device appears jailbroken/rooted",
            remediation="Use an unmodified device" if not passed else None,
        )

    def _check_screen_lock(self, device_info: dict[str, Any]) -> DeviceComplianceCheck:
        """Check if screen lock is enabled."""
        screen_lock_enabled = device_info.get("screen_lock_enabled")

        if screen_lock_enabled is None:
            return DeviceComplianceCheck(
                check_name="screen_lock",
                passed=False,
                required=True,
                details="Screen lock status unknown",
            )

        return DeviceComplianceCheck(
            check_name="screen_lock",
            passed=screen_lock_enabled,
            required=True,
            details="Screen lock enabled" if screen_lock_enabled else "Screen lock disabled",
            remediation="Enable screen lock with PIN/biometrics" if not screen_lock_enabled else None,
        )

    def _check_certificate(self, device_info: dict[str, Any]) -> DeviceComplianceCheck:
        """Check if device has valid certificate."""
        cert_valid = device_info.get("certificate_valid", False)
        cert_expiry = device_info.get("certificate_expiry")

        if not cert_valid:
            return DeviceComplianceCheck(
                check_name="certificate",
                passed=False,
                required=True,
                details="No valid device certificate",
                remediation="Enroll device and obtain certificate",
            )

        # Check if certificate is expiring soon
        if cert_expiry:
            days_until_expiry = (cert_expiry - time.time()) / 86400
            if days_until_expiry < 7:
                return DeviceComplianceCheck(
                    check_name="certificate",
                    passed=True,
                    required=True,
                    details=f"Certificate expires in {int(days_until_expiry)} days",
                    remediation="Renew device certificate",
                )

        return DeviceComplianceCheck(
            check_name="certificate",
            passed=True,
            required=True,
            details="Valid device certificate",
        )

    def _calculate_trust_score(
        self,
        fingerprint: DeviceFingerprint,
        device_info: dict[str, Any],
        compliance_checks: list[DeviceComplianceCheck],
        policy: DeviceTrustPolicy,
    ) -> float:
        """Calculate overall trust score."""
        score = 0.0
        weights = policy.weights

        # Base score from compliance checks
        required_checks = [c for c in compliance_checks if c.required]
        if required_checks:
            required_passed = sum(1 for c in required_checks if c.passed)
            score += (required_passed / len(required_checks)) * 50

        optional_checks = [c for c in compliance_checks if not c.required]
        if optional_checks:
            optional_passed = sum(1 for c in optional_checks if c.passed)
            score += (optional_passed / len(optional_checks)) * 20

        # Device identity factors
        if fingerprint.fingerprint_id in self._known_devices:
            score += 10  # Known device bonus

        if device_info.get("certificate_valid"):
            score += weights.get("certificate_valid", 0.15) * 100

        # Device history
        if fingerprint.seen_count > 10:
            score += 5  # Established device bonus
        if fingerprint.trust_level == DeviceTrustLevel.TRUSTED:
            score += 10

        # Risk factor penalties
        if fingerprint.is_suspicious:
            score -= len(fingerprint.risk_indicators) * 5

        if fingerprint.user_agent.is_bot:
            score -= 30

        return max(0.0, min(100.0, score))

    def _score_to_trust_level(self, score: float) -> DeviceTrustLevel:
        """Convert score to trust level."""
        if score >= 90:
            return DeviceTrustLevel.TRUSTED
        elif score >= 75:
            return DeviceTrustLevel.HIGH
        elif score >= 50:
            return DeviceTrustLevel.MEDIUM
        elif score >= 25:
            return DeviceTrustLevel.LOW
        else:
            return DeviceTrustLevel.UNKNOWN

    def _assess_security_posture(
        self,
        compliance_checks: list[DeviceComplianceCheck],
    ) -> SecurityPosture:
        """Assess overall security posture."""
        total = len(compliance_checks)
        if total == 0:
            return SecurityPosture.CRITICAL

        passed = sum(1 for c in compliance_checks if c.passed)
        ratio = passed / total

        required_failed = sum(1 for c in compliance_checks if c.required and not c.passed)

        if required_failed > 0:
            if required_failed >= 2:
                return SecurityPosture.CRITICAL
            return SecurityPosture.POOR

        if ratio >= 0.95:
            return SecurityPosture.EXCELLENT
        elif ratio >= 0.80:
            return SecurityPosture.GOOD
        elif ratio >= 0.60:
            return SecurityPosture.ACCEPTABLE
        else:
            return SecurityPosture.POOR

    def _assess_compliance(
        self,
        compliance_checks: list[DeviceComplianceCheck],
    ) -> ComplianceStatus:
        """Assess compliance status."""
        required_checks = [c for c in compliance_checks if c.required]

        if not required_checks:
            return ComplianceStatus.UNKNOWN

        required_passed = all(c.passed for c in required_checks)
        all_passed = all(c.passed for c in compliance_checks)

        if all_passed:
            return ComplianceStatus.COMPLIANT
        elif required_passed:
            return ComplianceStatus.PARTIALLY_COMPLIANT
        else:
            return ComplianceStatus.NON_COMPLIANT

    def _identify_risk_factors(
        self,
        fingerprint: DeviceFingerprint,
        device_info: dict[str, Any],
        compliance_checks: list[DeviceComplianceCheck],
    ) -> list[str]:
        """Identify risk factors for the device."""
        risks: list[str] = []

        # From fingerprint
        risks.extend(fingerprint.risk_indicators)

        # From compliance checks
        for check in compliance_checks:
            if not check.passed and check.required:
                risks.append(f"failed_{check.check_name}")

        # From device info
        if device_info.get("is_shared_device"):
            risks.append("shared_device")

        if device_info.get("is_byod"):
            risks.append("byod_device")

        if device_info.get("mdm_managed") is False:
            risks.append("unmanaged_device")

        return list(set(risks))

    def register_known_device(
        self,
        device_id: str,
        user_id: str,
        device_name: Optional[str] = None,
    ) -> None:
        """Register a device as known/trusted."""
        self._known_devices[device_id] = {
            "user_id": user_id,
            "device_name": device_name,
            "registered_at": time.time(),
        }

    def revoke_device_trust(self, device_id: str) -> bool:
        """Revoke trust for a device."""
        if device_id in self._known_devices:
            del self._known_devices[device_id]
        if device_id in self._cache:
            del self._cache[device_id]
        return True
