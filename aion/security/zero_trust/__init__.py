"""
Zero Trust Security Module.

Implements Zero Trust Architecture principles:
- Never trust, always verify
- Assume breach
- Verify explicitly
- Use least privilege access
- Microsegmentation

Components:
- Continuous verification
- Device trust scoring
- Context-aware access
- Microsegmentation
"""

from aion.security.zero_trust.verification import (
    ContinuousVerifier,
    VerificationContext,
    VerificationPolicy,
    VerificationResult,
)
from aion.security.zero_trust.device_trust import (
    DeviceTrustEvaluator,
    DeviceTrustScore,
    DeviceComplianceCheck,
)
from aion.security.zero_trust.access import (
    ContextAwareAccessController,
    AccessDecision,
    AccessPolicy,
)
from aion.security.zero_trust.microsegmentation import (
    Microsegment,
    MicrosegmentPolicy,
    SegmentationManager,
)

__all__ = [
    "ContinuousVerifier",
    "VerificationContext",
    "VerificationPolicy",
    "VerificationResult",
    "DeviceTrustEvaluator",
    "DeviceTrustScore",
    "DeviceComplianceCheck",
    "ContextAwareAccessController",
    "AccessDecision",
    "AccessPolicy",
    "Microsegment",
    "MicrosegmentPolicy",
    "SegmentationManager",
]
