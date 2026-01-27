"""
Plugin Security Module

Provides cryptographic code signing, verification, and
capability-based security for plugins.
"""

from aion.plugins.security.signing import (
    PluginSigner,
    PluginVerifier,
    SignatureInfo,
    SigningKey,
    VerifyingKey,
    SignatureError,
    InvalidSignatureError,
    ExpiredSignatureError,
)
from aion.plugins.security.capabilities import (
    Capability,
    CapabilityGrant,
    CapabilityChecker,
    CapabilityDeniedError,
    CapabilityPrompt,
)

__all__ = [
    "PluginSigner",
    "PluginVerifier",
    "SignatureInfo",
    "SigningKey",
    "VerifyingKey",
    "SignatureError",
    "InvalidSignatureError",
    "ExpiredSignatureError",
    "Capability",
    "CapabilityGrant",
    "CapabilityChecker",
    "CapabilityDeniedError",
    "CapabilityPrompt",
]
