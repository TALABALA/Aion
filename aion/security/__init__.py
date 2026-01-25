"""
AION Security & Access Control System

Comprehensive SOTA security infrastructure for AION including:

Core Security:
- Multi-method authentication (API key, JWT, OAuth2, WebAuthn/FIDO2, mTLS)
- Role-based and policy-based authorization (RBAC + ABAC)
- Multi-tenant isolation with quotas
- Comprehensive audit logging
- Enterprise rate limiting (including distributed Redis-backed)
- Agent permission boundaries
- Secure secret management with external KMS integration

Advanced Features:
- WebAuthn/FIDO2 passwordless authentication
- Device fingerprinting and risk scoring
- ML-based anomaly detection
- Zero Trust continuous verification
- mTLS client certificate authentication
- Post-quantum cryptography support
- Secure enclave with auto-unsealing
- Adaptive authentication with risk-based access control
- Microsegmentation for Zero Trust architecture
"""

from aion.security.types import (
    # Authentication
    AuthMethod,
    TokenType,
    TokenStatus,
    MFAMethod,
    Credentials,
    AuthToken,
    Session,
    AuthenticationResult,
    AuthenticationResultStatus,
    # Users and Identity
    User,
    UserStatus,
    MFAConfig,
    ServiceAccount,
    # Tenancy
    Tenant,
    TenantStatus,
    TenantTier,
    TenantQuotas,
    TenantUsage,
    # Authorization
    Permission,
    PermissionAction,
    Role,
    Policy,
    PolicyEffect,
    PolicyCondition,
    ConditionOperator,
    AuthorizationResult,
    AuthorizationResultStatus,
    # Security Context
    SecurityContext,
    # Agent Security
    AgentCapability,
    AgentPermissionBoundary,
    NetworkPolicy,
    ResourceLimits,
    # Audit
    AuditEvent,
    AuditEventType,
    AuditEventSeverity,
    # Secrets
    Secret,
    SecretType,
    # Rate Limiting
    RateLimitConfig,
    RateLimitResult,
    RateLimitState,
)

from aion.security.manager import (
    SecurityManager,
    get_security_manager,
    set_security_manager,
)

from aion.security.authentication import (
    AuthenticationService,
    TokenManager,
    SessionManager,
)

from aion.security.authorization import (
    AuthorizationService,
    BUILTIN_ROLES,
)

from aion.security.tenancy import (
    TenancyService,
    TIER_QUOTAS,
)

from aion.security.audit import (
    AuditLogger,
)

from aion.security.rate_limiting import (
    RateLimiter,
    RateLimitStrategy,
)

from aion.security.agent_security import (
    AgentBoundaryEnforcer,
)

from aion.security.secrets import (
    SecretManager,
)

from aion.security.middleware import (
    SecurityMiddleware,
    get_security_context,
    require_auth,
    require_admin,
    require_permission,
    require_role,
    require_scope,
    authenticated,
    authorized,
    rate_limit,
)

# SOTA Feature Imports
from aion.security.adaptive import (
    DeviceFingerprint,
    DeviceFingerprintManager,
    RiskAssessment,
    RiskEngine,
    RiskLevel,
    RiskScore,
)

from aion.security.adaptive.anomaly import (
    AnomalyDetectionService,
    AnomalyType,
    AnomalySeverity,
    EnsembleDetector,
    IsolationForest,
)

from aion.security.zero_trust import (
    ContinuousVerifier,
    VerificationResult,
    DeviceTrustEvaluator,
    DeviceTrustScore,
    ContextAwareAccessController,
    AccessDecision,
    SegmentationManager,
)

from aion.security.crypto import (
    PQCProvider,
    PQCAlgorithm,
    HybridEncryption,
    KyberKEM,
    DilithiumSigner,
)

from aion.security.secrets.kms import (
    KeyManagementService,
    KMSProvider,
    VaultBackend,
    AWSKMSBackend,
    AzureKeyVaultBackend,
)

from aion.security.secrets.enclave import (
    SecureEnclave,
    ShamirSecretSharing,
    UnsealMethod,
    SealStatus,
)

from aion.security.rate_limiting.distributed import (
    DistributedRateLimiter,
    DistributedRateLimitStrategy,
    MultiTierRateLimiter,
)

__all__ = [
    # Manager
    "SecurityManager",
    "get_security_manager",
    "set_security_manager",
    # Types - Authentication
    "AuthMethod",
    "TokenType",
    "TokenStatus",
    "MFAMethod",
    "Credentials",
    "AuthToken",
    "Session",
    "AuthenticationResult",
    "AuthenticationResultStatus",
    # Types - Users
    "User",
    "UserStatus",
    "MFAConfig",
    "ServiceAccount",
    # Types - Tenancy
    "Tenant",
    "TenantStatus",
    "TenantTier",
    "TenantQuotas",
    "TenantUsage",
    # Types - Authorization
    "Permission",
    "PermissionAction",
    "Role",
    "Policy",
    "PolicyEffect",
    "PolicyCondition",
    "ConditionOperator",
    "AuthorizationResult",
    "AuthorizationResultStatus",
    # Types - Context
    "SecurityContext",
    # Types - Agent Security
    "AgentCapability",
    "AgentPermissionBoundary",
    "NetworkPolicy",
    "ResourceLimits",
    # Types - Audit
    "AuditEvent",
    "AuditEventType",
    "AuditEventSeverity",
    # Types - Secrets
    "Secret",
    "SecretType",
    # Types - Rate Limiting
    "RateLimitConfig",
    "RateLimitResult",
    "RateLimitState",
    # Services
    "AuthenticationService",
    "TokenManager",
    "SessionManager",
    "AuthorizationService",
    "BUILTIN_ROLES",
    "TenancyService",
    "TIER_QUOTAS",
    "AuditLogger",
    "RateLimiter",
    "RateLimitStrategy",
    "AgentBoundaryEnforcer",
    "SecretManager",
    # Middleware
    "SecurityMiddleware",
    "get_security_context",
    "require_auth",
    "require_admin",
    "require_permission",
    "require_role",
    "require_scope",
    "authenticated",
    "authorized",
    "rate_limit",
    # SOTA - Adaptive Security
    "DeviceFingerprint",
    "DeviceFingerprintManager",
    "RiskAssessment",
    "RiskEngine",
    "RiskLevel",
    "RiskScore",
    # SOTA - Anomaly Detection
    "AnomalyDetectionService",
    "AnomalyType",
    "AnomalySeverity",
    "EnsembleDetector",
    "IsolationForest",
    # SOTA - Zero Trust
    "ContinuousVerifier",
    "VerificationResult",
    "DeviceTrustEvaluator",
    "DeviceTrustScore",
    "ContextAwareAccessController",
    "AccessDecision",
    "SegmentationManager",
    # SOTA - Post-Quantum Cryptography
    "PQCProvider",
    "PQCAlgorithm",
    "HybridEncryption",
    "KyberKEM",
    "DilithiumSigner",
    # SOTA - KMS Integration
    "KeyManagementService",
    "KMSProvider",
    "VaultBackend",
    "AWSKMSBackend",
    "AzureKeyVaultBackend",
    # SOTA - Secure Enclave
    "SecureEnclave",
    "ShamirSecretSharing",
    "UnsealMethod",
    "SealStatus",
    # SOTA - Distributed Rate Limiting
    "DistributedRateLimiter",
    "DistributedRateLimitStrategy",
    "MultiTierRateLimiter",
]
