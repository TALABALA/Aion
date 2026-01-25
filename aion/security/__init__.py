"""
AION Security & Access Control System

Comprehensive security infrastructure for AION including:
- Multi-method authentication (API key, JWT, OAuth2, session)
- Role-based and policy-based authorization (RBAC + ABAC)
- Multi-tenant isolation with quotas
- Comprehensive audit logging
- Enterprise rate limiting
- Agent permission boundaries
- Secure secret management
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
]
