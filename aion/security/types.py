"""
AION Security Types

Comprehensive type definitions for the security subsystem including
authentication, authorization, tenancy, audit, and agent security.

This module provides the foundational data structures for:
- Multi-factor authentication with multiple providers
- Fine-grained RBAC with hierarchical roles
- Policy-based access control with conditions
- Multi-tenant isolation with quotas
- Comprehensive audit logging
- Agent permission boundaries with capability-based security
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Literal,
    Optional,
    Protocol,
    Set,
    TypeVar,
    Union,
)


# =============================================================================
# Authentication Types
# =============================================================================


class AuthMethod(str, Enum):
    """Supported authentication methods."""

    API_KEY = "api_key"
    JWT = "jwt"
    OAUTH2 = "oauth2"
    BASIC = "basic"
    SESSION = "session"
    MTLS = "mtls"  # Mutual TLS
    SAML = "saml"
    OIDC = "oidc"  # OpenID Connect
    PASSKEY = "passkey"  # WebAuthn/FIDO2


class TokenType(str, Enum):
    """Types of authentication tokens."""

    ACCESS = "access"
    REFRESH = "refresh"
    API_KEY = "api_key"
    ID_TOKEN = "id_token"  # OIDC
    SERVICE_ACCOUNT = "service_account"
    IMPERSONATION = "impersonation"


class TokenStatus(str, Enum):
    """Token lifecycle status."""

    ACTIVE = "active"
    EXPIRED = "expired"
    REVOKED = "revoked"
    SUSPENDED = "suspended"
    PENDING_ROTATION = "pending_rotation"


class MFAMethod(str, Enum):
    """Multi-factor authentication methods."""

    TOTP = "totp"  # Time-based OTP
    SMS = "sms"
    EMAIL = "email"
    PUSH = "push"  # Push notification
    HARDWARE_KEY = "hardware_key"  # FIDO2/WebAuthn
    BACKUP_CODES = "backup_codes"


@dataclass
class Credentials:
    """Authentication credentials container."""

    method: AuthMethod

    # API Key authentication
    api_key: Optional[str] = None
    api_key_prefix: Optional[str] = None  # For key identification

    # JWT/OAuth2 authentication
    token: Optional[str] = None
    refresh_token: Optional[str] = None
    id_token: Optional[str] = None

    # Basic authentication
    username: Optional[str] = None
    password: Optional[str] = None

    # MFA
    mfa_code: Optional[str] = None
    mfa_method: Optional[MFAMethod] = None

    # OAuth2 specific
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    authorization_code: Optional[str] = None
    code_verifier: Optional[str] = None  # PKCE

    # Certificate-based
    client_certificate: Optional[str] = None
    certificate_chain: Optional[List[str]] = None

    # Session
    session_id: Optional[str] = None

    # Metadata
    device_id: Optional[str] = None
    device_fingerprint: Optional[str] = None

    def is_valid(self) -> bool:
        """Check if credentials have required fields for the method."""
        if self.method == AuthMethod.API_KEY:
            return bool(self.api_key)
        elif self.method == AuthMethod.JWT:
            return bool(self.token)
        elif self.method == AuthMethod.BASIC:
            return bool(self.username and self.password)
        elif self.method == AuthMethod.OAUTH2:
            return bool(self.token or self.authorization_code)
        elif self.method == AuthMethod.SESSION:
            return bool(self.session_id)
        return False


@dataclass
class AuthToken:
    """An authentication token with full lifecycle management."""

    token_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    token_type: TokenType = TokenType.ACCESS
    status: TokenStatus = TokenStatus.ACTIVE

    # Token value (stored as secure hash)
    token_hash: str = ""
    token_prefix: str = ""  # First few chars for identification

    # Ownership
    user_id: str = ""
    tenant_id: Optional[str] = None
    service_account_id: Optional[str] = None

    # Metadata
    name: str = ""
    description: str = ""
    scopes: List[str] = field(default_factory=list)
    audiences: List[str] = field(default_factory=list)

    # Security
    issuer: str = "aion"
    subject: str = ""
    jti: str = field(default_factory=lambda: str(uuid.uuid4()))  # JWT ID

    # Constraints
    allowed_ips: List[str] = field(default_factory=list)
    allowed_origins: List[str] = field(default_factory=list)
    allowed_user_agents: List[str] = field(default_factory=list)

    # Validity
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    not_before: Optional[datetime] = None  # Token not valid before
    last_used_at: Optional[datetime] = None
    last_rotated_at: Optional[datetime] = None

    # Revocation
    revoked: bool = False
    revoked_at: Optional[datetime] = None
    revoked_by: Optional[str] = None
    revocation_reason: Optional[str] = None

    # Rotation
    rotation_count: int = 0
    max_rotations: Optional[int] = None
    next_rotation_at: Optional[datetime] = None

    # Fingerprinting
    device_id: Optional[str] = None
    device_fingerprint: Optional[str] = None
    ip_at_creation: Optional[str] = None

    def is_valid(self) -> bool:
        """Check if token is currently valid."""
        if self.status != TokenStatus.ACTIVE:
            return False
        if self.revoked:
            return False
        now = datetime.now()
        if self.expires_at and now > self.expires_at:
            return False
        if self.not_before and now < self.not_before:
            return False
        return True

    def needs_rotation(self) -> bool:
        """Check if token needs rotation."""
        if self.next_rotation_at and datetime.now() > self.next_rotation_at:
            return True
        if self.max_rotations and self.rotation_count >= self.max_rotations:
            return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary (excluding sensitive data)."""
        return {
            "token_id": self.token_id,
            "token_type": self.token_type.value,
            "status": self.status.value,
            "token_prefix": self.token_prefix,
            "user_id": self.user_id,
            "tenant_id": self.tenant_id,
            "name": self.name,
            "scopes": self.scopes,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "last_used_at": (
                self.last_used_at.isoformat() if self.last_used_at else None
            ),
            "revoked": self.revoked,
        }


@dataclass
class Session:
    """A user session with security tracking."""

    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    tenant_id: Optional[str] = None

    # Session data
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Validity
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: datetime = field(
        default_factory=lambda: datetime.now() + timedelta(hours=24)
    )
    last_accessed_at: datetime = field(default_factory=datetime.now)
    last_activity_at: datetime = field(default_factory=datetime.now)

    # Idle timeout (session expires after inactivity)
    idle_timeout_minutes: int = 30
    absolute_timeout_hours: int = 24

    # Security context
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    device_id: Optional[str] = None
    device_fingerprint: Optional[str] = None
    geo_location: Optional[Dict[str, Any]] = None

    # Security flags
    is_elevated: bool = False  # Elevated privileges (e.g., after MFA)
    elevation_expires_at: Optional[datetime] = None
    requires_reauthentication: bool = False

    # Concurrent session control
    is_primary: bool = True
    concurrent_session_count: int = 1

    # State
    is_active: bool = True
    terminated_at: Optional[datetime] = None
    terminated_reason: Optional[str] = None

    def is_valid(self) -> bool:
        """Check if session is currently valid."""
        if not self.is_active:
            return False
        now = datetime.now()
        if now > self.expires_at:
            return False
        # Check idle timeout
        idle_expiry = self.last_activity_at + timedelta(
            minutes=self.idle_timeout_minutes
        )
        if now > idle_expiry:
            return False
        return True

    def is_elevated_valid(self) -> bool:
        """Check if elevated session is still valid."""
        if not self.is_elevated:
            return False
        if self.elevation_expires_at and datetime.now() > self.elevation_expires_at:
            return False
        return True

    def touch(self) -> None:
        """Update last activity time."""
        self.last_activity_at = datetime.now()
        self.last_accessed_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize session (excluding sensitive data)."""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "tenant_id": self.tenant_id,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "last_activity_at": self.last_activity_at.isoformat(),
            "is_elevated": self.is_elevated,
            "is_active": self.is_active,
        }


# =============================================================================
# User and Identity Types
# =============================================================================


class UserStatus(str, Enum):
    """User account status."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    LOCKED = "locked"  # Too many failed attempts
    PENDING_VERIFICATION = "pending_verification"
    PENDING_APPROVAL = "pending_approval"
    DELETED = "deleted"


@dataclass
class MFAConfig:
    """Multi-factor authentication configuration for a user."""

    enabled: bool = False
    methods: List[MFAMethod] = field(default_factory=list)
    primary_method: Optional[MFAMethod] = None

    # TOTP
    totp_secret: Optional[str] = None
    totp_verified: bool = False

    # Backup codes
    backup_codes_hash: List[str] = field(default_factory=list)
    backup_codes_used: int = 0

    # Phone/Email
    phone_number: Optional[str] = None
    phone_verified: bool = False

    # Hardware key
    hardware_key_credentials: List[Dict[str, Any]] = field(default_factory=list)

    # Recovery
    recovery_email: Optional[str] = None
    recovery_phone: Optional[str] = None


@dataclass
class User:
    """A user in the system with comprehensive security attributes."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Identity
    username: str = ""
    email: str = ""
    display_name: str = ""
    first_name: str = ""
    last_name: str = ""
    avatar_url: Optional[str] = None

    # Authentication
    password_hash: Optional[str] = None
    password_salt: Optional[str] = None
    password_algorithm: str = "argon2id"
    password_changed_at: Optional[datetime] = None
    password_expires_at: Optional[datetime] = None
    require_password_change: bool = False

    # MFA
    mfa: MFAConfig = field(default_factory=MFAConfig)

    # Tenant membership
    tenant_id: Optional[str] = None
    tenant_roles: Dict[str, List[str]] = field(
        default_factory=dict
    )  # tenant_id -> roles

    # Roles and permissions
    roles: List[str] = field(default_factory=list)
    direct_permissions: List[str] = field(default_factory=list)

    # Status
    status: UserStatus = UserStatus.ACTIVE
    status_reason: Optional[str] = None

    # Verification
    email_verified: bool = False
    email_verified_at: Optional[datetime] = None
    phone_verified: bool = False

    # Security
    failed_login_attempts: int = 0
    last_failed_login_at: Optional[datetime] = None
    lockout_until: Optional[datetime] = None
    security_questions: List[Dict[str, str]] = field(default_factory=list)

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    last_login_at: Optional[datetime] = None
    last_active_at: Optional[datetime] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    preferences: Dict[str, Any] = field(default_factory=dict)

    # OAuth/OIDC linked accounts
    linked_accounts: List[Dict[str, Any]] = field(default_factory=list)

    # Service account flag
    is_service_account: bool = False
    service_account_owner_id: Optional[str] = None

    def is_active(self) -> bool:
        """Check if user account is active."""
        if self.status != UserStatus.ACTIVE:
            return False
        if self.lockout_until and datetime.now() < self.lockout_until:
            return False
        return True

    def is_locked(self) -> bool:
        """Check if user is locked out."""
        return self.lockout_until is not None and datetime.now() < self.lockout_until

    def get_all_roles(self, tenant_id: Optional[str] = None) -> Set[str]:
        """Get all roles for the user, optionally scoped to a tenant."""
        roles = set(self.roles)
        if tenant_id and tenant_id in self.tenant_roles:
            roles.update(self.tenant_roles[tenant_id])
        return roles

    def to_dict(self, include_sensitive: bool = False) -> Dict[str, Any]:
        """Serialize user to dictionary."""
        data = {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "display_name": self.display_name,
            "roles": self.roles,
            "status": self.status.value,
            "tenant_id": self.tenant_id,
            "email_verified": self.email_verified,
            "mfa_enabled": self.mfa.enabled,
            "created_at": self.created_at.isoformat(),
            "last_login_at": (
                self.last_login_at.isoformat() if self.last_login_at else None
            ),
            "is_service_account": self.is_service_account,
        }
        if include_sensitive:
            data["metadata"] = self.metadata
            data["preferences"] = self.preferences
        return data


@dataclass
class ServiceAccount:
    """A service account for machine-to-machine authentication."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""

    # Owner
    owner_user_id: str = ""
    tenant_id: Optional[str] = None

    # Credentials
    api_key_ids: List[str] = field(default_factory=list)

    # Permissions
    roles: List[str] = field(default_factory=list)
    scopes: List[str] = field(default_factory=list)

    # Constraints
    allowed_ips: List[str] = field(default_factory=list)
    allowed_origins: List[str] = field(default_factory=list)

    # Status
    active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    last_used_at: Optional[datetime] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Tenant Types
# =============================================================================


class TenantStatus(str, Enum):
    """Tenant lifecycle status."""

    ACTIVE = "active"
    TRIAL = "trial"
    SUSPENDED = "suspended"
    PENDING_ACTIVATION = "pending_activation"
    DEACTIVATED = "deactivated"
    DELETED = "deleted"


class TenantTier(str, Enum):
    """Tenant subscription tier."""

    FREE = "free"
    STARTER = "starter"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"
    UNLIMITED = "unlimited"


@dataclass
class TenantQuotas:
    """Resource quotas for a tenant."""

    # User limits
    max_users: int = 100
    max_service_accounts: int = 10
    max_api_keys_per_user: int = 5

    # Agent limits
    max_agents: int = 10
    max_concurrent_agents: int = 5

    # Storage limits
    max_storage_gb: float = 10.0
    max_memory_vectors: int = 1000000
    max_knowledge_nodes: int = 100000

    # Rate limits
    max_requests_per_minute: int = 1000
    max_requests_per_hour: int = 50000
    max_requests_per_day: int = 500000

    # Token limits
    max_tokens_per_request: int = 100000
    max_tokens_per_day: int = 10000000

    # Tool limits
    max_tool_executions_per_day: int = 10000

    # Custom limits
    custom_limits: Dict[str, int] = field(default_factory=dict)


@dataclass
class TenantUsage:
    """Current resource usage for a tenant."""

    # User usage
    current_users: int = 0
    current_service_accounts: int = 0

    # Storage usage
    current_storage_gb: float = 0.0
    current_memory_vectors: int = 0
    current_knowledge_nodes: int = 0

    # Daily usage (reset daily)
    tokens_used_today: int = 0
    requests_today: int = 0
    tool_executions_today: int = 0

    # Monthly usage
    tokens_used_this_month: int = 0
    requests_this_month: int = 0

    # Last reset
    daily_reset_at: datetime = field(default_factory=datetime.now)
    monthly_reset_at: datetime = field(default_factory=datetime.now)


@dataclass
class Tenant:
    """A tenant (organization) in the multi-tenant system."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Identity
    name: str = ""
    slug: str = ""  # URL-safe identifier
    domain: Optional[str] = None  # Custom domain
    display_name: str = ""
    description: str = ""
    logo_url: Optional[str] = None

    # Subscription
    tier: TenantTier = TenantTier.FREE
    tier_expires_at: Optional[datetime] = None

    # Quotas and usage
    quotas: TenantQuotas = field(default_factory=TenantQuotas)
    usage: TenantUsage = field(default_factory=TenantUsage)

    # Status
    status: TenantStatus = TenantStatus.ACTIVE
    status_reason: Optional[str] = None
    trial_ends_at: Optional[datetime] = None

    # Settings
    settings: Dict[str, Any] = field(default_factory=dict)
    features: Dict[str, bool] = field(default_factory=dict)

    # Security settings
    require_mfa: bool = False
    allowed_auth_methods: List[AuthMethod] = field(
        default_factory=lambda: [AuthMethod.API_KEY, AuthMethod.JWT]
    )
    password_policy: Dict[str, Any] = field(default_factory=dict)
    session_settings: Dict[str, Any] = field(default_factory=dict)

    # Isolation
    data_isolation_level: str = "logical"  # logical, physical
    encryption_key_id: Optional[str] = None

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    activated_at: Optional[datetime] = None

    # Billing
    billing_email: Optional[str] = None
    billing_address: Optional[Dict[str, str]] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_active(self) -> bool:
        """Check if tenant is active."""
        return self.status in (TenantStatus.ACTIVE, TenantStatus.TRIAL)

    def is_within_quota(self, resource: str, amount: int = 1) -> bool:
        """Check if usage is within quota."""
        quota_map = {
            "users": (self.usage.current_users, self.quotas.max_users),
            "storage_gb": (self.usage.current_storage_gb, self.quotas.max_storage_gb),
            "tokens_today": (self.usage.tokens_used_today, self.quotas.max_tokens_per_day),
            "requests_today": (self.usage.requests_today, self.quotas.max_requests_per_day),
        }
        if resource in quota_map:
            current, limit = quota_map[resource]
            return current + amount <= limit
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Serialize tenant to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "slug": self.slug,
            "display_name": self.display_name,
            "tier": self.tier.value,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "quotas": {
                "max_users": self.quotas.max_users,
                "max_agents": self.quotas.max_agents,
                "max_storage_gb": self.quotas.max_storage_gb,
            },
            "usage": {
                "current_users": self.usage.current_users,
                "current_storage_gb": self.usage.current_storage_gb,
            },
        }


# =============================================================================
# Authorization Types
# =============================================================================


class PermissionAction(str, Enum):
    """Permission actions."""

    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    EXECUTE = "execute"
    ADMIN = "admin"
    LIST = "list"
    EXPORT = "export"
    IMPORT = "import"
    APPROVE = "approve"
    DENY = "deny"
    SHARE = "share"
    ALL = "*"


class ConditionOperator(str, Enum):
    """Operators for policy conditions."""

    EQUALS = "eq"
    NOT_EQUALS = "neq"
    GREATER_THAN = "gt"
    LESS_THAN = "lt"
    GREATER_EQUAL = "gte"
    LESS_EQUAL = "lte"
    IN = "in"
    NOT_IN = "not_in"
    CONTAINS = "contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    MATCHES = "matches"  # Regex
    EXISTS = "exists"


@dataclass
class PolicyCondition:
    """A condition for policy evaluation."""

    field: str  # e.g., "resource.owner_id", "context.ip_address"
    operator: ConditionOperator
    value: Any
    negate: bool = False

    def evaluate(self, context: Dict[str, Any]) -> bool:
        """Evaluate condition against context."""
        # Navigate to field value
        field_value = context
        for part in self.field.split("."):
            if isinstance(field_value, dict):
                field_value = field_value.get(part)
            else:
                field_value = getattr(field_value, part, None)
            if field_value is None:
                result = self.operator == ConditionOperator.EXISTS and not self.value
                return not result if self.negate else result

        # Evaluate based on operator
        result = self._evaluate_operator(field_value)
        return not result if self.negate else result

    def _evaluate_operator(self, field_value: Any) -> bool:
        """Evaluate the operator."""
        if self.operator == ConditionOperator.EQUALS:
            return field_value == self.value
        elif self.operator == ConditionOperator.NOT_EQUALS:
            return field_value != self.value
        elif self.operator == ConditionOperator.GREATER_THAN:
            return field_value > self.value
        elif self.operator == ConditionOperator.LESS_THAN:
            return field_value < self.value
        elif self.operator == ConditionOperator.GREATER_EQUAL:
            return field_value >= self.value
        elif self.operator == ConditionOperator.LESS_EQUAL:
            return field_value <= self.value
        elif self.operator == ConditionOperator.IN:
            return field_value in self.value
        elif self.operator == ConditionOperator.NOT_IN:
            return field_value not in self.value
        elif self.operator == ConditionOperator.CONTAINS:
            return self.value in field_value
        elif self.operator == ConditionOperator.STARTS_WITH:
            return str(field_value).startswith(str(self.value))
        elif self.operator == ConditionOperator.ENDS_WITH:
            return str(field_value).endswith(str(self.value))
        elif self.operator == ConditionOperator.MATCHES:
            import re

            return bool(re.match(self.value, str(field_value)))
        elif self.operator == ConditionOperator.EXISTS:
            return (field_value is not None) == self.value
        return False


@dataclass
class Permission:
    """A permission definition with optional conditions."""

    resource: str  # e.g., "agents", "goals", "memory", "*"
    action: PermissionAction
    conditions: List[PolicyCondition] = field(default_factory=list)
    description: str = ""

    def to_string(self) -> str:
        """Convert to string representation."""
        return f"{self.resource}:{self.action.value}"

    @classmethod
    def from_string(cls, perm_str: str) -> "Permission":
        """Parse from string representation."""
        parts = perm_str.split(":")
        return cls(
            resource=parts[0],
            action=PermissionAction(parts[1]) if len(parts) > 1 else PermissionAction.ALL,
        )

    def matches(self, resource: str, action: PermissionAction) -> bool:
        """Check if permission matches the requested resource and action."""
        # Check resource
        if self.resource != "*" and self.resource != resource:
            # Check hierarchical match (e.g., "agents" matches "agents.123")
            if not resource.startswith(f"{self.resource}."):
                return False

        # Check action
        if self.action != PermissionAction.ALL and self.action != action:
            return False

        return True


@dataclass
class Role:
    """A role with permissions."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    display_name: str = ""
    description: str = ""

    # Permissions
    permissions: List[Permission] = field(default_factory=list)

    # Hierarchy
    parent_role_id: Optional[str] = None
    child_role_ids: List[str] = field(default_factory=list)

    # Scope
    system_role: bool = False  # Built-in role
    tenant_id: Optional[str] = None  # Tenant-specific role

    # Constraints
    max_sessions: int = 10
    require_mfa: bool = False

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    created_by: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def has_permission(
        self,
        resource: str,
        action: PermissionAction,
        context: Dict[str, Any] = None,
    ) -> bool:
        """Check if role has a permission."""
        for perm in self.permissions:
            if perm.matches(resource, action):
                # Check conditions if present
                if perm.conditions and context:
                    if all(c.evaluate(context) for c in perm.conditions):
                        return True
                elif not perm.conditions:
                    return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize role to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "permissions": [p.to_string() for p in self.permissions],
            "system_role": self.system_role,
            "parent_role_id": self.parent_role_id,
        }


class PolicyEffect(str, Enum):
    """Policy effect."""

    ALLOW = "allow"
    DENY = "deny"


class PolicyEvaluationOrder(str, Enum):
    """Order of policy evaluation."""

    DENY_OVERRIDES = "deny_overrides"  # Any deny wins
    ALLOW_OVERRIDES = "allow_overrides"  # Any allow wins
    FIRST_MATCH = "first_match"  # First matching policy wins
    MOST_SPECIFIC = "most_specific"  # Most specific match wins


@dataclass
class Policy:
    """An access control policy."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""

    # Effect
    effect: PolicyEffect = PolicyEffect.ALLOW

    # Targets
    resource_type: str = ""  # e.g., "agent", "goal", "memory"
    resource_pattern: str = "*"  # Pattern matching (glob or regex)
    actions: List[PermissionAction] = field(default_factory=list)

    # Subjects (who this policy applies to)
    roles: List[str] = field(default_factory=list)
    users: List[str] = field(default_factory=list)
    groups: List[str] = field(default_factory=list)
    service_accounts: List[str] = field(default_factory=list)

    # Conditions
    conditions: List[PolicyCondition] = field(default_factory=list)

    # Priority (higher = evaluated first)
    priority: int = 0

    # Scope
    tenant_id: Optional[str] = None
    enabled: bool = True

    # Validity
    valid_from: Optional[datetime] = None
    valid_until: Optional[datetime] = None

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    created_by: Optional[str] = None
    version: int = 1
    tags: List[str] = field(default_factory=list)

    def is_active(self) -> bool:
        """Check if policy is currently active."""
        if not self.enabled:
            return False
        now = datetime.now()
        if self.valid_from and now < self.valid_from:
            return False
        if self.valid_until and now > self.valid_until:
            return False
        return True

    def matches_subject(
        self,
        user_id: Optional[str],
        roles: List[str],
        service_account_id: Optional[str] = None,
    ) -> bool:
        """Check if policy applies to the subject."""
        if not self.roles and not self.users and not self.service_accounts:
            return True  # Policy applies to everyone

        if user_id and user_id in self.users:
            return True
        if any(r in self.roles for r in roles):
            return True
        if service_account_id and service_account_id in self.service_accounts:
            return True
        return False

    def matches_resource(self, resource_type: str, resource_id: str = None) -> bool:
        """Check if policy applies to the resource."""
        if self.resource_type != "*" and self.resource_type != resource_type:
            return False

        if self.resource_pattern == "*":
            return True

        if resource_id:
            import fnmatch

            return fnmatch.fnmatch(resource_id, self.resource_pattern)

        return True


# =============================================================================
# Security Context
# =============================================================================


@dataclass
class SecurityContext:
    """Security context for a request/operation."""

    # Identity
    user_id: Optional[str] = None
    user: Optional[User] = None
    tenant_id: Optional[str] = None
    tenant: Optional[Tenant] = None
    service_account_id: Optional[str] = None

    # Authentication
    auth_method: Optional[AuthMethod] = None
    token: Optional[AuthToken] = None
    session: Optional[Session] = None
    authenticated_at: Optional[datetime] = None

    # Authorization
    roles: List[str] = field(default_factory=list)
    permissions: Set[str] = field(default_factory=set)
    effective_permissions: Set[str] = field(
        default_factory=set
    )  # After policy evaluation
    scopes: List[str] = field(default_factory=list)  # OAuth scopes

    # Request context
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    trace_id: Optional[str] = None
    span_id: Optional[str] = None

    # Client info
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    origin: Optional[str] = None
    device_id: Optional[str] = None
    geo_location: Optional[Dict[str, Any]] = None

    # Security flags
    is_elevated: bool = False  # Elevated privileges
    is_impersonating: bool = False
    impersonated_by: Optional[str] = None
    is_internal: bool = False  # Internal service call

    # Rate limiting
    rate_limit_key: Optional[str] = None
    rate_limit_remaining: Optional[int] = None

    # Timestamp
    created_at: datetime = field(default_factory=datetime.now)

    def is_authenticated(self) -> bool:
        """Check if context is authenticated."""
        return self.user_id is not None or self.service_account_id is not None

    def has_role(self, role: str) -> bool:
        """Check if context has a role."""
        return role in self.roles

    def has_any_role(self, roles: List[str]) -> bool:
        """Check if context has any of the roles."""
        return bool(set(roles) & set(self.roles))

    def has_all_roles(self, roles: List[str]) -> bool:
        """Check if context has all of the roles."""
        return set(roles).issubset(set(self.roles))

    def has_permission(self, permission: str) -> bool:
        """Check if context has a permission."""
        if "*:*" in self.permissions or "*:*" in self.effective_permissions:
            return True
        return permission in self.permissions or permission in self.effective_permissions

    def has_scope(self, scope: str) -> bool:
        """Check if context has an OAuth scope."""
        return scope in self.scopes or "*" in self.scopes

    def to_dict(self) -> Dict[str, Any]:
        """Serialize security context."""
        return {
            "user_id": self.user_id,
            "tenant_id": self.tenant_id,
            "service_account_id": self.service_account_id,
            "auth_method": self.auth_method.value if self.auth_method else None,
            "roles": self.roles,
            "permissions": list(self.permissions),
            "scopes": self.scopes,
            "request_id": self.request_id,
            "trace_id": self.trace_id,
            "ip_address": self.ip_address,
            "is_elevated": self.is_elevated,
            "is_impersonating": self.is_impersonating,
        }

    def __hash__(self):
        """Make context hashable for caching."""
        return hash(
            (
                self.user_id,
                self.tenant_id,
                self.request_id,
                tuple(sorted(self.roles)),
            )
        )


# =============================================================================
# Agent Security Types
# =============================================================================


class AgentCapability(str, Enum):
    """Agent capabilities that can be granted or restricted."""

    # Tool capabilities
    TOOL_WEB_BROWSE = "tool:web:browse"
    TOOL_WEB_FETCH = "tool:web:fetch"
    TOOL_FILE_READ = "tool:file:read"
    TOOL_FILE_WRITE = "tool:file:write"
    TOOL_FILE_DELETE = "tool:file:delete"
    TOOL_SHELL_EXECUTE = "tool:shell:execute"
    TOOL_CODE_EXECUTE = "tool:code:execute"
    TOOL_DATABASE_QUERY = "tool:database:query"
    TOOL_DATABASE_WRITE = "tool:database:write"
    TOOL_EMAIL_SEND = "tool:email:send"
    TOOL_API_CALL = "tool:api:call"

    # Memory capabilities
    MEMORY_READ = "memory:read"
    MEMORY_WRITE = "memory:write"
    MEMORY_DELETE = "memory:delete"
    MEMORY_SEARCH = "memory:search"

    # Knowledge graph capabilities
    KNOWLEDGE_READ = "knowledge:read"
    KNOWLEDGE_WRITE = "knowledge:write"
    KNOWLEDGE_QUERY = "knowledge:query"

    # Agent capabilities
    AGENT_SPAWN = "agent:spawn"
    AGENT_TERMINATE = "agent:terminate"
    AGENT_COMMUNICATE = "agent:communicate"

    # Goal capabilities
    GOAL_CREATE = "goal:create"
    GOAL_MODIFY = "goal:modify"
    GOAL_DELETE = "goal:delete"

    # System capabilities
    SYSTEM_ADMIN = "system:admin"
    SYSTEM_CONFIG = "system:config"


@dataclass
class NetworkPolicy:
    """Network access policy for an agent."""

    # Allowed domains/IPs
    allowed_domains: List[str] = field(default_factory=list)  # Glob patterns
    denied_domains: List[str] = field(default_factory=list)
    allowed_ips: List[str] = field(default_factory=list)  # CIDR notation
    denied_ips: List[str] = field(default_factory=list)

    # Port restrictions
    allowed_ports: List[int] = field(default_factory=lambda: [80, 443])
    denied_ports: List[int] = field(default_factory=list)

    # Protocol restrictions
    allowed_protocols: List[str] = field(default_factory=lambda: ["https", "wss"])

    # Rate limits for network calls
    max_requests_per_minute: int = 100
    max_bytes_per_minute: int = 10 * 1024 * 1024  # 10MB

    def is_allowed(self, domain: str, port: int = 443) -> bool:
        """Check if access to domain:port is allowed."""
        import fnmatch

        # Check denied first
        for pattern in self.denied_domains:
            if fnmatch.fnmatch(domain, pattern):
                return False

        if port in self.denied_ports:
            return False

        # Check allowed
        if self.allowed_domains:
            if not any(fnmatch.fnmatch(domain, p) for p in self.allowed_domains):
                return False

        if self.allowed_ports and port not in self.allowed_ports:
            return False

        return True


@dataclass
class ResourceLimits:
    """Resource limits for an agent."""

    # Token limits
    max_tokens_per_request: int = 100000
    max_tokens_per_session: int = 1000000
    max_tokens_per_day: int = 10000000

    # Call limits
    max_tool_calls_per_request: int = 50
    max_tool_calls_per_session: int = 500
    max_llm_calls_per_request: int = 10

    # Concurrency limits
    max_concurrent_requests: int = 5
    max_concurrent_tool_calls: int = 10

    # Time limits
    max_execution_time_seconds: int = 300
    max_session_duration_hours: int = 24

    # Memory limits
    max_memory_mb: int = 512
    max_context_size: int = 100000

    # Storage limits
    max_file_size_mb: int = 100
    max_total_storage_mb: int = 1000


@dataclass
class AgentPermissionBoundary:
    """Complete permission boundary for an agent."""

    agent_id: str = ""
    name: str = ""
    description: str = ""

    # Capabilities (whitelist approach)
    granted_capabilities: Set[AgentCapability] = field(default_factory=set)
    denied_capabilities: Set[AgentCapability] = field(default_factory=set)

    # Tool permissions
    allowed_tools: List[str] = field(default_factory=list)  # Tool names or "*"
    denied_tools: List[str] = field(default_factory=list)
    tool_configurations: Dict[str, Dict[str, Any]] = field(
        default_factory=dict
    )  # Tool-specific configs

    # Memory/Knowledge access
    allowed_memory_namespaces: List[str] = field(default_factory=list)
    denied_memory_namespaces: List[str] = field(default_factory=list)
    allowed_knowledge_graphs: List[str] = field(default_factory=list)
    denied_knowledge_graphs: List[str] = field(default_factory=list)

    # Network policy
    network_policy: NetworkPolicy = field(default_factory=NetworkPolicy)

    # Resource limits
    resource_limits: ResourceLimits = field(default_factory=ResourceLimits)

    # File system access
    allowed_paths: List[str] = field(default_factory=list)  # Glob patterns
    denied_paths: List[str] = field(default_factory=list)
    read_only_paths: List[str] = field(default_factory=list)

    # Time restrictions
    active_hours_start: Optional[int] = None  # 0-23
    active_hours_end: Optional[int] = None
    active_days: List[int] = field(
        default_factory=lambda: [0, 1, 2, 3, 4, 5, 6]
    )  # 0=Monday
    timezone: str = "UTC"

    # Approval requirements
    require_approval_for: List[str] = field(default_factory=list)  # Actions requiring approval
    auto_approve_low_risk: bool = True

    # Sandbox settings
    sandbox_enabled: bool = True
    sandbox_level: str = "standard"  # minimal, standard, strict

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    created_by: Optional[str] = None
    version: int = 1

    def has_capability(self, capability: AgentCapability) -> bool:
        """Check if agent has a capability."""
        if capability in self.denied_capabilities:
            return False
        return capability in self.granted_capabilities

    def can_use_tool(self, tool_name: str) -> bool:
        """Check if agent can use a tool."""
        if tool_name in self.denied_tools:
            return False
        if "*" in self.allowed_tools:
            return True
        return tool_name in self.allowed_tools

    def can_access_memory(self, namespace: str) -> bool:
        """Check if agent can access a memory namespace."""
        if namespace in self.denied_memory_namespaces:
            return False
        if "*" in self.allowed_memory_namespaces:
            return True
        return namespace in self.allowed_memory_namespaces

    def is_within_active_hours(self) -> bool:
        """Check if current time is within active hours."""
        if self.active_hours_start is None or self.active_hours_end is None:
            return True

        from datetime import datetime
        import pytz

        try:
            tz = pytz.timezone(self.timezone)
            now = datetime.now(tz)
        except Exception:
            now = datetime.now()

        current_hour = now.hour
        current_day = now.weekday()

        if current_day not in self.active_days:
            return False

        if self.active_hours_start <= self.active_hours_end:
            return self.active_hours_start <= current_hour < self.active_hours_end
        else:
            # Spans midnight
            return current_hour >= self.active_hours_start or current_hour < self.active_hours_end


# =============================================================================
# Audit Types
# =============================================================================


class AuditEventType(str, Enum):
    """Types of audit events."""

    # Authentication events
    AUTH_LOGIN_SUCCESS = "auth.login.success"
    AUTH_LOGIN_FAILURE = "auth.login.failure"
    AUTH_LOGOUT = "auth.logout"
    AUTH_MFA_SUCCESS = "auth.mfa.success"
    AUTH_MFA_FAILURE = "auth.mfa.failure"
    AUTH_PASSWORD_CHANGE = "auth.password.change"
    AUTH_PASSWORD_RESET = "auth.password.reset"

    # Token events
    TOKEN_CREATED = "token.created"
    TOKEN_ROTATED = "token.rotated"
    TOKEN_REVOKED = "token.revoked"
    TOKEN_EXPIRED = "token.expired"

    # Session events
    SESSION_CREATED = "session.created"
    SESSION_EXPIRED = "session.expired"
    SESSION_TERMINATED = "session.terminated"
    SESSION_ELEVATED = "session.elevated"

    # Authorization events
    AUTHZ_ACCESS_GRANTED = "authz.access.granted"
    AUTHZ_ACCESS_DENIED = "authz.access.denied"
    AUTHZ_PERMISSION_CHANGED = "authz.permission.changed"
    AUTHZ_ROLE_ASSIGNED = "authz.role.assigned"
    AUTHZ_ROLE_REMOVED = "authz.role.removed"

    # User management events
    USER_CREATED = "user.created"
    USER_UPDATED = "user.updated"
    USER_DELETED = "user.deleted"
    USER_LOCKED = "user.locked"
    USER_UNLOCKED = "user.unlocked"
    USER_SUSPENDED = "user.suspended"

    # Tenant events
    TENANT_CREATED = "tenant.created"
    TENANT_UPDATED = "tenant.updated"
    TENANT_SUSPENDED = "tenant.suspended"
    TENANT_QUOTA_EXCEEDED = "tenant.quota.exceeded"

    # Agent events
    AGENT_SPAWNED = "agent.spawned"
    AGENT_TERMINATED = "agent.terminated"
    AGENT_BOUNDARY_VIOLATION = "agent.boundary.violation"
    AGENT_TOOL_BLOCKED = "agent.tool.blocked"

    # Tool events
    TOOL_EXECUTED = "tool.executed"
    TOOL_FAILED = "tool.failed"
    TOOL_BLOCKED = "tool.blocked"

    # Data events
    DATA_READ = "data.read"
    DATA_WRITE = "data.write"
    DATA_DELETE = "data.delete"
    DATA_EXPORT = "data.export"
    DATA_IMPORT = "data.import"

    # Security events
    SECURITY_ALERT = "security.alert"
    SECURITY_THREAT_DETECTED = "security.threat.detected"
    SECURITY_POLICY_VIOLATION = "security.policy.violation"
    SECURITY_RATE_LIMITED = "security.rate.limited"

    # System events
    SYSTEM_CONFIG_CHANGED = "system.config.changed"
    SYSTEM_SECRET_ACCESSED = "system.secret.accessed"
    SYSTEM_SECRET_ROTATED = "system.secret.rotated"
    SYSTEM_EMERGENCY_STOP = "system.emergency.stop"


class AuditEventSeverity(str, Enum):
    """Severity levels for audit events."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AuditEvent:
    """A comprehensive audit log event."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)

    # Event classification
    event_type: AuditEventType = AuditEventType.DATA_READ
    severity: AuditEventSeverity = AuditEventSeverity.INFO
    category: str = ""  # High-level category

    # Description
    description: str = ""
    message: str = ""

    # Actor (who performed the action)
    actor_type: str = "user"  # user, service_account, agent, system
    actor_id: Optional[str] = None
    actor_name: Optional[str] = None
    actor_ip: Optional[str] = None
    actor_user_agent: Optional[str] = None

    # Subject (who was affected)
    subject_type: Optional[str] = None
    subject_id: Optional[str] = None

    # Resource (what was accessed/modified)
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    resource_name: Optional[str] = None

    # Tenant context
    tenant_id: Optional[str] = None

    # Request context
    request_id: Optional[str] = None
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    session_id: Optional[str] = None

    # Action details
    action: Optional[str] = None
    action_result: str = "success"  # success, failure, partial
    error_code: Optional[str] = None
    error_message: Optional[str] = None

    # Before/after state (for changes)
    old_value: Optional[Dict[str, Any]] = None
    new_value: Optional[Dict[str, Any]] = None
    changes: Optional[Dict[str, Any]] = None

    # Additional context
    details: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    # Compliance
    compliance_relevant: bool = False
    compliance_frameworks: List[str] = field(default_factory=list)  # GDPR, SOC2, HIPAA

    # Retention
    retention_days: int = 365
    immutable: bool = True

    # Verification
    checksum: Optional[str] = None

    def compute_checksum(self) -> str:
        """Compute checksum for integrity verification."""
        import json

        data = {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type.value,
            "actor_id": self.actor_id,
            "resource_id": self.resource_id,
            "action": self.action,
        }
        content = json.dumps(data, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize audit event."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type.value,
            "severity": self.severity.value,
            "description": self.description,
            "actor_type": self.actor_type,
            "actor_id": self.actor_id,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "tenant_id": self.tenant_id,
            "request_id": self.request_id,
            "action": self.action,
            "action_result": self.action_result,
            "error_message": self.error_message,
            "details": self.details,
            "tags": self.tags,
        }


# =============================================================================
# Secret Types
# =============================================================================


class SecretType(str, Enum):
    """Types of secrets."""

    API_KEY = "api_key"
    DATABASE_PASSWORD = "database_password"
    ENCRYPTION_KEY = "encryption_key"
    SIGNING_KEY = "signing_key"
    OAUTH_CLIENT_SECRET = "oauth_client_secret"
    WEBHOOK_SECRET = "webhook_secret"
    CERTIFICATE = "certificate"
    PRIVATE_KEY = "private_key"
    CONNECTION_STRING = "connection_string"
    GENERIC = "generic"


@dataclass
class Secret:
    """A managed secret."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""

    # Type
    secret_type: SecretType = SecretType.GENERIC

    # Value (encrypted at rest)
    encrypted_value: bytes = b""
    encryption_key_id: str = ""

    # Version
    version: int = 1
    previous_version_id: Optional[str] = None

    # Scope
    tenant_id: Optional[str] = None
    environment: str = "production"  # development, staging, production

    # Access control
    allowed_services: List[str] = field(default_factory=list)
    allowed_users: List[str] = field(default_factory=list)

    # Rotation
    rotation_enabled: bool = True
    rotation_interval_days: int = 90
    last_rotated_at: Optional[datetime] = None
    next_rotation_at: Optional[datetime] = None

    # Expiration
    expires_at: Optional[datetime] = None

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    created_by: Optional[str] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    def needs_rotation(self) -> bool:
        """Check if secret needs rotation."""
        if not self.rotation_enabled:
            return False
        if self.next_rotation_at and datetime.now() > self.next_rotation_at:
            return True
        return False


# =============================================================================
# Rate Limiting Types
# =============================================================================


@dataclass
class RateLimitConfig:
    """Rate limit configuration."""

    # Window-based limits
    requests_per_second: Optional[int] = None
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    requests_per_day: int = 10000

    # Token bucket
    bucket_size: int = 100
    refill_rate: float = 1.0  # tokens per second

    # Sliding window
    window_size_seconds: int = 60
    window_limit: int = 60

    # Concurrency limit
    max_concurrent: int = 10

    # Cost-based (for expensive operations)
    daily_cost_limit: int = 1000000  # arbitrary units
    operation_costs: Dict[str, int] = field(default_factory=dict)

    # Burst protection
    burst_limit: int = 20
    burst_window_seconds: int = 1


@dataclass
class RateLimitState:
    """Current rate limit state for a key."""

    key: str = ""

    # Window counters
    second_count: int = 0
    second_reset: datetime = field(default_factory=datetime.now)

    minute_count: int = 0
    minute_reset: datetime = field(default_factory=datetime.now)

    hour_count: int = 0
    hour_reset: datetime = field(default_factory=datetime.now)

    day_count: int = 0
    day_reset: datetime = field(default_factory=datetime.now)

    # Token bucket
    tokens: float = 0.0
    last_refill: datetime = field(default_factory=datetime.now)

    # Sliding window (timestamps of recent requests)
    request_timestamps: List[datetime] = field(default_factory=list)

    # Concurrency
    current_concurrent: int = 0

    # Cost tracking
    daily_cost: int = 0

    # Penalty (for abuse detection)
    penalty_until: Optional[datetime] = None
    penalty_count: int = 0


@dataclass
class RateLimitResult:
    """Result of a rate limit check."""

    allowed: bool = True
    limit_type: Optional[str] = None  # Which limit was hit
    current: int = 0
    limit: int = 0
    remaining: int = 0
    reset_at: Optional[datetime] = None
    retry_after_seconds: Optional[float] = None

    def to_headers(self) -> Dict[str, str]:
        """Convert to HTTP headers."""
        headers = {
            "X-RateLimit-Limit": str(self.limit),
            "X-RateLimit-Remaining": str(self.remaining),
        }
        if self.reset_at:
            headers["X-RateLimit-Reset"] = str(int(self.reset_at.timestamp()))
        if self.retry_after_seconds:
            headers["Retry-After"] = str(int(self.retry_after_seconds))
        return headers


# =============================================================================
# Authentication Result Types
# =============================================================================


class AuthenticationResultStatus(str, Enum):
    """Authentication result status."""

    SUCCESS = "success"
    FAILURE = "failure"
    MFA_REQUIRED = "mfa_required"
    PASSWORD_EXPIRED = "password_expired"
    ACCOUNT_LOCKED = "account_locked"
    ACCOUNT_SUSPENDED = "account_suspended"
    REQUIRES_EMAIL_VERIFICATION = "requires_email_verification"


@dataclass
class AuthenticationResult:
    """Result of an authentication attempt."""

    status: AuthenticationResultStatus
    context: Optional[SecurityContext] = None

    # On success
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    id_token: Optional[str] = None
    expires_in: Optional[int] = None

    # On MFA required
    mfa_token: Optional[str] = None
    mfa_methods: List[MFAMethod] = field(default_factory=list)

    # On failure
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    retry_after: Optional[int] = None  # Seconds

    # Metadata
    authenticated_at: Optional[datetime] = None
    auth_method: Optional[AuthMethod] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize result."""
        data = {
            "status": self.status.value,
            "authenticated_at": (
                self.authenticated_at.isoformat() if self.authenticated_at else None
            ),
        }
        if self.status == AuthenticationResultStatus.SUCCESS:
            data["access_token"] = self.access_token
            data["token_type"] = "Bearer"
            data["expires_in"] = self.expires_in
            if self.refresh_token:
                data["refresh_token"] = self.refresh_token
        elif self.status == AuthenticationResultStatus.MFA_REQUIRED:
            data["mfa_token"] = self.mfa_token
            data["mfa_methods"] = [m.value for m in self.mfa_methods]
        elif self.status == AuthenticationResultStatus.FAILURE:
            data["error"] = self.error_code
            data["error_description"] = self.error_message
            if self.retry_after:
                data["retry_after"] = self.retry_after
        return data


# =============================================================================
# Authorization Result Types
# =============================================================================


class AuthorizationResultStatus(str, Enum):
    """Authorization result status."""

    ALLOWED = "allowed"
    DENIED = "denied"
    REQUIRES_ELEVATION = "requires_elevation"
    REQUIRES_APPROVAL = "requires_approval"


@dataclass
class AuthorizationResult:
    """Result of an authorization check."""

    status: AuthorizationResultStatus
    resource: str = ""
    action: str = ""

    # Decision details
    decided_by: str = ""  # policy, role, default
    policy_id: Optional[str] = None
    role_name: Optional[str] = None

    # On denial
    reason: Optional[str] = None
    missing_permissions: List[str] = field(default_factory=list)
    missing_roles: List[str] = field(default_factory=list)

    # Conditions
    conditions_evaluated: int = 0
    conditions_passed: int = 0

    # Recommendations
    elevation_required: bool = False
    approval_required: bool = False
    approval_workflow: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize result."""
        return {
            "status": self.status.value,
            "resource": self.resource,
            "action": self.action,
            "decided_by": self.decided_by,
            "reason": self.reason,
        }
