"""
AION Security Manager

Central coordinator for all security services.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import structlog

from aion.security.types import (
    AuthenticationResult,
    AuthorizationResult,
    AuthorizationResultStatus,
    AuthMethod,
    Credentials,
    Permission,
    PermissionAction,
    RateLimitResult,
    SecurityContext,
    User,
    Role,
    Policy,
    PolicyEffect,
    Tenant,
    TenantTier,
    AgentPermissionBoundary,
    AuditEventType,
    Secret,
    SecretType,
)
from aion.security.authentication import AuthenticationService
from aion.security.authorization import AuthorizationService
from aion.security.tenancy import TenancyService
from aion.security.audit import AuditLogger
from aion.security.rate_limiting import RateLimiter, RateLimitStrategy
from aion.security.agent_security import AgentBoundaryEnforcer
from aion.security.secrets import SecretManager

logger = structlog.get_logger(__name__)


# Global security manager instance
_security_manager: Optional["SecurityManager"] = None


def get_security_manager() -> "SecurityManager":
    """Get the global security manager instance."""
    global _security_manager
    if _security_manager is None:
        _security_manager = SecurityManager()
    return _security_manager


def set_security_manager(manager: "SecurityManager") -> None:
    """Set the global security manager instance."""
    global _security_manager
    _security_manager = manager


class SecurityManager:
    """
    Central security manager orchestrating all security services.

    This is the main entry point for security operations in AION.

    Features:
    - Unified authentication across providers
    - Fine-grained authorization (RBAC + ABAC)
    - Multi-tenant isolation
    - Comprehensive audit logging
    - Rate limiting
    - Agent permission boundaries
    - Secret management
    """

    def __init__(
        self,
        jwt_secret: Optional[str] = None,
        rate_limit_strategy: RateLimitStrategy = RateLimitStrategy.TOKEN_BUCKET,
    ):
        # Core services
        self.auth = AuthenticationService(jwt_secret=jwt_secret)
        self.authz = AuthorizationService()
        self.tenancy = TenancyService()
        self.audit = AuditLogger()
        self.rate_limiter = RateLimiter(strategy=rate_limit_strategy)
        self.agent_boundaries = AgentBoundaryEnforcer()
        self.secrets = SecretManager()

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize all security services."""
        if self._initialized:
            return

        logger.info("Initializing Security Manager")

        # Initialize all services
        await self.auth.initialize()
        await self.authz.initialize()
        await self.tenancy.initialize()
        await self.audit.initialize()
        await self.rate_limiter.initialize()
        await self.agent_boundaries.initialize()
        await self.secrets.initialize()

        # Wire up event handlers
        self._setup_event_handlers()

        self._initialized = True
        logger.info("Security Manager initialized successfully")

    async def shutdown(self) -> None:
        """Shutdown all security services."""
        logger.info("Shutting down Security Manager")

        await self.audit.shutdown()
        await self.rate_limiter.shutdown()
        await self.auth.shutdown()

        self._initialized = False
        logger.info("Security Manager shutdown complete")

    def _setup_event_handlers(self) -> None:
        """Setup event handlers between services."""
        # Log authentication events
        self.auth.add_event_handler(self._on_auth_event)

        # Log authorization decisions
        self.authz.add_decision_handler(self._on_authz_decision)

        # Log agent boundary violations
        self.agent_boundaries.add_violation_handler(self._on_boundary_violation)

    def _on_auth_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Handle authentication events."""
        if event_type == "auth_success":
            self.audit.log_auth_success(
                context=SecurityContext(
                    user_id=data.get("user_id"),
                    ip_address=data.get("ip_address"),
                ),
                method=data.get("method", "unknown"),
            )
        elif event_type == "auth_failed":
            self.audit.log_auth_failure(
                username=data.get("username", "unknown"),
                reason=data.get("error", "unknown"),
                ip_address=data.get("ip_address"),
            )

    def _on_authz_decision(self, data: Dict[str, Any]) -> None:
        """Handle authorization decisions."""
        context = SecurityContext(
            user_id=data.get("user_id"),
        )

        if data.get("allowed"):
            self.audit.log_access_granted(
                context=context,
                resource=f"{data.get('resource')}:{data.get('resource_id', '')}",
                action=data.get("action", "unknown"),
                decided_by=data.get("decided_by", "unknown"),
            )
        else:
            self.audit.log_access_denied(
                context=context,
                resource=f"{data.get('resource')}:{data.get('resource_id', '')}",
                action=data.get("action", "unknown"),
            )

    def _on_boundary_violation(self, violation) -> None:
        """Handle agent boundary violations."""
        self.audit.log_agent_boundary_violation(
            agent_id=violation.agent_id,
            violation_type=violation.violation_type,
            details={
                "resource": violation.resource,
                "action": violation.action,
                **violation.details,
            },
        )

    # =========================================================================
    # Authentication
    # =========================================================================

    async def authenticate(
        self,
        credentials: Credentials,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        device_id: Optional[str] = None,
    ) -> AuthenticationResult:
        """
        Authenticate a request.

        This is the main authentication entry point.
        """
        # Check rate limit for authentication attempts
        rate_key = f"auth:{ip_address or 'unknown'}"
        rate_result = await self.rate_limiter.check(rate_key)

        if not rate_result.allowed:
            return AuthenticationResult(
                status="failure",
                error_code="rate_limited",
                error_message="Too many authentication attempts",
                retry_after=int(rate_result.retry_after_seconds or 60),
            )

        # Authenticate
        result = await self.auth.authenticate(
            credentials,
            ip_address=ip_address,
            user_agent=user_agent,
            device_id=device_id,
        )

        # Populate permissions in context
        if result.status.value == "success" and result.context:
            result.context.permissions = set(
                await self.authz.get_user_permissions(result.context)
            )

        return result

    async def validate_token(
        self,
        token: str,
        ip_address: Optional[str] = None,
    ) -> Tuple[Optional[SecurityContext], Optional[str]]:
        """
        Validate a token and return security context.

        Returns (context, error_message).
        """
        # Determine token type and validate
        if token.startswith("aion_"):
            credentials = Credentials(method=AuthMethod.API_KEY, api_key=token)
        elif token.count(".") == 2:
            credentials = Credentials(method=AuthMethod.JWT, token=token)
        else:
            credentials = Credentials(method=AuthMethod.SESSION, session_id=token)

        result = await self.authenticate(credentials, ip_address=ip_address)

        if result.status.value == "success":
            return result.context, None
        else:
            return None, result.error_message

    # =========================================================================
    # Authorization
    # =========================================================================

    async def authorize(
        self,
        context: SecurityContext,
        resource: str,
        action: PermissionAction,
        resource_id: Optional[str] = None,
        resource_attrs: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Check if action is authorized.

        This is the main authorization entry point.
        """
        result = await self.authz.check_permission(
            context,
            resource,
            action,
            resource_id=resource_id,
            resource_attrs=resource_attrs,
        )

        return result.status == AuthorizationResultStatus.ALLOWED

    async def check_permission(
        self,
        context: SecurityContext,
        resource: str,
        action: PermissionAction,
        resource_id: Optional[str] = None,
    ) -> AuthorizationResult:
        """
        Check permission with full result details.
        """
        return await self.authz.check_permission(
            context,
            resource,
            action,
            resource_id=resource_id,
        )

    async def require_permission(
        self,
        context: SecurityContext,
        resource: str,
        action: PermissionAction,
        resource_id: Optional[str] = None,
    ) -> None:
        """
        Require a permission or raise exception.
        """
        result = await self.check_permission(context, resource, action, resource_id)

        if result.status != AuthorizationResultStatus.ALLOWED:
            raise PermissionError(
                f"Permission denied: {resource}:{action.value}"
                + (f"/{resource_id}" if resource_id else "")
            )

    # =========================================================================
    # Rate Limiting
    # =========================================================================

    async def check_rate_limit(
        self,
        key: str,
        cost: int = 1,
    ) -> RateLimitResult:
        """Check rate limit for a key."""
        return await self.rate_limiter.check(key, cost)

    async def get_rate_limit_status(self, key: str) -> Dict[str, Any]:
        """Get rate limit status for a key."""
        return self.rate_limiter.get_status(key)

    # =========================================================================
    # Tenancy
    # =========================================================================

    async def get_tenant(self, tenant_id: str) -> Optional[Tenant]:
        """Get a tenant by ID."""
        return await self.tenancy.get_tenant(tenant_id)

    async def create_tenant(
        self,
        name: str,
        slug: str,
        tier: TenantTier = TenantTier.FREE,
    ) -> Tenant:
        """Create a new tenant."""
        return await self.tenancy.create_tenant(name, slug, tier)

    async def check_tenant_quota(
        self,
        tenant_id: str,
        resource: str,
        amount: int = 1,
    ) -> Tuple[bool, Optional[str]]:
        """Check if tenant is within quota."""
        return await self.tenancy.check_quota(tenant_id, resource, amount)

    async def record_tenant_usage(
        self,
        tenant_id: str,
        resource: str,
        amount: int = 1,
    ) -> bool:
        """Record resource usage for a tenant."""
        return await self.tenancy.record_usage(tenant_id, resource, amount)

    # =========================================================================
    # Agent Security
    # =========================================================================

    async def check_agent_permission(
        self,
        agent_id: str,
        action: str,
        resource: Optional[str] = None,
    ) -> Tuple[bool, Optional[str]]:
        """Check if an agent has permission for an action."""
        if action == "use_tool":
            return await self.agent_boundaries.check_tool_access(agent_id, resource)
        elif action == "network_request":
            return await self.agent_boundaries.check_network_access(agent_id, resource)
        elif action == "access_memory":
            return await self.agent_boundaries.check_memory_access(agent_id, resource)
        elif action == "access_file":
            return await self.agent_boundaries.check_file_access(agent_id, resource)
        else:
            # Generic capability check
            from aion.security.types import AgentCapability

            try:
                capability = AgentCapability(action)
                return await self.agent_boundaries.check_capability(agent_id, capability)
            except ValueError:
                return True, None  # Unknown action, allow by default

    async def set_agent_boundary(
        self,
        agent_id: str,
        boundary: AgentPermissionBoundary,
    ) -> None:
        """Set permission boundary for an agent."""
        self.agent_boundaries.set_boundary(agent_id, boundary)
        await self.authz.set_agent_boundary(agent_id, boundary)

    # =========================================================================
    # User Management
    # =========================================================================

    async def create_user(
        self,
        username: str,
        email: str,
        password: str,
        roles: Optional[List[str]] = None,
        tenant_id: Optional[str] = None,
    ) -> User:
        """Create a new user."""
        return await self.auth.create_user(
            username=username,
            email=email,
            password=password,
            roles=roles,
            tenant_id=tenant_id,
        )

    async def get_user(self, user_id: str) -> Optional[User]:
        """Get a user by ID."""
        return await self.auth.get_user(user_id)

    async def get_user_by_email(self, email: str) -> Optional[User]:
        """Get a user by email."""
        return await self.auth.get_user_by_email(email)

    # =========================================================================
    # Role and Policy Management
    # =========================================================================

    async def create_role(
        self,
        name: str,
        display_name: str,
        permissions: List[Permission],
        description: str = "",
    ) -> Role:
        """Create a new role."""
        return await self.authz.create_role(
            name=name,
            display_name=display_name,
            description=description,
            permissions=permissions,
        )

    async def create_policy(
        self,
        name: str,
        resource_type: str,
        actions: List[PermissionAction],
        effect: PolicyEffect = PolicyEffect.ALLOW,
        roles: Optional[List[str]] = None,
        users: Optional[List[str]] = None,
    ) -> Policy:
        """Create a new policy."""
        return await self.authz.create_policy(
            name=name,
            resource_type=resource_type,
            actions=actions,
            effect=effect,
            roles=roles,
            users=users,
        )

    # =========================================================================
    # Secret Management
    # =========================================================================

    async def create_secret(
        self,
        name: str,
        value: str,
        secret_type: SecretType = SecretType.GENERIC,
    ) -> Secret:
        """Create a new secret."""
        return await self.secrets.create_secret(
            name=name,
            value=value,
            secret_type=secret_type,
        )

    async def get_secret(self, name: str, accessor: Optional[str] = None) -> Optional[str]:
        """Get a secret value."""
        return await self.secrets.get_secret(name, accessor=accessor)

    # =========================================================================
    # Audit
    # =========================================================================

    async def query_audit_logs(
        self,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        resource_type: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Query audit logs."""
        events = await self.audit.query(
            user_id=user_id,
            tenant_id=tenant_id,
            resource_type=resource_type,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
        )
        return [e.to_dict() for e in events]

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive security statistics."""
        return {
            "authentication": self.auth.get_stats(),
            "authorization": self.authz.get_stats(),
            "tenancy": self.tenancy.get_stats(),
            "audit": self.audit.get_stats(),
            "rate_limiting": self.rate_limiter.get_stats(),
            "agent_boundaries": self.agent_boundaries.get_stats(),
            "secrets": self.secrets.get_stats(),
        }
