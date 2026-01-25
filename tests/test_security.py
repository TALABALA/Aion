"""
AION Security Module Tests

Comprehensive tests for the security & access control system.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from aion.security import (
    # Manager
    SecurityManager,
    get_security_manager,
    set_security_manager,
    # Types
    AuthMethod,
    Credentials,
    User,
    UserStatus,
    Permission,
    PermissionAction,
    Role,
    Policy,
    PolicyEffect,
    PolicyCondition,
    ConditionOperator,
    SecurityContext,
    Tenant,
    TenantTier,
    TenantStatus,
    AgentCapability,
    AgentPermissionBoundary,
    NetworkPolicy,
    ResourceLimits,
    AuditEventType,
    SecretType,
    RateLimitConfig,
    # Services
    AuthenticationService,
    TokenManager,
    SessionManager,
    AuthorizationService,
    BUILTIN_ROLES,
    TenancyService,
    AuditLogger,
    RateLimiter,
    RateLimitStrategy,
    AgentBoundaryEnforcer,
    SecretManager,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def security_manager():
    """Create and initialize a security manager."""
    manager = SecurityManager(jwt_secret="test_secret_key_for_testing_only")
    await manager.initialize()
    yield manager
    await manager.shutdown()


@pytest.fixture
async def auth_service():
    """Create and initialize authentication service."""
    service = AuthenticationService(jwt_secret="test_secret")
    await service.initialize()
    yield service
    await service.shutdown()


@pytest.fixture
async def token_manager():
    """Create and initialize token manager."""
    manager = TokenManager(jwt_secret="test_secret")
    await manager.initialize()
    yield manager
    await manager.shutdown()


@pytest.fixture
async def session_manager():
    """Create and initialize session manager."""
    manager = SessionManager()
    await manager.initialize()
    yield manager
    await manager.shutdown()


@pytest.fixture
async def authz_service():
    """Create and initialize authorization service."""
    service = AuthorizationService()
    await service.initialize()
    yield service


@pytest.fixture
async def tenancy_service():
    """Create and initialize tenancy service."""
    service = TenancyService()
    await service.initialize()
    yield service


@pytest.fixture
async def audit_logger():
    """Create and initialize audit logger."""
    logger = AuditLogger()
    await logger.initialize()
    yield logger
    await logger.shutdown()


@pytest.fixture
async def rate_limiter():
    """Create and initialize rate limiter."""
    limiter = RateLimiter()
    await limiter.initialize()
    yield limiter
    await limiter.shutdown()


@pytest.fixture
def agent_enforcer():
    """Create agent boundary enforcer."""
    return AgentBoundaryEnforcer()


@pytest.fixture
async def secret_manager():
    """Create and initialize secret manager."""
    manager = SecretManager()
    await manager.initialize()
    yield manager


# ============================================================================
# Token Manager Tests
# ============================================================================


class TestTokenManager:
    """Tests for token management."""

    @pytest.mark.asyncio
    async def test_create_api_key(self, token_manager):
        """Test API key creation."""
        result = await token_manager.create_api_key(
            user_id="user_123",
            name="test-key",
            scopes=["read", "write"],
        )

        assert result.raw_token.startswith("aion_")
        assert result.token_record.user_id == "user_123"
        assert result.token_record.name == "test-key"
        assert "read" in result.token_record.scopes

    @pytest.mark.asyncio
    async def test_validate_api_key(self, token_manager):
        """Test API key validation."""
        # Create key
        result = await token_manager.create_api_key(
            user_id="user_123",
            name="test-key",
        )

        # Validate
        token = await token_manager.validate_api_key(result.raw_token)

        assert token is not None
        assert token.user_id == "user_123"

    @pytest.mark.asyncio
    async def test_invalid_api_key(self, token_manager):
        """Test invalid API key rejection."""
        token = await token_manager.validate_api_key("invalid_key")
        assert token is None

    @pytest.mark.asyncio
    async def test_revoke_token(self, token_manager):
        """Test token revocation."""
        result = await token_manager.create_api_key(
            user_id="user_123",
            name="test-key",
        )

        # Revoke
        success = await token_manager.revoke_token(result.token_record.token_id)
        assert success

        # Validation should fail
        token = await token_manager.validate_api_key(result.raw_token)
        assert token is None

    @pytest.mark.asyncio
    async def test_jwt_creation_and_validation(self, token_manager):
        """Test JWT token creation and validation."""
        user = User(id="user_123", username="testuser", email="test@test.com")

        # Create access token
        result = await token_manager.create_access_token(user)

        assert result.raw_token is not None
        assert "." in result.raw_token  # JWT format

        # Validate
        claims, error = await token_manager.validate_jwt(result.raw_token)

        assert error is None
        assert claims["sub"] == "user_123"
        assert claims["username"] == "testuser"


# ============================================================================
# Session Manager Tests
# ============================================================================


class TestSessionManager:
    """Tests for session management."""

    @pytest.mark.asyncio
    async def test_create_session(self, session_manager):
        """Test session creation."""
        session = await session_manager.create_session(
            user_id="user_123",
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0",
        )

        assert session.user_id == "user_123"
        assert session.ip_address == "192.168.1.1"
        assert session.is_active

    @pytest.mark.asyncio
    async def test_get_session(self, session_manager):
        """Test session retrieval."""
        session = await session_manager.create_session(user_id="user_123")

        retrieved = await session_manager.get_session(session.session_id)

        assert retrieved is not None
        assert retrieved.user_id == "user_123"

    @pytest.mark.asyncio
    async def test_terminate_session(self, session_manager):
        """Test session termination."""
        session = await session_manager.create_session(user_id="user_123")

        success = await session_manager.terminate_session(session.session_id)
        assert success

        # Should not be valid anymore
        retrieved = await session_manager.get_session(session.session_id)
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_session_elevation(self, session_manager):
        """Test session elevation."""
        session = await session_manager.create_session(user_id="user_123")

        # Not elevated initially
        assert not session.is_elevated

        # Elevate
        success = await session_manager.elevate_session(session.session_id)
        assert success

        # Check elevation
        session = await session_manager.get_session(session.session_id)
        assert session.is_elevated


# ============================================================================
# Authentication Service Tests
# ============================================================================


class TestAuthenticationService:
    """Tests for authentication service."""

    @pytest.mark.asyncio
    async def test_create_user(self, auth_service):
        """Test user creation."""
        user = await auth_service.create_user(
            username="testuser",
            email="test@example.com",
            password="SecurePass123!",
        )

        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert "user" in user.roles

    @pytest.mark.asyncio
    async def test_duplicate_user_rejected(self, auth_service):
        """Test that duplicate users are rejected."""
        await auth_service.create_user(
            username="testuser",
            email="test@example.com",
            password="SecurePass123!",
        )

        with pytest.raises(ValueError):
            await auth_service.create_user(
                username="testuser2",
                email="test@example.com",  # Same email
                password="SecurePass123!",
            )

    @pytest.mark.asyncio
    async def test_password_authentication(self, auth_service):
        """Test password authentication."""
        await auth_service.create_user(
            username="testuser",
            email="test@example.com",
            password="SecurePass123!",
            require_email_verification=False,
        )

        credentials = Credentials(
            method=AuthMethod.BASIC,
            username="testuser",
            password="SecurePass123!",
        )

        result = await auth_service.authenticate(credentials)

        assert result.status.value == "success"
        assert result.access_token is not None

    @pytest.mark.asyncio
    async def test_wrong_password_rejected(self, auth_service):
        """Test that wrong password is rejected."""
        await auth_service.create_user(
            username="testuser",
            email="test@example.com",
            password="SecurePass123!",
            require_email_verification=False,
        )

        credentials = Credentials(
            method=AuthMethod.BASIC,
            username="testuser",
            password="WrongPassword!",
        )

        result = await auth_service.authenticate(credentials)

        assert result.status.value == "failure"


# ============================================================================
# Authorization Service Tests
# ============================================================================


class TestAuthorizationService:
    """Tests for authorization service."""

    @pytest.mark.asyncio
    async def test_builtin_roles_exist(self, authz_service):
        """Test that built-in roles are created."""
        assert "admin" in [r.name for r in authz_service._roles.values()]
        assert "user" in [r.name for r in authz_service._roles.values()]

    @pytest.mark.asyncio
    async def test_admin_has_all_permissions(self, authz_service):
        """Test that admin role has all permissions."""
        context = SecurityContext(
            user_id="admin_user",
            roles=["admin"],
        )

        result = await authz_service.check_permission(
            context,
            "any_resource",
            PermissionAction.ALL,
        )

        assert result.status.value == "allowed"

    @pytest.mark.asyncio
    async def test_user_permissions(self, authz_service):
        """Test user role permissions."""
        context = SecurityContext(
            user_id="normal_user",
            roles=["user"],
        )

        # User can read agents
        result = await authz_service.check_permission(
            context,
            "agents",
            PermissionAction.READ,
        )
        assert result.status.value == "allowed"

        # User cannot create agents
        result = await authz_service.check_permission(
            context,
            "agents",
            PermissionAction.CREATE,
        )
        assert result.status.value == "denied"

    @pytest.mark.asyncio
    async def test_create_custom_role(self, authz_service):
        """Test custom role creation."""
        role = await authz_service.create_role(
            name="custom_role",
            display_name="Custom Role",
            permissions=[
                Permission(resource="custom", action=PermissionAction.ALL),
            ],
        )

        assert role.name == "custom_role"

        # Test permission
        context = SecurityContext(
            user_id="test_user",
            roles=["custom_role"],
        )

        result = await authz_service.check_permission(
            context,
            "custom",
            PermissionAction.CREATE,
        )
        assert result.status.value == "allowed"

    @pytest.mark.asyncio
    async def test_policy_enforcement(self, authz_service):
        """Test policy-based authorization."""
        # Create a deny policy
        await authz_service.create_policy(
            name="deny_dangerous",
            resource_type="dangerous_resource",
            actions=[PermissionAction.ALL],
            effect=PolicyEffect.DENY,
            roles=["user"],
        )

        context = SecurityContext(
            user_id="test_user",
            roles=["user", "admin"],  # Even admin is denied by policy
        )

        result = await authz_service.check_permission(
            context,
            "dangerous_resource",
            PermissionAction.EXECUTE,
        )

        # Deny policy should take precedence
        assert result.status.value == "denied"


# ============================================================================
# Tenancy Service Tests
# ============================================================================


class TestTenancyService:
    """Tests for multi-tenancy service."""

    @pytest.mark.asyncio
    async def test_create_tenant(self, tenancy_service):
        """Test tenant creation."""
        tenant = await tenancy_service.create_tenant(
            name="Test Company",
            slug="test-company",
            tier=TenantTier.PROFESSIONAL,
        )

        assert tenant.name == "Test Company"
        assert tenant.slug == "test-company"
        assert tenant.tier == TenantTier.PROFESSIONAL

    @pytest.mark.asyncio
    async def test_tenant_quotas(self, tenancy_service):
        """Test tenant quota enforcement."""
        tenant = await tenancy_service.create_tenant(
            name="Free Tier",
            slug="free-tier",
            tier=TenantTier.FREE,
        )

        # Check quota
        within, error = await tenancy_service.check_quota(
            tenant.id,
            "users",
            1,
        )
        assert within

        # Record usage up to limit
        for _ in range(5):
            await tenancy_service.record_usage(tenant.id, "users", 1)

        # Should exceed quota now
        within, error = await tenancy_service.check_quota(
            tenant.id,
            "users",
            1,
        )
        assert not within
        assert "exceeded" in error.lower()

    @pytest.mark.asyncio
    async def test_tenant_suspension(self, tenancy_service):
        """Test tenant suspension."""
        tenant = await tenancy_service.create_tenant(
            name="Test",
            slug="test-suspend",
        )

        await tenancy_service.suspend_tenant(tenant.id, "test_reason")

        tenant = await tenancy_service.get_tenant(tenant.id)
        assert tenant.status == TenantStatus.SUSPENDED


# ============================================================================
# Audit Logger Tests
# ============================================================================


class TestAuditLogger:
    """Tests for audit logging."""

    @pytest.mark.asyncio
    async def test_log_event(self, audit_logger):
        """Test basic event logging."""
        context = SecurityContext(user_id="user_123")

        event = audit_logger.log(
            event_type=AuditEventType.DATA_READ,
            description="Test event",
            context=context,
            resource_type="test",
            resource_id="123",
        )

        assert event.event_type == AuditEventType.DATA_READ
        assert event.actor_id == "user_123"

    @pytest.mark.asyncio
    async def test_query_events(self, audit_logger):
        """Test event querying."""
        context = SecurityContext(user_id="user_456")

        # Log multiple events
        for i in range(5):
            audit_logger.log(
                event_type=AuditEventType.DATA_READ,
                description=f"Event {i}",
                context=context,
            )

        # Flush and query
        await audit_logger._flush()

        events = await audit_logger.query(user_id="user_456")

        assert len(events) == 5

    @pytest.mark.asyncio
    async def test_auth_logging_helpers(self, audit_logger):
        """Test authentication logging helpers."""
        context = SecurityContext(user_id="user_123")

        # Success
        event = audit_logger.log_auth_success(context, "password")
        assert event.event_type == AuditEventType.AUTH_LOGIN_SUCCESS

        # Failure
        event = audit_logger.log_auth_failure("baduser", "invalid_password")
        assert event.event_type == AuditEventType.AUTH_LOGIN_FAILURE
        assert not event.action_result == "success"


# ============================================================================
# Rate Limiter Tests
# ============================================================================


class TestRateLimiter:
    """Tests for rate limiting."""

    @pytest.mark.asyncio
    async def test_basic_rate_limit(self, rate_limiter):
        """Test basic rate limiting."""
        key = "test_user_1"

        # First request should succeed
        result = await rate_limiter.check(key)
        assert result.allowed

    @pytest.mark.asyncio
    async def test_rate_limit_exceeded(self, rate_limiter):
        """Test that rate limit is enforced."""
        key = "test_user_2"

        # Configure low limit
        rate_limiter.set_config(
            "test_user_2",
            RateLimitConfig(
                requests_per_minute=5,
                bucket_size=5,
                refill_rate=0.1,
            ),
        )

        # Make requests up to limit
        for _ in range(5):
            result = await rate_limiter.check(key)
            assert result.allowed

        # Next should be denied
        result = await rate_limiter.check(key)
        assert not result.allowed

    @pytest.mark.asyncio
    async def test_rate_limit_status(self, rate_limiter):
        """Test rate limit status reporting."""
        key = "test_user_3"

        await rate_limiter.check(key)

        status = rate_limiter.get_status(key)

        assert "token_bucket" in status
        assert "fixed_windows" in status


# ============================================================================
# Agent Boundary Tests
# ============================================================================


class TestAgentBoundaryEnforcer:
    """Tests for agent permission boundaries."""

    @pytest.mark.asyncio
    async def test_tool_allowlist(self, agent_enforcer):
        """Test tool allowlisting."""
        await agent_enforcer.initialize()

        boundary = AgentPermissionBoundary(
            allowed_tools=["safe_tool", "another_safe"],
            denied_tools=["dangerous_tool"],
        )

        agent_enforcer.set_boundary("agent_1", boundary)

        # Allowed tool
        allowed, reason = await agent_enforcer.check_tool_access("agent_1", "safe_tool")
        assert allowed

        # Denied tool
        allowed, reason = await agent_enforcer.check_tool_access("agent_1", "dangerous_tool")
        assert not allowed

        # Not in list
        allowed, reason = await agent_enforcer.check_tool_access("agent_1", "unknown_tool")
        assert not allowed

    @pytest.mark.asyncio
    async def test_network_policy(self, agent_enforcer):
        """Test network access policy."""
        await agent_enforcer.initialize()

        boundary = AgentPermissionBoundary(
            network_policy=NetworkPolicy(
                allowed_domains=["api.safe.com", "*.trusted.org"],
                denied_domains=["evil.com"],
            ),
        )

        agent_enforcer.set_boundary("agent_1", boundary)

        # Allowed domain
        allowed, _ = await agent_enforcer.check_network_access("agent_1", "api.safe.com")
        assert allowed

        # Denied domain
        allowed, _ = await agent_enforcer.check_network_access("agent_1", "evil.com")
        assert not allowed

    @pytest.mark.asyncio
    async def test_capability_check(self, agent_enforcer):
        """Test capability checking."""
        await agent_enforcer.initialize()

        boundary = AgentPermissionBoundary(
            granted_capabilities={AgentCapability.MEMORY_READ, AgentCapability.MEMORY_WRITE},
            denied_capabilities={AgentCapability.TOOL_SHELL_EXECUTE},
        )

        agent_enforcer.set_boundary("agent_1", boundary)

        # Granted
        allowed, _ = await agent_enforcer.check_capability("agent_1", AgentCapability.MEMORY_READ)
        assert allowed

        # Denied
        allowed, _ = await agent_enforcer.check_capability("agent_1", AgentCapability.TOOL_SHELL_EXECUTE)
        assert not allowed


# ============================================================================
# Secret Manager Tests
# ============================================================================


class TestSecretManager:
    """Tests for secret management."""

    @pytest.mark.asyncio
    async def test_create_secret(self, secret_manager):
        """Test secret creation."""
        secret = await secret_manager.create_secret(
            name="api_key",
            value="super_secret_value",
            secret_type=SecretType.API_KEY,
        )

        assert secret.name == "api_key"
        assert secret.secret_type == SecretType.API_KEY

    @pytest.mark.asyncio
    async def test_get_secret(self, secret_manager):
        """Test secret retrieval."""
        await secret_manager.create_secret(
            name="test_secret",
            value="secret_value_123",
        )

        value = await secret_manager.get_secret("test_secret")

        assert value == "secret_value_123"

    @pytest.mark.asyncio
    async def test_update_secret(self, secret_manager):
        """Test secret update."""
        await secret_manager.create_secret(
            name="updatable",
            value="old_value",
        )

        success = await secret_manager.update_secret("updatable", "new_value")
        assert success

        value = await secret_manager.get_secret("updatable")
        assert value == "new_value"

    @pytest.mark.asyncio
    async def test_delete_secret(self, secret_manager):
        """Test secret deletion."""
        await secret_manager.create_secret(
            name="deletable",
            value="value",
        )

        success = await secret_manager.delete_secret("deletable")
        assert success

        value = await secret_manager.get_secret("deletable")
        assert value is None


# ============================================================================
# Security Manager Integration Tests
# ============================================================================


class TestSecurityManagerIntegration:
    """Integration tests for the security manager."""

    @pytest.mark.asyncio
    async def test_full_auth_flow(self, security_manager):
        """Test complete authentication flow."""
        # Create user
        user = await security_manager.create_user(
            username="integration_user",
            email="integration@test.com",
            password="SecurePass123!",
        )

        # Authenticate
        credentials = Credentials(
            method=AuthMethod.BASIC,
            username="integration_user",
            password="SecurePass123!",
        )

        result = await security_manager.authenticate(credentials)

        assert result.status.value == "success"
        assert result.context.user_id == user.id

    @pytest.mark.asyncio
    async def test_authorization_check(self, security_manager):
        """Test authorization checking."""
        # Create user
        user = await security_manager.create_user(
            username="auth_test_user",
            email="auth_test@test.com",
            password="SecurePass123!",
            roles=["user"],
        )

        context = SecurityContext(
            user_id=user.id,
            roles=user.roles,
        )

        # Should be able to read conversations
        allowed = await security_manager.authorize(
            context,
            "conversations",
            PermissionAction.READ,
        )
        assert allowed

        # Should not be able to admin
        allowed = await security_manager.authorize(
            context,
            "system",
            PermissionAction.ADMIN,
        )
        assert not allowed

    @pytest.mark.asyncio
    async def test_rate_limiting_integration(self, security_manager):
        """Test rate limiting through security manager."""
        key = "integration_test_key"

        result = await security_manager.check_rate_limit(key)

        assert result.allowed
        assert result.remaining >= 0

    @pytest.mark.asyncio
    async def test_stats_retrieval(self, security_manager):
        """Test statistics retrieval."""
        stats = security_manager.get_stats()

        assert "authentication" in stats
        assert "authorization" in stats
        assert "tenancy" in stats
        assert "audit" in stats
        assert "rate_limiting" in stats


# ============================================================================
# Policy Condition Tests
# ============================================================================


class TestPolicyConditions:
    """Tests for policy conditions."""

    def test_equals_condition(self):
        """Test equals condition."""
        condition = PolicyCondition(
            field="user.role",
            operator=ConditionOperator.EQUALS,
            value="admin",
        )

        context = {"user": {"role": "admin"}}
        assert condition.evaluate(context)

        context = {"user": {"role": "user"}}
        assert not condition.evaluate(context)

    def test_in_condition(self):
        """Test IN condition."""
        condition = PolicyCondition(
            field="resource.type",
            operator=ConditionOperator.IN,
            value=["agent", "goal", "tool"],
        )

        context = {"resource": {"type": "agent"}}
        assert condition.evaluate(context)

        context = {"resource": {"type": "secret"}}
        assert not condition.evaluate(context)

    def test_greater_than_condition(self):
        """Test greater than condition."""
        condition = PolicyCondition(
            field="request.size",
            operator=ConditionOperator.GREATER_THAN,
            value=100,
        )

        context = {"request": {"size": 150}}
        assert condition.evaluate(context)

        context = {"request": {"size": 50}}
        assert not condition.evaluate(context)

    def test_negated_condition(self):
        """Test negated condition."""
        condition = PolicyCondition(
            field="user.suspended",
            operator=ConditionOperator.EQUALS,
            value=True,
            negate=True,
        )

        context = {"user": {"suspended": False}}
        assert condition.evaluate(context)

        context = {"user": {"suspended": True}}
        assert not condition.evaluate(context)


# ============================================================================
# Permission Tests
# ============================================================================


class TestPermissions:
    """Tests for permission matching."""

    def test_exact_match(self):
        """Test exact permission matching."""
        perm = Permission(resource="agents", action=PermissionAction.READ)

        assert perm.matches("agents", PermissionAction.READ)
        assert not perm.matches("agents", PermissionAction.WRITE)
        assert not perm.matches("goals", PermissionAction.READ)

    def test_wildcard_resource(self):
        """Test wildcard resource matching."""
        perm = Permission(resource="*", action=PermissionAction.READ)

        assert perm.matches("agents", PermissionAction.READ)
        assert perm.matches("goals", PermissionAction.READ)
        assert not perm.matches("agents", PermissionAction.WRITE)

    def test_wildcard_action(self):
        """Test wildcard action matching."""
        perm = Permission(resource="agents", action=PermissionAction.ALL)

        assert perm.matches("agents", PermissionAction.READ)
        assert perm.matches("agents", PermissionAction.WRITE)
        assert perm.matches("agents", PermissionAction.DELETE)

    def test_permission_string_conversion(self):
        """Test permission string conversion."""
        perm = Permission(resource="agents", action=PermissionAction.EXECUTE)

        assert perm.to_string() == "agents:execute"

        parsed = Permission.from_string("agents:execute")
        assert parsed.resource == "agents"
        assert parsed.action == PermissionAction.EXECUTE
