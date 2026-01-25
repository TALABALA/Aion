"""
AION Tenancy Service

Multi-tenant isolation with resource quotas and usage tracking.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import structlog

from aion.security.types import (
    Tenant,
    TenantQuotas,
    TenantStatus,
    TenantTier,
    TenantUsage,
)

logger = structlog.get_logger(__name__)


# Default quotas by tier
TIER_QUOTAS: Dict[TenantTier, TenantQuotas] = {
    TenantTier.FREE: TenantQuotas(
        max_users=5,
        max_service_accounts=2,
        max_api_keys_per_user=2,
        max_agents=2,
        max_concurrent_agents=1,
        max_storage_gb=1.0,
        max_memory_vectors=10000,
        max_knowledge_nodes=1000,
        max_requests_per_minute=60,
        max_requests_per_hour=1000,
        max_requests_per_day=10000,
        max_tokens_per_request=10000,
        max_tokens_per_day=100000,
        max_tool_executions_per_day=100,
    ),
    TenantTier.STARTER: TenantQuotas(
        max_users=25,
        max_service_accounts=5,
        max_api_keys_per_user=5,
        max_agents=5,
        max_concurrent_agents=2,
        max_storage_gb=10.0,
        max_memory_vectors=100000,
        max_knowledge_nodes=10000,
        max_requests_per_minute=300,
        max_requests_per_hour=5000,
        max_requests_per_day=50000,
        max_tokens_per_request=50000,
        max_tokens_per_day=1000000,
        max_tool_executions_per_day=1000,
    ),
    TenantTier.PROFESSIONAL: TenantQuotas(
        max_users=100,
        max_service_accounts=20,
        max_api_keys_per_user=10,
        max_agents=20,
        max_concurrent_agents=10,
        max_storage_gb=100.0,
        max_memory_vectors=1000000,
        max_knowledge_nodes=100000,
        max_requests_per_minute=1000,
        max_requests_per_hour=20000,
        max_requests_per_day=200000,
        max_tokens_per_request=100000,
        max_tokens_per_day=10000000,
        max_tool_executions_per_day=10000,
    ),
    TenantTier.ENTERPRISE: TenantQuotas(
        max_users=1000,
        max_service_accounts=100,
        max_api_keys_per_user=20,
        max_agents=100,
        max_concurrent_agents=50,
        max_storage_gb=1000.0,
        max_memory_vectors=10000000,
        max_knowledge_nodes=1000000,
        max_requests_per_minute=5000,
        max_requests_per_hour=100000,
        max_requests_per_day=1000000,
        max_tokens_per_request=200000,
        max_tokens_per_day=100000000,
        max_tool_executions_per_day=100000,
    ),
    TenantTier.UNLIMITED: TenantQuotas(
        max_users=999999,
        max_service_accounts=999999,
        max_api_keys_per_user=999,
        max_agents=999999,
        max_concurrent_agents=999,
        max_storage_gb=999999.0,
        max_memory_vectors=999999999,
        max_knowledge_nodes=999999999,
        max_requests_per_minute=999999,
        max_requests_per_hour=999999,
        max_requests_per_day=999999999,
        max_tokens_per_request=999999,
        max_tokens_per_day=999999999999,
        max_tool_executions_per_day=999999999,
    ),
}


class TenancyService:
    """
    Multi-tenant management service.

    Features:
    - Tenant lifecycle management
    - Resource quota enforcement
    - Usage tracking and billing
    - Tenant isolation
    - Feature flags per tenant
    """

    def __init__(self):
        self._tenants: Dict[str, Tenant] = {}
        self._tenants_by_slug: Dict[str, str] = {}
        self._tenants_by_domain: Dict[str, str] = {}

        # Usage tracking
        self._usage_updates: List[Dict[str, Any]] = []

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize tenancy service."""
        if self._initialized:
            return

        logger.info("Initializing Tenancy Service")

        # Create default tenant
        if not self._tenants:
            await self.create_tenant(
                name="Default",
                slug="default",
                tier=TenantTier.PROFESSIONAL,
            )

        self._initialized = True

    # =========================================================================
    # Tenant Lifecycle
    # =========================================================================

    async def create_tenant(
        self,
        name: str,
        slug: str,
        tier: TenantTier = TenantTier.FREE,
        domain: Optional[str] = None,
        settings: Optional[Dict[str, Any]] = None,
        trial_days: int = 14,
    ) -> Tenant:
        """Create a new tenant."""
        # Validate slug
        if not self._validate_slug(slug):
            raise ValueError("Invalid slug format")

        if slug.lower() in self._tenants_by_slug:
            raise ValueError(f"Slug already exists: {slug}")

        if domain and domain.lower() in self._tenants_by_domain:
            raise ValueError(f"Domain already exists: {domain}")

        # Get quotas for tier
        quotas = TIER_QUOTAS.get(tier, TIER_QUOTAS[TenantTier.FREE])

        tenant = Tenant(
            name=name,
            slug=slug.lower(),
            display_name=name,
            domain=domain.lower() if domain else None,
            tier=tier,
            quotas=quotas,
            usage=TenantUsage(),
            status=TenantStatus.TRIAL if tier == TenantTier.FREE else TenantStatus.ACTIVE,
            trial_ends_at=datetime.now() + timedelta(days=trial_days) if tier == TenantTier.FREE else None,
            settings=settings or {},
        )

        self._tenants[tenant.id] = tenant
        self._tenants_by_slug[slug.lower()] = tenant.id
        if domain:
            self._tenants_by_domain[domain.lower()] = tenant.id

        logger.info("Tenant created", tenant_id=tenant.id, name=name, tier=tier.value)

        return tenant

    async def get_tenant(self, tenant_id: str) -> Optional[Tenant]:
        """Get a tenant by ID."""
        return self._tenants.get(tenant_id)

    async def get_tenant_by_slug(self, slug: str) -> Optional[Tenant]:
        """Get a tenant by slug."""
        tenant_id = self._tenants_by_slug.get(slug.lower())
        return self._tenants.get(tenant_id) if tenant_id else None

    async def get_tenant_by_domain(self, domain: str) -> Optional[Tenant]:
        """Get a tenant by custom domain."""
        tenant_id = self._tenants_by_domain.get(domain.lower())
        return self._tenants.get(tenant_id) if tenant_id else None

    async def update_tenant(self, tenant: Tenant) -> None:
        """Update a tenant."""
        tenant.updated_at = datetime.now()
        self._tenants[tenant.id] = tenant

    async def suspend_tenant(
        self,
        tenant_id: str,
        reason: str = "policy_violation",
    ) -> bool:
        """Suspend a tenant."""
        tenant = self._tenants.get(tenant_id)
        if not tenant:
            return False

        tenant.status = TenantStatus.SUSPENDED
        tenant.status_reason = reason
        tenant.updated_at = datetime.now()

        logger.warning("Tenant suspended", tenant_id=tenant_id, reason=reason)

        return True

    async def activate_tenant(self, tenant_id: str) -> bool:
        """Activate a tenant."""
        tenant = self._tenants.get(tenant_id)
        if not tenant:
            return False

        tenant.status = TenantStatus.ACTIVE
        tenant.status_reason = None
        tenant.activated_at = datetime.now()
        tenant.updated_at = datetime.now()

        return True

    async def upgrade_tier(
        self,
        tenant_id: str,
        new_tier: TenantTier,
    ) -> bool:
        """Upgrade tenant to a new tier."""
        tenant = self._tenants.get(tenant_id)
        if not tenant:
            return False

        old_tier = tenant.tier
        tenant.tier = new_tier
        tenant.quotas = TIER_QUOTAS.get(new_tier, tenant.quotas)

        # Clear trial if upgrading
        if old_tier == TenantTier.FREE and new_tier != TenantTier.FREE:
            tenant.status = TenantStatus.ACTIVE
            tenant.trial_ends_at = None

        tenant.updated_at = datetime.now()

        logger.info(
            "Tenant tier upgraded",
            tenant_id=tenant_id,
            old_tier=old_tier.value,
            new_tier=new_tier.value,
        )

        return True

    # =========================================================================
    # Quota Management
    # =========================================================================

    async def check_quota(
        self,
        tenant_id: str,
        resource: str,
        amount: int = 1,
    ) -> tuple[bool, Optional[str]]:
        """
        Check if tenant is within quota for a resource.

        Returns (within_quota, error_message).
        """
        tenant = self._tenants.get(tenant_id)
        if not tenant:
            return False, "Tenant not found"

        if not tenant.is_active():
            return False, f"Tenant is {tenant.status.value}"

        # Check trial expiration
        if tenant.status == TenantStatus.TRIAL:
            if tenant.trial_ends_at and datetime.now() > tenant.trial_ends_at:
                return False, "Trial has expired"

        # Check specific quotas
        quota_checks = {
            "users": (tenant.usage.current_users, tenant.quotas.max_users),
            "service_accounts": (tenant.usage.current_service_accounts, tenant.quotas.max_service_accounts),
            "storage_gb": (tenant.usage.current_storage_gb, tenant.quotas.max_storage_gb),
            "memory_vectors": (tenant.usage.current_memory_vectors, tenant.quotas.max_memory_vectors),
            "knowledge_nodes": (tenant.usage.current_knowledge_nodes, tenant.quotas.max_knowledge_nodes),
            "tokens_today": (tenant.usage.tokens_used_today, tenant.quotas.max_tokens_per_day),
            "requests_today": (tenant.usage.requests_today, tenant.quotas.max_requests_per_day),
            "tool_executions_today": (tenant.usage.tool_executions_today, tenant.quotas.max_tool_executions_per_day),
        }

        if resource in quota_checks:
            current, limit = quota_checks[resource]
            if current + amount > limit:
                return False, f"Quota exceeded for {resource}: {current}/{limit}"

        return True, None

    async def record_usage(
        self,
        tenant_id: str,
        resource: str,
        amount: int = 1,
    ) -> bool:
        """Record resource usage for a tenant."""
        tenant = self._tenants.get(tenant_id)
        if not tenant:
            return False

        # Reset daily counters if needed
        now = datetime.now()
        if now.date() > tenant.usage.daily_reset_at.date():
            tenant.usage.tokens_used_today = 0
            tenant.usage.requests_today = 0
            tenant.usage.tool_executions_today = 0
            tenant.usage.daily_reset_at = now

        # Update usage
        if resource == "tokens":
            tenant.usage.tokens_used_today += amount
            tenant.usage.tokens_used_this_month += amount
        elif resource == "requests":
            tenant.usage.requests_today += amount
            tenant.usage.requests_this_month += amount
        elif resource == "tool_executions":
            tenant.usage.tool_executions_today += amount
        elif resource == "memory_vectors":
            tenant.usage.current_memory_vectors += amount
        elif resource == "knowledge_nodes":
            tenant.usage.current_knowledge_nodes += amount
        elif resource == "storage_gb":
            tenant.usage.current_storage_gb += amount
        elif resource == "users":
            tenant.usage.current_users += amount
        elif resource == "service_accounts":
            tenant.usage.current_service_accounts += amount

        return True

    async def get_usage(self, tenant_id: str) -> Optional[TenantUsage]:
        """Get usage for a tenant."""
        tenant = self._tenants.get(tenant_id)
        return tenant.usage if tenant else None

    # =========================================================================
    # Feature Flags
    # =========================================================================

    async def set_feature(
        self,
        tenant_id: str,
        feature: str,
        enabled: bool,
    ) -> bool:
        """Set a feature flag for a tenant."""
        tenant = self._tenants.get(tenant_id)
        if not tenant:
            return False

        tenant.features[feature] = enabled
        tenant.updated_at = datetime.now()

        return True

    async def check_feature(
        self,
        tenant_id: str,
        feature: str,
        default: bool = False,
    ) -> bool:
        """Check if a feature is enabled for a tenant."""
        tenant = self._tenants.get(tenant_id)
        if not tenant:
            return default

        return tenant.features.get(feature, default)

    # =========================================================================
    # Tenant Settings
    # =========================================================================

    async def get_setting(
        self,
        tenant_id: str,
        key: str,
        default: Any = None,
    ) -> Any:
        """Get a tenant setting."""
        tenant = self._tenants.get(tenant_id)
        if not tenant:
            return default

        return tenant.settings.get(key, default)

    async def set_setting(
        self,
        tenant_id: str,
        key: str,
        value: Any,
    ) -> bool:
        """Set a tenant setting."""
        tenant = self._tenants.get(tenant_id)
        if not tenant:
            return False

        tenant.settings[key] = value
        tenant.updated_at = datetime.now()

        return True

    # =========================================================================
    # Helpers
    # =========================================================================

    def _validate_slug(self, slug: str) -> bool:
        """Validate tenant slug format."""
        if not slug or len(slug) < 3 or len(slug) > 63:
            return False
        return bool(re.match(r"^[a-z0-9][a-z0-9-]*[a-z0-9]$", slug.lower()))

    def get_stats(self) -> Dict[str, Any]:
        """Get tenancy service statistics."""
        by_tier = {}
        by_status = {}

        for tenant in self._tenants.values():
            tier = tenant.tier.value
            by_tier[tier] = by_tier.get(tier, 0) + 1

            status = tenant.status.value
            by_status[status] = by_status.get(status, 0) + 1

        return {
            "total_tenants": len(self._tenants),
            "by_tier": by_tier,
            "by_status": by_status,
        }
