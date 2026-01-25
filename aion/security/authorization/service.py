"""
AION Authorization Service

Enterprise-grade authorization with RBAC, ABAC, and policy-based access control.
"""

from __future__ import annotations

import fnmatch
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import structlog

from aion.security.types import (
    AgentPermissionBoundary,
    AuthorizationResult,
    AuthorizationResultStatus,
    Permission,
    PermissionAction,
    Policy,
    PolicyCondition,
    PolicyEffect,
    PolicyEvaluationOrder,
    Role,
    SecurityContext,
    User,
)

logger = structlog.get_logger(__name__)


# =============================================================================
# Built-in Roles
# =============================================================================

BUILTIN_ROLES: Dict[str, Role] = {
    "admin": Role(
        id="role_admin",
        name="admin",
        display_name="Administrator",
        description="Full system access with all permissions",
        permissions=[
            Permission(resource="*", action=PermissionAction.ALL),
        ],
        system_role=True,
    ),
    "user": Role(
        id="role_user",
        name="user",
        display_name="User",
        description="Standard user access",
        permissions=[
            Permission(resource="conversations", action=PermissionAction.ALL),
            Permission(resource="memory", action=PermissionAction.ALL),
            Permission(resource="goals", action=PermissionAction.ALL),
            Permission(resource="agents", action=PermissionAction.READ),
            Permission(resource="agents", action=PermissionAction.EXECUTE),
            Permission(resource="tools", action=PermissionAction.READ),
            Permission(resource="tools", action=PermissionAction.EXECUTE),
            Permission(resource="knowledge", action=PermissionAction.READ),
        ],
        system_role=True,
    ),
    "operator": Role(
        id="role_operator",
        name="operator",
        display_name="Operator",
        description="System operator with elevated permissions",
        permissions=[
            Permission(resource="agents", action=PermissionAction.ALL),
            Permission(resource="goals", action=PermissionAction.ALL),
            Permission(resource="tools", action=PermissionAction.ALL),
            Permission(resource="memory", action=PermissionAction.ALL),
            Permission(resource="knowledge", action=PermissionAction.ALL),
            Permission(resource="config", action=PermissionAction.READ),
            Permission(resource="metrics", action=PermissionAction.READ),
        ],
        system_role=True,
    ),
    "readonly": Role(
        id="role_readonly",
        name="readonly",
        display_name="Read Only",
        description="Read-only access to all resources",
        permissions=[
            Permission(resource="*", action=PermissionAction.READ),
            Permission(resource="*", action=PermissionAction.LIST),
        ],
        system_role=True,
    ),
    "agent": Role(
        id="role_agent",
        name="agent",
        display_name="Agent",
        description="AI agent execution role",
        permissions=[
            Permission(resource="tools", action=PermissionAction.EXECUTE),
            Permission(resource="memory", action=PermissionAction.ALL),
            Permission(resource="knowledge", action=PermissionAction.READ),
            Permission(resource="knowledge", action=PermissionAction.CREATE),
        ],
        system_role=True,
    ),
    "service": Role(
        id="role_service",
        name="service",
        display_name="Service Account",
        description="Service account for machine-to-machine access",
        permissions=[
            Permission(resource="api", action=PermissionAction.ALL),
        ],
        system_role=True,
    ),
}


class AuthorizationService:
    """
    Authorization service with multiple access control models.

    Features:
    - Role-Based Access Control (RBAC)
    - Attribute-Based Access Control (ABAC)
    - Policy-based authorization
    - Hierarchical roles
    - Resource-level permissions
    - Agent permission boundaries
    - Audit integration
    """

    def __init__(
        self,
        evaluation_order: PolicyEvaluationOrder = PolicyEvaluationOrder.DENY_OVERRIDES,
        default_deny: bool = True,
    ):
        self.evaluation_order = evaluation_order
        self.default_deny = default_deny

        # Roles storage
        self._roles: Dict[str, Role] = {}
        self._roles_by_name: Dict[str, str] = {}  # name -> id

        # Policies storage
        self._policies: Dict[str, Policy] = {}
        self._policies_by_resource: Dict[str, List[str]] = {}  # resource -> [policy_ids]

        # Agent boundaries
        self._agent_boundaries: Dict[str, AgentPermissionBoundary] = {}

        # Permission cache
        self._permission_cache: Dict[str, Dict[str, bool]] = {}
        self._cache_ttl = 60  # seconds
        self._cache_timestamps: Dict[str, datetime] = {}

        # Event handlers
        self._decision_handlers: List[Callable] = []

        # Initialize built-in roles
        self._initialize_builtin_roles()

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize authorization service."""
        if self._initialized:
            return

        logger.info("Initializing Authorization Service")
        self._initialized = True

    def _initialize_builtin_roles(self) -> None:
        """Initialize built-in roles."""
        for role_name, role in BUILTIN_ROLES.items():
            self._roles[role.id] = role
            self._roles_by_name[role.name] = role.id

    # =========================================================================
    # Permission Checking
    # =========================================================================

    async def check_permission(
        self,
        context: SecurityContext,
        resource: str,
        action: PermissionAction,
        resource_id: Optional[str] = None,
        resource_attrs: Optional[Dict[str, Any]] = None,
    ) -> AuthorizationResult:
        """
        Check if context has permission for an action.

        This is the main authorization entry point.
        """
        start_time = datetime.now()

        # Check cache first
        cache_key = self._get_cache_key(context, resource, action, resource_id)
        cached = self._check_cache(cache_key)
        if cached is not None:
            return cached

        # Build evaluation context
        eval_context = self._build_eval_context(context, resource, resource_id, resource_attrs)

        # Check role-based permissions first
        role_result = await self._check_role_permissions(context, resource, action, eval_context)

        if role_result.status == AuthorizationResultStatus.ALLOWED:
            # Role allows - check if any policy denies
            policy_result = await self._evaluate_policies(
                context, resource, action, resource_id, eval_context
            )

            if policy_result.status == AuthorizationResultStatus.DENIED:
                result = policy_result
            else:
                result = role_result
        else:
            # Role doesn't allow - check if any policy allows
            policy_result = await self._evaluate_policies(
                context, resource, action, resource_id, eval_context
            )
            result = policy_result

        # Apply default deny if no decision
        if result.status not in (AuthorizationResultStatus.ALLOWED, AuthorizationResultStatus.DENIED):
            if self.default_deny:
                result = AuthorizationResult(
                    status=AuthorizationResultStatus.DENIED,
                    resource=resource,
                    action=action.value,
                    decided_by="default_deny",
                    reason="No matching permission or policy",
                )
            else:
                result = AuthorizationResult(
                    status=AuthorizationResultStatus.ALLOWED,
                    resource=resource,
                    action=action.value,
                    decided_by="default_allow",
                )

        # Cache result
        self._cache_result(cache_key, result)

        # Emit decision event
        await self._emit_decision(context, resource, action, resource_id, result)

        return result

    async def _check_role_permissions(
        self,
        context: SecurityContext,
        resource: str,
        action: PermissionAction,
        eval_context: Dict[str, Any],
    ) -> AuthorizationResult:
        """Check permissions based on roles."""
        for role_name in context.roles:
            role = self._get_role_by_name(role_name)
            if not role:
                continue

            # Check role permissions including inherited ones
            has_perm = await self._role_has_permission(role, resource, action, eval_context)

            if has_perm:
                return AuthorizationResult(
                    status=AuthorizationResultStatus.ALLOWED,
                    resource=resource,
                    action=action.value,
                    decided_by="role",
                    role_name=role.name,
                )

        return AuthorizationResult(
            status=AuthorizationResultStatus.DENIED,
            resource=resource,
            action=action.value,
            decided_by="role",
            reason="No role grants this permission",
            missing_permissions=[f"{resource}:{action.value}"],
        )

    async def _role_has_permission(
        self,
        role: Role,
        resource: str,
        action: PermissionAction,
        eval_context: Dict[str, Any],
    ) -> bool:
        """Check if a role has a specific permission."""
        # Check direct permissions
        for perm in role.permissions:
            if perm.matches(resource, action):
                # Check conditions if any
                if perm.conditions:
                    if all(c.evaluate(eval_context) for c in perm.conditions):
                        return True
                else:
                    return True

        # Check parent role
        if role.parent_role_id:
            parent = self._roles.get(role.parent_role_id)
            if parent:
                return await self._role_has_permission(parent, resource, action, eval_context)

        return False

    async def _evaluate_policies(
        self,
        context: SecurityContext,
        resource: str,
        action: PermissionAction,
        resource_id: Optional[str],
        eval_context: Dict[str, Any],
    ) -> AuthorizationResult:
        """Evaluate policies for authorization decision."""
        # Get applicable policies
        applicable_policies = self._get_applicable_policies(
            context, resource, action, resource_id
        )

        if not applicable_policies:
            return AuthorizationResult(
                status=AuthorizationResultStatus.DENIED,
                resource=resource,
                action=action.value,
                decided_by="no_policy",
            )

        # Sort by priority
        applicable_policies.sort(key=lambda p: p.priority, reverse=True)

        allow_policies = []
        deny_policies = []

        for policy in applicable_policies:
            # Evaluate conditions
            if policy.conditions:
                conditions_met = all(c.evaluate(eval_context) for c in policy.conditions)
                if not conditions_met:
                    continue

            if policy.effect == PolicyEffect.ALLOW:
                allow_policies.append(policy)
            else:
                deny_policies.append(policy)

        # Apply evaluation order
        if self.evaluation_order == PolicyEvaluationOrder.DENY_OVERRIDES:
            if deny_policies:
                policy = deny_policies[0]
                return AuthorizationResult(
                    status=AuthorizationResultStatus.DENIED,
                    resource=resource,
                    action=action.value,
                    decided_by="policy",
                    policy_id=policy.id,
                    reason=f"Denied by policy: {policy.name}",
                )
            if allow_policies:
                policy = allow_policies[0]
                return AuthorizationResult(
                    status=AuthorizationResultStatus.ALLOWED,
                    resource=resource,
                    action=action.value,
                    decided_by="policy",
                    policy_id=policy.id,
                )

        elif self.evaluation_order == PolicyEvaluationOrder.ALLOW_OVERRIDES:
            if allow_policies:
                policy = allow_policies[0]
                return AuthorizationResult(
                    status=AuthorizationResultStatus.ALLOWED,
                    resource=resource,
                    action=action.value,
                    decided_by="policy",
                    policy_id=policy.id,
                )
            if deny_policies:
                policy = deny_policies[0]
                return AuthorizationResult(
                    status=AuthorizationResultStatus.DENIED,
                    resource=resource,
                    action=action.value,
                    decided_by="policy",
                    policy_id=policy.id,
                    reason=f"Denied by policy: {policy.name}",
                )

        elif self.evaluation_order == PolicyEvaluationOrder.FIRST_MATCH:
            if applicable_policies:
                policy = applicable_policies[0]
                return AuthorizationResult(
                    status=(
                        AuthorizationResultStatus.ALLOWED
                        if policy.effect == PolicyEffect.ALLOW
                        else AuthorizationResultStatus.DENIED
                    ),
                    resource=resource,
                    action=action.value,
                    decided_by="policy",
                    policy_id=policy.id,
                )

        return AuthorizationResult(
            status=AuthorizationResultStatus.DENIED,
            resource=resource,
            action=action.value,
            decided_by="no_matching_policy",
        )

    def _get_applicable_policies(
        self,
        context: SecurityContext,
        resource: str,
        action: PermissionAction,
        resource_id: Optional[str],
    ) -> List[Policy]:
        """Get policies applicable to this request."""
        applicable = []

        for policy in self._policies.values():
            if not policy.is_active():
                continue

            # Check resource match
            if not policy.matches_resource(resource, resource_id):
                continue

            # Check action match
            if policy.actions and action not in policy.actions:
                if PermissionAction.ALL not in policy.actions:
                    continue

            # Check subject match
            if not policy.matches_subject(
                context.user_id,
                context.roles,
                context.service_account_id,
            ):
                continue

            # Check tenant
            if policy.tenant_id and policy.tenant_id != context.tenant_id:
                continue

            applicable.append(policy)

        return applicable

    # =========================================================================
    # Role Management
    # =========================================================================

    async def create_role(
        self,
        name: str,
        display_name: str,
        description: str = "",
        permissions: Optional[List[Permission]] = None,
        parent_role_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> Role:
        """Create a new role."""
        if name.lower() in self._roles_by_name:
            raise ValueError(f"Role already exists: {name}")

        role = Role(
            name=name,
            display_name=display_name,
            description=description,
            permissions=permissions or [],
            parent_role_id=parent_role_id,
            tenant_id=tenant_id,
        )

        self._roles[role.id] = role
        self._roles_by_name[name.lower()] = role.id

        # Clear cache
        self._clear_cache()

        logger.info("Role created", role_id=role.id, name=name)

        return role

    async def get_role(self, role_id: str) -> Optional[Role]:
        """Get a role by ID."""
        return self._roles.get(role_id)

    def _get_role_by_name(self, name: str) -> Optional[Role]:
        """Get a role by name."""
        role_id = self._roles_by_name.get(name.lower())
        return self._roles.get(role_id) if role_id else None

    async def update_role(self, role: Role) -> None:
        """Update a role."""
        if role.system_role:
            raise ValueError("Cannot modify system roles")

        role.updated_at = datetime.now()
        self._roles[role.id] = role
        self._clear_cache()

    async def delete_role(self, role_id: str) -> bool:
        """Delete a role."""
        role = self._roles.get(role_id)
        if not role:
            return False

        if role.system_role:
            raise ValueError("Cannot delete system roles")

        del self._roles[role_id]
        self._roles_by_name.pop(role.name.lower(), None)
        self._clear_cache()

        return True

    async def add_permission_to_role(
        self,
        role_id: str,
        permission: Permission,
    ) -> bool:
        """Add a permission to a role."""
        role = self._roles.get(role_id)
        if not role or role.system_role:
            return False

        role.permissions.append(permission)
        role.updated_at = datetime.now()
        self._clear_cache()

        return True

    async def remove_permission_from_role(
        self,
        role_id: str,
        resource: str,
        action: PermissionAction,
    ) -> bool:
        """Remove a permission from a role."""
        role = self._roles.get(role_id)
        if not role or role.system_role:
            return False

        role.permissions = [
            p for p in role.permissions
            if not (p.resource == resource and p.action == action)
        ]
        role.updated_at = datetime.now()
        self._clear_cache()

        return True

    async def get_user_permissions(self, context: SecurityContext) -> Set[str]:
        """Get all effective permissions for a user."""
        permissions = set()

        for role_name in context.roles:
            role = self._get_role_by_name(role_name)
            if role:
                permissions.update(await self._get_role_permissions(role))

        return permissions

    async def _get_role_permissions(self, role: Role) -> Set[str]:
        """Get all permissions for a role including inherited ones."""
        permissions = {p.to_string() for p in role.permissions}

        if role.parent_role_id:
            parent = self._roles.get(role.parent_role_id)
            if parent:
                permissions.update(await self._get_role_permissions(parent))

        return permissions

    # =========================================================================
    # Policy Management
    # =========================================================================

    async def create_policy(
        self,
        name: str,
        resource_type: str,
        actions: List[PermissionAction],
        effect: PolicyEffect = PolicyEffect.ALLOW,
        roles: Optional[List[str]] = None,
        users: Optional[List[str]] = None,
        conditions: Optional[List[PolicyCondition]] = None,
        resource_pattern: str = "*",
        priority: int = 0,
        tenant_id: Optional[str] = None,
        description: str = "",
    ) -> Policy:
        """Create a new policy."""
        policy = Policy(
            name=name,
            description=description,
            effect=effect,
            resource_type=resource_type,
            resource_pattern=resource_pattern,
            actions=actions,
            roles=roles or [],
            users=users or [],
            conditions=conditions or [],
            priority=priority,
            tenant_id=tenant_id,
        )

        self._policies[policy.id] = policy

        # Index by resource
        if resource_type not in self._policies_by_resource:
            self._policies_by_resource[resource_type] = []
        self._policies_by_resource[resource_type].append(policy.id)

        self._clear_cache()

        logger.info("Policy created", policy_id=policy.id, name=name)

        return policy

    async def get_policy(self, policy_id: str) -> Optional[Policy]:
        """Get a policy by ID."""
        return self._policies.get(policy_id)

    async def update_policy(self, policy: Policy) -> None:
        """Update a policy."""
        policy.updated_at = datetime.now()
        policy.version += 1
        self._policies[policy.id] = policy
        self._clear_cache()

    async def delete_policy(self, policy_id: str) -> bool:
        """Delete a policy."""
        policy = self._policies.pop(policy_id, None)
        if not policy:
            return False

        # Remove from index
        if policy.resource_type in self._policies_by_resource:
            self._policies_by_resource[policy.resource_type].remove(policy_id)

        self._clear_cache()
        return True

    async def enable_policy(self, policy_id: str) -> bool:
        """Enable a policy."""
        policy = self._policies.get(policy_id)
        if not policy:
            return False

        policy.enabled = True
        policy.updated_at = datetime.now()
        self._clear_cache()
        return True

    async def disable_policy(self, policy_id: str) -> bool:
        """Disable a policy."""
        policy = self._policies.get(policy_id)
        if not policy:
            return False

        policy.enabled = False
        policy.updated_at = datetime.now()
        self._clear_cache()
        return True

    # =========================================================================
    # Agent Boundaries
    # =========================================================================

    async def set_agent_boundary(
        self,
        agent_id: str,
        boundary: AgentPermissionBoundary,
    ) -> None:
        """Set permission boundary for an agent."""
        boundary.agent_id = agent_id
        boundary.updated_at = datetime.now()
        self._agent_boundaries[agent_id] = boundary

        logger.info("Agent boundary set", agent_id=agent_id)

    async def get_agent_boundary(
        self,
        agent_id: str,
    ) -> Optional[AgentPermissionBoundary]:
        """Get permission boundary for an agent."""
        return self._agent_boundaries.get(agent_id)

    async def check_agent_permission(
        self,
        agent_id: str,
        action: str,
        resource: Optional[str] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if an agent has permission for an action.

        Returns (allowed, reason).
        """
        boundary = self._agent_boundaries.get(agent_id)

        if not boundary:
            # No boundary = use default (allow with logging)
            return True, None

        # Check time restrictions
        if not boundary.is_within_active_hours():
            return False, "Outside active hours"

        # Check specific actions
        if action == "use_tool" and resource:
            if not boundary.can_use_tool(resource):
                return False, f"Tool not allowed: {resource}"

        if action == "access_memory" and resource:
            if not boundary.can_access_memory(resource):
                return False, f"Memory namespace not allowed: {resource}"

        # Check network access
        if action == "network_request" and resource:
            domain = resource.split("/")[0] if "/" in resource else resource
            if not boundary.network_policy.is_allowed(domain):
                return False, f"Domain not allowed: {domain}"

        return True, None

    async def remove_agent_boundary(self, agent_id: str) -> bool:
        """Remove permission boundary for an agent."""
        if agent_id in self._agent_boundaries:
            del self._agent_boundaries[agent_id]
            return True
        return False

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _build_eval_context(
        self,
        context: SecurityContext,
        resource: str,
        resource_id: Optional[str],
        resource_attrs: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Build evaluation context for policy conditions."""
        return {
            "user": context.user.to_dict() if context.user else {},
            "user_id": context.user_id,
            "tenant_id": context.tenant_id,
            "roles": context.roles,
            "ip_address": context.ip_address,
            "resource": {
                "type": resource,
                "id": resource_id,
                **(resource_attrs or {}),
            },
            "time": {
                "hour": datetime.now().hour,
                "day_of_week": datetime.now().weekday(),
                "timestamp": datetime.now().isoformat(),
            },
            "context": context.to_dict(),
        }

    def _get_cache_key(
        self,
        context: SecurityContext,
        resource: str,
        action: PermissionAction,
        resource_id: Optional[str],
    ) -> str:
        """Generate cache key for permission check."""
        return f"{context.user_id}:{context.tenant_id}:{':'.join(sorted(context.roles))}:{resource}:{action.value}:{resource_id or ''}"

    def _check_cache(self, cache_key: str) -> Optional[AuthorizationResult]:
        """Check permission cache."""
        if cache_key not in self._permission_cache:
            return None

        timestamp = self._cache_timestamps.get(cache_key)
        if not timestamp:
            return None

        if (datetime.now() - timestamp).total_seconds() > self._cache_ttl:
            del self._permission_cache[cache_key]
            del self._cache_timestamps[cache_key]
            return None

        return self._permission_cache[cache_key]

    def _cache_result(self, cache_key: str, result: AuthorizationResult) -> None:
        """Cache authorization result."""
        self._permission_cache[cache_key] = result
        self._cache_timestamps[cache_key] = datetime.now()

    def _clear_cache(self) -> None:
        """Clear the permission cache."""
        self._permission_cache.clear()
        self._cache_timestamps.clear()

    async def _emit_decision(
        self,
        context: SecurityContext,
        resource: str,
        action: PermissionAction,
        resource_id: Optional[str],
        result: AuthorizationResult,
    ) -> None:
        """Emit authorization decision event."""
        for handler in self._decision_handlers:
            try:
                handler({
                    "user_id": context.user_id,
                    "resource": resource,
                    "resource_id": resource_id,
                    "action": action.value,
                    "allowed": result.status == AuthorizationResultStatus.ALLOWED,
                    "decided_by": result.decided_by,
                    "policy_id": result.policy_id,
                    "role_name": result.role_name,
                })
            except Exception as e:
                logger.error(f"Decision handler error: {e}")

    def add_decision_handler(self, handler: Callable) -> None:
        """Add an authorization decision handler."""
        self._decision_handlers.append(handler)

    def get_stats(self) -> Dict[str, Any]:
        """Get authorization service statistics."""
        return {
            "roles_count": len(self._roles),
            "policies_count": len(self._policies),
            "agent_boundaries_count": len(self._agent_boundaries),
            "cache_size": len(self._permission_cache),
            "system_roles": [r.name for r in self._roles.values() if r.system_role],
        }
