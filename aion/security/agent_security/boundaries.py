"""
AION Agent Security Boundaries

Permission boundaries and sandboxing for AI agents.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set

import structlog

from aion.security.types import (
    AgentCapability,
    AgentPermissionBoundary,
    NetworkPolicy,
    ResourceLimits,
)

logger = structlog.get_logger(__name__)


@dataclass
class BoundaryViolation:
    """A boundary violation event."""

    agent_id: str
    violation_type: str
    resource: Optional[str] = None
    action: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class AgentBoundaryEnforcer:
    """
    Enforces permission boundaries for AI agents.

    Features:
    - Capability-based access control
    - Tool allowlisting/denylisting
    - Network access control
    - Resource limits enforcement
    - Time-based restrictions
    - Violation tracking
    """

    def __init__(self):
        # Boundaries by agent ID
        self._boundaries: Dict[str, AgentPermissionBoundary] = {}

        # Default boundary for agents without explicit config
        self._default_boundary = AgentPermissionBoundary(
            granted_capabilities={
                AgentCapability.MEMORY_READ,
                AgentCapability.MEMORY_WRITE,
                AgentCapability.KNOWLEDGE_READ,
                AgentCapability.TOOL_API_CALL,
            },
            allowed_tools=["*"],
            denied_tools=["shell", "bash", "exec", "system"],
            network_policy=NetworkPolicy(
                allowed_protocols=["https"],
                max_requests_per_minute=60,
            ),
            resource_limits=ResourceLimits(),
        )

        # Violation history
        self._violations: List[BoundaryViolation] = []

        # Violation handlers
        self._violation_handlers: List[Callable[[BoundaryViolation], None]] = []

        # Usage tracking
        self._usage: Dict[str, Dict[str, int]] = {}

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the boundary enforcer."""
        if self._initialized:
            return

        logger.info("Initializing Agent Boundary Enforcer")
        self._initialized = True

    # =========================================================================
    # Boundary Management
    # =========================================================================

    def set_boundary(
        self,
        agent_id: str,
        boundary: AgentPermissionBoundary,
    ) -> None:
        """Set permission boundary for an agent."""
        boundary.agent_id = agent_id
        boundary.updated_at = datetime.now()
        self._boundaries[agent_id] = boundary

        logger.info("Agent boundary set", agent_id=agent_id)

    def get_boundary(self, agent_id: str) -> AgentPermissionBoundary:
        """Get permission boundary for an agent."""
        return self._boundaries.get(agent_id, self._default_boundary)

    def remove_boundary(self, agent_id: str) -> bool:
        """Remove custom boundary for an agent."""
        if agent_id in self._boundaries:
            del self._boundaries[agent_id]
            return True
        return False

    def set_default_boundary(self, boundary: AgentPermissionBoundary) -> None:
        """Set the default boundary for agents."""
        self._default_boundary = boundary

    # =========================================================================
    # Permission Checking
    # =========================================================================

    async def check_capability(
        self,
        agent_id: str,
        capability: AgentCapability,
    ) -> tuple[bool, Optional[str]]:
        """
        Check if agent has a capability.

        Returns (allowed, reason).
        """
        boundary = self.get_boundary(agent_id)

        # Check time restrictions first
        if not boundary.is_within_active_hours():
            return False, "Outside active hours"

        # Check denied capabilities
        if capability in boundary.denied_capabilities:
            self._record_violation(
                agent_id,
                "capability_denied",
                action=capability.value,
            )
            return False, f"Capability denied: {capability.value}"

        # Check granted capabilities
        if capability in boundary.granted_capabilities:
            return True, None

        # Default deny
        return False, f"Capability not granted: {capability.value}"

    async def check_tool_access(
        self,
        agent_id: str,
        tool_name: str,
        tool_config: Optional[Dict[str, Any]] = None,
    ) -> tuple[bool, Optional[str]]:
        """
        Check if agent can use a tool.

        Returns (allowed, reason).
        """
        boundary = self.get_boundary(agent_id)

        # Check time restrictions
        if not boundary.is_within_active_hours():
            return False, "Outside active hours"

        # Check denied tools first
        for pattern in boundary.denied_tools:
            if fnmatch.fnmatch(tool_name.lower(), pattern.lower()):
                self._record_violation(
                    agent_id,
                    "tool_denied",
                    resource=tool_name,
                )
                return False, f"Tool denied: {tool_name}"

        # Check allowed tools
        if "*" in boundary.allowed_tools:
            return True, None

        for pattern in boundary.allowed_tools:
            if fnmatch.fnmatch(tool_name.lower(), pattern.lower()):
                return True, None

        # Not explicitly allowed
        self._record_violation(
            agent_id,
            "tool_not_allowed",
            resource=tool_name,
        )
        return False, f"Tool not allowed: {tool_name}"

    async def check_network_access(
        self,
        agent_id: str,
        domain: str,
        port: int = 443,
        protocol: str = "https",
    ) -> tuple[bool, Optional[str]]:
        """
        Check if agent can access a network resource.

        Returns (allowed, reason).
        """
        boundary = self.get_boundary(agent_id)
        policy = boundary.network_policy

        # Check protocol
        if protocol not in policy.allowed_protocols:
            self._record_violation(
                agent_id,
                "protocol_denied",
                resource=domain,
                details={"protocol": protocol},
            )
            return False, f"Protocol not allowed: {protocol}"

        # Check denied domains
        for pattern in policy.denied_domains:
            if fnmatch.fnmatch(domain.lower(), pattern.lower()):
                self._record_violation(
                    agent_id,
                    "domain_denied",
                    resource=domain,
                )
                return False, f"Domain denied: {domain}"

        # Check denied IPs
        if domain in policy.denied_ips:
            return False, f"IP denied: {domain}"

        # Check denied ports
        if port in policy.denied_ports:
            return False, f"Port denied: {port}"

        # Check allowed domains (if specified)
        if policy.allowed_domains:
            allowed = False
            for pattern in policy.allowed_domains:
                if fnmatch.fnmatch(domain.lower(), pattern.lower()):
                    allowed = True
                    break
            if not allowed:
                self._record_violation(
                    agent_id,
                    "domain_not_allowed",
                    resource=domain,
                )
                return False, f"Domain not allowed: {domain}"

        # Check allowed ports (if specified)
        if policy.allowed_ports and port not in policy.allowed_ports:
            return False, f"Port not allowed: {port}"

        # Check rate limit for network requests
        usage_key = f"{agent_id}:network"
        if not self._check_usage_limit(usage_key, policy.max_requests_per_minute):
            return False, "Network rate limit exceeded"

        self._increment_usage(usage_key)
        return True, None

    async def check_memory_access(
        self,
        agent_id: str,
        namespace: str,
        operation: str = "read",
    ) -> tuple[bool, Optional[str]]:
        """
        Check if agent can access a memory namespace.

        Returns (allowed, reason).
        """
        boundary = self.get_boundary(agent_id)

        # Check denied namespaces
        for pattern in boundary.denied_memory_namespaces:
            if fnmatch.fnmatch(namespace.lower(), pattern.lower()):
                self._record_violation(
                    agent_id,
                    "memory_namespace_denied",
                    resource=namespace,
                    action=operation,
                )
                return False, f"Memory namespace denied: {namespace}"

        # Check allowed namespaces
        if "*" in boundary.allowed_memory_namespaces:
            return True, None

        for pattern in boundary.allowed_memory_namespaces:
            if fnmatch.fnmatch(namespace.lower(), pattern.lower()):
                return True, None

        # Not explicitly allowed
        return False, f"Memory namespace not allowed: {namespace}"

    async def check_file_access(
        self,
        agent_id: str,
        file_path: str,
        operation: str = "read",
    ) -> tuple[bool, Optional[str]]:
        """
        Check if agent can access a file path.

        Returns (allowed, reason).
        """
        boundary = self.get_boundary(agent_id)

        # Normalize path
        file_path = file_path.replace("\\", "/")

        # Check denied paths
        for pattern in boundary.denied_paths:
            if fnmatch.fnmatch(file_path, pattern):
                self._record_violation(
                    agent_id,
                    "path_denied",
                    resource=file_path,
                    action=operation,
                )
                return False, f"Path denied: {file_path}"

        # Check read-only paths for write operations
        if operation in ("write", "delete"):
            for pattern in boundary.read_only_paths:
                if fnmatch.fnmatch(file_path, pattern):
                    return False, f"Path is read-only: {file_path}"

        # Check allowed paths
        if not boundary.allowed_paths:
            return True, None  # No restrictions

        for pattern in boundary.allowed_paths:
            if fnmatch.fnmatch(file_path, pattern):
                return True, None

        return False, f"Path not allowed: {file_path}"

    # =========================================================================
    # Resource Limits
    # =========================================================================

    async def check_resource_limit(
        self,
        agent_id: str,
        resource: str,
        amount: int = 1,
    ) -> tuple[bool, Optional[str]]:
        """
        Check if agent is within resource limits.

        Returns (allowed, reason).
        """
        boundary = self.get_boundary(agent_id)
        limits = boundary.resource_limits

        limit_checks = {
            "tokens_per_request": limits.max_tokens_per_request,
            "tool_calls_per_request": limits.max_tool_calls_per_request,
            "llm_calls_per_request": limits.max_llm_calls_per_request,
            "concurrent_requests": limits.max_concurrent_requests,
            "context_size": limits.max_context_size,
        }

        if resource in limit_checks:
            limit = limit_checks[resource]
            usage_key = f"{agent_id}:{resource}"
            current = self._get_usage(usage_key)

            if current + amount > limit:
                return False, f"Resource limit exceeded: {resource} ({current}/{limit})"

        return True, None

    def record_resource_usage(
        self,
        agent_id: str,
        resource: str,
        amount: int = 1,
    ) -> None:
        """Record resource usage for an agent."""
        usage_key = f"{agent_id}:{resource}"
        self._increment_usage(usage_key, amount)

    def reset_request_usage(self, agent_id: str) -> None:
        """Reset per-request usage counters."""
        keys_to_reset = [
            f"{agent_id}:tokens_per_request",
            f"{agent_id}:tool_calls_per_request",
            f"{agent_id}:llm_calls_per_request",
        ]
        for key in keys_to_reset:
            if key in self._usage:
                self._usage[key] = {"count": 0, "last_reset": datetime.now()}

    # =========================================================================
    # Approval Requirements
    # =========================================================================

    async def requires_approval(
        self,
        agent_id: str,
        action: str,
    ) -> bool:
        """Check if an action requires human approval."""
        boundary = self.get_boundary(agent_id)

        if action in boundary.require_approval_for:
            return True

        # Check if action pattern matches
        for pattern in boundary.require_approval_for:
            if fnmatch.fnmatch(action.lower(), pattern.lower()):
                return True

        return False

    # =========================================================================
    # Violation Tracking
    # =========================================================================

    def _record_violation(
        self,
        agent_id: str,
        violation_type: str,
        resource: Optional[str] = None,
        action: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a boundary violation."""
        violation = BoundaryViolation(
            agent_id=agent_id,
            violation_type=violation_type,
            resource=resource,
            action=action,
            details=details or {},
        )

        self._violations.append(violation)

        # Limit history
        if len(self._violations) > 10000:
            self._violations = self._violations[-5000:]

        logger.warning(
            "Agent boundary violation",
            agent_id=agent_id,
            violation_type=violation_type,
            resource=resource,
        )

        # Notify handlers
        for handler in self._violation_handlers:
            try:
                handler(violation)
            except Exception as e:
                logger.error(f"Violation handler error: {e}")

    def get_violations(
        self,
        agent_id: Optional[str] = None,
        violation_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[BoundaryViolation]:
        """Get recorded violations."""
        violations = self._violations

        if agent_id:
            violations = [v for v in violations if v.agent_id == agent_id]

        if violation_type:
            violations = [v for v in violations if v.violation_type == violation_type]

        return violations[-limit:]

    def add_violation_handler(
        self,
        handler: Callable[[BoundaryViolation], None],
    ) -> None:
        """Add a violation handler."""
        self._violation_handlers.append(handler)

    # =========================================================================
    # Usage Tracking
    # =========================================================================

    def _get_usage(self, key: str) -> int:
        """Get current usage for a key."""
        if key not in self._usage:
            return 0
        return self._usage[key].get("count", 0)

    def _increment_usage(self, key: str, amount: int = 1) -> None:
        """Increment usage for a key."""
        if key not in self._usage:
            self._usage[key] = {"count": 0, "last_reset": datetime.now()}
        self._usage[key]["count"] += amount

    def _check_usage_limit(self, key: str, limit: int) -> bool:
        """Check if usage is within limit."""
        current = self._get_usage(key)
        return current < limit

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get boundary enforcer statistics."""
        violation_counts = {}
        for v in self._violations[-1000:]:
            vtype = v.violation_type
            violation_counts[vtype] = violation_counts.get(vtype, 0) + 1

        return {
            "boundaries_configured": len(self._boundaries),
            "total_violations": len(self._violations),
            "recent_violation_types": violation_counts,
            "tracked_usage_keys": len(self._usage),
        }
