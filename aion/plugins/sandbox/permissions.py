"""
AION Plugin Permission System

Permission checking and enforcement for plugins.
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set
from urllib.parse import urlparse

import structlog

from aion.plugins.types import PluginPermissions, PermissionLevel

logger = structlog.get_logger(__name__)


class PermissionViolation(Exception):
    """Raised when a permission check fails."""

    def __init__(
        self,
        plugin_id: str,
        operation: str,
        resource: str,
        message: str = "",
    ):
        self.plugin_id = plugin_id
        self.operation = operation
        self.resource = resource
        super().__init__(
            message or f"Plugin '{plugin_id}' denied permission for {operation} on {resource}"
        )


@dataclass
class PermissionRequest:
    """A permission request from a plugin."""

    operation: str
    resource: str
    plugin_id: str
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PermissionDecision:
    """Result of permission evaluation."""

    allowed: bool
    reason: str = ""
    conditions: List[str] = field(default_factory=list)


class PermissionChecker:
    """
    Checks permissions for plugin operations.

    Features:
    - Network access control
    - File system access control
    - Database access control
    - API access control
    - Subprocess control
    - Custom permission rules
    """

    def __init__(self, permissions: PluginPermissions):
        self._permissions = permissions
        self._custom_rules: Dict[str, Callable[[str], bool]] = {}
        self._audit_log: List[PermissionRequest] = []
        self._violations: List[PermissionViolation] = []

    @property
    def permissions(self) -> PluginPermissions:
        """Get the permissions configuration."""
        return self._permissions

    def check(self, operation: str, resource: str) -> bool:
        """
        Check if an operation is permitted.

        Args:
            operation: Operation type
            resource: Resource identifier

        Returns:
            True if permitted
        """
        decision = self.evaluate(operation, resource)
        return decision.allowed

    def require(
        self,
        operation: str,
        resource: str,
        plugin_id: str = "unknown",
    ) -> None:
        """
        Require permission, raising if denied.

        Args:
            operation: Operation type
            resource: Resource identifier
            plugin_id: Plugin identifier

        Raises:
            PermissionViolation: If permission denied
        """
        decision = self.evaluate(operation, resource)

        if not decision.allowed:
            violation = PermissionViolation(
                plugin_id, operation, resource, decision.reason
            )
            self._violations.append(violation)
            raise violation

    def evaluate(self, operation: str, resource: str) -> PermissionDecision:
        """
        Evaluate a permission request.

        Args:
            operation: Operation type
            resource: Resource identifier

        Returns:
            PermissionDecision with result and reason
        """
        # Check permission level first
        if self._permissions.level == PermissionLevel.FULL:
            return PermissionDecision(allowed=True, reason="Full permissions granted")

        # Check custom rules first
        if operation in self._custom_rules:
            allowed = self._custom_rules[operation](resource)
            return PermissionDecision(
                allowed=allowed,
                reason=f"Custom rule for {operation}",
            )

        # Route to specific checkers
        checkers = {
            "network": self._check_network,
            "http": self._check_network,
            "file_read": self._check_file_read,
            "file_write": self._check_file_write,
            "file": self._check_file_write,
            "database": self._check_database,
            "memory": self._check_memory,
            "subprocess": self._check_subprocess,
            "api": self._check_api,
            "environment": self._check_environment,
        }

        checker = checkers.get(operation)
        if checker:
            return checker(resource)

        # Unknown operation - deny by default
        return PermissionDecision(
            allowed=False,
            reason=f"Unknown operation: {operation}",
        )

    # === Network Access ===

    def _check_network(self, resource: str) -> PermissionDecision:
        """Check network access permission."""
        if not self._permissions.network_access:
            return PermissionDecision(
                allowed=False,
                reason="Network access not permitted",
            )

        # Parse URL/domain
        if resource.startswith(("http://", "https://")):
            parsed = urlparse(resource)
            domain = parsed.netloc
            port = parsed.port
        else:
            domain = resource.split(":")[0]
            port = int(resource.split(":")[1]) if ":" in resource else None

        # Check blocked domains
        if self._matches_patterns(domain, self._permissions.blocked_domains):
            return PermissionDecision(
                allowed=False,
                reason=f"Domain '{domain}' is blocked",
            )

        # Check allowed domains
        if self._permissions.allowed_domains:
            if not self._matches_patterns(domain, self._permissions.allowed_domains):
                return PermissionDecision(
                    allowed=False,
                    reason=f"Domain '{domain}' not in allowed list",
                )

        # Check allowed ports
        if port and self._permissions.allowed_ports:
            if port not in self._permissions.allowed_ports:
                return PermissionDecision(
                    allowed=False,
                    reason=f"Port {port} not in allowed list",
                )

        return PermissionDecision(allowed=True)

    # === File System Access ===

    def _check_file_read(self, resource: str) -> PermissionDecision:
        """Check file read permission."""
        if not self._permissions.file_system_access:
            return PermissionDecision(
                allowed=False,
                reason="File system access not permitted",
            )

        return self._check_path(resource)

    def _check_file_write(self, resource: str) -> PermissionDecision:
        """Check file write permission."""
        if not self._permissions.file_system_access:
            return PermissionDecision(
                allowed=False,
                reason="File system access not permitted",
            )

        # Check if path is read-only
        path = Path(resource).resolve()

        for readonly_path in self._permissions.read_only_paths:
            if self._path_matches(path, Path(readonly_path)):
                return PermissionDecision(
                    allowed=False,
                    reason=f"Path '{resource}' is read-only",
                )

        return self._check_path(resource)

    def _check_path(self, resource: str) -> PermissionDecision:
        """Check if path access is allowed."""
        path = Path(resource).resolve()

        # Check blocked paths
        for blocked in self._permissions.blocked_paths:
            if self._path_matches(path, Path(blocked)):
                return PermissionDecision(
                    allowed=False,
                    reason=f"Path '{resource}' is blocked",
                )

        # Check allowed paths
        if self._permissions.allowed_paths:
            for allowed in self._permissions.allowed_paths:
                if self._path_matches(path, Path(allowed)):
                    return PermissionDecision(allowed=True)

            return PermissionDecision(
                allowed=False,
                reason=f"Path '{resource}' not in allowed paths",
            )

        return PermissionDecision(allowed=True)

    def _path_matches(self, path: Path, pattern: Path) -> bool:
        """Check if path matches or is under pattern."""
        try:
            pattern = pattern.resolve()
            # Check if path is same as or under pattern
            return path == pattern or pattern in path.parents
        except Exception:
            return False

    # === Database Access ===

    def _check_database(self, resource: str) -> PermissionDecision:
        """Check database access permission."""
        if not self._permissions.database_access:
            return PermissionDecision(
                allowed=False,
                reason="Database access not permitted",
            )

        # Resource is table name
        table = resource

        # Check allowed tables
        if self._permissions.allowed_tables:
            if table not in self._permissions.allowed_tables:
                return PermissionDecision(
                    allowed=False,
                    reason=f"Table '{table}' not in allowed list",
                )

        return PermissionDecision(allowed=True)

    # === Memory Access ===

    def _check_memory(self, resource: str) -> PermissionDecision:
        """Check memory namespace access permission."""
        if not self._permissions.memory_access:
            return PermissionDecision(
                allowed=False,
                reason="Memory access not permitted",
            )

        # Resource is namespace
        namespace = resource

        if self._permissions.allowed_memory_namespaces:
            if namespace not in self._permissions.allowed_memory_namespaces:
                return PermissionDecision(
                    allowed=False,
                    reason=f"Memory namespace '{namespace}' not allowed",
                )

        return PermissionDecision(allowed=True)

    # === Subprocess ===

    def _check_subprocess(self, resource: str) -> PermissionDecision:
        """Check subprocess execution permission."""
        if not self._permissions.subprocess_access:
            return PermissionDecision(
                allowed=False,
                reason="Subprocess execution not permitted",
            )

        # Resource is command
        command = resource.split()[0] if resource else ""

        if self._permissions.allowed_commands:
            if not self._matches_patterns(command, self._permissions.allowed_commands):
                return PermissionDecision(
                    allowed=False,
                    reason=f"Command '{command}' not in allowed list",
                )

        return PermissionDecision(allowed=True)

    # === API Access ===

    def _check_api(self, resource: str) -> PermissionDecision:
        """Check API access permission."""
        # Resource is API endpoint
        endpoint = resource

        if self._permissions.api_access:
            for allowed in self._permissions.api_access:
                if allowed.endswith("*"):
                    if endpoint.startswith(allowed[:-1]):
                        return PermissionDecision(allowed=True)
                elif endpoint == allowed:
                    return PermissionDecision(allowed=True)

            return PermissionDecision(
                allowed=False,
                reason=f"API endpoint '{endpoint}' not in allowed list",
            )

        # If no API restrictions, allow all
        return PermissionDecision(allowed=True)

    # === Environment ===

    def _check_environment(self, resource: str) -> PermissionDecision:
        """Check environment variable access permission."""
        if not self._permissions.environment_access:
            return PermissionDecision(
                allowed=False,
                reason="Environment access not permitted",
            )

        # Resource is variable name
        var_name = resource

        if self._permissions.allowed_env_vars:
            if var_name not in self._permissions.allowed_env_vars:
                return PermissionDecision(
                    allowed=False,
                    reason=f"Environment variable '{var_name}' not allowed",
                )

        return PermissionDecision(allowed=True)

    # === Utilities ===

    def _matches_patterns(self, value: str, patterns: List[str]) -> bool:
        """Check if value matches any pattern (supports wildcards)."""
        for pattern in patterns:
            if pattern.startswith("*."):
                # Wildcard subdomain
                if value.endswith(pattern[1:]) or value == pattern[2:]:
                    return True
            elif fnmatch.fnmatch(value, pattern):
                return True
            elif value == pattern:
                return True
        return False

    # === Custom Rules ===

    def add_rule(self, operation: str, rule: Callable[[str], bool]) -> None:
        """Add a custom permission rule."""
        self._custom_rules[operation] = rule

    def remove_rule(self, operation: str) -> None:
        """Remove a custom permission rule."""
        self._custom_rules.pop(operation, None)

    # === Audit ===

    def get_violations(self) -> List[PermissionViolation]:
        """Get all permission violations."""
        return self._violations.copy()

    def clear_violations(self) -> None:
        """Clear violation history."""
        self._violations.clear()


class PermissionBuilder:
    """
    Builder for creating PluginPermissions instances.

    Usage:
        permissions = (PermissionBuilder()
            .allow_network(["api.example.com"])
            .allow_files(["/tmp"])
            .build())
    """

    def __init__(self):
        self._level = PermissionLevel.MINIMAL
        self._network_access = False
        self._allowed_domains: List[str] = []
        self._blocked_domains: List[str] = []
        self._file_system_access = False
        self._allowed_paths: List[str] = []
        self._blocked_paths: List[str] = []
        self._database_access = False
        self._allowed_tables: List[str] = []
        self._memory_access = False
        self._subprocess_access = False
        self._api_access: List[str] = []

    def level(self, level: PermissionLevel) -> "PermissionBuilder":
        """Set permission level."""
        self._level = level
        return self

    def allow_network(
        self,
        domains: Optional[List[str]] = None,
        block: Optional[List[str]] = None,
    ) -> "PermissionBuilder":
        """Allow network access."""
        self._network_access = True
        if domains:
            self._allowed_domains.extend(domains)
        if block:
            self._blocked_domains.extend(block)
        return self

    def allow_files(
        self,
        paths: Optional[List[str]] = None,
        block: Optional[List[str]] = None,
    ) -> "PermissionBuilder":
        """Allow file system access."""
        self._file_system_access = True
        if paths:
            self._allowed_paths.extend(paths)
        if block:
            self._blocked_paths.extend(block)
        return self

    def allow_database(
        self,
        tables: Optional[List[str]] = None,
    ) -> "PermissionBuilder":
        """Allow database access."""
        self._database_access = True
        if tables:
            self._allowed_tables.extend(tables)
        return self

    def allow_memory(self) -> "PermissionBuilder":
        """Allow memory access."""
        self._memory_access = True
        return self

    def allow_subprocess(
        self,
        commands: Optional[List[str]] = None,
    ) -> "PermissionBuilder":
        """Allow subprocess execution."""
        self._subprocess_access = True
        return self

    def allow_api(self, endpoints: List[str]) -> "PermissionBuilder":
        """Allow API access."""
        self._api_access.extend(endpoints)
        return self

    def build(self) -> PluginPermissions:
        """Build the PluginPermissions instance."""
        return PluginPermissions(
            level=self._level,
            network_access=self._network_access,
            allowed_domains=self._allowed_domains,
            blocked_domains=self._blocked_domains,
            file_system_access=self._file_system_access,
            allowed_paths=self._allowed_paths,
            blocked_paths=self._blocked_paths,
            database_access=self._database_access,
            allowed_tables=self._allowed_tables,
            memory_access=self._memory_access,
            subprocess_access=self._subprocess_access,
            api_access=self._api_access,
        )
