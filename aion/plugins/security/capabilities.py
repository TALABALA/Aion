"""
Capability-Based Security for Plugins

Implements Deno-style capability-based security where plugins
must explicitly request permissions at runtime.
"""

from __future__ import annotations

import asyncio
import fnmatch
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional, Set, Union

import structlog

logger = structlog.get_logger(__name__)


class Capability(str, Enum):
    """Available capabilities that plugins can request."""

    # File system
    READ_FILE = "read_file"  # --allow-read
    WRITE_FILE = "write_file"  # --allow-write

    # Network
    NETWORK_CONNECT = "network_connect"  # --allow-net
    NETWORK_LISTEN = "network_listen"

    # Environment
    ENV_READ = "env_read"  # --allow-env
    ENV_WRITE = "env_write"

    # Subprocess
    SUBPROCESS_RUN = "subprocess_run"  # --allow-run

    # System
    SYSTEM_INFO = "system_info"  # --allow-sys
    HIGH_RESOLUTION_TIME = "high_resolution_time"

    # FFI
    FFI_LOAD = "ffi_load"  # --allow-ffi

    # AION specific
    KERNEL_ACCESS = "kernel_access"
    MEMORY_ACCESS = "memory_access"
    TOOL_REGISTER = "tool_register"
    AGENT_SPAWN = "agent_spawn"
    EVENT_EMIT = "event_emit"
    HOOK_REGISTER = "hook_register"
    API_CALL = "api_call"


class CapabilityDeniedError(Exception):
    """Raised when a capability check fails."""

    def __init__(
        self,
        capability: Capability,
        resource: Optional[str] = None,
        plugin_id: Optional[str] = None,
    ):
        self.capability = capability
        self.resource = resource
        self.plugin_id = plugin_id
        msg = f"Capability denied: {capability.value}"
        if resource:
            msg += f" for resource: {resource}"
        if plugin_id:
            msg += f" (plugin: {plugin_id})"
        super().__init__(msg)


@dataclass
class CapabilityGrant:
    """A granted capability with optional restrictions."""

    capability: Capability
    granted: bool = True
    granted_at: float = 0.0
    expires_at: Optional[float] = None

    # Resource restrictions
    allowed_paths: list[str] = field(default_factory=list)  # For file ops
    allowed_hosts: list[str] = field(default_factory=list)  # For network
    allowed_env_vars: list[str] = field(default_factory=list)  # For env
    allowed_commands: list[str] = field(default_factory=list)  # For subprocess

    # Flags
    prompt_on_first_use: bool = False
    log_usage: bool = True

    def is_expired(self) -> bool:
        """Check if grant has expired."""
        if self.expires_at is None:
            return False
        import time
        return time.time() > self.expires_at

    def allows_path(self, path: Union[str, Path]) -> bool:
        """Check if path is allowed."""
        if not self.allowed_paths:
            return True  # No restrictions
        path_str = str(Path(path).resolve())
        return any(
            fnmatch.fnmatch(path_str, pattern) or path_str.startswith(pattern)
            for pattern in self.allowed_paths
        )

    def allows_host(self, host: str) -> bool:
        """Check if host is allowed."""
        if not self.allowed_hosts:
            return True
        return any(
            fnmatch.fnmatch(host, pattern)
            for pattern in self.allowed_hosts
        )

    def allows_env_var(self, var: str) -> bool:
        """Check if environment variable is allowed."""
        if not self.allowed_env_vars:
            return True
        return any(
            fnmatch.fnmatch(var, pattern)
            for pattern in self.allowed_env_vars
        )

    def allows_command(self, command: str) -> bool:
        """Check if command is allowed."""
        if not self.allowed_commands:
            return True
        return any(
            fnmatch.fnmatch(command, pattern)
            for pattern in self.allowed_commands
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "capability": self.capability.value,
            "granted": self.granted,
            "granted_at": self.granted_at,
            "expires_at": self.expires_at,
            "allowed_paths": self.allowed_paths,
            "allowed_hosts": self.allowed_hosts,
            "allowed_env_vars": self.allowed_env_vars,
            "allowed_commands": self.allowed_commands,
        }


@dataclass
class CapabilityPrompt:
    """Prompt for user to grant a capability."""

    plugin_id: str
    capability: Capability
    resource: Optional[str]
    reason: Optional[str]
    response: Optional[bool] = None
    responded_at: Optional[float] = None


class CapabilityChecker:
    """
    Checks and enforces capability grants for plugins.

    Implements capability-based security similar to Deno's permission system.
    """

    def __init__(
        self,
        plugin_id: str,
        prompt_callback: Optional[Callable[[CapabilityPrompt], asyncio.Future[bool]]] = None,
        auto_deny_unknown: bool = True,
    ):
        self.plugin_id = plugin_id
        self.prompt_callback = prompt_callback
        self.auto_deny_unknown = auto_deny_unknown

        self._grants: dict[Capability, CapabilityGrant] = {}
        self._denied: Set[Capability] = set()
        self._usage_log: list[dict[str, Any]] = []
        self._pending_prompts: dict[str, CapabilityPrompt] = {}

    def grant(
        self,
        capability: Capability,
        **kwargs,
    ) -> CapabilityGrant:
        """Grant a capability to the plugin."""
        import time
        grant = CapabilityGrant(
            capability=capability,
            granted=True,
            granted_at=time.time(),
            **kwargs,
        )
        self._grants[capability] = grant
        self._denied.discard(capability)

        logger.info(
            "Capability granted",
            plugin_id=self.plugin_id,
            capability=capability.value,
        )
        return grant

    def deny(self, capability: Capability) -> None:
        """Explicitly deny a capability."""
        self._denied.add(capability)
        self._grants.pop(capability, None)

        logger.info(
            "Capability denied",
            plugin_id=self.plugin_id,
            capability=capability.value,
        )

    def revoke(self, capability: Capability) -> None:
        """Revoke a previously granted capability."""
        self._grants.pop(capability, None)
        logger.info(
            "Capability revoked",
            plugin_id=self.plugin_id,
            capability=capability.value,
        )

    def has_capability(self, capability: Capability) -> bool:
        """Check if capability is granted (without enforcement)."""
        if capability in self._denied:
            return False
        grant = self._grants.get(capability)
        return grant is not None and grant.granted and not grant.is_expired()

    def check(
        self,
        capability: Capability,
        resource: Optional[str] = None,
    ) -> None:
        """
        Check capability and raise if not granted.

        Args:
            capability: Required capability
            resource: Optional resource (path, host, etc.)

        Raises:
            CapabilityDeniedError: If capability not granted
        """
        # Check explicit denial
        if capability in self._denied:
            raise CapabilityDeniedError(capability, resource, self.plugin_id)

        # Check grant
        grant = self._grants.get(capability)
        if grant is None:
            if self.auto_deny_unknown:
                raise CapabilityDeniedError(capability, resource, self.plugin_id)
            return  # Allow if auto_deny_unknown is False

        if not grant.granted or grant.is_expired():
            raise CapabilityDeniedError(capability, resource, self.plugin_id)

        # Check resource restrictions
        if resource:
            if capability in (Capability.READ_FILE, Capability.WRITE_FILE):
                if not grant.allows_path(resource):
                    raise CapabilityDeniedError(capability, resource, self.plugin_id)
            elif capability == Capability.NETWORK_CONNECT:
                if not grant.allows_host(resource):
                    raise CapabilityDeniedError(capability, resource, self.plugin_id)
            elif capability in (Capability.ENV_READ, Capability.ENV_WRITE):
                if not grant.allows_env_var(resource):
                    raise CapabilityDeniedError(capability, resource, self.plugin_id)
            elif capability == Capability.SUBPROCESS_RUN:
                if not grant.allows_command(resource):
                    raise CapabilityDeniedError(capability, resource, self.plugin_id)

        # Log usage
        if grant.log_usage:
            self._log_usage(capability, resource)

    async def check_or_prompt(
        self,
        capability: Capability,
        resource: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> bool:
        """
        Check capability, prompting user if needed.

        Args:
            capability: Required capability
            resource: Optional resource
            reason: Reason for request

        Returns:
            True if granted, False if denied
        """
        try:
            self.check(capability, resource)
            return True
        except CapabilityDeniedError:
            pass

        # Capability not granted, prompt user
        if self.prompt_callback is None:
            return False

        prompt = CapabilityPrompt(
            plugin_id=self.plugin_id,
            capability=capability,
            resource=resource,
            reason=reason,
        )

        try:
            result = await self.prompt_callback(prompt)
            import time
            prompt.response = result
            prompt.responded_at = time.time()

            if result:
                # User granted permission
                self.grant(
                    capability,
                    allowed_paths=[resource] if resource and capability in (
                        Capability.READ_FILE, Capability.WRITE_FILE
                    ) else [],
                    allowed_hosts=[resource] if resource and capability == Capability.NETWORK_CONNECT else [],
                )
                return True
            else:
                self.deny(capability)
                return False

        except Exception as e:
            logger.error(
                "Capability prompt failed",
                plugin_id=self.plugin_id,
                capability=capability.value,
                error=str(e),
            )
            return False

    def check_file_read(self, path: Union[str, Path]) -> None:
        """Check file read capability."""
        self.check(Capability.READ_FILE, str(path))

    def check_file_write(self, path: Union[str, Path]) -> None:
        """Check file write capability."""
        self.check(Capability.WRITE_FILE, str(path))

    def check_network(self, host: str) -> None:
        """Check network capability."""
        self.check(Capability.NETWORK_CONNECT, host)

    def check_env_read(self, var: str) -> None:
        """Check environment read capability."""
        self.check(Capability.ENV_READ, var)

    def check_subprocess(self, command: str) -> None:
        """Check subprocess capability."""
        self.check(Capability.SUBPROCESS_RUN, command)

    def grant_all(self) -> None:
        """Grant all capabilities (dangerous - for development only)."""
        logger.warning(
            "Granting all capabilities",
            plugin_id=self.plugin_id,
        )
        for cap in Capability:
            self.grant(cap)

    def revoke_all(self) -> None:
        """Revoke all capabilities."""
        self._grants.clear()
        self._denied = set(Capability)

    def _log_usage(self, capability: Capability, resource: Optional[str]) -> None:
        """Log capability usage."""
        import time
        self._usage_log.append({
            "timestamp": time.time(),
            "capability": capability.value,
            "resource": resource,
        })
        # Keep only last 1000 entries
        if len(self._usage_log) > 1000:
            self._usage_log = self._usage_log[-1000:]

    def get_grants(self) -> dict[str, CapabilityGrant]:
        """Get all current grants."""
        return {cap.value: grant for cap, grant in self._grants.items()}

    def get_usage_log(self) -> list[dict[str, Any]]:
        """Get capability usage log."""
        return self._usage_log.copy()

    def get_stats(self) -> dict[str, Any]:
        """Get capability statistics."""
        return {
            "plugin_id": self.plugin_id,
            "granted": [cap.value for cap in self._grants if self._grants[cap].granted],
            "denied": [cap.value for cap in self._denied],
            "usage_count": len(self._usage_log),
        }


class CapabilityManager:
    """
    Manages capabilities across multiple plugins.

    Provides centralized capability management and policy enforcement.
    """

    def __init__(
        self,
        default_prompt_callback: Optional[Callable] = None,
        auto_deny_unknown: bool = True,
    ):
        self.default_prompt_callback = default_prompt_callback
        self.auto_deny_unknown = auto_deny_unknown
        self._checkers: dict[str, CapabilityChecker] = {}
        self._global_policies: dict[Capability, bool] = {}

    def get_checker(self, plugin_id: str) -> CapabilityChecker:
        """Get or create capability checker for a plugin."""
        if plugin_id not in self._checkers:
            self._checkers[plugin_id] = CapabilityChecker(
                plugin_id=plugin_id,
                prompt_callback=self.default_prompt_callback,
                auto_deny_unknown=self.auto_deny_unknown,
            )
        return self._checkers[plugin_id]

    def remove_checker(self, plugin_id: str) -> None:
        """Remove capability checker for a plugin."""
        self._checkers.pop(plugin_id, None)

    def set_global_policy(self, capability: Capability, allowed: bool) -> None:
        """Set a global policy for a capability."""
        self._global_policies[capability] = allowed

    def grant_from_manifest(
        self,
        plugin_id: str,
        permissions: dict[str, Any],
    ) -> None:
        """
        Grant capabilities based on plugin manifest permissions.

        Args:
            plugin_id: Plugin identifier
            permissions: Permissions from manifest
        """
        checker = self.get_checker(plugin_id)

        # Map manifest permissions to capabilities
        permission_map = {
            "file_read": (Capability.READ_FILE, "paths"),
            "file_write": (Capability.WRITE_FILE, "paths"),
            "network": (Capability.NETWORK_CONNECT, "hosts"),
            "env": (Capability.ENV_READ, "vars"),
            "subprocess": (Capability.SUBPROCESS_RUN, "commands"),
            "kernel": (Capability.KERNEL_ACCESS, None),
            "memory": (Capability.MEMORY_ACCESS, None),
            "tools": (Capability.TOOL_REGISTER, None),
            "agents": (Capability.AGENT_SPAWN, None),
            "events": (Capability.EVENT_EMIT, None),
            "hooks": (Capability.HOOK_REGISTER, None),
            "api": (Capability.API_CALL, None),
        }

        for perm_name, (capability, restriction_key) in permission_map.items():
            if perm_name in permissions:
                perm_value = permissions[perm_name]

                # Check global policy
                if self._global_policies.get(capability) is False:
                    checker.deny(capability)
                    continue

                kwargs = {}
                if restriction_key and isinstance(perm_value, dict):
                    if restriction_key == "paths":
                        kwargs["allowed_paths"] = perm_value.get("paths", [])
                    elif restriction_key == "hosts":
                        kwargs["allowed_hosts"] = perm_value.get("hosts", [])
                    elif restriction_key == "vars":
                        kwargs["allowed_env_vars"] = perm_value.get("vars", [])
                    elif restriction_key == "commands":
                        kwargs["allowed_commands"] = perm_value.get("commands", [])

                checker.grant(capability, **kwargs)

    def get_stats(self) -> dict[str, Any]:
        """Get statistics for all plugins."""
        return {
            "total_plugins": len(self._checkers),
            "global_policies": {
                cap.value: allowed
                for cap, allowed in self._global_policies.items()
            },
            "per_plugin": {
                pid: checker.get_stats()
                for pid, checker in self._checkers.items()
            },
        }
