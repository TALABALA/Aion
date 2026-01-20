"""
AION Security & Process Isolation

Enterprise-grade security with:
- Capability-based security model
- Process sandboxing
- Resource isolation
- Audit logging
- Access control
- Secure inter-process communication
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import secrets
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, Flag, auto
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar, Union

import structlog

logger = structlog.get_logger(__name__)

F = TypeVar("F", bound=Callable)


# === Capability-Based Security ===

class Capability(Flag):
    """Process capabilities (fine-grained permissions)."""
    NONE = 0

    # File system
    FS_READ = auto()
    FS_WRITE = auto()
    FS_EXECUTE = auto()
    FS_DELETE = auto()

    # Network
    NET_CONNECT = auto()
    NET_LISTEN = auto()
    NET_RAW = auto()

    # Process
    PROC_SPAWN = auto()
    PROC_KILL = auto()
    PROC_PTRACE = auto()

    # System
    SYS_ADMIN = auto()
    SYS_TIME = auto()
    SYS_MODULE = auto()

    # IPC
    IPC_SEND = auto()
    IPC_RECEIVE = auto()
    IPC_LOCK = auto()

    # Memory
    MEM_LOCK = auto()
    MEM_SHARED = auto()

    # AI-specific
    AI_LLM_ACCESS = auto()
    AI_TOOL_USE = auto()
    AI_MEMORY_ACCESS = auto()
    AI_INTERNET = auto()

    # Dangerous
    PRIVILEGED = auto()

    # Common sets
    @classmethod
    def basic_agent(cls) -> "Capability":
        """Basic capabilities for an AI agent."""
        return (
            cls.FS_READ |
            cls.NET_CONNECT |
            cls.IPC_SEND |
            cls.IPC_RECEIVE |
            cls.AI_LLM_ACCESS |
            cls.AI_TOOL_USE |
            cls.AI_MEMORY_ACCESS
        )

    @classmethod
    def full_agent(cls) -> "Capability":
        """Full capabilities for a trusted agent."""
        return (
            cls.basic_agent() |
            cls.FS_WRITE |
            cls.PROC_SPAWN |
            cls.AI_INTERNET
        )


@dataclass
class CapabilityToken:
    """Token granting specific capabilities."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    capabilities: Capability = Capability.NONE
    process_id: str = ""
    issuer: str = ""
    subject: str = ""
    issued_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    revoked: bool = False

    # Restrictions
    allowed_paths: List[str] = field(default_factory=list)
    allowed_hosts: List[str] = field(default_factory=list)
    allowed_tools: List[str] = field(default_factory=list)
    max_tokens: Optional[int] = None
    rate_limit: Optional[int] = None  # Calls per minute

    # Signature for verification
    signature: str = ""

    def has_capability(self, cap: Capability) -> bool:
        """Check if token has a capability."""
        return bool(self.capabilities & cap)

    def is_valid(self) -> bool:
        """Check if token is valid."""
        if self.revoked:
            return False
        if self.expires_at and datetime.now() > self.expires_at:
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "capabilities": self.capabilities.value,
            "process_id": self.process_id,
            "issuer": self.issuer,
            "subject": self.subject,
            "issued_at": self.issued_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "revoked": self.revoked,
            "allowed_paths": self.allowed_paths,
            "allowed_hosts": self.allowed_hosts,
            "allowed_tools": self.allowed_tools,
        }


class CapabilityManager:
    """
    Manages capability tokens for processes.

    Provides:
    - Token issuance and verification
    - Capability inheritance
    - Token revocation
    - Delegation
    """

    def __init__(self, secret_key: Optional[str] = None):
        self._secret_key = secret_key or secrets.token_hex(32)
        self._tokens: Dict[str, CapabilityToken] = {}
        self._process_tokens: Dict[str, str] = {}  # process_id -> token_id
        self._revocation_list: Set[str] = set()

    def issue_token(
        self,
        capabilities: Capability,
        process_id: str,
        issuer: str = "system",
        duration_seconds: Optional[int] = None,
        **restrictions,
    ) -> CapabilityToken:
        """Issue a new capability token."""
        expires_at = None
        if duration_seconds:
            expires_at = datetime.now() + timedelta(seconds=duration_seconds)

        token = CapabilityToken(
            capabilities=capabilities,
            process_id=process_id,
            issuer=issuer,
            expires_at=expires_at,
            **restrictions,
        )

        # Sign token
        token.signature = self._sign_token(token)

        self._tokens[token.id] = token
        self._process_tokens[process_id] = token.id

        logger.info(
            "Capability token issued",
            token_id=token.id,
            process_id=process_id,
            capabilities=capabilities.value,
        )

        return token

    def verify_token(self, token: CapabilityToken) -> bool:
        """Verify a token's signature and validity."""
        if token.id in self._revocation_list:
            return False

        if not token.is_valid():
            return False

        expected_sig = self._sign_token(token)
        return hmac.compare_digest(token.signature, expected_sig)

    def check_capability(
        self,
        process_id: str,
        capability: Capability,
        **context,
    ) -> bool:
        """Check if a process has a capability."""
        token_id = self._process_tokens.get(process_id)
        if not token_id:
            return False

        token = self._tokens.get(token_id)
        if not token or not self.verify_token(token):
            return False

        if not token.has_capability(capability):
            return False

        # Check restrictions
        if "path" in context and token.allowed_paths:
            path = context["path"]
            if not any(path.startswith(p) for p in token.allowed_paths):
                return False

        if "host" in context and token.allowed_hosts:
            host = context["host"]
            if host not in token.allowed_hosts:
                return False

        if "tool" in context and token.allowed_tools:
            tool = context["tool"]
            if tool not in token.allowed_tools:
                return False

        return True

    def revoke_token(self, token_id: str) -> bool:
        """Revoke a token."""
        token = self._tokens.get(token_id)
        if not token:
            return False

        token.revoked = True
        self._revocation_list.add(token_id)
        self._process_tokens.pop(token.process_id, None)

        logger.info("Capability token revoked", token_id=token_id)
        return True

    def delegate_capabilities(
        self,
        parent_token: CapabilityToken,
        capabilities: Capability,
        child_process_id: str,
        **restrictions,
    ) -> Optional[CapabilityToken]:
        """Delegate a subset of capabilities to a child process."""
        # Can only delegate capabilities parent has
        delegatable = parent_token.capabilities & capabilities

        if delegatable == Capability.NONE:
            logger.warning("No capabilities to delegate")
            return None

        # Inherit restrictions
        merged_restrictions = {
            "allowed_paths": restrictions.get("allowed_paths") or parent_token.allowed_paths,
            "allowed_hosts": restrictions.get("allowed_hosts") or parent_token.allowed_hosts,
            "allowed_tools": restrictions.get("allowed_tools") or parent_token.allowed_tools,
        }

        return self.issue_token(
            capabilities=delegatable,
            process_id=child_process_id,
            issuer=parent_token.process_id,
            **merged_restrictions,
        )

    def get_process_capabilities(self, process_id: str) -> Optional[Capability]:
        """Get capabilities for a process."""
        token_id = self._process_tokens.get(process_id)
        if not token_id:
            return None

        token = self._tokens.get(token_id)
        if not token or not token.is_valid():
            return None

        return token.capabilities

    def _sign_token(self, token: CapabilityToken) -> str:
        """Sign a token."""
        data = f"{token.id}:{token.capabilities.value}:{token.process_id}:{token.issuer}"
        return hmac.new(
            self._secret_key.encode(),
            data.encode(),
            hashlib.sha256,
        ).hexdigest()


def require_capability(capability: Capability) -> Callable[[F], F]:
    """Decorator to require a capability for a function."""
    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            # Expect self to have process_id and capability_manager
            process_id = getattr(self, "process_id", None)
            cap_manager = getattr(self, "capability_manager", None)

            if not process_id or not cap_manager:
                raise PermissionError("Capability check not configured")

            if not cap_manager.check_capability(process_id, capability):
                raise PermissionError(f"Missing capability: {capability.name}")

            return await func(self, *args, **kwargs)
        return wrapper
    return decorator


# === Process Sandboxing ===

@dataclass
class SandboxConfig:
    """Configuration for process sandbox."""
    # File system
    allowed_paths: List[str] = field(default_factory=lambda: ["/tmp"])
    read_only_paths: List[str] = field(default_factory=list)
    hidden_paths: List[str] = field(default_factory=lambda: ["/etc/passwd", "/etc/shadow"])

    # Network
    allow_network: bool = False
    allowed_hosts: List[str] = field(default_factory=list)
    allowed_ports: List[int] = field(default_factory=list)

    # Resources
    max_memory_mb: int = 512
    max_cpu_percent: float = 50.0
    max_file_descriptors: int = 100
    max_processes: int = 10
    max_file_size_mb: int = 100

    # Time
    max_execution_time: float = 3600.0
    max_cpu_time: float = 300.0

    # Capabilities
    capabilities: Capability = Capability.NONE


class Sandbox:
    """
    Process sandbox for isolation.

    Provides:
    - File system isolation
    - Network restrictions
    - Resource limits
    - System call filtering (conceptual)
    """

    def __init__(self, config: SandboxConfig):
        self.config = config
        self._violations: List[Dict[str, Any]] = []

    def check_path_access(self, path: str, write: bool = False) -> bool:
        """Check if path access is allowed."""
        # Normalize path
        path = os.path.normpath(os.path.abspath(path))

        # Check hidden paths
        for hidden in self.config.hidden_paths:
            if path.startswith(hidden):
                self._record_violation("path_hidden", path=path)
                return False

        # Check read-only paths
        if write:
            for readonly in self.config.read_only_paths:
                if path.startswith(readonly):
                    self._record_violation("path_readonly", path=path)
                    return False

        # Check allowed paths
        for allowed in self.config.allowed_paths:
            if path.startswith(allowed):
                return True

        self._record_violation("path_denied", path=path)
        return False

    def check_network_access(self, host: str, port: int) -> bool:
        """Check if network access is allowed."""
        if not self.config.allow_network:
            self._record_violation("network_disabled", host=host, port=port)
            return False

        # Check allowed hosts
        if self.config.allowed_hosts and host not in self.config.allowed_hosts:
            self._record_violation("host_denied", host=host)
            return False

        # Check allowed ports
        if self.config.allowed_ports and port not in self.config.allowed_ports:
            self._record_violation("port_denied", port=port)
            return False

        return True

    def check_resource_limit(self, resource: str, value: float) -> bool:
        """Check if resource usage is within limits."""
        limits = {
            "memory_mb": self.config.max_memory_mb,
            "cpu_percent": self.config.max_cpu_percent,
            "file_descriptors": self.config.max_file_descriptors,
            "processes": self.config.max_processes,
            "file_size_mb": self.config.max_file_size_mb,
        }

        limit = limits.get(resource)
        if limit is None:
            return True

        if value > limit:
            self._record_violation(
                "resource_exceeded",
                resource=resource,
                value=value,
                limit=limit,
            )
            return False

        return True

    def _record_violation(self, violation_type: str, **details) -> None:
        """Record a security violation."""
        self._violations.append({
            "type": violation_type,
            "timestamp": datetime.now().isoformat(),
            **details,
        })

        logger.warning(
            "Sandbox violation",
            violation_type=violation_type,
            **details,
        )

    def get_violations(self) -> List[Dict[str, Any]]:
        """Get recorded violations."""
        return self._violations.copy()

    def clear_violations(self) -> None:
        """Clear recorded violations."""
        self._violations.clear()


class SandboxManager:
    """Manages sandboxes for multiple processes."""

    def __init__(self):
        self._sandboxes: Dict[str, Sandbox] = {}
        self._default_config = SandboxConfig()

    def create_sandbox(
        self,
        process_id: str,
        config: Optional[SandboxConfig] = None,
    ) -> Sandbox:
        """Create a sandbox for a process."""
        sandbox = Sandbox(config or self._default_config)
        self._sandboxes[process_id] = sandbox
        return sandbox

    def get_sandbox(self, process_id: str) -> Optional[Sandbox]:
        """Get sandbox for a process."""
        return self._sandboxes.get(process_id)

    def destroy_sandbox(self, process_id: str) -> None:
        """Destroy a sandbox."""
        self._sandboxes.pop(process_id, None)


# === Audit Logging ===

class AuditEventType(Enum):
    """Types of audit events."""
    # Authentication
    AUTH_LOGIN = auto()
    AUTH_LOGOUT = auto()
    AUTH_FAILED = auto()
    AUTH_TOKEN_ISSUED = auto()
    AUTH_TOKEN_REVOKED = auto()

    # Process
    PROCESS_SPAWN = auto()
    PROCESS_TERMINATE = auto()
    PROCESS_CAPABILITY_CHANGE = auto()

    # File system
    FS_READ = auto()
    FS_WRITE = auto()
    FS_DELETE = auto()
    FS_EXECUTE = auto()

    # Network
    NET_CONNECT = auto()
    NET_LISTEN = auto()
    NET_DATA_TRANSFER = auto()

    # AI
    AI_LLM_CALL = auto()
    AI_TOOL_USE = auto()
    AI_MEMORY_ACCESS = auto()

    # Security
    SEC_VIOLATION = auto()
    SEC_CAPABILITY_DENIED = auto()
    SEC_SANDBOX_BREACH = auto()

    # Admin
    ADMIN_CONFIG_CHANGE = auto()
    ADMIN_USER_CHANGE = auto()


@dataclass
class AuditEvent:
    """An audit log event."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: AuditEventType = AuditEventType.PROCESS_SPAWN
    timestamp: datetime = field(default_factory=datetime.now)
    process_id: Optional[str] = None
    user_id: Optional[str] = None
    action: str = ""
    resource: str = ""
    outcome: str = "success"  # success, failure, denied
    details: Dict[str, Any] = field(default_factory=dict)

    # Context
    source_ip: Optional[str] = None
    session_id: Optional[str] = None
    correlation_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "event_type": self.event_type.name,
            "timestamp": self.timestamp.isoformat(),
            "process_id": self.process_id,
            "user_id": self.user_id,
            "action": self.action,
            "resource": self.resource,
            "outcome": self.outcome,
            "details": self.details,
            "source_ip": self.source_ip,
            "session_id": self.session_id,
            "correlation_id": self.correlation_id,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


class AuditSink(ABC):
    """Abstract sink for audit events."""

    @abstractmethod
    async def write(self, event: AuditEvent) -> None:
        """Write an audit event."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close the sink."""
        pass


class ConsoleAuditSink(AuditSink):
    """Write audit events to console."""

    async def write(self, event: AuditEvent) -> None:
        logger.info(
            "AUDIT",
            event_type=event.event_type.name,
            action=event.action,
            resource=event.resource,
            outcome=event.outcome,
            process_id=event.process_id,
        )

    async def close(self) -> None:
        pass


class FileAuditSink(AuditSink):
    """Write audit events to a file."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self._file = None
        self._lock = asyncio.Lock()

    async def write(self, event: AuditEvent) -> None:
        async with self._lock:
            if self._file is None:
                self._file = open(self.file_path, "a")

            self._file.write(event.to_json() + "\n")
            self._file.flush()

    async def close(self) -> None:
        if self._file:
            self._file.close()


class AuditLogger:
    """
    Audit logging system.

    Features:
    - Multiple sinks (file, console, remote)
    - Event filtering
    - Async writing
    - Correlation tracking
    """

    def __init__(
        self,
        sinks: Optional[List[AuditSink]] = None,
        buffer_size: int = 100,
        flush_interval: float = 5.0,
    ):
        self.sinks = sinks or [ConsoleAuditSink()]
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval

        self._buffer: List[AuditEvent] = []
        self._buffer_lock = asyncio.Lock()
        self._flush_task: Optional[asyncio.Task] = None
        self._shutdown = False

        # Filtering
        self._enabled_types: Set[AuditEventType] = set(AuditEventType)
        self._disabled_types: Set[AuditEventType] = set()

        # Stats
        self._stats = {
            "events_logged": 0,
            "events_dropped": 0,
            "flush_count": 0,
        }

    async def start(self) -> None:
        """Start the audit logger."""
        self._flush_task = asyncio.create_task(self._flush_loop())
        logger.info("Audit logger started")

    async def stop(self) -> None:
        """Stop the audit logger."""
        self._shutdown = True

        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass

        # Final flush
        await self._flush()

        # Close sinks
        for sink in self.sinks:
            await sink.close()

        logger.info("Audit logger stopped", events_logged=self._stats["events_logged"])

    async def log(
        self,
        event_type: AuditEventType,
        action: str,
        resource: str = "",
        outcome: str = "success",
        process_id: Optional[str] = None,
        user_id: Optional[str] = None,
        **details,
    ) -> None:
        """Log an audit event."""
        # Check if type is enabled
        if event_type in self._disabled_types:
            return
        if self._enabled_types and event_type not in self._enabled_types:
            return

        event = AuditEvent(
            event_type=event_type,
            action=action,
            resource=resource,
            outcome=outcome,
            process_id=process_id,
            user_id=user_id,
            details=details,
        )

        async with self._buffer_lock:
            self._buffer.append(event)
            self._stats["events_logged"] += 1

            if len(self._buffer) >= self.buffer_size:
                await self._flush()

    async def _flush(self) -> None:
        """Flush buffer to sinks."""
        async with self._buffer_lock:
            if not self._buffer:
                return

            events = self._buffer[:]
            self._buffer.clear()

        for event in events:
            for sink in self.sinks:
                try:
                    await sink.write(event)
                except Exception as e:
                    logger.error(f"Audit sink error: {e}")
                    self._stats["events_dropped"] += 1

        self._stats["flush_count"] += 1

    async def _flush_loop(self) -> None:
        """Background flush loop."""
        while not self._shutdown:
            try:
                await asyncio.sleep(self.flush_interval)
                await self._flush()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Audit flush error: {e}")

    def enable_type(self, event_type: AuditEventType) -> None:
        """Enable logging for an event type."""
        self._enabled_types.add(event_type)
        self._disabled_types.discard(event_type)

    def disable_type(self, event_type: AuditEventType) -> None:
        """Disable logging for an event type."""
        self._disabled_types.add(event_type)
        self._enabled_types.discard(event_type)

    def get_stats(self) -> Dict[str, Any]:
        return {
            **self._stats,
            "buffer_size": len(self._buffer),
        }


# === Security Manager ===

class SecurityManager:
    """
    Central security manager.

    Coordinates:
    - Capability management
    - Sandboxing
    - Audit logging
    - Access control
    """

    def __init__(
        self,
        secret_key: Optional[str] = None,
        audit_sinks: Optional[List[AuditSink]] = None,
    ):
        self.capability_manager = CapabilityManager(secret_key)
        self.sandbox_manager = SandboxManager()
        self.audit_logger = AuditLogger(audit_sinks)

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize security manager."""
        if self._initialized:
            return

        await self.audit_logger.start()
        self._initialized = True

        await self.audit_logger.log(
            AuditEventType.ADMIN_CONFIG_CHANGE,
            "security_manager_initialized",
        )

    async def shutdown(self) -> None:
        """Shutdown security manager."""
        await self.audit_logger.stop()

    def register_process(
        self,
        process_id: str,
        capabilities: Capability = Capability.NONE,
        sandbox_config: Optional[SandboxConfig] = None,
    ) -> Tuple[CapabilityToken, Sandbox]:
        """Register a new process with security controls."""
        # Issue capability token
        token = self.capability_manager.issue_token(
            capabilities=capabilities,
            process_id=process_id,
        )

        # Create sandbox
        sandbox = self.sandbox_manager.create_sandbox(
            process_id,
            sandbox_config,
        )

        # Audit
        asyncio.create_task(self.audit_logger.log(
            AuditEventType.PROCESS_SPAWN,
            "process_registered",
            resource=process_id,
            capabilities=capabilities.value,
        ))

        return token, sandbox

    def unregister_process(self, process_id: str) -> None:
        """Unregister a process."""
        # Revoke token
        token_id = self.capability_manager._process_tokens.get(process_id)
        if token_id:
            self.capability_manager.revoke_token(token_id)

        # Destroy sandbox
        self.sandbox_manager.destroy_sandbox(process_id)

        # Audit
        asyncio.create_task(self.audit_logger.log(
            AuditEventType.PROCESS_TERMINATE,
            "process_unregistered",
            resource=process_id,
        ))

    async def check_access(
        self,
        process_id: str,
        capability: Capability,
        resource: str = "",
        **context,
    ) -> bool:
        """Check if access should be granted."""
        # Check capability
        has_capability = self.capability_manager.check_capability(
            process_id,
            capability,
            **context,
        )

        # Check sandbox restrictions
        sandbox = self.sandbox_manager.get_sandbox(process_id)
        sandbox_allowed = True

        if sandbox:
            if capability & (Capability.FS_READ | Capability.FS_WRITE | Capability.FS_DELETE):
                sandbox_allowed = sandbox.check_path_access(
                    resource,
                    write=bool(capability & (Capability.FS_WRITE | Capability.FS_DELETE)),
                )
            elif capability & Capability.NET_CONNECT:
                host, port = context.get("host", ""), context.get("port", 0)
                sandbox_allowed = sandbox.check_network_access(host, port)

        granted = has_capability and sandbox_allowed

        # Audit
        await self.audit_logger.log(
            AuditEventType.SEC_CAPABILITY_DENIED if not granted else AuditEventType.FS_READ,
            f"access_check_{capability.name}",
            resource=resource,
            outcome="success" if granted else "denied",
            process_id=process_id,
        )

        return granted


# Global instance
_security_manager: Optional[SecurityManager] = None


def get_security_manager() -> SecurityManager:
    """Get the global security manager."""
    global _security_manager
    if _security_manager is None:
        _security_manager = SecurityManager()
    return _security_manager


async def initialize_security() -> SecurityManager:
    """Initialize and return the global security manager."""
    manager = get_security_manager()
    await manager.initialize()
    return manager
