"""
AION Audit Logger

Comprehensive audit logging for compliance and security monitoring.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections import deque
from dataclasses import asdict
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

import structlog

from aion.security.types import (
    AuditEvent,
    AuditEventSeverity,
    AuditEventType,
    SecurityContext,
)

logger = structlog.get_logger(__name__)


class AuditExporter:
    """Base class for audit exporters."""

    async def export(self, event: AuditEvent) -> bool:
        """Export an audit event. Returns True on success."""
        raise NotImplementedError


class ConsoleExporter(AuditExporter):
    """Export audit events to console."""

    async def export(self, event: AuditEvent) -> bool:
        severity_map = {
            AuditEventSeverity.DEBUG: logger.debug,
            AuditEventSeverity.INFO: logger.info,
            AuditEventSeverity.WARNING: logger.warning,
            AuditEventSeverity.ERROR: logger.error,
            AuditEventSeverity.CRITICAL: logger.critical,
        }

        log_fn = severity_map.get(event.severity, logger.info)
        log_fn(
            f"AUDIT: {event.event_type.value}",
            audit_id=event.id,
            actor=event.actor_id,
            resource=f"{event.resource_type}:{event.resource_id}",
            action=event.action,
            result=event.action_result,
        )
        return True


class FileExporter(AuditExporter):
    """Export audit events to a file."""

    def __init__(self, file_path: str):
        self.file_path = file_path

    async def export(self, event: AuditEvent) -> bool:
        try:
            with open(self.file_path, "a") as f:
                f.write(json.dumps(event.to_dict()) + "\n")
            return True
        except Exception as e:
            logger.error(f"Failed to export audit event to file: {e}")
            return False


class AuditLogger:
    """
    Comprehensive audit logging system.

    Features:
    - Structured audit events
    - Multiple exporters (console, file, webhook, etc.)
    - Buffered writes for performance
    - Query capabilities
    - Event integrity verification
    - Compliance tagging (GDPR, SOC2, HIPAA)
    """

    def __init__(
        self,
        buffer_size: int = 10000,
        flush_interval: float = 5.0,
        max_events: int = 1000000,
        enable_checksums: bool = True,
    ):
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        self.max_events = max_events
        self.enable_checksums = enable_checksums

        # Event buffer
        self._buffer: deque = deque(maxlen=buffer_size)

        # Persistent storage
        self._events: List[AuditEvent] = []

        # Indexes for fast querying
        self._by_user: Dict[str, List[int]] = {}
        self._by_tenant: Dict[str, List[int]] = {}
        self._by_resource: Dict[str, List[int]] = {}
        self._by_type: Dict[str, List[int]] = {}

        # Exporters
        self._exporters: List[AuditExporter] = []

        # Event handlers (for real-time alerting)
        self._handlers: List[Callable[[AuditEvent], None]] = []

        # Flush task
        self._flush_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize audit logger."""
        if self._initialized:
            return

        logger.info("Initializing Audit Logger")

        # Add default console exporter
        self._exporters.append(ConsoleExporter())

        # Start flush loop
        self._flush_task = asyncio.create_task(self._flush_loop())

        self._initialized = True

    async def shutdown(self) -> None:
        """Shutdown and flush remaining events."""
        self._shutdown_event.set()

        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass

        # Final flush
        await self._flush()

        self._initialized = False

    # =========================================================================
    # Logging Methods
    # =========================================================================

    def log(
        self,
        event_type: AuditEventType,
        description: str,
        context: Optional[SecurityContext] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        action: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        success: bool = True,
        error_message: Optional[str] = None,
        severity: AuditEventSeverity = AuditEventSeverity.INFO,
        compliance_frameworks: Optional[List[str]] = None,
    ) -> AuditEvent:
        """
        Log an audit event.

        This is the primary logging method.
        """
        event = AuditEvent(
            event_type=event_type,
            severity=severity,
            description=description,
            message=description,
            actor_type="user" if context and context.user_id else "system",
            actor_id=context.user_id if context else None,
            actor_ip=context.ip_address if context else None,
            actor_user_agent=context.user_agent if context else None,
            tenant_id=context.tenant_id if context else None,
            resource_type=resource_type,
            resource_id=resource_id,
            request_id=context.request_id if context else None,
            trace_id=context.trace_id if context else None,
            session_id=context.session.session_id if context and context.session else None,
            action=action,
            action_result="success" if success else "failure",
            error_message=error_message,
            details=details or {},
            compliance_relevant=bool(compliance_frameworks),
            compliance_frameworks=compliance_frameworks or [],
        )

        # Compute checksum if enabled
        if self.enable_checksums:
            event.checksum = event.compute_checksum()

        # Add to buffer
        self._buffer.append(event)

        # Notify handlers
        for handler in self._handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Audit handler error: {e}")

        return event

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    def log_auth_success(
        self,
        context: SecurityContext,
        method: str,
    ) -> AuditEvent:
        """Log successful authentication."""
        return self.log(
            event_type=AuditEventType.AUTH_LOGIN_SUCCESS,
            description=f"User authenticated via {method}",
            context=context,
            action="login",
            details={"auth_method": method},
        )

    def log_auth_failure(
        self,
        username: str,
        reason: str,
        ip_address: Optional[str] = None,
        method: str = "password",
    ) -> AuditEvent:
        """Log failed authentication."""
        return self.log(
            event_type=AuditEventType.AUTH_LOGIN_FAILURE,
            description=f"Authentication failed for {username}: {reason}",
            action="login",
            details={
                "username": username,
                "reason": reason,
                "auth_method": method,
                "ip_address": ip_address,
            },
            success=False,
            error_message=reason,
            severity=AuditEventSeverity.WARNING,
        )

    def log_access_granted(
        self,
        context: SecurityContext,
        resource: str,
        action: str,
        decided_by: str = "role",
    ) -> AuditEvent:
        """Log access granted."""
        return self.log(
            event_type=AuditEventType.AUTHZ_ACCESS_GRANTED,
            description=f"Access granted: {action} on {resource}",
            context=context,
            resource_type=resource.split(":")[0] if ":" in resource else resource,
            resource_id=resource.split(":")[1] if ":" in resource else None,
            action=action,
            details={"decided_by": decided_by},
        )

    def log_access_denied(
        self,
        context: SecurityContext,
        resource: str,
        action: str,
        reason: str = "permission_denied",
    ) -> AuditEvent:
        """Log access denied."""
        return self.log(
            event_type=AuditEventType.AUTHZ_ACCESS_DENIED,
            description=f"Access denied: {action} on {resource}",
            context=context,
            resource_type=resource.split(":")[0] if ":" in resource else resource,
            resource_id=resource.split(":")[1] if ":" in resource else None,
            action=action,
            details={"reason": reason},
            success=False,
            error_message=reason,
            severity=AuditEventSeverity.WARNING,
        )

    def log_tool_execution(
        self,
        context: SecurityContext,
        tool_name: str,
        agent_id: Optional[str] = None,
        success: bool = True,
        error: Optional[str] = None,
        duration_ms: Optional[float] = None,
    ) -> AuditEvent:
        """Log tool execution."""
        return self.log(
            event_type=AuditEventType.TOOL_EXECUTED,
            description=f"Tool executed: {tool_name}",
            context=context,
            resource_type="tool",
            resource_id=tool_name,
            action="execute",
            details={
                "agent_id": agent_id,
                "duration_ms": duration_ms,
            },
            success=success,
            error_message=error,
            severity=AuditEventSeverity.INFO if success else AuditEventSeverity.WARNING,
        )

    def log_data_access(
        self,
        context: SecurityContext,
        resource_type: str,
        resource_id: str,
        action: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> AuditEvent:
        """Log data access."""
        event_map = {
            "read": AuditEventType.DATA_READ,
            "write": AuditEventType.DATA_WRITE,
            "delete": AuditEventType.DATA_DELETE,
            "export": AuditEventType.DATA_EXPORT,
            "import": AuditEventType.DATA_IMPORT,
        }

        return self.log(
            event_type=event_map.get(action, AuditEventType.DATA_READ),
            description=f"Data {action}: {resource_type}/{resource_id}",
            context=context,
            resource_type=resource_type,
            resource_id=resource_id,
            action=action,
            details=details,
            compliance_frameworks=["GDPR"] if action in ("read", "export") else None,
        )

    def log_security_alert(
        self,
        description: str,
        context: Optional[SecurityContext] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> AuditEvent:
        """Log a security alert."""
        return self.log(
            event_type=AuditEventType.SECURITY_ALERT,
            description=description,
            context=context,
            details=details,
            success=False,
            severity=AuditEventSeverity.CRITICAL,
        )

    def log_agent_boundary_violation(
        self,
        agent_id: str,
        violation_type: str,
        context: Optional[SecurityContext] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> AuditEvent:
        """Log an agent boundary violation."""
        return self.log(
            event_type=AuditEventType.AGENT_BOUNDARY_VIOLATION,
            description=f"Agent boundary violation: {violation_type}",
            context=context,
            resource_type="agent",
            resource_id=agent_id,
            action=violation_type,
            details=details,
            success=False,
            severity=AuditEventSeverity.WARNING,
        )

    # =========================================================================
    # Query Methods
    # =========================================================================

    async def query(
        self,
        event_types: Optional[List[AuditEventType]] = None,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        success: Optional[bool] = None,
        severity: Optional[AuditEventSeverity] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[AuditEvent]:
        """Query audit events with filters."""
        # Flush buffer first
        await self._flush()

        events = self._events

        # Apply filters
        if event_types:
            events = [e for e in events if e.event_type in event_types]

        if user_id:
            events = [e for e in events if e.actor_id == user_id]

        if tenant_id:
            events = [e for e in events if e.tenant_id == tenant_id]

        if resource_type:
            events = [e for e in events if e.resource_type == resource_type]

        if resource_id:
            events = [e for e in events if e.resource_id == resource_id]

        if start_time:
            events = [e for e in events if e.timestamp >= start_time]

        if end_time:
            events = [e for e in events if e.timestamp <= end_time]

        if success is not None:
            result_filter = "success" if success else "failure"
            events = [e for e in events if e.action_result == result_filter]

        if severity:
            events = [e for e in events if e.severity == severity]

        # Sort by timestamp descending
        events = sorted(events, key=lambda e: e.timestamp, reverse=True)

        return events[offset : offset + limit]

    async def count(
        self,
        event_types: Optional[List[AuditEventType]] = None,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> int:
        """Count audit events matching filters."""
        events = await self.query(
            event_types=event_types,
            user_id=user_id,
            tenant_id=tenant_id,
            start_time=start_time,
            end_time=end_time,
            limit=self.max_events,
        )
        return len(events)

    async def get_event(self, event_id: str) -> Optional[AuditEvent]:
        """Get a specific audit event by ID."""
        for event in self._events:
            if event.id == event_id:
                return event
        return None

    # =========================================================================
    # Export
    # =========================================================================

    def add_exporter(self, exporter: AuditExporter) -> None:
        """Add an audit event exporter."""
        self._exporters.append(exporter)

    def add_handler(self, handler: Callable[[AuditEvent], None]) -> None:
        """Add a real-time event handler."""
        self._handlers.append(handler)

    async def export_events(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        format: str = "json",
    ) -> str:
        """Export events to a string format."""
        events = await self.query(
            start_time=start_time,
            end_time=end_time,
            limit=self.max_events,
        )

        if format == "json":
            return json.dumps([e.to_dict() for e in events], indent=2)
        elif format == "jsonl":
            return "\n".join(json.dumps(e.to_dict()) for e in events)
        else:
            raise ValueError(f"Unsupported format: {format}")

    # =========================================================================
    # Flush
    # =========================================================================

    async def _flush_loop(self) -> None:
        """Background flush loop."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(self.flush_interval)
                await self._flush()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Audit flush error: {e}")

    async def _flush(self) -> None:
        """Flush buffer to storage and exporters."""
        while self._buffer:
            event = self._buffer.popleft()

            # Store event
            idx = len(self._events)
            self._events.append(event)

            # Update indexes
            if event.actor_id:
                if event.actor_id not in self._by_user:
                    self._by_user[event.actor_id] = []
                self._by_user[event.actor_id].append(idx)

            if event.tenant_id:
                if event.tenant_id not in self._by_tenant:
                    self._by_tenant[event.tenant_id] = []
                self._by_tenant[event.tenant_id].append(idx)

            if event.resource_type:
                key = f"{event.resource_type}:{event.resource_id or '*'}"
                if key not in self._by_resource:
                    self._by_resource[key] = []
                self._by_resource[key].append(idx)

            type_key = event.event_type.value
            if type_key not in self._by_type:
                self._by_type[type_key] = []
            self._by_type[type_key].append(idx)

            # Export
            for exporter in self._exporters[1:]:  # Skip console (already logged)
                try:
                    await exporter.export(event)
                except Exception as e:
                    logger.error(f"Audit export error: {e}")

        # Trim events if needed
        if len(self._events) > self.max_events:
            trim_count = len(self._events) - self.max_events
            self._events = self._events[trim_count:]
            # Rebuild indexes (expensive but rare)
            self._rebuild_indexes()

    def _rebuild_indexes(self) -> None:
        """Rebuild all indexes after trimming."""
        self._by_user.clear()
        self._by_tenant.clear()
        self._by_resource.clear()
        self._by_type.clear()

        for idx, event in enumerate(self._events):
            if event.actor_id:
                if event.actor_id not in self._by_user:
                    self._by_user[event.actor_id] = []
                self._by_user[event.actor_id].append(idx)

            if event.tenant_id:
                if event.tenant_id not in self._by_tenant:
                    self._by_tenant[event.tenant_id] = []
                self._by_tenant[event.tenant_id].append(idx)

            if event.resource_type:
                key = f"{event.resource_type}:{event.resource_id or '*'}"
                if key not in self._by_resource:
                    self._by_resource[key] = []
                self._by_resource[key].append(idx)

            type_key = event.event_type.value
            if type_key not in self._by_type:
                self._by_type[type_key] = []
            self._by_type[type_key].append(idx)

    def get_stats(self) -> Dict[str, Any]:
        """Get audit logger statistics."""
        return {
            "buffer_size": len(self._buffer),
            "total_events": len(self._events),
            "exporters": len(self._exporters),
            "handlers": len(self._handlers),
            "unique_users": len(self._by_user),
            "unique_tenants": len(self._by_tenant),
        }
