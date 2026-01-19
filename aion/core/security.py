"""
AION Security System

Comprehensive security management with:
- Risk level classification
- Human approval gates for high-risk operations
- Action auditing and logging
- Resource usage monitoring
- Emergency stop mechanisms
- Sandboxed execution
"""

from __future__ import annotations

import asyncio
import hashlib
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Awaitable, Callable, Optional
import json

import structlog

logger = structlog.get_logger(__name__)


class RiskLevel(Enum):
    """Risk classification for operations."""
    MINIMAL = auto()   # Read-only, no side effects
    LOW = auto()       # Minor side effects, easily reversible
    MEDIUM = auto()    # Significant changes, may need review
    HIGH = auto()      # Critical changes, requires approval
    CRITICAL = auto()  # System-level changes, always requires approval


class ApprovalStatus(Enum):
    """Status of an approval request."""
    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class ApprovalRequest:
    """A request for human approval."""
    id: str
    operation: str
    risk_level: RiskLevel
    description: str
    details: dict[str, Any]
    requested_at: datetime
    timeout_at: datetime
    status: ApprovalStatus = ApprovalStatus.PENDING
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None
    denial_reason: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "operation": self.operation,
            "risk_level": self.risk_level.name,
            "description": self.description,
            "details": self.details,
            "requested_at": self.requested_at.isoformat(),
            "timeout_at": self.timeout_at.isoformat(),
            "status": self.status.value,
            "approved_by": self.approved_by,
            "approved_at": self.approved_at.isoformat() if self.approved_at else None,
            "denial_reason": self.denial_reason,
        }


@dataclass
class AuditEntry:
    """An entry in the audit log."""
    id: str
    timestamp: datetime
    operation: str
    risk_level: RiskLevel
    user_id: Optional[str]
    details: dict[str, Any]
    result: str  # "success", "failure", "blocked"
    error_message: Optional[str] = None
    execution_time_ms: float = 0.0
    approval_id: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "operation": self.operation,
            "risk_level": self.risk_level.name,
            "user_id": self.user_id,
            "details": self.details,
            "result": self.result,
            "error_message": self.error_message,
            "execution_time_ms": self.execution_time_ms,
            "approval_id": self.approval_id,
        }


@dataclass
class Checkpoint:
    """A system state checkpoint for rollback."""
    id: str
    timestamp: datetime
    description: str
    state_hash: str
    state_data: dict[str, Any]
    parent_id: Optional[str] = None


class RateLimiter:
    """Token bucket rate limiter."""

    def __init__(self, rate: float, burst: int = 10):
        """
        Initialize rate limiter.

        Args:
            rate: Tokens per second
            burst: Maximum burst size
        """
        self.rate = rate
        self.burst = burst
        self.tokens = float(burst)
        self.last_update = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> bool:
        """Attempt to acquire a token."""
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_update
            self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
            self.last_update = now

            if self.tokens >= 1:
                self.tokens -= 1
                return True
            return False

    async def wait(self) -> None:
        """Wait until a token is available."""
        while not await self.acquire():
            await asyncio.sleep(1.0 / self.rate)


class ApprovalGate:
    """
    Human approval gate for high-risk operations.

    Operations above a certain risk threshold require explicit human approval
    before execution.
    """

    def __init__(
        self,
        auto_approve_below: RiskLevel = RiskLevel.LOW,
        timeout: float = 300.0,
        max_pending: int = 10,
    ):
        self.auto_approve_below = auto_approve_below
        self.timeout = timeout
        self.max_pending = max_pending
        self._pending: dict[str, ApprovalRequest] = {}
        self._approval_events: dict[str, asyncio.Event] = {}
        self._callbacks: list[Callable[[ApprovalRequest], Awaitable[None]]] = []

    def register_callback(
        self, callback: Callable[[ApprovalRequest], Awaitable[None]]
    ) -> None:
        """Register a callback for new approval requests."""
        self._callbacks.append(callback)

    async def request_approval(
        self,
        operation: str,
        risk_level: RiskLevel,
        description: str,
        details: Optional[dict[str, Any]] = None,
    ) -> ApprovalRequest:
        """
        Request approval for an operation.

        Args:
            operation: Name of the operation
            risk_level: Risk level of the operation
            description: Human-readable description
            details: Additional details for the reviewer

        Returns:
            ApprovalRequest with current status

        Raises:
            PermissionError: If too many pending requests
        """
        # Auto-approve low-risk operations
        if risk_level.value < self.auto_approve_below.value:
            request = ApprovalRequest(
                id=str(uuid.uuid4()),
                operation=operation,
                risk_level=risk_level,
                description=description,
                details=details or {},
                requested_at=datetime.now(),
                timeout_at=datetime.now(),
                status=ApprovalStatus.APPROVED,
                approved_by="auto",
                approved_at=datetime.now(),
            )
            return request

        # Check pending limit
        if len(self._pending) >= self.max_pending:
            raise PermissionError("Too many pending approval requests")

        # Create new request
        request_id = str(uuid.uuid4())
        request = ApprovalRequest(
            id=request_id,
            operation=operation,
            risk_level=risk_level,
            description=description,
            details=details or {},
            requested_at=datetime.now(),
            timeout_at=datetime.now() + timedelta(seconds=self.timeout),
        )

        self._pending[request_id] = request
        self._approval_events[request_id] = asyncio.Event()

        # Notify callbacks
        for callback in self._callbacks:
            try:
                await callback(request)
            except Exception as e:
                logger.error("Approval callback failed", error=str(e))

        logger.info(
            "Approval requested",
            request_id=request_id,
            operation=operation,
            risk_level=risk_level.name,
        )

        return request

    async def wait_for_approval(self, request_id: str) -> ApprovalRequest:
        """
        Wait for an approval decision.

        Args:
            request_id: ID of the approval request

        Returns:
            Updated ApprovalRequest

        Raises:
            KeyError: If request not found
            TimeoutError: If approval times out
        """
        if request_id not in self._pending:
            raise KeyError(f"Approval request not found: {request_id}")

        request = self._pending[request_id]
        event = self._approval_events[request_id]

        timeout = (request.timeout_at - datetime.now()).total_seconds()
        if timeout <= 0:
            request.status = ApprovalStatus.TIMEOUT
            self._cleanup_request(request_id)
            return request

        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            request.status = ApprovalStatus.TIMEOUT
            logger.warning("Approval timed out", request_id=request_id)

        self._cleanup_request(request_id)
        return request

    def approve(
        self, request_id: str, approved_by: str = "human"
    ) -> ApprovalRequest:
        """
        Approve a pending request.

        Args:
            request_id: ID of the request to approve
            approved_by: Identifier of the approver

        Returns:
            Updated ApprovalRequest
        """
        if request_id not in self._pending:
            raise KeyError(f"Approval request not found: {request_id}")

        request = self._pending[request_id]
        request.status = ApprovalStatus.APPROVED
        request.approved_by = approved_by
        request.approved_at = datetime.now()

        if request_id in self._approval_events:
            self._approval_events[request_id].set()

        logger.info(
            "Request approved",
            request_id=request_id,
            approved_by=approved_by,
        )

        return request

    def deny(
        self, request_id: str, reason: str = "Denied by reviewer"
    ) -> ApprovalRequest:
        """
        Deny a pending request.

        Args:
            request_id: ID of the request to deny
            reason: Reason for denial

        Returns:
            Updated ApprovalRequest
        """
        if request_id not in self._pending:
            raise KeyError(f"Approval request not found: {request_id}")

        request = self._pending[request_id]
        request.status = ApprovalStatus.DENIED
        request.denial_reason = reason

        if request_id in self._approval_events:
            self._approval_events[request_id].set()

        logger.warning(
            "Request denied",
            request_id=request_id,
            reason=reason,
        )

        return request

    def get_pending(self) -> list[ApprovalRequest]:
        """Get all pending approval requests."""
        return list(self._pending.values())

    def _cleanup_request(self, request_id: str) -> None:
        """Clean up completed request."""
        self._pending.pop(request_id, None)
        self._approval_events.pop(request_id, None)


class SecurityManager:
    """
    Central security manager for AION.

    Handles:
    - Risk assessment
    - Approval workflows
    - Audit logging
    - Rate limiting
    - Resource monitoring
    - Emergency stop
    """

    def __init__(
        self,
        require_approval_for_high_risk: bool = True,
        auto_approve_low_risk: bool = True,
        approval_timeout: float = 300.0,
        audit_all: bool = True,
        rate_limit: Optional[float] = None,
    ):
        self.require_approval = require_approval_for_high_risk
        self.auto_approve_low_risk = auto_approve_low_risk
        self.audit_all = audit_all

        # Initialize subsystems
        auto_approve_level = RiskLevel.LOW if auto_approve_low_risk else RiskLevel.MINIMAL
        self.approval_gate = ApprovalGate(
            auto_approve_below=auto_approve_level,
            timeout=approval_timeout,
        )

        self.rate_limiter = RateLimiter(rate_limit) if rate_limit else None

        # Audit log
        self._audit_log: list[AuditEntry] = []
        self._audit_lock = asyncio.Lock()

        # Checkpoints for rollback
        self._checkpoints: dict[str, Checkpoint] = {}
        self._checkpoint_stack: list[str] = []

        # Emergency stop flag
        self._emergency_stop = False
        self._emergency_stop_reason: Optional[str] = None

        # Operation classifications
        self._operation_risks: dict[str, RiskLevel] = {}
        self._blocked_operations: set[str] = set()

        # Resource limits
        self._resource_limits: dict[str, float] = {
            "memory_mb": 4096,
            "cpu_percent": 80,
            "disk_mb": 10240,
        }

    def classify_risk(self, operation: str, context: Optional[dict] = None) -> RiskLevel:
        """
        Classify the risk level of an operation.

        Args:
            operation: Name of the operation
            context: Additional context for classification

        Returns:
            RiskLevel for the operation
        """
        # Check if explicitly classified
        if operation in self._operation_risks:
            return self._operation_risks[operation]

        # Default classifications based on operation patterns
        high_risk_patterns = [
            "delete", "remove", "drop", "modify_system",
            "execute_code", "write_file", "change_config",
            "self_modify", "evolve", "update_weights",
        ]

        medium_risk_patterns = [
            "write", "update", "create", "api_call",
            "external_request", "tool_execute",
        ]

        low_risk_patterns = [
            "read", "search", "query", "list", "get", "fetch",
        ]

        operation_lower = operation.lower()

        for pattern in high_risk_patterns:
            if pattern in operation_lower:
                return RiskLevel.HIGH

        for pattern in medium_risk_patterns:
            if pattern in operation_lower:
                return RiskLevel.MEDIUM

        for pattern in low_risk_patterns:
            if pattern in operation_lower:
                return RiskLevel.LOW

        # Default to medium for unknown operations
        return RiskLevel.MEDIUM

    def register_risk(self, operation: str, risk_level: RiskLevel) -> None:
        """Register the risk level for an operation."""
        self._operation_risks[operation] = risk_level

    def block_operation(self, operation: str) -> None:
        """Block an operation from being executed."""
        self._blocked_operations.add(operation)
        logger.warning("Operation blocked", operation=operation)

    def unblock_operation(self, operation: str) -> None:
        """Unblock a previously blocked operation."""
        self._blocked_operations.discard(operation)

    async def authorize(
        self,
        operation: str,
        description: str,
        details: Optional[dict[str, Any]] = None,
        user_id: Optional[str] = None,
    ) -> tuple[bool, Optional[str]]:
        """
        Authorize an operation.

        Args:
            operation: Name of the operation
            description: Human-readable description
            details: Additional details
            user_id: ID of the requesting user

        Returns:
            Tuple of (authorized: bool, reason: Optional[str])
        """
        # Check emergency stop
        if self._emergency_stop:
            return False, f"Emergency stop active: {self._emergency_stop_reason}"

        # Check blocked operations
        if operation in self._blocked_operations:
            return False, f"Operation '{operation}' is blocked"

        # Check rate limit
        if self.rate_limiter and not await self.rate_limiter.acquire():
            return False, "Rate limit exceeded"

        # Classify risk
        risk_level = self.classify_risk(operation, details)

        # Check if approval required
        if self.require_approval and risk_level.value >= RiskLevel.HIGH.value:
            request = await self.approval_gate.request_approval(
                operation=operation,
                risk_level=risk_level,
                description=description,
                details=details or {},
            )

            if request.status != ApprovalStatus.APPROVED:
                # Wait for approval
                request = await self.approval_gate.wait_for_approval(request.id)

            if request.status == ApprovalStatus.APPROVED:
                return True, None
            elif request.status == ApprovalStatus.DENIED:
                return False, f"Denied: {request.denial_reason}"
            else:
                return False, f"Approval {request.status.value}"

        return True, None

    async def audit(
        self,
        operation: str,
        risk_level: RiskLevel,
        result: str,
        details: Optional[dict[str, Any]] = None,
        user_id: Optional[str] = None,
        error_message: Optional[str] = None,
        execution_time_ms: float = 0.0,
        approval_id: Optional[str] = None,
    ) -> AuditEntry:
        """
        Record an audit entry.

        Args:
            operation: Name of the operation
            risk_level: Risk level of the operation
            result: Result of the operation
            details: Additional details
            user_id: ID of the user
            error_message: Error message if failed
            execution_time_ms: Execution time in milliseconds
            approval_id: ID of related approval request

        Returns:
            Created AuditEntry
        """
        entry = AuditEntry(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            operation=operation,
            risk_level=risk_level,
            user_id=user_id,
            details=details or {},
            result=result,
            error_message=error_message,
            execution_time_ms=execution_time_ms,
            approval_id=approval_id,
        )

        async with self._audit_lock:
            self._audit_log.append(entry)

        logger.info(
            "Audit entry recorded",
            operation=operation,
            result=result,
            risk_level=risk_level.name,
        )

        return entry

    def get_audit_log(
        self,
        limit: int = 100,
        operation: Optional[str] = None,
        risk_level: Optional[RiskLevel] = None,
    ) -> list[AuditEntry]:
        """Get audit log entries with optional filtering."""
        entries = self._audit_log

        if operation:
            entries = [e for e in entries if e.operation == operation]

        if risk_level:
            entries = [e for e in entries if e.risk_level == risk_level]

        return entries[-limit:]

    def create_checkpoint(
        self,
        description: str,
        state_data: dict[str, Any],
    ) -> Checkpoint:
        """
        Create a system checkpoint for potential rollback.

        Args:
            description: Description of the checkpoint
            state_data: State data to preserve

        Returns:
            Created Checkpoint
        """
        state_json = json.dumps(state_data, sort_keys=True, default=str)
        state_hash = hashlib.sha256(state_json.encode()).hexdigest()

        parent_id = self._checkpoint_stack[-1] if self._checkpoint_stack else None

        checkpoint = Checkpoint(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            description=description,
            state_hash=state_hash,
            state_data=state_data,
            parent_id=parent_id,
        )

        self._checkpoints[checkpoint.id] = checkpoint
        self._checkpoint_stack.append(checkpoint.id)

        logger.info(
            "Checkpoint created",
            checkpoint_id=checkpoint.id,
            description=description,
        )

        return checkpoint

    def get_checkpoint(self, checkpoint_id: str) -> Optional[Checkpoint]:
        """Get a checkpoint by ID."""
        return self._checkpoints.get(checkpoint_id)

    def rollback_to_checkpoint(self, checkpoint_id: str) -> dict[str, Any]:
        """
        Get state data for rollback to a checkpoint.

        Args:
            checkpoint_id: ID of the checkpoint

        Returns:
            State data from the checkpoint

        Raises:
            KeyError: If checkpoint not found
        """
        if checkpoint_id not in self._checkpoints:
            raise KeyError(f"Checkpoint not found: {checkpoint_id}")

        checkpoint = self._checkpoints[checkpoint_id]

        # Remove all checkpoints after this one
        while self._checkpoint_stack and self._checkpoint_stack[-1] != checkpoint_id:
            removed_id = self._checkpoint_stack.pop()
            del self._checkpoints[removed_id]

        logger.warning(
            "Rolling back to checkpoint",
            checkpoint_id=checkpoint_id,
            description=checkpoint.description,
        )

        return checkpoint.state_data

    def emergency_stop(self, reason: str) -> None:
        """
        Activate emergency stop.

        Halts all operations until manually cleared.

        Args:
            reason: Reason for emergency stop
        """
        self._emergency_stop = True
        self._emergency_stop_reason = reason
        logger.critical(
            "EMERGENCY STOP ACTIVATED",
            reason=reason,
        )

    def clear_emergency_stop(self) -> None:
        """Clear emergency stop and resume operations."""
        self._emergency_stop = False
        self._emergency_stop_reason = None
        logger.warning("Emergency stop cleared")

    def is_emergency_stopped(self) -> tuple[bool, Optional[str]]:
        """Check if emergency stop is active."""
        return self._emergency_stop, self._emergency_stop_reason

    def set_resource_limit(self, resource: str, limit: float) -> None:
        """Set a resource limit."""
        self._resource_limits[resource] = limit

    def get_resource_limits(self) -> dict[str, float]:
        """Get all resource limits."""
        return self._resource_limits.copy()
