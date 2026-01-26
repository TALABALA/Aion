"""
AION Approval Manager

Human-in-the-loop approval gates for workflows.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional
import uuid

import structlog

from aion.automation.types import (
    ApprovalRequest,
    ApprovalResponse,
    ApprovalStatus,
)

logger = structlog.get_logger(__name__)


class ApprovalManager:
    """
    Manages approval requests for workflows.

    Features:
    - Request creation and tracking
    - Approval/rejection handling
    - Timeout management
    - Notification integration
    - Audit trail
    """

    def __init__(
        self,
        default_timeout_hours: float = 24.0,
        notification_handler: Optional[Callable] = None,
    ):
        self.default_timeout_hours = default_timeout_hours
        self.notification_handler = notification_handler

        # Pending requests
        self._requests: Dict[str, ApprovalRequest] = {}

        # Waiting futures
        self._waiters: Dict[str, asyncio.Future] = {}

        # Event callbacks
        self._on_request_created: List[Callable] = []
        self._on_request_responded: List[Callable] = []
        self._on_request_expired: List[Callable] = []

        # Background task
        self._expiry_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the approval manager."""
        if self._initialized:
            return

        # Start expiry check loop
        self._expiry_task = asyncio.create_task(self._expiry_loop())

        self._initialized = True
        logger.info("Approval manager initialized")

    async def shutdown(self) -> None:
        """Shutdown the approval manager."""
        self._shutdown_event.set()

        if self._expiry_task:
            self._expiry_task.cancel()
            try:
                await self._expiry_task
            except asyncio.CancelledError:
                pass

        # Cancel all waiting futures
        for future in self._waiters.values():
            if not future.done():
                future.cancel()

        self._initialized = False

    # === Request Management ===

    async def create_request(
        self,
        execution_id: str,
        step_id: str,
        message: str,
        approvers: List[str] = None,
        timeout_hours: float = None,
        workflow_id: str = "",
        workflow_name: str = "",
        title: str = "",
        details: Dict[str, Any] = None,
        requires_all: bool = False,
    ) -> ApprovalRequest:
        """Create an approval request."""
        timeout = timeout_hours or self.default_timeout_hours
        expires_at = datetime.now() + timedelta(hours=timeout)

        request = ApprovalRequest(
            execution_id=execution_id,
            step_id=step_id,
            workflow_id=workflow_id,
            workflow_name=workflow_name,
            title=title or "Approval Required",
            message=message,
            details=details or {},
            approvers=approvers or [],
            requires_all=requires_all,
            expires_at=expires_at,
        )

        self._requests[request.id] = request

        # Fire callbacks
        await self._fire_callbacks(self._on_request_created, request)

        # Send notifications
        if self.notification_handler:
            try:
                await self.notification_handler(request)
            except Exception as e:
                logger.error("notification_error", error=str(e))

        logger.info(
            "approval_request_created",
            request_id=request.id,
            execution_id=execution_id,
            approvers=approvers,
            expires_at=expires_at.isoformat(),
        )

        return request

    async def get_request(
        self,
        request_id: str,
    ) -> Optional[ApprovalRequest]:
        """Get an approval request by ID."""
        return self._requests.get(request_id)

    async def list_requests(
        self,
        status: Optional[ApprovalStatus] = None,
        approver: Optional[str] = None,
        execution_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[ApprovalRequest]:
        """List approval requests with filters."""
        requests = list(self._requests.values())

        if status:
            requests = [r for r in requests if r.status == status]

        if approver:
            requests = [r for r in requests if approver in r.approvers]

        if execution_id:
            requests = [r for r in requests if r.execution_id == execution_id]

        if workflow_id:
            requests = [r for r in requests if r.workflow_id == workflow_id]

        # Sort by created_at descending
        requests.sort(key=lambda r: r.created_at, reverse=True)

        return requests[:limit]

    async def get_pending_for_approver(
        self,
        approver: str,
    ) -> List[ApprovalRequest]:
        """Get pending requests for a specific approver."""
        return [
            r for r in self._requests.values()
            if r.status == ApprovalStatus.PENDING
            and approver in r.approvers
        ]

    # === Approval/Rejection ===

    async def approve(
        self,
        request_id: str,
        approver: str,
        message: str = "",
    ) -> bool:
        """Approve a request."""
        request = self._requests.get(request_id)
        if not request:
            logger.warning("approval_request_not_found", request_id=request_id)
            return False

        if request.status != ApprovalStatus.PENDING:
            logger.warning("approval_request_not_pending", request_id=request_id)
            return False

        if request.is_expired():
            request.status = ApprovalStatus.EXPIRED
            return False

        # Check if approver is authorized
        if request.approvers and approver not in request.approvers:
            logger.warning(
                "approver_not_authorized",
                request_id=request_id,
                approver=approver,
            )
            return False

        # Record approval
        request.approve(approver, message)

        logger.info(
            "approval_request_approved",
            request_id=request_id,
            approver=approver,
        )

        # Fire callbacks
        await self._fire_callbacks(self._on_request_responded, request)

        # Resolve waiter if approved
        if request.status == ApprovalStatus.APPROVED:
            self._resolve_waiter(request_id, ApprovalResult(
                approved=True,
                approved_by=approver,
                message=message,
            ))

        return True

    async def reject(
        self,
        request_id: str,
        approver: str,
        message: str = "",
    ) -> bool:
        """Reject a request."""
        request = self._requests.get(request_id)
        if not request:
            return False

        if request.status != ApprovalStatus.PENDING:
            return False

        # Record rejection
        request.reject(approver, message)

        logger.info(
            "approval_request_rejected",
            request_id=request_id,
            approver=approver,
        )

        # Fire callbacks
        await self._fire_callbacks(self._on_request_responded, request)

        # Resolve waiter
        self._resolve_waiter(request_id, ApprovalResult(
            approved=False,
            approved_by=approver,
            message=message,
        ))

        return True

    async def cancel(
        self,
        request_id: str,
    ) -> bool:
        """Cancel a pending request."""
        request = self._requests.get(request_id)
        if not request:
            return False

        if request.status != ApprovalStatus.PENDING:
            return False

        request.status = ApprovalStatus.CANCELLED

        # Resolve waiter
        self._resolve_waiter(request_id, ApprovalResult(
            approved=False,
            message="Request cancelled",
        ))

        return True

    # === Waiting ===

    async def wait_for_decision(
        self,
        request_id: str,
        timeout_hours: float = None,
    ) -> "ApprovalResult":
        """
        Wait for an approval decision.

        Args:
            request_id: Request to wait for
            timeout_hours: Override timeout

        Returns:
            ApprovalResult with decision
        """
        request = self._requests.get(request_id)
        if not request:
            return ApprovalResult(approved=False, message="Request not found")

        # Check if already decided
        if request.status == ApprovalStatus.APPROVED:
            return ApprovalResult(
                approved=True,
                approved_by=request.responded_by,
                message=request.response_message,
            )
        elif request.status in (ApprovalStatus.REJECTED, ApprovalStatus.EXPIRED, ApprovalStatus.CANCELLED):
            return ApprovalResult(
                approved=False,
                approved_by=request.responded_by,
                message=request.response_message or f"Request {request.status.value}",
            )

        # Create a future to wait on
        future = asyncio.get_event_loop().create_future()
        self._waiters[request_id] = future

        # Calculate timeout
        timeout = timeout_hours or self.default_timeout_hours
        timeout_seconds = timeout * 3600

        try:
            result = await asyncio.wait_for(future, timeout=timeout_seconds)
            return result

        except asyncio.TimeoutError:
            # Mark as expired
            if request.status == ApprovalStatus.PENDING:
                request.status = ApprovalStatus.EXPIRED
                await self._fire_callbacks(self._on_request_expired, request)

            return ApprovalResult(approved=False, message="Approval timed out")

        except asyncio.CancelledError:
            return ApprovalResult(approved=False, message="Wait cancelled")

        finally:
            self._waiters.pop(request_id, None)

    def _resolve_waiter(
        self,
        request_id: str,
        result: "ApprovalResult",
    ) -> None:
        """Resolve a waiting future."""
        future = self._waiters.get(request_id)
        if future and not future.done():
            future.set_result(result)

    # === Expiry Loop ===

    async def _expiry_loop(self) -> None:
        """Background loop to check for expired requests."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(60)  # Check every minute

                now = datetime.now()

                for request in list(self._requests.values()):
                    if request.status != ApprovalStatus.PENDING:
                        continue

                    if request.is_expired():
                        request.status = ApprovalStatus.EXPIRED

                        logger.info(
                            "approval_request_expired",
                            request_id=request.id,
                        )

                        await self._fire_callbacks(self._on_request_expired, request)

                        # Resolve waiter
                        self._resolve_waiter(request.id, ApprovalResult(
                            approved=False,
                            message="Request expired",
                        ))

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("expiry_loop_error", error=str(e))

    # === Event Callbacks ===

    def on_request_created(self, callback: Callable) -> None:
        """Register callback for request creation."""
        self._on_request_created.append(callback)

    def on_request_responded(self, callback: Callable) -> None:
        """Register callback for request response."""
        self._on_request_responded.append(callback)

    def on_request_expired(self, callback: Callable) -> None:
        """Register callback for request expiry."""
        self._on_request_expired.append(callback)

    async def _fire_callbacks(
        self,
        callbacks: List[Callable],
        *args,
    ) -> None:
        """Fire callbacks."""
        for callback in callbacks:
            try:
                result = callback(*args)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error("callback_error", error=str(e))

    # === Statistics ===

    def get_stats(self) -> Dict[str, Any]:
        """Get approval manager statistics."""
        status_counts = {}
        for status in ApprovalStatus:
            status_counts[status.value] = len([
                r for r in self._requests.values()
                if r.status == status
            ])

        return {
            "total_requests": len(self._requests),
            "pending": status_counts.get("pending", 0),
            "approved": status_counts.get("approved", 0),
            "rejected": status_counts.get("rejected", 0),
            "expired": status_counts.get("expired", 0),
            "active_waiters": len(self._waiters),
        }


# === Result Types ===


from dataclasses import dataclass


@dataclass
class ApprovalResult:
    """Result of an approval decision."""
    approved: bool
    approved_by: Optional[str] = None
    message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "approved": self.approved,
            "approved_by": self.approved_by,
            "message": self.message,
        }
