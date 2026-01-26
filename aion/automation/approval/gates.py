"""
AION Approval Gates

Different types of approval gates for workflows.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import uuid

import structlog

from aion.automation.types import ApprovalRequest, ApprovalStatus

logger = structlog.get_logger(__name__)


class ApprovalGate(ABC):
    """
    Base class for approval gates.

    Approval gates define the rules for how approvals work.
    """

    @abstractmethod
    async def check(
        self,
        request: ApprovalRequest,
    ) -> bool:
        """
        Check if the approval conditions are met.

        Returns:
            True if approved, False otherwise
        """
        pass

    @abstractmethod
    async def can_approve(
        self,
        request: ApprovalRequest,
        approver: str,
    ) -> bool:
        """
        Check if a user can approve a request.

        Returns:
            True if the user can approve
        """
        pass


class SingleApproverGate(ApprovalGate):
    """
    Single approver gate.

    Any one approver can approve the request.
    """

    async def check(
        self,
        request: ApprovalRequest,
    ) -> bool:
        """Check if any approver has approved."""
        approvals = [r for r in request.responses if r.decision == "approved"]
        return len(approvals) > 0

    async def can_approve(
        self,
        request: ApprovalRequest,
        approver: str,
    ) -> bool:
        """Check if user is in approvers list."""
        if not request.approvers:
            return True  # Anyone can approve
        return approver in request.approvers


class MultiApproverGate(ApprovalGate):
    """
    Multi-approver gate.

    Requires multiple approvers based on configuration.
    """

    def __init__(
        self,
        min_approvals: int = 2,
        require_all: bool = False,
    ):
        self.min_approvals = min_approvals
        self.require_all = require_all

    async def check(
        self,
        request: ApprovalRequest,
    ) -> bool:
        """Check if enough approvers have approved."""
        approvals = [r for r in request.responses if r.decision == "approved"]
        approval_count = len(approvals)

        if self.require_all or request.requires_all:
            return approval_count >= len(request.approvers)
        else:
            return approval_count >= self.min_approvals

    async def can_approve(
        self,
        request: ApprovalRequest,
        approver: str,
    ) -> bool:
        """Check if user is in approvers list and hasn't already approved."""
        if not request.approvers:
            return True

        if approver not in request.approvers:
            return False

        # Check if already responded
        already_responded = any(
            r.approver == approver for r in request.responses
        )
        return not already_responded


class TimeoutGate(ApprovalGate):
    """
    Timeout gate.

    Auto-approves or auto-rejects after a timeout.
    """

    def __init__(
        self,
        timeout_hours: float = 24.0,
        auto_approve: bool = False,
    ):
        self.timeout_hours = timeout_hours
        self.auto_approve = auto_approve

    async def check(
        self,
        request: ApprovalRequest,
    ) -> bool:
        """Check approval status with timeout handling."""
        # Check for explicit approval
        approvals = [r for r in request.responses if r.decision == "approved"]
        if approvals:
            return True

        # Check for explicit rejection
        rejections = [r for r in request.responses if r.decision == "rejected"]
        if rejections:
            return False

        # Check timeout
        if request.is_expired():
            return self.auto_approve

        return False

    async def can_approve(
        self,
        request: ApprovalRequest,
        approver: str,
    ) -> bool:
        """Check if request is still pending."""
        return request.status == ApprovalStatus.PENDING


class HierarchicalGate(ApprovalGate):
    """
    Hierarchical approval gate.

    Requires approval from users at different hierarchy levels.
    """

    def __init__(
        self,
        levels: List[List[str]] = None,
    ):
        # levels is a list of approver groups, e.g.:
        # [["manager1", "manager2"], ["director"], ["vp"]]
        self.levels = levels or []

    async def check(
        self,
        request: ApprovalRequest,
    ) -> bool:
        """Check if all levels have approved."""
        approvals = {r.approver for r in request.responses if r.decision == "approved"}

        for level in self.levels:
            # At least one approver from each level must approve
            if not any(approver in approvals for approver in level):
                return False

        return True

    async def can_approve(
        self,
        request: ApprovalRequest,
        approver: str,
    ) -> bool:
        """Check if approver is in any level and current level allows approval."""
        # Find which level the approver is in
        approver_level = None
        for i, level in enumerate(self.levels):
            if approver in level:
                approver_level = i
                break

        if approver_level is None:
            return False

        # Check if previous levels are approved
        approvals = {r.approver for r in request.responses if r.decision == "approved"}

        for i in range(approver_level):
            if not any(a in approvals for a in self.levels[i]):
                return False  # Previous level not yet approved

        return True


class ConditionalGate(ApprovalGate):
    """
    Conditional approval gate.

    Approval requirements depend on conditions.
    """

    def __init__(
        self,
        condition_evaluator: Any = None,
        high_risk_approvers: List[str] = None,
        low_risk_approvers: List[str] = None,
    ):
        self.condition_evaluator = condition_evaluator
        self.high_risk_approvers = high_risk_approvers or []
        self.low_risk_approvers = low_risk_approvers or []

    async def check(
        self,
        request: ApprovalRequest,
    ) -> bool:
        """Check approval based on risk level."""
        risk_level = request.details.get("risk_level", "low")
        approvals = [r for r in request.responses if r.decision == "approved"]

        if risk_level == "high":
            # Require high-risk approver
            return any(
                r.approver in self.high_risk_approvers
                for r in approvals
            )
        else:
            # Any approver works for low risk
            return len(approvals) > 0

    async def can_approve(
        self,
        request: ApprovalRequest,
        approver: str,
    ) -> bool:
        """Check if approver matches risk level."""
        risk_level = request.details.get("risk_level", "low")

        if risk_level == "high":
            return approver in self.high_risk_approvers
        else:
            return approver in (self.low_risk_approvers + self.high_risk_approvers)


class QuorumGate(ApprovalGate):
    """
    Quorum-based approval gate.

    Requires a percentage of approvers to approve.
    """

    def __init__(
        self,
        quorum_percentage: float = 0.5,  # 50%
    ):
        self.quorum_percentage = quorum_percentage

    async def check(
        self,
        request: ApprovalRequest,
    ) -> bool:
        """Check if quorum is reached."""
        if not request.approvers:
            return True

        approvals = len([r for r in request.responses if r.decision == "approved"])
        required = int(len(request.approvers) * self.quorum_percentage)

        # At least 1 approval required
        required = max(1, required)

        return approvals >= required

    async def can_approve(
        self,
        request: ApprovalRequest,
        approver: str,
    ) -> bool:
        """Check if user is in approvers list."""
        if not request.approvers:
            return True
        return approver in request.approvers


# === Gate Factory ===


def create_gate(
    gate_type: str,
    config: Dict[str, Any] = None,
) -> ApprovalGate:
    """
    Create an approval gate by type.

    Args:
        gate_type: Type of gate (single, multi, timeout, hierarchical, quorum)
        config: Gate configuration

    Returns:
        ApprovalGate instance
    """
    config = config or {}

    gates = {
        "single": lambda: SingleApproverGate(),
        "multi": lambda: MultiApproverGate(
            min_approvals=config.get("min_approvals", 2),
            require_all=config.get("require_all", False),
        ),
        "timeout": lambda: TimeoutGate(
            timeout_hours=config.get("timeout_hours", 24.0),
            auto_approve=config.get("auto_approve", False),
        ),
        "hierarchical": lambda: HierarchicalGate(
            levels=config.get("levels", []),
        ),
        "quorum": lambda: QuorumGate(
            quorum_percentage=config.get("quorum_percentage", 0.5),
        ),
        "conditional": lambda: ConditionalGate(
            high_risk_approvers=config.get("high_risk_approvers", []),
            low_risk_approvers=config.get("low_risk_approvers", []),
        ),
    }

    factory = gates.get(gate_type.lower())
    if not factory:
        raise ValueError(f"Unknown gate type: {gate_type}")

    return factory()
