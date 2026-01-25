"""
AION Workflow Approvals

Human-in-the-loop approval gates:
- Approval request creation
- Multi-approver support
- Timeout handling
- Approval/rejection flow
"""

from aion.automation.approval.manager import ApprovalManager
from aion.automation.approval.gates import (
    ApprovalGate,
    SingleApproverGate,
    MultiApproverGate,
    TimeoutGate,
)

__all__ = [
    "ApprovalManager",
    "ApprovalGate",
    "SingleApproverGate",
    "MultiApproverGate",
    "TimeoutGate",
]
