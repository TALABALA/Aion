"""
AION Saga Pattern Implementation

Implements the Saga pattern for distributed transactions with:
- Choreography-based sagas
- Orchestration-based sagas
- Compensation transactions
- Semantic lock management
"""

from aion.automation.saga.orchestrator import (
    SagaOrchestrator,
    SagaDefinition,
    SagaStep,
    SagaState,
    SagaStatus,
)
from aion.automation.saga.compensation import (
    CompensationManager,
    CompensationAction,
    CompensationResult,
)
from aion.automation.saga.semantic_lock import (
    SemanticLockManager,
    Lock,
    LockType,
)

__all__ = [
    "SagaOrchestrator",
    "SagaDefinition",
    "SagaStep",
    "SagaState",
    "SagaStatus",
    "CompensationManager",
    "CompensationAction",
    "CompensationResult",
    "SemanticLockManager",
    "Lock",
    "LockType",
]
