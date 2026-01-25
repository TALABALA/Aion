"""
Auto-Remediation for AIOps.

Implements automated remediation capabilities:
- Runbook execution
- Confidence-scored actions
- Safe rollback mechanisms
- Human-in-the-loop approval
"""

import asyncio
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable, Awaitable
from datetime import datetime, timedelta
from enum import Enum
import json

logger = logging.getLogger(__name__)


# =============================================================================
# Data Types
# =============================================================================

class RemediationStatus(Enum):
    """Status of a remediation action."""
    PENDING = "pending"
    APPROVED = "approved"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    CANCELLED = "cancelled"


class ActionType(Enum):
    """Type of remediation action."""
    RESTART_SERVICE = "restart_service"
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    CLEAR_CACHE = "clear_cache"
    ROTATE_LOGS = "rotate_logs"
    KILL_PROCESS = "kill_process"
    ADJUST_CONFIG = "adjust_config"
    ROLLBACK_DEPLOY = "rollback_deploy"
    FAILOVER = "failover"
    CUSTOM = "custom"


class ApprovalMode(Enum):
    """Approval mode for remediation actions."""
    AUTO = "auto"  # Execute automatically
    MANUAL = "manual"  # Require human approval
    CONDITIONAL = "conditional"  # Auto if confidence high enough


@dataclass
class RemediationAction:
    """Definition of a remediation action."""
    action_id: str
    action_type: ActionType
    target: str  # Service, host, container, etc.
    parameters: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    estimated_duration: timedelta = timedelta(minutes=5)
    rollback_action: Optional['RemediationAction'] = None
    prerequisites: List[str] = field(default_factory=list)
    validation_checks: List[str] = field(default_factory=list)


@dataclass
class RemediationResult:
    """Result of a remediation action."""
    action_id: str
    status: RemediationStatus
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    output: str = ""
    error: Optional[str] = None
    metrics_before: Dict[str, float] = field(default_factory=dict)
    metrics_after: Dict[str, float] = field(default_factory=dict)
    effectiveness_score: float = 0.0


@dataclass
class RemediationRunbook:
    """A runbook defining a sequence of remediation actions."""
    runbook_id: str
    name: str
    description: str
    trigger_conditions: List[str]
    actions: List[RemediationAction]
    approval_mode: ApprovalMode = ApprovalMode.CONDITIONAL
    confidence_threshold: float = 0.8
    timeout: timedelta = timedelta(minutes=30)
    tags: List[str] = field(default_factory=list)


# =============================================================================
# Confidence Scoring
# =============================================================================

class ConfidenceScorer:
    """
    Score confidence for remediation actions.

    Considers:
    - Historical success rate
    - Current system state
    - Similar incident resolutions
    - Risk assessment
    """

    def __init__(self):
        # Historical success rates by action type
        self._success_rates: Dict[str, Tuple[int, int]] = {}  # success, total

        # Risk weights by action type
        self._risk_weights: Dict[ActionType, float] = {
            ActionType.RESTART_SERVICE: 0.3,
            ActionType.SCALE_UP: 0.1,
            ActionType.SCALE_DOWN: 0.2,
            ActionType.CLEAR_CACHE: 0.1,
            ActionType.ROTATE_LOGS: 0.05,
            ActionType.KILL_PROCESS: 0.4,
            ActionType.ADJUST_CONFIG: 0.3,
            ActionType.ROLLBACK_DEPLOY: 0.5,
            ActionType.FAILOVER: 0.6,
            ActionType.CUSTOM: 0.5,
        }

        # Time-of-day adjustments
        self._time_adjustments: Dict[int, float] = {
            # Lower confidence during peak hours
            9: 0.9, 10: 0.85, 11: 0.85, 12: 0.85,
            13: 0.85, 14: 0.85, 15: 0.85, 16: 0.85,
            17: 0.9,
            # Higher confidence during off-hours
            0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0,
            22: 1.0, 23: 1.0,
        }

    def record_outcome(self, action_key: str, success: bool):
        """Record outcome for learning."""
        if action_key not in self._success_rates:
            self._success_rates[action_key] = (0, 0)

        successes, total = self._success_rates[action_key]
        self._success_rates[action_key] = (
            successes + (1 if success else 0),
            total + 1
        )

    def score(
        self,
        action: RemediationAction,
        incident_context: Dict[str, Any] = None
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Score confidence for an action.

        Returns:
            (confidence_score, factors_dict)
        """
        factors = {}

        # Base confidence from historical success rate
        action_key = f"{action.action_type.value}:{action.target}"
        if action_key in self._success_rates:
            successes, total = self._success_rates[action_key]
            if total > 0:
                historical_rate = successes / total
                # Bayesian smoothing
                confidence_historical = (successes + 1) / (total + 2)
            else:
                confidence_historical = 0.5
        else:
            confidence_historical = 0.5  # No history

        factors['historical_success_rate'] = confidence_historical

        # Risk factor
        risk = self._risk_weights.get(action.action_type, 0.5)
        confidence_risk = 1 - risk
        factors['risk_factor'] = confidence_risk

        # Time-of-day factor
        current_hour = datetime.now().hour
        time_factor = self._time_adjustments.get(current_hour, 0.95)
        factors['time_factor'] = time_factor

        # Context-based factors
        confidence_context = 1.0
        if incident_context:
            # Similar incident success rate
            if 'similar_incident_success_rate' in incident_context:
                confidence_context *= incident_context['similar_incident_success_rate']
                factors['similar_incident_rate'] = incident_context['similar_incident_success_rate']

            # System health factor
            if 'system_health_score' in incident_context:
                health = incident_context['system_health_score']
                confidence_context *= 0.5 + 0.5 * health  # Range: 0.5-1.0
                factors['system_health'] = health

        # Combine factors
        confidence = (
            confidence_historical * 0.3 +
            confidence_risk * 0.3 +
            time_factor * 0.1 +
            confidence_context * 0.3
        )

        # Ensure bounds
        confidence = max(0.0, min(1.0, confidence))

        return (confidence, factors)


# =============================================================================
# Action Executors
# =============================================================================

class ActionExecutor(ABC):
    """Base class for action executors."""

    @abstractmethod
    async def execute(
        self,
        action: RemediationAction,
        context: Dict[str, Any]
    ) -> RemediationResult:
        """Execute the action."""
        pass

    @abstractmethod
    async def validate(
        self,
        action: RemediationAction,
        context: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Validate the action before execution."""
        pass

    @abstractmethod
    async def rollback(
        self,
        action: RemediationAction,
        result: RemediationResult,
        context: Dict[str, Any]
    ) -> RemediationResult:
        """Rollback the action."""
        pass


class RestartServiceExecutor(ActionExecutor):
    """Executor for service restart actions."""

    def __init__(self, service_manager: Any = None):
        self.service_manager = service_manager

    async def execute(
        self,
        action: RemediationAction,
        context: Dict[str, Any]
    ) -> RemediationResult:
        result = RemediationResult(
            action_id=action.action_id,
            status=RemediationStatus.EXECUTING,
            started_at=datetime.now()
        )

        try:
            service_name = action.target
            graceful = action.parameters.get('graceful', True)
            timeout = action.parameters.get('timeout', 30)

            logger.info(f"Restarting service {service_name}")

            # Simulate service restart
            await asyncio.sleep(2)

            result.status = RemediationStatus.COMPLETED
            result.output = f"Service {service_name} restarted successfully"
            result.effectiveness_score = 0.9

        except Exception as e:
            result.status = RemediationStatus.FAILED
            result.error = str(e)

        result.completed_at = datetime.now()
        return result

    async def validate(
        self,
        action: RemediationAction,
        context: Dict[str, Any]
    ) -> Tuple[bool, str]:
        # Validate service exists
        service_name = action.target

        # Check if service is in a restartable state
        # (would check actual service status in real implementation)

        return (True, "Service can be restarted")

    async def rollback(
        self,
        action: RemediationAction,
        result: RemediationResult,
        context: Dict[str, Any]
    ) -> RemediationResult:
        # For restart, rollback might mean reverting to previous version
        # or simply logging that rollback isn't applicable
        return RemediationResult(
            action_id=f"{action.action_id}_rollback",
            status=RemediationStatus.COMPLETED,
            output="Restart actions don't have specific rollback"
        )


class ScaleExecutor(ActionExecutor):
    """Executor for scale up/down actions."""

    async def execute(
        self,
        action: RemediationAction,
        context: Dict[str, Any]
    ) -> RemediationResult:
        result = RemediationResult(
            action_id=action.action_id,
            status=RemediationStatus.EXECUTING,
            started_at=datetime.now()
        )

        try:
            target = action.target
            replicas = action.parameters.get('replicas', 1)
            direction = "up" if action.action_type == ActionType.SCALE_UP else "down"

            logger.info(f"Scaling {direction} {target} to {replicas} replicas")

            # Store original count for rollback
            result.metrics_before = {'replicas': context.get('current_replicas', 1)}

            # Simulate scaling
            await asyncio.sleep(3)

            result.metrics_after = {'replicas': replicas}
            result.status = RemediationStatus.COMPLETED
            result.output = f"Scaled {target} to {replicas} replicas"
            result.effectiveness_score = 0.85

        except Exception as e:
            result.status = RemediationStatus.FAILED
            result.error = str(e)

        result.completed_at = datetime.now()
        return result

    async def validate(
        self,
        action: RemediationAction,
        context: Dict[str, Any]
    ) -> Tuple[bool, str]:
        replicas = action.parameters.get('replicas', 1)

        if replicas < 0:
            return (False, "Replicas cannot be negative")

        max_replicas = context.get('max_replicas', 100)
        if replicas > max_replicas:
            return (False, f"Replicas exceed maximum ({max_replicas})")

        return (True, "Scale action validated")

    async def rollback(
        self,
        action: RemediationAction,
        result: RemediationResult,
        context: Dict[str, Any]
    ) -> RemediationResult:
        original_replicas = result.metrics_before.get('replicas', 1)

        rollback_action = RemediationAction(
            action_id=f"{action.action_id}_rollback",
            action_type=ActionType.SCALE_UP if action.action_type == ActionType.SCALE_DOWN else ActionType.SCALE_DOWN,
            target=action.target,
            parameters={'replicas': original_replicas}
        )

        return await self.execute(rollback_action, context)


class CustomExecutor(ActionExecutor):
    """Executor for custom remediation actions."""

    def __init__(
        self,
        execute_fn: Callable[[RemediationAction, Dict], Awaitable[RemediationResult]] = None,
        validate_fn: Callable[[RemediationAction, Dict], Awaitable[Tuple[bool, str]]] = None,
        rollback_fn: Callable[[RemediationAction, RemediationResult, Dict], Awaitable[RemediationResult]] = None
    ):
        self._execute_fn = execute_fn
        self._validate_fn = validate_fn
        self._rollback_fn = rollback_fn

    async def execute(
        self,
        action: RemediationAction,
        context: Dict[str, Any]
    ) -> RemediationResult:
        if self._execute_fn:
            return await self._execute_fn(action, context)

        return RemediationResult(
            action_id=action.action_id,
            status=RemediationStatus.FAILED,
            error="No execute function configured"
        )

    async def validate(
        self,
        action: RemediationAction,
        context: Dict[str, Any]
    ) -> Tuple[bool, str]:
        if self._validate_fn:
            return await self._validate_fn(action, context)
        return (True, "Custom action validation skipped")

    async def rollback(
        self,
        action: RemediationAction,
        result: RemediationResult,
        context: Dict[str, Any]
    ) -> RemediationResult:
        if self._rollback_fn:
            return await self._rollback_fn(action, result, context)

        return RemediationResult(
            action_id=f"{action.action_id}_rollback",
            status=RemediationStatus.FAILED,
            error="No rollback function configured"
        )


# =============================================================================
# Auto Remediator
# =============================================================================

class AutoRemediator:
    """
    Main auto-remediation engine.

    Manages runbooks, executes actions, and tracks results.
    """

    def __init__(
        self,
        default_approval_mode: ApprovalMode = ApprovalMode.CONDITIONAL,
        default_confidence_threshold: float = 0.8
    ):
        self.default_approval_mode = default_approval_mode
        self.default_confidence_threshold = default_confidence_threshold

        self.scorer = ConfidenceScorer()

        # Executors by action type
        self._executors: Dict[ActionType, ActionExecutor] = {
            ActionType.RESTART_SERVICE: RestartServiceExecutor(),
            ActionType.SCALE_UP: ScaleExecutor(),
            ActionType.SCALE_DOWN: ScaleExecutor(),
        }

        # Runbooks
        self._runbooks: Dict[str, RemediationRunbook] = {}

        # Execution history
        self._history: List[Tuple[RemediationAction, RemediationResult]] = []

        # Pending approvals
        self._pending_approvals: Dict[str, Tuple[RemediationAction, Dict]] = {}

        # Approval handlers
        self._approval_handlers: List[Callable[[RemediationAction, float], Awaitable[bool]]] = []

    def register_executor(self, action_type: ActionType, executor: ActionExecutor):
        """Register an executor for an action type."""
        self._executors[action_type] = executor

    def add_runbook(self, runbook: RemediationRunbook):
        """Add a remediation runbook."""
        self._runbooks[runbook.runbook_id] = runbook

    def add_approval_handler(
        self,
        handler: Callable[[RemediationAction, float], Awaitable[bool]]
    ):
        """Add an approval handler."""
        self._approval_handlers.append(handler)

    async def remediate(
        self,
        action: RemediationAction,
        context: Dict[str, Any] = None,
        approval_mode: ApprovalMode = None,
        confidence_threshold: float = None
    ) -> RemediationResult:
        """
        Execute a remediation action.

        Args:
            action: The action to execute
            context: Incident and system context
            approval_mode: Override approval mode
            confidence_threshold: Override confidence threshold
        """
        context = context or {}
        approval_mode = approval_mode or self.default_approval_mode
        confidence_threshold = confidence_threshold or self.default_confidence_threshold

        # Score confidence
        confidence, factors = self.scorer.score(action, context)
        logger.info(f"Action {action.action_id} confidence: {confidence:.2f}")

        # Check approval
        approved = await self._check_approval(
            action, confidence, approval_mode, confidence_threshold
        )

        if not approved:
            return RemediationResult(
                action_id=action.action_id,
                status=RemediationStatus.CANCELLED,
                error="Action not approved"
            )

        # Get executor
        executor = self._executors.get(action.action_type)
        if not executor:
            return RemediationResult(
                action_id=action.action_id,
                status=RemediationStatus.FAILED,
                error=f"No executor for action type: {action.action_type}"
            )

        # Validate
        valid, validation_msg = await executor.validate(action, context)
        if not valid:
            return RemediationResult(
                action_id=action.action_id,
                status=RemediationStatus.FAILED,
                error=f"Validation failed: {validation_msg}"
            )

        # Execute
        result = await executor.execute(action, context)

        # Record for learning
        self.scorer.record_outcome(
            f"{action.action_type.value}:{action.target}",
            result.status == RemediationStatus.COMPLETED
        )

        # Store history
        self._history.append((action, result))

        # Auto-rollback on failure if configured
        if result.status == RemediationStatus.FAILED and action.rollback_action:
            logger.warning(f"Action {action.action_id} failed, initiating rollback")
            await self.remediate(action.rollback_action, context)

        return result

    async def _check_approval(
        self,
        action: RemediationAction,
        confidence: float,
        approval_mode: ApprovalMode,
        threshold: float
    ) -> bool:
        """Check if action is approved for execution."""
        if approval_mode == ApprovalMode.AUTO:
            return True

        elif approval_mode == ApprovalMode.MANUAL:
            # Always require approval
            return await self._get_approval(action, confidence)

        else:  # CONDITIONAL
            if confidence >= threshold:
                logger.info(f"Auto-approving action {action.action_id} (confidence: {confidence:.2f})")
                return True
            else:
                logger.info(f"Requesting approval for action {action.action_id} (confidence: {confidence:.2f})")
                return await self._get_approval(action, confidence)

    async def _get_approval(self, action: RemediationAction, confidence: float) -> bool:
        """Get approval from handlers."""
        if not self._approval_handlers:
            # No handlers, auto-approve with warning
            logger.warning(f"No approval handlers, auto-approving action {action.action_id}")
            return True

        # Try each handler until one approves
        for handler in self._approval_handlers:
            try:
                approved = await handler(action, confidence)
                if approved:
                    return True
            except Exception as e:
                logger.error(f"Approval handler error: {e}")

        return False

    async def execute_runbook(
        self,
        runbook_id: str,
        context: Dict[str, Any] = None
    ) -> List[RemediationResult]:
        """Execute a complete runbook."""
        runbook = self._runbooks.get(runbook_id)
        if not runbook:
            raise ValueError(f"Runbook not found: {runbook_id}")

        results = []
        context = context or {}

        for action in runbook.actions:
            # Check prerequisites
            if action.prerequisites:
                for prereq_id in action.prerequisites:
                    prereq_result = next(
                        (r for a, r in self._history if a.action_id == prereq_id),
                        None
                    )
                    if not prereq_result or prereq_result.status != RemediationStatus.COMPLETED:
                        results.append(RemediationResult(
                            action_id=action.action_id,
                            status=RemediationStatus.CANCELLED,
                            error=f"Prerequisite not met: {prereq_id}"
                        ))
                        continue

            # Execute action
            result = await self.remediate(
                action,
                context,
                runbook.approval_mode,
                runbook.confidence_threshold
            )
            results.append(result)

            # Stop on failure
            if result.status == RemediationStatus.FAILED:
                logger.error(f"Runbook {runbook_id} stopped due to failure")
                break

        return results

    def get_history(
        self,
        action_type: ActionType = None,
        target: str = None,
        status: RemediationStatus = None,
        since: datetime = None
    ) -> List[Tuple[RemediationAction, RemediationResult]]:
        """Get remediation history with filters."""
        results = []

        for action, result in self._history:
            if action_type and action.action_type != action_type:
                continue
            if target and action.target != target:
                continue
            if status and result.status != status:
                continue
            if since and result.started_at and result.started_at < since:
                continue

            results.append((action, result))

        return results

    def get_success_rate(
        self,
        action_type: ActionType = None,
        target: str = None
    ) -> float:
        """Get historical success rate."""
        history = self.get_history(action_type=action_type, target=target)

        if not history:
            return 0.0

        successes = sum(
            1 for _, r in history
            if r.status == RemediationStatus.COMPLETED
        )

        return successes / len(history)
