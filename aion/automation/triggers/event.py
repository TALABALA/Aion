"""
AION Event Trigger Handler

Internal event triggers.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import structlog

from aion.automation.types import Trigger, TriggerType, WorkflowEvent
from aion.automation.triggers.manager import BaseTriggerHandler

logger = structlog.get_logger(__name__)


class EventTriggerHandler(BaseTriggerHandler):
    """
    Handler for internal event triggers.

    Features:
    - Event type filtering
    - Source filtering
    - Pattern matching
    - Event history
    """

    def __init__(self, manager):
        super().__init__(manager)
        self._event_history: List[WorkflowEvent] = []
        self._max_history = 1000

    async def register(self, trigger: Trigger) -> None:
        """Register an event trigger."""
        config = trigger.config

        logger.info(
            "event_trigger_registered",
            trigger_id=trigger.id,
            event_type=config.event_type,
            event_source=config.event_source,
            has_filter=bool(config.event_filter),
        )

    async def unregister(self, trigger: Trigger) -> None:
        """Unregister an event trigger."""
        logger.info("event_trigger_unregistered", trigger_id=trigger.id)

    async def emit(
        self,
        event_type: str,
        data: Dict[str, Any],
        source: str = None,
        workflow_id: str = None,
        execution_id: str = None,
    ) -> List[str]:
        """
        Emit an event and trigger matching workflows.

        Args:
            event_type: Type of event
            data: Event data
            source: Event source identifier
            workflow_id: Optional workflow context
            execution_id: Optional execution context

        Returns:
            List of triggered execution IDs
        """
        event = WorkflowEvent(
            event_type=event_type,
            source=source or "system",
            data=data,
            workflow_id=workflow_id,
            execution_id=execution_id,
        )

        # Store in history
        self._event_history.append(event)
        if len(self._event_history) > self._max_history:
            self._event_history = self._event_history[-self._max_history:]

        # Trigger matching workflows
        execution_ids = await self.manager.handle_event(
            event_type=event_type,
            event_data=data,
            source=source,
        )

        logger.info(
            "event_emitted",
            event_type=event_type,
            source=source,
            triggered=len(execution_ids),
        )

        return execution_ids

    def get_event_history(
        self,
        event_type: str = None,
        source: str = None,
        limit: int = 100,
    ) -> List[WorkflowEvent]:
        """Get event history with optional filters."""
        events = self._event_history

        if event_type:
            events = [e for e in events if e.event_type == event_type]

        if source:
            events = [e for e in events if e.source == source]

        # Return most recent first
        return list(reversed(events[-limit:]))

    def clear_history(self) -> int:
        """Clear event history."""
        count = len(self._event_history)
        self._event_history.clear()
        return count


# === Standard Event Types ===


class StandardEvents:
    """Standard event type constants."""

    # Workflow events
    WORKFLOW_CREATED = "workflow.created"
    WORKFLOW_UPDATED = "workflow.updated"
    WORKFLOW_DELETED = "workflow.deleted"
    WORKFLOW_ACTIVATED = "workflow.activated"
    WORKFLOW_PAUSED = "workflow.paused"

    # Execution events
    EXECUTION_STARTED = "execution.started"
    EXECUTION_COMPLETED = "execution.completed"
    EXECUTION_FAILED = "execution.failed"
    EXECUTION_CANCELLED = "execution.cancelled"

    # Step events
    STEP_STARTED = "step.started"
    STEP_COMPLETED = "step.completed"
    STEP_FAILED = "step.failed"
    STEP_SKIPPED = "step.skipped"

    # Approval events
    APPROVAL_REQUESTED = "approval.requested"
    APPROVAL_APPROVED = "approval.approved"
    APPROVAL_REJECTED = "approval.rejected"
    APPROVAL_EXPIRED = "approval.expired"

    # Agent events
    AGENT_SPAWNED = "agent.spawned"
    AGENT_TERMINATED = "agent.terminated"
    AGENT_MESSAGE = "agent.message"

    # Goal events
    GOAL_CREATED = "goal.created"
    GOAL_COMPLETED = "goal.completed"
    GOAL_FAILED = "goal.failed"
    GOAL_PAUSED = "goal.paused"

    # System events
    SYSTEM_STARTED = "system.started"
    SYSTEM_SHUTDOWN = "system.shutdown"
    SYSTEM_ERROR = "system.error"
    SYSTEM_WARNING = "system.warning"

    # Alert events
    ALERT_FIRED = "alert.fired"
    ALERT_RESOLVED = "alert.resolved"
    ALERT_ACKNOWLEDGED = "alert.acknowledged"


def create_event(
    event_type: str,
    data: Dict[str, Any] = None,
    source: str = None,
) -> WorkflowEvent:
    """Create a workflow event."""
    return WorkflowEvent(
        event_type=event_type,
        source=source or "system",
        data=data or {},
    )
