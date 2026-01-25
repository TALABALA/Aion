"""
AION Trigger Manager

Manages workflow triggers of all types.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

import structlog

from aion.automation.types import (
    Trigger,
    TriggerConfig,
    TriggerType,
    WorkflowEvent,
)

if TYPE_CHECKING:
    from aion.automation.engine import WorkflowEngine

logger = structlog.get_logger(__name__)


class TriggerManager:
    """
    Manages workflow triggers.

    Features:
    - Schedule (cron) triggers
    - Webhook triggers
    - Event triggers
    - Data change triggers
    - Rate limiting
    """

    def __init__(self, engine: Optional["WorkflowEngine"] = None):
        self.engine = engine

        # Registered triggers by workflow
        self._triggers: Dict[str, List[Trigger]] = {}

        # Trigger instances by ID
        self._trigger_instances: Dict[str, Trigger] = {}

        # Webhook registry
        self._webhooks: Dict[str, Trigger] = {}  # path -> trigger

        # Event subscriptions
        self._event_subscriptions: Dict[str, List[Trigger]] = {}  # event_type -> triggers

        # Data change subscriptions
        self._data_subscriptions: Dict[str, List[Trigger]] = {}  # source -> triggers

        # Trigger handlers
        self._handlers: Dict[TriggerType, "BaseTriggerHandler"] = {}

        # Schedule task
        self._schedule_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

        # Event bus reference (for kernel integration)
        self._event_bus = None

        self._initialized = False

    def set_engine(self, engine: "WorkflowEngine") -> None:
        """Set the workflow engine (for deferred initialization)."""
        self.engine = engine

    def set_event_bus(self, event_bus: Any) -> None:
        """Set the event bus for kernel integration."""
        self._event_bus = event_bus

    async def initialize(self) -> None:
        """Initialize the trigger manager."""
        if self._initialized:
            return

        # Initialize trigger handlers
        from aion.automation.triggers.schedule import ScheduleTriggerHandler
        from aion.automation.triggers.webhook import WebhookTriggerHandler
        from aion.automation.triggers.event import EventTriggerHandler
        from aion.automation.triggers.data import DataChangeTriggerHandler
        from aion.automation.triggers.manual import ManualTriggerHandler

        self._handlers[TriggerType.SCHEDULE] = ScheduleTriggerHandler(self)
        self._handlers[TriggerType.WEBHOOK] = WebhookTriggerHandler(self)
        self._handlers[TriggerType.EVENT] = EventTriggerHandler(self)
        self._handlers[TriggerType.DATA_CHANGE] = DataChangeTriggerHandler(self)
        self._handlers[TriggerType.MANUAL] = ManualTriggerHandler(self)

        # Start schedule loop
        self._schedule_task = asyncio.create_task(self._schedule_loop())

        self._initialized = True
        logger.info("Trigger manager initialized")

    async def shutdown(self) -> None:
        """Shutdown the trigger manager."""
        self._shutdown_event.set()

        if self._schedule_task:
            self._schedule_task.cancel()
            try:
                await self._schedule_task
            except asyncio.CancelledError:
                pass

        self._initialized = False
        logger.info("Trigger manager shutdown")

    # === Trigger Registration ===

    async def register(
        self,
        workflow_id: str,
        config: TriggerConfig,
    ) -> Trigger:
        """Register a trigger for a workflow."""
        trigger = Trigger(
            workflow_id=workflow_id,
            config=config,
        )

        # Store by workflow
        if workflow_id not in self._triggers:
            self._triggers[workflow_id] = []
        self._triggers[workflow_id].append(trigger)

        # Store by ID
        self._trigger_instances[trigger.id] = trigger

        # Register with handler
        handler = self._handlers.get(config.trigger_type)
        if handler:
            await handler.register(trigger)

        # Type-specific registration
        if config.trigger_type == TriggerType.WEBHOOK:
            path = config.webhook_path or f"/webhooks/{trigger.id}"
            self._webhooks[path] = trigger
            logger.info("webhook_registered", path=path, trigger_id=trigger.id)

        elif config.trigger_type == TriggerType.EVENT:
            event_type = config.event_type or "*"
            if event_type not in self._event_subscriptions:
                self._event_subscriptions[event_type] = []
            self._event_subscriptions[event_type].append(trigger)
            logger.info("event_subscription_registered", event_type=event_type, trigger_id=trigger.id)

        elif config.trigger_type == TriggerType.DATA_CHANGE:
            source = config.data_source or "*"
            if source not in self._data_subscriptions:
                self._data_subscriptions[source] = []
            self._data_subscriptions[source].append(trigger)
            logger.info("data_subscription_registered", source=source, trigger_id=trigger.id)

        elif config.trigger_type == TriggerType.SCHEDULE:
            trigger.next_trigger_at = self._calculate_next_trigger(config)
            logger.info(
                "schedule_registered",
                cron=config.cron_expression,
                next_trigger=trigger.next_trigger_at,
            )

        logger.info(
            "trigger_registered",
            trigger_id=trigger.id,
            type=config.trigger_type.value,
            workflow_id=workflow_id,
        )

        return trigger

    async def unregister(self, trigger_id: str) -> bool:
        """Unregister a trigger."""
        trigger = self._trigger_instances.pop(trigger_id, None)
        if not trigger:
            return False

        # Remove from workflow list
        workflow_triggers = self._triggers.get(trigger.workflow_id, [])
        self._triggers[trigger.workflow_id] = [
            t for t in workflow_triggers if t.id != trigger_id
        ]

        # Remove from type-specific registries
        config = trigger.config

        if config.trigger_type == TriggerType.WEBHOOK:
            path = config.webhook_path or f"/webhooks/{trigger.id}"
            self._webhooks.pop(path, None)

        elif config.trigger_type == TriggerType.EVENT:
            event_type = config.event_type or "*"
            if event_type in self._event_subscriptions:
                self._event_subscriptions[event_type] = [
                    t for t in self._event_subscriptions[event_type]
                    if t.id != trigger_id
                ]

        elif config.trigger_type == TriggerType.DATA_CHANGE:
            source = config.data_source or "*"
            if source in self._data_subscriptions:
                self._data_subscriptions[source] = [
                    t for t in self._data_subscriptions[source]
                    if t.id != trigger_id
                ]

        # Unregister from handler
        handler = self._handlers.get(config.trigger_type)
        if handler:
            await handler.unregister(trigger)

        logger.info("trigger_unregistered", trigger_id=trigger_id)
        return True

    async def unregister_all(self, workflow_id: str) -> None:
        """Unregister all triggers for a workflow."""
        triggers = self._triggers.pop(workflow_id, [])

        for trigger in triggers:
            await self.unregister(trigger.id)

    async def get_trigger(self, trigger_id: str) -> Optional[Trigger]:
        """Get a trigger by ID."""
        return self._trigger_instances.get(trigger_id)

    async def list_triggers(
        self,
        workflow_id: Optional[str] = None,
        trigger_type: Optional[TriggerType] = None,
    ) -> List[Trigger]:
        """List triggers with optional filters."""
        if workflow_id:
            triggers = self._triggers.get(workflow_id, [])
        else:
            triggers = list(self._trigger_instances.values())

        if trigger_type:
            triggers = [t for t in triggers if t.config.trigger_type == trigger_type]

        return triggers

    # === Trigger Handlers ===

    async def handle_webhook(
        self,
        path: str,
        data: Dict[str, Any],
        headers: Dict[str, str] = None,
        method: str = "POST",
    ) -> Optional[str]:
        """
        Handle a webhook trigger.

        Returns:
            Execution ID if triggered, None otherwise
        """
        trigger = self._webhooks.get(path)
        if not trigger or not trigger.config.enabled:
            logger.debug("webhook_not_found", path=path)
            return None

        # Check method
        allowed_methods = trigger.config.webhook_methods or ["POST"]
        if method not in allowed_methods:
            logger.debug("webhook_method_not_allowed", path=path, method=method)
            return None

        # Verify secret if configured
        if trigger.config.webhook_secret:
            received_secret = (headers or {}).get("x-webhook-secret")
            if received_secret != trigger.config.webhook_secret:
                logger.warning("webhook_secret_mismatch", path=path)
                return None

        # Check rate limiting
        if not await self._check_rate_limit(trigger):
            logger.warning("webhook_rate_limited", path=path)
            return None

        # Execute workflow
        execution = await self.engine.execute(
            workflow_id=trigger.workflow_id,
            trigger_type=TriggerType.WEBHOOK,
            trigger_data={
                "body": data,
                "headers": headers or {},
                "path": path,
                "method": method,
            },
            trigger_id=trigger.id,
        )

        # Update trigger stats
        trigger.last_triggered_at = datetime.now()
        trigger.trigger_count += 1
        trigger.consecutive_failures = 0

        logger.info(
            "webhook_triggered",
            path=path,
            execution_id=execution.id,
        )

        return execution.id

    async def handle_event(
        self,
        event_type: str,
        event_data: Dict[str, Any],
        source: str = None,
    ) -> List[str]:
        """
        Handle an event trigger.

        Returns:
            List of execution IDs
        """
        execution_ids = []

        # Find matching triggers
        triggers = []
        triggers.extend(self._event_subscriptions.get(event_type, []))
        triggers.extend(self._event_subscriptions.get("*", []))  # Wildcard

        for trigger in triggers:
            if not trigger.config.enabled:
                continue

            # Check source filter
            if trigger.config.event_source and source != trigger.config.event_source:
                continue

            # Check event filter
            if trigger.config.event_filter:
                if not self._matches_filter(event_data, trigger.config.event_filter):
                    continue

            # Check rate limiting
            if not await self._check_rate_limit(trigger):
                continue

            # Execute workflow
            execution = await self.engine.execute(
                workflow_id=trigger.workflow_id,
                trigger_type=TriggerType.EVENT,
                trigger_data={
                    "event_type": event_type,
                    "data": event_data,
                    "source": source,
                },
                trigger_id=trigger.id,
            )

            execution_ids.append(execution.id)

            # Update trigger stats
            trigger.last_triggered_at = datetime.now()
            trigger.trigger_count += 1

        if execution_ids:
            logger.info(
                "event_triggered",
                event_type=event_type,
                executions=len(execution_ids),
            )

        return execution_ids

    async def handle_data_change(
        self,
        source: str,
        operation: str,
        data: Dict[str, Any],
    ) -> List[str]:
        """
        Handle a data change trigger.

        Returns:
            List of execution IDs
        """
        execution_ids = []

        # Find matching triggers
        triggers = []
        triggers.extend(self._data_subscriptions.get(source, []))
        triggers.extend(self._data_subscriptions.get("*", []))  # Wildcard

        for trigger in triggers:
            if not trigger.config.enabled:
                continue

            # Check operation filter
            if trigger.config.data_operation and trigger.config.data_operation != operation:
                continue

            # Check data filter
            if trigger.config.data_filter:
                if not self._matches_filter(data, trigger.config.data_filter):
                    continue

            # Check rate limiting
            if not await self._check_rate_limit(trigger):
                continue

            # Execute workflow
            execution = await self.engine.execute(
                workflow_id=trigger.workflow_id,
                trigger_type=TriggerType.DATA_CHANGE,
                trigger_data={
                    "source": source,
                    "operation": operation,
                    "data": data,
                },
                trigger_id=trigger.id,
            )

            execution_ids.append(execution.id)

            # Update trigger stats
            trigger.last_triggered_at = datetime.now()
            trigger.trigger_count += 1

        if execution_ids:
            logger.info(
                "data_change_triggered",
                source=source,
                operation=operation,
                executions=len(execution_ids),
            )

        return execution_ids

    async def handle_manual(
        self,
        workflow_id: str,
        inputs: Dict[str, Any] = None,
        initiated_by: str = None,
    ) -> Optional[str]:
        """
        Handle a manual trigger.

        Returns:
            Execution ID
        """
        execution = await self.engine.execute(
            workflow_id=workflow_id,
            trigger_type=TriggerType.MANUAL,
            trigger_data={
                "inputs": inputs or {},
                "initiated_by": initiated_by,
            },
            inputs=inputs,
            initiated_by=initiated_by,
        )

        logger.info(
            "manual_triggered",
            workflow_id=workflow_id,
            execution_id=execution.id,
            initiated_by=initiated_by,
        )

        return execution.id

    # === Helper Methods ===

    def _matches_filter(
        self,
        data: Dict[str, Any],
        filter_spec: Dict[str, Any],
    ) -> bool:
        """Check if data matches a filter specification."""
        for key, expected in filter_spec.items():
            # Support nested keys with dot notation
            parts = key.split(".")
            actual = data
            for part in parts:
                if isinstance(actual, dict):
                    actual = actual.get(part)
                else:
                    actual = None
                    break

            if actual != expected:
                return False
        return True

    async def _check_rate_limit(self, trigger: Trigger) -> bool:
        """Check if trigger is within rate limits."""
        config = trigger.config

        # Check minimum interval
        if config.min_interval_seconds > 0 and trigger.last_triggered_at:
            elapsed = (datetime.now() - trigger.last_triggered_at).total_seconds()
            if elapsed < config.min_interval_seconds:
                return False

        # Check max triggers per hour
        if config.max_triggers_per_hour:
            # Simple check: reset count every hour
            if trigger.last_triggered_at:
                hours_since = (datetime.now() - trigger.last_triggered_at).total_seconds() / 3600
                if hours_since < 1:
                    # Within the same hour
                    # Note: This is a simplified check. For production,
                    # use a sliding window or token bucket.
                    pass

        return True

    def _calculate_next_trigger(self, config: TriggerConfig) -> Optional[datetime]:
        """Calculate next trigger time from cron expression."""
        if not config.cron_expression:
            return None

        try:
            from croniter import croniter

            cron = croniter(config.cron_expression, datetime.now())
            return cron.get_next(datetime)
        except ImportError:
            logger.warning("croniter_not_installed")
            return None
        except Exception as e:
            logger.error("cron_parse_error", error=str(e))
            return None

    # === Schedule Loop ===

    async def _schedule_loop(self) -> None:
        """Background loop for schedule triggers."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(60)  # Check every minute

                now = datetime.now()

                for workflow_id, triggers in list(self._triggers.items()):
                    for trigger in triggers:
                        if trigger.config.trigger_type != TriggerType.SCHEDULE:
                            continue

                        if not trigger.config.enabled:
                            continue

                        if trigger.next_trigger_at and now >= trigger.next_trigger_at:
                            try:
                                # Execute workflow
                                execution = await self.engine.execute(
                                    workflow_id=workflow_id,
                                    trigger_type=TriggerType.SCHEDULE,
                                    trigger_data={
                                        "scheduled_time": trigger.next_trigger_at.isoformat(),
                                        "cron": trigger.config.cron_expression,
                                    },
                                    trigger_id=trigger.id,
                                )

                                # Update trigger stats
                                trigger.last_triggered_at = now
                                trigger.trigger_count += 1
                                trigger.consecutive_failures = 0

                                logger.info(
                                    "schedule_triggered",
                                    workflow_id=workflow_id,
                                    execution_id=execution.id,
                                )

                            except Exception as e:
                                trigger.consecutive_failures += 1
                                logger.error(
                                    "schedule_trigger_error",
                                    workflow_id=workflow_id,
                                    error=str(e),
                                )

                            # Calculate next trigger time
                            trigger.next_trigger_at = self._calculate_next_trigger(trigger.config)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("schedule_loop_error", error=str(e))

    # === Statistics ===

    def get_stats(self) -> Dict[str, Any]:
        """Get trigger manager statistics."""
        type_counts = {}
        for trigger_type in TriggerType:
            type_counts[trigger_type.value] = len([
                t for t in self._trigger_instances.values()
                if t.config.trigger_type == trigger_type
            ])

        return {
            "total_triggers": len(self._trigger_instances),
            "by_type": type_counts,
            "webhooks_registered": len(self._webhooks),
            "event_subscriptions": len(self._event_subscriptions),
            "data_subscriptions": len(self._data_subscriptions),
            "workflows_with_triggers": len(self._triggers),
        }


class BaseTriggerHandler:
    """Base class for trigger handlers."""

    def __init__(self, manager: TriggerManager):
        self.manager = manager

    async def register(self, trigger: Trigger) -> None:
        """Register a trigger."""
        pass

    async def unregister(self, trigger: Trigger) -> None:
        """Unregister a trigger."""
        pass

    async def handle(self, trigger: Trigger, data: Dict[str, Any]) -> Optional[str]:
        """Handle a trigger event."""
        pass
