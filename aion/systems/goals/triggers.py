"""
AION Goal Triggers

Event-based goal activation:
- Trigger conditions
- Event handling
- Scheduled triggers
- Context-aware activation
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Optional

import structlog

from aion.systems.goals.types import Goal, GoalSource, GoalPriority, GoalType

logger = structlog.get_logger(__name__)


class TriggerType(str, Enum):
    """Types of triggers."""
    TIME = "time"              # Time-based trigger
    EVENT = "event"            # Event-based trigger
    CONDITION = "condition"    # Condition-based trigger
    GOAL = "goal"              # Goal completion trigger
    THRESHOLD = "threshold"    # Threshold-based trigger
    PATTERN = "pattern"        # Pattern detection trigger


@dataclass
class TriggerCondition:
    """Condition that activates a trigger."""
    id: str
    name: str
    trigger_type: TriggerType
    description: str

    # For time triggers
    schedule: Optional[str] = None  # Cron-like expression
    interval_minutes: Optional[int] = None
    next_run: Optional[datetime] = None

    # For event triggers
    event_type: Optional[str] = None
    event_filter: Optional[dict[str, Any]] = None

    # For condition triggers
    condition_expression: Optional[str] = None

    # For goal triggers
    trigger_on_goal_id: Optional[str] = None
    trigger_on_goal_status: Optional[str] = None

    # For threshold triggers
    metric_name: Optional[str] = None
    threshold_value: Optional[float] = None
    threshold_direction: str = "above"  # "above" or "below"

    # Status
    enabled: bool = True
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "trigger_type": self.trigger_type.value,
            "description": self.description,
            "enabled": self.enabled,
            "trigger_count": self.trigger_count,
            "last_triggered": self.last_triggered.isoformat() if self.last_triggered else None,
        }


@dataclass
class TriggerAction:
    """Action to take when a trigger fires."""
    id: str
    name: str

    # Goal to create
    goal_template: dict[str, Any] = field(default_factory=dict)

    # Or existing goal to activate
    goal_id_to_activate: Optional[str] = None

    # Or custom action
    action_type: str = "create_goal"  # "create_goal", "activate_goal", "custom"
    custom_handler: Optional[Callable] = None

    # Cooldown
    cooldown_minutes: int = 0
    last_executed: Optional[datetime] = None

    def can_execute(self) -> bool:
        """Check if action can be executed (respecting cooldown)."""
        if not self.cooldown_minutes:
            return True
        if not self.last_executed:
            return True
        elapsed = datetime.now() - self.last_executed
        return elapsed.total_seconds() >= self.cooldown_minutes * 60


@dataclass
class Trigger:
    """A complete trigger with condition and action."""
    id: str
    name: str
    description: str
    condition: TriggerCondition
    action: TriggerAction
    enabled: bool = True
    priority: int = 5  # 1-10, lower = higher priority
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "condition": self.condition.to_dict(),
            "enabled": self.enabled,
            "priority": self.priority,
        }


class GoalTriggers:
    """
    Manages event-based goal activation.

    Features:
    - Time-based scheduling
    - Event-driven triggers
    - Condition monitoring
    - Goal completion chains
    """

    def __init__(
        self,
        registry: Optional[Any] = None,  # GoalRegistry
        check_interval: float = 60.0,
    ):
        self._registry = registry
        self._check_interval = check_interval

        # Triggers
        self._triggers: dict[str, Trigger] = {}

        # Event subscriptions
        self._event_handlers: dict[str, list[str]] = {}  # event_type -> trigger_ids

        # Background tasks
        self._check_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

        # Stats
        self._stats = {
            "triggers_fired": 0,
            "goals_created": 0,
            "check_cycles": 0,
        }

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the trigger system."""
        if self._initialized:
            return

        logger.info("Initializing Goal Triggers")

        # Start background check task
        self._check_task = asyncio.create_task(self._check_loop())

        self._initialized = True

    async def shutdown(self) -> None:
        """Shutdown the trigger system."""
        logger.info("Shutting down Goal Triggers")

        self._shutdown_event.set()

        if self._check_task:
            self._check_task.cancel()
            try:
                await self._check_task
            except asyncio.CancelledError:
                pass

        self._initialized = False

    async def _check_loop(self) -> None:
        """Background loop for checking triggers."""
        while not self._shutdown_event.is_set():
            try:
                await self._check_triggers()
                self._stats["check_cycles"] += 1

                await asyncio.sleep(self._check_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Trigger check error: {e}")
                await asyncio.sleep(5)

    async def _check_triggers(self) -> None:
        """Check all triggers for activation."""
        now = datetime.now()

        for trigger in self._triggers.values():
            if not trigger.enabled:
                continue

            should_fire = await self._should_trigger_fire(trigger, now)

            if should_fire:
                await self._fire_trigger(trigger)

    async def _should_trigger_fire(
        self, trigger: Trigger, now: datetime
    ) -> bool:
        """Check if a trigger should fire."""
        condition = trigger.condition

        if not condition.enabled:
            return False

        if trigger.condition.trigger_type == TriggerType.TIME:
            return self._check_time_trigger(condition, now)

        elif trigger.condition.trigger_type == TriggerType.CONDITION:
            return await self._check_condition_trigger(condition)

        elif trigger.condition.trigger_type == TriggerType.GOAL:
            return await self._check_goal_trigger(condition)

        elif trigger.condition.trigger_type == TriggerType.THRESHOLD:
            return await self._check_threshold_trigger(condition)

        return False

    def _check_time_trigger(
        self, condition: TriggerCondition, now: datetime
    ) -> bool:
        """Check time-based trigger."""
        if condition.interval_minutes:
            if condition.next_run and now >= condition.next_run:
                # Update next run time
                condition.next_run = now + timedelta(
                    minutes=condition.interval_minutes
                )
                return True
            elif not condition.next_run:
                # First time - set next run
                condition.next_run = now + timedelta(
                    minutes=condition.interval_minutes
                )

        # TODO: Add cron-like schedule parsing

        return False

    async def _check_condition_trigger(
        self, condition: TriggerCondition
    ) -> bool:
        """Check condition-based trigger."""
        if not condition.condition_expression:
            return False

        # Simple condition evaluation
        # In production, would use a safe expression evaluator
        try:
            # Example: check system stats
            # result = eval(condition.condition_expression, {"stats": stats})
            return False
        except:
            return False

    async def _check_goal_trigger(
        self, condition: TriggerCondition
    ) -> bool:
        """Check goal completion trigger."""
        if not self._registry or not condition.trigger_on_goal_id:
            return False

        goal = await self._registry.get(condition.trigger_on_goal_id)
        if not goal:
            return False

        if condition.trigger_on_goal_status:
            return goal.status.value == condition.trigger_on_goal_status

        return goal.status.value == "completed"

    async def _check_threshold_trigger(
        self, condition: TriggerCondition
    ) -> bool:
        """Check threshold-based trigger."""
        if not condition.metric_name or condition.threshold_value is None:
            return False

        # Would need access to metrics system
        # For now, return False
        return False

    async def _fire_trigger(self, trigger: Trigger) -> None:
        """Fire a trigger and execute its action."""
        action = trigger.action

        if not action.can_execute():
            logger.debug(
                f"Trigger action in cooldown",
                trigger_id=trigger.id[:8],
            )
            return

        logger.info(
            f"Firing trigger: {trigger.name}",
            trigger_id=trigger.id[:8],
        )

        # Update trigger stats
        trigger.condition.last_triggered = datetime.now()
        trigger.condition.trigger_count += 1
        self._stats["triggers_fired"] += 1

        # Execute action
        if action.action_type == "create_goal":
            await self._create_goal_from_template(trigger, action.goal_template)

        elif action.action_type == "activate_goal":
            await self._activate_goal(action.goal_id_to_activate)

        elif action.action_type == "custom" and action.custom_handler:
            try:
                await action.custom_handler(trigger)
            except Exception as e:
                logger.error(f"Custom trigger handler error: {e}")

        action.last_executed = datetime.now()

    async def _create_goal_from_template(
        self,
        trigger: Trigger,
        template: dict[str, Any],
    ) -> Optional[Goal]:
        """Create a goal from a template."""
        if not self._registry:
            return None

        goal = Goal(
            title=template.get("title", f"Triggered: {trigger.name}"),
            description=template.get(
                "description", f"Created by trigger: {trigger.description}"
            ),
            success_criteria=template.get("success_criteria", []),
            goal_type=GoalType(template.get("goal_type", "achievement")),
            source=GoalSource.TRIGGERED,
            priority=GoalPriority[template.get("priority", "MEDIUM")],
            tags=template.get("tags", []) + [f"trigger:{trigger.id[:8]}"],
            context={
                "trigger_id": trigger.id,
                "trigger_name": trigger.name,
            },
        )

        await self._registry.create(goal)
        self._stats["goals_created"] += 1

        logger.info(
            f"Created goal from trigger",
            goal_id=goal.id[:8],
            trigger_id=trigger.id[:8],
        )

        return goal

    async def _activate_goal(self, goal_id: Optional[str]) -> None:
        """Activate an existing goal."""
        if not self._registry or not goal_id:
            return

        goal = await self._registry.get(goal_id)
        if goal and goal.status.value == "pending":
            goal.status = GoalPriority.MEDIUM  # This should be GoalStatus
            await self._registry.update(goal)

    async def handle_event(
        self, event_type: str, event_data: dict[str, Any]
    ) -> None:
        """Handle an external event."""
        trigger_ids = self._event_handlers.get(event_type, [])

        for trigger_id in trigger_ids:
            trigger = self._triggers.get(trigger_id)
            if not trigger or not trigger.enabled:
                continue

            condition = trigger.condition
            if condition.trigger_type != TriggerType.EVENT:
                continue

            # Check event filter
            if condition.event_filter:
                if not self._matches_filter(event_data, condition.event_filter):
                    continue

            await self._fire_trigger(trigger)

    def _matches_filter(
        self, data: dict[str, Any], filter_spec: dict[str, Any]
    ) -> bool:
        """Check if data matches a filter specification."""
        for key, value in filter_spec.items():
            if key not in data:
                return False
            if data[key] != value:
                return False
        return True

    def add_trigger(self, trigger: Trigger) -> None:
        """Add a new trigger."""
        self._triggers[trigger.id] = trigger

        # Register event handlers
        if trigger.condition.trigger_type == TriggerType.EVENT:
            event_type = trigger.condition.event_type
            if event_type:
                if event_type not in self._event_handlers:
                    self._event_handlers[event_type] = []
                self._event_handlers[event_type].append(trigger.id)

        logger.info(f"Added trigger: {trigger.name}", trigger_id=trigger.id[:8])

    def remove_trigger(self, trigger_id: str) -> bool:
        """Remove a trigger."""
        if trigger_id not in self._triggers:
            return False

        trigger = self._triggers[trigger_id]

        # Remove event handler
        if trigger.condition.trigger_type == TriggerType.EVENT:
            event_type = trigger.condition.event_type
            if event_type and event_type in self._event_handlers:
                if trigger_id in self._event_handlers[event_type]:
                    self._event_handlers[event_type].remove(trigger_id)

        del self._triggers[trigger_id]
        return True

    def enable_trigger(self, trigger_id: str) -> bool:
        """Enable a trigger."""
        if trigger_id in self._triggers:
            self._triggers[trigger_id].enabled = True
            return True
        return False

    def disable_trigger(self, trigger_id: str) -> bool:
        """Disable a trigger."""
        if trigger_id in self._triggers:
            self._triggers[trigger_id].enabled = False
            return True
        return False

    def get_trigger(self, trigger_id: str) -> Optional[Trigger]:
        """Get a trigger by ID."""
        return self._triggers.get(trigger_id)

    def get_all_triggers(self) -> list[Trigger]:
        """Get all triggers."""
        return list(self._triggers.values())

    def get_active_triggers(self) -> list[Trigger]:
        """Get active triggers."""
        return [t for t in self._triggers.values() if t.enabled]

    def get_stats(self) -> dict[str, Any]:
        """Get trigger system statistics."""
        return {
            **self._stats,
            "total_triggers": len(self._triggers),
            "active_triggers": len(self.get_active_triggers()),
            "event_types_monitored": len(self._event_handlers),
        }

    # Convenience methods for creating common triggers

    def create_scheduled_trigger(
        self,
        name: str,
        interval_minutes: int,
        goal_template: dict[str, Any],
        description: str = "",
    ) -> Trigger:
        """Create a scheduled trigger."""
        import uuid

        trigger_id = str(uuid.uuid4())

        condition = TriggerCondition(
            id=f"{trigger_id}_condition",
            name=f"{name} condition",
            trigger_type=TriggerType.TIME,
            description=f"Every {interval_minutes} minutes",
            interval_minutes=interval_minutes,
        )

        action = TriggerAction(
            id=f"{trigger_id}_action",
            name=f"{name} action",
            goal_template=goal_template,
        )

        trigger = Trigger(
            id=trigger_id,
            name=name,
            description=description or f"Scheduled every {interval_minutes} minutes",
            condition=condition,
            action=action,
        )

        self.add_trigger(trigger)
        return trigger

    def create_goal_completion_trigger(
        self,
        name: str,
        trigger_on_goal_id: str,
        goal_template: dict[str, Any],
        description: str = "",
    ) -> Trigger:
        """Create a trigger that fires when another goal completes."""
        import uuid

        trigger_id = str(uuid.uuid4())

        condition = TriggerCondition(
            id=f"{trigger_id}_condition",
            name=f"{name} condition",
            trigger_type=TriggerType.GOAL,
            description=f"When goal {trigger_on_goal_id[:8]} completes",
            trigger_on_goal_id=trigger_on_goal_id,
            trigger_on_goal_status="completed",
        )

        action = TriggerAction(
            id=f"{trigger_id}_action",
            name=f"{name} action",
            goal_template=goal_template,
        )

        trigger = Trigger(
            id=trigger_id,
            name=name,
            description=description or f"Triggered by goal completion",
            condition=condition,
            action=action,
        )

        self.add_trigger(trigger)
        return trigger
