"""
AION Goal Registry

Persistent storage and retrieval of goals:
- CRUD operations
- Query by status, priority, type
- Event history
- Statistics
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Any, Callable, Optional

import structlog

from aion.systems.goals.types import (
    Goal,
    GoalStatus,
    GoalPriority,
    GoalType,
    GoalSource,
    GoalEvent,
    Objective,
    GoalProgress,
)

logger = structlog.get_logger(__name__)


class GoalRegistry:
    """
    Registry for goal storage and retrieval.

    Features:
    - In-memory cache with optional persistence
    - Query by multiple criteria
    - Event history tracking
    - Statistics and analytics
    - Subscription for goal events
    """

    def __init__(
        self,
        persistence: Optional[Any] = None,  # GoalPersistence
        max_events_in_memory: int = 10000,
    ):
        self._persistence = persistence
        self._max_events = max_events_in_memory

        # In-memory storage
        self._goals: dict[str, Goal] = {}
        self._objectives: dict[str, Objective] = {}
        self._events: list[GoalEvent] = []

        # Indices for fast lookup
        self._status_index: dict[GoalStatus, set[str]] = {
            status: set() for status in GoalStatus
        }
        self._priority_index: dict[GoalPriority, set[str]] = {
            priority: set() for priority in GoalPriority
        }
        self._type_index: dict[GoalType, set[str]] = {
            goal_type: set() for goal_type in GoalType
        }
        self._tag_index: dict[str, set[str]] = {}
        self._parent_index: dict[str, set[str]] = {}  # parent_id -> child_ids

        # Event subscribers
        self._event_subscribers: list[Callable[[GoalEvent], None]] = []

        # Lock for thread safety
        self._lock = asyncio.Lock()

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the registry."""
        if self._initialized:
            return

        logger.info("Initializing Goal Registry")

        # Load from persistence if available
        if self._persistence:
            await self._load_from_persistence()

        self._initialized = True
        logger.info(
            "Goal Registry initialized",
            goals_count=len(self._goals),
            objectives_count=len(self._objectives),
        )

    async def shutdown(self) -> None:
        """Shutdown and persist."""
        logger.info("Shutting down Goal Registry")

        if self._persistence:
            await self._save_to_persistence()

        self._initialized = False

    # === Goal CRUD ===

    async def create(self, goal: Goal) -> str:
        """
        Create a new goal.

        Args:
            goal: The goal to create

        Returns:
            The goal ID
        """
        async with self._lock:
            if goal.id in self._goals:
                raise ValueError(f"Goal {goal.id} already exists")

            self._goals[goal.id] = goal
            self._update_indices(goal)

            # Record creation event
            event = GoalEvent(
                goal_id=goal.id,
                event_type="created",
                description=f"Goal created: {goal.title}",
                data={"goal": goal.to_dict()},
            )
            await self._add_event_internal(event)

            # Persist
            if self._persistence:
                await self._persistence.save_goal(goal)

        logger.info("Goal created", goal_id=goal.id[:8], title=goal.title)
        return goal.id

    async def get(self, goal_id: str) -> Optional[Goal]:
        """Get a goal by ID."""
        return self._goals.get(goal_id)

    async def update(self, goal: Goal) -> bool:
        """
        Update a goal.

        Args:
            goal: The goal with updated values

        Returns:
            True if updated, False if not found
        """
        async with self._lock:
            if goal.id not in self._goals:
                return False

            old_goal = self._goals[goal.id]

            # Update indices if status/priority/type changed
            if old_goal.status != goal.status:
                self._status_index[old_goal.status].discard(goal.id)
                self._status_index[goal.status].add(goal.id)

            if old_goal.priority != goal.priority:
                self._priority_index[old_goal.priority].discard(goal.id)
                self._priority_index[goal.priority].add(goal.id)

            if old_goal.goal_type != goal.goal_type:
                self._type_index[old_goal.goal_type].discard(goal.id)
                self._type_index[goal.goal_type].add(goal.id)

            # Update tags index
            old_tags = set(old_goal.tags)
            new_tags = set(goal.tags)
            for tag in old_tags - new_tags:
                if tag in self._tag_index:
                    self._tag_index[tag].discard(goal.id)
            for tag in new_tags - old_tags:
                if tag not in self._tag_index:
                    self._tag_index[tag] = set()
                self._tag_index[tag].add(goal.id)

            self._goals[goal.id] = goal

            # Persist
            if self._persistence:
                await self._persistence.save_goal(goal)

        return True

    async def delete(self, goal_id: str) -> bool:
        """
        Delete a goal.

        Args:
            goal_id: The goal ID to delete

        Returns:
            True if deleted, False if not found
        """
        async with self._lock:
            if goal_id not in self._goals:
                return False

            goal = self._goals[goal_id]
            self._remove_from_indices(goal)
            del self._goals[goal_id]

            # Record deletion event
            event = GoalEvent(
                goal_id=goal_id,
                event_type="deleted",
                description=f"Goal deleted: {goal.title}",
            )
            await self._add_event_internal(event)

            # Delete from persistence
            if self._persistence:
                await self._persistence.delete_goal(goal_id)

        logger.info("Goal deleted", goal_id=goal_id[:8])
        return True

    # === Queries ===

    async def get_all(self) -> list[Goal]:
        """Get all goals."""
        return list(self._goals.values())

    async def get_by_status(self, status: GoalStatus) -> list[Goal]:
        """Get goals by status."""
        goal_ids = self._status_index.get(status, set())
        return [self._goals[gid] for gid in goal_ids if gid in self._goals]

    async def get_by_priority(self, priority: GoalPriority) -> list[Goal]:
        """Get goals by priority."""
        goal_ids = self._priority_index.get(priority, set())
        return [self._goals[gid] for gid in goal_ids if gid in self._goals]

    async def get_by_type(self, goal_type: GoalType) -> list[Goal]:
        """Get goals by type."""
        goal_ids = self._type_index.get(goal_type, set())
        return [self._goals[gid] for gid in goal_ids if gid in self._goals]

    async def get_by_source(self, source: GoalSource) -> list[Goal]:
        """Get goals by source."""
        return [g for g in self._goals.values() if g.source == source]

    async def get_active(self) -> list[Goal]:
        """Get all active goals (pending or active status)."""
        active = []
        for status in (GoalStatus.PENDING, GoalStatus.ACTIVE):
            active.extend(await self.get_by_status(status))
        return active

    async def get_children(self, parent_id: str) -> list[Goal]:
        """Get child goals of a parent."""
        child_ids = self._parent_index.get(parent_id, set())
        return [self._goals[gid] for gid in child_ids if gid in self._goals]

    async def get_by_tag(self, tag: str) -> list[Goal]:
        """Get goals with a specific tag."""
        goal_ids = self._tag_index.get(tag, set())
        return [self._goals[gid] for gid in goal_ids if gid in self._goals]

    async def get_overdue(self) -> list[Goal]:
        """Get goals that are past their deadline."""
        return [g for g in self._goals.values() if g.is_overdue()]

    async def get_blocked(self) -> list[Goal]:
        """Get blocked goals."""
        return await self.get_by_status(GoalStatus.BLOCKED)

    async def search(
        self,
        status: Optional[GoalStatus] = None,
        priority: Optional[GoalPriority] = None,
        goal_type: Optional[GoalType] = None,
        source: Optional[GoalSource] = None,
        tags: Optional[list[str]] = None,
        created_after: Optional[datetime] = None,
        created_before: Optional[datetime] = None,
        has_deadline: Optional[bool] = None,
        limit: int = 100,
    ) -> list[Goal]:
        """
        Search goals with multiple criteria.

        Args:
            status: Filter by status
            priority: Filter by priority
            goal_type: Filter by type
            source: Filter by source
            tags: Filter by tags (all must match)
            created_after: Filter by creation time
            created_before: Filter by creation time
            has_deadline: Filter by deadline presence
            limit: Maximum results

        Returns:
            List of matching goals
        """
        results = list(self._goals.values())

        if status:
            results = [g for g in results if g.status == status]

        if priority:
            results = [g for g in results if g.priority == priority]

        if goal_type:
            results = [g for g in results if g.goal_type == goal_type]

        if source:
            results = [g for g in results if g.source == source]

        if tags:
            results = [g for g in results if all(t in g.tags for t in tags)]

        if created_after:
            results = [g for g in results if g.created_at >= created_after]

        if created_before:
            results = [g for g in results if g.created_at <= created_before]

        if has_deadline is not None:
            if has_deadline:
                results = [g for g in results if g.deadline is not None]
            else:
                results = [g for g in results if g.deadline is None]

        return results[:limit]

    # === Objectives ===

    async def create_objective(self, objective: Objective) -> str:
        """Create a new objective."""
        async with self._lock:
            self._objectives[objective.id] = objective

            if self._persistence:
                await self._persistence.save_objective(objective)

        logger.info(
            "Objective created", objective_id=objective.id[:8], name=objective.name
        )
        return objective.id

    async def get_objective(self, objective_id: str) -> Optional[Objective]:
        """Get an objective."""
        return self._objectives.get(objective_id)

    async def update_objective(self, objective: Objective) -> bool:
        """Update an objective."""
        async with self._lock:
            if objective.id not in self._objectives:
                return False

            self._objectives[objective.id] = objective

            if self._persistence:
                await self._persistence.save_objective(objective)

        return True

    async def delete_objective(self, objective_id: str) -> bool:
        """Delete an objective."""
        async with self._lock:
            if objective_id not in self._objectives:
                return False

            del self._objectives[objective_id]

            if self._persistence:
                await self._persistence.delete_objective(objective_id)

        return True

    async def get_all_objectives(self) -> list[Objective]:
        """Get all objectives."""
        return list(self._objectives.values())

    async def get_active_objectives(self) -> list[Objective]:
        """Get active objectives."""
        return [o for o in self._objectives.values() if o.active]

    # === Events ===

    async def add_event(self, event: GoalEvent) -> None:
        """Add a goal event."""
        async with self._lock:
            await self._add_event_internal(event)

    async def _add_event_internal(self, event: GoalEvent) -> None:
        """Internal event addition (assumes lock held)."""
        self._events.append(event)

        # Keep only recent events in memory
        if len(self._events) > self._max_events:
            self._events = self._events[-(self._max_events // 2) :]

        # Notify subscribers
        for subscriber in self._event_subscribers:
            try:
                subscriber(event)
            except Exception as e:
                logger.error("Event subscriber error", error=str(e))

        # Persist
        if self._persistence:
            await self._persistence.save_event(event)

    async def get_events(
        self,
        goal_id: Optional[str] = None,
        event_type: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> list[GoalEvent]:
        """Get goal events."""
        events = self._events

        if goal_id:
            events = [e for e in events if e.goal_id == goal_id]

        if event_type:
            events = [e for e in events if e.event_type == event_type]

        if since:
            events = [e for e in events if e.timestamp >= since]

        return events[-limit:]

    def subscribe_to_events(self, callback: Callable[[GoalEvent], None]) -> None:
        """Subscribe to goal events."""
        self._event_subscribers.append(callback)

    def unsubscribe_from_events(self, callback: Callable[[GoalEvent], None]) -> None:
        """Unsubscribe from goal events."""
        if callback in self._event_subscribers:
            self._event_subscribers.remove(callback)

    # === Progress Tracking ===

    async def update_progress(self, progress: GoalProgress) -> bool:
        """
        Update goal progress.

        Args:
            progress: The progress update

        Returns:
            True if updated
        """
        goal = await self.get(progress.goal_id)
        if not goal:
            return False

        # Update goal metrics
        goal.metrics.update_progress(progress.progress_percent)
        goal.status = progress.status

        if progress.artifacts_created:
            goal.artifacts.extend(progress.artifacts_created)

        await self.update(goal)

        # Record progress event
        event = GoalEvent(
            goal_id=progress.goal_id,
            event_type="progress",
            description=progress.message or f"Progress updated to {progress.progress_percent:.1f}%",
            data=progress.to_dict(),
        )
        await self.add_event(event)

        return True

    # === Statistics ===

    def get_stats(self) -> dict[str, Any]:
        """Get registry statistics."""
        status_counts = {}
        for status in GoalStatus:
            status_counts[status.value] = len(self._status_index.get(status, set()))

        priority_counts = {}
        for priority in GoalPriority:
            priority_counts[priority.name] = len(
                self._priority_index.get(priority, set())
            )

        type_counts = {}
        for goal_type in GoalType:
            type_counts[goal_type.value] = len(self._type_index.get(goal_type, set()))

        source_counts = {}
        for source in GoalSource:
            source_counts[source.value] = len(
                [g for g in self._goals.values() if g.source == source]
            )

        # Calculate completion rate
        completed = status_counts.get("completed", 0)
        failed = status_counts.get("failed", 0)
        total_terminal = completed + failed
        completion_rate = completed / total_terminal if total_terminal > 0 else 0.0

        # Calculate average progress of active goals
        active_goals = [g for g in self._goals.values() if g.is_active()]
        avg_progress = (
            sum(g.metrics.progress_percent for g in active_goals) / len(active_goals)
            if active_goals
            else 0.0
        )

        return {
            "total_goals": len(self._goals),
            "total_objectives": len(self._objectives),
            "total_events": len(self._events),
            "by_status": status_counts,
            "by_priority": priority_counts,
            "by_type": type_counts,
            "by_source": source_counts,
            "overdue_count": len([g for g in self._goals.values() if g.is_overdue()]),
            "completion_rate": completion_rate,
            "avg_active_progress": avg_progress,
        }

    # === Index Management ===

    def _update_indices(self, goal: Goal) -> None:
        """Update all indices for a goal."""
        self._status_index[goal.status].add(goal.id)
        self._priority_index[goal.priority].add(goal.id)
        self._type_index[goal.goal_type].add(goal.id)

        for tag in goal.tags:
            if tag not in self._tag_index:
                self._tag_index[tag] = set()
            self._tag_index[tag].add(goal.id)

        if goal.parent_goal_id:
            if goal.parent_goal_id not in self._parent_index:
                self._parent_index[goal.parent_goal_id] = set()
            self._parent_index[goal.parent_goal_id].add(goal.id)

    def _remove_from_indices(self, goal: Goal) -> None:
        """Remove goal from all indices."""
        self._status_index[goal.status].discard(goal.id)
        self._priority_index[goal.priority].discard(goal.id)
        self._type_index[goal.goal_type].discard(goal.id)

        for tag in goal.tags:
            if tag in self._tag_index:
                self._tag_index[tag].discard(goal.id)

        if goal.parent_goal_id and goal.parent_goal_id in self._parent_index:
            self._parent_index[goal.parent_goal_id].discard(goal.id)

    # === Persistence ===

    async def _load_from_persistence(self) -> None:
        """Load goals from persistence."""
        if not self._persistence:
            return

        try:
            goals = await self._persistence.load_all_goals()
            for goal in goals:
                self._goals[goal.id] = goal
                self._update_indices(goal)

            objectives = await self._persistence.load_all_objectives()
            for objective in objectives:
                self._objectives[objective.id] = objective

            events = await self._persistence.load_recent_events(limit=self._max_events)
            self._events = events

            logger.info(
                "Loaded from persistence",
                goals=len(goals),
                objectives=len(objectives),
                events=len(events),
            )
        except Exception as e:
            logger.error("Failed to load from persistence", error=str(e))

    async def _save_to_persistence(self) -> None:
        """Save all goals to persistence."""
        if not self._persistence:
            return

        try:
            for goal in self._goals.values():
                await self._persistence.save_goal(goal)

            for objective in self._objectives.values():
                await self._persistence.save_objective(objective)

            logger.info(
                "Saved to persistence",
                goals=len(self._goals),
                objectives=len(self._objectives),
            )
        except Exception as e:
            logger.error("Failed to save to persistence", error=str(e))
