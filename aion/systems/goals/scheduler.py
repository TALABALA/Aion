"""
AION Goal Scheduler

Manages goal execution scheduling:
- Priority-based scheduling
- Resource allocation
- Dependency management
- Time-based triggers
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Any, Callable, Optional
import heapq

import structlog

from aion.systems.goals.types import (
    Goal,
    GoalStatus,
    GoalPriority,
    GoalEvent,
)

logger = structlog.get_logger(__name__)


class GoalScheduler:
    """
    Schedules and coordinates goal execution.

    Features:
    - Priority queue based scheduling
    - Concurrent goal execution (with limits)
    - Dependency resolution
    - Resource-aware scheduling
    - Time-based activation
    """

    def __init__(
        self,
        registry: Any,  # GoalRegistry
        executor: Any,  # GoalExecutor
        max_concurrent_goals: int = 3,
        scheduling_interval: float = 60.0,
    ):
        self._registry = registry
        self._executor = executor
        self._max_concurrent_goals = max_concurrent_goals
        self._scheduling_interval = scheduling_interval

        # Active executions
        self._active_goals: dict[str, asyncio.Task] = {}

        # Scheduling state
        self._schedule_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        self._paused = False

        # Statistics
        self._stats = {
            "goals_scheduled": 0,
            "goals_completed": 0,
            "goals_failed": 0,
            "scheduling_cycles": 0,
            "total_execution_time": 0.0,
        }

        # Callbacks
        self._completion_callbacks: list[Callable[[Goal, dict], None]] = []

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the scheduler."""
        if self._initialized:
            return

        logger.info("Initializing Goal Scheduler")

        # Start scheduling loop
        self._schedule_task = asyncio.create_task(self._scheduling_loop())

        self._initialized = True
        logger.info(
            "Goal Scheduler initialized",
            max_concurrent=self._max_concurrent_goals,
            interval=self._scheduling_interval,
        )

    async def shutdown(self) -> None:
        """Shutdown the scheduler."""
        logger.info("Shutting down Goal Scheduler")

        self._shutdown_event.set()

        # Cancel scheduling loop
        if self._schedule_task:
            self._schedule_task.cancel()
            try:
                await self._schedule_task
            except asyncio.CancelledError:
                pass

        # Stop active goals gracefully
        for goal_id, task in list(self._active_goals.items()):
            await self.pause_goal(goal_id)

        self._initialized = False

    async def _scheduling_loop(self) -> None:
        """Main scheduling loop."""
        while not self._shutdown_event.is_set():
            try:
                if not self._paused:
                    await self._schedule_cycle()
                    self._stats["scheduling_cycles"] += 1

                await asyncio.sleep(self._scheduling_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scheduling cycle error: {e}")
                await asyncio.sleep(5)  # Brief pause before retry

    async def _schedule_cycle(self) -> None:
        """Run one scheduling cycle."""
        # Get all pending goals
        pending_goals = await self._registry.get_by_status(GoalStatus.PENDING)

        # Check if we can start more goals
        available_slots = self._max_concurrent_goals - len(self._active_goals)

        if available_slots <= 0:
            return

        # Sort by priority score
        ready_goals = []
        for goal in pending_goals:
            if self._can_start_goal(goal):
                score = self._calculate_priority_score(goal)
                heapq.heappush(ready_goals, (-score, goal.id, goal))

        # Start goals
        started = 0
        while ready_goals and started < available_slots:
            _, _, goal = heapq.heappop(ready_goals)
            success = await self.start_goal(goal.id)
            if success:
                started += 1

        # Check blocked goals
        await self._check_blocked_goals()

        # Check for overdue goals
        await self._check_overdue_goals()

    def _can_start_goal(self, goal: Goal) -> bool:
        """Check if a goal can be started."""
        now = datetime.now()

        for constraint in goal.constraints:
            # Time constraints
            if constraint.earliest_start and now < constraint.earliest_start:
                return False

            # Dependency constraints
            for dep_id in constraint.depends_on_goals:
                dep_goal = self._registry._goals.get(dep_id)
                if dep_goal and dep_goal.status != GoalStatus.COMPLETED:
                    return False

            # Approval constraints
            if constraint.requires_approval and not constraint.approved_at:
                return False

        return True

    def _calculate_priority_score(self, goal: Goal) -> float:
        """Calculate priority score for scheduling."""
        score = 100.0

        # Base priority (lower priority value = higher score)
        score += (6 - goal.priority.value) * 20

        # Deadline urgency
        if goal.deadline:
            time_until_deadline = (goal.deadline - datetime.now()).total_seconds()
            if time_until_deadline < 3600:  # Less than 1 hour
                score += 50
            elif time_until_deadline < 86400:  # Less than 1 day
                score += 30
            elif time_until_deadline < 604800:  # Less than 1 week
                score += 10
            elif time_until_deadline < 0:  # Overdue
                score += 100  # Highest urgency

        # Age bonus (older pending goals get slight priority)
        age_hours = (datetime.now() - goal.created_at).total_seconds() / 3600
        score += min(age_hours * 0.5, 10)

        # Expected value bonus
        score += goal.expected_value * 10

        # Dependency satisfaction bonus
        all_deps_met = all(
            self._registry._goals.get(dep_id, Goal()).status == GoalStatus.COMPLETED
            for constraint in goal.constraints
            for dep_id in constraint.depends_on_goals
        )
        if all_deps_met:
            score += 5

        return score

    async def start_goal(self, goal_id: str) -> bool:
        """Start executing a goal."""
        goal = await self._registry.get(goal_id)
        if not goal:
            return False

        if goal.id in self._active_goals:
            return False  # Already running

        # Update status
        goal.status = GoalStatus.ACTIVE
        goal.started_at = datetime.now()
        await self._registry.update(goal)

        # Record event
        await self._registry.add_event(
            GoalEvent(
                goal_id=goal_id,
                event_type="started",
                description="Goal execution started",
            )
        )

        # Start execution task
        task = asyncio.create_task(self._execute_goal(goal))
        self._active_goals[goal_id] = task

        self._stats["goals_scheduled"] += 1

        logger.info(f"Started goal: {goal.title}", goal_id=goal_id[:8])

        return True

    async def _execute_goal(self, goal: Goal) -> None:
        """Execute a goal (runs in background task)."""
        start_time = datetime.now()

        try:
            # Execute via executor
            result = await self._executor.execute(goal)

            # Update goal based on result
            if result.get("success"):
                goal.status = GoalStatus.COMPLETED
                goal.completed_at = datetime.now()
                goal.outcome = result.get("outcome", "Completed successfully")
                goal.artifacts = result.get("artifacts", [])
                goal.learnings = result.get("learnings", [])
                self._stats["goals_completed"] += 1
            else:
                goal.status = GoalStatus.FAILED
                goal.completed_at = datetime.now()
                goal.outcome = result.get("outcome", "Failed")
                goal.status_reason = result.get("reason", "Unknown failure")
                self._stats["goals_failed"] += 1

            # Update metrics
            goal.metrics.progress_percent = (
                100.0 if result.get("success") else goal.metrics.progress_percent
            )

            if result.get("metrics"):
                metrics = result["metrics"]
                goal.metrics.tokens_used = metrics.get(
                    "tokens_used", goal.metrics.tokens_used
                )
                goal.metrics.time_spent_seconds = metrics.get(
                    "duration_seconds", goal.metrics.time_spent_seconds
                )

            await self._registry.update(goal)

            # Record event
            await self._registry.add_event(
                GoalEvent(
                    goal_id=goal.id,
                    event_type="completed" if result.get("success") else "failed",
                    description=goal.outcome or "",
                    data=result,
                )
            )

            # Notify callbacks
            for callback in self._completion_callbacks:
                try:
                    callback(goal, result)
                except Exception as e:
                    logger.error(f"Completion callback error: {e}")

        except asyncio.CancelledError:
            # Goal was paused/cancelled
            goal.status = GoalStatus.PAUSED
            await self._registry.update(goal)

        except Exception as e:
            logger.error(f"Goal execution error: {e}", goal_id=goal.id[:8])
            goal.status = GoalStatus.FAILED
            goal.outcome = str(e)
            goal.status_reason = "Execution error"
            await self._registry.update(goal)
            self._stats["goals_failed"] += 1

        finally:
            self._active_goals.pop(goal.id, None)
            execution_time = (datetime.now() - start_time).total_seconds()
            self._stats["total_execution_time"] += execution_time

    async def pause_goal(self, goal_id: str) -> bool:
        """Pause a running goal."""
        task = self._active_goals.get(goal_id)
        if not task:
            return False

        # Signal executor to pause
        self._executor.pause_execution(goal_id)

        # Cancel the task
        task.cancel()

        try:
            await task
        except asyncio.CancelledError:
            pass

        goal = await self._registry.get(goal_id)
        if goal:
            goal.status = GoalStatus.PAUSED
            await self._registry.update(goal)

            await self._registry.add_event(
                GoalEvent(
                    goal_id=goal_id,
                    event_type="paused",
                    description="Goal execution paused",
                )
            )

        logger.info(f"Paused goal", goal_id=goal_id[:8])
        return True

    async def resume_goal(self, goal_id: str) -> bool:
        """Resume a paused goal."""
        goal = await self._registry.get(goal_id)
        if not goal or goal.status != GoalStatus.PAUSED:
            return False

        goal.status = GoalStatus.PENDING
        await self._registry.update(goal)

        await self._registry.add_event(
            GoalEvent(
                goal_id=goal_id,
                event_type="resumed",
                description="Goal queued for resumption",
            )
        )

        logger.info(f"Resumed goal", goal_id=goal_id[:8])
        return True

    async def stop_goal(self, goal_id: str, reason: str = "Stopped") -> bool:
        """Stop a running goal."""
        task = self._active_goals.get(goal_id)
        if task:
            self._executor.cancel_execution(goal_id)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        goal = await self._registry.get(goal_id)
        if goal:
            goal.status = GoalStatus.ABANDONED
            goal.status_reason = reason
            goal.completed_at = datetime.now()
            await self._registry.update(goal)

            await self._registry.add_event(
                GoalEvent(
                    goal_id=goal_id,
                    event_type="stopped",
                    description=reason,
                )
            )

        return True

    async def _check_blocked_goals(self) -> None:
        """Check blocked goals for unblocking."""
        blocked_goals = await self._registry.get_by_status(GoalStatus.BLOCKED)

        for goal in blocked_goals:
            # Check if blocking conditions are resolved
            can_unblock = True

            for constraint in goal.constraints:
                for dep_id in constraint.depends_on_goals:
                    dep_goal = await self._registry.get(dep_id)
                    if dep_goal and dep_goal.status != GoalStatus.COMPLETED:
                        can_unblock = False
                        break

            if can_unblock:
                goal.status = GoalStatus.PENDING
                goal.status_reason = None
                await self._registry.update(goal)

                await self._registry.add_event(
                    GoalEvent(
                        goal_id=goal.id,
                        event_type="unblocked",
                        description="Blocking conditions resolved",
                    )
                )

    async def _check_overdue_goals(self) -> None:
        """Check for overdue goals and handle them."""
        overdue_goals = await self._registry.get_overdue()

        for goal in overdue_goals:
            if goal.status in (GoalStatus.PENDING, GoalStatus.PAUSED):
                # Increase priority for overdue goals
                if goal.priority.value > GoalPriority.HIGH.value:
                    goal.priority = GoalPriority.HIGH
                    await self._registry.update(goal)

                    await self._registry.add_event(
                        GoalEvent(
                            goal_id=goal.id,
                            event_type="priority_escalated",
                            description="Goal is overdue, priority escalated",
                        )
                    )

    def pause_scheduler(self) -> None:
        """Pause the scheduler."""
        self._paused = True
        logger.info("Scheduler paused")

    def resume_scheduler(self) -> None:
        """Resume the scheduler."""
        self._paused = False
        logger.info("Scheduler resumed")

    def is_paused(self) -> bool:
        """Check if scheduler is paused."""
        return self._paused

    def get_active_goals(self) -> list[str]:
        """Get IDs of currently active goals."""
        return list(self._active_goals.keys())

    def get_stats(self) -> dict[str, Any]:
        """Get scheduler statistics."""
        return {
            **self._stats,
            "active_goals": len(self._active_goals),
            "max_concurrent": self._max_concurrent_goals,
            "paused": self._paused,
            "scheduling_interval": self._scheduling_interval,
        }

    def set_max_concurrent(self, max_concurrent: int) -> None:
        """Set maximum concurrent goals."""
        self._max_concurrent_goals = max(1, max_concurrent)

    def set_scheduling_interval(self, interval: float) -> None:
        """Set scheduling interval."""
        self._scheduling_interval = max(1.0, interval)

    def on_completion(self, callback: Callable[[Goal, dict], None]) -> None:
        """Register completion callback."""
        self._completion_callbacks.append(callback)
