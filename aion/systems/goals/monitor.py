"""
AION Goal Monitor

Progress tracking and monitoring for autonomous goals:
- Real-time progress updates
- Health monitoring
- Performance analytics
- Alerts and notifications
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Any, Callable, Optional

import structlog

from aion.systems.goals.types import (
    Goal,
    GoalStatus,
    GoalEvent,
    GoalProgress,
)

logger = structlog.get_logger(__name__)


class GoalMonitor:
    """
    Monitors goal progress and health.

    Features:
    - Real-time progress tracking
    - Stall detection
    - Performance metrics
    - Alerting system
    """

    def __init__(
        self,
        registry: Any,  # GoalRegistry
        reasoner: Optional[Any] = None,  # GoalReasoner
        monitoring_interval: float = 30.0,
        stall_threshold_minutes: int = 15,
    ):
        self._registry = registry
        self._reasoner = reasoner
        self._monitoring_interval = monitoring_interval
        self._stall_threshold = timedelta(minutes=stall_threshold_minutes)

        # Monitoring state
        self._monitor_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

        # Tracking
        self._last_progress: dict[str, tuple[float, datetime]] = {}
        self._stalled_goals: set[str] = set()

        # Alerts
        self._alert_callbacks: list[Callable[[str, Goal, dict], None]] = []

        # Statistics
        self._stats = {
            "monitoring_cycles": 0,
            "alerts_triggered": 0,
            "stalls_detected": 0,
        }

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the monitor."""
        if self._initialized:
            return

        logger.info("Initializing Goal Monitor")

        # Start monitoring loop
        self._monitor_task = asyncio.create_task(self._monitoring_loop())

        self._initialized = True

    async def shutdown(self) -> None:
        """Shutdown the monitor."""
        logger.info("Shutting down Goal Monitor")

        self._shutdown_event.set()

        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

        self._initialized = False

    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while not self._shutdown_event.is_set():
            try:
                await self._monitoring_cycle()
                self._stats["monitoring_cycles"] += 1

                await asyncio.sleep(self._monitoring_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring cycle error: {e}")
                await asyncio.sleep(5)

    async def _monitoring_cycle(self) -> None:
        """Run one monitoring cycle."""
        # Get active goals
        active_goals = await self._registry.get_active()

        for goal in active_goals:
            await self._monitor_goal(goal)

        # Clean up tracking for completed goals
        active_ids = {g.id for g in active_goals}
        for goal_id in list(self._last_progress.keys()):
            if goal_id not in active_ids:
                del self._last_progress[goal_id]
                self._stalled_goals.discard(goal_id)

    async def _monitor_goal(self, goal: Goal) -> None:
        """Monitor a single goal."""
        now = datetime.now()
        current_progress = goal.metrics.progress_percent

        # Check for stalls
        if goal.id in self._last_progress:
            last_progress, last_time = self._last_progress[goal.id]

            if current_progress == last_progress:
                # No progress since last check
                time_since_progress = now - last_time

                if time_since_progress > self._stall_threshold:
                    if goal.id not in self._stalled_goals:
                        self._stalled_goals.add(goal.id)
                        self._stats["stalls_detected"] += 1
                        await self._trigger_alert(
                            "stall",
                            goal,
                            {
                                "time_since_progress": time_since_progress.total_seconds(),
                                "current_progress": current_progress,
                            },
                        )
            else:
                # Progress made
                self._last_progress[goal.id] = (current_progress, now)
                self._stalled_goals.discard(goal.id)
        else:
            # First time tracking this goal
            self._last_progress[goal.id] = (current_progress, now)

        # Check for deadline approaching
        if goal.deadline:
            time_until_deadline = goal.time_until_deadline()
            if time_until_deadline:
                hours_remaining = time_until_deadline.total_seconds() / 3600

                if hours_remaining < 1 and current_progress < 90:
                    await self._trigger_alert(
                        "deadline_imminent",
                        goal,
                        {
                            "hours_remaining": hours_remaining,
                            "progress": current_progress,
                        },
                    )
                elif hours_remaining < 24 and current_progress < 50:
                    await self._trigger_alert(
                        "deadline_approaching",
                        goal,
                        {
                            "hours_remaining": hours_remaining,
                            "progress": current_progress,
                        },
                    )

        # Check resource usage
        for constraint in goal.constraints:
            if constraint.max_tokens and goal.metrics.tokens_used > constraint.max_tokens * 0.9:
                await self._trigger_alert(
                    "resource_high",
                    goal,
                    {
                        "resource": "tokens",
                        "used": goal.metrics.tokens_used,
                        "limit": constraint.max_tokens,
                    },
                )

    async def _trigger_alert(
        self,
        alert_type: str,
        goal: Goal,
        data: dict[str, Any],
    ) -> None:
        """Trigger an alert."""
        self._stats["alerts_triggered"] += 1

        # Record event
        await self._registry.add_event(
            GoalEvent(
                goal_id=goal.id,
                event_type=f"alert_{alert_type}",
                description=f"Alert: {alert_type}",
                data=data,
            )
        )

        # Notify callbacks
        for callback in self._alert_callbacks:
            try:
                callback(alert_type, goal, data)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")

        logger.warning(
            f"Goal alert: {alert_type}",
            goal_id=goal.id[:8],
            goal_title=goal.title,
            data=data,
        )

    async def get_goal_health(self, goal_id: str) -> dict[str, Any]:
        """Get health status for a goal."""
        goal = await self._registry.get(goal_id)
        if not goal:
            return {"error": "Goal not found"}

        health = {
            "goal_id": goal_id,
            "status": goal.status.value,
            "progress": goal.metrics.progress_percent,
            "healthy": True,
            "issues": [],
        }

        # Check for stall
        if goal_id in self._stalled_goals:
            health["healthy"] = False
            health["issues"].append("Goal is stalled (no progress)")

        # Check deadline
        if goal.is_overdue():
            health["healthy"] = False
            health["issues"].append("Goal is overdue")
        elif goal.deadline:
            remaining = goal.time_until_deadline()
            if remaining and remaining.total_seconds() < 3600:
                health["issues"].append("Deadline imminent")

        # Check resource usage
        for constraint in goal.constraints:
            if constraint.max_tokens and goal.metrics.tokens_used > constraint.max_tokens * 0.9:
                health["issues"].append("Approaching token limit")
            if constraint.max_cost_dollars:
                # Estimate cost (rough approximation)
                estimated_cost = goal.metrics.tokens_used * 0.00001  # Example rate
                if estimated_cost > constraint.max_cost_dollars * 0.9:
                    health["issues"].append("Approaching cost limit")

        return health

    async def get_progress_report(self, goal_id: str) -> dict[str, Any]:
        """Get detailed progress report for a goal."""
        goal = await self._registry.get(goal_id)
        if not goal:
            return {"error": "Goal not found"}

        events = await self._registry.get_events(goal_id=goal_id, limit=20)

        report = {
            "goal_id": goal_id,
            "title": goal.title,
            "status": goal.status.value,
            "progress_percent": goal.metrics.progress_percent,
            "started_at": goal.started_at.isoformat() if goal.started_at else None,
            "duration": None,
            "metrics": goal.metrics.to_dict(),
            "recent_events": [
                {
                    "type": e.event_type,
                    "description": e.description,
                    "timestamp": e.timestamp.isoformat(),
                }
                for e in events
            ],
            "subgoals": [],
        }

        # Calculate duration
        if goal.started_at:
            end_time = goal.completed_at or datetime.now()
            duration = end_time - goal.started_at
            report["duration"] = {
                "seconds": duration.total_seconds(),
                "human": self._format_duration(duration),
            }

        # Get subgoal status
        for subgoal_id in goal.subgoal_ids:
            subgoal = await self._registry.get(subgoal_id)
            if subgoal:
                report["subgoals"].append(
                    {
                        "id": subgoal_id,
                        "title": subgoal.title,
                        "status": subgoal.status.value,
                        "progress": subgoal.metrics.progress_percent,
                    }
                )

        # Get LLM assessment if available
        if self._reasoner and goal.status == GoalStatus.ACTIVE:
            try:
                assessment = await self._reasoner.assess_progress(
                    goal,
                    [e.to_dict() for e in events],
                    [],  # artifacts
                )
                report["assessment"] = assessment
            except Exception as e:
                logger.warning(f"Could not get progress assessment: {e}")

        return report

    async def get_system_health(self) -> dict[str, Any]:
        """Get overall goal system health."""
        all_goals = await self._registry.get_all()
        active_goals = await self._registry.get_active()

        stats = self._registry.get_stats()

        health = {
            "overall_status": "healthy",
            "issues": [],
            "summary": {
                "total_goals": len(all_goals),
                "active_goals": len(active_goals),
                "stalled_goals": len(self._stalled_goals),
                "overdue_goals": stats.get("overdue_count", 0),
                "completion_rate": stats.get("completion_rate", 0.0),
            },
            "by_status": stats.get("by_status", {}),
            "monitoring_stats": self._stats,
        }

        # Check for issues
        if len(self._stalled_goals) > 0:
            health["issues"].append(f"{len(self._stalled_goals)} goals are stalled")

        if stats.get("overdue_count", 0) > 0:
            health["issues"].append(f"{stats['overdue_count']} goals are overdue")

        completion_rate = stats.get("completion_rate", 0.0)
        if completion_rate < 0.5 and len(all_goals) > 10:
            health["issues"].append(f"Low completion rate: {completion_rate:.0%}")

        if health["issues"]:
            health["overall_status"] = "degraded"

        return health

    def _format_duration(self, duration: timedelta) -> str:
        """Format duration for human readability."""
        total_seconds = int(duration.total_seconds())

        if total_seconds < 60:
            return f"{total_seconds}s"
        elif total_seconds < 3600:
            minutes = total_seconds // 60
            seconds = total_seconds % 60
            return f"{minutes}m {seconds}s"
        else:
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            return f"{hours}h {minutes}m"

    def on_alert(self, callback: Callable[[str, Goal, dict], None]) -> None:
        """Register alert callback."""
        self._alert_callbacks.append(callback)

    def get_stalled_goals(self) -> list[str]:
        """Get list of stalled goal IDs."""
        return list(self._stalled_goals)

    def get_stats(self) -> dict[str, Any]:
        """Get monitor statistics."""
        return {
            **self._stats,
            "stalled_goals": len(self._stalled_goals),
            "monitoring_interval": self._monitoring_interval,
            "stall_threshold_minutes": self._stall_threshold.total_seconds() / 60,
        }
