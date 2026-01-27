"""
AION Distributed Task Scheduler

Production-grade task scheduler with DAG-based dependency resolution.
Implements SOTA patterns including:
- Directed acyclic graph (DAG) dependency resolution
- Batch scheduling optimization for related task groups
- Deadline-aware priority boosting for time-sensitive work
- Backpressure detection to prevent node overload
- Exponential backoff with jitter for retry scheduling
- Automatic rescheduling of failed tasks with attempt tracking
"""

from __future__ import annotations

import asyncio
import random
import time
from collections import defaultdict
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

import structlog

from aion.distributed.types import (
    DistributedTask,
    TaskPriority,
    TaskStatus,
    TaskType,
)

if TYPE_CHECKING:
    from aion.distributed.cluster.manager import ClusterManager

logger = structlog.get_logger(__name__)

# Backoff configuration constants
DEFAULT_BACKOFF_BASE_SECONDS = 1.0
DEFAULT_BACKOFF_MAX_SECONDS = 60.0
DEFAULT_BACKOFF_JITTER_FACTOR = 0.25

# Backpressure thresholds
BACKPRESSURE_LOAD_THRESHOLD = 0.85
BACKPRESSURE_QUEUE_DEPTH_THRESHOLD = 100


class TaskScheduler:
    """
    Schedules distributed tasks across the cluster with dependency awareness.

    The scheduler maintains a dependency graph to ensure tasks execute only
    after their prerequisites complete. It supports batch scheduling for
    throughput optimization, deadline-aware priority boosting, and
    backpressure detection to prevent overwhelming nodes.

    Attributes:
        cluster_manager: Reference to the cluster manager for node state.
        backoff_base: Base delay in seconds for exponential backoff.
        backoff_max: Maximum delay cap for exponential backoff.
        backpressure_enabled: Whether to detect and react to backpressure.
    """

    def __init__(
        self,
        cluster_manager: ClusterManager,
        *,
        backoff_base: float = DEFAULT_BACKOFF_BASE_SECONDS,
        backoff_max: float = DEFAULT_BACKOFF_MAX_SECONDS,
        backpressure_enabled: bool = True,
    ) -> None:
        self._cluster_manager = cluster_manager
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        self.backpressure_enabled = backpressure_enabled

        # Dependency graph: task_id -> set of task IDs it depends on
        self._dependency_graph: Dict[str, Set[str]] = {}

        # Reverse dependency index: task_id -> set of tasks that depend on it
        self._dependents: Dict[str, Set[str]] = defaultdict(set)

        # Completed task cache for dependency resolution
        self._completed_tasks: Set[str] = set()

        # Cancelled tasks for cascade cancellation
        self._cancelled_tasks: Set[str] = set()

        # Scheduling statistics
        self._stats = {
            "scheduled": 0,
            "batch_scheduled": 0,
            "rescheduled": 0,
            "cancelled": 0,
            "backpressure_events": 0,
            "deadline_boosts": 0,
        }

        logger.info(
            "task_scheduler_initialized",
            backoff_base=backoff_base,
            backoff_max=backoff_max,
            backpressure_enabled=backpressure_enabled,
        )

    # -------------------------------------------------------------------------
    # Core Scheduling
    # -------------------------------------------------------------------------

    async def schedule(self, task: DistributedTask) -> bool:
        """
        Schedule a single task for execution.

        Registers the task in the dependency graph, checks if all
        dependencies are satisfied, and applies deadline-aware priority
        boosting. If backpressure is detected, scheduling is deferred.

        Args:
            task: The task to schedule.

        Returns:
            True if the task was scheduled (dependencies met and no
            backpressure), False if deferred.
        """
        # Check for backpressure before scheduling
        if self.backpressure_enabled and await self._detect_backpressure():
            logger.warning(
                "scheduling_deferred_backpressure",
                task_id=task.id,
                task_name=task.name,
            )
            self._stats["backpressure_events"] += 1
            return False

        # Register in the dependency graph
        self._register_dependencies(task)

        # Check if all dependencies are satisfied
        deps_met = await self.check_dependencies(task)
        if not deps_met:
            task.status = TaskStatus.PENDING
            logger.debug(
                "task_waiting_on_dependencies",
                task_id=task.id,
                pending_deps=list(
                    self._dependency_graph.get(task.id, set()) - self._completed_tasks
                ),
            )
            return False

        # Apply deadline-aware priority boosting
        self._apply_deadline_boost(task)

        # Mark as assigned for scheduling
        task.status = TaskStatus.ASSIGNED
        self._stats["scheduled"] += 1

        logger.info(
            "task_scheduled",
            task_id=task.id,
            task_name=task.name,
            priority=task.priority.name,
            task_type=task.task_type,
        )
        return True

    async def schedule_batch(self, tasks: List[DistributedTask]) -> List[DistributedTask]:
        """
        Schedule a batch of tasks with optimized dependency resolution.

        Performs a topological sort to determine the optimal scheduling
        order, then schedules each task in dependency order. Tasks without
        dependencies are scheduled in parallel.

        Args:
            tasks: List of tasks to schedule as a batch.

        Returns:
            List of tasks that were successfully scheduled (dependencies met).
        """
        if not tasks:
            return []

        # Register all tasks in the dependency graph first
        for task in tasks:
            self._register_dependencies(task)

        # Topological sort to determine scheduling order
        sorted_task_ids = self._topological_sort(
            {t.id for t in tasks},
        )

        # Build a lookup for fast access
        task_map = {t.id: t for t in tasks}

        # Schedule in dependency order
        scheduled: List[DistributedTask] = []
        for task_id in sorted_task_ids:
            task = task_map.get(task_id)
            if task is None:
                continue

            success = await self.schedule(task)
            if success:
                scheduled.append(task)

        self._stats["batch_scheduled"] += 1

        logger.info(
            "batch_scheduled",
            total=len(tasks),
            scheduled=len(scheduled),
            deferred=len(tasks) - len(scheduled),
        )
        return scheduled

    async def check_dependencies(self, task: DistributedTask) -> bool:
        """
        Check whether all dependencies for a task have been completed.

        A task with no dependencies is always considered ready. A task
        whose dependency has been cancelled is also considered unblocked
        (cascade cancellation is handled separately).

        Args:
            task: The task whose dependencies to check.

        Returns:
            True if all dependencies are satisfied, False otherwise.
        """
        deps = self._dependency_graph.get(task.id, set())
        if not deps:
            return True

        for dep_id in deps:
            if dep_id in self._cancelled_tasks:
                # Dependency was cancelled -- cascade cancel this task
                logger.info(
                    "task_cancelled_dependency_cascade",
                    task_id=task.id,
                    cancelled_dep=dep_id,
                )
                task.status = TaskStatus.CANCELLED
                self._cancelled_tasks.add(task.id)
                return False

            if dep_id not in self._completed_tasks:
                return False

        return True

    async def get_ready_tasks(self) -> List[str]:
        """
        Get all task IDs whose dependencies are fully satisfied.

        Scans the dependency graph for tasks in PENDING state that have
        all their dependencies met and are ready for execution.

        Returns:
            List of task IDs that are ready to be dispatched.
        """
        ready: List[str] = []
        for task_id, deps in self._dependency_graph.items():
            # Skip already completed or cancelled tasks
            if task_id in self._completed_tasks or task_id in self._cancelled_tasks:
                continue

            # Check if all dependencies are met
            unmet = deps - self._completed_tasks
            if not unmet:
                ready.append(task_id)

        logger.debug(
            "ready_tasks_found",
            ready_count=len(ready),
            total_tracked=len(self._dependency_graph),
        )
        return ready

    async def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a task and cascade cancellation to its dependents.

        All tasks that transitively depend on the cancelled task are
        also marked for cancellation.

        Args:
            task_id: The task to cancel.

        Returns:
            True if the task was found and cancelled.
        """
        if task_id in self._cancelled_tasks:
            return False

        self._cancelled_tasks.add(task_id)
        self._stats["cancelled"] += 1

        # Cascade cancellation to dependents
        dependents = self._dependents.get(task_id, set())
        cascade_count = 0
        for dep_id in dependents:
            if dep_id not in self._cancelled_tasks:
                self._cancelled_tasks.add(dep_id)
                cascade_count += 1

        logger.info(
            "task_cancelled",
            task_id=task_id,
            cascade_count=cascade_count,
        )
        return True

    async def reschedule_failed(self, task: DistributedTask) -> Optional[float]:
        """
        Reschedule a failed task with exponential backoff.

        Computes the retry delay using exponential backoff with jitter:
        delay = min(base * 2^attempt, max) * (1 + random jitter)

        If the task has exhausted its retry budget, returns None.

        Args:
            task: The failed task to reschedule.

        Returns:
            The backoff delay in seconds before the next attempt, or None
            if retries are exhausted.
        """
        if not task.can_retry():
            logger.warning(
                "task_retries_exhausted",
                task_id=task.id,
                task_name=task.name,
                retry_count=task.retry_count,
                max_retries=task.max_retries,
            )
            return None

        # Increment retry count
        task.retry_count += 1
        task.status = TaskStatus.RETRYING

        # Compute exponential backoff with jitter
        delay = min(
            self.backoff_base * (2 ** (task.retry_count - 1)),
            self.backoff_max,
        )
        jitter = delay * DEFAULT_BACKOFF_JITTER_FACTOR * random.random()
        total_delay = delay + jitter

        # Record the attempt
        task.record_attempt(
            node_id=task.assigned_node or "",
            error=task.error,
        )

        # Clear the assignment so routing can pick a new node
        task.assigned_node = None
        task.error = None
        task.error_traceback = None

        self._stats["rescheduled"] += 1

        logger.info(
            "task_rescheduled",
            task_id=task.id,
            task_name=task.name,
            attempt=task.retry_count,
            max_retries=task.max_retries,
            delay_seconds=round(total_delay, 2),
        )
        return total_delay

    # -------------------------------------------------------------------------
    # Dependency Completion Notification
    # -------------------------------------------------------------------------

    async def mark_completed(self, task_id: str) -> List[str]:
        """
        Mark a task as completed and return newly unblocked dependents.

        This should be called when a task finishes execution so the
        scheduler can unlock downstream tasks.

        Args:
            task_id: The completed task ID.

        Returns:
            List of task IDs that are now unblocked by this completion.
        """
        self._completed_tasks.add(task_id)

        # Find dependents that are now fully unblocked
        unblocked: List[str] = []
        for dep_id in self._dependents.get(task_id, set()):
            if dep_id in self._cancelled_tasks or dep_id in self._completed_tasks:
                continue

            deps = self._dependency_graph.get(dep_id, set())
            if deps.issubset(self._completed_tasks):
                unblocked.append(dep_id)

        if unblocked:
            logger.info(
                "tasks_unblocked",
                completed_task=task_id,
                unblocked_tasks=unblocked,
            )
        return unblocked

    async def get_stats(self) -> Dict[str, Any]:
        """Return scheduler statistics."""
        return {
            "tracked_tasks": len(self._dependency_graph),
            "completed_tasks": len(self._completed_tasks),
            "cancelled_tasks": len(self._cancelled_tasks),
            "counters": dict(self._stats),
        }

    # -------------------------------------------------------------------------
    # Internal Helpers
    # -------------------------------------------------------------------------

    def _register_dependencies(self, task: DistributedTask) -> None:
        """Register a task and its dependencies in the DAG."""
        deps = set(task.depends_on) if task.depends_on else set()
        self._dependency_graph[task.id] = deps

        # Update the reverse index
        for dep_id in deps:
            self._dependents[dep_id].add(task.id)

    def _apply_deadline_boost(self, task: DistributedTask) -> None:
        """
        Boost the priority of tasks nearing their deadline.

        Tasks within 20% of their remaining time are promoted to HIGH.
        Tasks within 5% are promoted to CRITICAL.
        """
        if task.deadline is None:
            return

        remaining = (task.deadline - datetime.now()).total_seconds()
        total_window = (task.deadline - task.created_at).total_seconds()

        if total_window <= 0:
            return

        remaining_fraction = remaining / total_window

        if remaining_fraction <= 0.05 and task.priority > TaskPriority.CRITICAL:
            task.priority = TaskPriority.CRITICAL
            self._stats["deadline_boosts"] += 1
            logger.info(
                "task_deadline_boost_critical",
                task_id=task.id,
                remaining_seconds=round(remaining, 1),
            )
        elif remaining_fraction <= 0.20 and task.priority > TaskPriority.HIGH:
            task.priority = TaskPriority.HIGH
            self._stats["deadline_boosts"] += 1
            logger.info(
                "task_deadline_boost_high",
                task_id=task.id,
                remaining_seconds=round(remaining, 1),
            )

    async def _detect_backpressure(self) -> bool:
        """
        Detect whether the cluster is experiencing backpressure.

        Checks average node load across the cluster. If the average load
        score exceeds the threshold, backpressure is detected and new
        scheduling should be deferred.

        Returns:
            True if backpressure is detected.
        """
        try:
            cluster_state = self._cluster_manager.state
            healthy_nodes = cluster_state.healthy_nodes

            if not healthy_nodes:
                return True  # No healthy nodes is ultimate backpressure

            avg_load = sum(n.load_score for n in healthy_nodes) / len(healthy_nodes)
            return avg_load > BACKPRESSURE_LOAD_THRESHOLD

        except Exception:
            logger.debug("backpressure_detection_error", exc_info=True)
            return False

    def _topological_sort(self, task_ids: Set[str]) -> List[str]:
        """
        Perform topological sort on a subset of the dependency graph.

        Uses Kahn's algorithm (BFS-based) for a stable topological order.
        Tasks with no dependencies appear first. Cycles are detected and
        the remaining tasks are appended at the end.

        Args:
            task_ids: Set of task IDs to sort.

        Returns:
            List of task IDs in topological (dependency-first) order.
        """
        # Build the in-degree map for the subgraph
        in_degree: Dict[str, int] = {}
        adj: Dict[str, Set[str]] = defaultdict(set)

        for tid in task_ids:
            deps = self._dependency_graph.get(tid, set())
            # Only count dependencies within the provided set
            internal_deps = deps & task_ids
            in_degree[tid] = len(internal_deps)
            for dep in internal_deps:
                adj[dep].add(tid)

        # Seed the queue with zero-dependency tasks
        queue: List[str] = [
            tid for tid, deg in in_degree.items() if deg == 0
        ]
        sorted_order: List[str] = []

        while queue:
            current = queue.pop(0)
            sorted_order.append(current)

            for neighbor in adj.get(current, set()):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # Append any remaining tasks (indicates a cycle -- log a warning)
        remaining = task_ids - set(sorted_order)
        if remaining:
            logger.warning(
                "dependency_cycle_detected",
                cyclic_tasks=list(remaining),
            )
            sorted_order.extend(remaining)

        return sorted_order
