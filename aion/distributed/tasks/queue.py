"""
AION Distributed Task Queue

Production-grade priority task queue with thread-safe async operations.
Implements SOTA patterns including:
- Heap-based priority ordering (lower priority value = higher priority)
- Idempotency key deduplication within a configurable window
- Dead letter queue for tasks that exhaust retry budgets
- TTL-based task expiration enforcement
- Per-node task tracking for locality-aware scheduling
- Comprehensive queue statistics for observability
"""

from __future__ import annotations

import asyncio
import heapq
import time
from collections import defaultdict
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

import structlog

from aion.distributed.types import DistributedTask, TaskPriority, TaskStatus

if TYPE_CHECKING:
    from aion.distributed.cluster.manager import ClusterManager

logger = structlog.get_logger(__name__)


class DistributedTaskQueue:
    """
    Thread-safe distributed task queue with priority ordering.

    Tasks are ordered by priority (lower value = higher priority) and then by
    creation timestamp for FIFO ordering within the same priority level.
    The queue supports deduplication via idempotency keys, automatic TTL
    enforcement, and a dead letter queue for tasks that exceed their retry
    budget.

    Attributes:
        cluster_manager: Reference to the cluster manager for node awareness.
        max_queue_size: Maximum number of tasks allowed in the queue.
        task_ttl_seconds: Default time-to-live for tasks in seconds.
        dead_letter_enabled: Whether to route exhausted tasks to dead letter.
        dead_letter_max_size: Maximum capacity of the dead letter queue.
        deduplication_window_seconds: Time window for idempotency deduplication.
    """

    def __init__(
        self,
        cluster_manager: ClusterManager,
        *,
        max_queue_size: int = 10000,
        task_ttl_seconds: int = 3600,
        dead_letter_enabled: bool = True,
        dead_letter_max_size: int = 1000,
        deduplication_window_seconds: int = 300,
    ) -> None:
        self._cluster_manager = cluster_manager
        self.max_queue_size = max_queue_size
        self.task_ttl_seconds = task_ttl_seconds
        self.dead_letter_enabled = dead_letter_enabled
        self.dead_letter_max_size = dead_letter_max_size
        self.deduplication_window_seconds = deduplication_window_seconds

        # Primary task storage keyed by task ID
        self._tasks: Dict[str, DistributedTask] = {}

        # Heap-based priority queue: (priority, created_timestamp, task_id)
        self._heap: List[Tuple[int, float, str]] = []

        # Per-node task index for locality queries
        self._tasks_by_node: Dict[str, Set[str]] = defaultdict(set)

        # Deduplication index: idempotency_key -> (task_id, expiry_timestamp)
        self._idempotency_index: Dict[str, Tuple[str, float]] = {}

        # Dead letter queue storage
        self._dead_letter: Dict[str, DistributedTask] = {}

        # Concurrency lock for thread-safe heap operations
        self._lock = asyncio.Lock()

        # Queue statistics counters
        self._stats = {
            "enqueued": 0,
            "dequeued": 0,
            "failed": 0,
            "deduplicated": 0,
            "expired": 0,
            "dead_lettered": 0,
        }

        logger.info(
            "distributed_task_queue_initialized",
            max_queue_size=max_queue_size,
            task_ttl_seconds=task_ttl_seconds,
            dead_letter_enabled=dead_letter_enabled,
        )

    # -------------------------------------------------------------------------
    # Core Operations
    # -------------------------------------------------------------------------

    async def enqueue(self, task: DistributedTask) -> bool:
        """
        Add a task to the priority queue.

        Performs deduplication check if the task has an idempotency key.
        Rejects the task if the queue is at capacity.

        Args:
            task: The distributed task to enqueue.

        Returns:
            True if the task was successfully enqueued, False otherwise.
        """
        async with self._lock:
            # Capacity check
            if len(self._tasks) >= self.max_queue_size:
                logger.warning(
                    "task_queue_full",
                    task_id=task.id,
                    queue_size=len(self._tasks),
                    max_size=self.max_queue_size,
                )
                return False

            # Deduplication check via idempotency key
            if task.idempotency_key:
                existing = self._idempotency_index.get(task.idempotency_key)
                if existing is not None:
                    existing_id, expiry = existing
                    if time.time() < expiry and existing_id in self._tasks:
                        logger.debug(
                            "task_deduplicated",
                            task_id=task.id,
                            idempotency_key=task.idempotency_key,
                            existing_task_id=existing_id,
                        )
                        self._stats["deduplicated"] += 1
                        return False

                # Register the idempotency key with expiry
                expiry_ts = time.time() + self.deduplication_window_seconds
                self._idempotency_index[task.idempotency_key] = (task.id, expiry_ts)

            # Set task to queued status
            task.status = TaskStatus.QUEUED

            # Store the task
            self._tasks[task.id] = task

            # Push onto the priority heap
            created_ts = task.created_at.timestamp()
            heapq.heappush(self._heap, (task.priority.value, created_ts, task.id))

            # Track node assignment if already assigned
            if task.assigned_node:
                self._tasks_by_node[task.assigned_node].add(task.id)

            self._stats["enqueued"] += 1

            logger.debug(
                "task_enqueued",
                task_id=task.id,
                task_name=task.name,
                priority=task.priority.name,
                queue_size=len(self._tasks),
            )
            return True

    async def dequeue(self) -> Optional[DistributedTask]:
        """
        Remove and return the highest-priority task from the queue.

        Skips tasks that have been removed, are expired, or are no longer
        in a queued state. Expired tasks are moved to dead letter if enabled.

        Returns:
            The highest-priority task, or None if the queue is empty.
        """
        async with self._lock:
            while self._heap:
                priority_val, created_ts, task_id = heapq.heappop(self._heap)

                # Skip if the task was already removed
                task = self._tasks.get(task_id)
                if task is None:
                    continue

                # Skip if no longer in a queued state (already picked up)
                if task.status != TaskStatus.QUEUED:
                    continue

                # Enforce TTL expiration
                age_seconds = (datetime.now() - task.created_at).total_seconds()
                if age_seconds > self.task_ttl_seconds:
                    await self._expire_task(task)
                    continue

                # Enforce deadline expiration
                if task.is_expired:
                    await self._expire_task(task)
                    continue

                self._stats["dequeued"] += 1

                logger.debug(
                    "task_dequeued",
                    task_id=task.id,
                    task_name=task.name,
                    priority=task.priority.name,
                )
                return task

            return None

    async def get_task(self, task_id: str) -> Optional[DistributedTask]:
        """
        Retrieve a task by its ID without removing it from the queue.

        Args:
            task_id: The unique identifier of the task.

        Returns:
            The task if found, None otherwise.
        """
        return self._tasks.get(task_id)

    async def update_task(self, task: DistributedTask) -> None:
        """
        Update an existing task in the queue.

        Maintains node index consistency when the assigned node changes.

        Args:
            task: The task with updated fields.
        """
        async with self._lock:
            old_task = self._tasks.get(task.id)
            if old_task is None:
                logger.warning("task_update_not_found", task_id=task.id)
                return

            # Update node index if assignment changed
            old_node = old_task.assigned_node
            new_node = task.assigned_node

            if old_node != new_node:
                if old_node and task.id in self._tasks_by_node.get(old_node, set()):
                    self._tasks_by_node[old_node].discard(task.id)
                if new_node:
                    self._tasks_by_node[new_node].add(task.id)

            # Track failure statistics
            if task.status == TaskStatus.FAILED:
                self._stats["failed"] += 1

            self._tasks[task.id] = task

            logger.debug(
                "task_updated",
                task_id=task.id,
                status=task.status.value,
                assigned_node=task.assigned_node,
            )

    async def get_tasks_by_node(self, node_id: str) -> List[DistributedTask]:
        """
        Get all tasks assigned to a specific node.

        Args:
            node_id: The node identifier.

        Returns:
            List of tasks assigned to the node.
        """
        task_ids = self._tasks_by_node.get(node_id, set())
        return [
            self._tasks[tid]
            for tid in task_ids
            if tid in self._tasks
        ]

    async def get_all(self) -> List[DistributedTask]:
        """
        Return all tasks currently in the queue.

        Returns:
            List of all tracked tasks (including non-queued states).
        """
        return list(self._tasks.values())

    async def remove_task(self, task_id: str) -> Optional[DistributedTask]:
        """
        Remove a task from the queue entirely.

        The task is removed from all indexes. The heap entry becomes a
        no-op that is skipped on the next dequeue.

        Args:
            task_id: The task to remove.

        Returns:
            The removed task if found, None otherwise.
        """
        async with self._lock:
            task = self._tasks.pop(task_id, None)
            if task is None:
                return None

            # Clean up node index
            if task.assigned_node:
                self._tasks_by_node[task.assigned_node].discard(task_id)

            # Clean up idempotency index
            if task.idempotency_key and task.idempotency_key in self._idempotency_index:
                stored_id, _ = self._idempotency_index[task.idempotency_key]
                if stored_id == task_id:
                    del self._idempotency_index[task.idempotency_key]

            logger.debug("task_removed", task_id=task_id)
            return task

    # -------------------------------------------------------------------------
    # Query and Statistics
    # -------------------------------------------------------------------------

    async def size(self) -> int:
        """Return the total number of tasks in the queue."""
        return len(self._tasks)

    async def pending_count(self) -> int:
        """Return the number of tasks in a queued or pending state."""
        return sum(
            1
            for t in self._tasks.values()
            if t.status in (TaskStatus.PENDING, TaskStatus.QUEUED)
        )

    async def get_stats(self) -> Dict[str, Any]:
        """
        Return comprehensive queue statistics.

        Returns:
            Dictionary containing queue size, state counts, and
            cumulative operation counters.
        """
        status_counts: Dict[str, int] = defaultdict(int)
        for task in self._tasks.values():
            status_counts[task.status.value] += 1

        return {
            "total_tasks": len(self._tasks),
            "heap_size": len(self._heap),
            "dead_letter_size": len(self._dead_letter),
            "status_counts": dict(status_counts),
            "counters": dict(self._stats),
            "idempotency_keys_tracked": len(self._idempotency_index),
            "nodes_with_tasks": len(self._tasks_by_node),
        }

    # -------------------------------------------------------------------------
    # Dead Letter Queue
    # -------------------------------------------------------------------------

    async def get_dead_letter_tasks(self) -> List[DistributedTask]:
        """Return all tasks in the dead letter queue."""
        return list(self._dead_letter.values())

    async def requeue_dead_letter(self, task_id: str) -> bool:
        """
        Move a task from the dead letter queue back to the main queue.

        Resets the retry count and status so the task gets another chance.

        Args:
            task_id: The dead letter task to requeue.

        Returns:
            True if the task was successfully requeued.
        """
        task = self._dead_letter.pop(task_id, None)
        if task is None:
            return False

        task.retry_count = 0
        task.status = TaskStatus.PENDING
        task.error = None
        task.error_traceback = None

        return await self.enqueue(task)

    # -------------------------------------------------------------------------
    # Internal Helpers
    # -------------------------------------------------------------------------

    async def _expire_task(self, task: DistributedTask) -> None:
        """
        Handle expiration of a task that has exceeded its TTL or deadline.

        Moves the task to the dead letter queue if enabled, otherwise
        marks it as failed and removes it.

        Args:
            task: The expired task.
        """
        self._stats["expired"] += 1

        logger.info(
            "task_expired",
            task_id=task.id,
            task_name=task.name,
            age_seconds=(datetime.now() - task.created_at).total_seconds(),
        )

        if self.dead_letter_enabled:
            await self._move_to_dead_letter(task)
        else:
            task.status = TaskStatus.TIMEOUT
            task.completed_at = datetime.now()
            self._tasks.pop(task.id, None)

    async def _move_to_dead_letter(self, task: DistributedTask) -> None:
        """
        Move a task to the dead letter queue.

        Evicts the oldest dead letter entry if the DLQ is at capacity.

        Args:
            task: The task to move to dead letter.
        """
        task.status = TaskStatus.DEAD_LETTER
        task.completed_at = datetime.now()

        # Remove from main storage
        self._tasks.pop(task.id, None)
        if task.assigned_node:
            self._tasks_by_node[task.assigned_node].discard(task.id)

        # Enforce DLQ capacity by evicting the oldest entry
        if len(self._dead_letter) >= self.dead_letter_max_size:
            oldest_key = next(iter(self._dead_letter))
            del self._dead_letter[oldest_key]

        self._dead_letter[task.id] = task
        self._stats["dead_lettered"] += 1

        logger.warning(
            "task_moved_to_dead_letter",
            task_id=task.id,
            task_name=task.name,
            retry_count=task.retry_count,
            max_retries=task.max_retries,
        )

    async def cleanup_expired_idempotency_keys(self) -> int:
        """
        Purge expired entries from the idempotency deduplication index.

        Should be called periodically to prevent unbounded memory growth.

        Returns:
            The number of expired keys removed.
        """
        now = time.time()
        expired_keys = [
            key
            for key, (_, expiry) in self._idempotency_index.items()
            if now >= expiry
        ]
        for key in expired_keys:
            del self._idempotency_index[key]

        if expired_keys:
            logger.debug(
                "idempotency_keys_cleaned",
                removed_count=len(expired_keys),
            )
        return len(expired_keys)
