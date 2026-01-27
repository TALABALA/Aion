"""
AION Recovery Manager

Production-grade recovery orchestration implementing:
- Structured recovery plans with step-by-step progress tracking
- Task recovery: reassign orphaned tasks from failed nodes
- Data recovery: re-replicate under-replicated shards to healthy nodes
- Grace period before triggering recovery (transient failure tolerance)
- Concurrent recovery limiter to prevent recovery storms / cascades
- Exponential back-off for retry attempts during recovery steps
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

import structlog

from aion.distributed.types import (
    DistributedTask,
    NodeStatus,
    ShardInfo,
    TaskStatus,
)

if TYPE_CHECKING:
    from aion.distributed.cluster.manager import ClusterManager

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Recovery plan data structures
# ---------------------------------------------------------------------------


class RecoveryStepStatus(str, Enum):
    """Status of an individual recovery step."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class RecoveryPlanStatus(str, Enum):
    """Overall status of a recovery plan."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    PARTIALLY_COMPLETED = "partially_completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class RecoveryStep:
    """A single step within a recovery plan.

    Each step represents a discrete, retriable unit of work such as
    reassigning a task or re-replicating a shard.
    """

    step_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    status: RecoveryStepStatus = RecoveryStepStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    retries: int = 0
    max_retries: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecoveryPlan:
    """Full recovery plan for a failed node.

    Aggregates multiple :class:`RecoveryStep` instances and tracks
    overall progress, timing, and the identity of the failed node.
    """

    plan_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    node_id: str = ""
    status: RecoveryPlanStatus = RecoveryPlanStatus.PENDING
    steps: List[RecoveryStep] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None

    @property
    def progress(self) -> float:
        """Fraction of steps completed (0.0 -- 1.0)."""
        if not self.steps:
            return 0.0
        done = sum(
            1
            for s in self.steps
            if s.status in (RecoveryStepStatus.COMPLETED, RecoveryStepStatus.SKIPPED)
        )
        return done / len(self.steps)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "node_id": self.node_id,
            "status": self.status.value,
            "progress": round(self.progress, 4),
            "total_steps": len(self.steps),
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": (
                self.completed_at.isoformat() if self.completed_at else None
            ),
            "error": self.error,
            "steps": [
                {
                    "step_id": s.step_id,
                    "name": s.name,
                    "status": s.status.value,
                    "retries": s.retries,
                    "error": s.error,
                }
                for s in self.steps
            ],
        }


# ---------------------------------------------------------------------------
# RecoveryManager
# ---------------------------------------------------------------------------


class RecoveryManager:
    """Orchestrates recovery procedures after node failures.

    The manager provides three levels of recovery:

    1. **Task recovery** -- orphaned tasks on the failed node are
       reassigned to healthy nodes that satisfy capability requirements.
    2. **Data recovery** -- under-replicated shards previously hosted on
       the failed node are re-replicated to maintain the configured
       replication factor.
    3. **Full recovery plan** -- a structured, step-by-step plan that
       sequences task and data recovery with progress tracking.

    Recovery is deliberately delayed by a configurable *grace period* to
    tolerate transient failures (e.g. a node rebooting).  A concurrent
    recovery limiter prevents recovery storms that could cascade into
    further failures.

    Args:
        cluster_manager: The :class:`ClusterManager` that owns this
                         recovery manager.
        grace_period: Seconds to wait before starting recovery for a
                      failed node.  Default ``10.0``.
        max_concurrent_recoveries: Maximum number of recovery plans that
                                   may execute simultaneously.  Default ``3``.
    """

    def __init__(
        self,
        cluster_manager: ClusterManager,
        grace_period: float = 10.0,
        max_concurrent_recoveries: int = 3,
    ) -> None:
        self._cluster_manager = cluster_manager
        self._grace_period = grace_period
        self._max_concurrent = max_concurrent_recoveries

        # Active / completed recovery plans keyed by plan_id
        self._plans: Dict[str, RecoveryPlan] = {}

        # Nodes currently being recovered (prevents duplicate plans)
        self._recovering_nodes: Set[str] = {}

        # Semaphore for concurrent recovery limiting
        self._semaphore = asyncio.Semaphore(max_concurrent_recoveries)

        # Grace-period timers keyed by node_id
        self._grace_timers: Dict[str, asyncio.Task[None]] = {}

        logger.info(
            "recovery_manager.init",
            grace_period=grace_period,
            max_concurrent=max_concurrent_recoveries,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def start_recovery_plan(self, node_id: str) -> str:
        """Create and execute a full recovery plan for *node_id*.

        A grace period elapses before recovery begins.  If the node
        comes back within the grace period the plan is cancelled.

        Returns:
            The ``plan_id`` of the newly created recovery plan.
        """
        if node_id in self._recovering_nodes:
            logger.warning(
                "recovery_manager.already_recovering",
                node_id=node_id,
            )
            # Return existing plan id
            for plan in self._plans.values():
                if plan.node_id == node_id and plan.status in (
                    RecoveryPlanStatus.PENDING,
                    RecoveryPlanStatus.IN_PROGRESS,
                ):
                    return plan.plan_id
            return ""

        plan = RecoveryPlan(node_id=node_id)
        self._plans[plan.plan_id] = plan
        self._recovering_nodes.add(node_id)

        logger.info(
            "recovery_manager.plan_created",
            plan_id=plan.plan_id,
            node_id=node_id,
        )

        # Schedule the actual recovery after the grace period
        timer = asyncio.create_task(
            self._grace_then_execute(plan)
        )
        self._grace_timers[node_id] = timer

        return plan.plan_id

    async def cancel_recovery(self, node_id: str) -> bool:
        """Cancel an in-progress or pending recovery for *node_id*.

        Typically called when the node rejoins the cluster during the
        grace period.

        Returns:
            ``True`` if a plan was successfully cancelled.
        """
        timer = self._grace_timers.pop(node_id, None)
        if timer is not None and not timer.done():
            timer.cancel()

        cancelled = False
        for plan in self._plans.values():
            if plan.node_id == node_id and plan.status in (
                RecoveryPlanStatus.PENDING,
                RecoveryPlanStatus.IN_PROGRESS,
            ):
                plan.status = RecoveryPlanStatus.CANCELLED
                plan.completed_at = datetime.now()
                cancelled = True

        self._recovering_nodes.discard(node_id)
        if cancelled:
            logger.info("recovery_manager.plan_cancelled", node_id=node_id)
        return cancelled

    async def recover_node(self, node_id: str) -> Dict[str, Any]:
        """Execute immediate full recovery for *node_id* (no grace period).

        Returns:
            A summary dict of the recovery outcome.
        """
        task_result = await self.recover_tasks(node_id)
        data_result = await self.recover_data(node_id)
        return {
            "node_id": node_id,
            "tasks": task_result,
            "data": data_result,
        }

    async def recover_tasks(self, node_id: str) -> Dict[str, Any]:
        """Reassign all tasks from a failed node to healthy peers.

        Returns:
            Summary with counts of reassigned and failed-to-reassign tasks.
        """
        task_queue = getattr(self._cluster_manager, "_task_queue", None)
        reassigned = 0
        failed = 0

        if task_queue is not None:
            orphaned: List[DistributedTask] = await task_queue.get_tasks_for_node(
                node_id
            )
            for task in orphaned:
                if task.is_terminal:
                    continue
                try:
                    task.assigned_node = None
                    task.status = TaskStatus.PENDING
                    task.excluded_nodes.add(node_id)
                    await self._cluster_manager.submit_task(task)
                    reassigned += 1
                    logger.debug(
                        "recovery_manager.task_reassigned",
                        task_id=task.id,
                        old_node=node_id,
                    )
                except Exception as exc:
                    failed += 1
                    logger.error(
                        "recovery_manager.task_reassign_failed",
                        task_id=task.id,
                        error=str(exc),
                    )
        else:
            logger.warning("recovery_manager.no_task_queue")

        result = {
            "node_id": node_id,
            "reassigned": reassigned,
            "failed": failed,
        }
        logger.info("recovery_manager.tasks_recovered", **result)
        return result

    async def recover_data(self, node_id: str) -> Dict[str, Any]:
        """Re-replicate under-replicated shards after losing *node_id*.

        Iterates all known shards to find those that included *node_id*
        as primary or replica, then triggers re-replication to healthy
        nodes to restore the configured replication factor.

        Returns:
            Summary with lists of repaired shard IDs and failures.
        """
        state = self._cluster_manager.state
        replication_factor = state.replication_factor
        repaired: List[str] = []
        failed_shards: List[str] = []

        # Collect shards that reference the failed node
        shards = self._get_shards_for_node(node_id)

        for shard in shards:
            try:
                # Remove failed node from the shard's replica set
                if shard.primary_node == node_id:
                    # Promote a replica to primary
                    if shard.replica_nodes:
                        shard.primary_node = shard.replica_nodes.pop(0)
                    else:
                        shard.primary_node = ""
                else:
                    shard.replica_nodes = [
                        r for r in shard.replica_nodes if r != node_id
                    ]

                # Determine how many new replicas are needed
                current_copies = (
                    (1 if shard.primary_node else 0) + len(shard.replica_nodes)
                )
                deficit = max(0, replication_factor - current_copies)

                if deficit > 0:
                    new_targets = self._select_replication_targets(
                        shard, deficit, exclude={node_id}
                    )
                    shard.replica_nodes.extend(new_targets)

                    logger.info(
                        "recovery_manager.shard_repaired",
                        shard_id=shard.shard_id,
                        new_replicas=new_targets,
                    )

                repaired.append(shard.shard_id)
            except Exception as exc:
                failed_shards.append(shard.shard_id)
                logger.error(
                    "recovery_manager.shard_repair_failed",
                    shard_id=shard.shard_id,
                    error=str(exc),
                )

        result = {
            "node_id": node_id,
            "repaired": repaired,
            "failed": failed_shards,
            "total_shards": len(shards),
        }
        logger.info("recovery_manager.data_recovered", **result)
        return result

    def get_recovery_status(self, recovery_id: str) -> Dict[str, Any]:
        """Return the current status of a recovery plan.

        Args:
            recovery_id: The ``plan_id`` returned by
                         :meth:`start_recovery_plan`.

        Returns:
            A dict representation of the plan, or an error dict if the
            plan is not found.
        """
        plan = self._plans.get(recovery_id)
        if plan is None:
            return {"error": "recovery_plan_not_found", "plan_id": recovery_id}
        return plan.to_dict()

    def get_all_recovery_statuses(self) -> Dict[str, Dict[str, Any]]:
        """Return statuses for all known recovery plans."""
        return {pid: plan.to_dict() for pid, plan in self._plans.items()}

    # ------------------------------------------------------------------
    # Grace period + execution
    # ------------------------------------------------------------------

    async def _grace_then_execute(self, plan: RecoveryPlan) -> None:
        """Wait for the grace period, then execute the recovery plan."""
        node_id = plan.node_id
        try:
            logger.info(
                "recovery_manager.grace_period_start",
                node_id=node_id,
                seconds=self._grace_period,
            )
            await asyncio.sleep(self._grace_period)

            # Check if the node came back during the grace period
            node = self._cluster_manager.state.nodes.get(node_id)
            if node is not None and node.status == NodeStatus.HEALTHY:
                plan.status = RecoveryPlanStatus.CANCELLED
                plan.completed_at = datetime.now()
                self._recovering_nodes.discard(node_id)
                logger.info(
                    "recovery_manager.node_recovered_during_grace",
                    node_id=node_id,
                )
                return

            # Execute under the concurrency limiter
            async with self._semaphore:
                await self._execute_plan(plan)

        except asyncio.CancelledError:
            plan.status = RecoveryPlanStatus.CANCELLED
            plan.completed_at = datetime.now()
            logger.info(
                "recovery_manager.plan_cancelled_during_grace",
                node_id=node_id,
            )
        except Exception:
            plan.status = RecoveryPlanStatus.FAILED
            plan.completed_at = datetime.now()
            logger.exception(
                "recovery_manager.plan_execution_failed",
                node_id=node_id,
            )
        finally:
            self._recovering_nodes.discard(node_id)
            self._grace_timers.pop(node_id, None)

    async def _execute_plan(self, plan: RecoveryPlan) -> None:
        """Build and execute the steps of a recovery plan."""
        plan.status = RecoveryPlanStatus.IN_PROGRESS
        plan.started_at = datetime.now()

        # Build steps
        plan.steps = [
            RecoveryStep(
                name="task_recovery",
                description=f"Reassign orphaned tasks from node {plan.node_id}",
                metadata={"node_id": plan.node_id},
            ),
            RecoveryStep(
                name="data_recovery",
                description=f"Re-replicate under-replicated shards from node {plan.node_id}",
                metadata={"node_id": plan.node_id},
            ),
            RecoveryStep(
                name="cluster_state_update",
                description=f"Update cluster state to reflect loss of node {plan.node_id}",
                metadata={"node_id": plan.node_id},
            ),
        ]

        # Execute each step sequentially
        all_ok = True
        for step in plan.steps:
            step.status = RecoveryStepStatus.IN_PROGRESS
            step.started_at = datetime.now()

            success = await self._execute_step(step, plan.node_id)
            if success:
                step.status = RecoveryStepStatus.COMPLETED
            else:
                step.status = RecoveryStepStatus.FAILED
                all_ok = False
            step.completed_at = datetime.now()

        plan.completed_at = datetime.now()
        if all_ok:
            plan.status = RecoveryPlanStatus.COMPLETED
        elif any(s.status == RecoveryStepStatus.COMPLETED for s in plan.steps):
            plan.status = RecoveryPlanStatus.PARTIALLY_COMPLETED
        else:
            plan.status = RecoveryPlanStatus.FAILED

        logger.info(
            "recovery_manager.plan_completed",
            plan_id=plan.plan_id,
            status=plan.status.value,
            progress=plan.progress,
        )

    async def _execute_step(self, step: RecoveryStep, node_id: str) -> bool:
        """Execute a single recovery step with retries and back-off."""
        for attempt in range(step.max_retries + 1):
            step.retries = attempt
            try:
                if step.name == "task_recovery":
                    await self.recover_tasks(node_id)
                elif step.name == "data_recovery":
                    await self.recover_data(node_id)
                elif step.name == "cluster_state_update":
                    await self._update_cluster_state(node_id)
                else:
                    logger.warning(
                        "recovery_manager.unknown_step",
                        step=step.name,
                    )
                    step.status = RecoveryStepStatus.SKIPPED
                return True
            except Exception as exc:
                step.error = str(exc)
                logger.warning(
                    "recovery_manager.step_attempt_failed",
                    step=step.name,
                    attempt=attempt,
                    error=str(exc),
                )
                if attempt < step.max_retries:
                    backoff = min(2 ** attempt, 30)
                    await asyncio.sleep(backoff)
        return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _update_cluster_state(self, node_id: str) -> None:
        """Update the cluster state after removing a failed node."""
        state = self._cluster_manager.state
        if node_id in state.nodes:
            node = state.nodes[node_id]
            node.status = NodeStatus.OFFLINE
        state.increment_epoch()
        logger.info(
            "recovery_manager.cluster_state_updated",
            node_id=node_id,
            epoch=state.epoch,
        )

    def _get_shards_for_node(self, node_id: str) -> List[ShardInfo]:
        """Retrieve all shards where *node_id* is primary or replica.

        Scans the cluster state for shard metadata.  Returns an empty
        list if no shard registry is available.
        """
        state = self._cluster_manager.state
        shard_registry: Dict[str, ShardInfo] = getattr(state, "shards", {})
        result: List[ShardInfo] = []
        for shard in shard_registry.values():
            if shard.primary_node == node_id or node_id in shard.replica_nodes:
                result.append(shard)
        return result

    def _select_replication_targets(
        self,
        shard: ShardInfo,
        count: int,
        exclude: Optional[Set[str]] = None,
    ) -> List[str]:
        """Choose healthy nodes as new replica targets for a shard.

        Prefers nodes with the lowest load score that are not already
        hosting the shard and are not in the *exclude* set.  Returns up
        to *count* node IDs.
        """
        exclude = exclude or set()
        existing = {shard.primary_node} | set(shard.replica_nodes) | exclude
        state = self._cluster_manager.state

        candidates = [
            node
            for nid, node in state.nodes.items()
            if nid not in existing and node.status == NodeStatus.HEALTHY
        ]

        # Sort by load score ascending (prefer least loaded)
        candidates.sort(key=lambda n: n.load_score)
        return [c.id for c in candidates[:count]]
