"""
AION DAG-based Task Scheduler

Dependency-aware scheduling with:
- Directed Acyclic Graph (DAG) task definitions
- Topological execution ordering
- Parallel execution of independent tasks
- Fair-share scheduling between groups
- Priority inheritance
- Deadline scheduling
- Dynamic dependency resolution
"""

from __future__ import annotations

import asyncio
import heapq
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import structlog

from aion.systems.process.models import (
    ProcessPriority,
    ResourceLimits,
    ProcessState,
)

logger = structlog.get_logger(__name__)


class TaskStatus(Enum):
    """Status of a DAG task."""
    PENDING = auto()
    WAITING = auto()      # Waiting for dependencies
    READY = auto()        # Dependencies satisfied
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    SKIPPED = auto()
    CANCELLED = auto()


class TriggerRule(Enum):
    """Rules for when a task can be triggered."""
    ALL_SUCCESS = auto()      # All upstream tasks succeeded
    ALL_FAILED = auto()       # All upstream tasks failed
    ALL_DONE = auto()         # All upstream tasks completed (any status)
    ONE_SUCCESS = auto()      # At least one upstream succeeded
    ONE_FAILED = auto()       # At least one upstream failed
    NONE_FAILED = auto()      # No upstream tasks failed
    NONE_SKIPPED = auto()     # No upstream tasks skipped


@dataclass
class DAGTask:
    """A task in a DAG workflow."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    handler: str = ""
    params: Dict[str, Any] = field(default_factory=dict)

    # Dependencies
    upstream_tasks: Set[str] = field(default_factory=set)
    downstream_tasks: Set[str] = field(default_factory=set)
    trigger_rule: TriggerRule = TriggerRule.ALL_SUCCESS

    # Execution
    status: TaskStatus = TaskStatus.PENDING
    priority: ProcessPriority = ProcessPriority.NORMAL
    retry_count: int = 0
    max_retries: int = 3
    retry_delay_seconds: float = 30.0
    timeout_seconds: float = 3600.0

    # Scheduling
    deadline: Optional[datetime] = None
    earliest_start: Optional[datetime] = None
    weight: float = 1.0  # For fair-share scheduling

    # Resource requirements
    resource_requirements: Dict[str, float] = field(default_factory=dict)
    limits: Optional[ResourceLimits] = None

    # Execution tracking
    run_id: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: float = 0.0
    result: Any = None
    error: Optional[str] = None
    attempt: int = 0

    # Fair-share tracking
    group: Optional[str] = None
    accumulated_runtime: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "handler": self.handler,
            "status": self.status.name,
            "priority": self.priority.value,
            "upstream_tasks": list(self.upstream_tasks),
            "downstream_tasks": list(self.downstream_tasks),
            "trigger_rule": self.trigger_rule.name,
            "retry_count": self.retry_count,
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
            "error": self.error,
            "group": self.group,
        }


@dataclass
class DAGRun:
    """An execution instance of a DAG."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    dag_id: str = ""
    state: str = "running"  # running, success, failed, cancelled
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    task_states: Dict[str, TaskStatus] = field(default_factory=dict)
    execution_order: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DAGDefinition:
    """Definition of a DAG workflow."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    tasks: Dict[str, DAGTask] = field(default_factory=dict)
    default_priority: ProcessPriority = ProcessPriority.NORMAL
    max_active_runs: int = 1
    concurrency: int = 16  # Max parallel tasks
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

    def add_task(self, task: DAGTask) -> None:
        """Add a task to the DAG."""
        self.tasks[task.id] = task

    def set_dependency(self, upstream_id: str, downstream_id: str) -> None:
        """Set a dependency between tasks."""
        if upstream_id in self.tasks and downstream_id in self.tasks:
            self.tasks[upstream_id].downstream_tasks.add(downstream_id)
            self.tasks[downstream_id].upstream_tasks.add(upstream_id)

    def validate(self) -> List[str]:
        """Validate the DAG structure."""
        errors = []

        # Check for cycles
        if self._has_cycle():
            errors.append("DAG contains a cycle")

        # Check for missing dependencies
        for task_id, task in self.tasks.items():
            for dep_id in task.upstream_tasks:
                if dep_id not in self.tasks:
                    errors.append(f"Task {task_id} has missing dependency: {dep_id}")

        return errors

    def _has_cycle(self) -> bool:
        """Detect cycles using DFS."""
        WHITE, GRAY, BLACK = 0, 1, 2
        colors = {task_id: WHITE for task_id in self.tasks}

        def dfs(task_id: str) -> bool:
            colors[task_id] = GRAY
            for downstream in self.tasks[task_id].downstream_tasks:
                if colors[downstream] == GRAY:
                    return True
                if colors[downstream] == WHITE and dfs(downstream):
                    return True
            colors[task_id] = BLACK
            return False

        for task_id in self.tasks:
            if colors[task_id] == WHITE:
                if dfs(task_id):
                    return True
        return False

    def get_topological_order(self) -> List[str]:
        """Get topological ordering of tasks."""
        in_degree = {task_id: len(task.upstream_tasks)
                     for task_id, task in self.tasks.items()}
        queue = deque([tid for tid, deg in in_degree.items() if deg == 0])
        order = []

        while queue:
            task_id = queue.popleft()
            order.append(task_id)

            for downstream in self.tasks[task_id].downstream_tasks:
                in_degree[downstream] -= 1
                if in_degree[downstream] == 0:
                    queue.append(downstream)

        return order

    def get_root_tasks(self) -> List[str]:
        """Get tasks with no upstream dependencies."""
        return [tid for tid, task in self.tasks.items()
                if not task.upstream_tasks]

    def get_leaf_tasks(self) -> List[str]:
        """Get tasks with no downstream dependencies."""
        return [tid for tid, task in self.tasks.items()
                if not task.downstream_tasks]


@dataclass
class FairShareGroup:
    """Group for fair-share scheduling."""
    id: str
    name: str
    weight: float = 1.0
    min_share: float = 0.0  # Minimum guaranteed share
    max_share: float = 1.0  # Maximum allowed share
    accumulated_deficit: float = 0.0
    running_tasks: int = 0
    queued_tasks: int = 0
    total_runtime: float = 0.0
    last_scheduled: Optional[datetime] = None


class DAGScheduler:
    """
    DAG-based task scheduler with fair-share support.

    Features:
    - Topological execution ordering
    - Parallel execution of independent tasks
    - Fair-share scheduling between groups
    - Priority inheritance from critical path
    - Deadline-based scheduling
    - Dynamic dependency resolution
    """

    def __init__(
        self,
        max_concurrent_tasks: int = 16,
        enable_fair_share: bool = True,
        fair_share_interval: float = 1.0,
        priority_inheritance: bool = True,
    ):
        self.max_concurrent_tasks = max_concurrent_tasks
        self.enable_fair_share = enable_fair_share
        self.fair_share_interval = fair_share_interval
        self.priority_inheritance = priority_inheritance

        # DAG storage
        self._dags: Dict[str, DAGDefinition] = {}
        self._runs: Dict[str, DAGRun] = {}
        self._active_runs: Dict[str, Set[str]] = defaultdict(set)  # dag_id -> run_ids

        # Task queues
        self._ready_queue: List[Tuple[float, str, str]] = []  # (priority, run_id, task_id)
        self._running_tasks: Dict[str, Tuple[str, str]] = {}  # task_key -> (run_id, task_id)
        self._task_futures: Dict[str, asyncio.Task] = {}

        # Fair-share groups
        self._groups: Dict[str, FairShareGroup] = {}
        self._default_group = FairShareGroup(id="default", name="Default", weight=1.0)
        self._groups["default"] = self._default_group

        # Handlers
        self._handlers: Dict[str, Callable] = {}

        # Background tasks
        self._scheduler_task: Optional[asyncio.Task] = None
        self._fair_share_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

        # Statistics
        self._stats = {
            "dags_registered": 0,
            "runs_started": 0,
            "runs_completed": 0,
            "runs_failed": 0,
            "tasks_executed": 0,
            "tasks_succeeded": 0,
            "tasks_failed": 0,
            "tasks_retried": 0,
        }

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the DAG scheduler."""
        if self._initialized:
            return

        logger.info("Initializing DAG Scheduler")

        self._scheduler_task = asyncio.create_task(self._scheduler_loop())

        if self.enable_fair_share:
            self._fair_share_task = asyncio.create_task(self._fair_share_loop())

        self._initialized = True
        logger.info("DAG Scheduler initialized")

    async def shutdown(self) -> None:
        """Shutdown the scheduler."""
        logger.info("Shutting down DAG Scheduler")

        self._shutdown_event.set()

        for task in [self._scheduler_task, self._fair_share_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Cancel running tasks
        for future in self._task_futures.values():
            future.cancel()

        logger.info("DAG Scheduler shutdown complete")

    def register_handler(self, name: str, handler: Callable) -> None:
        """Register a task handler."""
        self._handlers[name] = handler

    def register_dag(self, dag: DAGDefinition) -> bool:
        """Register a DAG definition."""
        errors = dag.validate()
        if errors:
            logger.error(f"DAG validation failed: {errors}")
            return False

        self._dags[dag.id] = dag
        self._stats["dags_registered"] += 1

        # Apply priority inheritance
        if self.priority_inheritance:
            self._apply_priority_inheritance(dag)

        logger.info(f"Registered DAG: {dag.name} ({dag.id})")
        return True

    def _apply_priority_inheritance(self, dag: DAGDefinition) -> None:
        """Apply priority inheritance from downstream critical tasks."""
        # Process in reverse topological order
        order = dag.get_topological_order()

        for task_id in reversed(order):
            task = dag.tasks[task_id]

            # Inherit highest priority from downstream
            for downstream_id in task.downstream_tasks:
                downstream = dag.tasks[downstream_id]
                if downstream.priority.value > task.priority.value:
                    task.priority = downstream.priority

            # Inherit deadline from downstream
            for downstream_id in task.downstream_tasks:
                downstream = dag.tasks[downstream_id]
                if downstream.deadline and (not task.deadline or downstream.deadline < task.deadline):
                    # Account for our expected duration
                    task.deadline = downstream.deadline - timedelta(seconds=task.timeout_seconds)

    def create_group(self, group: FairShareGroup) -> None:
        """Create a fair-share group."""
        self._groups[group.id] = group
        logger.info(f"Created fair-share group: {group.name}")

    async def trigger_dag(
        self,
        dag_id: str,
        context: Optional[Dict[str, Any]] = None,
        priority: Optional[ProcessPriority] = None,
    ) -> Optional[str]:
        """
        Trigger a DAG execution.

        Returns run ID or None if failed.
        """
        dag = self._dags.get(dag_id)
        if not dag:
            logger.error(f"DAG not found: {dag_id}")
            return None

        # Check max active runs
        if len(self._active_runs[dag_id]) >= dag.max_active_runs:
            logger.warning(f"Max active runs reached for DAG: {dag_id}")
            return None

        # Create run
        run = DAGRun(
            dag_id=dag_id,
            context=context or {},
        )

        # Initialize task states
        for task_id, task in dag.tasks.items():
            run.task_states[task_id] = TaskStatus.PENDING

            # Clone task for this run with overridden priority
            if priority:
                task.priority = priority

        self._runs[run.id] = run
        self._active_runs[dag_id].add(run.id)
        self._stats["runs_started"] += 1

        # Queue root tasks
        for task_id in dag.get_root_tasks():
            self._queue_task(run.id, task_id, dag.tasks[task_id])

        logger.info(f"Triggered DAG run: {dag.name} ({run.id})")
        return run.id

    def _queue_task(self, run_id: str, task_id: str, task: DAGTask) -> None:
        """Add task to ready queue."""
        run = self._runs.get(run_id)
        if not run:
            return

        run.task_states[task_id] = TaskStatus.READY

        # Calculate effective priority
        # Lower value = higher priority
        priority_score = -task.priority.value * 1000

        # Deadline-based priority boost
        if task.deadline:
            time_to_deadline = (task.deadline - datetime.now()).total_seconds()
            if time_to_deadline < task.timeout_seconds * 2:
                priority_score -= 500  # Urgent boost

        # Fair-share adjustment
        if self.enable_fair_share and task.group:
            group = self._groups.get(task.group, self._default_group)
            # Higher deficit = higher priority
            priority_score -= group.accumulated_deficit * 100

        heapq.heappush(self._ready_queue, (priority_score, run_id, task_id))

        # Update group stats
        if task.group:
            group = self._groups.get(task.group, self._default_group)
            group.queued_tasks += 1

    async def _scheduler_loop(self) -> None:
        """Main scheduler loop."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(0.01)

                # Process ready queue
                while (self._ready_queue and
                       len(self._running_tasks) < self.max_concurrent_tasks):

                    _, run_id, task_id = heapq.heappop(self._ready_queue)

                    run = self._runs.get(run_id)
                    if not run or run.state != "running":
                        continue

                    dag = self._dags.get(run.dag_id)
                    if not dag:
                        continue

                    task = dag.tasks.get(task_id)
                    if not task:
                        continue

                    # Check if task is still ready
                    if run.task_states.get(task_id) != TaskStatus.READY:
                        continue

                    # Check dependencies based on trigger rule
                    if not self._check_trigger_rule(run, dag, task):
                        continue

                    # Execute task
                    await self._execute_task(run, dag, task)

                # Check for completed runs
                self._check_completed_runs()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scheduler loop error: {e}")

    def _check_trigger_rule(
        self,
        run: DAGRun,
        dag: DAGDefinition,
        task: DAGTask,
    ) -> bool:
        """Check if task's trigger rule is satisfied."""
        upstream_states = [
            run.task_states.get(uid, TaskStatus.PENDING)
            for uid in task.upstream_tasks
        ]

        if not upstream_states:
            return True  # No dependencies

        rule = task.trigger_rule

        success_count = sum(1 for s in upstream_states if s == TaskStatus.COMPLETED)
        failed_count = sum(1 for s in upstream_states if s == TaskStatus.FAILED)
        done_count = sum(1 for s in upstream_states
                         if s in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.SKIPPED))
        skipped_count = sum(1 for s in upstream_states if s == TaskStatus.SKIPPED)

        total = len(upstream_states)

        if rule == TriggerRule.ALL_SUCCESS:
            return success_count == total
        elif rule == TriggerRule.ALL_FAILED:
            return failed_count == total
        elif rule == TriggerRule.ALL_DONE:
            return done_count == total
        elif rule == TriggerRule.ONE_SUCCESS:
            return success_count >= 1
        elif rule == TriggerRule.ONE_FAILED:
            return failed_count >= 1
        elif rule == TriggerRule.NONE_FAILED:
            return failed_count == 0 and done_count == total
        elif rule == TriggerRule.NONE_SKIPPED:
            return skipped_count == 0 and done_count == total

        return False

    async def _execute_task(
        self,
        run: DAGRun,
        dag: DAGDefinition,
        task: DAGTask,
    ) -> None:
        """Execute a task."""
        task_key = f"{run.id}:{task.id}"

        run.task_states[task.id] = TaskStatus.RUNNING
        task.status = TaskStatus.RUNNING
        task.run_id = run.id
        task.started_at = datetime.now()
        task.attempt += 1

        self._running_tasks[task_key] = (run.id, task.id)

        # Update group stats
        if task.group:
            group = self._groups.get(task.group, self._default_group)
            group.running_tasks += 1
            group.queued_tasks = max(0, group.queued_tasks - 1)
            group.last_scheduled = datetime.now()

        # Execute
        self._task_futures[task_key] = asyncio.create_task(
            self._run_task(run, dag, task, task_key)
        )

        self._stats["tasks_executed"] += 1
        logger.debug(f"Started task: {task.name} (attempt {task.attempt})")

    async def _run_task(
        self,
        run: DAGRun,
        dag: DAGDefinition,
        task: DAGTask,
        task_key: str,
    ) -> None:
        """Run a task with timeout and error handling."""
        start_time = datetime.now()

        try:
            handler = self._handlers.get(task.handler)
            if not handler:
                raise ValueError(f"Handler not found: {task.handler}")

            # Prepare context
            context = {
                **run.context,
                "task_id": task.id,
                "task_name": task.name,
                "run_id": run.id,
                "dag_id": dag.id,
                "attempt": task.attempt,
            }

            # Add upstream results
            for upstream_id in task.upstream_tasks:
                upstream_task = dag.tasks.get(upstream_id)
                if upstream_task and upstream_task.result is not None:
                    context[f"upstream_{upstream_id}"] = upstream_task.result

            # Execute with timeout
            result = await asyncio.wait_for(
                self._call_handler(handler, task.params, context),
                timeout=task.timeout_seconds,
            )

            task.result = result
            task.status = TaskStatus.COMPLETED
            run.task_states[task.id] = TaskStatus.COMPLETED
            task.error = None

            self._stats["tasks_succeeded"] += 1
            logger.info(f"Task completed: {task.name}")

        except asyncio.TimeoutError:
            task.error = "Task timed out"
            await self._handle_task_failure(run, dag, task)

        except asyncio.CancelledError:
            task.status = TaskStatus.CANCELLED
            run.task_states[task.id] = TaskStatus.CANCELLED
            raise

        except Exception as e:
            task.error = str(e)
            await self._handle_task_failure(run, dag, task)

        finally:
            # Update timing
            task.completed_at = datetime.now()
            task.duration_seconds = (task.completed_at - start_time).total_seconds()

            # Update group stats
            if task.group:
                group = self._groups.get(task.group, self._default_group)
                group.running_tasks = max(0, group.running_tasks - 1)
                group.total_runtime += task.duration_seconds

            # Cleanup
            self._running_tasks.pop(task_key, None)
            self._task_futures.pop(task_key, None)

            # Record execution order
            if task.id not in run.execution_order:
                run.execution_order.append(task.id)

            # Queue downstream tasks
            self._queue_downstream_tasks(run, dag, task)

    async def _call_handler(
        self,
        handler: Callable,
        params: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Any:
        """Call task handler."""
        result = handler(params, context)
        if asyncio.iscoroutine(result):
            return await result
        return result

    async def _handle_task_failure(
        self,
        run: DAGRun,
        dag: DAGDefinition,
        task: DAGTask,
    ) -> None:
        """Handle task failure with retry logic."""
        task.retry_count += 1

        if task.retry_count <= task.max_retries:
            # Schedule retry
            self._stats["tasks_retried"] += 1
            logger.warning(
                f"Task {task.name} failed, scheduling retry {task.retry_count}/{task.max_retries}"
            )

            await asyncio.sleep(task.retry_delay_seconds)

            task.status = TaskStatus.READY
            run.task_states[task.id] = TaskStatus.READY
            self._queue_task(run.id, task.id, task)
        else:
            # Mark as failed
            task.status = TaskStatus.FAILED
            run.task_states[task.id] = TaskStatus.FAILED
            self._stats["tasks_failed"] += 1
            logger.error(f"Task {task.name} failed after {task.retry_count} retries: {task.error}")

    def _queue_downstream_tasks(
        self,
        run: DAGRun,
        dag: DAGDefinition,
        completed_task: DAGTask,
    ) -> None:
        """Queue downstream tasks if their dependencies are satisfied."""
        for downstream_id in completed_task.downstream_tasks:
            downstream = dag.tasks.get(downstream_id)
            if not downstream:
                continue

            current_state = run.task_states.get(downstream_id)
            if current_state not in (TaskStatus.PENDING, TaskStatus.WAITING):
                continue

            # Check if we should skip this task
            if completed_task.status == TaskStatus.FAILED:
                if downstream.trigger_rule not in (
                    TriggerRule.ALL_FAILED,
                    TriggerRule.ONE_FAILED,
                    TriggerRule.ALL_DONE,
                ):
                    # Check if all upstream are done
                    all_done = all(
                        run.task_states.get(uid) in (
                            TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.SKIPPED
                        )
                        for uid in downstream.upstream_tasks
                    )
                    if all_done:
                        downstream.status = TaskStatus.SKIPPED
                        run.task_states[downstream_id] = TaskStatus.SKIPPED
                        self._queue_downstream_tasks(run, dag, downstream)
                        continue

            # Check if ready
            if self._check_trigger_rule(run, dag, downstream):
                self._queue_task(run.id, downstream_id, downstream)
            else:
                run.task_states[downstream_id] = TaskStatus.WAITING

    def _check_completed_runs(self) -> None:
        """Check for completed DAG runs."""
        for run_id, run in list(self._runs.items()):
            if run.state != "running":
                continue

            dag = self._dags.get(run.dag_id)
            if not dag:
                continue

            # Check if all tasks are in terminal state
            all_terminal = all(
                state in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.SKIPPED, TaskStatus.CANCELLED)
                for state in run.task_states.values()
            )

            if all_terminal:
                failed = any(
                    state == TaskStatus.FAILED
                    for state in run.task_states.values()
                )

                run.state = "failed" if failed else "success"
                run.completed_at = datetime.now()

                self._active_runs[run.dag_id].discard(run_id)

                if failed:
                    self._stats["runs_failed"] += 1
                else:
                    self._stats["runs_completed"] += 1

                logger.info(f"DAG run {run.state}: {dag.name} ({run_id})")

    async def _fair_share_loop(self) -> None:
        """Update fair-share deficits."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(self.fair_share_interval)

                # Calculate total weight
                total_weight = sum(g.weight for g in self._groups.values())
                if total_weight == 0:
                    continue

                # Update deficits
                for group in self._groups.values():
                    fair_share = group.weight / total_weight

                    # Calculate actual share
                    total_running = sum(g.running_tasks for g in self._groups.values())
                    actual_share = group.running_tasks / total_running if total_running > 0 else 0

                    # Update deficit
                    # Positive deficit means group is under-served
                    group.accumulated_deficit += (fair_share - actual_share) * self.fair_share_interval

                    # Clamp deficit
                    group.accumulated_deficit = max(
                        -10.0,
                        min(10.0, group.accumulated_deficit)
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Fair-share loop error: {e}")

    # === Query Methods ===

    def get_dag(self, dag_id: str) -> Optional[DAGDefinition]:
        """Get DAG by ID."""
        return self._dags.get(dag_id)

    def get_run(self, run_id: str) -> Optional[DAGRun]:
        """Get DAG run by ID."""
        return self._runs.get(run_id)

    def get_active_runs(self, dag_id: Optional[str] = None) -> List[DAGRun]:
        """Get active DAG runs."""
        if dag_id:
            run_ids = self._active_runs.get(dag_id, set())
            return [self._runs[rid] for rid in run_ids if rid in self._runs]
        else:
            return [r for r in self._runs.values() if r.state == "running"]

    def get_group_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get fair-share group statistics."""
        return {
            gid: {
                "name": g.name,
                "weight": g.weight,
                "running_tasks": g.running_tasks,
                "queued_tasks": g.queued_tasks,
                "total_runtime": g.total_runtime,
                "accumulated_deficit": g.accumulated_deficit,
            }
            for gid, g in self._groups.items()
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics."""
        return {
            **self._stats,
            "active_dags": len(self._dags),
            "active_runs": sum(len(runs) for runs in self._active_runs.values()),
            "running_tasks": len(self._running_tasks),
            "queued_tasks": len(self._ready_queue),
            "groups": len(self._groups),
        }

    async def cancel_run(self, run_id: str) -> bool:
        """Cancel a DAG run."""
        run = self._runs.get(run_id)
        if not run or run.state != "running":
            return False

        run.state = "cancelled"
        run.completed_at = datetime.now()

        # Cancel running tasks
        for task_key, (rid, tid) in list(self._running_tasks.items()):
            if rid == run_id:
                future = self._task_futures.get(task_key)
                if future:
                    future.cancel()

        self._active_runs[run.dag_id].discard(run_id)
        logger.info(f"Cancelled DAG run: {run_id}")
        return True

    # === Context Manager ===

    async def __aenter__(self) -> "DAGScheduler":
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.shutdown()


# === Convenience Functions for DAG Building ===

def create_dag(
    name: str,
    description: str = "",
    **kwargs,
) -> DAGDefinition:
    """Create a new DAG definition."""
    return DAGDefinition(
        name=name,
        description=description,
        **kwargs,
    )


def create_task(
    name: str,
    handler: str,
    params: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> DAGTask:
    """Create a new DAG task."""
    return DAGTask(
        name=name,
        handler=handler,
        params=params or {},
        **kwargs,
    )
