"""
AION Workflow Service - Complete Runtime

Production-grade workflow service integrating:
- Deterministic workflow execution
- Activity workers with heartbeats
- Query and signal handling
- Workflow lifecycle management
- Search attribute indexing
- Visibility and history APIs
"""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Type,
)

import structlog

from aion.automation.runtime.deterministic import (
    ActivityContext,
    ActivityResult,
    Command,
    CommandType,
    HistoryEvent,
    HistoryEventType,
    ReplayContext,
    ReplayMode,
    SearchAttributeIndex,
    WorkflowContext,
    WorkflowExecution,
    WorkflowExecutionStatus,
    WorkflowHistory,
    WorkflowWorker,
)

logger = structlog.get_logger(__name__)


# =============================================================================
# Activity Heartbeat Manager
# =============================================================================


@dataclass
class ActivityHeartbeat:
    """Heartbeat information for an activity."""
    activity_id: str
    workflow_id: str
    run_id: str
    last_heartbeat: datetime
    details: Any = None
    timeout_seconds: float = 30.0


class ActivityHeartbeatManager:
    """
    Manages activity heartbeats.

    Features:
    - Tracks heartbeats for running activities
    - Detects timed-out activities
    - Stores heartbeat details for recovery
    """

    def __init__(self):
        self._heartbeats: Dict[str, ActivityHeartbeat] = {}
        self._lock = asyncio.Lock()
        self._check_task: Optional[asyncio.Task] = None
        self._timeout_callbacks: List[Callable[[str, str], None]] = []

    async def start(self) -> None:
        """Start heartbeat monitoring."""
        self._check_task = asyncio.create_task(self._check_loop())

    async def stop(self) -> None:
        """Stop heartbeat monitoring."""
        if self._check_task:
            self._check_task.cancel()
            try:
                await self._check_task
            except asyncio.CancelledError:
                pass

    def on_timeout(self, callback: Callable[[str, str], None]) -> None:
        """Register timeout callback (activity_id, workflow_id)."""
        self._timeout_callbacks.append(callback)

    async def record_heartbeat(
        self,
        activity_id: str,
        workflow_id: str,
        run_id: str,
        details: Any = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        """Record a heartbeat from an activity."""
        async with self._lock:
            self._heartbeats[activity_id] = ActivityHeartbeat(
                activity_id=activity_id,
                workflow_id=workflow_id,
                run_id=run_id,
                last_heartbeat=datetime.now(),
                details=details,
                timeout_seconds=timeout_seconds,
            )

        logger.debug(
            "Activity heartbeat recorded",
            activity_id=activity_id,
            workflow_id=workflow_id,
        )

    async def get_heartbeat_details(self, activity_id: str) -> Any:
        """Get last heartbeat details for recovery."""
        async with self._lock:
            hb = self._heartbeats.get(activity_id)
            return hb.details if hb else None

    async def remove_activity(self, activity_id: str) -> None:
        """Remove activity from tracking."""
        async with self._lock:
            self._heartbeats.pop(activity_id, None)

    async def _check_loop(self) -> None:
        """Check for timed-out activities."""
        while True:
            try:
                await asyncio.sleep(5)

                now = datetime.now()
                timed_out = []

                async with self._lock:
                    for activity_id, hb in list(self._heartbeats.items()):
                        elapsed = (now - hb.last_heartbeat).total_seconds()
                        if elapsed > hb.timeout_seconds:
                            timed_out.append((activity_id, hb.workflow_id))
                            del self._heartbeats[activity_id]

                for activity_id, workflow_id in timed_out:
                    logger.warning(
                        "Activity heartbeat timeout",
                        activity_id=activity_id,
                        workflow_id=workflow_id,
                    )
                    for callback in self._timeout_callbacks:
                        try:
                            callback(activity_id, workflow_id)
                        except Exception as e:
                            logger.error("Timeout callback error", error=str(e))

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Heartbeat check error", error=str(e))


# =============================================================================
# Workflow State Store
# =============================================================================


class WorkflowStateStore:
    """
    Stores workflow execution state.

    In production, this would be backed by a database.
    """

    def __init__(self):
        self._executions: Dict[str, WorkflowExecution] = {}  # workflow_id -> execution
        self._histories: Dict[str, WorkflowHistory] = {}  # run_id -> history
        self._by_status: Dict[WorkflowExecutionStatus, Set[str]] = {
            status: set() for status in WorkflowExecutionStatus
        }
        self._lock = asyncio.Lock()

    async def save_execution(self, execution: WorkflowExecution) -> None:
        """Save workflow execution state."""
        async with self._lock:
            # Update status index
            old_exec = self._executions.get(execution.workflow_id)
            if old_exec:
                self._by_status[old_exec.status].discard(execution.workflow_id)

            self._executions[execution.workflow_id] = execution
            self._by_status[execution.status].add(execution.workflow_id)

            # Save history
            if execution.history:
                self._histories[execution.run_id] = execution.history

    async def get_execution(self, workflow_id: str) -> Optional[WorkflowExecution]:
        """Get workflow execution by ID."""
        async with self._lock:
            return self._executions.get(workflow_id)

    async def get_history(self, run_id: str) -> Optional[WorkflowHistory]:
        """Get workflow history by run ID."""
        async with self._lock:
            return self._histories.get(run_id)

    async def list_executions(
        self,
        status: Optional[WorkflowExecutionStatus] = None,
        workflow_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[WorkflowExecution]:
        """List workflow executions."""
        async with self._lock:
            if status:
                workflow_ids = list(self._by_status[status])[:limit]
            else:
                workflow_ids = list(self._executions.keys())[:limit]

            executions = []
            for wf_id in workflow_ids:
                exec = self._executions.get(wf_id)
                if exec:
                    if workflow_type and exec.workflow_type != workflow_type:
                        continue
                    executions.append(exec)

            return executions

    async def delete_execution(self, workflow_id: str) -> bool:
        """Delete a workflow execution."""
        async with self._lock:
            exec = self._executions.pop(workflow_id, None)
            if exec:
                self._by_status[exec.status].discard(workflow_id)
                self._histories.pop(exec.run_id, None)
                return True
            return False


# =============================================================================
# Signal Manager
# =============================================================================


class SignalManager:
    """
    Manages signals to workflows.
    """

    def __init__(self):
        self._pending_signals: Dict[str, List[Tuple[str, Any]]] = {}  # workflow_id -> [(name, payload)]
        self._signal_callbacks: Dict[str, Callable] = {}
        self._lock = asyncio.Lock()

    async def send_signal(
        self,
        workflow_id: str,
        signal_name: str,
        payload: Any = None,
    ) -> None:
        """Send a signal to a workflow."""
        async with self._lock:
            if workflow_id not in self._pending_signals:
                self._pending_signals[workflow_id] = []

            self._pending_signals[workflow_id].append((signal_name, payload))

        logger.info(
            "Signal sent",
            workflow_id=workflow_id,
            signal_name=signal_name,
        )

        # Notify callback if registered
        callback = self._signal_callbacks.get(workflow_id)
        if callback:
            try:
                callback(signal_name, payload)
            except Exception as e:
                logger.error("Signal callback error", error=str(e))

    async def get_pending_signals(
        self,
        workflow_id: str,
    ) -> List[Tuple[str, Any]]:
        """Get and clear pending signals for a workflow."""
        async with self._lock:
            signals = self._pending_signals.pop(workflow_id, [])
            return signals

    def register_callback(
        self,
        workflow_id: str,
        callback: Callable[[str, Any], None],
    ) -> None:
        """Register signal callback for a workflow."""
        self._signal_callbacks[workflow_id] = callback

    def unregister_callback(self, workflow_id: str) -> None:
        """Unregister signal callback."""
        self._signal_callbacks.pop(workflow_id, None)


# =============================================================================
# Query Manager
# =============================================================================


class QueryManager:
    """
    Manages queries to running workflows.
    """

    def __init__(self):
        self._handlers: Dict[str, Dict[str, Callable]] = {}  # workflow_id -> {query_name -> handler}

    def register_handler(
        self,
        workflow_id: str,
        query_name: str,
        handler: Callable[[Any], Any],
    ) -> None:
        """Register a query handler."""
        if workflow_id not in self._handlers:
            self._handlers[workflow_id] = {}
        self._handlers[workflow_id][query_name] = handler

    def unregister_handlers(self, workflow_id: str) -> None:
        """Unregister all handlers for a workflow."""
        self._handlers.pop(workflow_id, None)

    async def query(
        self,
        workflow_id: str,
        query_name: str,
        args: Any = None,
    ) -> Any:
        """
        Query a running workflow.

        Queries are read-only and don't affect workflow state.
        """
        handlers = self._handlers.get(workflow_id, {})
        handler = handlers.get(query_name)

        if not handler:
            raise ValueError(f"No handler for query '{query_name}' in workflow {workflow_id}")

        return handler(args)

    def list_queries(self, workflow_id: str) -> List[str]:
        """List available queries for a workflow."""
        return list(self._handlers.get(workflow_id, {}).keys())


# =============================================================================
# Task Queue
# =============================================================================


@dataclass
class WorkflowTask:
    """A workflow task to be processed."""
    workflow_id: str
    run_id: str
    task_queue: str
    history: WorkflowHistory
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ActivityTask:
    """An activity task to be processed."""
    activity_id: str
    activity_type: str
    workflow_id: str
    run_id: str
    task_queue: str
    args: Any
    scheduled_event_id: int
    heartbeat_timeout: Optional[float] = None
    start_to_close_timeout: Optional[float] = None
    created_at: datetime = field(default_factory=datetime.now)


class TaskQueue:
    """
    Task queue for workflow and activity tasks.
    """

    def __init__(self, name: str):
        self.name = name
        self._workflow_tasks: asyncio.Queue[WorkflowTask] = asyncio.Queue()
        self._activity_tasks: asyncio.Queue[ActivityTask] = asyncio.Queue()

    async def enqueue_workflow_task(self, task: WorkflowTask) -> None:
        """Enqueue a workflow task."""
        await self._workflow_tasks.put(task)

    async def enqueue_activity_task(self, task: ActivityTask) -> None:
        """Enqueue an activity task."""
        await self._activity_tasks.put(task)

    async def dequeue_workflow_task(
        self,
        timeout: Optional[float] = None,
    ) -> Optional[WorkflowTask]:
        """Dequeue a workflow task."""
        try:
            if timeout:
                return await asyncio.wait_for(
                    self._workflow_tasks.get(),
                    timeout=timeout,
                )
            return self._workflow_tasks.get_nowait()
        except (asyncio.TimeoutError, asyncio.QueueEmpty):
            return None

    async def dequeue_activity_task(
        self,
        timeout: Optional[float] = None,
    ) -> Optional[ActivityTask]:
        """Dequeue an activity task."""
        try:
            if timeout:
                return await asyncio.wait_for(
                    self._activity_tasks.get(),
                    timeout=timeout,
                )
            return self._activity_tasks.get_nowait()
        except (asyncio.TimeoutError, asyncio.QueueEmpty):
            return None

    @property
    def workflow_task_count(self) -> int:
        return self._workflow_tasks.qsize()

    @property
    def activity_task_count(self) -> int:
        return self._activity_tasks.qsize()


# =============================================================================
# Workflow Service
# =============================================================================


class WorkflowService:
    """
    Complete workflow service.

    Provides:
    - Workflow start/cancel/terminate
    - Activity execution with heartbeats
    - Signal and query handling
    - Search attribute indexing
    - History and visibility APIs
    """

    def __init__(self):
        self._state_store = WorkflowStateStore()
        self._heartbeat_manager = ActivityHeartbeatManager()
        self._signal_manager = SignalManager()
        self._query_manager = QueryManager()
        self._search_index = SearchAttributeIndex()

        self._task_queues: Dict[str, TaskQueue] = {}
        self._workers: Dict[str, WorkflowWorker] = {}

        self._timer_tasks: Dict[str, asyncio.Task] = {}

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the service."""
        if self._initialized:
            return

        await self._heartbeat_manager.start()
        self._heartbeat_manager.on_timeout(self._on_activity_timeout)

        self._initialized = True
        logger.info("Workflow service initialized")

    async def shutdown(self) -> None:
        """Shutdown the service."""
        await self._heartbeat_manager.stop()

        # Cancel all timer tasks
        for timer_task in self._timer_tasks.values():
            timer_task.cancel()

        self._initialized = False
        logger.info("Workflow service shutdown")

    def _get_task_queue(self, name: str) -> TaskQueue:
        """Get or create a task queue."""
        if name not in self._task_queues:
            self._task_queues[name] = TaskQueue(name)
        return self._task_queues[name]

    def register_worker(self, worker: WorkflowWorker) -> None:
        """Register a workflow worker."""
        self._workers[worker.task_queue] = worker

    # -------------------------------------------------------------------------
    # Workflow Lifecycle
    # -------------------------------------------------------------------------

    async def start_workflow(
        self,
        workflow_type: str,
        args: Any = None,
        *,
        workflow_id: Optional[str] = None,
        task_queue: str = "default",
        execution_timeout: Optional[float] = None,
        run_timeout: Optional[float] = None,
        search_attributes: Optional[Dict[str, Any]] = None,
    ) -> WorkflowExecution:
        """
        Start a new workflow execution.
        """
        workflow_id = workflow_id or str(uuid.uuid4())
        run_id = str(uuid.uuid4())

        # Check for existing
        existing = await self._state_store.get_execution(workflow_id)
        if existing and existing.status == WorkflowExecutionStatus.RUNNING:
            raise ValueError(f"Workflow {workflow_id} already running")

        # Create history
        history = WorkflowHistory(workflow_id, run_id)

        # Add start event
        history.add_event(
            HistoryEventType.WORKFLOW_EXECUTION_STARTED,
            {
                "workflow_type": workflow_type,
                "args": args,
                "task_queue": task_queue,
                "execution_timeout": execution_timeout,
                "run_timeout": run_timeout,
            },
        )

        # Create execution
        execution = WorkflowExecution(
            workflow_id=workflow_id,
            run_id=run_id,
            workflow_type=workflow_type,
            task_queue=task_queue,
            args=args,
            history=history,
            search_attributes=search_attributes or {},
            execution_timeout=execution_timeout,
        )

        await self._state_store.save_execution(execution)

        # Index search attributes
        if search_attributes:
            self._search_index.index_workflow(workflow_id, search_attributes)

        # Queue workflow task
        task_queue_obj = self._get_task_queue(task_queue)
        await task_queue_obj.enqueue_workflow_task(
            WorkflowTask(
                workflow_id=workflow_id,
                run_id=run_id,
                task_queue=task_queue,
                history=history,
            )
        )

        logger.info(
            "Workflow started",
            workflow_id=workflow_id,
            workflow_type=workflow_type,
        )

        return execution

    async def cancel_workflow(
        self,
        workflow_id: str,
        reason: Optional[str] = None,
    ) -> bool:
        """Request cancellation of a workflow."""
        execution = await self._state_store.get_execution(workflow_id)
        if not execution or execution.status != WorkflowExecutionStatus.RUNNING:
            return False

        # Add cancel request event
        if execution.history:
            execution.history.add_event(
                HistoryEventType.WORKFLOW_EXECUTION_CANCEL_REQUESTED,
                {"reason": reason},
            )

        # Signal the workflow
        await self._signal_manager.send_signal(
            workflow_id,
            "__cancel__",
            {"reason": reason},
        )

        logger.info("Workflow cancel requested", workflow_id=workflow_id)
        return True

    async def terminate_workflow(
        self,
        workflow_id: str,
        reason: Optional[str] = None,
    ) -> bool:
        """Forcefully terminate a workflow."""
        execution = await self._state_store.get_execution(workflow_id)
        if not execution or execution.status != WorkflowExecutionStatus.RUNNING:
            return False

        # Update status
        execution.status = WorkflowExecutionStatus.TERMINATED
        execution.close_time = datetime.now()
        execution.error = reason

        if execution.history:
            execution.history.add_event(
                HistoryEventType.WORKFLOW_EXECUTION_TERMINATED,
                {"reason": reason},
            )

        await self._state_store.save_execution(execution)

        # Cleanup
        self._query_manager.unregister_handlers(workflow_id)
        self._signal_manager.unregister_callback(workflow_id)

        logger.info("Workflow terminated", workflow_id=workflow_id)
        return True

    async def get_workflow(self, workflow_id: str) -> Optional[WorkflowExecution]:
        """Get workflow execution."""
        return await self._state_store.get_execution(workflow_id)

    async def get_workflow_history(
        self,
        workflow_id: str,
        run_id: Optional[str] = None,
    ) -> Optional[WorkflowHistory]:
        """Get workflow history."""
        if run_id:
            return await self._state_store.get_history(run_id)

        execution = await self._state_store.get_execution(workflow_id)
        if execution:
            return execution.history
        return None

    async def list_workflows(
        self,
        status: Optional[WorkflowExecutionStatus] = None,
        workflow_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[WorkflowExecution]:
        """List workflow executions."""
        return await self._state_store.list_executions(status, workflow_type, limit)

    async def search_workflows(
        self,
        query: Dict[str, Any],
    ) -> List[WorkflowExecution]:
        """Search workflows by search attributes."""
        workflow_ids = self._search_index.search(query)

        executions = []
        for wf_id in workflow_ids:
            exec = await self._state_store.get_execution(wf_id)
            if exec:
                executions.append(exec)

        return executions

    # -------------------------------------------------------------------------
    # Signals and Queries
    # -------------------------------------------------------------------------

    async def signal_workflow(
        self,
        workflow_id: str,
        signal_name: str,
        payload: Any = None,
    ) -> None:
        """Send a signal to a workflow."""
        execution = await self._state_store.get_execution(workflow_id)
        if not execution:
            raise ValueError(f"Workflow not found: {workflow_id}")

        if execution.status != WorkflowExecutionStatus.RUNNING:
            raise ValueError(f"Workflow not running: {workflow_id}")

        # Record in history
        if execution.history:
            execution.history.add_event(
                HistoryEventType.WORKFLOW_EXECUTION_SIGNALED,
                {
                    "signal_name": signal_name,
                    "payload": payload,
                },
            )
            await self._state_store.save_execution(execution)

        # Deliver signal
        await self._signal_manager.send_signal(workflow_id, signal_name, payload)

    async def query_workflow(
        self,
        workflow_id: str,
        query_name: str,
        args: Any = None,
    ) -> Any:
        """Query a running workflow."""
        execution = await self._state_store.get_execution(workflow_id)
        if not execution:
            raise ValueError(f"Workflow not found: {workflow_id}")

        if execution.status != WorkflowExecutionStatus.RUNNING:
            raise ValueError(f"Workflow not running: {workflow_id}")

        return await self._query_manager.query(workflow_id, query_name, args)

    # -------------------------------------------------------------------------
    # Activity Heartbeats
    # -------------------------------------------------------------------------

    async def activity_heartbeat(
        self,
        activity_id: str,
        workflow_id: str,
        run_id: str,
        details: Any = None,
    ) -> bool:
        """
        Record activity heartbeat.

        Returns False if activity should be cancelled.
        """
        execution = await self._state_store.get_execution(workflow_id)
        if not execution or execution.status != WorkflowExecutionStatus.RUNNING:
            return False  # Workflow no longer running

        await self._heartbeat_manager.record_heartbeat(
            activity_id=activity_id,
            workflow_id=workflow_id,
            run_id=run_id,
            details=details,
        )

        return True

    def _on_activity_timeout(self, activity_id: str, workflow_id: str) -> None:
        """Handle activity heartbeat timeout."""
        logger.warning(
            "Activity heartbeat timeout",
            activity_id=activity_id,
            workflow_id=workflow_id,
        )
        # In production, this would trigger activity failure

    # -------------------------------------------------------------------------
    # Worker Loop
    # -------------------------------------------------------------------------

    async def run_worker(
        self,
        worker: WorkflowWorker,
        *,
        shutdown_event: Optional[asyncio.Event] = None,
    ) -> None:
        """
        Run a workflow worker loop.
        """
        self.register_worker(worker)
        task_queue = self._get_task_queue(worker.task_queue)

        shutdown = shutdown_event or asyncio.Event()

        logger.info("Worker started", task_queue=worker.task_queue)

        while not shutdown.is_set():
            try:
                # Process workflow tasks
                wf_task = await task_queue.dequeue_workflow_task(timeout=1.0)
                if wf_task:
                    await self._process_workflow_task(worker, wf_task)
                    continue

                # Process activity tasks
                act_task = await task_queue.dequeue_activity_task(timeout=1.0)
                if act_task:
                    await self._process_activity_task(worker, act_task)
                    continue

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Worker error", error=str(e))
                await asyncio.sleep(1)

        logger.info("Worker stopped", task_queue=worker.task_queue)

    async def _process_workflow_task(
        self,
        worker: WorkflowWorker,
        task: WorkflowTask,
    ) -> None:
        """Process a workflow task."""
        execution = await self._state_store.get_execution(task.workflow_id)
        if not execution:
            logger.warning("Execution not found", workflow_id=task.workflow_id)
            return

        # Record workflow task started
        task.history.add_event(
            HistoryEventType.WORKFLOW_TASK_STARTED,
            {},
        )

        try:
            # Execute workflow task
            commands, result, error = await worker.execute_workflow_task(
                execution,
                task.history,
            )

            # Record workflow task completed
            task.history.add_event(
                HistoryEventType.WORKFLOW_TASK_COMPLETED,
                {"commands": [c.to_dict() for c in commands]},
            )

            # Process commands
            await self._process_commands(execution, task.history, commands)

            # Check if completed
            if result is not None:
                execution.status = WorkflowExecutionStatus.COMPLETED
                execution.result = result
                execution.close_time = datetime.now()

                task.history.add_event(
                    HistoryEventType.WORKFLOW_EXECUTION_COMPLETED,
                    {"result": result},
                )

            elif error:
                execution.status = WorkflowExecutionStatus.FAILED
                execution.error = error
                execution.close_time = datetime.now()

                task.history.add_event(
                    HistoryEventType.WORKFLOW_EXECUTION_FAILED,
                    {"error": error},
                )

            execution.history = task.history
            await self._state_store.save_execution(execution)

        except Exception as e:
            logger.error(
                "Workflow task failed",
                workflow_id=task.workflow_id,
                error=str(e),
            )

            task.history.add_event(
                HistoryEventType.WORKFLOW_TASK_FAILED,
                {"error": str(e)},
            )

    async def _process_commands(
        self,
        execution: WorkflowExecution,
        history: WorkflowHistory,
        commands: List[Command],
    ) -> None:
        """Process commands from workflow execution."""
        task_queue = self._get_task_queue(execution.task_queue)

        for command in commands:
            if command.type == CommandType.SCHEDULE_ACTIVITY:
                await self._schedule_activity(execution, history, command, task_queue)

            elif command.type == CommandType.START_TIMER:
                await self._start_timer(execution, history, command, task_queue)

            elif command.type == CommandType.RECORD_MARKER:
                history.add_event(
                    HistoryEventType.MARKER_RECORDED,
                    command.attributes,
                )

            elif command.type == CommandType.UPSERT_SEARCH_ATTRIBUTES:
                history.add_event(
                    HistoryEventType.UPSERT_WORKFLOW_SEARCH_ATTRIBUTES,
                    command.attributes,
                )
                # Update index
                self._search_index.index_workflow(
                    execution.workflow_id,
                    command.attributes.get("search_attributes", {}),
                )

            elif command.type == CommandType.START_CHILD_WORKFLOW:
                await self._start_child_workflow(execution, history, command)

            elif command.type == CommandType.CONTINUE_AS_NEW:
                await self._continue_as_new(execution, history, command)

    async def _schedule_activity(
        self,
        execution: WorkflowExecution,
        history: WorkflowHistory,
        command: Command,
        task_queue: TaskQueue,
    ) -> None:
        """Schedule an activity task."""
        attrs = command.attributes

        # Record scheduled event
        scheduled_event = history.add_event(
            HistoryEventType.ACTIVITY_TASK_SCHEDULED,
            {
                "activity_id": attrs["activity_id"],
                "activity_type": attrs["activity_type"],
                "args": attrs.get("args"),
                "task_queue": attrs.get("task_queue") or execution.task_queue,
                "schedule_to_close_timeout": attrs.get("schedule_to_close_timeout"),
                "start_to_close_timeout": attrs.get("start_to_close_timeout"),
                "heartbeat_timeout": attrs.get("heartbeat_timeout"),
            },
        )

        # Create activity task
        activity_task = ActivityTask(
            activity_id=attrs["activity_id"],
            activity_type=attrs["activity_type"],
            workflow_id=execution.workflow_id,
            run_id=execution.run_id,
            task_queue=attrs.get("task_queue") or execution.task_queue,
            args=attrs.get("args"),
            scheduled_event_id=scheduled_event.event_id,
            heartbeat_timeout=attrs.get("heartbeat_timeout"),
            start_to_close_timeout=attrs.get("start_to_close_timeout"),
        )

        await task_queue.enqueue_activity_task(activity_task)

    async def _process_activity_task(
        self,
        worker: WorkflowWorker,
        task: ActivityTask,
    ) -> None:
        """Process an activity task."""
        execution = await self._state_store.get_execution(task.workflow_id)
        if not execution or not execution.history:
            return

        history = execution.history

        # Record started
        started_event = history.add_event(
            HistoryEventType.ACTIVITY_TASK_STARTED,
            {"activity_id": task.activity_id},
            scheduled_event_id=task.scheduled_event_id,
        )

        # Setup heartbeat callback
        async def heartbeat_callback(details: Any) -> None:
            await self.activity_heartbeat(
                task.activity_id,
                task.workflow_id,
                task.run_id,
                details,
            )

        # Execute activity
        ActivityContext._current = ActivityContext(
            activity_id=task.activity_id,
            activity_type=task.activity_type,
            workflow_id=task.workflow_id,
            run_id=task.run_id,
            heartbeat_timeout=task.heartbeat_timeout,
        )
        ActivityContext._current.set_heartbeat_callback(heartbeat_callback)

        try:
            result = await worker.execute_activity(
                task.activity_type,
                task.args,
                task.workflow_id,
                task.run_id,
                task.activity_id,
                task.heartbeat_timeout,
            )

            if result.success:
                history.add_event(
                    HistoryEventType.ACTIVITY_TASK_COMPLETED,
                    {
                        "activity_id": task.activity_id,
                        "result": result.result,
                    },
                    scheduled_event_id=task.scheduled_event_id,
                    started_event_id=started_event.event_id,
                )
            else:
                history.add_event(
                    HistoryEventType.ACTIVITY_TASK_FAILED,
                    {
                        "activity_id": task.activity_id,
                        "error": result.error,
                        "error_type": result.error_type,
                    },
                    scheduled_event_id=task.scheduled_event_id,
                    started_event_id=started_event.event_id,
                )

        finally:
            ActivityContext._current = None
            await self._heartbeat_manager.remove_activity(task.activity_id)

        await self._state_store.save_execution(execution)

        # Queue another workflow task to process result
        tq = self._get_task_queue(execution.task_queue)
        await tq.enqueue_workflow_task(
            WorkflowTask(
                workflow_id=execution.workflow_id,
                run_id=execution.run_id,
                task_queue=execution.task_queue,
                history=history,
            )
        )

    async def _start_timer(
        self,
        execution: WorkflowExecution,
        history: WorkflowHistory,
        command: Command,
        task_queue: TaskQueue,
    ) -> None:
        """Start a timer."""
        attrs = command.attributes
        timer_id = attrs["timer_id"]
        duration = attrs["duration_seconds"]

        # Record timer started
        started_event = history.add_event(
            HistoryEventType.TIMER_STARTED,
            {
                "timer_id": timer_id,
                "duration_seconds": duration,
                "fire_at": attrs["fire_at"],
            },
        )

        # Create timer task
        async def timer_fire():
            await asyncio.sleep(duration)

            # Record timer fired
            exec = await self._state_store.get_execution(execution.workflow_id)
            if exec and exec.history:
                exec.history.add_event(
                    HistoryEventType.TIMER_FIRED,
                    {"timer_id": timer_id},
                    scheduled_event_id=started_event.event_id,
                )
                await self._state_store.save_execution(exec)

                # Queue workflow task
                await task_queue.enqueue_workflow_task(
                    WorkflowTask(
                        workflow_id=exec.workflow_id,
                        run_id=exec.run_id,
                        task_queue=exec.task_queue,
                        history=exec.history,
                    )
                )

            self._timer_tasks.pop(timer_id, None)

        self._timer_tasks[timer_id] = asyncio.create_task(timer_fire())

    async def _start_child_workflow(
        self,
        execution: WorkflowExecution,
        history: WorkflowHistory,
        command: Command,
    ) -> None:
        """Start a child workflow."""
        attrs = command.attributes

        history.add_event(
            HistoryEventType.START_CHILD_WORKFLOW_EXECUTION_INITIATED,
            {
                "workflow_id": attrs["workflow_id"],
                "run_id": attrs["run_id"],
                "workflow_type": attrs["workflow_type"],
                "args": attrs.get("args"),
            },
        )

        # Start the child workflow
        await self.start_workflow(
            workflow_type=attrs["workflow_type"],
            args=attrs.get("args"),
            workflow_id=attrs["workflow_id"],
            task_queue=attrs.get("task_queue") or execution.task_queue,
            execution_timeout=attrs.get("execution_timeout"),
        )

    async def _continue_as_new(
        self,
        execution: WorkflowExecution,
        history: WorkflowHistory,
        command: Command,
    ) -> None:
        """Continue workflow as new."""
        attrs = command.attributes

        history.add_event(
            HistoryEventType.WORKFLOW_EXECUTION_CONTINUED_AS_NEW,
            attrs,
        )

        execution.status = WorkflowExecutionStatus.CONTINUED_AS_NEW
        execution.close_time = datetime.now()
        await self._state_store.save_execution(execution)

        # Start new execution
        await self.start_workflow(
            workflow_type=attrs.get("workflow_type") or execution.workflow_type,
            args=attrs.get("args"),
            workflow_id=execution.workflow_id,  # Same workflow ID
            task_queue=attrs.get("task_queue") or execution.task_queue,
        )


# =============================================================================
# Factory Functions
# =============================================================================


async def create_workflow_service() -> WorkflowService:
    """Create and initialize a workflow service."""
    service = WorkflowService()
    await service.initialize()
    return service
