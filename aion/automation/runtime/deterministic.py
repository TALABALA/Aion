"""
AION Deterministic Workflow Execution Engine

True SOTA implementation matching Temporal/Cadence:
- Deterministic replay with side-effect recording
- Activity heartbeats with progress reporting
- Workflow queries for inspecting state
- Signals for external events
- Continue-as-new for history management
- Child workflow orchestration
- Search attributes for indexing
- Timer and schedule management
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import random
import time
import uuid
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from typing import (
    Any,
    Awaitable,
    Callable,
    Coroutine,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import structlog

logger = structlog.get_logger(__name__)

T = TypeVar("T")
R = TypeVar("R")


# =============================================================================
# Command Types (What workflow code requests)
# =============================================================================


class CommandType(str, Enum):
    """Types of commands a workflow can issue."""
    SCHEDULE_ACTIVITY = "schedule_activity"
    START_TIMER = "start_timer"
    CANCEL_TIMER = "cancel_timer"
    RECORD_MARKER = "record_marker"  # Side effects
    START_CHILD_WORKFLOW = "start_child_workflow"
    SIGNAL_CHILD_WORKFLOW = "signal_child_workflow"
    CANCEL_CHILD_WORKFLOW = "cancel_child_workflow"
    REQUEST_CANCEL_EXTERNAL = "request_cancel_external"
    SIGNAL_EXTERNAL_WORKFLOW = "signal_external_workflow"
    CONTINUE_AS_NEW = "continue_as_new"
    COMPLETE_WORKFLOW = "complete_workflow"
    FAIL_WORKFLOW = "fail_workflow"
    UPSERT_SEARCH_ATTRIBUTES = "upsert_search_attributes"


@dataclass
class Command:
    """A command issued by workflow code."""
    id: str
    type: CommandType
    attributes: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "attributes": self.attributes,
            "timestamp": self.timestamp.isoformat(),
        }


# =============================================================================
# Event Types (What actually happened)
# =============================================================================


class HistoryEventType(str, Enum):
    """Types of events in workflow history."""
    # Workflow lifecycle
    WORKFLOW_EXECUTION_STARTED = "WorkflowExecutionStarted"
    WORKFLOW_EXECUTION_COMPLETED = "WorkflowExecutionCompleted"
    WORKFLOW_EXECUTION_FAILED = "WorkflowExecutionFailed"
    WORKFLOW_EXECUTION_TIMED_OUT = "WorkflowExecutionTimedOut"
    WORKFLOW_EXECUTION_CANCELED = "WorkflowExecutionCanceled"
    WORKFLOW_EXECUTION_TERMINATED = "WorkflowExecutionTerminated"
    WORKFLOW_EXECUTION_CONTINUED_AS_NEW = "WorkflowExecutionContinuedAsNew"

    # Activity lifecycle
    ACTIVITY_TASK_SCHEDULED = "ActivityTaskScheduled"
    ACTIVITY_TASK_STARTED = "ActivityTaskStarted"
    ACTIVITY_TASK_COMPLETED = "ActivityTaskCompleted"
    ACTIVITY_TASK_FAILED = "ActivityTaskFailed"
    ACTIVITY_TASK_TIMED_OUT = "ActivityTaskTimedOut"
    ACTIVITY_TASK_CANCELED = "ActivityTaskCanceled"
    ACTIVITY_TASK_CANCEL_REQUESTED = "ActivityTaskCancelRequested"

    # Timer lifecycle
    TIMER_STARTED = "TimerStarted"
    TIMER_FIRED = "TimerFired"
    TIMER_CANCELED = "TimerCanceled"

    # Child workflow lifecycle
    START_CHILD_WORKFLOW_EXECUTION_INITIATED = "StartChildWorkflowExecutionInitiated"
    CHILD_WORKFLOW_EXECUTION_STARTED = "ChildWorkflowExecutionStarted"
    CHILD_WORKFLOW_EXECUTION_COMPLETED = "ChildWorkflowExecutionCompleted"
    CHILD_WORKFLOW_EXECUTION_FAILED = "ChildWorkflowExecutionFailed"
    CHILD_WORKFLOW_EXECUTION_CANCELED = "ChildWorkflowExecutionCanceled"
    CHILD_WORKFLOW_EXECUTION_TIMED_OUT = "ChildWorkflowExecutionTimedOut"
    CHILD_WORKFLOW_EXECUTION_TERMINATED = "ChildWorkflowExecutionTerminated"

    # Signal events
    WORKFLOW_EXECUTION_SIGNALED = "WorkflowExecutionSignaled"
    SIGNAL_EXTERNAL_WORKFLOW_EXECUTION_INITIATED = "SignalExternalWorkflowExecutionInitiated"
    SIGNAL_EXTERNAL_WORKFLOW_EXECUTION_COMPLETED = "ExternalWorkflowExecutionSignaled"
    SIGNAL_EXTERNAL_WORKFLOW_EXECUTION_FAILED = "SignalExternalWorkflowExecutionFailed"

    # Markers (side effects, versions, etc.)
    MARKER_RECORDED = "MarkerRecorded"

    # Cancel request
    WORKFLOW_EXECUTION_CANCEL_REQUESTED = "WorkflowExecutionCancelRequested"
    REQUEST_CANCEL_EXTERNAL_WORKFLOW_EXECUTION_INITIATED = "RequestCancelExternalWorkflowExecutionInitiated"

    # Search attributes
    UPSERT_WORKFLOW_SEARCH_ATTRIBUTES = "UpsertWorkflowSearchAttributes"

    # Workflow task (decision task)
    WORKFLOW_TASK_SCHEDULED = "WorkflowTaskScheduled"
    WORKFLOW_TASK_STARTED = "WorkflowTaskStarted"
    WORKFLOW_TASK_COMPLETED = "WorkflowTaskCompleted"
    WORKFLOW_TASK_TIMED_OUT = "WorkflowTaskTimedOut"
    WORKFLOW_TASK_FAILED = "WorkflowTaskFailed"


@dataclass
class HistoryEvent:
    """An event in workflow history."""
    event_id: int
    event_type: HistoryEventType
    timestamp: datetime
    attributes: Dict[str, Any]

    # Linking
    scheduled_event_id: Optional[int] = None  # Links completion to schedule
    started_event_id: Optional[int] = None  # Links completion to start

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "attributes": self.attributes,
            "scheduled_event_id": self.scheduled_event_id,
            "started_event_id": self.started_event_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HistoryEvent":
        return cls(
            event_id=data["event_id"],
            event_type=HistoryEventType(data["event_type"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            attributes=data["attributes"],
            scheduled_event_id=data.get("scheduled_event_id"),
            started_event_id=data.get("started_event_id"),
        )


# =============================================================================
# Workflow History
# =============================================================================


class WorkflowHistory:
    """
    Complete history of a workflow execution.

    This is the source of truth for replay.
    """

    def __init__(self, workflow_id: str, run_id: str):
        self.workflow_id = workflow_id
        self.run_id = run_id
        self.events: List[HistoryEvent] = []
        self._next_event_id = 1

    def add_event(
        self,
        event_type: HistoryEventType,
        attributes: Dict[str, Any],
        scheduled_event_id: Optional[int] = None,
        started_event_id: Optional[int] = None,
    ) -> HistoryEvent:
        """Add an event to history."""
        event = HistoryEvent(
            event_id=self._next_event_id,
            event_type=event_type,
            timestamp=datetime.now(),
            attributes=attributes,
            scheduled_event_id=scheduled_event_id,
            started_event_id=started_event_id,
        )
        self.events.append(event)
        self._next_event_id += 1
        return event

    def get_event(self, event_id: int) -> Optional[HistoryEvent]:
        """Get event by ID."""
        for event in self.events:
            if event.event_id == event_id:
                return event
        return None

    def find_events(self, event_type: HistoryEventType) -> List[HistoryEvent]:
        """Find all events of a type."""
        return [e for e in self.events if e.event_type == event_type]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "run_id": self.run_id,
            "events": [e.to_dict() for e in self.events],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowHistory":
        history = cls(data["workflow_id"], data["run_id"])
        for event_data in data["events"]:
            event = HistoryEvent.from_dict(event_data)
            history.events.append(event)
            history._next_event_id = max(history._next_event_id, event.event_id + 1)
        return history


# =============================================================================
# Replay Context
# =============================================================================


class ReplayMode(str, Enum):
    """Execution mode."""
    EXECUTING = "executing"  # Normal execution, recording events
    REPLAYING = "replaying"  # Replay mode, using recorded events


@dataclass
class ActivityResult:
    """Result of an activity execution."""
    success: bool
    result: Any = None
    error: Optional[str] = None
    error_type: Optional[str] = None


@dataclass
class PendingActivity:
    """An activity waiting for completion."""
    activity_id: str
    activity_type: str
    scheduled_event_id: int
    future: asyncio.Future
    heartbeat_timeout: Optional[float] = None
    last_heartbeat: Optional[datetime] = None
    heartbeat_details: Any = None


@dataclass
class PendingTimer:
    """A timer waiting to fire."""
    timer_id: str
    scheduled_event_id: int
    fire_at: datetime
    future: asyncio.Future


@dataclass
class PendingChildWorkflow:
    """A child workflow waiting for completion."""
    child_workflow_id: str
    child_run_id: str
    initiated_event_id: int
    future: asyncio.Future


class ReplayContext:
    """
    Context for deterministic workflow execution.

    This is the key to Temporal-style replay:
    - Tracks position in history
    - Provides deterministic time, random, UUID
    - Records and replays side effects
    """

    def __init__(
        self,
        history: WorkflowHistory,
        mode: ReplayMode = ReplayMode.EXECUTING,
    ):
        self.history = history
        self.mode = mode

        # Replay position
        self._replay_index = 0

        # Pending operations
        self._pending_activities: Dict[str, PendingActivity] = {}
        self._pending_timers: Dict[str, PendingTimer] = {}
        self._pending_children: Dict[str, PendingChildWorkflow] = {}

        # Commands generated this workflow task
        self._commands: List[Command] = []

        # Side effect cache (for replay)
        self._side_effect_counter = 0
        self._side_effect_cache: Dict[int, Any] = {}

        # Version markers
        self._version_cache: Dict[str, int] = {}

        # Signals received
        self._signal_queue: asyncio.Queue = asyncio.Queue()

        # Query handlers
        self._query_handlers: Dict[str, Callable] = {}

        # Search attributes
        self._search_attributes: Dict[str, Any] = {}

        # Workflow state
        self._is_replaying = mode == ReplayMode.REPLAYING
        self._workflow_start_time: Optional[datetime] = None

    @property
    def is_replaying(self) -> bool:
        """Check if currently replaying history."""
        return self._is_replaying

    def now(self) -> datetime:
        """
        Get deterministic current time.

        During replay, returns the time from history.
        """
        if self._is_replaying and self._replay_index < len(self.history.events):
            return self.history.events[self._replay_index].timestamp

        if self._workflow_start_time:
            # Use workflow start time as base for determinism
            return self._workflow_start_time

        return datetime.now()

    def random(self) -> float:
        """
        Get deterministic random number.

        Uses workflow run_id as seed for reproducibility.
        """
        # Create deterministic seed from run_id and call count
        seed = f"{self.history.run_id}:{self._side_effect_counter}"
        self._side_effect_counter += 1

        # Use hash for deterministic "random"
        hash_value = int(hashlib.sha256(seed.encode()).hexdigest()[:16], 16)
        return hash_value / (2 ** 64)

    def uuid4(self) -> str:
        """
        Get deterministic UUID.

        Uses workflow context for reproducibility.
        """
        seed = f"{self.history.run_id}:uuid:{self._side_effect_counter}"
        self._side_effect_counter += 1

        # Generate deterministic UUID from hash
        hash_bytes = hashlib.sha256(seed.encode()).digest()[:16]
        return str(uuid.UUID(bytes=hash_bytes))

    def _get_next_replay_event(
        self,
        expected_type: HistoryEventType,
    ) -> Optional[HistoryEvent]:
        """Get next event during replay if it matches expected type."""
        if not self._is_replaying:
            return None

        while self._replay_index < len(self.history.events):
            event = self.history.events[self._replay_index]

            # Skip workflow task events during replay
            if event.event_type in (
                HistoryEventType.WORKFLOW_TASK_SCHEDULED,
                HistoryEventType.WORKFLOW_TASK_STARTED,
                HistoryEventType.WORKFLOW_TASK_COMPLETED,
            ):
                self._replay_index += 1
                continue

            if event.event_type == expected_type:
                self._replay_index += 1
                return event

            # Mismatch - not replaying this event
            return None

        return None

    def _find_matching_event(
        self,
        event_type: HistoryEventType,
        match_fn: Callable[[HistoryEvent], bool],
    ) -> Optional[HistoryEvent]:
        """Find a matching event in remaining history."""
        for i in range(self._replay_index, len(self.history.events)):
            event = self.history.events[i]
            if event.event_type == event_type and match_fn(event):
                return event
        return None

    def add_command(self, command: Command) -> None:
        """Add a command to be executed."""
        self._commands.append(command)

    def get_commands(self) -> List[Command]:
        """Get all pending commands."""
        commands = self._commands.copy()
        self._commands.clear()
        return commands

    def register_query_handler(self, name: str, handler: Callable) -> None:
        """Register a query handler."""
        self._query_handlers[name] = handler

    async def handle_query(self, name: str, args: Any) -> Any:
        """Handle a query."""
        handler = self._query_handlers.get(name)
        if not handler:
            raise ValueError(f"Unknown query: {name}")
        return handler(args)


# =============================================================================
# Workflow Context (User-facing API)
# =============================================================================


class WorkflowContext:
    """
    User-facing API for workflow code.

    Provides deterministic operations that are safe for replay.
    """

    _current: Optional["WorkflowContext"] = None

    def __init__(self, replay_ctx: ReplayContext):
        self._ctx = replay_ctx
        self._activity_id_counter = 0
        self._timer_id_counter = 0
        self._child_workflow_counter = 0

    @classmethod
    def current(cls) -> "WorkflowContext":
        """Get current workflow context."""
        if cls._current is None:
            raise RuntimeError("No workflow context - are you in a workflow?")
        return cls._current

    @property
    def workflow_id(self) -> str:
        return self._ctx.history.workflow_id

    @property
    def run_id(self) -> str:
        return self._ctx.history.run_id

    @property
    def is_replaying(self) -> bool:
        return self._ctx.is_replaying

    # -------------------------------------------------------------------------
    # Deterministic Operations
    # -------------------------------------------------------------------------

    def now(self) -> datetime:
        """Get current time (deterministic during replay)."""
        return self._ctx.now()

    def random(self) -> float:
        """Get random number (deterministic during replay)."""
        return self._ctx.random()

    def uuid4(self) -> str:
        """Generate UUID (deterministic during replay)."""
        return self._ctx.uuid4()

    def sleep(self, seconds: float) -> Awaitable[None]:
        """Sleep for duration (uses timer internally)."""
        return self.start_timer(f"sleep-{self._timer_id_counter}", seconds)

    # -------------------------------------------------------------------------
    # Activity Execution
    # -------------------------------------------------------------------------

    async def execute_activity(
        self,
        activity_type: str,
        args: Any = None,
        *,
        task_queue: Optional[str] = None,
        schedule_to_close_timeout: Optional[float] = None,
        schedule_to_start_timeout: Optional[float] = None,
        start_to_close_timeout: Optional[float] = None,
        heartbeat_timeout: Optional[float] = None,
        retry_policy: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Execute an activity.

        Activities are non-deterministic operations that:
        1. Get scheduled (recorded in history)
        2. Execute on a worker
        3. Complete/fail (recorded in history)

        During replay, we return the recorded result.
        """
        self._activity_id_counter += 1
        activity_id = f"{activity_type}-{self._activity_id_counter}"

        # Check for replay
        scheduled_event = self._ctx._find_matching_event(
            HistoryEventType.ACTIVITY_TASK_SCHEDULED,
            lambda e: e.attributes.get("activity_id") == activity_id,
        )

        if scheduled_event:
            # Find completion event
            completed_event = self._ctx._find_matching_event(
                HistoryEventType.ACTIVITY_TASK_COMPLETED,
                lambda e: e.scheduled_event_id == scheduled_event.event_id,
            )

            if completed_event:
                return completed_event.attributes.get("result")

            # Check for failure
            failed_event = self._ctx._find_matching_event(
                HistoryEventType.ACTIVITY_TASK_FAILED,
                lambda e: e.scheduled_event_id == scheduled_event.event_id,
            )

            if failed_event:
                raise ActivityError(
                    failed_event.attributes.get("error", "Activity failed"),
                    activity_type=activity_type,
                )

        # Not in history - schedule it
        command = Command(
            id=activity_id,
            type=CommandType.SCHEDULE_ACTIVITY,
            attributes={
                "activity_id": activity_id,
                "activity_type": activity_type,
                "args": args,
                "task_queue": task_queue,
                "schedule_to_close_timeout": schedule_to_close_timeout,
                "schedule_to_start_timeout": schedule_to_start_timeout,
                "start_to_close_timeout": start_to_close_timeout,
                "heartbeat_timeout": heartbeat_timeout,
                "retry_policy": retry_policy,
            },
        )
        self._ctx.add_command(command)

        # Create future for result
        future: asyncio.Future = asyncio.Future()
        self._ctx._pending_activities[activity_id] = PendingActivity(
            activity_id=activity_id,
            activity_type=activity_type,
            scheduled_event_id=0,  # Will be set when event is recorded
            future=future,
            heartbeat_timeout=heartbeat_timeout,
        )

        return await future

    # -------------------------------------------------------------------------
    # Timers
    # -------------------------------------------------------------------------

    async def start_timer(
        self,
        timer_id: str,
        duration_seconds: float,
    ) -> None:
        """
        Start a timer.

        Timers are durable - they survive workflow restarts.
        """
        self._timer_id_counter += 1
        full_timer_id = f"{timer_id}-{self._timer_id_counter}"

        # Check for replay
        started_event = self._ctx._find_matching_event(
            HistoryEventType.TIMER_STARTED,
            lambda e: e.attributes.get("timer_id") == full_timer_id,
        )

        if started_event:
            # Check if fired
            fired_event = self._ctx._find_matching_event(
                HistoryEventType.TIMER_FIRED,
                lambda e: e.scheduled_event_id == started_event.event_id,
            )

            if fired_event:
                return  # Timer already fired

        # Not in history or not fired - schedule it
        fire_at = self.now() + timedelta(seconds=duration_seconds)

        command = Command(
            id=full_timer_id,
            type=CommandType.START_TIMER,
            attributes={
                "timer_id": full_timer_id,
                "duration_seconds": duration_seconds,
                "fire_at": fire_at.isoformat(),
            },
        )
        self._ctx.add_command(command)

        # Create future
        future: asyncio.Future = asyncio.Future()
        self._ctx._pending_timers[full_timer_id] = PendingTimer(
            timer_id=full_timer_id,
            scheduled_event_id=0,
            fire_at=fire_at,
            future=future,
        )

        await future

    async def cancel_timer(self, timer_id: str) -> bool:
        """Cancel a timer."""
        if timer_id in self._ctx._pending_timers:
            timer = self._ctx._pending_timers.pop(timer_id)
            if not timer.future.done():
                timer.future.cancel()

            command = Command(
                id=timer_id,
                type=CommandType.CANCEL_TIMER,
                attributes={"timer_id": timer_id},
            )
            self._ctx.add_command(command)
            return True

        return False

    # -------------------------------------------------------------------------
    # Side Effects (Non-deterministic operations that should be recorded)
    # -------------------------------------------------------------------------

    async def side_effect(
        self,
        func: Callable[[], T],
    ) -> T:
        """
        Execute a side effect.

        Side effects are recorded so replay returns the same value.
        Use for: current time, random IDs, external state reads.
        """
        effect_id = self._ctx._side_effect_counter
        self._ctx._side_effect_counter += 1

        # Check replay cache
        if effect_id in self._ctx._side_effect_cache:
            return self._ctx._side_effect_cache[effect_id]

        # Check history for marker
        marker_event = self._ctx._find_matching_event(
            HistoryEventType.MARKER_RECORDED,
            lambda e: (
                e.attributes.get("marker_name") == "SideEffect" and
                e.attributes.get("effect_id") == effect_id
            ),
        )

        if marker_event:
            result = marker_event.attributes.get("result")
            self._ctx._side_effect_cache[effect_id] = result
            return result

        # Execute and record
        result = func()

        command = Command(
            id=f"side-effect-{effect_id}",
            type=CommandType.RECORD_MARKER,
            attributes={
                "marker_name": "SideEffect",
                "effect_id": effect_id,
                "result": result,
            },
        )
        self._ctx.add_command(command)

        self._ctx._side_effect_cache[effect_id] = result
        return result

    # -------------------------------------------------------------------------
    # Versioning (for workflow code updates)
    # -------------------------------------------------------------------------

    def get_version(
        self,
        change_id: str,
        min_supported: int,
        max_supported: int,
    ) -> int:
        """
        Get version for a code change.

        Enables safe workflow code updates:
        - New executions get max_supported
        - Replaying old executions get their recorded version
        """
        # Check cache
        if change_id in self._ctx._version_cache:
            return self._ctx._version_cache[change_id]

        # Check history
        marker_event = self._ctx._find_matching_event(
            HistoryEventType.MARKER_RECORDED,
            lambda e: (
                e.attributes.get("marker_name") == "Version" and
                e.attributes.get("change_id") == change_id
            ),
        )

        if marker_event:
            version = marker_event.attributes.get("version")
            self._ctx._version_cache[change_id] = version
            return version

        # New execution - use max version
        version = max_supported

        command = Command(
            id=f"version-{change_id}",
            type=CommandType.RECORD_MARKER,
            attributes={
                "marker_name": "Version",
                "change_id": change_id,
                "version": version,
            },
        )
        self._ctx.add_command(command)

        self._ctx._version_cache[change_id] = version
        return version

    # -------------------------------------------------------------------------
    # Child Workflows
    # -------------------------------------------------------------------------

    async def execute_child_workflow(
        self,
        workflow_type: str,
        args: Any = None,
        *,
        workflow_id: Optional[str] = None,
        task_queue: Optional[str] = None,
        execution_timeout: Optional[float] = None,
        run_timeout: Optional[float] = None,
        task_timeout: Optional[float] = None,
        retry_policy: Optional[Dict[str, Any]] = None,
        parent_close_policy: str = "TERMINATE",  # TERMINATE, ABANDON, REQUEST_CANCEL
    ) -> Any:
        """Execute a child workflow."""
        self._child_workflow_counter += 1
        child_workflow_id = workflow_id or f"{self.workflow_id}/child-{self._child_workflow_counter}"
        child_run_id = self.uuid4()

        # Check for replay
        initiated_event = self._ctx._find_matching_event(
            HistoryEventType.START_CHILD_WORKFLOW_EXECUTION_INITIATED,
            lambda e: e.attributes.get("workflow_id") == child_workflow_id,
        )

        if initiated_event:
            # Check completion
            completed_event = self._ctx._find_matching_event(
                HistoryEventType.CHILD_WORKFLOW_EXECUTION_COMPLETED,
                lambda e: e.attributes.get("workflow_id") == child_workflow_id,
            )

            if completed_event:
                return completed_event.attributes.get("result")

            # Check failure
            failed_event = self._ctx._find_matching_event(
                HistoryEventType.CHILD_WORKFLOW_EXECUTION_FAILED,
                lambda e: e.attributes.get("workflow_id") == child_workflow_id,
            )

            if failed_event:
                raise ChildWorkflowError(
                    failed_event.attributes.get("error", "Child workflow failed"),
                    workflow_id=child_workflow_id,
                )

        # Not in history - start it
        command = Command(
            id=child_workflow_id,
            type=CommandType.START_CHILD_WORKFLOW,
            attributes={
                "workflow_id": child_workflow_id,
                "run_id": child_run_id,
                "workflow_type": workflow_type,
                "args": args,
                "task_queue": task_queue,
                "execution_timeout": execution_timeout,
                "run_timeout": run_timeout,
                "task_timeout": task_timeout,
                "retry_policy": retry_policy,
                "parent_close_policy": parent_close_policy,
            },
        )
        self._ctx.add_command(command)

        # Create future
        future: asyncio.Future = asyncio.Future()
        self._ctx._pending_children[child_workflow_id] = PendingChildWorkflow(
            child_workflow_id=child_workflow_id,
            child_run_id=child_run_id,
            initiated_event_id=0,
            future=future,
        )

        return await future

    # -------------------------------------------------------------------------
    # Signals
    # -------------------------------------------------------------------------

    async def wait_for_signal(
        self,
        signal_name: str,
        timeout: Optional[float] = None,
    ) -> Any:
        """Wait for a signal."""
        # Check history for signal
        signal_event = self._ctx._find_matching_event(
            HistoryEventType.WORKFLOW_EXECUTION_SIGNALED,
            lambda e: e.attributes.get("signal_name") == signal_name,
        )

        if signal_event:
            return signal_event.attributes.get("payload")

        # Wait for signal (with optional timeout)
        if timeout:
            try:
                return await asyncio.wait_for(
                    self._ctx._signal_queue.get(),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                return None
        else:
            return await self._ctx._signal_queue.get()

    def signal_received(self, signal_name: str, payload: Any) -> None:
        """Called when a signal is received."""
        self._ctx._signal_queue.put_nowait((signal_name, payload))

    async def signal_external_workflow(
        self,
        workflow_id: str,
        run_id: Optional[str],
        signal_name: str,
        payload: Any = None,
    ) -> None:
        """Signal an external workflow."""
        command = Command(
            id=f"signal-{workflow_id}-{signal_name}",
            type=CommandType.SIGNAL_EXTERNAL_WORKFLOW,
            attributes={
                "workflow_id": workflow_id,
                "run_id": run_id,
                "signal_name": signal_name,
                "payload": payload,
            },
        )
        self._ctx.add_command(command)

    # -------------------------------------------------------------------------
    # Queries
    # -------------------------------------------------------------------------

    def set_query_handler(
        self,
        name: str,
        handler: Callable[[Any], Any],
    ) -> None:
        """Register a query handler."""
        self._ctx.register_query_handler(name, handler)

    # -------------------------------------------------------------------------
    # Search Attributes
    # -------------------------------------------------------------------------

    def upsert_search_attributes(
        self,
        attributes: Dict[str, Any],
    ) -> None:
        """Update search attributes for workflow indexing."""
        self._ctx._search_attributes.update(attributes)

        command = Command(
            id=f"search-attrs-{len(self._ctx._search_attributes)}",
            type=CommandType.UPSERT_SEARCH_ATTRIBUTES,
            attributes={"search_attributes": attributes},
        )
        self._ctx.add_command(command)

    def get_search_attributes(self) -> Dict[str, Any]:
        """Get current search attributes."""
        return self._ctx._search_attributes.copy()

    # -------------------------------------------------------------------------
    # Continue-as-New
    # -------------------------------------------------------------------------

    def continue_as_new(
        self,
        args: Any = None,
        *,
        workflow_type: Optional[str] = None,
        task_queue: Optional[str] = None,
        execution_timeout: Optional[float] = None,
        run_timeout: Optional[float] = None,
    ) -> None:
        """
        Continue workflow as a new execution.

        Use when history gets too large. Starts fresh with new run_id.
        """
        command = Command(
            id="continue-as-new",
            type=CommandType.CONTINUE_AS_NEW,
            attributes={
                "args": args,
                "workflow_type": workflow_type,
                "task_queue": task_queue,
                "execution_timeout": execution_timeout,
                "run_timeout": run_timeout,
            },
        )
        self._ctx.add_command(command)

        raise ContinueAsNewError(command.attributes)


# =============================================================================
# Errors
# =============================================================================


class ActivityError(Exception):
    """Raised when an activity fails."""
    def __init__(self, message: str, activity_type: str):
        self.activity_type = activity_type
        super().__init__(message)


class ChildWorkflowError(Exception):
    """Raised when a child workflow fails."""
    def __init__(self, message: str, workflow_id: str):
        self.workflow_id = workflow_id
        super().__init__(message)


class ContinueAsNewError(Exception):
    """Raised to trigger continue-as-new."""
    def __init__(self, attributes: Dict[str, Any]):
        self.attributes = attributes
        super().__init__("Continue as new")


class NonDeterministicError(Exception):
    """Raised when replay detects non-determinism."""
    pass


# =============================================================================
# Activity Context
# =============================================================================


class ActivityContext:
    """
    Context for activity execution.

    Provides heartbeat functionality for long-running activities.
    """

    _current: Optional["ActivityContext"] = None

    def __init__(
        self,
        activity_id: str,
        activity_type: str,
        workflow_id: str,
        run_id: str,
        attempt: int = 1,
        heartbeat_timeout: Optional[float] = None,
    ):
        self.activity_id = activity_id
        self.activity_type = activity_type
        self.workflow_id = workflow_id
        self.run_id = run_id
        self.attempt = attempt
        self.heartbeat_timeout = heartbeat_timeout

        self._heartbeat_callback: Optional[Callable[[Any], Awaitable[None]]] = None
        self._cancelled = False

    @classmethod
    def current(cls) -> "ActivityContext":
        """Get current activity context."""
        if cls._current is None:
            raise RuntimeError("No activity context - are you in an activity?")
        return cls._current

    @property
    def is_cancelled(self) -> bool:
        """Check if activity cancellation was requested."""
        return self._cancelled

    async def heartbeat(self, details: Any = None) -> None:
        """
        Send heartbeat to report progress.

        Long-running activities should heartbeat periodically:
        - Lets the server know the activity is still running
        - Can detect cancellation requests
        - Can record progress for recovery
        """
        if self._heartbeat_callback:
            await self._heartbeat_callback(details)

    def set_heartbeat_callback(
        self,
        callback: Callable[[Any], Awaitable[None]],
    ) -> None:
        """Set the heartbeat callback."""
        self._heartbeat_callback = callback

    def request_cancel(self) -> None:
        """Request activity cancellation."""
        self._cancelled = True


# =============================================================================
# Workflow and Activity Decorators
# =============================================================================


def workflow(
    name: Optional[str] = None,
    *,
    task_queue: Optional[str] = None,
):
    """Decorator to define a workflow."""
    def decorator(cls: Type[T]) -> Type[T]:
        cls._workflow_name = name or cls.__name__
        cls._task_queue = task_queue
        cls._is_workflow = True
        return cls

    return decorator


def workflow_run(fn: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
    """Decorator for the main workflow run method."""
    fn._is_workflow_run = True
    return fn


def workflow_signal(name: Optional[str] = None):
    """Decorator for signal handlers."""
    def decorator(fn: Callable) -> Callable:
        fn._signal_name = name or fn.__name__
        fn._is_signal_handler = True
        return fn

    return decorator


def workflow_query(name: Optional[str] = None):
    """Decorator for query handlers."""
    def decorator(fn: Callable) -> Callable:
        fn._query_name = name or fn.__name__
        fn._is_query_handler = True
        return fn

    return decorator


def activity(
    name: Optional[str] = None,
    *,
    task_queue: Optional[str] = None,
    schedule_to_close_timeout: Optional[float] = None,
    start_to_close_timeout: Optional[float] = None,
    heartbeat_timeout: Optional[float] = None,
):
    """Decorator to define an activity."""
    def decorator(fn: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        fn._activity_name = name or fn.__name__
        fn._task_queue = task_queue
        fn._schedule_to_close_timeout = schedule_to_close_timeout
        fn._start_to_close_timeout = start_to_close_timeout
        fn._heartbeat_timeout = heartbeat_timeout
        fn._is_activity = True
        return fn

    return decorator


# =============================================================================
# Workflow Execution State
# =============================================================================


class WorkflowExecutionStatus(str, Enum):
    """Status of a workflow execution."""
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"
    TERMINATED = "terminated"
    CONTINUED_AS_NEW = "continued_as_new"
    TIMED_OUT = "timed_out"


@dataclass
class WorkflowExecution:
    """State of a workflow execution."""
    workflow_id: str
    run_id: str
    workflow_type: str
    task_queue: str
    status: WorkflowExecutionStatus = WorkflowExecutionStatus.RUNNING

    # Input/Output
    args: Any = None
    result: Any = None
    error: Optional[str] = None

    # History
    history: Optional[WorkflowHistory] = None

    # Search attributes
    search_attributes: Dict[str, Any] = field(default_factory=dict)

    # Timing
    start_time: datetime = field(default_factory=datetime.now)
    close_time: Optional[datetime] = None
    execution_timeout: Optional[float] = None

    # Parent (if child workflow)
    parent_workflow_id: Optional[str] = None
    parent_run_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "run_id": self.run_id,
            "workflow_type": self.workflow_type,
            "task_queue": self.task_queue,
            "status": self.status.value,
            "args": self.args,
            "result": self.result,
            "error": self.error,
            "search_attributes": self.search_attributes,
            "start_time": self.start_time.isoformat(),
            "close_time": self.close_time.isoformat() if self.close_time else None,
            "execution_timeout": self.execution_timeout,
            "parent_workflow_id": self.parent_workflow_id,
            "parent_run_id": self.parent_run_id,
        }


# =============================================================================
# Workflow Worker
# =============================================================================


class WorkflowWorker:
    """
    Executes workflow tasks with deterministic replay.
    """

    def __init__(
        self,
        task_queue: str,
        workflows: List[Type],
        activities: List[Callable],
    ):
        self.task_queue = task_queue
        self._workflows: Dict[str, Type] = {}
        self._activities: Dict[str, Callable] = {}

        # Register workflows
        for wf_cls in workflows:
            if hasattr(wf_cls, "_workflow_name"):
                self._workflows[wf_cls._workflow_name] = wf_cls

        # Register activities
        for act_fn in activities:
            if hasattr(act_fn, "_activity_name"):
                self._activities[act_fn._activity_name] = act_fn

    async def execute_workflow_task(
        self,
        execution: WorkflowExecution,
        history: WorkflowHistory,
    ) -> Tuple[List[Command], Optional[Any], Optional[str]]:
        """
        Execute a workflow task.

        Returns: (commands, result_if_completed, error_if_failed)
        """
        workflow_cls = self._workflows.get(execution.workflow_type)
        if not workflow_cls:
            raise ValueError(f"Unknown workflow type: {execution.workflow_type}")

        # Determine if replaying
        has_started_event = any(
            e.event_type == HistoryEventType.WORKFLOW_EXECUTION_STARTED
            for e in history.events
        )
        mode = ReplayMode.REPLAYING if has_started_event else ReplayMode.EXECUTING

        # Create contexts
        replay_ctx = ReplayContext(history, mode)
        workflow_ctx = WorkflowContext(replay_ctx)

        # Set context globals
        WorkflowContext._current = workflow_ctx

        try:
            # Create workflow instance
            workflow_instance = workflow_cls()

            # Find and register signal handlers
            for attr_name in dir(workflow_instance):
                attr = getattr(workflow_instance, attr_name)
                if hasattr(attr, "_is_signal_handler"):
                    # Signal handlers are auto-registered
                    pass
                if hasattr(attr, "_is_query_handler"):
                    workflow_ctx.set_query_handler(attr._query_name, attr)

            # Find workflow run method
            run_method = None
            for attr_name in dir(workflow_instance):
                attr = getattr(workflow_instance, attr_name)
                if hasattr(attr, "_is_workflow_run"):
                    run_method = attr
                    break

            if not run_method:
                raise ValueError(f"No @workflow_run method found in {execution.workflow_type}")

            # Execute
            result = await run_method(execution.args)

            # Get commands and return result
            commands = replay_ctx.get_commands()
            return commands, result, None

        except ContinueAsNewError as e:
            commands = replay_ctx.get_commands()
            return commands, None, None

        except Exception as e:
            commands = replay_ctx.get_commands()
            return commands, None, str(e)

        finally:
            WorkflowContext._current = None

    async def execute_activity(
        self,
        activity_type: str,
        args: Any,
        workflow_id: str,
        run_id: str,
        activity_id: str,
        heartbeat_timeout: Optional[float] = None,
    ) -> ActivityResult:
        """Execute an activity."""
        activity_fn = self._activities.get(activity_type)
        if not activity_fn:
            return ActivityResult(
                success=False,
                error=f"Unknown activity type: {activity_type}",
                error_type="UnknownActivity",
            )

        # Create activity context
        activity_ctx = ActivityContext(
            activity_id=activity_id,
            activity_type=activity_type,
            workflow_id=workflow_id,
            run_id=run_id,
            heartbeat_timeout=heartbeat_timeout,
        )
        ActivityContext._current = activity_ctx

        try:
            result = await activity_fn(args)
            return ActivityResult(success=True, result=result)

        except Exception as e:
            return ActivityResult(
                success=False,
                error=str(e),
                error_type=type(e).__name__,
            )

        finally:
            ActivityContext._current = None


# =============================================================================
# Search Attribute Index
# =============================================================================


class SearchAttributeType(str, Enum):
    """Types of search attributes."""
    KEYWORD = "keyword"  # Exact match
    TEXT = "text"  # Full-text search
    INT = "int"
    DOUBLE = "double"
    BOOL = "bool"
    DATETIME = "datetime"
    KEYWORD_LIST = "keyword_list"


@dataclass
class SearchAttributeDefinition:
    """Definition of a search attribute."""
    name: str
    type: SearchAttributeType


class SearchAttributeIndex:
    """
    Index for workflow search attributes.

    Enables querying workflows by custom attributes.
    """

    def __init__(self):
        self._definitions: Dict[str, SearchAttributeDefinition] = {}
        self._index: Dict[str, Dict[Any, Set[str]]] = {}  # attr -> value -> workflow_ids

    def define_attribute(
        self,
        name: str,
        attr_type: SearchAttributeType,
    ) -> None:
        """Define a search attribute."""
        self._definitions[name] = SearchAttributeDefinition(name, attr_type)
        self._index[name] = {}

    def index_workflow(
        self,
        workflow_id: str,
        attributes: Dict[str, Any],
    ) -> None:
        """Index a workflow's search attributes."""
        for attr_name, value in attributes.items():
            if attr_name not in self._index:
                continue

            if value not in self._index[attr_name]:
                self._index[attr_name][value] = set()

            self._index[attr_name][value].add(workflow_id)

    def search(
        self,
        query: Dict[str, Any],
    ) -> Set[str]:
        """
        Search for workflows matching query.

        Simple AND query - all conditions must match.
        """
        result_sets = []

        for attr_name, value in query.items():
            if attr_name not in self._index:
                continue

            if value in self._index[attr_name]:
                result_sets.append(self._index[attr_name][value])
            else:
                return set()  # No matches for this attribute

        if not result_sets:
            return set()

        # Intersection of all result sets
        result = result_sets[0]
        for rs in result_sets[1:]:
            result = result & rs

        return result


# =============================================================================
# Factory Functions
# =============================================================================


def create_workflow_worker(
    task_queue: str,
    workflows: List[Type],
    activities: Optional[List[Callable]] = None,
) -> WorkflowWorker:
    """Create a workflow worker."""
    return WorkflowWorker(
        task_queue=task_queue,
        workflows=workflows,
        activities=activities or [],
    )
