"""
AION Event Store - Event Sourcing for Durable Workflows

Implements true event sourcing with:
- Immutable event log
- Deterministic replay
- Snapshotting for performance
- Event versioning
- Concurrent access safety
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar, Union

import structlog

logger = structlog.get_logger(__name__)


class EventType(str, Enum):
    """Types of workflow events."""
    # Workflow lifecycle
    WORKFLOW_STARTED = "workflow.started"
    WORKFLOW_COMPLETED = "workflow.completed"
    WORKFLOW_FAILED = "workflow.failed"
    WORKFLOW_CANCELLED = "workflow.cancelled"
    WORKFLOW_PAUSED = "workflow.paused"
    WORKFLOW_RESUMED = "workflow.resumed"

    # Step lifecycle
    STEP_STARTED = "step.started"
    STEP_COMPLETED = "step.completed"
    STEP_FAILED = "step.failed"
    STEP_SKIPPED = "step.skipped"
    STEP_RETRIED = "step.retried"

    # Action events
    ACTION_INVOKED = "action.invoked"
    ACTION_COMPLETED = "action.completed"
    ACTION_FAILED = "action.failed"

    # Approval events
    APPROVAL_REQUESTED = "approval.requested"
    APPROVAL_GRANTED = "approval.granted"
    APPROVAL_DENIED = "approval.denied"
    APPROVAL_TIMEOUT = "approval.timeout"

    # Timer events
    TIMER_STARTED = "timer.started"
    TIMER_FIRED = "timer.fired"
    TIMER_CANCELLED = "timer.cancelled"

    # Signal events
    SIGNAL_RECEIVED = "signal.received"
    SIGNAL_SENT = "signal.sent"

    # State events
    STATE_CHANGED = "state.changed"
    VARIABLE_SET = "variable.set"

    # Compensation events
    COMPENSATION_STARTED = "compensation.started"
    COMPENSATION_COMPLETED = "compensation.completed"
    COMPENSATION_FAILED = "compensation.failed"

    # Snapshot events
    SNAPSHOT_CREATED = "snapshot.created"


@dataclass
class Event:
    """
    Immutable event in the event store.

    Events are the source of truth for workflow state.
    They are never modified, only appended.
    """
    id: str
    execution_id: str
    sequence_number: int
    event_type: EventType
    timestamp: datetime
    data: Dict[str, Any]

    # Metadata
    version: int = 1
    causation_id: Optional[str] = None  # ID of event that caused this one
    correlation_id: Optional[str] = None  # ID linking related events
    actor: Optional[str] = None  # Who/what generated this event

    # Integrity
    checksum: Optional[str] = None

    def __post_init__(self):
        if self.checksum is None:
            self.checksum = self._compute_checksum()

    def _compute_checksum(self) -> str:
        """Compute SHA-256 checksum for integrity verification."""
        content = f"{self.id}:{self.execution_id}:{self.sequence_number}:{self.event_type}:{json.dumps(self.data, sort_keys=True)}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def verify_integrity(self) -> bool:
        """Verify event hasn't been tampered with."""
        return self.checksum == self._compute_checksum()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize event to dictionary."""
        return {
            "id": self.id,
            "execution_id": self.execution_id,
            "sequence_number": self.sequence_number,
            "event_type": self.event_type.value if isinstance(self.event_type, EventType) else self.event_type,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
            "version": self.version,
            "causation_id": self.causation_id,
            "correlation_id": self.correlation_id,
            "actor": self.actor,
            "checksum": self.checksum,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Event":
        """Deserialize event from dictionary."""
        return cls(
            id=data["id"],
            execution_id=data["execution_id"],
            sequence_number=data["sequence_number"],
            event_type=EventType(data["event_type"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            data=data["data"],
            version=data.get("version", 1),
            causation_id=data.get("causation_id"),
            correlation_id=data.get("correlation_id"),
            actor=data.get("actor"),
            checksum=data.get("checksum"),
        )


@dataclass
class Snapshot:
    """
    Point-in-time snapshot of workflow state.

    Snapshots optimize replay by providing a starting point
    instead of replaying from the beginning.
    """
    id: str
    execution_id: str
    sequence_number: int  # Event sequence this snapshot is valid up to
    timestamp: datetime
    state: Dict[str, Any]
    version: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "execution_id": self.execution_id,
            "sequence_number": self.sequence_number,
            "timestamp": self.timestamp.isoformat(),
            "state": self.state,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Snapshot":
        return cls(
            id=data["id"],
            execution_id=data["execution_id"],
            sequence_number=data["sequence_number"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            state=data["state"],
            version=data.get("version", 1),
        )


class EventStoreBackend(ABC):
    """Abstract base class for event store backends."""

    @abstractmethod
    async def append(self, event: Event) -> None:
        """Append an event to the store."""
        pass

    @abstractmethod
    async def get_events(
        self,
        execution_id: str,
        from_sequence: int = 0,
        to_sequence: Optional[int] = None,
    ) -> List[Event]:
        """Get events for an execution."""
        pass

    @abstractmethod
    async def get_latest_sequence(self, execution_id: str) -> int:
        """Get the latest sequence number for an execution."""
        pass

    @abstractmethod
    async def save_snapshot(self, snapshot: Snapshot) -> None:
        """Save a snapshot."""
        pass

    @abstractmethod
    async def get_latest_snapshot(self, execution_id: str) -> Optional[Snapshot]:
        """Get the latest snapshot for an execution."""
        pass

    @abstractmethod
    async def get_all_execution_ids(self) -> List[str]:
        """Get all execution IDs in the store."""
        pass


class InMemoryEventStore(EventStoreBackend):
    """In-memory event store for development/testing."""

    def __init__(self):
        self._events: Dict[str, List[Event]] = {}
        self._snapshots: Dict[str, List[Snapshot]] = {}
        self._lock = asyncio.Lock()

    async def append(self, event: Event) -> None:
        async with self._lock:
            if event.execution_id not in self._events:
                self._events[event.execution_id] = []
            self._events[event.execution_id].append(event)

    async def get_events(
        self,
        execution_id: str,
        from_sequence: int = 0,
        to_sequence: Optional[int] = None,
    ) -> List[Event]:
        events = self._events.get(execution_id, [])
        filtered = [e for e in events if e.sequence_number >= from_sequence]
        if to_sequence is not None:
            filtered = [e for e in filtered if e.sequence_number <= to_sequence]
        return sorted(filtered, key=lambda e: e.sequence_number)

    async def get_latest_sequence(self, execution_id: str) -> int:
        events = self._events.get(execution_id, [])
        if not events:
            return -1
        return max(e.sequence_number for e in events)

    async def save_snapshot(self, snapshot: Snapshot) -> None:
        async with self._lock:
            if snapshot.execution_id not in self._snapshots:
                self._snapshots[snapshot.execution_id] = []
            self._snapshots[snapshot.execution_id].append(snapshot)

    async def get_latest_snapshot(self, execution_id: str) -> Optional[Snapshot]:
        snapshots = self._snapshots.get(execution_id, [])
        if not snapshots:
            return None
        return max(snapshots, key=lambda s: s.sequence_number)

    async def get_all_execution_ids(self) -> List[str]:
        return list(self._events.keys())


class FileEventStore(EventStoreBackend):
    """File-based event store with append-only logs."""

    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        self.events_path = self.base_path / "events"
        self.snapshots_path = self.base_path / "snapshots"
        self._lock = asyncio.Lock()

        # Create directories
        self.events_path.mkdir(parents=True, exist_ok=True)
        self.snapshots_path.mkdir(parents=True, exist_ok=True)

    def _event_file(self, execution_id: str) -> Path:
        return self.events_path / f"{execution_id}.jsonl"

    def _snapshot_file(self, execution_id: str) -> Path:
        return self.snapshots_path / f"{execution_id}.json"

    async def append(self, event: Event) -> None:
        async with self._lock:
            file_path = self._event_file(event.execution_id)
            with open(file_path, "a") as f:
                f.write(json.dumps(event.to_dict()) + "\n")

    async def get_events(
        self,
        execution_id: str,
        from_sequence: int = 0,
        to_sequence: Optional[int] = None,
    ) -> List[Event]:
        file_path = self._event_file(execution_id)
        if not file_path.exists():
            return []

        events = []
        with open(file_path, "r") as f:
            for line in f:
                if line.strip():
                    event = Event.from_dict(json.loads(line))
                    if event.sequence_number >= from_sequence:
                        if to_sequence is None or event.sequence_number <= to_sequence:
                            events.append(event)

        return sorted(events, key=lambda e: e.sequence_number)

    async def get_latest_sequence(self, execution_id: str) -> int:
        events = await self.get_events(execution_id)
        if not events:
            return -1
        return max(e.sequence_number for e in events)

    async def save_snapshot(self, snapshot: Snapshot) -> None:
        async with self._lock:
            file_path = self._snapshot_file(snapshot.execution_id)

            # Load existing snapshots
            snapshots = []
            if file_path.exists():
                with open(file_path, "r") as f:
                    snapshots = json.load(f)

            # Add new snapshot
            snapshots.append(snapshot.to_dict())

            # Save
            with open(file_path, "w") as f:
                json.dump(snapshots, f, indent=2)

    async def get_latest_snapshot(self, execution_id: str) -> Optional[Snapshot]:
        file_path = self._snapshot_file(execution_id)
        if not file_path.exists():
            return None

        with open(file_path, "r") as f:
            snapshots = json.load(f)

        if not snapshots:
            return None

        latest = max(snapshots, key=lambda s: s["sequence_number"])
        return Snapshot.from_dict(latest)

    async def get_all_execution_ids(self) -> List[str]:
        return [
            f.stem for f in self.events_path.glob("*.jsonl")
        ]


class RedisEventStore(EventStoreBackend):
    """Redis-based event store for distributed deployments."""

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        prefix: str = "aion:events:",
    ):
        self.redis_url = redis_url
        self.prefix = prefix
        self._client = None

    async def _get_client(self):
        if self._client is None:
            try:
                import redis.asyncio as redis
                self._client = redis.from_url(self.redis_url)
            except ImportError:
                raise ImportError("redis package required for RedisEventStore")
        return self._client

    def _events_key(self, execution_id: str) -> str:
        return f"{self.prefix}events:{execution_id}"

    def _snapshots_key(self, execution_id: str) -> str:
        return f"{self.prefix}snapshots:{execution_id}"

    def _sequence_key(self, execution_id: str) -> str:
        return f"{self.prefix}sequence:{execution_id}"

    async def append(self, event: Event) -> None:
        client = await self._get_client()

        # Use Redis list for append-only log
        await client.rpush(
            self._events_key(event.execution_id),
            json.dumps(event.to_dict())
        )

        # Update sequence counter
        await client.set(
            self._sequence_key(event.execution_id),
            event.sequence_number
        )

    async def get_events(
        self,
        execution_id: str,
        from_sequence: int = 0,
        to_sequence: Optional[int] = None,
    ) -> List[Event]:
        client = await self._get_client()

        # Get all events from Redis list
        raw_events = await client.lrange(
            self._events_key(execution_id),
            0, -1
        )

        events = []
        for raw in raw_events:
            event = Event.from_dict(json.loads(raw))
            if event.sequence_number >= from_sequence:
                if to_sequence is None or event.sequence_number <= to_sequence:
                    events.append(event)

        return sorted(events, key=lambda e: e.sequence_number)

    async def get_latest_sequence(self, execution_id: str) -> int:
        client = await self._get_client()
        seq = await client.get(self._sequence_key(execution_id))
        return int(seq) if seq else -1

    async def save_snapshot(self, snapshot: Snapshot) -> None:
        client = await self._get_client()
        await client.rpush(
            self._snapshots_key(snapshot.execution_id),
            json.dumps(snapshot.to_dict())
        )

    async def get_latest_snapshot(self, execution_id: str) -> Optional[Snapshot]:
        client = await self._get_client()

        # Get last snapshot
        raw = await client.lindex(
            self._snapshots_key(execution_id),
            -1
        )

        if not raw:
            return None

        return Snapshot.from_dict(json.loads(raw))

    async def get_all_execution_ids(self) -> List[str]:
        client = await self._get_client()
        keys = await client.keys(f"{self.prefix}events:*")
        return [k.decode().replace(f"{self.prefix}events:", "") for k in keys]


class EventStore:
    """
    Main event store interface with replay capabilities.

    Features:
    - Append-only event log
    - Deterministic replay
    - Snapshotting for performance
    - Event versioning
    - Integrity verification
    """

    def __init__(
        self,
        backend: Optional[EventStoreBackend] = None,
        snapshot_interval: int = 100,  # Create snapshot every N events
    ):
        self.backend = backend or InMemoryEventStore()
        self.snapshot_interval = snapshot_interval

        # Event handlers for projections
        self._projectors: List[Callable[[Event], None]] = []

        # Sequence counters per execution
        self._sequences: Dict[str, int] = {}
        self._sequence_lock = asyncio.Lock()

    async def _next_sequence(self, execution_id: str) -> int:
        """Get next sequence number for an execution."""
        async with self._sequence_lock:
            if execution_id not in self._sequences:
                # Load from backend
                self._sequences[execution_id] = await self.backend.get_latest_sequence(execution_id)

            self._sequences[execution_id] += 1
            return self._sequences[execution_id]

    async def append(
        self,
        execution_id: str,
        event_type: EventType,
        data: Dict[str, Any],
        causation_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        actor: Optional[str] = None,
    ) -> Event:
        """Append a new event to the store."""
        sequence = await self._next_sequence(execution_id)

        event = Event(
            id=str(uuid.uuid4()),
            execution_id=execution_id,
            sequence_number=sequence,
            event_type=event_type,
            timestamp=datetime.now(),
            data=data,
            causation_id=causation_id,
            correlation_id=correlation_id,
            actor=actor,
        )

        await self.backend.append(event)

        # Notify projectors
        for projector in self._projectors:
            try:
                projector(event)
            except Exception as e:
                logger.error("Projector failed", error=str(e))

        # Check if we should create a snapshot
        if sequence > 0 and sequence % self.snapshot_interval == 0:
            await self._create_snapshot(execution_id, sequence)

        logger.debug(
            "Event appended",
            execution_id=execution_id,
            event_type=event_type.value,
            sequence=sequence,
        )

        return event

    async def _create_snapshot(self, execution_id: str, sequence: int) -> None:
        """Create a snapshot for faster replay."""
        # This should be implemented by the workflow engine
        # by providing the current state
        pass

    async def save_snapshot(
        self,
        execution_id: str,
        state: Dict[str, Any],
    ) -> Snapshot:
        """Save a snapshot of current state."""
        sequence = await self.backend.get_latest_sequence(execution_id)

        snapshot = Snapshot(
            id=str(uuid.uuid4()),
            execution_id=execution_id,
            sequence_number=sequence,
            timestamp=datetime.now(),
            state=state,
        )

        await self.backend.save_snapshot(snapshot)

        logger.info(
            "Snapshot created",
            execution_id=execution_id,
            sequence=sequence,
        )

        return snapshot

    async def get_events(
        self,
        execution_id: str,
        from_sequence: int = 0,
        to_sequence: Optional[int] = None,
    ) -> List[Event]:
        """Get events for replay."""
        return await self.backend.get_events(
            execution_id,
            from_sequence,
            to_sequence,
        )

    async def get_events_since_snapshot(
        self,
        execution_id: str,
    ) -> tuple[Optional[Snapshot], List[Event]]:
        """
        Get latest snapshot and subsequent events.

        This is the optimized replay path - start from snapshot
        and only replay events after it.
        """
        snapshot = await self.backend.get_latest_snapshot(execution_id)

        if snapshot:
            events = await self.backend.get_events(
                execution_id,
                from_sequence=snapshot.sequence_number + 1,
            )
        else:
            events = await self.backend.get_events(execution_id)

        return snapshot, events

    def add_projector(self, projector: Callable[[Event], None]) -> None:
        """Add an event projector for live updates."""
        self._projectors.append(projector)

    def remove_projector(self, projector: Callable[[Event], None]) -> None:
        """Remove an event projector."""
        self._projectors.remove(projector)

    async def verify_integrity(self, execution_id: str) -> bool:
        """Verify all events for an execution have valid checksums."""
        events = await self.backend.get_events(execution_id)

        for event in events:
            if not event.verify_integrity():
                logger.error(
                    "Event integrity check failed",
                    execution_id=execution_id,
                    event_id=event.id,
                    sequence=event.sequence_number,
                )
                return False

        return True

    async def get_execution_timeline(
        self,
        execution_id: str,
    ) -> List[Dict[str, Any]]:
        """Get a human-readable timeline of execution events."""
        events = await self.backend.get_events(execution_id)

        timeline = []
        for event in events:
            timeline.append({
                "sequence": event.sequence_number,
                "timestamp": event.timestamp.isoformat(),
                "type": event.event_type.value,
                "summary": self._event_summary(event),
                "data": event.data,
            })

        return timeline

    def _event_summary(self, event: Event) -> str:
        """Generate human-readable summary of an event."""
        data = event.data

        if event.event_type == EventType.WORKFLOW_STARTED:
            return f"Workflow '{data.get('workflow_name', 'unknown')}' started"
        elif event.event_type == EventType.STEP_STARTED:
            return f"Step '{data.get('step_id', 'unknown')}' started"
        elif event.event_type == EventType.STEP_COMPLETED:
            return f"Step '{data.get('step_id', 'unknown')}' completed"
        elif event.event_type == EventType.STEP_FAILED:
            return f"Step '{data.get('step_id', 'unknown')}' failed: {data.get('error', 'unknown')}"
        elif event.event_type == EventType.ACTION_INVOKED:
            return f"Action '{data.get('action_type', 'unknown')}' invoked"
        elif event.event_type == EventType.APPROVAL_REQUESTED:
            return f"Approval requested from {data.get('approvers', [])}"
        elif event.event_type == EventType.APPROVAL_GRANTED:
            return f"Approval granted by {data.get('approver', 'unknown')}"
        elif event.event_type == EventType.COMPENSATION_STARTED:
            return f"Compensation started for step '{data.get('step_id', 'unknown')}'"
        else:
            return event.event_type.value


class WorkflowReplayer:
    """
    Replays workflow execution from event store.

    Features:
    - Deterministic replay
    - Time-travel debugging
    - State reconstruction
    - Partial replay (from snapshot)
    """

    def __init__(self, event_store: EventStore):
        self.event_store = event_store

    async def replay(
        self,
        execution_id: str,
        to_sequence: Optional[int] = None,
        state_handler: Optional[Callable[[Event, Dict], Dict]] = None,
    ) -> Dict[str, Any]:
        """
        Replay execution to reconstruct state.

        Args:
            execution_id: The execution to replay
            to_sequence: Optional sequence to replay up to (for time-travel)
            state_handler: Custom handler to apply events to state

        Returns:
            Reconstructed state
        """
        # Get snapshot and subsequent events
        snapshot, events = await self.event_store.get_events_since_snapshot(
            execution_id
        )

        # Filter events if replaying to specific point
        if to_sequence is not None:
            events = [e for e in events if e.sequence_number <= to_sequence]

        # Start from snapshot state or empty state
        state = snapshot.state.copy() if snapshot else {
            "status": "pending",
            "current_step": None,
            "completed_steps": [],
            "failed_steps": [],
            "variables": {},
            "step_outputs": {},
            "error": None,
        }

        # Apply events
        for event in events:
            if state_handler:
                state = state_handler(event, state)
            else:
                state = self._apply_event(event, state)

        return state

    def _apply_event(self, event: Event, state: Dict[str, Any]) -> Dict[str, Any]:
        """Apply an event to state (default handler)."""
        data = event.data

        if event.event_type == EventType.WORKFLOW_STARTED:
            state["status"] = "running"
            state["started_at"] = event.timestamp.isoformat()
            state["workflow_id"] = data.get("workflow_id")
            state["inputs"] = data.get("inputs", {})

        elif event.event_type == EventType.WORKFLOW_COMPLETED:
            state["status"] = "completed"
            state["completed_at"] = event.timestamp.isoformat()
            state["result"] = data.get("result")

        elif event.event_type == EventType.WORKFLOW_FAILED:
            state["status"] = "failed"
            state["failed_at"] = event.timestamp.isoformat()
            state["error"] = data.get("error")

        elif event.event_type == EventType.WORKFLOW_CANCELLED:
            state["status"] = "cancelled"
            state["cancelled_at"] = event.timestamp.isoformat()
            state["cancel_reason"] = data.get("reason")

        elif event.event_type == EventType.STEP_STARTED:
            state["current_step"] = data.get("step_id")

        elif event.event_type == EventType.STEP_COMPLETED:
            step_id = data.get("step_id")
            if step_id not in state["completed_steps"]:
                state["completed_steps"].append(step_id)
            state["step_outputs"][step_id] = data.get("output")
            if state["current_step"] == step_id:
                state["current_step"] = None

        elif event.event_type == EventType.STEP_FAILED:
            step_id = data.get("step_id")
            if step_id not in state["failed_steps"]:
                state["failed_steps"].append(step_id)
            state["step_errors"] = state.get("step_errors", {})
            state["step_errors"][step_id] = data.get("error")

        elif event.event_type == EventType.VARIABLE_SET:
            state["variables"][data.get("name")] = data.get("value")

        elif event.event_type == EventType.APPROVAL_REQUESTED:
            state["pending_approval"] = {
                "request_id": data.get("request_id"),
                "step_id": data.get("step_id"),
                "approvers": data.get("approvers"),
            }

        elif event.event_type == EventType.APPROVAL_GRANTED:
            state.pop("pending_approval", None)

        elif event.event_type == EventType.APPROVAL_DENIED:
            state.pop("pending_approval", None)
            state["approval_denied"] = True

        elif event.event_type == EventType.COMPENSATION_STARTED:
            state["compensating"] = True
            state["compensation_step"] = data.get("step_id")

        elif event.event_type == EventType.COMPENSATION_COMPLETED:
            state["compensated_steps"] = state.get("compensated_steps", [])
            state["compensated_steps"].append(data.get("step_id"))

        return state

    async def get_state_at(
        self,
        execution_id: str,
        timestamp: datetime,
    ) -> Dict[str, Any]:
        """Get state at a specific point in time."""
        events = await self.event_store.get_events(execution_id)

        # Find the sequence number at that timestamp
        target_sequence = None
        for event in events:
            if event.timestamp <= timestamp:
                target_sequence = event.sequence_number
            else:
                break

        if target_sequence is None:
            return {}

        return await self.replay(execution_id, to_sequence=target_sequence)

    async def diff_states(
        self,
        execution_id: str,
        from_sequence: int,
        to_sequence: int,
    ) -> Dict[str, Any]:
        """Get the difference between two states."""
        state_from = await self.replay(execution_id, to_sequence=from_sequence)
        state_to = await self.replay(execution_id, to_sequence=to_sequence)

        # Simple diff
        diff = {
            "from_sequence": from_sequence,
            "to_sequence": to_sequence,
            "changes": {},
        }

        all_keys = set(state_from.keys()) | set(state_to.keys())
        for key in all_keys:
            old_val = state_from.get(key)
            new_val = state_to.get(key)
            if old_val != new_val:
                diff["changes"][key] = {
                    "old": old_val,
                    "new": new_val,
                }

        return diff
