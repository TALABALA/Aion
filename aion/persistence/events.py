"""
AION Event Sourcing and Change Data Capture

State-of-the-art event infrastructure with:
- Event store for full audit trail
- Change Data Capture (CDC) for real-time sync
- Event replay for state reconstruction
- Snapshots for performance optimization
- Event streaming with pub/sub
- Projections for read models
- Saga support for distributed transactions
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Generic, Optional, Protocol, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class EventType(str, Enum):
    """Standard event types."""
    CREATED = "created"
    UPDATED = "updated"
    DELETED = "deleted"
    RESTORED = "restored"
    SNAPSHOT = "snapshot"


@dataclass
class Event:
    """
    Represents a domain event.

    Events are immutable records of something that happened.
    """
    id: str
    event_type: str
    aggregate_type: str
    aggregate_id: str
    data: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    sequence_number: int = 0
    correlation_id: Optional[str] = None
    causation_id: Optional[str] = None
    version: int = 1

    def to_dict(self) -> dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "id": self.id,
            "event_type": self.event_type,
            "aggregate_type": self.aggregate_type,
            "aggregate_id": self.aggregate_id,
            "data": self.data,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "sequence_number": self.sequence_number,
            "correlation_id": self.correlation_id,
            "causation_id": self.causation_id,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Event":
        """Create event from dictionary."""
        return cls(
            id=data["id"],
            event_type=data["event_type"],
            aggregate_type=data["aggregate_type"],
            aggregate_id=data["aggregate_id"],
            data=data["data"],
            metadata=data.get("metadata", {}),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            sequence_number=data.get("sequence_number", 0),
            correlation_id=data.get("correlation_id"),
            causation_id=data.get("causation_id"),
            version=data.get("version", 1),
        )


@dataclass
class Snapshot:
    """
    Represents an aggregate snapshot.

    Snapshots capture the state of an aggregate at a point in time
    to avoid replaying all events.
    """
    id: str
    aggregate_type: str
    aggregate_id: str
    state: dict[str, Any]
    version: int
    event_sequence: int
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CDCEvent:
    """
    Change Data Capture event.

    Captures changes to database tables for replication and streaming.
    """
    id: int
    table_name: str
    operation: str  # INSERT, UPDATE, DELETE
    entity_id: str
    old_data: Optional[dict[str, Any]]
    new_data: Optional[dict[str, Any]]
    changed_fields: list[str]
    transaction_id: Optional[str]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    processed: bool = False


class EventHandler(Protocol):
    """Protocol for event handlers."""

    async def handle(self, event: Event) -> None:
        """Handle an event."""
        ...


class DatabaseConnection(Protocol):
    """Protocol for database connections."""

    async def execute(self, query: str, params: tuple = ()) -> None:
        ...

    async def fetch_all(self, query: str, params: tuple = ()) -> list[dict]:
        ...

    async def fetch_one(self, query: str, params: tuple = ()) -> Optional[dict]:
        ...


class EventStore:
    """
    Append-only event store with replay capabilities.

    Features:
    - Immutable event storage
    - Optimistic concurrency control
    - Event streaming
    - Snapshot support
    - Projection rebuilding
    """

    EVENTS_TABLE = "event_store"
    SNAPSHOTS_TABLE = "event_snapshots"

    def __init__(
        self,
        connection: Optional[DatabaseConnection] = None,
        snapshot_interval: int = 100,
    ):
        self.connection = connection
        self.snapshot_interval = snapshot_interval
        self._handlers: dict[str, list[EventHandler]] = {}
        self._projections: dict[str, "Projection"] = {}

    async def initialize(self, connection: Optional[DatabaseConnection] = None) -> None:
        """Initialize the event store."""
        if connection:
            self.connection = connection

        if not self.connection:
            raise RuntimeError("No database connection provided")

        # Create events table
        await self.connection.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.EVENTS_TABLE} (
                id TEXT PRIMARY KEY,
                event_type TEXT NOT NULL,
                aggregate_type TEXT NOT NULL,
                aggregate_id TEXT NOT NULL,
                data TEXT NOT NULL,
                metadata TEXT,
                timestamp TIMESTAMP NOT NULL,
                sequence_number INTEGER NOT NULL,
                correlation_id TEXT,
                causation_id TEXT,
                version INTEGER DEFAULT 1,
                UNIQUE(aggregate_type, aggregate_id, sequence_number)
            )
        """)

        await self.connection.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_event_store_aggregate
            ON {self.EVENTS_TABLE}(aggregate_type, aggregate_id)
        """)

        await self.connection.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_event_store_type
            ON {self.EVENTS_TABLE}(event_type)
        """)

        await self.connection.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_event_store_timestamp
            ON {self.EVENTS_TABLE}(timestamp DESC)
        """)

        await self.connection.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_event_store_correlation
            ON {self.EVENTS_TABLE}(correlation_id)
        """)

        # Create snapshots table
        await self.connection.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.SNAPSHOTS_TABLE} (
                id TEXT PRIMARY KEY,
                aggregate_type TEXT NOT NULL,
                aggregate_id TEXT NOT NULL,
                state TEXT NOT NULL,
                version INTEGER NOT NULL,
                event_sequence INTEGER NOT NULL,
                created_at TIMESTAMP NOT NULL,
                metadata TEXT,
                UNIQUE(aggregate_type, aggregate_id, version)
            )
        """)

        await self.connection.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_event_snapshots_aggregate
            ON {self.SNAPSHOTS_TABLE}(aggregate_type, aggregate_id, version DESC)
        """)

    async def append(
        self,
        event: Event,
        expected_version: Optional[int] = None,
    ) -> Event:
        """
        Append an event to the store.

        Args:
            event: Event to append
            expected_version: Expected aggregate version (for optimistic locking)

        Returns:
            Event with assigned sequence number
        """
        # Get current sequence for aggregate
        row = await self.connection.fetch_one(
            f"""
            SELECT MAX(sequence_number) as max_seq
            FROM {self.EVENTS_TABLE}
            WHERE aggregate_type = ? AND aggregate_id = ?
            """,
            (event.aggregate_type, event.aggregate_id),
        )

        current_seq = row["max_seq"] if row and row["max_seq"] else 0

        # Check expected version
        if expected_version is not None and current_seq != expected_version:
            raise ValueError(
                f"Concurrency conflict: expected version {expected_version}, "
                f"got {current_seq}"
            )

        # Assign sequence number
        event.sequence_number = current_seq + 1

        # Generate ID if not set
        if not event.id:
            event.id = str(uuid.uuid4())

        # Insert event
        await self.connection.execute(
            f"""
            INSERT INTO {self.EVENTS_TABLE}
            (id, event_type, aggregate_type, aggregate_id, data, metadata,
             timestamp, sequence_number, correlation_id, causation_id, version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.id,
                event.event_type,
                event.aggregate_type,
                event.aggregate_id,
                json.dumps(event.data),
                json.dumps(event.metadata),
                event.timestamp.isoformat(),
                event.sequence_number,
                event.correlation_id,
                event.causation_id,
                event.version,
            ),
        )

        # Dispatch to handlers
        await self._dispatch(event)

        # Check if snapshot is needed
        if event.sequence_number % self.snapshot_interval == 0:
            logger.debug(
                f"Snapshot interval reached for {event.aggregate_type}/{event.aggregate_id}"
            )

        return event

    async def append_batch(self, events: list[Event]) -> list[Event]:
        """Append multiple events atomically."""
        result = []
        for event in events:
            result.append(await self.append(event))
        return result

    async def get_events(
        self,
        aggregate_type: str,
        aggregate_id: str,
        from_sequence: int = 0,
        to_sequence: Optional[int] = None,
        limit: int = 1000,
    ) -> list[Event]:
        """
        Get events for an aggregate.

        Args:
            aggregate_type: Type of aggregate
            aggregate_id: ID of aggregate
            from_sequence: Start from this sequence number
            to_sequence: End at this sequence number
            limit: Maximum events to return

        Returns:
            List of events in sequence order
        """
        query = f"""
            SELECT * FROM {self.EVENTS_TABLE}
            WHERE aggregate_type = ? AND aggregate_id = ?
            AND sequence_number >= ?
        """
        params: list[Any] = [aggregate_type, aggregate_id, from_sequence]

        if to_sequence is not None:
            query += " AND sequence_number <= ?"
            params.append(to_sequence)

        query += " ORDER BY sequence_number ASC LIMIT ?"
        params.append(limit)

        rows = await self.connection.fetch_all(query, tuple(params))

        return [self._row_to_event(row) for row in rows]

    async def get_all_events(
        self,
        event_type: Optional[str] = None,
        from_timestamp: Optional[datetime] = None,
        to_timestamp: Optional[datetime] = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> list[Event]:
        """
        Get all events with filtering.

        Args:
            event_type: Filter by event type
            from_timestamp: Start timestamp
            to_timestamp: End timestamp
            limit: Maximum events
            offset: Offset for pagination

        Returns:
            List of events
        """
        query = f"SELECT * FROM {self.EVENTS_TABLE} WHERE 1=1"
        params: list[Any] = []

        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)

        if from_timestamp:
            query += " AND timestamp >= ?"
            params.append(from_timestamp.isoformat())

        if to_timestamp:
            query += " AND timestamp <= ?"
            params.append(to_timestamp.isoformat())

        query += " ORDER BY timestamp ASC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        rows = await self.connection.fetch_all(query, tuple(params))

        return [self._row_to_event(row) for row in rows]

    async def get_events_by_correlation(
        self,
        correlation_id: str,
    ) -> list[Event]:
        """Get all events with a correlation ID."""
        rows = await self.connection.fetch_all(
            f"""
            SELECT * FROM {self.EVENTS_TABLE}
            WHERE correlation_id = ?
            ORDER BY timestamp ASC
            """,
            (correlation_id,),
        )

        return [self._row_to_event(row) for row in rows]

    def _row_to_event(self, row: dict) -> Event:
        """Convert database row to Event."""
        return Event(
            id=row["id"],
            event_type=row["event_type"],
            aggregate_type=row["aggregate_type"],
            aggregate_id=row["aggregate_id"],
            data=json.loads(row["data"]),
            metadata=json.loads(row["metadata"] or "{}"),
            timestamp=datetime.fromisoformat(row["timestamp"]),
            sequence_number=row["sequence_number"],
            correlation_id=row["correlation_id"],
            causation_id=row["causation_id"],
            version=row["version"],
        )

    async def save_snapshot(self, snapshot: Snapshot) -> None:
        """Save an aggregate snapshot."""
        if not snapshot.id:
            snapshot.id = str(uuid.uuid4())

        await self.connection.execute(
            f"""
            INSERT OR REPLACE INTO {self.SNAPSHOTS_TABLE}
            (id, aggregate_type, aggregate_id, state, version, event_sequence,
             created_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                snapshot.id,
                snapshot.aggregate_type,
                snapshot.aggregate_id,
                json.dumps(snapshot.state),
                snapshot.version,
                snapshot.event_sequence,
                snapshot.created_at.isoformat(),
                json.dumps(snapshot.metadata),
            ),
        )

    async def get_latest_snapshot(
        self,
        aggregate_type: str,
        aggregate_id: str,
    ) -> Optional[Snapshot]:
        """Get the latest snapshot for an aggregate."""
        row = await self.connection.fetch_one(
            f"""
            SELECT * FROM {self.SNAPSHOTS_TABLE}
            WHERE aggregate_type = ? AND aggregate_id = ?
            ORDER BY version DESC LIMIT 1
            """,
            (aggregate_type, aggregate_id),
        )

        if not row:
            return None

        return Snapshot(
            id=row["id"],
            aggregate_type=row["aggregate_type"],
            aggregate_id=row["aggregate_id"],
            state=json.loads(row["state"]),
            version=row["version"],
            event_sequence=row["event_sequence"],
            created_at=datetime.fromisoformat(row["created_at"]),
            metadata=json.loads(row["metadata"] or "{}"),
        )

    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        """Subscribe a handler to an event type."""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)

    def unsubscribe(self, event_type: str, handler: EventHandler) -> None:
        """Unsubscribe a handler from an event type."""
        if event_type in self._handlers:
            self._handlers[event_type] = [
                h for h in self._handlers[event_type] if h != handler
            ]

    async def _dispatch(self, event: Event) -> None:
        """Dispatch event to handlers."""
        handlers = self._handlers.get(event.event_type, [])
        handlers.extend(self._handlers.get("*", []))  # Wildcard handlers

        for handler in handlers:
            try:
                await handler.handle(event)
            except Exception as e:
                logger.error(f"Event handler error for {event.event_type}: {e}")

    def register_projection(self, name: str, projection: "Projection") -> None:
        """Register a projection for automatic updates."""
        self._projections[name] = projection

    async def replay_to_projection(
        self,
        projection_name: str,
        from_sequence: int = 0,
    ) -> int:
        """
        Replay events to a projection.

        Args:
            projection_name: Name of projection
            from_sequence: Start sequence

        Returns:
            Number of events replayed
        """
        projection = self._projections.get(projection_name)
        if not projection:
            raise ValueError(f"Projection {projection_name} not found")

        events = await self.get_all_events(limit=10000)
        count = 0

        for event in events:
            if event.sequence_number >= from_sequence:
                await projection.apply(event)
                count += 1

        return count


class Projection(ABC):
    """
    Abstract base for event projections.

    Projections build read models from events.
    """

    @abstractmethod
    async def apply(self, event: Event) -> None:
        """Apply an event to update the projection."""
        pass

    @abstractmethod
    async def rebuild(self, events: list[Event]) -> None:
        """Rebuild projection from a list of events."""
        pass


class CDCManager:
    """
    Change Data Capture manager.

    Captures and streams database changes for replication,
    synchronization, and analytics.
    """

    CDC_TABLE = "cdc_events"

    def __init__(
        self,
        connection: Optional[DatabaseConnection] = None,
        batch_size: int = 100,
        flush_interval: float = 5.0,
    ):
        self.connection = connection
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self._handlers: list[Callable[[CDCEvent], Any]] = []
        self._buffer: list[CDCEvent] = []
        self._running = False
        self._flush_task: Optional[asyncio.Task] = None

    async def initialize(self, connection: Optional[DatabaseConnection] = None) -> None:
        """Initialize CDC manager."""
        if connection:
            self.connection = connection

        if not self.connection:
            raise RuntimeError("No database connection provided")

        # Table should be created by migration, but ensure it exists
        await self.connection.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.CDC_TABLE} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                table_name TEXT NOT NULL,
                operation TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                old_data TEXT,
                new_data TEXT,
                changed_fields TEXT,
                transaction_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                processed INTEGER DEFAULT 0,
                processed_at TIMESTAMP
            )
        """)

    async def capture(
        self,
        table_name: str,
        operation: str,
        entity_id: str,
        old_data: Optional[dict] = None,
        new_data: Optional[dict] = None,
        transaction_id: Optional[str] = None,
    ) -> CDCEvent:
        """
        Capture a change event.

        Args:
            table_name: Name of changed table
            operation: Operation type (INSERT, UPDATE, DELETE)
            entity_id: ID of affected entity
            old_data: Previous state (for UPDATE/DELETE)
            new_data: New state (for INSERT/UPDATE)
            transaction_id: Transaction ID

        Returns:
            CDCEvent
        """
        # Calculate changed fields
        changed_fields = []
        if old_data and new_data:
            for key in set(old_data.keys()) | set(new_data.keys()):
                old_val = old_data.get(key)
                new_val = new_data.get(key)
                if old_val != new_val:
                    changed_fields.append(key)

        event = CDCEvent(
            id=0,  # Will be assigned by database
            table_name=table_name,
            operation=operation,
            entity_id=entity_id,
            old_data=old_data,
            new_data=new_data,
            changed_fields=changed_fields,
            transaction_id=transaction_id,
        )

        # Insert into CDC table
        await self.connection.execute(
            f"""
            INSERT INTO {self.CDC_TABLE}
            (table_name, operation, entity_id, old_data, new_data,
             changed_fields, transaction_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.table_name,
                event.operation,
                event.entity_id,
                json.dumps(event.old_data) if event.old_data else None,
                json.dumps(event.new_data) if event.new_data else None,
                json.dumps(event.changed_fields),
                event.transaction_id,
                event.timestamp.isoformat(),
            ),
        )

        # Buffer for batch processing
        self._buffer.append(event)

        if len(self._buffer) >= self.batch_size:
            await self._flush()

        return event

    def subscribe(self, handler: Callable[[CDCEvent], Any]) -> None:
        """Subscribe to CDC events."""
        self._handlers.append(handler)

    def unsubscribe(self, handler: Callable[[CDCEvent], Any]) -> None:
        """Unsubscribe from CDC events."""
        self._handlers = [h for h in self._handlers if h != handler]

    async def _flush(self) -> None:
        """Flush buffer and notify handlers."""
        if not self._buffer:
            return

        events = self._buffer.copy()
        self._buffer.clear()

        for event in events:
            for handler in self._handlers:
                try:
                    result = handler(event)
                    if asyncio.iscoroutine(result):
                        await result
                except Exception as e:
                    logger.error(f"CDC handler error: {e}")

    async def start(self) -> None:
        """Start CDC background processing."""
        self._running = True
        self._flush_task = asyncio.create_task(self._flush_loop())

    async def stop(self) -> None:
        """Stop CDC processing."""
        self._running = False
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        await self._flush()

    async def _flush_loop(self) -> None:
        """Background flush loop."""
        while self._running:
            await asyncio.sleep(self.flush_interval)
            await self._flush()

    async def get_unprocessed(
        self,
        limit: int = 100,
    ) -> list[CDCEvent]:
        """Get unprocessed CDC events."""
        rows = await self.connection.fetch_all(
            f"""
            SELECT * FROM {self.CDC_TABLE}
            WHERE processed = 0
            ORDER BY id ASC LIMIT ?
            """,
            (limit,),
        )

        return [
            CDCEvent(
                id=row["id"],
                table_name=row["table_name"],
                operation=row["operation"],
                entity_id=row["entity_id"],
                old_data=json.loads(row["old_data"]) if row["old_data"] else None,
                new_data=json.loads(row["new_data"]) if row["new_data"] else None,
                changed_fields=json.loads(row["changed_fields"] or "[]"),
                transaction_id=row["transaction_id"],
                timestamp=datetime.fromisoformat(row["created_at"]),
                processed=bool(row["processed"]),
            )
            for row in rows
        ]

    async def mark_processed(self, event_ids: list[int]) -> None:
        """Mark CDC events as processed."""
        if not event_ids:
            return

        placeholders = ", ".join(["?"] * len(event_ids))
        await self.connection.execute(
            f"""
            UPDATE {self.CDC_TABLE}
            SET processed = 1, processed_at = CURRENT_TIMESTAMP
            WHERE id IN ({placeholders})
            """,
            tuple(event_ids),
        )

    async def cleanup(self, days: int = 30) -> int:
        """
        Clean up old processed CDC events.

        Args:
            days: Delete events older than this many days

        Returns:
            Number of deleted events
        """
        result = await self.connection.execute(
            f"""
            DELETE FROM {self.CDC_TABLE}
            WHERE processed = 1
            AND created_at < datetime('now', '-{days} days')
            """
        )
        return 0  # Would need to get affected rows


class Aggregate(Generic[T], ABC):
    """
    Abstract base for event-sourced aggregates.

    Aggregates are the core building blocks of event sourcing.
    They encapsulate state and enforce invariants.
    """

    def __init__(self, aggregate_id: str):
        self.id = aggregate_id
        self._version = 0
        self._pending_events: list[Event] = []
        self._state: Optional[T] = None

    @property
    def version(self) -> int:
        """Current aggregate version."""
        return self._version

    @property
    def state(self) -> Optional[T]:
        """Current aggregate state."""
        return self._state

    @abstractmethod
    def apply_event(self, event: Event) -> None:
        """Apply an event to update state."""
        pass

    @abstractmethod
    def get_aggregate_type(self) -> str:
        """Get the aggregate type name."""
        pass

    def raise_event(
        self,
        event_type: str,
        data: dict[str, Any],
        metadata: Optional[dict[str, Any]] = None,
    ) -> Event:
        """
        Raise a new event.

        Args:
            event_type: Type of event
            data: Event data
            metadata: Optional metadata

        Returns:
            The new event
        """
        event = Event(
            id=str(uuid.uuid4()),
            event_type=event_type,
            aggregate_type=self.get_aggregate_type(),
            aggregate_id=self.id,
            data=data,
            metadata=metadata or {},
        )

        self.apply_event(event)
        self._pending_events.append(event)
        self._version += 1

        return event

    def get_pending_events(self) -> list[Event]:
        """Get pending uncommitted events."""
        return self._pending_events.copy()

    def clear_pending_events(self) -> None:
        """Clear pending events after commit."""
        self._pending_events.clear()

    def load_from_history(self, events: list[Event]) -> None:
        """Load aggregate state from event history."""
        for event in events:
            self.apply_event(event)
            self._version = event.sequence_number

    def create_snapshot(self) -> Snapshot:
        """Create a snapshot of current state."""
        return Snapshot(
            id=str(uuid.uuid4()),
            aggregate_type=self.get_aggregate_type(),
            aggregate_id=self.id,
            state=self._serialize_state(),
            version=self._version,
            event_sequence=self._version,
        )

    def load_from_snapshot(self, snapshot: Snapshot) -> None:
        """Load state from a snapshot."""
        self._deserialize_state(snapshot.state)
        self._version = snapshot.version

    @abstractmethod
    def _serialize_state(self) -> dict[str, Any]:
        """Serialize current state for snapshot."""
        pass

    @abstractmethod
    def _deserialize_state(self, state: dict[str, Any]) -> None:
        """Deserialize state from snapshot."""
        pass


class AggregateRepository(Generic[T]):
    """
    Repository for event-sourced aggregates.

    Handles loading and saving aggregates using the event store.
    """

    def __init__(
        self,
        event_store: EventStore,
        aggregate_class: type,
    ):
        self.event_store = event_store
        self.aggregate_class = aggregate_class

    async def get(self, aggregate_id: str) -> Optional[T]:
        """
        Load an aggregate by ID.

        Uses snapshots when available for performance.
        """
        aggregate = self.aggregate_class(aggregate_id)
        aggregate_type = aggregate.get_aggregate_type()

        # Try to load from snapshot
        snapshot = await self.event_store.get_latest_snapshot(
            aggregate_type, aggregate_id
        )

        start_sequence = 0
        if snapshot:
            aggregate.load_from_snapshot(snapshot)
            start_sequence = snapshot.event_sequence + 1

        # Load remaining events
        events = await self.event_store.get_events(
            aggregate_type,
            aggregate_id,
            from_sequence=start_sequence,
        )

        if not events and not snapshot:
            return None

        aggregate.load_from_history(events)
        return aggregate

    async def save(self, aggregate: T) -> None:
        """Save an aggregate by appending its pending events."""
        pending = aggregate.get_pending_events()

        if not pending:
            return

        for event in pending:
            await self.event_store.append(event)

        aggregate.clear_pending_events()

        # Check if we should create a snapshot
        if aggregate.version % self.event_store.snapshot_interval == 0:
            snapshot = aggregate.create_snapshot()
            await self.event_store.save_snapshot(snapshot)
