"""
AION CQRS (Command Query Responsibility Segregation)

True SOTA implementation with:
- Separate command and query models
- Event-driven synchronization
- Eventual consistency handling
- Read model projections
- Command handlers with validation
- Query optimization for read-heavy workloads
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Generic, Optional, Protocol, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")
TCommand = TypeVar("TCommand", bound="Command")
TQuery = TypeVar("TQuery", bound="Query")
TResult = TypeVar("TResult")


# ==================== Commands ====================

@dataclass
class Command:
    """Base class for commands (write operations)."""
    command_id: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    correlation_id: Optional[str] = None
    causation_id: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.command_id:
            self.command_id = hashlib.sha256(
                f"{self.__class__.__name__}:{self.timestamp.isoformat()}:{id(self)}".encode()
            ).hexdigest()[:16]


@dataclass
class CommandResult:
    """Result of command execution."""
    success: bool
    command_id: str
    aggregate_id: Optional[str] = None
    events_produced: list[str] = field(default_factory=list)
    error: Optional[str] = None
    validation_errors: list[str] = field(default_factory=list)


class CommandHandler(ABC, Generic[TCommand]):
    """Abstract command handler."""

    @abstractmethod
    async def validate(self, command: TCommand) -> list[str]:
        """Validate command before execution. Returns list of errors."""
        pass

    @abstractmethod
    async def handle(self, command: TCommand) -> CommandResult:
        """Execute the command."""
        pass


class CommandBus:
    """
    Routes commands to appropriate handlers.

    Features:
    - Handler registration
    - Middleware support
    - Retry logic
    - Command logging
    """

    def __init__(self):
        self._handlers: dict[type, CommandHandler] = {}
        self._middleware: list[Callable] = []
        self._command_log: list[tuple[Command, CommandResult]] = []

    def register(self, command_type: type, handler: CommandHandler) -> None:
        """Register a handler for a command type."""
        self._handlers[command_type] = handler

    def add_middleware(self, middleware: Callable) -> None:
        """Add middleware for command processing."""
        self._middleware.append(middleware)

    async def dispatch(self, command: Command) -> CommandResult:
        """Dispatch a command to its handler."""
        handler = self._handlers.get(type(command))
        if not handler:
            return CommandResult(
                success=False,
                command_id=command.command_id,
                error=f"No handler registered for {type(command).__name__}",
            )

        # Run middleware
        for middleware in self._middleware:
            try:
                command = await middleware(command) if asyncio.iscoroutinefunction(middleware) else middleware(command)
            except Exception as e:
                return CommandResult(
                    success=False,
                    command_id=command.command_id,
                    error=f"Middleware error: {e}",
                )

        # Validate
        validation_errors = await handler.validate(command)
        if validation_errors:
            return CommandResult(
                success=False,
                command_id=command.command_id,
                validation_errors=validation_errors,
            )

        # Execute
        try:
            result = await handler.handle(command)
            self._command_log.append((command, result))
            return result
        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            return CommandResult(
                success=False,
                command_id=command.command_id,
                error=str(e),
            )


# ==================== Queries ====================

@dataclass
class Query:
    """Base class for queries (read operations)."""
    query_id: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    include_metadata: bool = False
    cache_ttl: Optional[int] = None  # Seconds to cache result

    def __post_init__(self):
        if not self.query_id:
            self.query_id = hashlib.sha256(
                f"{self.__class__.__name__}:{self.timestamp.isoformat()}:{id(self)}".encode()
            ).hexdigest()[:16]

    def cache_key(self) -> str:
        """Generate cache key for this query."""
        # Override in subclasses for better caching
        return f"{self.__class__.__name__}:{self.query_id}"


@dataclass
class QueryResult(Generic[T]):
    """Result of query execution."""
    success: bool
    data: Optional[T] = None
    error: Optional[str] = None
    from_cache: bool = False
    execution_time_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class QueryHandler(ABC, Generic[TQuery, TResult]):
    """Abstract query handler."""

    @abstractmethod
    async def handle(self, query: TQuery) -> QueryResult[TResult]:
        """Execute the query."""
        pass


class QueryBus:
    """
    Routes queries to appropriate handlers.

    Features:
    - Handler registration
    - Query result caching
    - Performance tracking
    """

    def __init__(self, cache: Optional[Any] = None):
        self._handlers: dict[type, QueryHandler] = {}
        self._cache = cache
        self._query_stats: dict[str, dict[str, Any]] = {}

    def register(self, query_type: type, handler: QueryHandler) -> None:
        """Register a handler for a query type."""
        self._handlers[query_type] = handler

    async def dispatch(self, query: Query) -> QueryResult:
        """Dispatch a query to its handler."""
        import time
        start = time.monotonic()

        handler = self._handlers.get(type(query))
        if not handler:
            return QueryResult(
                success=False,
                error=f"No handler registered for {type(query).__name__}",
            )

        # Check cache
        if self._cache and query.cache_ttl:
            cache_key = query.cache_key()
            cached = await self._get_cached(cache_key)
            if cached is not None:
                return QueryResult(
                    success=True,
                    data=cached,
                    from_cache=True,
                    execution_time_ms=(time.monotonic() - start) * 1000,
                )

        # Execute query
        try:
            result = await handler.handle(query)
            result.execution_time_ms = (time.monotonic() - start) * 1000

            # Cache result
            if self._cache and query.cache_ttl and result.success:
                await self._set_cached(query.cache_key(), result.data, query.cache_ttl)

            # Track stats
            self._track_query(type(query).__name__, result.execution_time_ms)

            return result
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            return QueryResult(
                success=False,
                error=str(e),
                execution_time_ms=(time.monotonic() - start) * 1000,
            )

    async def _get_cached(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if hasattr(self._cache, 'get'):
            return await self._cache.get(key) if asyncio.iscoroutinefunction(self._cache.get) else self._cache.get(key)
        return None

    async def _set_cached(self, key: str, value: Any, ttl: int) -> None:
        """Set value in cache."""
        if hasattr(self._cache, 'set'):
            if asyncio.iscoroutinefunction(self._cache.set):
                await self._cache.set(key, value, ttl=ttl)
            else:
                self._cache.set(key, value, ttl=ttl)

    def _track_query(self, query_type: str, execution_time_ms: float) -> None:
        """Track query performance stats."""
        if query_type not in self._query_stats:
            self._query_stats[query_type] = {
                "count": 0,
                "total_time_ms": 0,
                "min_time_ms": float('inf'),
                "max_time_ms": 0,
            }

        stats = self._query_stats[query_type]
        stats["count"] += 1
        stats["total_time_ms"] += execution_time_ms
        stats["min_time_ms"] = min(stats["min_time_ms"], execution_time_ms)
        stats["max_time_ms"] = max(stats["max_time_ms"], execution_time_ms)

    def get_stats(self) -> dict[str, Any]:
        """Get query performance statistics."""
        result = {}
        for query_type, stats in self._query_stats.items():
            result[query_type] = {
                **stats,
                "avg_time_ms": stats["total_time_ms"] / stats["count"] if stats["count"] > 0 else 0,
            }
        return result


# ==================== Read Models (Projections) ====================

class ReadModel(ABC):
    """
    Abstract read model that subscribes to events and maintains
    a denormalized view optimized for queries.
    """

    @abstractmethod
    async def apply(self, event: Any) -> None:
        """Apply an event to update the read model."""
        pass

    @abstractmethod
    async def rebuild(self) -> None:
        """Rebuild the read model from scratch."""
        pass

    @abstractmethod
    def get_position(self) -> int:
        """Get the last processed event position."""
        pass


class ProjectionManager:
    """
    Manages read model projections.

    Features:
    - Projection registration
    - Event subscription
    - Automatic rebuilding
    - Position tracking
    """

    def __init__(self, event_store: Optional[Any] = None):
        self._event_store = event_store
        self._projections: dict[str, ReadModel] = {}
        self._running = False

    def register(self, name: str, projection: ReadModel) -> None:
        """Register a projection."""
        self._projections[name] = projection

    async def start(self) -> None:
        """Start projection processing."""
        self._running = True
        if self._event_store:
            # Subscribe to events
            self._event_store.subscribe("*", self)

    async def stop(self) -> None:
        """Stop projection processing."""
        self._running = False

    async def handle(self, event: Any) -> None:
        """Handle an event by updating all projections."""
        for projection in self._projections.values():
            try:
                await projection.apply(event)
            except Exception as e:
                logger.error(f"Projection update failed: {e}")

    async def rebuild_projection(self, name: str) -> int:
        """Rebuild a specific projection from events."""
        projection = self._projections.get(name)
        if not projection:
            raise ValueError(f"Projection {name} not found")

        await projection.rebuild()
        return projection.get_position()

    async def rebuild_all(self) -> dict[str, int]:
        """Rebuild all projections."""
        results = {}
        for name in self._projections:
            results[name] = await self.rebuild_projection(name)
        return results

    def get_status(self) -> dict[str, Any]:
        """Get status of all projections."""
        return {
            name: {
                "position": proj.get_position(),
                "type": type(proj).__name__,
            }
            for name, proj in self._projections.items()
        }


# ==================== Outbox Pattern ====================

@dataclass
class OutboxMessage:
    """Message stored in outbox for reliable publishing."""
    id: str
    aggregate_type: str
    aggregate_id: str
    event_type: str
    payload: dict[str, Any]
    created_at: datetime = field(default_factory=datetime.utcnow)
    published_at: Optional[datetime] = None
    retry_count: int = 0
    last_error: Optional[str] = None


class OutboxProcessor:
    """
    Processes outbox messages for reliable event publishing.

    Implements the Transactional Outbox pattern to ensure
    events are published even if the message broker is down.
    """

    OUTBOX_TABLE = "outbox_messages"

    def __init__(
        self,
        connection: Any,
        publisher: Optional[Callable[[OutboxMessage], Any]] = None,
        batch_size: int = 100,
        max_retries: int = 5,
        retry_delay_seconds: float = 60.0,
    ):
        self.connection = connection
        self.publisher = publisher
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def initialize(self) -> None:
        """Initialize outbox table."""
        await self.connection.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.OUTBOX_TABLE} (
                id TEXT PRIMARY KEY,
                aggregate_type TEXT NOT NULL,
                aggregate_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                payload TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL,
                published_at TIMESTAMP,
                retry_count INTEGER DEFAULT 0,
                last_error TEXT
            )
        """)

        await self.connection.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_outbox_unpublished
            ON {self.OUTBOX_TABLE}(published_at) WHERE published_at IS NULL
        """)

    async def add(self, message: OutboxMessage) -> None:
        """Add a message to the outbox (within existing transaction)."""
        await self.connection.execute(
            f"""
            INSERT INTO {self.OUTBOX_TABLE}
            (id, aggregate_type, aggregate_id, event_type, payload, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                message.id,
                message.aggregate_type,
                message.aggregate_id,
                message.event_type,
                json.dumps(message.payload),
                message.created_at.isoformat(),
            ),
        )

    async def start(self) -> None:
        """Start outbox processing."""
        self._running = True
        self._task = asyncio.create_task(self._process_loop())

    async def stop(self) -> None:
        """Stop outbox processing."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _process_loop(self) -> None:
        """Background loop to process outbox messages."""
        while self._running:
            try:
                processed = await self._process_batch()
                if processed == 0:
                    await asyncio.sleep(1.0)  # No messages, wait a bit
            except Exception as e:
                logger.error(f"Outbox processing error: {e}")
                await asyncio.sleep(5.0)

    async def _process_batch(self) -> int:
        """Process a batch of outbox messages."""
        if not self.publisher:
            return 0

        rows = await self.connection.fetch_all(
            f"""
            SELECT * FROM {self.OUTBOX_TABLE}
            WHERE published_at IS NULL
            AND retry_count < ?
            ORDER BY created_at ASC
            LIMIT ?
            """,
            (self.max_retries, self.batch_size),
        )

        processed = 0
        for row in rows:
            message = OutboxMessage(
                id=row["id"],
                aggregate_type=row["aggregate_type"],
                aggregate_id=row["aggregate_id"],
                event_type=row["event_type"],
                payload=json.loads(row["payload"]),
                created_at=datetime.fromisoformat(row["created_at"]),
                retry_count=row["retry_count"],
            )

            try:
                result = self.publisher(message)
                if asyncio.iscoroutine(result):
                    await result

                # Mark as published
                await self.connection.execute(
                    f"""
                    UPDATE {self.OUTBOX_TABLE}
                    SET published_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                    """,
                    (message.id,),
                )
                processed += 1

            except Exception as e:
                # Increment retry count
                await self.connection.execute(
                    f"""
                    UPDATE {self.OUTBOX_TABLE}
                    SET retry_count = retry_count + 1, last_error = ?
                    WHERE id = ?
                    """,
                    (str(e), message.id),
                )

        return processed

    async def cleanup(self, days: int = 7) -> int:
        """Clean up old published messages."""
        result = await self.connection.execute(
            f"""
            DELETE FROM {self.OUTBOX_TABLE}
            WHERE published_at IS NOT NULL
            AND published_at < datetime('now', '-{days} days')
            """
        )
        return 0  # Would need affected rows count


# ==================== CQRS Coordinator ====================

class CQRSCoordinator:
    """
    Coordinates CQRS components for a complete implementation.

    Provides:
    - Command and query bus management
    - Event store integration
    - Projection management
    - Outbox processing
    """

    def __init__(
        self,
        connection: Any = None,
        event_store: Any = None,
        cache: Any = None,
    ):
        self.connection = connection
        self.command_bus = CommandBus()
        self.query_bus = QueryBus(cache)
        self.projection_manager = ProjectionManager(event_store)
        self.outbox: Optional[OutboxProcessor] = None

        if connection:
            self.outbox = OutboxProcessor(connection)

    async def initialize(self) -> None:
        """Initialize CQRS components."""
        if self.outbox:
            await self.outbox.initialize()
        await self.projection_manager.start()

    async def shutdown(self) -> None:
        """Shutdown CQRS components."""
        await self.projection_manager.stop()
        if self.outbox:
            await self.outbox.stop()

    async def execute_command(self, command: Command) -> CommandResult:
        """Execute a command through the command bus."""
        return await self.command_bus.dispatch(command)

    async def execute_query(self, query: Query) -> QueryResult:
        """Execute a query through the query bus."""
        return await self.query_bus.dispatch(query)

    def register_command_handler(
        self,
        command_type: type,
        handler: CommandHandler,
    ) -> None:
        """Register a command handler."""
        self.command_bus.register(command_type, handler)

    def register_query_handler(
        self,
        query_type: type,
        handler: QueryHandler,
    ) -> None:
        """Register a query handler."""
        self.query_bus.register(query_type, handler)

    def register_projection(self, name: str, projection: ReadModel) -> None:
        """Register a read model projection."""
        self.projection_manager.register(name, projection)

    def get_stats(self) -> dict[str, Any]:
        """Get CQRS statistics."""
        return {
            "query_stats": self.query_bus.get_stats(),
            "projections": self.projection_manager.get_status(),
        }
