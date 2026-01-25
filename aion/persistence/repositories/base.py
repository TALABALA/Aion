"""
AION Base Repository

Abstract base repository with comprehensive features:
- Type-safe CRUD operations
- Batch operations
- Optimistic locking
- Query builders
- Caching integration
- Event emission for CDC
"""

from __future__ import annotations

import json
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Generic,
    Optional,
    TypeVar,
    Union,
)

import structlog

from aion.persistence.database import DatabaseManager
from aion.persistence.backends.redis_cache import CacheManager

logger = structlog.get_logger(__name__)

T = TypeVar("T")


class ChangeType(str, Enum):
    """Types of data changes for CDC."""
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"


@dataclass
class ChangeEvent:
    """Change event for change data capture."""
    id: str
    entity_type: str
    entity_id: str
    change_type: ChangeType
    old_data: Optional[dict[str, Any]] = None
    new_data: Optional[dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryOptions:
    """Options for database queries."""
    limit: int = 100
    offset: int = 0
    order_by: str = "created_at DESC"
    include_deleted: bool = False
    use_cache: bool = True
    cache_ttl: Optional[int] = None


@dataclass
class BatchResult:
    """Result of a batch operation."""
    total: int
    successful: int
    failed: int
    errors: list[tuple[str, str]] = field(default_factory=list)


class BaseRepository(ABC, Generic[T]):
    """
    Abstract base repository with comprehensive CRUD operations.

    Features:
    - Type-safe entity operations
    - Batch create/update/delete
    - Optimistic locking support
    - Query builders
    - Caching integration
    - Change data capture
    """

    def __init__(
        self,
        db: DatabaseManager,
        cache: Optional[CacheManager] = None,
    ):
        self.db = db
        self.cache = cache
        self._table_name: str = ""
        self._primary_key: str = "id"
        self._version_column: Optional[str] = None  # For optimistic locking
        self._soft_delete_column: Optional[str] = "deleted_at"
        self._change_handlers: list[Callable[[ChangeEvent], None]] = []

    @property
    def table_name(self) -> str:
        return self._table_name

    # === Abstract Methods ===

    @abstractmethod
    def _serialize(self, entity: T) -> dict[str, Any]:
        """Serialize entity to database row."""
        pass

    @abstractmethod
    def _deserialize(self, row: dict[str, Any]) -> T:
        """Deserialize database row to entity."""
        pass

    # === CRUD Operations ===

    async def get(self, id: str, options: Optional[QueryOptions] = None) -> Optional[T]:
        """Get entity by ID."""
        options = options or QueryOptions()

        # Try cache first
        if self.cache and options.use_cache:
            cache_key = self._cache_key(id)
            cached = await self.cache.get(cache_key)
            if cached is not None:
                return self._deserialize(cached)

        # Query database
        where = f"{self._primary_key} = ?"
        params = [id]

        if self._soft_delete_column and not options.include_deleted:
            where += f" AND {self._soft_delete_column} IS NULL"

        query = f"SELECT * FROM {self._table_name} WHERE {where}"
        row = await self.db.fetch_one(query, tuple(params))

        if row is None:
            return None

        # Update cache
        if self.cache and options.use_cache:
            await self.cache.set(
                cache_key,
                dict(row),
                ttl=options.cache_ttl,
            )

        return self._deserialize(row)

    async def get_many(
        self,
        ids: list[str],
        options: Optional[QueryOptions] = None,
    ) -> dict[str, T]:
        """Get multiple entities by IDs."""
        options = options or QueryOptions()
        result: dict[str, T] = {}

        # Try cache first
        if self.cache and options.use_cache:
            cache_keys = {id: self._cache_key(id) for id in ids}
            cached = await self.cache._local_cache._cache  # Direct access for batch
            uncached_ids = []

            for id, key in cache_keys.items():
                if key in cached and not cached[key].is_expired():
                    result[id] = self._deserialize(cached[key].value)
                else:
                    uncached_ids.append(id)

            if not uncached_ids:
                return result

            ids = uncached_ids

        # Query database
        if not ids:
            return result

        placeholders = ", ".join(["?" for _ in ids])
        where = f"{self._primary_key} IN ({placeholders})"

        if self._soft_delete_column and not options.include_deleted:
            where += f" AND {self._soft_delete_column} IS NULL"

        query = f"SELECT * FROM {self._table_name} WHERE {where}"
        rows = await self.db.fetch_all(query, tuple(ids))

        for row in rows:
            entity_id = row[self._primary_key]
            result[entity_id] = self._deserialize(row)

            # Update cache
            if self.cache and options.use_cache:
                await self.cache.set(
                    self._cache_key(entity_id),
                    dict(row),
                    ttl=options.cache_ttl,
                )

        return result

    async def get_all(self, options: Optional[QueryOptions] = None) -> list[T]:
        """Get all entities with pagination."""
        options = options or QueryOptions()

        where = "1=1"
        if self._soft_delete_column and not options.include_deleted:
            where = f"{self._soft_delete_column} IS NULL"

        query = f"""
            SELECT * FROM {self._table_name}
            WHERE {where}
            ORDER BY {options.order_by}
            LIMIT ? OFFSET ?
        """

        rows = await self.db.fetch_all(query, (options.limit, options.offset))
        return [self._deserialize(row) for row in rows]

    async def create(self, entity: T) -> str:
        """Create a new entity."""
        data = self._serialize(entity)

        # Generate ID if not present
        if self._primary_key not in data or not data[self._primary_key]:
            data[self._primary_key] = str(uuid.uuid4())

        # Set timestamps
        if "created_at" not in data:
            data["created_at"] = datetime.now().isoformat()

        # Initialize version for optimistic locking
        if self._version_column and self._version_column not in data:
            data[self._version_column] = 1

        columns = ", ".join(data.keys())
        placeholders = ", ".join(["?" for _ in data])
        query = f"INSERT INTO {self._table_name} ({columns}) VALUES ({placeholders})"

        async with self.db.transaction() as conn:
            await conn.execute(query, tuple(data.values()))

        # Emit change event
        await self._emit_change(
            data[self._primary_key],
            ChangeType.INSERT,
            new_data=data,
        )

        # Invalidate cache
        if self.cache:
            await self.cache.delete(self._cache_key(data[self._primary_key]))

        return data[self._primary_key]

    async def create_many(self, entities: list[T]) -> BatchResult:
        """Create multiple entities."""
        result = BatchResult(total=len(entities), successful=0, failed=0)

        for entity in entities:
            try:
                await self.create(entity)
                result.successful += 1
            except Exception as e:
                result.failed += 1
                entity_data = self._serialize(entity)
                result.errors.append((
                    entity_data.get(self._primary_key, "unknown"),
                    str(e),
                ))

        return result

    async def update(
        self,
        id: str,
        entity: T,
        check_version: bool = False,
    ) -> bool:
        """
        Update an existing entity.

        Args:
            id: Entity ID
            entity: Updated entity
            check_version: Use optimistic locking

        Returns:
            True if updated, False if not found or version conflict
        """
        data = self._serialize(entity)
        data.pop(self._primary_key, None)
        data.pop("created_at", None)

        if not data:
            return False

        # Set update timestamp
        data["updated_at"] = datetime.now().isoformat()

        # Get old data for CDC
        old_data = None
        if self._change_handlers:
            old_entity = await self.get(id, QueryOptions(use_cache=False))
            if old_entity:
                old_data = self._serialize(old_entity)

        # Build update query
        set_clause = ", ".join([f"{k} = ?" for k in data.keys()])
        where = f"{self._primary_key} = ?"
        params = list(data.values()) + [id]

        # Optimistic locking
        if check_version and self._version_column:
            current_version = old_data.get(self._version_column, 0) if old_data else 0
            where += f" AND {self._version_column} = ?"
            params.append(current_version)
            data[self._version_column] = current_version + 1

        query = f"UPDATE {self._table_name} SET {set_clause} WHERE {where}"

        async with self.db.transaction() as conn:
            result = await conn.execute(query, tuple(params))

        # Check if update happened (for optimistic locking)
        if check_version and result == 0:
            logger.warning(
                "Optimistic lock conflict",
                entity_id=id,
                table=self._table_name,
            )
            return False

        # Emit change event
        await self._emit_change(
            id,
            ChangeType.UPDATE,
            old_data=old_data,
            new_data=data,
        )

        # Invalidate cache
        if self.cache:
            await self.cache.delete(self._cache_key(id))

        return True

    async def update_fields(
        self,
        id: str,
        fields: dict[str, Any],
    ) -> bool:
        """Update specific fields of an entity."""
        if not fields:
            return False

        fields["updated_at"] = datetime.now().isoformat()

        set_clause = ", ".join([f"{k} = ?" for k in fields.keys()])
        query = f"UPDATE {self._table_name} SET {set_clause} WHERE {self._primary_key} = ?"

        async with self.db.transaction() as conn:
            await conn.execute(query, (*fields.values(), id))

        # Invalidate cache
        if self.cache:
            await self.cache.delete(self._cache_key(id))

        return True

    async def delete(self, id: str, hard: bool = False) -> bool:
        """
        Delete an entity.

        Args:
            id: Entity ID
            hard: If True, permanently delete. If False, soft delete.

        Returns:
            True if deleted
        """
        # Get old data for CDC
        old_data = None
        if self._change_handlers:
            old_entity = await self.get(id, QueryOptions(
                use_cache=False,
                include_deleted=True,
            ))
            if old_entity:
                old_data = self._serialize(old_entity)

        if hard or not self._soft_delete_column:
            query = f"DELETE FROM {self._table_name} WHERE {self._primary_key} = ?"
        else:
            query = f"""
                UPDATE {self._table_name}
                SET {self._soft_delete_column} = ?
                WHERE {self._primary_key} = ?
            """

        async with self.db.transaction() as conn:
            if hard or not self._soft_delete_column:
                await conn.execute(query, (id,))
            else:
                await conn.execute(query, (datetime.now().isoformat(), id))

        # Emit change event
        await self._emit_change(
            id,
            ChangeType.DELETE,
            old_data=old_data,
        )

        # Invalidate cache
        if self.cache:
            await self.cache.delete(self._cache_key(id))

        return True

    async def delete_many(self, ids: list[str], hard: bool = False) -> BatchResult:
        """Delete multiple entities."""
        result = BatchResult(total=len(ids), successful=0, failed=0)

        for id in ids:
            try:
                await self.delete(id, hard=hard)
                result.successful += 1
            except Exception as e:
                result.failed += 1
                result.errors.append((id, str(e)))

        return result

    async def exists(self, id: str) -> bool:
        """Check if entity exists."""
        query = f"SELECT 1 FROM {self._table_name} WHERE {self._primary_key} = ?"

        if self._soft_delete_column:
            query += f" AND {self._soft_delete_column} IS NULL"

        result = await self.db.fetch_one(query, (id,))
        return result is not None

    async def count(
        self,
        where: Optional[str] = None,
        params: Optional[tuple] = None,
    ) -> int:
        """Count entities."""
        query = f"SELECT COUNT(*) as count FROM {self._table_name}"

        conditions = []
        if where:
            conditions.append(where)
        if self._soft_delete_column:
            conditions.append(f"{self._soft_delete_column} IS NULL")

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        result = await self.db.fetch_one(query, params)
        return result["count"] if result else 0

    # === Query Builders ===

    async def find_where(
        self,
        where: str,
        params: tuple,
        options: Optional[QueryOptions] = None,
    ) -> list[T]:
        """Find entities matching a condition."""
        options = options or QueryOptions()

        conditions = [where]
        if self._soft_delete_column and not options.include_deleted:
            conditions.append(f"{self._soft_delete_column} IS NULL")

        query = f"""
            SELECT * FROM {self._table_name}
            WHERE {" AND ".join(conditions)}
            ORDER BY {options.order_by}
            LIMIT ? OFFSET ?
        """

        rows = await self.db.fetch_all(query, (*params, options.limit, options.offset))
        return [self._deserialize(row) for row in rows]

    async def find_one_where(
        self,
        where: str,
        params: tuple,
        options: Optional[QueryOptions] = None,
    ) -> Optional[T]:
        """Find one entity matching a condition."""
        options = options or QueryOptions()
        options.limit = 1

        results = await self.find_where(where, params, options)
        return results[0] if results else None

    async def stream(
        self,
        batch_size: int = 100,
        options: Optional[QueryOptions] = None,
    ) -> AsyncGenerator[T, None]:
        """Stream entities in batches."""
        options = options or QueryOptions()
        offset = 0

        while True:
            options.offset = offset
            options.limit = batch_size

            batch = await self.get_all(options)
            if not batch:
                break

            for entity in batch:
                yield entity

            offset += batch_size

    # === Caching ===

    def _cache_key(self, id: str) -> str:
        """Generate cache key for entity."""
        return f"{self._table_name}:{id}"

    async def invalidate_cache(self, id: Optional[str] = None) -> None:
        """Invalidate cache for entity or all entities."""
        if not self.cache:
            return

        if id:
            await self.cache.delete(self._cache_key(id))
        else:
            await self.cache.invalidate_pattern(f"{self._table_name}:*")

    # === Change Data Capture ===

    def on_change(self, handler: Callable[[ChangeEvent], None]) -> None:
        """Register a change handler for CDC."""
        self._change_handlers.append(handler)

    async def _emit_change(
        self,
        entity_id: str,
        change_type: ChangeType,
        old_data: Optional[dict] = None,
        new_data: Optional[dict] = None,
    ) -> None:
        """Emit a change event."""
        if not self._change_handlers:
            return

        event = ChangeEvent(
            id=str(uuid.uuid4()),
            entity_type=self._table_name,
            entity_id=entity_id,
            change_type=change_type,
            old_data=old_data,
            new_data=new_data,
        )

        for handler in self._change_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception as e:
                logger.error(
                    "Change handler error",
                    error=str(e),
                    event_id=event.id,
                )

    # === Serialization Helpers ===

    @staticmethod
    def _to_json(obj: Any) -> str:
        """Convert object to JSON string."""
        return json.dumps(obj, default=str)

    @staticmethod
    def _from_json(s: Optional[str]) -> Any:
        """Parse JSON string."""
        if not s:
            return None
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _to_datetime(s: Optional[str]) -> Optional[datetime]:
        """Parse datetime string."""
        if not s:
            return None
        if isinstance(s, datetime):
            return s
        try:
            return datetime.fromisoformat(s)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _from_datetime(dt: Optional[datetime]) -> Optional[str]:
        """Convert datetime to ISO string."""
        if not dt:
            return None
        return dt.isoformat()


# Import asyncio for the emit_change method
import asyncio
