"""
AION Configuration Repository

Persistence for system and user configurations:
- Namespace-based configuration storage
- Session management
- User preferences
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
import uuid

import structlog

from aion.persistence.database import DatabaseManager
from aion.persistence.backends.redis_cache import CacheManager

logger = structlog.get_logger(__name__)


@dataclass
class ConfigEntry:
    """A configuration entry."""
    id: str
    namespace: str
    key: str
    value: Any
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class Session:
    """A user session."""
    id: str
    user_id: Optional[str] = None
    context: dict[str, Any] = field(default_factory=dict)
    working_memory: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_active_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ConfigRepository:
    """
    Repository for configuration persistence.

    Features:
    - Namespace-based organization
    - Type-safe value storage
    - Session management
    - Caching for frequently accessed configs
    """

    def __init__(
        self,
        db: DatabaseManager,
        cache: Optional[CacheManager] = None,
    ):
        self.db = db
        self.cache = cache

    @staticmethod
    def _to_json(obj: Any) -> str:
        """Convert object to JSON string."""
        import json
        return json.dumps(obj, default=str)

    @staticmethod
    def _from_json(s: Optional[str]) -> Any:
        """Parse JSON string."""
        if not s:
            return None
        try:
            import json
            return json.loads(s)
        except Exception:
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

    # === Configuration Operations ===

    async def get(
        self,
        namespace: str,
        key: str,
        default: Any = None,
    ) -> Any:
        """Get a configuration value."""
        # Try cache first
        cache_key = f"config:{namespace}:{key}"
        if self.cache:
            cached = await self.cache.get(cache_key)
            if cached is not None:
                return cached

        query = """
            SELECT value FROM configurations
            WHERE namespace = ? AND key = ?
        """
        row = await self.db.fetch_one(query, (namespace, key))

        if not row:
            return default

        value = self._from_json(row["value"])

        # Update cache
        if self.cache:
            await self.cache.set(cache_key, value, ttl=300)

        return value

    async def set(
        self,
        namespace: str,
        key: str,
        value: Any,
    ) -> None:
        """Set a configuration value."""
        config_id = str(uuid.uuid4())
        now = datetime.now().isoformat()

        query = """
            INSERT INTO configurations (id, namespace, key, value, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(namespace, key) DO UPDATE SET
                value = excluded.value,
                updated_at = excluded.updated_at
        """

        await self.db.execute(query, (
            config_id,
            namespace,
            key,
            self._to_json(value),
            now,
            now,
        ))

        # Invalidate cache
        if self.cache:
            await self.cache.delete(f"config:{namespace}:{key}")

    async def delete(
        self,
        namespace: str,
        key: str,
    ) -> bool:
        """Delete a configuration value."""
        query = "DELETE FROM configurations WHERE namespace = ? AND key = ?"
        await self.db.execute(query, (namespace, key))

        # Invalidate cache
        if self.cache:
            await self.cache.delete(f"config:{namespace}:{key}")

        return True

    async def get_namespace(
        self,
        namespace: str,
    ) -> dict[str, Any]:
        """Get all configurations in a namespace."""
        query = """
            SELECT key, value FROM configurations
            WHERE namespace = ?
        """
        rows = await self.db.fetch_all(query, (namespace,))

        return {
            row["key"]: self._from_json(row["value"])
            for row in rows
        }

    async def delete_namespace(
        self,
        namespace: str,
    ) -> int:
        """Delete all configurations in a namespace."""
        query = "DELETE FROM configurations WHERE namespace = ?"
        await self.db.execute(query, (namespace,))

        # Invalidate cache
        if self.cache:
            await self.cache.invalidate_pattern(f"config:{namespace}:*")

        return 0

    async def list_namespaces(self) -> list[str]:
        """List all configuration namespaces."""
        query = "SELECT DISTINCT namespace FROM configurations ORDER BY namespace"
        rows = await self.db.fetch_all(query)
        return [row["namespace"] for row in rows]

    # === Session Operations ===

    async def create_session(
        self,
        user_id: Optional[str] = None,
        expires_in_seconds: Optional[int] = None,
        metadata: Optional[dict] = None,
    ) -> Session:
        """Create a new session."""
        session_id = str(uuid.uuid4())
        now = datetime.now()

        expires_at = None
        if expires_in_seconds:
            from datetime import timedelta
            expires_at = now + timedelta(seconds=expires_in_seconds)

        session = Session(
            id=session_id,
            user_id=user_id,
            created_at=now,
            last_active_at=now,
            expires_at=expires_at,
            metadata=metadata or {},
        )

        query = """
            INSERT INTO sessions (id, user_id, context, working_memory, created_at, last_active_at, expires_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """

        await self.db.execute(query, (
            session.id,
            session.user_id,
            self._to_json(session.context),
            self._to_json(session.working_memory),
            session.created_at.isoformat(),
            session.last_active_at.isoformat(),
            session.expires_at.isoformat() if session.expires_at else None,
            self._to_json(session.metadata),
        ))

        return session

    async def get_session(
        self,
        session_id: str,
    ) -> Optional[Session]:
        """Get a session by ID."""
        query = "SELECT * FROM sessions WHERE id = ?"
        row = await self.db.fetch_one(query, (session_id,))

        if not row:
            return None

        session = Session(
            id=row["id"],
            user_id=row.get("user_id"),
            context=self._from_json(row.get("context")) or {},
            working_memory=self._from_json(row.get("working_memory")) or {},
            created_at=self._to_datetime(row.get("created_at")) or datetime.now(),
            last_active_at=self._to_datetime(row.get("last_active_at")) or datetime.now(),
            expires_at=self._to_datetime(row.get("expires_at")),
            metadata=self._from_json(row.get("metadata")) or {},
        )

        # Check if expired
        if session.expires_at and datetime.now() > session.expires_at:
            await self.delete_session(session_id)
            return None

        return session

    async def update_session(
        self,
        session_id: str,
        context: Optional[dict] = None,
        working_memory: Optional[dict] = None,
        metadata: Optional[dict] = None,
    ) -> bool:
        """Update a session."""
        updates = ["last_active_at = ?"]
        params = [datetime.now().isoformat()]

        if context is not None:
            updates.append("context = ?")
            params.append(self._to_json(context))

        if working_memory is not None:
            updates.append("working_memory = ?")
            params.append(self._to_json(working_memory))

        if metadata is not None:
            updates.append("metadata = ?")
            params.append(self._to_json(metadata))

        params.append(session_id)

        query = f"UPDATE sessions SET {', '.join(updates)} WHERE id = ?"
        await self.db.execute(query, tuple(params))

        return True

    async def delete_session(
        self,
        session_id: str,
    ) -> bool:
        """Delete a session."""
        query = "DELETE FROM sessions WHERE id = ?"
        await self.db.execute(query, (session_id,))
        return True

    async def find_user_sessions(
        self,
        user_id: str,
    ) -> list[Session]:
        """Find all sessions for a user."""
        query = """
            SELECT * FROM sessions
            WHERE user_id = ?
            ORDER BY last_active_at DESC
        """
        rows = await self.db.fetch_all(query, (user_id,))

        sessions = []
        for row in rows:
            session = Session(
                id=row["id"],
                user_id=row.get("user_id"),
                context=self._from_json(row.get("context")) or {},
                working_memory=self._from_json(row.get("working_memory")) or {},
                created_at=self._to_datetime(row.get("created_at")) or datetime.now(),
                last_active_at=self._to_datetime(row.get("last_active_at")) or datetime.now(),
                expires_at=self._to_datetime(row.get("expires_at")),
                metadata=self._from_json(row.get("metadata")) or {},
            )

            # Skip expired sessions
            if session.expires_at and datetime.now() > session.expires_at:
                continue

            sessions.append(session)

        return sessions

    async def cleanup_expired_sessions(self) -> int:
        """Delete expired sessions."""
        query = """
            DELETE FROM sessions
            WHERE expires_at IS NOT NULL AND expires_at < ?
        """
        await self.db.execute(query, (datetime.now().isoformat(),))
        return 0

    async def cleanup_inactive_sessions(
        self,
        inactive_hours: int = 24,
    ) -> int:
        """Delete inactive sessions."""
        query = """
            DELETE FROM sessions
            WHERE last_active_at < datetime('now', '-' || ? || ' hours')
        """
        await self.db.execute(query, (inactive_hours,))
        return 0

    # === Metadata Storage ===

    async def get_metadata(
        self,
        key: str,
    ) -> Optional[str]:
        """Get system metadata value."""
        query = "SELECT value FROM aion_metadata WHERE key = ?"
        row = await self.db.fetch_one(query, (key,))
        return row["value"] if row else None

    async def set_metadata(
        self,
        key: str,
        value: str,
    ) -> None:
        """Set system metadata value."""
        query = """
            INSERT INTO aion_metadata (key, value, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value = excluded.value,
                updated_at = excluded.updated_at
        """
        await self.db.execute(query, (key, value, datetime.now().isoformat()))

    async def get_all_metadata(self) -> dict[str, str]:
        """Get all system metadata."""
        query = "SELECT key, value FROM aion_metadata"
        rows = await self.db.fetch_all(query)
        return {row["key"]: row["value"] for row in rows}
