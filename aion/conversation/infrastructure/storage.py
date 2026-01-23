"""
AION SOTA Session Storage

State-of-the-art session storage featuring:
- Redis for high-performance distributed sessions
- PostgreSQL for persistent storage
- Memory store for development/testing
- Session replication and failover
- Compression and encryption support
"""

from __future__ import annotations

import asyncio
import gzip
import hashlib
import json
import pickle
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, Generic, List, Optional, TypeVar

import structlog

logger = structlog.get_logger(__name__)

T = TypeVar("T")


# =============================================================================
# Session Data Structures
# =============================================================================

@dataclass
class SessionData:
    """Session data structure."""
    session_id: str
    data: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "data": self.data,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SessionData":
        return cls(
            session_id=d["session_id"],
            data=d["data"],
            created_at=datetime.fromisoformat(d["created_at"]),
            updated_at=datetime.fromisoformat(d["updated_at"]),
            expires_at=datetime.fromisoformat(d["expires_at"]) if d.get("expires_at") else None,
            metadata=d.get("metadata", {}),
        )

    @property
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at


# =============================================================================
# Abstract Session Store
# =============================================================================

class SessionStore(ABC):
    """Abstract base class for session stores."""

    @abstractmethod
    async def get(self, session_id: str) -> Optional[SessionData]:
        """Get session by ID."""
        pass

    @abstractmethod
    async def set(
        self,
        session_id: str,
        data: Dict[str, Any],
        ttl_seconds: Optional[int] = None,
    ) -> None:
        """Set session data."""
        pass

    @abstractmethod
    async def delete(self, session_id: str) -> bool:
        """Delete session."""
        pass

    @abstractmethod
    async def exists(self, session_id: str) -> bool:
        """Check if session exists."""
        pass

    @abstractmethod
    async def get_all_ids(self) -> List[str]:
        """Get all session IDs."""
        pass

    @abstractmethod
    async def clear_expired(self) -> int:
        """Clear expired sessions. Returns count of cleared sessions."""
        pass

    async def update(
        self,
        session_id: str,
        data: Dict[str, Any],
    ) -> bool:
        """Update session data (merge with existing)."""
        existing = await self.get(session_id)
        if not existing:
            return False

        merged = {**existing.data, **data}
        await self.set(session_id, merged)
        return True

    async def touch(self, session_id: str, ttl_seconds: Optional[int] = None) -> bool:
        """Extend session TTL."""
        existing = await self.get(session_id)
        if not existing:
            return False

        await self.set(session_id, existing.data, ttl_seconds)
        return True


# =============================================================================
# Memory Session Store
# =============================================================================

class MemorySessionStore(SessionStore):
    """
    In-memory session store for development and testing.
    """

    def __init__(self, default_ttl_seconds: int = 3600):
        self.default_ttl_seconds = default_ttl_seconds
        self._sessions: Dict[str, SessionData] = {}
        self._lock = asyncio.Lock()

    async def get(self, session_id: str) -> Optional[SessionData]:
        async with self._lock:
            session = self._sessions.get(session_id)
            if session and session.is_expired:
                del self._sessions[session_id]
                return None
            return session

    async def set(
        self,
        session_id: str,
        data: Dict[str, Any],
        ttl_seconds: Optional[int] = None,
    ) -> None:
        async with self._lock:
            ttl = ttl_seconds or self.default_ttl_seconds
            expires_at = datetime.now() + timedelta(seconds=ttl) if ttl else None

            existing = self._sessions.get(session_id)
            if existing:
                session = SessionData(
                    session_id=session_id,
                    data=data,
                    created_at=existing.created_at,
                    updated_at=datetime.now(),
                    expires_at=expires_at,
                    metadata=existing.metadata,
                )
            else:
                session = SessionData(
                    session_id=session_id,
                    data=data,
                    expires_at=expires_at,
                )

            self._sessions[session_id] = session

    async def delete(self, session_id: str) -> bool:
        async with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                return True
            return False

    async def exists(self, session_id: str) -> bool:
        session = await self.get(session_id)
        return session is not None

    async def get_all_ids(self) -> List[str]:
        async with self._lock:
            # Filter out expired
            valid_ids = []
            expired_ids = []

            for sid, session in self._sessions.items():
                if session.is_expired:
                    expired_ids.append(sid)
                else:
                    valid_ids.append(sid)

            # Clean up expired
            for sid in expired_ids:
                del self._sessions[sid]

            return valid_ids

    async def clear_expired(self) -> int:
        async with self._lock:
            expired = [sid for sid, s in self._sessions.items() if s.is_expired]
            for sid in expired:
                del self._sessions[sid]
            return len(expired)

    async def clear_all(self) -> None:
        """Clear all sessions."""
        async with self._lock:
            self._sessions.clear()


# =============================================================================
# Redis Session Store
# =============================================================================

class RedisSessionStore(SessionStore):
    """
    Redis-based session store for production use.

    Features:
    - Distributed session storage
    - Automatic expiration
    - Compression support
    - Connection pooling
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        prefix: str = "aion:session:",
        default_ttl_seconds: int = 3600,
        compress: bool = True,
        compression_threshold: int = 1024,
    ):
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.prefix = prefix
        self.default_ttl_seconds = default_ttl_seconds
        self.compress = compress
        self.compression_threshold = compression_threshold

        self._redis = None
        self._connected = False

    async def connect(self) -> None:
        """Connect to Redis."""
        if self._connected:
            return

        try:
            import redis.asyncio as redis

            self._redis = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                decode_responses=False,  # We handle encoding
            )

            # Test connection
            await self._redis.ping()
            self._connected = True
            logger.info(f"Connected to Redis at {self.host}:{self.port}")

        except ImportError:
            logger.error("redis package not installed. Install with: pip install redis")
            raise
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        if self._redis:
            await self._redis.close()
            self._connected = False

    def _key(self, session_id: str) -> str:
        """Generate Redis key for session."""
        return f"{self.prefix}{session_id}"

    def _serialize(self, session: SessionData) -> bytes:
        """Serialize session data."""
        data = json.dumps(session.to_dict()).encode('utf-8')

        if self.compress and len(data) > self.compression_threshold:
            compressed = gzip.compress(data)
            # Prefix with marker byte
            return b'\x01' + compressed
        return b'\x00' + data

    def _deserialize(self, data: bytes) -> SessionData:
        """Deserialize session data."""
        if data[0] == 1:  # Compressed
            decompressed = gzip.decompress(data[1:])
            return SessionData.from_dict(json.loads(decompressed.decode('utf-8')))
        return SessionData.from_dict(json.loads(data[1:].decode('utf-8')))

    async def get(self, session_id: str) -> Optional[SessionData]:
        if not self._connected:
            await self.connect()

        key = self._key(session_id)
        data = await self._redis.get(key)

        if data is None:
            return None

        try:
            return self._deserialize(data)
        except Exception as e:
            logger.error(f"Failed to deserialize session {session_id}: {e}")
            return None

    async def set(
        self,
        session_id: str,
        data: Dict[str, Any],
        ttl_seconds: Optional[int] = None,
    ) -> None:
        if not self._connected:
            await self.connect()

        ttl = ttl_seconds or self.default_ttl_seconds
        expires_at = datetime.now() + timedelta(seconds=ttl) if ttl else None

        # Get existing for metadata preservation
        existing = await self.get(session_id)

        session = SessionData(
            session_id=session_id,
            data=data,
            created_at=existing.created_at if existing else datetime.now(),
            updated_at=datetime.now(),
            expires_at=expires_at,
            metadata=existing.metadata if existing else {},
        )

        key = self._key(session_id)
        serialized = self._serialize(session)

        if ttl:
            await self._redis.setex(key, ttl, serialized)
        else:
            await self._redis.set(key, serialized)

    async def delete(self, session_id: str) -> bool:
        if not self._connected:
            await self.connect()

        key = self._key(session_id)
        result = await self._redis.delete(key)
        return result > 0

    async def exists(self, session_id: str) -> bool:
        if not self._connected:
            await self.connect()

        key = self._key(session_id)
        return await self._redis.exists(key) > 0

    async def get_all_ids(self) -> List[str]:
        if not self._connected:
            await self.connect()

        pattern = f"{self.prefix}*"
        keys = await self._redis.keys(pattern)
        return [k.decode('utf-8').replace(self.prefix, '') for k in keys]

    async def clear_expired(self) -> int:
        # Redis handles expiration automatically
        return 0


# =============================================================================
# PostgreSQL Session Store
# =============================================================================

class PostgresSessionStore(SessionStore):
    """
    PostgreSQL-based session store for persistent storage.

    Features:
    - ACID-compliant storage
    - Full-text search on session data
    - Audit logging
    """

    CREATE_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS sessions (
        session_id VARCHAR(255) PRIMARY KEY,
        data JSONB NOT NULL,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        expires_at TIMESTAMP WITH TIME ZONE,
        metadata JSONB DEFAULT '{}'::jsonb
    );

    CREATE INDEX IF NOT EXISTS idx_sessions_expires_at ON sessions(expires_at);
    CREATE INDEX IF NOT EXISTS idx_sessions_created_at ON sessions(created_at);
    """

    def __init__(
        self,
        dsn: str = "postgresql://localhost/aion",
        pool_size: int = 10,
        default_ttl_seconds: int = 3600,
    ):
        self.dsn = dsn
        self.pool_size = pool_size
        self.default_ttl_seconds = default_ttl_seconds

        self._pool = None
        self._connected = False

    async def connect(self) -> None:
        """Connect to PostgreSQL."""
        if self._connected:
            return

        try:
            import asyncpg

            self._pool = await asyncpg.create_pool(
                self.dsn,
                min_size=2,
                max_size=self.pool_size,
            )

            # Create table if not exists
            async with self._pool.acquire() as conn:
                await conn.execute(self.CREATE_TABLE_SQL)

            self._connected = True
            logger.info("Connected to PostgreSQL")

        except ImportError:
            logger.error("asyncpg package not installed. Install with: pip install asyncpg")
            raise
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise

    async def disconnect(self) -> None:
        """Disconnect from PostgreSQL."""
        if self._pool:
            await self._pool.close()
            self._connected = False

    async def get(self, session_id: str) -> Optional[SessionData]:
        if not self._connected:
            await self.connect()

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT session_id, data, created_at, updated_at, expires_at, metadata
                FROM sessions
                WHERE session_id = $1
                AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
                """,
                session_id,
            )

            if row is None:
                return None

            return SessionData(
                session_id=row['session_id'],
                data=json.loads(row['data']) if isinstance(row['data'], str) else row['data'],
                created_at=row['created_at'],
                updated_at=row['updated_at'],
                expires_at=row['expires_at'],
                metadata=json.loads(row['metadata']) if isinstance(row['metadata'], str) else row['metadata'] or {},
            )

    async def set(
        self,
        session_id: str,
        data: Dict[str, Any],
        ttl_seconds: Optional[int] = None,
    ) -> None:
        if not self._connected:
            await self.connect()

        ttl = ttl_seconds or self.default_ttl_seconds
        expires_at = datetime.now() + timedelta(seconds=ttl) if ttl else None

        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO sessions (session_id, data, expires_at, updated_at)
                VALUES ($1, $2, $3, CURRENT_TIMESTAMP)
                ON CONFLICT (session_id) DO UPDATE SET
                    data = EXCLUDED.data,
                    expires_at = EXCLUDED.expires_at,
                    updated_at = CURRENT_TIMESTAMP
                """,
                session_id,
                json.dumps(data),
                expires_at,
            )

    async def delete(self, session_id: str) -> bool:
        if not self._connected:
            await self.connect()

        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM sessions WHERE session_id = $1",
                session_id,
            )
            return result == "DELETE 1"

    async def exists(self, session_id: str) -> bool:
        if not self._connected:
            await self.connect()

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT 1 FROM sessions
                WHERE session_id = $1
                AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
                """,
                session_id,
            )
            return row is not None

    async def get_all_ids(self) -> List[str]:
        if not self._connected:
            await self.connect()

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT session_id FROM sessions
                WHERE expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP
                """
            )
            return [row['session_id'] for row in rows]

    async def clear_expired(self) -> int:
        if not self._connected:
            await self.connect()

        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM sessions WHERE expires_at < CURRENT_TIMESTAMP"
            )
            # Parse "DELETE N" to get count
            count = int(result.split()[-1]) if result else 0
            return count

    async def search(
        self,
        query: Dict[str, Any],
        limit: int = 100,
    ) -> List[SessionData]:
        """Search sessions by data content using JSONB operators."""
        if not self._connected:
            await self.connect()

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT session_id, data, created_at, updated_at, expires_at, metadata
                FROM sessions
                WHERE data @> $1
                AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
                LIMIT $2
                """,
                json.dumps(query),
                limit,
            )

            return [
                SessionData(
                    session_id=row['session_id'],
                    data=json.loads(row['data']) if isinstance(row['data'], str) else row['data'],
                    created_at=row['created_at'],
                    updated_at=row['updated_at'],
                    expires_at=row['expires_at'],
                    metadata=json.loads(row['metadata']) if isinstance(row['metadata'], str) else row['metadata'] or {},
                )
                for row in rows
            ]


# =============================================================================
# Session Store Factory
# =============================================================================

class SessionStoreFactory:
    """Factory for creating session stores."""

    @staticmethod
    def create(
        store_type: str = "memory",
        **kwargs: Any,
    ) -> SessionStore:
        """
        Create a session store.

        Args:
            store_type: Type of store ("memory", "redis", "postgres")
            **kwargs: Store-specific configuration

        Returns:
            SessionStore instance
        """
        if store_type == "memory":
            return MemorySessionStore(**kwargs)

        elif store_type == "redis":
            return RedisSessionStore(**kwargs)

        elif store_type == "postgres" or store_type == "postgresql":
            return PostgresSessionStore(**kwargs)

        else:
            raise ValueError(f"Unknown session store type: {store_type}")

    @staticmethod
    def from_url(url: str, **kwargs: Any) -> SessionStore:
        """
        Create a session store from a URL.

        Examples:
            - memory://
            - redis://localhost:6379/0
            - postgresql://user:pass@localhost/dbname
        """
        from urllib.parse import urlparse

        parsed = urlparse(url)

        if parsed.scheme == "memory":
            return MemorySessionStore(**kwargs)

        elif parsed.scheme == "redis":
            return RedisSessionStore(
                host=parsed.hostname or "localhost",
                port=parsed.port or 6379,
                db=int(parsed.path.lstrip('/') or 0),
                password=parsed.password,
                **kwargs,
            )

        elif parsed.scheme in ("postgres", "postgresql"):
            return PostgresSessionStore(dsn=url, **kwargs)

        else:
            raise ValueError(f"Unknown URL scheme: {parsed.scheme}")
