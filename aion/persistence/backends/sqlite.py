"""
AION SQLite Backend

Production-ready SQLite implementation with:
- WAL mode for concurrency
- Connection pooling
- Query optimization
- Automatic cleanup
"""

from __future__ import annotations

import asyncio
import sqlite3
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator, Optional
import time

import aiosqlite
import structlog

from aion.persistence.backends.base import BaseBackend, BackendCapabilities
from aion.persistence.config import SQLiteConfig, ConnectionPoolConfig
from aion.persistence.database import (
    DatabaseConnection,
    ConnectionPool,
    ConnectionStats,
)

logger = structlog.get_logger(__name__)


class SQLiteConnection(DatabaseConnection):
    """SQLite connection wrapper implementing DatabaseConnection interface."""

    def __init__(self, conn: aiosqlite.Connection):
        self._conn = conn
        self._in_transaction = False

    @staticmethod
    def _dict_factory(cursor: sqlite3.Cursor, row: tuple) -> dict[str, Any]:
        """Convert row to dictionary."""
        return {
            col[0]: row[idx]
            for idx, col in enumerate(cursor.description)
        }

    async def execute(
        self,
        query: str,
        params: Optional[tuple] = None,
    ) -> Any:
        """Execute a query."""
        cursor = await self._conn.execute(query, params or ())
        return cursor.lastrowid

    async def execute_many(
        self,
        query: str,
        params_list: list[tuple],
    ) -> int:
        """Execute a query with multiple parameter sets."""
        await self._conn.executemany(query, params_list)
        return len(params_list)

    async def fetch_one(
        self,
        query: str,
        params: Optional[tuple] = None,
    ) -> Optional[dict[str, Any]]:
        """Fetch one row."""
        self._conn.row_factory = self._dict_factory
        cursor = await self._conn.execute(query, params or ())
        row = await cursor.fetchone()
        return row

    async def fetch_all(
        self,
        query: str,
        params: Optional[tuple] = None,
    ) -> list[dict[str, Any]]:
        """Fetch all rows."""
        self._conn.row_factory = self._dict_factory
        cursor = await self._conn.execute(query, params or ())
        rows = await cursor.fetchall()
        return rows

    async def fetch_value(
        self,
        query: str,
        params: Optional[tuple] = None,
    ) -> Any:
        """Fetch a single value."""
        cursor = await self._conn.execute(query, params or ())
        row = await cursor.fetchone()
        return row[0] if row else None

    async def commit(self) -> None:
        """Commit the transaction."""
        await self._conn.commit()
        self._in_transaction = False

    async def rollback(self) -> None:
        """Rollback the transaction."""
        await self._conn.rollback()
        self._in_transaction = False

    async def savepoint(self, name: str) -> None:
        """Create a savepoint."""
        await self._conn.execute(f"SAVEPOINT {name}")

    async def rollback_to_savepoint(self, name: str) -> None:
        """Rollback to a savepoint."""
        await self._conn.execute(f"ROLLBACK TO SAVEPOINT {name}")

    async def release_savepoint(self, name: str) -> None:
        """Release a savepoint."""
        await self._conn.execute(f"RELEASE SAVEPOINT {name}")


class SQLiteConnectionPool(ConnectionPool):
    """
    SQLite connection pool.

    Note: SQLite with WAL mode allows concurrent reads but
    only one writer at a time. This pool manages connection
    lifecycle and provides proper locking.
    """

    def __init__(
        self,
        sqlite_config: SQLiteConfig,
        pool_config: ConnectionPoolConfig,
    ):
        super().__init__(pool_config)
        self.sqlite_config = sqlite_config
        self._connections: list[aiosqlite.Connection] = []
        self._available: asyncio.Queue[aiosqlite.Connection] = asyncio.Queue()
        self._write_lock = asyncio.Lock()
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the connection pool."""
        if self._initialized:
            return

        logger.info(
            "Initializing SQLite connection pool",
            path=str(self.sqlite_config.path),
        )

        # Ensure directory exists
        if self.sqlite_config.path != Path(":memory:"):
            self.sqlite_config.path.parent.mkdir(parents=True, exist_ok=True)

        # Create initial connections
        for _ in range(self.config.min_connections):
            conn = await self._create_connection()
            self._connections.append(conn)
            await self._available.put(conn)

        self._stats.total_connections = len(self._connections)
        self._stats.idle_connections = len(self._connections)
        self._initialized = True

        logger.info(
            "SQLite pool initialized",
            connections=len(self._connections),
        )

    async def shutdown(self) -> None:
        """Shutdown the connection pool."""
        logger.info("Shutting down SQLite connection pool")

        # Close all connections
        for conn in self._connections:
            try:
                await conn.close()
            except Exception as e:
                logger.warning("Error closing connection", error=str(e))

        self._connections.clear()
        self._available = asyncio.Queue()
        self._initialized = False

        logger.info("SQLite pool shutdown complete")

    async def _create_connection(self) -> aiosqlite.Connection:
        """Create a new SQLite connection with optimal settings."""
        path = str(self.sqlite_config.path)

        conn = await aiosqlite.connect(
            path,
            timeout=self.sqlite_config.busy_timeout / 1000,
        )

        # Apply performance settings
        pragmas = [
            f"PRAGMA journal_mode = {self.sqlite_config.journal_mode}",
            f"PRAGMA synchronous = {self.sqlite_config.synchronous}",
            f"PRAGMA cache_size = {self.sqlite_config.cache_size}",
            f"PRAGMA temp_store = {self.sqlite_config.temp_store}",
            f"PRAGMA mmap_size = {self.sqlite_config.mmap_size}",
            f"PRAGMA busy_timeout = {self.sqlite_config.busy_timeout}",
            f"PRAGMA foreign_keys = {'ON' if self.sqlite_config.foreign_keys else 'OFF'}",
            f"PRAGMA auto_vacuum = {self.sqlite_config.auto_vacuum}",
        ]

        for pragma in pragmas:
            await conn.execute(pragma)

        return conn

    async def acquire(self) -> DatabaseConnection:
        """Acquire a connection from the pool."""
        try:
            # Try to get from available pool with timeout
            conn = await asyncio.wait_for(
                self._available.get(),
                timeout=self.config.connection_timeout,
            )
            self._stats.idle_connections -= 1
            self._stats.active_connections += 1
            return SQLiteConnection(conn)

        except asyncio.TimeoutError:
            # Pool exhausted, create new if under limit
            if len(self._connections) < self.config.max_connections:
                conn = await self._create_connection()
                self._connections.append(conn)
                self._stats.total_connections += 1
                self._stats.active_connections += 1
                return SQLiteConnection(conn)

            raise RuntimeError("Connection pool exhausted")

    async def release(self, conn: DatabaseConnection) -> None:
        """Release a connection back to the pool."""
        if isinstance(conn, SQLiteConnection):
            try:
                # Rollback any uncommitted transaction
                await conn.rollback()
            except Exception:
                pass

            await self._available.put(conn._conn)
            self._stats.active_connections -= 1
            self._stats.idle_connections += 1

    async def health_check(self) -> bool:
        """Check pool health."""
        try:
            conn = await asyncio.wait_for(
                self._available.get(),
                timeout=5.0,
            )
            try:
                cursor = await conn.execute(self.config.health_check_query)
                await cursor.fetchone()
                return True
            finally:
                await self._available.put(conn)

        except Exception as e:
            logger.error("SQLite health check failed", error=str(e))
            return False

    @asynccontextmanager
    async def write_lock(self) -> AsyncGenerator[None, None]:
        """Context manager for write operations."""
        async with self._write_lock:
            yield


class SQLiteBackend(BaseBackend):
    """SQLite backend implementation."""

    def __init__(
        self,
        config: SQLiteConfig,
        pool_config: ConnectionPoolConfig,
    ):
        self._config = config
        self._pool = SQLiteConnectionPool(config, pool_config)
        self._initialized = False

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            supports_json=True,
            supports_arrays=False,  # Use JSON instead
            supports_transactions=True,
            supports_savepoints=True,
            supports_concurrent_writes=False,  # WAL helps but still single writer
            supports_full_text_search=True,  # FTS5
            supports_advisory_locks=False,
            supports_listen_notify=False,
            max_query_params=999,
            max_blob_size=1_000_000_000,
        )

    @property
    def placeholder(self) -> str:
        return "?"

    def convert_placeholders(self, query: str, count: int) -> str:
        """No conversion needed for SQLite."""
        return query

    async def initialize(self) -> None:
        """Initialize the backend."""
        await self._pool.initialize()
        self._initialized = True

    async def shutdown(self) -> None:
        """Shutdown the backend."""
        await self._pool.shutdown()
        self._initialized = False

    async def execute_ddl(self, ddl: str) -> None:
        """Execute DDL statements."""
        async with self._pool.connection() as conn:
            await conn.execute(ddl)
            await conn.commit()

    async def table_exists(self, table_name: str) -> bool:
        """Check if a table exists."""
        async with self._pool.connection() as conn:
            result = await conn.fetch_one(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table_name,),
            )
            return result is not None

    async def get_table_info(self, table_name: str) -> list[dict[str, Any]]:
        """Get table column information."""
        async with self._pool.connection() as conn:
            return await conn.fetch_all(f"PRAGMA table_info({table_name})")

    async def vacuum(self) -> None:
        """Optimize the database."""
        async with self._pool.connection() as conn:
            await conn.execute("VACUUM")
            await conn.execute("ANALYZE")
            await conn.commit()

    async def get_database_size(self) -> int:
        """Get database file size in bytes."""
        if self._config.path == Path(":memory:"):
            return 0
        return self._config.path.stat().st_size if self._config.path.exists() else 0

    async def checkpoint(self) -> None:
        """Force a WAL checkpoint."""
        async with self._pool.connection() as conn:
            await conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            await conn.commit()

    async def integrity_check(self) -> list[str]:
        """Run integrity check on the database."""
        async with self._pool.connection() as conn:
            results = await conn.fetch_all("PRAGMA integrity_check")
            return [r.get("integrity_check", str(r)) for r in results]
