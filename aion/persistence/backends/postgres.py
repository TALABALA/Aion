"""
AION PostgreSQL Backend

Production-ready PostgreSQL implementation with:
- Connection pooling via asyncpg
- Advisory locks
- LISTEN/NOTIFY support
- Prepared statements
- Statement caching
"""

from __future__ import annotations

import asyncio
import re
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, AsyncGenerator, Optional
import json

import structlog

from aion.persistence.backends.base import BaseBackend, BackendCapabilities
from aion.persistence.config import PostgreSQLConfig, ConnectionPoolConfig
from aion.persistence.database import (
    DatabaseConnection,
    ConnectionPool,
    ConnectionStats,
)

logger = structlog.get_logger(__name__)

# Optional asyncpg import
try:
    import asyncpg
    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False
    asyncpg = None


class PostgreSQLConnection(DatabaseConnection):
    """PostgreSQL connection wrapper implementing DatabaseConnection interface."""

    def __init__(self, conn: "asyncpg.Connection", transaction: Optional["asyncpg.Transaction"] = None):
        self._conn = conn
        self._transaction = transaction

    @staticmethod
    def _convert_placeholders(query: str) -> str:
        """Convert ? placeholders to $1, $2, etc."""
        counter = [0]

        def replacer(match):
            counter[0] += 1
            return f"${counter[0]}"

        return re.sub(r'\?', replacer, query)

    async def execute(
        self,
        query: str,
        params: Optional[tuple] = None,
    ) -> Any:
        """Execute a query."""
        query = self._convert_placeholders(query)
        result = await self._conn.execute(query, *(params or ()))
        # Extract row count from "INSERT 0 1" style strings
        if result and ' ' in result:
            parts = result.split()
            if parts[-1].isdigit():
                return int(parts[-1])
        return 0

    async def execute_many(
        self,
        query: str,
        params_list: list[tuple],
    ) -> int:
        """Execute a query with multiple parameter sets."""
        query = self._convert_placeholders(query)
        await self._conn.executemany(query, params_list)
        return len(params_list)

    async def fetch_one(
        self,
        query: str,
        params: Optional[tuple] = None,
    ) -> Optional[dict[str, Any]]:
        """Fetch one row."""
        query = self._convert_placeholders(query)
        row = await self._conn.fetchrow(query, *(params or ()))
        return dict(row) if row else None

    async def fetch_all(
        self,
        query: str,
        params: Optional[tuple] = None,
    ) -> list[dict[str, Any]]:
        """Fetch all rows."""
        query = self._convert_placeholders(query)
        rows = await self._conn.fetch(query, *(params or ()))
        return [dict(row) for row in rows]

    async def fetch_value(
        self,
        query: str,
        params: Optional[tuple] = None,
    ) -> Any:
        """Fetch a single value."""
        query = self._convert_placeholders(query)
        return await self._conn.fetchval(query, *(params or ()))

    async def commit(self) -> None:
        """Commit the transaction."""
        if self._transaction:
            await self._transaction.commit()
            self._transaction = None

    async def rollback(self) -> None:
        """Rollback the transaction."""
        if self._transaction:
            await self._transaction.rollback()
            self._transaction = None

    async def savepoint(self, name: str) -> None:
        """Create a savepoint."""
        await self._conn.execute(f"SAVEPOINT {name}")

    async def rollback_to_savepoint(self, name: str) -> None:
        """Rollback to a savepoint."""
        await self._conn.execute(f"ROLLBACK TO SAVEPOINT {name}")

    async def release_savepoint(self, name: str) -> None:
        """Release a savepoint."""
        await self._conn.execute(f"RELEASE SAVEPOINT {name}")


class PostgreSQLConnectionPool(ConnectionPool):
    """
    PostgreSQL connection pool using asyncpg.

    Features:
    - High-performance connection pooling
    - Statement caching
    - JSON/JSONB support
    - Array type support
    """

    def __init__(
        self,
        pg_config: PostgreSQLConfig,
        pool_config: ConnectionPoolConfig,
    ):
        super().__init__(pool_config)
        self.pg_config = pg_config
        self._pool: Optional["asyncpg.Pool"] = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the connection pool."""
        if self._initialized:
            return

        if not ASYNCPG_AVAILABLE:
            raise ImportError(
                "asyncpg is required for PostgreSQL backend. "
                "Install with: pip install asyncpg"
            )

        logger.info(
            "Initializing PostgreSQL connection pool",
            host=self.pg_config.host,
            database=self.pg_config.database,
        )

        # SSL configuration
        ssl = None
        if self.pg_config.ssl_mode == "require":
            import ssl as ssl_module
            ssl = ssl_module.create_default_context()
            if self.pg_config.ssl_cert:
                ssl.load_cert_chain(
                    str(self.pg_config.ssl_cert),
                    str(self.pg_config.ssl_key) if self.pg_config.ssl_key else None,
                )
            if self.pg_config.ssl_root_cert:
                ssl.load_verify_locations(str(self.pg_config.ssl_root_cert))

        # Custom type codecs
        async def init_connection(conn):
            # Enable JSON decoding
            await conn.set_type_codec(
                'json',
                encoder=json.dumps,
                decoder=json.loads,
                schema='pg_catalog',
            )
            await conn.set_type_codec(
                'jsonb',
                encoder=json.dumps,
                decoder=json.loads,
                schema='pg_catalog',
            )

        self._pool = await asyncpg.create_pool(
            host=self.pg_config.host,
            port=self.pg_config.port,
            database=self.pg_config.database,
            user=self.pg_config.user,
            password=self.pg_config.password,
            min_size=self.config.min_connections,
            max_size=self.config.max_connections,
            command_timeout=self.config.connection_timeout,
            max_inactive_connection_lifetime=self.config.idle_timeout,
            ssl=ssl,
            init=init_connection,
            statement_cache_size=self.pg_config.statement_cache_size,
        )

        self._stats.total_connections = self._pool.get_size()
        self._initialized = True

        logger.info(
            "PostgreSQL pool initialized",
            size=self._pool.get_size(),
        )

    async def shutdown(self) -> None:
        """Shutdown the connection pool."""
        logger.info("Shutting down PostgreSQL connection pool")

        if self._pool:
            await self._pool.close()
            self._pool = None

        self._initialized = False
        logger.info("PostgreSQL pool shutdown complete")

    async def acquire(self) -> DatabaseConnection:
        """Acquire a connection from the pool."""
        if not self._pool:
            raise RuntimeError("Pool not initialized")

        conn = await self._pool.acquire(timeout=self.config.connection_timeout)
        transaction = conn.transaction()
        await transaction.start()

        self._stats.active_connections = self._pool.get_size() - self._pool.get_idle_size()
        self._stats.idle_connections = self._pool.get_idle_size()

        return PostgreSQLConnection(conn, transaction)

    async def release(self, conn: DatabaseConnection) -> None:
        """Release a connection back to the pool."""
        if isinstance(conn, PostgreSQLConnection) and self._pool:
            try:
                # Ensure transaction is ended
                if conn._transaction:
                    try:
                        await conn._transaction.rollback()
                    except Exception:
                        pass
            finally:
                await self._pool.release(conn._conn)

            self._stats.active_connections = self._pool.get_size() - self._pool.get_idle_size()
            self._stats.idle_connections = self._pool.get_idle_size()

    async def health_check(self) -> bool:
        """Check pool health."""
        if not self._pool:
            return False

        try:
            async with self._pool.acquire(timeout=5.0) as conn:
                await conn.fetchval(self.config.health_check_query)
                return True
        except Exception as e:
            logger.error("PostgreSQL health check failed", error=str(e))
            return False


class PostgreSQLBackend(BaseBackend):
    """PostgreSQL backend implementation."""

    def __init__(
        self,
        config: PostgreSQLConfig,
        pool_config: ConnectionPoolConfig,
    ):
        self._config = config
        self._pool = PostgreSQLConnectionPool(config, pool_config)
        self._initialized = False

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            supports_json=True,
            supports_arrays=True,
            supports_transactions=True,
            supports_savepoints=True,
            supports_concurrent_writes=True,
            supports_full_text_search=True,
            supports_advisory_locks=True,
            supports_listen_notify=True,
            max_query_params=65535,
            max_blob_size=1_073_741_823,  # ~1GB
        )

    @property
    def placeholder(self) -> str:
        return "$"

    def convert_placeholders(self, query: str, count: int) -> str:
        """Convert ? placeholders to $1, $2, etc."""
        result = query
        for i in range(count, 0, -1):
            result = result.replace("?", f"${i}", 1)
        return result

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
        query = """
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_schema = $1
                AND table_name = $2
            )
        """
        async with self._pool.connection() as conn:
            result = await conn.fetch_value(query, (self._config.schema, table_name))
            return bool(result)

    async def get_table_info(self, table_name: str) -> list[dict[str, Any]]:
        """Get table column information."""
        query = """
            SELECT column_name, data_type, is_nullable, column_default
            FROM information_schema.columns
            WHERE table_schema = $1
            AND table_name = $2
            ORDER BY ordinal_position
        """
        async with self._pool.connection() as conn:
            return await conn.fetch_all(query, (self._config.schema, table_name))

    async def vacuum(self) -> None:
        """Optimize the database."""
        # VACUUM cannot run inside a transaction
        if not self._pool._pool:
            return

        async with self._pool._pool.acquire() as conn:
            await conn.execute("VACUUM ANALYZE")

    async def get_database_size(self) -> int:
        """Get database size in bytes."""
        query = "SELECT pg_database_size(current_database())"
        async with self._pool.connection() as conn:
            return await conn.fetch_value(query, None) or 0

    async def acquire_advisory_lock(
        self,
        lock_id: int,
        exclusive: bool = True,
        timeout_ms: int = 10000,
    ) -> bool:
        """Acquire an advisory lock."""
        if exclusive:
            query = "SELECT pg_try_advisory_lock($1)"
        else:
            query = "SELECT pg_try_advisory_lock_shared($1)"

        async with self._pool.connection() as conn:
            result = await conn.fetch_value(query, (lock_id,))
            return bool(result)

    async def release_advisory_lock(
        self,
        lock_id: int,
        exclusive: bool = True,
    ) -> bool:
        """Release an advisory lock."""
        if exclusive:
            query = "SELECT pg_advisory_unlock($1)"
        else:
            query = "SELECT pg_advisory_unlock_shared($1)"

        async with self._pool.connection() as conn:
            result = await conn.fetch_value(query, (lock_id,))
            return bool(result)

    async def listen(self, channel: str) -> AsyncGenerator[dict[str, Any], None]:
        """Listen to a PostgreSQL notification channel."""
        if not self._pool._pool:
            return

        async with self._pool._pool.acquire() as conn:
            await conn.add_listener(channel, lambda *args: None)
            try:
                while True:
                    notification = await conn.wait_for_notify()
                    yield {
                        "channel": notification.channel,
                        "payload": notification.payload,
                        "pid": notification.pid,
                    }
            finally:
                await conn.remove_listener(channel, lambda *args: None)

    async def notify(self, channel: str, payload: str) -> None:
        """Send a notification to a channel."""
        async with self._pool.connection() as conn:
            await conn.execute(f"NOTIFY {channel}, $1", (payload,))
            await conn.commit()
