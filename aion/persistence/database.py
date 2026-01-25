"""
AION Database Connection Management

State-of-the-art database abstraction with:
- Async connection pooling with health monitoring
- Circuit breaker pattern for resilience
- Query execution with automatic retry
- Transaction support with savepoints
- Connection lifecycle management
- Query logging and metrics
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, AsyncGenerator, Callable, Optional, TypeVar
import uuid

import structlog

from aion.persistence.config import (
    PersistenceConfig,
    DatabaseBackend,
    CircuitBreakerConfig,
    ConnectionPoolConfig,
)

logger = structlog.get_logger(__name__)

T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if recovered


@dataclass
class QueryStats:
    """Statistics for a single query execution."""
    query_hash: str
    duration_ms: float
    rows_affected: int
    success: bool
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ConnectionStats:
    """Statistics for database connections."""
    total_connections: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    failed_connections: int = 0
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    total_query_time_ms: float = 0.0
    slow_queries: int = 0
    last_health_check: Optional[datetime] = None
    healthy: bool = True


class CircuitBreaker:
    """
    Circuit breaker for database resilience.

    Prevents cascade failures by temporarily stopping
    requests when the database is unhealthy.
    """

    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._half_open_calls = 0
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        return self._state

    async def can_execute(self) -> bool:
        """Check if a request can be executed."""
        if not self.config.enabled:
            return True

        async with self._lock:
            if self._state == CircuitState.CLOSED:
                return True

            if self._state == CircuitState.OPEN:
                # Check if timeout has passed
                if self._last_failure_time:
                    elapsed = time.time() - self._last_failure_time
                    if elapsed >= self.config.timeout:
                        self._state = CircuitState.HALF_OPEN
                        self._half_open_calls = 0
                        logger.info("Circuit breaker entering half-open state")
                        return True
                return False

            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls < self.config.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False

        return True

    async def record_success(self) -> None:
        """Record a successful operation."""
        if not self.config.enabled:
            return

        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._success_count = 0
                    logger.info("Circuit breaker closed after recovery")

            elif self._state == CircuitState.CLOSED:
                # Reset failure count on success
                self._failure_count = 0

    async def record_failure(self) -> None:
        """Record a failed operation."""
        if not self.config.enabled:
            return

        async with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                # Any failure in half-open goes back to open
                self._state = CircuitState.OPEN
                self._success_count = 0
                logger.warning("Circuit breaker reopened after failure in half-open")

            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self.config.failure_threshold:
                    self._state = CircuitState.OPEN
                    logger.warning(
                        "Circuit breaker opened",
                        failure_count=self._failure_count,
                    )


class DatabaseConnection(ABC):
    """Abstract database connection interface."""

    @abstractmethod
    async def execute(
        self,
        query: str,
        params: Optional[tuple] = None,
    ) -> Any:
        """Execute a query without returning results."""
        pass

    @abstractmethod
    async def execute_many(
        self,
        query: str,
        params_list: list[tuple],
    ) -> int:
        """Execute a query with multiple parameter sets."""
        pass

    @abstractmethod
    async def fetch_one(
        self,
        query: str,
        params: Optional[tuple] = None,
    ) -> Optional[dict[str, Any]]:
        """Fetch one row as a dictionary."""
        pass

    @abstractmethod
    async def fetch_all(
        self,
        query: str,
        params: Optional[tuple] = None,
    ) -> list[dict[str, Any]]:
        """Fetch all rows as dictionaries."""
        pass

    @abstractmethod
    async def fetch_value(
        self,
        query: str,
        params: Optional[tuple] = None,
    ) -> Any:
        """Fetch a single value."""
        pass

    @abstractmethod
    async def commit(self) -> None:
        """Commit the current transaction."""
        pass

    @abstractmethod
    async def rollback(self) -> None:
        """Rollback the current transaction."""
        pass

    @abstractmethod
    async def savepoint(self, name: str) -> None:
        """Create a savepoint."""
        pass

    @abstractmethod
    async def rollback_to_savepoint(self, name: str) -> None:
        """Rollback to a savepoint."""
        pass

    @abstractmethod
    async def release_savepoint(self, name: str) -> None:
        """Release a savepoint."""
        pass

    async def __aenter__(self) -> "DatabaseConnection":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type:
            await self.rollback()
        else:
            await self.commit()


class ConnectionPool(ABC):
    """Abstract connection pool interface."""

    def __init__(self, config: ConnectionPoolConfig):
        self.config = config
        self._stats = ConnectionStats()
        self._health_check_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

    @property
    def stats(self) -> ConnectionStats:
        return self._stats

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the connection pool."""
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the connection pool."""
        pass

    @abstractmethod
    async def acquire(self) -> DatabaseConnection:
        """Acquire a connection from the pool."""
        pass

    @abstractmethod
    async def release(self, conn: DatabaseConnection) -> None:
        """Release a connection back to the pool."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Perform a health check on the pool."""
        pass

    async def start_health_checks(self) -> None:
        """Start periodic health checks."""
        if self._health_check_task is not None:
            return

        self._health_check_task = asyncio.create_task(self._health_check_loop())

    async def stop_health_checks(self) -> None:
        """Stop health checks."""
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
            self._health_check_task = None

    async def _health_check_loop(self) -> None:
        """Background health check loop."""
        while True:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                healthy = await self.health_check()
                self._stats.healthy = healthy
                self._stats.last_health_check = datetime.now()

                if not healthy:
                    logger.warning("Database health check failed")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Health check error", error=str(e))
                self._stats.healthy = False

    @asynccontextmanager
    async def connection(self) -> AsyncGenerator[DatabaseConnection, None]:
        """Context manager for connection acquisition."""
        conn = await self.acquire()
        try:
            yield conn
        finally:
            await self.release(conn)


class DatabaseManager:
    """
    Central database manager for AION.

    Handles:
    - Connection pooling with multiple backends
    - Query execution with retries and circuit breaker
    - Transaction management
    - Query logging and metrics
    - Health monitoring
    """

    def __init__(self, config: PersistenceConfig):
        self.config = config
        self._pool: Optional[ConnectionPool] = None
        self._circuit_breaker = CircuitBreaker(config.circuit_breaker)
        self._stats = ConnectionStats()
        self._query_stats: list[QueryStats] = []
        self._initialized = False
        self._lock = asyncio.Lock()

    @property
    def backend(self) -> DatabaseBackend:
        return self.config.backend

    @property
    def stats(self) -> ConnectionStats:
        return self._stats

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    async def initialize(self) -> None:
        """Initialize the database manager."""
        if self._initialized:
            return

        async with self._lock:
            if self._initialized:
                return

            logger.info(
                "Initializing database manager",
                backend=self.config.backend.value,
            )

            # Create backend-specific pool
            if self.config.backend == DatabaseBackend.SQLITE:
                from aion.persistence.backends.sqlite import SQLiteConnectionPool
                self._pool = SQLiteConnectionPool(
                    self.config.sqlite,
                    self.config.pool,
                )

            elif self.config.backend == DatabaseBackend.POSTGRESQL:
                from aion.persistence.backends.postgres import PostgreSQLConnectionPool
                self._pool = PostgreSQLConnectionPool(
                    self.config.postgresql,
                    self.config.pool,
                )

            elif self.config.backend == DatabaseBackend.MEMORY:
                from aion.persistence.backends.sqlite import SQLiteConnectionPool
                from aion.persistence.config import SQLiteConfig
                # Use in-memory SQLite for testing
                memory_config = SQLiteConfig(path=":memory:")
                self._pool = SQLiteConnectionPool(
                    memory_config,
                    self.config.pool,
                )

            await self._pool.initialize()
            await self._pool.start_health_checks()

            self._initialized = True
            logger.info("Database manager initialized")

    async def shutdown(self) -> None:
        """Shutdown the database manager."""
        if not self._initialized:
            return

        async with self._lock:
            logger.info("Shutting down database manager")

            if self._pool:
                await self._pool.stop_health_checks()
                await self._pool.shutdown()
                self._pool = None

            self._initialized = False
            logger.info("Database manager shutdown complete")

    async def _check_circuit_breaker(self) -> None:
        """Check circuit breaker before executing."""
        if not await self._circuit_breaker.can_execute():
            raise ConnectionError("Circuit breaker is open - database unavailable")

    async def _execute_with_retry(
        self,
        operation: Callable[..., T],
        *args,
        **kwargs,
    ) -> T:
        """Execute an operation with retry logic."""
        last_error: Optional[Exception] = None

        for attempt in range(self.config.pool.retry_attempts):
            await self._check_circuit_breaker()

            try:
                result = await operation(*args, **kwargs)
                await self._circuit_breaker.record_success()
                return result

            except Exception as e:
                last_error = e
                await self._circuit_breaker.record_failure()
                self._stats.failed_queries += 1

                if attempt < self.config.pool.retry_attempts - 1:
                    delay = (
                        self.config.pool.retry_delay *
                        (self.config.pool.retry_backoff_multiplier ** attempt)
                    )
                    logger.warning(
                        "Query failed, retrying",
                        attempt=attempt + 1,
                        delay=delay,
                        error=str(e),
                    )
                    await asyncio.sleep(delay)

        raise last_error or RuntimeError("Query failed after all retries")

    def _log_query(
        self,
        query: str,
        duration_ms: float,
        success: bool,
        error: Optional[str] = None,
        rows: int = 0,
    ) -> None:
        """Log query execution."""
        import hashlib
        query_hash = hashlib.md5(query.encode()).hexdigest()[:8]

        # Update stats
        self._stats.total_queries += 1
        self._stats.total_query_time_ms += duration_ms

        if success:
            self._stats.successful_queries += 1
        else:
            self._stats.failed_queries += 1

        if duration_ms > self.config.slow_query_threshold_ms:
            self._stats.slow_queries += 1

        # Log if enabled
        if self.config.log_queries or (
            self.config.log_slow_queries and
            duration_ms > self.config.slow_query_threshold_ms
        ):
            logger.debug(
                "Query executed",
                query_hash=query_hash,
                duration_ms=round(duration_ms, 2),
                rows=rows,
                success=success,
                error=error,
            )

        # Store query stats (limited)
        stats = QueryStats(
            query_hash=query_hash,
            duration_ms=duration_ms,
            rows_affected=rows,
            success=success,
            error=error,
        )
        self._query_stats.append(stats)

        # Limit stored stats
        if len(self._query_stats) > 10000:
            self._query_stats = self._query_stats[-5000:]

    @asynccontextmanager
    async def connection(self) -> AsyncGenerator[DatabaseConnection, None]:
        """Get a database connection."""
        if not self._initialized:
            await self.initialize()

        await self._check_circuit_breaker()

        async with self._pool.connection() as conn:
            yield conn

    @asynccontextmanager
    async def transaction(self) -> AsyncGenerator[DatabaseConnection, None]:
        """Get a connection with transaction management."""
        async with self.connection() as conn:
            try:
                yield conn
                await conn.commit()
            except Exception:
                await conn.rollback()
                raise

    async def execute(
        self,
        query: str,
        params: Optional[tuple] = None,
    ) -> Any:
        """Execute a query."""
        start_time = time.time()
        error_msg: Optional[str] = None

        try:
            async def _execute():
                async with self.connection() as conn:
                    return await conn.execute(query, params)

            result = await self._execute_with_retry(_execute)
            return result

        except Exception as e:
            error_msg = str(e)
            raise

        finally:
            duration_ms = (time.time() - start_time) * 1000
            self._log_query(query, duration_ms, error_msg is None, error_msg)

    async def execute_many(
        self,
        query: str,
        params_list: list[tuple],
    ) -> int:
        """Execute a query with multiple parameter sets."""
        start_time = time.time()
        error_msg: Optional[str] = None
        rows = 0

        try:
            async def _execute():
                async with self.connection() as conn:
                    return await conn.execute_many(query, params_list)

            rows = await self._execute_with_retry(_execute)
            return rows

        except Exception as e:
            error_msg = str(e)
            raise

        finally:
            duration_ms = (time.time() - start_time) * 1000
            self._log_query(query, duration_ms, error_msg is None, error_msg, rows)

    async def fetch_one(
        self,
        query: str,
        params: Optional[tuple] = None,
    ) -> Optional[dict[str, Any]]:
        """Fetch one row."""
        start_time = time.time()
        error_msg: Optional[str] = None

        try:
            async def _fetch():
                async with self.connection() as conn:
                    return await conn.fetch_one(query, params)

            return await self._execute_with_retry(_fetch)

        except Exception as e:
            error_msg = str(e)
            raise

        finally:
            duration_ms = (time.time() - start_time) * 1000
            self._log_query(query, duration_ms, error_msg is None, error_msg, 1)

    async def fetch_all(
        self,
        query: str,
        params: Optional[tuple] = None,
    ) -> list[dict[str, Any]]:
        """Fetch all rows."""
        start_time = time.time()
        error_msg: Optional[str] = None
        rows = 0

        try:
            async def _fetch():
                async with self.connection() as conn:
                    return await conn.fetch_all(query, params)

            result = await self._execute_with_retry(_fetch)
            rows = len(result)
            return result

        except Exception as e:
            error_msg = str(e)
            raise

        finally:
            duration_ms = (time.time() - start_time) * 1000
            self._log_query(query, duration_ms, error_msg is None, error_msg, rows)

    async def fetch_value(
        self,
        query: str,
        params: Optional[tuple] = None,
    ) -> Any:
        """Fetch a single value."""
        start_time = time.time()
        error_msg: Optional[str] = None

        try:
            async def _fetch():
                async with self.connection() as conn:
                    return await conn.fetch_value(query, params)

            return await self._execute_with_retry(_fetch)

        except Exception as e:
            error_msg = str(e)
            raise

        finally:
            duration_ms = (time.time() - start_time) * 1000
            self._log_query(query, duration_ms, error_msg is None, error_msg, 1)

    async def health_check(self) -> dict[str, Any]:
        """Check database health."""
        try:
            pool_healthy = await self._pool.health_check() if self._pool else False
            circuit_state = self._circuit_breaker.state.value

            return {
                "healthy": pool_healthy and circuit_state != CircuitState.OPEN.value,
                "backend": self.config.backend.value,
                "circuit_breaker": circuit_state,
                "pool_stats": self._pool.stats if self._pool else None,
                "query_stats": {
                    "total": self._stats.total_queries,
                    "successful": self._stats.successful_queries,
                    "failed": self._stats.failed_queries,
                    "slow": self._stats.slow_queries,
                    "avg_time_ms": (
                        self._stats.total_query_time_ms / self._stats.total_queries
                        if self._stats.total_queries > 0 else 0
                    ),
                },
            }

        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "backend": self.config.backend.value,
            }

    def get_recent_queries(self, limit: int = 100) -> list[QueryStats]:
        """Get recent query statistics."""
        return self._query_stats[-limit:]

    def get_slow_queries(self, threshold_ms: Optional[float] = None) -> list[QueryStats]:
        """Get slow queries."""
        threshold = threshold_ms or self.config.slow_query_threshold_ms
        return [q for q in self._query_stats if q.duration_ms > threshold]
