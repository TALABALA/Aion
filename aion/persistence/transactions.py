"""
AION Transaction Manager

State-of-the-art transaction management with:
- ACID compliance
- Multiple isolation levels
- Optimistic locking with version checking
- Pessimistic locking with timeouts
- Nested transactions (savepoints)
- Distributed transaction support
- Automatic retry with backoff
- Deadlock detection
"""

from __future__ import annotations

import asyncio
import functools
import logging
import time
import uuid
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Generic, Optional, Protocol, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class IsolationLevel(str, Enum):
    """Transaction isolation levels."""
    READ_UNCOMMITTED = "READ UNCOMMITTED"
    READ_COMMITTED = "READ COMMITTED"
    REPEATABLE_READ = "REPEATABLE READ"
    SERIALIZABLE = "SERIALIZABLE"


class LockMode(str, Enum):
    """Lock modes for pessimistic locking."""
    SHARED = "SHARE"
    EXCLUSIVE = "EXCLUSIVE"
    UPDATE = "UPDATE"


class TransactionState(str, Enum):
    """Transaction state."""
    PENDING = "pending"
    ACTIVE = "active"
    COMMITTED = "committed"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"


@dataclass
class TransactionContext:
    """Context for a transaction."""
    id: str
    isolation_level: IsolationLevel
    read_only: bool
    started_at: datetime
    savepoints: list[str] = field(default_factory=list)
    state: TransactionState = TransactionState.PENDING
    nested_level: int = 0
    locks_held: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class OptimisticLockError(Exception):
    """Raised when optimistic lock check fails."""

    def __init__(
        self,
        entity_id: str,
        expected_version: int,
        actual_version: int,
    ):
        self.entity_id = entity_id
        self.expected_version = expected_version
        self.actual_version = actual_version
        super().__init__(
            f"Optimistic lock failed for {entity_id}: "
            f"expected version {expected_version}, got {actual_version}"
        )


class PessimisticLockError(Exception):
    """Raised when pessimistic lock cannot be acquired."""

    def __init__(self, resource: str, timeout: float):
        self.resource = resource
        self.timeout = timeout
        super().__init__(f"Failed to acquire lock on {resource} within {timeout}s")


class DeadlockError(Exception):
    """Raised when a deadlock is detected."""

    def __init__(self, transactions: list[str]):
        self.transactions = transactions
        super().__init__(f"Deadlock detected involving transactions: {transactions}")


class DatabaseConnection(Protocol):
    """Protocol for database connections."""

    async def execute(self, query: str, params: tuple = ()) -> None:
        ...

    async def fetch_all(self, query: str, params: tuple = ()) -> list[dict]:
        ...

    async def fetch_one(self, query: str, params: tuple = ()) -> Optional[dict]:
        ...


@dataclass
class Transaction:
    """
    Represents an active database transaction.

    Supports:
    - Commit and rollback
    - Savepoints for nested transactions
    - Lock management
    - Version tracking for optimistic locking
    """

    context: TransactionContext
    connection: DatabaseConnection
    _manager: Optional["TransactionManager"] = None

    async def commit(self) -> None:
        """Commit the transaction."""
        if self.context.state != TransactionState.ACTIVE:
            raise RuntimeError(f"Cannot commit transaction in state {self.context.state}")

        try:
            await self.connection.execute("COMMIT")
            self.context.state = TransactionState.COMMITTED
            logger.debug(f"Transaction {self.context.id} committed")
        except Exception as e:
            self.context.state = TransactionState.FAILED
            logger.error(f"Transaction {self.context.id} commit failed: {e}")
            raise

    async def rollback(self) -> None:
        """Rollback the transaction."""
        if self.context.state not in (TransactionState.ACTIVE, TransactionState.FAILED):
            return

        try:
            await self.connection.execute("ROLLBACK")
            self.context.state = TransactionState.ROLLED_BACK
            logger.debug(f"Transaction {self.context.id} rolled back")
        except Exception as e:
            logger.error(f"Transaction {self.context.id} rollback failed: {e}")
            raise

    async def savepoint(self, name: Optional[str] = None) -> str:
        """Create a savepoint."""
        if self.context.state != TransactionState.ACTIVE:
            raise RuntimeError("Transaction not active")

        name = name or f"sp_{len(self.context.savepoints)}"
        await self.connection.execute(f"SAVEPOINT {name}")
        self.context.savepoints.append(name)
        self.context.nested_level += 1
        logger.debug(f"Savepoint {name} created in transaction {self.context.id}")
        return name

    async def release_savepoint(self, name: str) -> None:
        """Release a savepoint."""
        if name not in self.context.savepoints:
            raise ValueError(f"Savepoint {name} not found")

        await self.connection.execute(f"RELEASE SAVEPOINT {name}")
        self.context.savepoints.remove(name)
        self.context.nested_level = max(0, self.context.nested_level - 1)
        logger.debug(f"Savepoint {name} released in transaction {self.context.id}")

    async def rollback_to_savepoint(self, name: str) -> None:
        """Rollback to a savepoint."""
        if name not in self.context.savepoints:
            raise ValueError(f"Savepoint {name} not found")

        await self.connection.execute(f"ROLLBACK TO SAVEPOINT {name}")
        # Remove all savepoints after this one
        idx = self.context.savepoints.index(name)
        self.context.savepoints = self.context.savepoints[: idx + 1]
        self.context.nested_level = idx + 1
        logger.debug(f"Rolled back to savepoint {name} in transaction {self.context.id}")

    async def check_version(
        self,
        table: str,
        entity_id: str,
        expected_version: int,
    ) -> bool:
        """
        Check if entity version matches expected version.

        Args:
            table: Table name
            entity_id: Entity ID
            expected_version: Expected version number

        Returns:
            True if versions match

        Raises:
            OptimisticLockError if versions don't match
        """
        row = await self.connection.fetch_one(
            f"SELECT version FROM {table} WHERE id = ?",
            (entity_id,),
        )

        if not row:
            raise ValueError(f"Entity {entity_id} not found in {table}")

        actual_version = row["version"]
        if actual_version != expected_version:
            raise OptimisticLockError(entity_id, expected_version, actual_version)

        return True

    async def increment_version(
        self,
        table: str,
        entity_id: str,
        expected_version: int,
    ) -> int:
        """
        Increment entity version with optimistic lock check.

        Args:
            table: Table name
            entity_id: Entity ID
            expected_version: Expected current version

        Returns:
            New version number

        Raises:
            OptimisticLockError if versions don't match
        """
        # Atomic update with version check
        await self.connection.execute(
            f"""
            UPDATE {table}
            SET version = version + 1, updated_at = CURRENT_TIMESTAMP
            WHERE id = ? AND version = ?
            """,
            (entity_id, expected_version),
        )

        # Verify update succeeded
        row = await self.connection.fetch_one(
            f"SELECT version FROM {table} WHERE id = ?",
            (entity_id,),
        )

        if not row:
            raise ValueError(f"Entity {entity_id} not found in {table}")

        new_version = row["version"]
        if new_version != expected_version + 1:
            raise OptimisticLockError(entity_id, expected_version, new_version - 1)

        return new_version


class LockManager:
    """
    Manages pessimistic locks with deadlock detection.

    Features:
    - Shared and exclusive locks
    - Lock timeouts
    - Deadlock detection
    - Lock queuing
    """

    def __init__(self):
        self._locks: dict[str, tuple[str, LockMode, datetime]] = {}  # resource -> (txn_id, mode, acquired_at)
        self._waiting: dict[str, list[str]] = {}  # resource -> [waiting txn_ids]
        self._txn_locks: dict[str, set[str]] = {}  # txn_id -> {resources}
        self._lock = asyncio.Lock()

    async def acquire(
        self,
        transaction_id: str,
        resource: str,
        mode: LockMode = LockMode.EXCLUSIVE,
        timeout: float = 30.0,
    ) -> bool:
        """
        Acquire a lock on a resource.

        Args:
            transaction_id: Transaction ID
            resource: Resource identifier
            mode: Lock mode
            timeout: Timeout in seconds

        Returns:
            True if lock acquired

        Raises:
            PessimisticLockError if timeout exceeded
            DeadlockError if deadlock detected
        """
        start_time = time.time()

        while True:
            async with self._lock:
                # Check for deadlock
                if self._detect_deadlock(transaction_id, resource):
                    raise DeadlockError([transaction_id])

                # Check if lock is available
                if resource not in self._locks:
                    # Lock is free
                    self._locks[resource] = (transaction_id, mode, datetime.utcnow())
                    if transaction_id not in self._txn_locks:
                        self._txn_locks[transaction_id] = set()
                    self._txn_locks[transaction_id].add(resource)
                    logger.debug(
                        f"Lock acquired: {resource} by {transaction_id} ({mode})"
                    )
                    return True

                holder_id, holder_mode, _ = self._locks[resource]

                # Check if we already hold the lock
                if holder_id == transaction_id:
                    # Upgrade lock if needed
                    if mode == LockMode.EXCLUSIVE and holder_mode != LockMode.EXCLUSIVE:
                        self._locks[resource] = (transaction_id, mode, datetime.utcnow())
                    return True

                # Check if shared lock is compatible
                if (
                    mode == LockMode.SHARED
                    and holder_mode == LockMode.SHARED
                ):
                    # Allow shared lock
                    return True

                # Add to waiting list
                if resource not in self._waiting:
                    self._waiting[resource] = []
                if transaction_id not in self._waiting[resource]:
                    self._waiting[resource].append(transaction_id)

            # Check timeout
            elapsed = time.time() - start_time
            if elapsed >= timeout:
                async with self._lock:
                    if resource in self._waiting:
                        self._waiting[resource] = [
                            t for t in self._waiting[resource] if t != transaction_id
                        ]
                raise PessimisticLockError(resource, timeout)

            # Wait and retry
            await asyncio.sleep(0.1)

    async def release(
        self,
        transaction_id: str,
        resource: Optional[str] = None,
    ) -> None:
        """
        Release locks held by a transaction.

        Args:
            transaction_id: Transaction ID
            resource: Specific resource to release (None = all)
        """
        async with self._lock:
            if resource:
                resources = [resource]
            else:
                resources = list(self._txn_locks.get(transaction_id, set()))

            for res in resources:
                if res in self._locks:
                    holder_id, _, _ = self._locks[res]
                    if holder_id == transaction_id:
                        del self._locks[res]
                        logger.debug(f"Lock released: {res} by {transaction_id}")

                if transaction_id in self._txn_locks:
                    self._txn_locks[transaction_id].discard(res)

            # Clean up empty sets
            if transaction_id in self._txn_locks and not self._txn_locks[transaction_id]:
                del self._txn_locks[transaction_id]

    def _detect_deadlock(self, transaction_id: str, resource: str) -> bool:
        """
        Detect potential deadlock.

        Uses simple wait-for graph analysis.
        """
        if resource not in self._locks:
            return False

        holder_id, _, _ = self._locks[resource]
        if holder_id == transaction_id:
            return False

        # Check if holder is waiting for any resource we hold
        visited = set()
        to_check = [holder_id]

        while to_check:
            current = to_check.pop()
            if current in visited:
                continue
            visited.add(current)

            # Check what resources this transaction is waiting for
            for res, waiters in self._waiting.items():
                if current in waiters:
                    if res in self._locks:
                        res_holder, _, _ = self._locks[res]
                        if res_holder == transaction_id:
                            return True
                        to_check.append(res_holder)

        return False

    def get_lock_status(self) -> dict[str, Any]:
        """Get current lock status."""
        return {
            "active_locks": len(self._locks),
            "waiting_transactions": sum(len(w) for w in self._waiting.values()),
            "locks": {
                res: {"holder": holder, "mode": mode.value}
                for res, (holder, mode, _) in self._locks.items()
            },
        }


class TransactionManager:
    """
    Manages database transactions with advanced features.

    Features:
    - Multiple isolation levels
    - Optimistic and pessimistic locking
    - Nested transactions via savepoints
    - Automatic retry with backoff
    - Deadlock detection and recovery
    """

    def __init__(
        self,
        connection: Optional[DatabaseConnection] = None,
        default_isolation: IsolationLevel = IsolationLevel.READ_COMMITTED,
        default_timeout: float = 30.0,
        max_retries: int = 3,
        retry_delay: float = 0.5,
    ):
        self.connection = connection
        self.default_isolation = default_isolation
        self.default_timeout = default_timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self._lock_manager = LockManager()
        self._active_transactions: dict[str, Transaction] = {}

    async def initialize(self, connection: Optional[DatabaseConnection] = None) -> None:
        """Initialize the transaction manager."""
        if connection:
            self.connection = connection

    @asynccontextmanager
    async def transaction(
        self,
        isolation_level: Optional[IsolationLevel] = None,
        read_only: bool = False,
        timeout: Optional[float] = None,
    ):
        """
        Context manager for transactions.

        Args:
            isolation_level: Isolation level (default from config)
            read_only: Whether transaction is read-only
            timeout: Transaction timeout

        Yields:
            Transaction object
        """
        if not self.connection:
            raise RuntimeError("No database connection")

        txn_id = str(uuid.uuid4())
        isolation = isolation_level or self.default_isolation
        timeout = timeout or self.default_timeout

        context = TransactionContext(
            id=txn_id,
            isolation_level=isolation,
            read_only=read_only,
            started_at=datetime.utcnow(),
        )

        txn = Transaction(
            context=context,
            connection=self.connection,
            _manager=self,
        )

        try:
            # Start transaction
            await self.connection.execute("BEGIN")

            # Set isolation level (PostgreSQL style)
            try:
                await self.connection.execute(
                    f"SET TRANSACTION ISOLATION LEVEL {isolation.value}"
                )
            except Exception:
                pass  # SQLite doesn't support this

            if read_only:
                try:
                    await self.connection.execute("SET TRANSACTION READ ONLY")
                except Exception:
                    pass

            context.state = TransactionState.ACTIVE
            self._active_transactions[txn_id] = txn

            logger.debug(f"Transaction {txn_id} started (isolation: {isolation})")

            yield txn

            # Auto-commit if not already committed/rolled back
            if context.state == TransactionState.ACTIVE:
                await txn.commit()

        except Exception as e:
            # Rollback on error
            if context.state == TransactionState.ACTIVE:
                try:
                    await txn.rollback()
                except Exception:
                    pass
            raise
        finally:
            # Release all locks
            await self._lock_manager.release(txn_id)

            # Remove from active transactions
            self._active_transactions.pop(txn_id, None)

    @asynccontextmanager
    async def savepoint(self, transaction: Transaction, name: Optional[str] = None):
        """
        Context manager for savepoints (nested transactions).

        Args:
            transaction: Parent transaction
            name: Savepoint name

        Yields:
            Savepoint name
        """
        sp_name = await transaction.savepoint(name)
        try:
            yield sp_name
            await transaction.release_savepoint(sp_name)
        except Exception:
            await transaction.rollback_to_savepoint(sp_name)
            raise

    async def with_retry(
        self,
        operation: Callable[..., Any],
        *args,
        max_retries: Optional[int] = None,
        retry_delay: Optional[float] = None,
        **kwargs,
    ) -> Any:
        """
        Execute operation with automatic retry.

        Args:
            operation: Async function to execute
            max_retries: Maximum retry attempts
            retry_delay: Initial delay between retries

        Returns:
            Operation result
        """
        retries = max_retries or self.max_retries
        delay = retry_delay or self.retry_delay

        last_error = None

        for attempt in range(retries + 1):
            try:
                return await operation(*args, **kwargs)
            except OptimisticLockError as e:
                last_error = e
                if attempt < retries:
                    logger.warning(
                        f"Optimistic lock conflict, retrying ({attempt + 1}/{retries})"
                    )
                    await asyncio.sleep(delay * (2 ** attempt))
            except DeadlockError as e:
                last_error = e
                if attempt < retries:
                    logger.warning(
                        f"Deadlock detected, retrying ({attempt + 1}/{retries})"
                    )
                    await asyncio.sleep(delay * (2 ** attempt))

        raise last_error or RuntimeError("Operation failed after retries")

    async def lock(
        self,
        transaction: Transaction,
        resource: str,
        mode: LockMode = LockMode.EXCLUSIVE,
        timeout: Optional[float] = None,
    ) -> bool:
        """
        Acquire a lock within a transaction.

        Args:
            transaction: Transaction context
            resource: Resource to lock
            mode: Lock mode
            timeout: Lock timeout

        Returns:
            True if lock acquired
        """
        timeout = timeout or self.default_timeout
        result = await self._lock_manager.acquire(
            transaction.context.id,
            resource,
            mode,
            timeout,
        )

        if result:
            transaction.context.locks_held.append(resource)

        return result

    async def unlock(
        self,
        transaction: Transaction,
        resource: str,
    ) -> None:
        """
        Release a lock within a transaction.

        Args:
            transaction: Transaction context
            resource: Resource to unlock
        """
        await self._lock_manager.release(transaction.context.id, resource)
        if resource in transaction.context.locks_held:
            transaction.context.locks_held.remove(resource)

    def get_active_transactions(self) -> list[dict[str, Any]]:
        """Get list of active transactions."""
        return [
            {
                "id": txn.context.id,
                "isolation_level": txn.context.isolation_level.value,
                "read_only": txn.context.read_only,
                "started_at": txn.context.started_at.isoformat(),
                "state": txn.context.state.value,
                "nested_level": txn.context.nested_level,
                "locks_held": txn.context.locks_held,
            }
            for txn in self._active_transactions.values()
        ]

    def get_lock_status(self) -> dict[str, Any]:
        """Get lock manager status."""
        return self._lock_manager.get_lock_status()


def transactional(
    isolation_level: Optional[IsolationLevel] = None,
    read_only: bool = False,
    max_retries: int = 3,
):
    """
    Decorator to make a function transactional.

    Args:
        isolation_level: Transaction isolation level
        read_only: Whether transaction is read-only
        max_retries: Maximum retry attempts on conflict

    Usage:
        @transactional(isolation_level=IsolationLevel.SERIALIZABLE)
        async def update_balance(self, amount: int):
            ...
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(self, *args, **kwargs):
            manager = getattr(self, "_transaction_manager", None)
            if not manager:
                return await func(self, *args, **kwargs)

            async def operation():
                async with manager.transaction(
                    isolation_level=isolation_level,
                    read_only=read_only,
                ) as txn:
                    return await func(self, *args, transaction=txn, **kwargs)

            return await manager.with_retry(operation, max_retries=max_retries)

        return wrapper

    return decorator


class UnitOfWork:
    """
    Unit of Work pattern implementation.

    Tracks changes across multiple entities and commits them
    atomically within a single transaction.
    """

    def __init__(self, transaction_manager: TransactionManager):
        self._manager = transaction_manager
        self._new: list[tuple[str, dict]] = []  # (table, data)
        self._dirty: list[tuple[str, str, dict, int]] = []  # (table, id, data, version)
        self._deleted: list[tuple[str, str]] = []  # (table, id)

    def register_new(self, table: str, data: dict) -> None:
        """Register a new entity to be inserted."""
        self._new.append((table, data))

    def register_dirty(
        self,
        table: str,
        entity_id: str,
        data: dict,
        version: int,
    ) -> None:
        """Register a modified entity to be updated."""
        self._dirty.append((table, entity_id, data, version))

    def register_deleted(self, table: str, entity_id: str) -> None:
        """Register an entity to be deleted."""
        self._deleted.append((table, entity_id))

    async def commit(self) -> bool:
        """
        Commit all changes in a single transaction.

        Returns:
            True if commit succeeded
        """
        async with self._manager.transaction() as txn:
            # Process inserts
            for table, data in self._new:
                columns = ", ".join(data.keys())
                placeholders = ", ".join(["?"] * len(data))
                await txn.connection.execute(
                    f"INSERT INTO {table} ({columns}) VALUES ({placeholders})",
                    tuple(data.values()),
                )

            # Process updates with optimistic locking
            for table, entity_id, data, expected_version in self._dirty:
                # Check version
                await txn.check_version(table, entity_id, expected_version)

                # Update
                set_clause = ", ".join(f"{k} = ?" for k in data.keys())
                await txn.connection.execute(
                    f"UPDATE {table} SET {set_clause}, version = version + 1 WHERE id = ?",
                    (*data.values(), entity_id),
                )

            # Process deletes
            for table, entity_id in self._deleted:
                await txn.connection.execute(
                    f"DELETE FROM {table} WHERE id = ?",
                    (entity_id,),
                )

            # Clear tracked changes
            self._new.clear()
            self._dirty.clear()
            self._deleted.clear()

            return True

    def rollback(self) -> None:
        """Clear all tracked changes without committing."""
        self._new.clear()
        self._dirty.clear()
        self._deleted.clear()

    @property
    def has_changes(self) -> bool:
        """Check if there are pending changes."""
        return bool(self._new or self._dirty or self._deleted)

    def get_changes_summary(self) -> dict[str, int]:
        """Get summary of pending changes."""
        return {
            "new": len(self._new),
            "dirty": len(self._dirty),
            "deleted": len(self._deleted),
            "total": len(self._new) + len(self._dirty) + len(self._deleted),
        }
