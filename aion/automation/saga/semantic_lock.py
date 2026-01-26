"""
AION Semantic Lock Manager

Manages semantic locks for saga transactions.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set

import structlog

logger = structlog.get_logger(__name__)


class LockType(str, Enum):
    """Type of semantic lock."""
    EXCLUSIVE = "exclusive"  # No other locks allowed
    SHARED = "shared"  # Multiple readers, no writers
    INTENT_EXCLUSIVE = "intent_exclusive"  # Intent to acquire exclusive
    COUNTERMEASURE = "countermeasure"  # Block other sagas


class LockMode(str, Enum):
    """Lock acquisition mode."""
    BLOCKING = "blocking"  # Wait for lock
    NON_BLOCKING = "non_blocking"  # Fail immediately if can't acquire
    TIMEOUT = "timeout"  # Wait with timeout


@dataclass
class Lock:
    """A semantic lock on a resource."""
    id: str
    resource_id: str
    lock_type: LockType
    holder_id: str  # Saga/execution ID
    holder_name: Optional[str] = None

    # Timing
    acquired_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    ttl_seconds: int = 300  # 5 minutes default

    # State
    released: bool = False
    released_at: Optional[datetime] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.expires_at is None:
            self.expires_at = self.acquired_at + timedelta(seconds=self.ttl_seconds)

    def is_expired(self) -> bool:
        """Check if lock has expired."""
        if self.released:
            return True
        if self.expires_at:
            return datetime.now() > self.expires_at
        return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "resource_id": self.resource_id,
            "lock_type": self.lock_type.value,
            "holder_id": self.holder_id,
            "holder_name": self.holder_name,
            "acquired_at": self.acquired_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "ttl_seconds": self.ttl_seconds,
            "released": self.released,
            "released_at": self.released_at.isoformat() if self.released_at else None,
        }


class SemanticLockManager:
    """
    Manages semantic locks for saga transactions.

    Features:
    - Exclusive and shared locks
    - Deadlock detection
    - Lock timeout and renewal
    - Lock queuing
    - Countermeasure locks
    """

    def __init__(
        self,
        default_ttl: int = 300,
        deadlock_check_interval: float = 5.0,
        cleanup_interval: float = 60.0,
    ):
        self.default_ttl = default_ttl
        self.deadlock_check_interval = deadlock_check_interval
        self.cleanup_interval = cleanup_interval

        # Active locks by resource
        self._locks: Dict[str, List[Lock]] = {}

        # Lock wait queues
        self._queues: Dict[str, asyncio.Queue] = {}

        # Holder to locks mapping (for deadlock detection)
        self._holder_locks: Dict[str, Set[str]] = {}
        self._holder_waiting: Dict[str, str] = {}  # holder -> resource waiting for

        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._deadlock_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the lock manager."""
        if self._initialized:
            return

        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        self._deadlock_task = asyncio.create_task(self._deadlock_detection_loop())

        self._initialized = True
        logger.info("Semantic lock manager initialized")

    async def shutdown(self) -> None:
        """Shutdown the lock manager."""
        self._shutdown_event.set()

        if self._cleanup_task:
            self._cleanup_task.cancel()
        if self._deadlock_task:
            self._deadlock_task.cancel()

        # Release all locks
        for resource_id in list(self._locks.keys()):
            for lock in list(self._locks.get(resource_id, [])):
                await self.release(lock.id)

        self._initialized = False
        logger.info("Semantic lock manager shutdown")

    async def acquire(
        self,
        resource_id: str,
        holder_id: str,
        lock_type: LockType = LockType.EXCLUSIVE,
        mode: LockMode = LockMode.BLOCKING,
        timeout_seconds: Optional[float] = None,
        ttl_seconds: Optional[int] = None,
        holder_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Lock]:
        """
        Acquire a lock on a resource.

        Args:
            resource_id: Resource to lock
            holder_id: Saga/execution ID acquiring the lock
            lock_type: Type of lock
            mode: Blocking, non-blocking, or timeout
            timeout_seconds: Timeout for TIMEOUT mode
            ttl_seconds: Lock TTL
            holder_name: Optional name for the holder
            metadata: Additional lock metadata

        Returns:
            Lock if acquired, None if failed
        """
        ttl = ttl_seconds or self.default_ttl

        # Check if we can acquire immediately
        if self._can_acquire(resource_id, holder_id, lock_type):
            return await self._do_acquire(
                resource_id, holder_id, lock_type, ttl, holder_name, metadata
            )

        # Handle based on mode
        if mode == LockMode.NON_BLOCKING:
            logger.debug(f"Cannot acquire lock on {resource_id} (non-blocking)")
            return None

        elif mode == LockMode.TIMEOUT:
            return await self._acquire_with_timeout(
                resource_id, holder_id, lock_type, timeout_seconds or 30,
                ttl, holder_name, metadata
            )

        else:  # BLOCKING
            return await self._acquire_blocking(
                resource_id, holder_id, lock_type, ttl, holder_name, metadata
            )

    def _can_acquire(
        self,
        resource_id: str,
        holder_id: str,
        lock_type: LockType,
    ) -> bool:
        """Check if a lock can be acquired."""
        existing = self._locks.get(resource_id, [])
        active = [l for l in existing if not l.is_expired()]

        if not active:
            return True

        # Check compatibility
        for lock in active:
            # Same holder can upgrade/add compatible locks
            if lock.holder_id == holder_id:
                continue

            # Check lock type compatibility
            if lock_type == LockType.EXCLUSIVE:
                return False  # Exclusive requires no other locks

            if lock_type == LockType.SHARED:
                if lock.lock_type in [LockType.EXCLUSIVE, LockType.INTENT_EXCLUSIVE]:
                    return False

            if lock_type == LockType.INTENT_EXCLUSIVE:
                if lock.lock_type == LockType.EXCLUSIVE:
                    return False

            if lock_type == LockType.COUNTERMEASURE:
                return False  # Countermeasure blocks everything

            if lock.lock_type == LockType.COUNTERMEASURE:
                return False  # Blocked by countermeasure

        return True

    async def _do_acquire(
        self,
        resource_id: str,
        holder_id: str,
        lock_type: LockType,
        ttl: int,
        holder_name: Optional[str],
        metadata: Optional[Dict],
    ) -> Lock:
        """Actually acquire the lock."""
        lock = Lock(
            id=str(uuid.uuid4()),
            resource_id=resource_id,
            lock_type=lock_type,
            holder_id=holder_id,
            holder_name=holder_name,
            ttl_seconds=ttl,
            metadata=metadata or {},
        )

        if resource_id not in self._locks:
            self._locks[resource_id] = []
        self._locks[resource_id].append(lock)

        if holder_id not in self._holder_locks:
            self._holder_locks[holder_id] = set()
        self._holder_locks[holder_id].add(resource_id)

        # Remove from waiting
        self._holder_waiting.pop(holder_id, None)

        logger.debug(
            "Lock acquired",
            resource_id=resource_id,
            holder_id=holder_id,
            lock_type=lock_type.value,
        )

        return lock

    async def _acquire_blocking(
        self,
        resource_id: str,
        holder_id: str,
        lock_type: LockType,
        ttl: int,
        holder_name: Optional[str],
        metadata: Optional[Dict],
    ) -> Lock:
        """Acquire with blocking wait."""
        # Track waiting for deadlock detection
        self._holder_waiting[holder_id] = resource_id

        # Create queue if needed
        if resource_id not in self._queues:
            self._queues[resource_id] = asyncio.Queue()

        # Wait for notification
        while True:
            if self._can_acquire(resource_id, holder_id, lock_type):
                return await self._do_acquire(
                    resource_id, holder_id, lock_type, ttl, holder_name, metadata
                )

            # Wait for release notification
            try:
                await asyncio.wait_for(
                    self._queues[resource_id].get(),
                    timeout=1.0,
                )
            except asyncio.TimeoutError:
                pass

    async def _acquire_with_timeout(
        self,
        resource_id: str,
        holder_id: str,
        lock_type: LockType,
        timeout: float,
        ttl: int,
        holder_name: Optional[str],
        metadata: Optional[Dict],
    ) -> Optional[Lock]:
        """Acquire with timeout."""
        start_time = time.time()

        self._holder_waiting[holder_id] = resource_id

        while time.time() - start_time < timeout:
            if self._can_acquire(resource_id, holder_id, lock_type):
                return await self._do_acquire(
                    resource_id, holder_id, lock_type, ttl, holder_name, metadata
                )

            await asyncio.sleep(0.1)

        self._holder_waiting.pop(holder_id, None)
        logger.debug(f"Lock acquisition timed out for {resource_id}")
        return None

    async def release(self, lock_id: str) -> bool:
        """
        Release a lock.

        Args:
            lock_id: ID of the lock to release

        Returns:
            True if released, False if not found
        """
        for resource_id, locks in self._locks.items():
            for lock in locks:
                if lock.id == lock_id and not lock.released:
                    lock.released = True
                    lock.released_at = datetime.now()

                    # Notify waiters
                    if resource_id in self._queues:
                        try:
                            self._queues[resource_id].put_nowait(True)
                        except asyncio.QueueFull:
                            pass

                    # Update holder tracking
                    if lock.holder_id in self._holder_locks:
                        self._holder_locks[lock.holder_id].discard(resource_id)

                    logger.debug(
                        "Lock released",
                        lock_id=lock_id,
                        resource_id=resource_id,
                    )

                    return True

        return False

    async def release_all(self, holder_id: str) -> int:
        """
        Release all locks held by a holder.

        Args:
            holder_id: The holder whose locks to release

        Returns:
            Number of locks released
        """
        released = 0

        for resource_id, locks in list(self._locks.items()):
            for lock in locks:
                if lock.holder_id == holder_id and not lock.released:
                    await self.release(lock.id)
                    released += 1

        return released

    async def renew(self, lock_id: str, ttl_seconds: Optional[int] = None) -> bool:
        """
        Renew a lock's TTL.

        Args:
            lock_id: ID of the lock to renew
            ttl_seconds: New TTL (or use default)

        Returns:
            True if renewed, False if not found
        """
        ttl = ttl_seconds or self.default_ttl

        for locks in self._locks.values():
            for lock in locks:
                if lock.id == lock_id and not lock.released:
                    lock.expires_at = datetime.now() + timedelta(seconds=ttl)
                    lock.ttl_seconds = ttl
                    logger.debug(f"Lock renewed: {lock_id}")
                    return True

        return False

    def get_locks(self, resource_id: str) -> List[Lock]:
        """Get all active locks on a resource."""
        return [l for l in self._locks.get(resource_id, []) if not l.is_expired()]

    def get_holder_locks(self, holder_id: str) -> List[Lock]:
        """Get all locks held by a holder."""
        result = []
        for locks in self._locks.values():
            for lock in locks:
                if lock.holder_id == holder_id and not lock.is_expired():
                    result.append(lock)
        return result

    async def _cleanup_loop(self) -> None:
        """Background task to clean up expired locks."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(self.cleanup_interval)

                expired_count = 0
                for resource_id in list(self._locks.keys()):
                    locks = self._locks.get(resource_id, [])
                    active = [l for l in locks if not l.is_expired()]
                    expired_count += len(locks) - len(active)
                    self._locks[resource_id] = active

                    if not active:
                        del self._locks[resource_id]
                        if resource_id in self._queues:
                            del self._queues[resource_id]

                if expired_count > 0:
                    logger.debug(f"Cleaned up {expired_count} expired locks")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Lock cleanup error: {e}")

    async def _deadlock_detection_loop(self) -> None:
        """Background task to detect deadlocks."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(self.deadlock_check_interval)

                # Build wait-for graph
                # holder -> [holders it's waiting for]
                wait_for: Dict[str, Set[str]] = {}

                for holder_id, resource_id in self._holder_waiting.items():
                    wait_for[holder_id] = set()
                    for lock in self._locks.get(resource_id, []):
                        if lock.holder_id != holder_id and not lock.is_expired():
                            wait_for[holder_id].add(lock.holder_id)

                # Detect cycles using DFS
                visited = set()
                path = set()

                def has_cycle(node: str) -> Optional[List[str]]:
                    visited.add(node)
                    path.add(node)

                    for neighbor in wait_for.get(node, set()):
                        if neighbor in path:
                            return [neighbor, node]
                        if neighbor not in visited:
                            cycle = has_cycle(neighbor)
                            if cycle:
                                return cycle

                    path.remove(node)
                    return None

                for holder_id in wait_for:
                    if holder_id not in visited:
                        cycle = has_cycle(holder_id)
                        if cycle:
                            logger.warning(
                                "Deadlock detected!",
                                cycle=cycle,
                            )
                            # Could implement automatic resolution here

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Deadlock detection error: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get lock manager statistics."""
        total_locks = sum(
            len([l for l in locks if not l.is_expired()])
            for locks in self._locks.values()
        )

        by_type = {}
        for locks in self._locks.values():
            for lock in locks:
                if not lock.is_expired():
                    t = lock.lock_type.value
                    by_type[t] = by_type.get(t, 0) + 1

        return {
            "total_locks": total_locks,
            "locked_resources": len(self._locks),
            "waiting_holders": len(self._holder_waiting),
            "by_type": by_type,
        }


class LockContext:
    """
    Context manager for automatic lock acquisition and release.

    Usage:
        async with LockContext(manager, "order:123", holder_id) as lock:
            # Do work with resource
            pass
        # Lock automatically released
    """

    def __init__(
        self,
        manager: SemanticLockManager,
        resource_id: str,
        holder_id: str,
        lock_type: LockType = LockType.EXCLUSIVE,
        timeout_seconds: Optional[float] = None,
    ):
        self.manager = manager
        self.resource_id = resource_id
        self.holder_id = holder_id
        self.lock_type = lock_type
        self.timeout_seconds = timeout_seconds
        self._lock: Optional[Lock] = None

    async def __aenter__(self) -> Lock:
        mode = LockMode.TIMEOUT if self.timeout_seconds else LockMode.BLOCKING

        self._lock = await self.manager.acquire(
            resource_id=self.resource_id,
            holder_id=self.holder_id,
            lock_type=self.lock_type,
            mode=mode,
            timeout_seconds=self.timeout_seconds,
        )

        if self._lock is None:
            raise TimeoutError(f"Failed to acquire lock on {self.resource_id}")

        return self._lock

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._lock:
            await self.manager.release(self._lock.id)

        return False
