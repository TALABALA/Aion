"""
AION Distributed Memory - Distributed Cache

Two-tier caching layer combining a local LRU cache with a distributed
shard-based cache.  Supports TTL-based expiry, cache invalidation
broadcasting, cache-aside pattern, and hit/miss rate tracking.
"""

from __future__ import annotations

import asyncio
import fnmatch
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set

import structlog

if TYPE_CHECKING:
    from aion.distributed.cluster.manager import ClusterManager

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_LOCAL_MAX_SIZE = 4096
_EXPIRY_SWEEP_INTERVAL = 10.0  # seconds between background expiry sweeps


# ---------------------------------------------------------------------------
# Cache entry wrapper
# ---------------------------------------------------------------------------


@dataclass
class CacheEntry:
    """Single cache entry with optional TTL."""
    key: str
    value: Any
    created_at: float = field(default_factory=time.monotonic)
    ttl_seconds: Optional[float] = None
    access_count: int = 0
    last_accessed: float = field(default_factory=time.monotonic)

    @property
    def is_expired(self) -> bool:
        """Check whether this entry has exceeded its TTL."""
        if self.ttl_seconds is None:
            return False
        return (time.monotonic() - self.created_at) > self.ttl_seconds

    def touch(self) -> None:
        """Record an access."""
        self.access_count += 1
        self.last_accessed = time.monotonic()


# ---------------------------------------------------------------------------
# Local LRU Cache
# ---------------------------------------------------------------------------


class _LocalLRUCache:
    """
    Thread-safe LRU cache backed by an ``OrderedDict``.

    ``move_to_end`` is used on every access so the least-recently-used
    items are always at the front, and eviction is O(1) via ``popitem``.
    """

    def __init__(self, max_size: int = DEFAULT_LOCAL_MAX_SIZE) -> None:
        self._max_size = max_size
        self._store: OrderedDict[str, CacheEntry] = OrderedDict()
        self._hits: int = 0
        self._misses: int = 0

    def get(self, key: str) -> Optional[CacheEntry]:
        entry = self._store.get(key)
        if entry is None:
            self._misses += 1
            return None
        if entry.is_expired:
            self._store.pop(key, None)
            self._misses += 1
            return None
        # Move to end (most recently used)
        self._store.move_to_end(key)
        entry.touch()
        self._hits += 1
        return entry

    def put(self, key: str, value: Any, ttl_seconds: Optional[float] = None) -> None:
        if key in self._store:
            self._store.move_to_end(key)
            self._store[key] = CacheEntry(key=key, value=value, ttl_seconds=ttl_seconds)
        else:
            if len(self._store) >= self._max_size:
                self._store.popitem(last=False)  # evict LRU
            self._store[key] = CacheEntry(key=key, value=value, ttl_seconds=ttl_seconds)

    def delete(self, key: str) -> bool:
        return self._store.pop(key, None) is not None

    def invalidate_pattern(self, pattern: str) -> int:
        """Remove all keys matching *pattern* (fnmatch-style)."""
        to_remove = [k for k in self._store if fnmatch.fnmatch(k, pattern)]
        for k in to_remove:
            del self._store[k]
        return len(to_remove)

    def clear(self) -> None:
        self._store.clear()

    def evict_expired(self) -> int:
        """Remove all expired entries. Returns count of evicted items."""
        expired = [k for k, v in self._store.items() if v.is_expired]
        for k in expired:
            del self._store[k]
        return len(expired)

    @property
    def size(self) -> int:
        return len(self._store)

    @property
    def stats(self) -> Dict[str, Any]:
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        return {
            "size": self.size,
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(hit_rate, 4),
        }


# ---------------------------------------------------------------------------
# Distributed Cache
# ---------------------------------------------------------------------------


class DistributedCache:
    """
    Two-tier distributed cache.

    Tier 1 -- fast, node-local LRU cache.
    Tier 2 -- shard-based distributed store for cross-node sharing.

    Reads follow the **cache-aside** pattern: check local cache, then
    distributed cache, then fall through to the caller-provided loader
    function.  Writes update both tiers.  Invalidation is broadcast to
    every node so stale local copies are purged cluster-wide.
    """

    def __init__(
        self,
        cluster_manager: ClusterManager,
        *,
        max_local_size: int = DEFAULT_LOCAL_MAX_SIZE,
    ) -> None:
        self._cluster = cluster_manager
        self._local = _LocalLRUCache(max_size=max_local_size)
        self._distributed_store: Dict[str, CacheEntry] = {}
        self._invalidation_listeners: List[Callable[[str], Any]] = []
        self._lock = asyncio.Lock()
        self._expiry_task: Optional[asyncio.Task[None]] = None
        self._running = False

        # Global stats
        self._distributed_hits: int = 0
        self._distributed_misses: int = 0
        self._invalidations_sent: int = 0
        self._invalidations_received: int = 0

    # -- lifecycle --------------------------------------------------------

    async def start(self) -> None:
        """Start the background expiry sweep."""
        if self._running:
            return
        self._running = True
        self._expiry_task = asyncio.create_task(self._expiry_loop())
        logger.info("distributed_cache.started")

    async def stop(self) -> None:
        """Stop the background expiry sweep and clear all state."""
        self._running = False
        if self._expiry_task is not None:
            self._expiry_task.cancel()
            try:
                await self._expiry_task
            except asyncio.CancelledError:
                pass
            self._expiry_task = None
        logger.info("distributed_cache.stopped")

    # -- public API -------------------------------------------------------

    async def get(self, key: str) -> Optional[Any]:
        """
        Read a value from the cache (cache-aside).

        Checks local LRU first, then the distributed store.
        Returns ``None`` on a total miss.
        """
        # Tier 1: local LRU
        entry = self._local.get(key)
        if entry is not None:
            return entry.value

        # Tier 2: distributed
        async with self._lock:
            dist_entry = self._distributed_store.get(key)
        if dist_entry is not None and not dist_entry.is_expired:
            dist_entry.touch()
            self._distributed_hits += 1
            # Promote to local LRU
            self._local.put(key, dist_entry.value, dist_entry.ttl_seconds)
            return dist_entry.value

        self._distributed_misses += 1
        return None

    async def get_or_load(
        self,
        key: str,
        loader: Callable[[], Any],
        ttl_seconds: Optional[float] = None,
    ) -> Any:
        """
        Cache-aside with automatic population.

        If the key is not in any tier, call *loader* to fetch the value,
        store it, and return it.
        """
        value = await self.get(key)
        if value is not None:
            return value

        # Miss -- load and populate both tiers
        value = loader() if not asyncio.iscoroutinefunction(loader) else await loader()
        await self.set(key, value, ttl_seconds=ttl_seconds)
        return value

    async def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[float] = None,
    ) -> None:
        """Store a value in both tiers."""
        entry = CacheEntry(key=key, value=value, ttl_seconds=ttl_seconds)

        # Tier 1
        self._local.put(key, value, ttl_seconds)

        # Tier 2
        async with self._lock:
            self._distributed_store[key] = entry

        logger.debug("distributed_cache.set", key=key,
                      ttl=ttl_seconds)

    async def delete(self, key: str) -> bool:
        """Remove a key from both tiers and broadcast invalidation."""
        local_deleted = self._local.delete(key)
        async with self._lock:
            dist_deleted = self._distributed_store.pop(key, None) is not None

        if local_deleted or dist_deleted:
            await self._broadcast_invalidation(key)

        return local_deleted or dist_deleted

    async def invalidate_pattern(self, pattern: str) -> int:
        """
        Remove all keys matching *pattern* from both tiers.

        Uses ``fnmatch``-style wildcards (e.g. ``user:*``).
        """
        local_count = self._local.invalidate_pattern(pattern)

        async with self._lock:
            to_remove = [
                k for k in self._distributed_store
                if fnmatch.fnmatch(k, pattern)
            ]
            for k in to_remove:
                del self._distributed_store[k]

        total = local_count + len(to_remove)
        if total > 0:
            await self._broadcast_invalidation(pattern)
        logger.info("distributed_cache.invalidate_pattern",
                     pattern=pattern, removed=total)
        return total

    async def clear(self) -> None:
        """Purge all entries from both tiers."""
        self._local.clear()
        async with self._lock:
            self._distributed_store.clear()
        await self._broadcast_invalidation("*")
        logger.info("distributed_cache.cleared")

    async def get_stats(self) -> Dict[str, Any]:
        """Return comprehensive cache statistics."""
        dist_total = self._distributed_hits + self._distributed_misses
        dist_hit_rate = (
            self._distributed_hits / dist_total if dist_total > 0 else 0.0
        )

        async with self._lock:
            dist_size = len(self._distributed_store)

        return {
            "local": self._local.stats,
            "distributed": {
                "size": dist_size,
                "hits": self._distributed_hits,
                "misses": self._distributed_misses,
                "hit_rate": round(dist_hit_rate, 4),
            },
            "invalidations_sent": self._invalidations_sent,
            "invalidations_received": self._invalidations_received,
        }

    # -- invalidation -----------------------------------------------------

    def on_invalidation(self, callback: Callable[[str], Any]) -> None:
        """Register a callback for incoming invalidation events."""
        self._invalidation_listeners.append(callback)

    async def handle_remote_invalidation(self, key_or_pattern: str) -> None:
        """Handle an invalidation message received from another node."""
        self._invalidations_received += 1
        if "*" in key_or_pattern or "?" in key_or_pattern:
            self._local.invalidate_pattern(key_or_pattern)
            async with self._lock:
                to_remove = [
                    k for k in self._distributed_store
                    if fnmatch.fnmatch(k, key_or_pattern)
                ]
                for k in to_remove:
                    del self._distributed_store[k]
        else:
            self._local.delete(key_or_pattern)
            async with self._lock:
                self._distributed_store.pop(key_or_pattern, None)

        logger.debug("distributed_cache.remote_invalidation",
                      pattern=key_or_pattern)

    async def _broadcast_invalidation(self, key_or_pattern: str) -> None:
        """Broadcast an invalidation to all cluster nodes."""
        self._invalidations_sent += 1
        for listener in self._invalidation_listeners:
            try:
                result = listener(key_or_pattern)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as exc:
                logger.warning("distributed_cache.invalidation_broadcast_error",
                               error=str(exc))

        logger.debug("distributed_cache.broadcast_invalidation",
                      key_or_pattern=key_or_pattern)

    # -- background expiry ------------------------------------------------

    async def _expiry_loop(self) -> None:
        """Periodically sweep both tiers for expired entries."""
        while self._running:
            try:
                await asyncio.sleep(_EXPIRY_SWEEP_INTERVAL)
                local_evicted = self._local.evict_expired()

                async with self._lock:
                    expired_keys = [
                        k for k, v in self._distributed_store.items()
                        if v.is_expired
                    ]
                    for k in expired_keys:
                        del self._distributed_store[k]

                total = local_evicted + len(expired_keys)
                if total > 0:
                    logger.debug("distributed_cache.expiry_sweep",
                                  local=local_evicted,
                                  distributed=len(expired_keys))
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("distributed_cache.expiry_error", error=str(exc))
