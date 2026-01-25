"""
AION Redis Cache Layer

High-performance caching layer with:
- Async Redis operations
- Cache invalidation strategies
- TTL management
- Distributed locking
- Pub/sub for cache invalidation
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, AsyncGenerator, Callable, Optional, TypeVar
import hashlib

import structlog

from aion.persistence.config import CacheConfig, CacheStrategy

logger = structlog.get_logger(__name__)

T = TypeVar("T")

# Optional redis import
try:
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    aioredis = None


@dataclass
class CacheStats:
    """Cache statistics."""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    evictions: int = 0
    errors: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


@dataclass
class CacheEntry:
    """A cache entry with metadata."""
    key: str
    value: Any
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    access_count: int = 0
    last_accessed: Optional[datetime] = None

    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at


class MemoryCache:
    """
    In-memory LRU cache.

    Used when Redis is not available or as a local cache
    layer in front of Redis.
    """

    def __init__(
        self,
        max_size_mb: int = 256,
        default_ttl: int = 300,
    ):
        self.max_size_mb = max_size_mb
        self.default_ttl = default_ttl
        self._cache: dict[str, CacheEntry] = {}
        self._stats = CacheStats()
        self._lock = asyncio.Lock()

    @property
    def stats(self) -> CacheStats:
        return self._stats

    async def get(self, key: str) -> Optional[Any]:
        """Get a value from cache."""
        async with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                self._stats.misses += 1
                return None

            if entry.is_expired():
                del self._cache[key]
                self._stats.misses += 1
                return None

            entry.access_count += 1
            entry.last_accessed = datetime.now()
            self._stats.hits += 1

            return entry.value

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> None:
        """Set a value in cache."""
        async with self._lock:
            expires_at = None
            if ttl is not None:
                expires_at = datetime.now() + timedelta(seconds=ttl)
            elif self.default_ttl > 0:
                expires_at = datetime.now() + timedelta(seconds=self.default_ttl)

            entry = CacheEntry(
                key=key,
                value=value,
                expires_at=expires_at,
            )

            self._cache[key] = entry
            self._stats.sets += 1

            # Evict if necessary
            await self._evict_if_needed()

    async def delete(self, key: str) -> bool:
        """Delete a key from cache."""
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._stats.deletes += 1
                return True
            return False

    async def clear(self) -> None:
        """Clear all cache entries."""
        async with self._lock:
            self._cache.clear()

    async def exists(self, key: str) -> bool:
        """Check if a key exists."""
        entry = self._cache.get(key)
        return entry is not None and not entry.is_expired()

    async def keys(self, pattern: str = "*") -> list[str]:
        """Get keys matching a pattern."""
        import fnmatch
        return [
            k for k in self._cache.keys()
            if fnmatch.fnmatch(k, pattern)
        ]

    async def _evict_if_needed(self) -> None:
        """Evict entries if cache is too large."""
        # Simple LRU eviction
        while self._estimate_size() > self.max_size_mb * 1024 * 1024:
            # Find least recently used entry
            if not self._cache:
                break

            oldest_key = min(
                self._cache.keys(),
                key=lambda k: self._cache[k].last_accessed or self._cache[k].created_at,
            )
            del self._cache[oldest_key]
            self._stats.evictions += 1

    def _estimate_size(self) -> int:
        """Estimate cache size in bytes."""
        # Rough estimation
        try:
            return len(json.dumps({k: str(v.value) for k, v in self._cache.items()}))
        except Exception:
            return len(self._cache) * 1000  # Assume 1KB per entry


class RedisCache:
    """
    Redis-backed distributed cache.

    Features:
    - Async operations
    - Connection pooling
    - Pub/sub for cache invalidation
    - Distributed locking
    """

    def __init__(self, config: CacheConfig):
        self.config = config
        self._client: Optional["aioredis.Redis"] = None
        self._pubsub: Optional["aioredis.client.PubSub"] = None
        self._stats = CacheStats()
        self._initialized = False
        self._invalidation_handlers: dict[str, list[Callable]] = {}

    @property
    def stats(self) -> CacheStats:
        return self._stats

    async def initialize(self) -> None:
        """Initialize Redis connection."""
        if self._initialized:
            return

        if not REDIS_AVAILABLE:
            raise ImportError(
                "redis is required for Redis cache. "
                "Install with: pip install redis"
            )

        logger.info(
            "Initializing Redis cache",
            host=self.config.redis_host,
            port=self.config.redis_port,
        )

        self._client = aioredis.Redis(
            host=self.config.redis_host,
            port=self.config.redis_port,
            db=self.config.redis_db,
            password=self.config.redis_password,
            ssl=self.config.redis_ssl,
            decode_responses=True,
        )

        # Test connection
        await self._client.ping()

        self._initialized = True
        logger.info("Redis cache initialized")

    async def shutdown(self) -> None:
        """Shutdown Redis connection."""
        if self._pubsub:
            await self._pubsub.close()

        if self._client:
            await self._client.close()

        self._initialized = False
        logger.info("Redis cache shutdown complete")

    def _make_key(self, key: str) -> str:
        """Create prefixed cache key."""
        return f"{self.config.redis_prefix}{key}"

    async def get(self, key: str) -> Optional[Any]:
        """Get a value from cache."""
        try:
            full_key = self._make_key(key)
            value = await self._client.get(full_key)

            if value is None:
                self._stats.misses += 1
                return None

            self._stats.hits += 1
            return json.loads(value)

        except Exception as e:
            logger.error("Redis get error", key=key, error=str(e))
            self._stats.errors += 1
            return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> bool:
        """Set a value in cache."""
        try:
            full_key = self._make_key(key)
            serialized = json.dumps(value, default=str)
            ttl = ttl or self.config.ttl_seconds

            if ttl > 0:
                await self._client.setex(full_key, ttl, serialized)
            else:
                await self._client.set(full_key, serialized)

            self._stats.sets += 1
            return True

        except Exception as e:
            logger.error("Redis set error", key=key, error=str(e))
            self._stats.errors += 1
            return False

    async def delete(self, key: str) -> bool:
        """Delete a key from cache."""
        try:
            full_key = self._make_key(key)
            result = await self._client.delete(full_key)
            self._stats.deletes += 1
            return result > 0

        except Exception as e:
            logger.error("Redis delete error", key=key, error=str(e))
            self._stats.errors += 1
            return False

    async def delete_pattern(self, pattern: str) -> int:
        """Delete keys matching a pattern."""
        try:
            full_pattern = self._make_key(pattern)
            keys = []
            async for key in self._client.scan_iter(match=full_pattern):
                keys.append(key)

            if keys:
                deleted = await self._client.delete(*keys)
                self._stats.deletes += deleted
                return deleted
            return 0

        except Exception as e:
            logger.error("Redis delete pattern error", pattern=pattern, error=str(e))
            self._stats.errors += 1
            return 0

    async def exists(self, key: str) -> bool:
        """Check if a key exists."""
        try:
            full_key = self._make_key(key)
            return await self._client.exists(full_key) > 0
        except Exception as e:
            logger.error("Redis exists error", key=key, error=str(e))
            self._stats.errors += 1
            return False

    async def mget(self, keys: list[str]) -> dict[str, Any]:
        """Get multiple values."""
        try:
            full_keys = [self._make_key(k) for k in keys]
            values = await self._client.mget(full_keys)

            result = {}
            for key, value in zip(keys, values):
                if value is not None:
                    result[key] = json.loads(value)
                    self._stats.hits += 1
                else:
                    self._stats.misses += 1

            return result

        except Exception as e:
            logger.error("Redis mget error", error=str(e))
            self._stats.errors += 1
            return {}

    async def mset(
        self,
        items: dict[str, Any],
        ttl: Optional[int] = None,
    ) -> bool:
        """Set multiple values."""
        try:
            pipeline = self._client.pipeline()
            ttl = ttl or self.config.ttl_seconds

            for key, value in items.items():
                full_key = self._make_key(key)
                serialized = json.dumps(value, default=str)
                if ttl > 0:
                    pipeline.setex(full_key, ttl, serialized)
                else:
                    pipeline.set(full_key, serialized)

            await pipeline.execute()
            self._stats.sets += len(items)
            return True

        except Exception as e:
            logger.error("Redis mset error", error=str(e))
            self._stats.errors += 1
            return False

    # Distributed locking

    async def acquire_lock(
        self,
        name: str,
        timeout: int = 10,
        blocking_timeout: Optional[int] = None,
    ) -> Optional[str]:
        """
        Acquire a distributed lock.

        Returns a lock token if successful, None otherwise.
        """
        import uuid
        token = str(uuid.uuid4())
        lock_key = self._make_key(f"lock:{name}")

        try:
            if blocking_timeout is not None:
                end_time = time.time() + blocking_timeout
                while time.time() < end_time:
                    acquired = await self._client.set(
                        lock_key,
                        token,
                        nx=True,
                        ex=timeout,
                    )
                    if acquired:
                        return token
                    await asyncio.sleep(0.1)
                return None
            else:
                acquired = await self._client.set(
                    lock_key,
                    token,
                    nx=True,
                    ex=timeout,
                )
                return token if acquired else None

        except Exception as e:
            logger.error("Redis lock acquire error", name=name, error=str(e))
            return None

    async def release_lock(self, name: str, token: str) -> bool:
        """
        Release a distributed lock.

        Only releases if the token matches (prevents releasing others' locks).
        """
        lock_key = self._make_key(f"lock:{name}")

        # Use Lua script for atomic check-and-delete
        script = """
        if redis.call("get", KEYS[1]) == ARGV[1] then
            return redis.call("del", KEYS[1])
        else
            return 0
        end
        """

        try:
            result = await self._client.eval(script, 1, lock_key, token)
            return result == 1
        except Exception as e:
            logger.error("Redis lock release error", name=name, error=str(e))
            return False

    # Pub/Sub for cache invalidation

    async def publish_invalidation(self, keys: list[str]) -> None:
        """Publish cache invalidation event."""
        channel = self._make_key("invalidation")
        message = json.dumps({"keys": keys, "timestamp": time.time()})
        await self._client.publish(channel, message)

    async def subscribe_invalidation(
        self,
    ) -> AsyncGenerator[list[str], None]:
        """Subscribe to cache invalidation events."""
        channel = self._make_key("invalidation")

        self._pubsub = self._client.pubsub()
        await self._pubsub.subscribe(channel)

        try:
            async for message in self._pubsub.listen():
                if message["type"] == "message":
                    data = json.loads(message["data"])
                    yield data.get("keys", [])
        finally:
            await self._pubsub.unsubscribe(channel)


class CacheManager:
    """
    Unified cache manager supporting multiple backends.

    Implements cache-aside pattern with optional write-through
    or write-behind strategies.
    """

    def __init__(self, config: CacheConfig):
        self.config = config
        self._local_cache = MemoryCache(
            max_size_mb=config.max_size_mb // 4,  # Use 1/4 for local
            default_ttl=config.ttl_seconds // 2,  # Shorter TTL for local
        )
        self._remote_cache: Optional[RedisCache] = None
        self._stats = CacheStats()
        self._initialized = False

    @property
    def stats(self) -> CacheStats:
        return CacheStats(
            hits=self._local_cache.stats.hits + (
                self._remote_cache.stats.hits if self._remote_cache else 0
            ),
            misses=self._local_cache.stats.misses + (
                self._remote_cache.stats.misses if self._remote_cache else 0
            ),
            sets=self._local_cache.stats.sets + (
                self._remote_cache.stats.sets if self._remote_cache else 0
            ),
            deletes=self._local_cache.stats.deletes + (
                self._remote_cache.stats.deletes if self._remote_cache else 0
            ),
            evictions=self._local_cache.stats.evictions,
            errors=self._remote_cache.stats.errors if self._remote_cache else 0,
        )

    async def initialize(self) -> None:
        """Initialize cache backends."""
        if self._initialized:
            return

        if self.config.backend == "redis" and REDIS_AVAILABLE:
            self._remote_cache = RedisCache(self.config)
            await self._remote_cache.initialize()

        self._initialized = True

    async def shutdown(self) -> None:
        """Shutdown cache backends."""
        if self._remote_cache:
            await self._remote_cache.shutdown()

        self._initialized = False

    async def get(self, key: str) -> Optional[Any]:
        """Get a value from cache (tries local first, then remote)."""
        # Try local cache first
        value = await self._local_cache.get(key)
        if value is not None:
            return value

        # Try remote cache
        if self._remote_cache:
            value = await self._remote_cache.get(key)
            if value is not None:
                # Populate local cache
                await self._local_cache.set(key, value)
                return value

        return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> bool:
        """Set a value in cache."""
        # Always set in local cache
        await self._local_cache.set(key, value, ttl)

        # Set in remote cache based on strategy
        if self._remote_cache:
            if self.config.strategy == CacheStrategy.WRITE_THROUGH:
                return await self._remote_cache.set(key, value, ttl)
            elif self.config.strategy == CacheStrategy.WRITE_BEHIND:
                # Async write to remote (fire and forget)
                asyncio.create_task(self._remote_cache.set(key, value, ttl))
            # WRITE_AROUND: Don't write to remote cache

        return True

    async def delete(self, key: str) -> bool:
        """Delete a key from all caches."""
        await self._local_cache.delete(key)

        if self._remote_cache:
            await self._remote_cache.delete(key)
            # Publish invalidation for other instances
            await self._remote_cache.publish_invalidation([key])

        return True

    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate keys matching a pattern."""
        count = 0

        # Local cache
        for key in await self._local_cache.keys(pattern):
            await self._local_cache.delete(key)
            count += 1

        # Remote cache
        if self._remote_cache:
            count += await self._remote_cache.delete_pattern(pattern)

        return count

    def cache_key(self, *parts: Any) -> str:
        """Generate a cache key from parts."""
        key_string = ":".join(str(p) for p in parts)
        return hashlib.md5(key_string.encode()).hexdigest()

    async def get_or_set(
        self,
        key: str,
        factory: Callable[[], Any],
        ttl: Optional[int] = None,
    ) -> Any:
        """Get from cache or compute and cache."""
        value = await self.get(key)
        if value is not None:
            return value

        # Compute value
        if asyncio.iscoroutinefunction(factory):
            value = await factory()
        else:
            value = factory()

        await self.set(key, value, ttl)
        return value
