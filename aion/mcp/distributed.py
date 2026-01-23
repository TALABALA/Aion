"""
AION MCP Distributed Primitives

Redis/etcd-backed distributed versions of resilience patterns for horizontal scaling:
- Distributed Circuit Breaker with shared state
- Distributed Rate Limiter with sliding window
- Distributed Cache with stampede prevention
- Distributed Locks for coordination
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

import structlog

logger = structlog.get_logger(__name__)

T = TypeVar("T")


# ============================================
# Backend Abstraction
# ============================================

class DistributedBackend(ABC):
    """Abstract backend for distributed state storage."""

    @abstractmethod
    async def get(self, key: str) -> Optional[str]:
        """Get a value by key."""
        pass

    @abstractmethod
    async def set(
        self,
        key: str,
        value: str,
        ttl: Optional[float] = None,
    ) -> bool:
        """Set a value with optional TTL."""
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete a key."""
        pass

    @abstractmethod
    async def incr(self, key: str, amount: int = 1) -> int:
        """Increment a counter."""
        pass

    @abstractmethod
    async def expire(self, key: str, ttl: float) -> bool:
        """Set expiration on a key."""
        pass

    @abstractmethod
    async def setnx(self, key: str, value: str, ttl: Optional[float] = None) -> bool:
        """Set if not exists (for locks)."""
        pass

    @abstractmethod
    async def eval_script(
        self,
        script: str,
        keys: List[str],
        args: List[str],
    ) -> Any:
        """Execute a Lua script (Redis) or equivalent."""
        pass

    @abstractmethod
    async def zadd(
        self,
        key: str,
        score: float,
        member: str,
    ) -> int:
        """Add to sorted set."""
        pass

    @abstractmethod
    async def zremrangebyscore(
        self,
        key: str,
        min_score: float,
        max_score: float,
    ) -> int:
        """Remove from sorted set by score range."""
        pass

    @abstractmethod
    async def zcard(self, key: str) -> int:
        """Get sorted set cardinality."""
        pass

    @abstractmethod
    async def publish(self, channel: str, message: str) -> int:
        """Publish message to channel."""
        pass

    @abstractmethod
    async def subscribe(
        self,
        channel: str,
        callback: Callable[[str], Awaitable[None]],
    ) -> None:
        """Subscribe to channel."""
        pass


class RedisBackend(DistributedBackend):
    """Redis implementation of distributed backend."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        ssl: bool = False,
        ssl_cert_reqs: Optional[str] = None,
        ssl_ca_certs: Optional[str] = None,
        ssl_certfile: Optional[str] = None,
        ssl_keyfile: Optional[str] = None,
        prefix: str = "aion:mcp:",
        pool_size: int = 10,
    ):
        """
        Initialize Redis backend.

        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            password: Redis password
            ssl: Enable SSL/TLS
            ssl_*: SSL configuration for mTLS
            prefix: Key prefix for namespacing
            pool_size: Connection pool size
        """
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.ssl = ssl
        self.ssl_config = {
            "ssl_cert_reqs": ssl_cert_reqs,
            "ssl_ca_certs": ssl_ca_certs,
            "ssl_certfile": ssl_certfile,
            "ssl_keyfile": ssl_keyfile,
        }
        self.prefix = prefix
        self.pool_size = pool_size

        self._redis: Optional[Any] = None
        self._pubsub: Optional[Any] = None
        self._subscriptions: Dict[str, Callable] = {}

    async def connect(self) -> None:
        """Connect to Redis."""
        try:
            import redis.asyncio as redis

            ssl_kwargs = {}
            if self.ssl:
                ssl_kwargs["ssl"] = True
                for k, v in self.ssl_config.items():
                    if v is not None:
                        ssl_kwargs[k] = v

            self._redis = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                max_connections=self.pool_size,
                decode_responses=True,
                **ssl_kwargs,
            )

            # Test connection
            await self._redis.ping()
            logger.info(
                "Connected to Redis",
                host=self.host,
                port=self.port,
                ssl=self.ssl,
            )

        except ImportError:
            raise ImportError(
                "redis package required for distributed backend. "
                "Install with: pip install redis[hiredis]"
            )

    async def close(self) -> None:
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()
            self._redis = None

    def _key(self, key: str) -> str:
        """Add prefix to key."""
        return f"{self.prefix}{key}"

    async def get(self, key: str) -> Optional[str]:
        return await self._redis.get(self._key(key))

    async def set(
        self,
        key: str,
        value: str,
        ttl: Optional[float] = None,
    ) -> bool:
        if ttl:
            return await self._redis.setex(self._key(key), int(ttl), value)
        return await self._redis.set(self._key(key), value)

    async def delete(self, key: str) -> bool:
        return await self._redis.delete(self._key(key)) > 0

    async def incr(self, key: str, amount: int = 1) -> int:
        return await self._redis.incrby(self._key(key), amount)

    async def expire(self, key: str, ttl: float) -> bool:
        return await self._redis.expire(self._key(key), int(ttl))

    async def setnx(self, key: str, value: str, ttl: Optional[float] = None) -> bool:
        if ttl:
            return await self._redis.set(
                self._key(key),
                value,
                nx=True,
                ex=int(ttl),
            )
        return await self._redis.setnx(self._key(key), value)

    async def eval_script(
        self,
        script: str,
        keys: List[str],
        args: List[str],
    ) -> Any:
        prefixed_keys = [self._key(k) for k in keys]
        return await self._redis.eval(script, len(prefixed_keys), *prefixed_keys, *args)

    async def zadd(self, key: str, score: float, member: str) -> int:
        return await self._redis.zadd(self._key(key), {member: score})

    async def zremrangebyscore(
        self,
        key: str,
        min_score: float,
        max_score: float,
    ) -> int:
        return await self._redis.zremrangebyscore(
            self._key(key),
            min_score,
            max_score,
        )

    async def zcard(self, key: str) -> int:
        return await self._redis.zcard(self._key(key))

    async def publish(self, channel: str, message: str) -> int:
        return await self._redis.publish(self._key(channel), message)

    async def subscribe(
        self,
        channel: str,
        callback: Callable[[str], Awaitable[None]],
    ) -> None:
        if not self._pubsub:
            self._pubsub = self._redis.pubsub()

        await self._pubsub.subscribe(self._key(channel))
        self._subscriptions[channel] = callback

        # Start listener if not already running
        asyncio.create_task(self._listen())

    async def _listen(self) -> None:
        """Listen for pubsub messages."""
        async for message in self._pubsub.listen():
            if message["type"] == "message":
                channel = message["channel"].replace(self.prefix, "")
                if channel in self._subscriptions:
                    await self._subscriptions[channel](message["data"])


class InMemoryBackend(DistributedBackend):
    """
    In-memory implementation for testing/single-node deployment.

    Provides the same interface as Redis but uses local memory.
    """

    def __init__(self, prefix: str = "aion:mcp:"):
        self.prefix = prefix
        self._data: Dict[str, Tuple[str, Optional[float]]] = {}  # value, expiry
        self._sorted_sets: Dict[str, Dict[str, float]] = {}
        self._channels: Dict[str, List[Callable]] = {}
        self._lock = asyncio.Lock()

    def _key(self, key: str) -> str:
        return f"{self.prefix}{key}"

    def _is_expired(self, key: str) -> bool:
        if key not in self._data:
            return True
        _, expiry = self._data[key]
        if expiry and time.time() > expiry:
            del self._data[key]
            return True
        return False

    async def get(self, key: str) -> Optional[str]:
        async with self._lock:
            k = self._key(key)
            if self._is_expired(k):
                return None
            return self._data[k][0]

    async def set(
        self,
        key: str,
        value: str,
        ttl: Optional[float] = None,
    ) -> bool:
        async with self._lock:
            expiry = time.time() + ttl if ttl else None
            self._data[self._key(key)] = (value, expiry)
            return True

    async def delete(self, key: str) -> bool:
        async with self._lock:
            k = self._key(key)
            if k in self._data:
                del self._data[k]
                return True
            return False

    async def incr(self, key: str, amount: int = 1) -> int:
        async with self._lock:
            k = self._key(key)
            if k not in self._data or self._is_expired(k):
                self._data[k] = ("0", None)
            val = int(self._data[k][0]) + amount
            self._data[k] = (str(val), self._data[k][1])
            return val

    async def expire(self, key: str, ttl: float) -> bool:
        async with self._lock:
            k = self._key(key)
            if k in self._data:
                self._data[k] = (self._data[k][0], time.time() + ttl)
                return True
            return False

    async def setnx(self, key: str, value: str, ttl: Optional[float] = None) -> bool:
        async with self._lock:
            k = self._key(key)
            if k not in self._data or self._is_expired(k):
                expiry = time.time() + ttl if ttl else None
                self._data[k] = (value, expiry)
                return True
            return False

    async def eval_script(
        self,
        script: str,
        keys: List[str],
        args: List[str],
    ) -> Any:
        # Simplified script execution for common patterns
        # Real implementation would need a Lua interpreter
        raise NotImplementedError("Lua scripts not supported in InMemoryBackend")

    async def zadd(self, key: str, score: float, member: str) -> int:
        async with self._lock:
            k = self._key(key)
            if k not in self._sorted_sets:
                self._sorted_sets[k] = {}
            is_new = member not in self._sorted_sets[k]
            self._sorted_sets[k][member] = score
            return 1 if is_new else 0

    async def zremrangebyscore(
        self,
        key: str,
        min_score: float,
        max_score: float,
    ) -> int:
        async with self._lock:
            k = self._key(key)
            if k not in self._sorted_sets:
                return 0
            to_remove = [
                m for m, s in self._sorted_sets[k].items()
                if min_score <= s <= max_score
            ]
            for m in to_remove:
                del self._sorted_sets[k][m]
            return len(to_remove)

    async def zcard(self, key: str) -> int:
        async with self._lock:
            k = self._key(key)
            return len(self._sorted_sets.get(k, {}))

    async def publish(self, channel: str, message: str) -> int:
        async with self._lock:
            k = self._key(channel)
            callbacks = self._channels.get(k, [])
            for cb in callbacks:
                asyncio.create_task(cb(message))
            return len(callbacks)

    async def subscribe(
        self,
        channel: str,
        callback: Callable[[str], Awaitable[None]],
    ) -> None:
        async with self._lock:
            k = self._key(channel)
            if k not in self._channels:
                self._channels[k] = []
            self._channels[k].append(callback)


# ============================================
# Distributed Lock
# ============================================

class DistributedLock:
    """
    Distributed lock using Redlock algorithm.

    Provides mutual exclusion across multiple nodes.
    """

    def __init__(
        self,
        backend: DistributedBackend,
        name: str,
        ttl: float = 30.0,
        retry_delay: float = 0.1,
        retry_count: int = 3,
    ):
        """
        Initialize distributed lock.

        Args:
            backend: Distributed backend
            name: Lock name
            ttl: Lock TTL in seconds
            retry_delay: Delay between retries
            retry_count: Number of retry attempts
        """
        self.backend = backend
        self.name = name
        self.ttl = ttl
        self.retry_delay = retry_delay
        self.retry_count = retry_count

        self._lock_key = f"lock:{name}"
        self._lock_value: Optional[str] = None

    async def acquire(self, blocking: bool = True) -> bool:
        """
        Acquire the lock.

        Args:
            blocking: Whether to block until acquired

        Returns:
            True if acquired
        """
        self._lock_value = str(uuid.uuid4())

        for attempt in range(self.retry_count if blocking else 1):
            if await self.backend.setnx(
                self._lock_key,
                self._lock_value,
                ttl=self.ttl,
            ):
                return True

            if blocking and attempt < self.retry_count - 1:
                await asyncio.sleep(self.retry_delay)

        return False

    async def release(self) -> bool:
        """Release the lock."""
        if not self._lock_value:
            return False

        # Only release if we own the lock
        current = await self.backend.get(self._lock_key)
        if current == self._lock_value:
            await self.backend.delete(self._lock_key)
            self._lock_value = None
            return True

        return False

    async def extend(self, ttl: Optional[float] = None) -> bool:
        """Extend lock TTL."""
        if not self._lock_value:
            return False

        current = await self.backend.get(self._lock_key)
        if current == self._lock_value:
            return await self.backend.expire(
                self._lock_key,
                ttl or self.ttl,
            )
        return False

    @asynccontextmanager
    async def __call__(self):
        """Use lock as async context manager."""
        if await self.acquire():
            try:
                yield
            finally:
                await self.release()
        else:
            raise RuntimeError(f"Failed to acquire lock: {self.name}")

    async def __aenter__(self):
        if not await self.acquire():
            raise RuntimeError(f"Failed to acquire lock: {self.name}")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.release()


# ============================================
# Distributed Circuit Breaker
# ============================================

class DistributedCircuitBreaker:
    """
    Distributed circuit breaker with sliding window failure tracking.

    Features:
    - Redis-backed state for cluster-wide circuit breaking
    - Sliding window for accurate failure rate calculation
    - Gradual recovery in half-open state
    - Adaptive thresholds based on error patterns
    """

    # Lua script for atomic state transitions
    TRANSITION_SCRIPT = """
    local state_key = KEYS[1]
    local failures_key = KEYS[2]
    local current_state = redis.call('GET', state_key) or 'closed'
    local new_state = ARGV[1]
    local ttl = tonumber(ARGV[2])

    redis.call('SET', state_key, new_state)
    if ttl > 0 then
        redis.call('EXPIRE', state_key, ttl)
    end

    if new_state == 'closed' then
        redis.call('DEL', failures_key)
    end

    return current_state
    """

    def __init__(
        self,
        backend: DistributedBackend,
        name: str,
        failure_threshold: int = 5,
        success_threshold: int = 3,
        timeout: float = 30.0,
        window_size: float = 60.0,
        half_open_max_calls: int = 3,
        failure_rate_threshold: float = 0.5,
        adaptive: bool = True,
        min_calls_for_adaptive: int = 10,
    ):
        """
        Initialize distributed circuit breaker.

        Args:
            backend: Distributed backend
            name: Circuit breaker name
            failure_threshold: Failures before opening
            success_threshold: Successes in half-open before closing
            timeout: Time in open state before trying half-open
            window_size: Sliding window size in seconds
            half_open_max_calls: Max concurrent calls in half-open
            failure_rate_threshold: Failure rate to trigger open
            adaptive: Enable adaptive thresholds
            min_calls_for_adaptive: Min calls before adaptive kicks in
        """
        self.backend = backend
        self.name = name
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.timeout = timeout
        self.window_size = window_size
        self.half_open_max_calls = half_open_max_calls
        self.failure_rate_threshold = failure_rate_threshold
        self.adaptive = adaptive
        self.min_calls_for_adaptive = min_calls_for_adaptive

        # Keys
        self._state_key = f"cb:{name}:state"
        self._failures_key = f"cb:{name}:failures"
        self._successes_key = f"cb:{name}:successes"
        self._half_open_count_key = f"cb:{name}:half_open_count"
        self._metrics_key = f"cb:{name}:metrics"
        self._opened_at_key = f"cb:{name}:opened_at"

        # Local cache for performance
        self._local_state: Optional[str] = None
        self._local_state_time: float = 0
        self._state_cache_ttl: float = 0.5  # Cache state for 500ms

    async def get_state(self) -> str:
        """Get current circuit state."""
        # Check local cache
        if (
            self._local_state
            and time.time() - self._local_state_time < self._state_cache_ttl
        ):
            return self._local_state

        state = await self.backend.get(self._state_key)
        self._local_state = state or "closed"

        # Check if should transition from open to half-open
        if self._local_state == "open":
            opened_at = await self.backend.get(self._opened_at_key)
            if opened_at and time.time() - float(opened_at) >= self.timeout:
                await self._transition_to("half_open")
                self._local_state = "half_open"

        self._local_state_time = time.time()
        return self._local_state

    async def _transition_to(self, new_state: str) -> str:
        """Transition to new state atomically."""
        try:
            old_state = await self.backend.eval_script(
                self.TRANSITION_SCRIPT,
                [self._state_key, self._failures_key],
                [new_state, str(int(self.timeout * 2))],
            )
        except NotImplementedError:
            # Fallback for InMemoryBackend
            old_state = await self.backend.get(self._state_key) or "closed"
            await self.backend.set(self._state_key, new_state)
            if new_state == "closed":
                await self.backend.delete(self._failures_key)

        self._local_state = new_state
        self._local_state_time = time.time()

        if new_state == "open":
            await self.backend.set(self._opened_at_key, str(time.time()))

        logger.info(
            "Circuit breaker state transition",
            name=self.name,
            old_state=old_state,
            new_state=new_state,
        )

        return old_state

    async def record_success(self) -> None:
        """Record a successful call."""
        now = time.time()

        # Add to sliding window
        await self.backend.zadd(self._successes_key, now, f"{now}:{uuid.uuid4()}")
        await self.backend.zremrangebyscore(
            self._successes_key,
            0,
            now - self.window_size,
        )

        state = await self.get_state()

        if state == "half_open":
            # Count successes in half-open
            success_count = await self.backend.zcard(self._successes_key)
            if success_count >= self.success_threshold:
                await self._transition_to("closed")

            # Decrement half-open counter
            await self.backend.incr(self._half_open_count_key, -1)

    async def record_failure(self, error_type: Optional[str] = None) -> None:
        """Record a failed call."""
        now = time.time()

        # Add to sliding window with error type
        member = f"{now}:{error_type or 'unknown'}:{uuid.uuid4()}"
        await self.backend.zadd(self._failures_key, now, member)
        await self.backend.zremrangebyscore(
            self._failures_key,
            0,
            now - self.window_size,
        )

        state = await self.get_state()

        if state == "half_open":
            # Any failure in half-open reopens circuit
            await self._transition_to("open")
            await self.backend.incr(self._half_open_count_key, -1)
        elif state == "closed":
            # Check if should open
            failure_count = await self.backend.zcard(self._failures_key)

            should_open = False

            if failure_count >= self.failure_threshold:
                should_open = True
            elif self.adaptive:
                # Check failure rate
                success_count = await self.backend.zcard(self._successes_key)
                total = failure_count + success_count

                if total >= self.min_calls_for_adaptive:
                    failure_rate = failure_count / total
                    if failure_rate >= self.failure_rate_threshold:
                        should_open = True
                        logger.info(
                            "Adaptive circuit breaker triggered",
                            name=self.name,
                            failure_rate=failure_rate,
                        )

            if should_open:
                await self._transition_to("open")

    async def allow_request(self) -> bool:
        """Check if request is allowed."""
        state = await self.get_state()

        if state == "closed":
            return True

        if state == "open":
            return False

        # Half-open: allow limited requests
        count = await self.backend.incr(self._half_open_count_key)
        if count <= self.half_open_max_calls:
            return True

        # Too many concurrent half-open requests
        await self.backend.incr(self._half_open_count_key, -1)
        return False

    async def __aenter__(self):
        if not await self.allow_request():
            state = await self.get_state()
            raise CircuitBreakerOpenError(
                f"Circuit breaker {self.name} is {state}",
                name=self.name,
                state=state,
            )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            await self.record_success()
        else:
            await self.record_failure(exc_type.__name__ if exc_type else None)
        return False

    async def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        state = await self.get_state()
        failure_count = await self.backend.zcard(self._failures_key)
        success_count = await self.backend.zcard(self._successes_key)
        total = failure_count + success_count

        return {
            "name": self.name,
            "state": state,
            "failure_count": failure_count,
            "success_count": success_count,
            "failure_rate": failure_count / total if total > 0 else 0.0,
            "window_size": self.window_size,
            "adaptive": self.adaptive,
        }


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open."""

    def __init__(self, message: str, name: str, state: str):
        super().__init__(message)
        self.name = name
        self.state = state


# ============================================
# Distributed Rate Limiter
# ============================================

class DistributedRateLimiter:
    """
    Distributed sliding window rate limiter.

    Uses Redis sorted sets for accurate distributed rate limiting.
    """

    def __init__(
        self,
        backend: DistributedBackend,
        name: str,
        rate: float,
        window: float = 1.0,
        burst: Optional[int] = None,
    ):
        """
        Initialize distributed rate limiter.

        Args:
            backend: Distributed backend
            name: Rate limiter name
            rate: Requests per window
            window: Window size in seconds
            burst: Optional burst allowance
        """
        self.backend = backend
        self.name = name
        self.rate = rate
        self.window = window
        self.burst = burst or int(rate * 1.5)

        self._key = f"rl:{name}"

    async def acquire(self, tokens: int = 1) -> bool:
        """
        Acquire rate limit tokens.

        Args:
            tokens: Number of tokens to acquire

        Returns:
            True if allowed
        """
        now = time.time()
        window_start = now - self.window

        # Clean old entries
        await self.backend.zremrangebyscore(self._key, 0, window_start)

        # Check current count
        current = await self.backend.zcard(self._key)

        if current + tokens > self.rate:
            return False

        # Add new entries
        for i in range(tokens):
            await self.backend.zadd(self._key, now, f"{now}:{uuid.uuid4()}")

        # Set expiry
        await self.backend.expire(self._key, self.window * 2)

        return True

    async def get_remaining(self) -> int:
        """Get remaining tokens in current window."""
        now = time.time()
        window_start = now - self.window

        await self.backend.zremrangebyscore(self._key, 0, window_start)
        current = await self.backend.zcard(self._key)

        return max(0, int(self.rate) - current)

    async def reset(self) -> None:
        """Reset the rate limiter."""
        await self.backend.delete(self._key)


# ============================================
# Distributed Cache with Stampede Prevention
# ============================================

class DistributedCache(Generic[T]):
    """
    Distributed cache with single-flight stampede prevention.

    Features:
    - Redis-backed distributed cache
    - Single-flight pattern prevents thundering herd
    - Probabilistic early expiration
    - Async refresh
    """

    def __init__(
        self,
        backend: DistributedBackend,
        name: str,
        default_ttl: float = 300.0,
        early_expiry_probability: float = 0.1,
        early_expiry_window: float = 30.0,
        serializer: Optional[Callable[[T], str]] = None,
        deserializer: Optional[Callable[[str], T]] = None,
    ):
        """
        Initialize distributed cache.

        Args:
            backend: Distributed backend
            name: Cache name
            default_ttl: Default TTL in seconds
            early_expiry_probability: Probability of early refresh
            early_expiry_window: Window before expiry for early refresh
            serializer: Custom serializer
            deserializer: Custom deserializer
        """
        self.backend = backend
        self.name = name
        self.default_ttl = default_ttl
        self.early_expiry_probability = early_expiry_probability
        self.early_expiry_window = early_expiry_window
        self.serializer = serializer or json.dumps
        self.deserializer = deserializer or json.loads

        self._prefix = f"cache:{name}:"
        self._lock_prefix = f"cache_lock:{name}:"
        self._inflight: Dict[str, asyncio.Future] = {}

    def _cache_key(self, key: str) -> str:
        return f"{self._prefix}{key}"

    def _lock_key(self, key: str) -> str:
        return f"{self._lock_prefix}{key}"

    async def get(
        self,
        key: str,
        loader: Optional[Callable[[], Awaitable[T]]] = None,
        ttl: Optional[float] = None,
    ) -> Optional[T]:
        """
        Get value from cache with optional loading.

        Uses single-flight pattern: if multiple requests miss cache
        simultaneously, only one will load while others wait.

        Args:
            key: Cache key
            loader: Optional async function to load value on miss
            ttl: Optional TTL override

        Returns:
            Cached value or None
        """
        cache_key = self._cache_key(key)

        # Try to get from cache
        data = await self.backend.get(cache_key)

        if data:
            try:
                entry = json.loads(data)
                value = self.deserializer(entry["value"])

                # Check for probabilistic early expiry
                if self._should_early_refresh(entry):
                    asyncio.create_task(
                        self._background_refresh(key, loader, ttl)
                    )

                return value
            except (json.JSONDecodeError, KeyError):
                pass

        # Cache miss
        if not loader:
            return None

        # Single-flight: check if already loading
        if key in self._inflight:
            return await self._inflight[key]

        # Try to acquire lock
        lock_key = self._lock_key(key)
        lock_value = str(uuid.uuid4())

        if await self.backend.setnx(lock_key, lock_value, ttl=30):
            # We got the lock, load the value
            future: asyncio.Future = asyncio.Future()
            self._inflight[key] = future

            try:
                value = await loader()
                await self.set(key, value, ttl)
                future.set_result(value)
                return value
            except Exception as e:
                future.set_exception(e)
                raise
            finally:
                del self._inflight[key]
                # Release lock
                current = await self.backend.get(lock_key)
                if current == lock_value:
                    await self.backend.delete(lock_key)
        else:
            # Someone else is loading, wait for them
            for _ in range(100):  # Max 10 seconds wait
                await asyncio.sleep(0.1)

                if key in self._inflight:
                    return await self._inflight[key]

                data = await self.backend.get(cache_key)
                if data:
                    entry = json.loads(data)
                    return self.deserializer(entry["value"])

            # Timeout, try to load ourselves
            return await loader()

    async def set(
        self,
        key: str,
        value: T,
        ttl: Optional[float] = None,
    ) -> None:
        """Set value in cache."""
        cache_key = self._cache_key(key)
        actual_ttl = ttl or self.default_ttl

        entry = {
            "value": self.serializer(value),
            "created_at": time.time(),
            "ttl": actual_ttl,
        }

        await self.backend.set(
            cache_key,
            json.dumps(entry),
            ttl=actual_ttl,
        )

    async def delete(self, key: str) -> bool:
        """Delete from cache."""
        return await self.backend.delete(self._cache_key(key))

    async def clear(self) -> None:
        """Clear all entries (pattern delete)."""
        # Note: This requires SCAN in production
        logger.warning("Cache clear not fully implemented for distributed cache")

    def _should_early_refresh(self, entry: Dict) -> bool:
        """Check if should probabilistically refresh early."""
        import random

        created_at = entry.get("created_at", 0)
        ttl = entry.get("ttl", self.default_ttl)
        age = time.time() - created_at

        # Check if in early expiry window
        if age < ttl - self.early_expiry_window:
            return False

        # Probabilistic refresh
        return random.random() < self.early_expiry_probability

    async def _background_refresh(
        self,
        key: str,
        loader: Optional[Callable[[], Awaitable[T]]],
        ttl: Optional[float],
    ) -> None:
        """Refresh cache entry in background."""
        if not loader:
            return

        try:
            value = await loader()
            await self.set(key, value, ttl)
            logger.debug(f"Background cache refresh for {key}")
        except Exception as e:
            logger.warning(f"Background cache refresh failed: {e}")


# ============================================
# Factory Functions
# ============================================

_global_backend: Optional[DistributedBackend] = None


async def init_distributed_backend(
    backend_type: str = "memory",
    **kwargs,
) -> DistributedBackend:
    """
    Initialize global distributed backend.

    Args:
        backend_type: "redis" or "memory"
        **kwargs: Backend-specific configuration

    Returns:
        Initialized backend
    """
    global _global_backend

    if backend_type == "redis":
        backend = RedisBackend(**kwargs)
        await backend.connect()
    else:
        backend = InMemoryBackend(**kwargs)

    _global_backend = backend
    logger.info(f"Initialized distributed backend: {backend_type}")
    return backend


def get_distributed_backend() -> Optional[DistributedBackend]:
    """Get global distributed backend."""
    return _global_backend


async def close_distributed_backend() -> None:
    """Close global distributed backend."""
    global _global_backend

    if _global_backend and isinstance(_global_backend, RedisBackend):
        await _global_backend.close()

    _global_backend = None
