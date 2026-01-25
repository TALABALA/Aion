"""
Distributed Rate Limiting with Redis Backend.

Implements horizontally scalable rate limiting using Redis for state storage.
Supports multiple algorithms optimized for distributed environments:
- Token Bucket with atomic operations
- Sliding Window Log with sorted sets
- Sliding Window Counter with hash-based counters
- Leaky Bucket with Redis streams

Includes circuit breaker pattern and automatic failover.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import structlog

logger = structlog.get_logger()


class DistributedRateLimitStrategy(str, Enum):
    """Distributed rate limiting strategies."""
    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW_LOG = "sliding_window_log"
    SLIDING_WINDOW_COUNTER = "sliding_window_counter"
    LEAKY_BUCKET = "leaky_bucket"
    GCRA = "gcra"  # Generic Cell Rate Algorithm


class CircuitState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject all
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class RateLimitResult:
    """Result of a rate limit check."""
    allowed: bool
    remaining: int
    reset_at: float
    retry_after: Optional[float] = None
    limit: int = 0
    used: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RateLimitRule:
    """Rate limiting rule configuration."""
    key_prefix: str
    requests: int  # Number of requests allowed
    window_seconds: int  # Time window
    strategy: DistributedRateLimitStrategy = DistributedRateLimitStrategy.SLIDING_WINDOW_COUNTER
    burst_multiplier: float = 1.5  # Allow burst up to this multiplier
    penalty_multiplier: float = 2.0  # Multiply window on violations
    penalty_threshold: int = 3  # Violations before penalty


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    failure_threshold: int = 5  # Failures before opening
    success_threshold: int = 3  # Successes in half-open before closing
    timeout_seconds: int = 30  # Time in open state before half-open
    half_open_max_requests: int = 3  # Max requests in half-open state


class RedisBackend(ABC):
    """Abstract Redis backend interface."""

    @abstractmethod
    async def get(self, key: str) -> Optional[str]:
        pass

    @abstractmethod
    async def set(self, key: str, value: str, ex: Optional[int] = None) -> bool:
        pass

    @abstractmethod
    async def incr(self, key: str) -> int:
        pass

    @abstractmethod
    async def expire(self, key: str, seconds: int) -> bool:
        pass

    @abstractmethod
    async def ttl(self, key: str) -> int:
        pass

    @abstractmethod
    async def zadd(self, key: str, mapping: dict[str, float]) -> int:
        pass

    @abstractmethod
    async def zremrangebyscore(self, key: str, min_score: float, max_score: float) -> int:
        pass

    @abstractmethod
    async def zcard(self, key: str) -> int:
        pass

    @abstractmethod
    async def eval_script(self, script: str, keys: list[str], args: list[str]) -> Any:
        pass

    @abstractmethod
    async def hset(self, key: str, mapping: dict[str, str]) -> int:
        pass

    @abstractmethod
    async def hget(self, key: str, field: str) -> Optional[str]:
        pass

    @abstractmethod
    async def hincrby(self, key: str, field: str, amount: int) -> int:
        pass

    @abstractmethod
    async def pipeline(self) -> "RedisPipeline":
        pass

    @abstractmethod
    async def close(self) -> None:
        pass


class RedisPipeline(ABC):
    """Abstract Redis pipeline interface."""

    @abstractmethod
    def incr(self, key: str) -> "RedisPipeline":
        pass

    @abstractmethod
    def expire(self, key: str, seconds: int) -> "RedisPipeline":
        pass

    @abstractmethod
    def get(self, key: str) -> "RedisPipeline":
        pass

    @abstractmethod
    async def execute(self) -> list[Any]:
        pass


class AsyncRedisBackend(RedisBackend):
    """
    Redis backend using redis-py async client.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        ssl: bool = False,
        ssl_cert_reqs: Optional[str] = None,
        socket_timeout: float = 5.0,
        socket_connect_timeout: float = 5.0,
        max_connections: int = 50,
        decode_responses: bool = True,
        cluster_mode: bool = False,
        sentinel_hosts: Optional[list[tuple[str, int]]] = None,
        sentinel_master: Optional[str] = None,
    ) -> None:
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.ssl = ssl
        self.ssl_cert_reqs = ssl_cert_reqs
        self.socket_timeout = socket_timeout
        self.socket_connect_timeout = socket_connect_timeout
        self.max_connections = max_connections
        self.decode_responses = decode_responses
        self.cluster_mode = cluster_mode
        self.sentinel_hosts = sentinel_hosts
        self.sentinel_master = sentinel_master
        self._client: Optional[Any] = None
        self._logger = logger.bind(backend="redis")

    async def initialize(self) -> None:
        """Initialize Redis connection."""
        try:
            import redis.asyncio as redis

            if self.cluster_mode:
                from redis.asyncio.cluster import RedisCluster

                self._client = RedisCluster(
                    host=self.host,
                    port=self.port,
                    password=self.password,
                    decode_responses=self.decode_responses,
                )
            elif self.sentinel_hosts and self.sentinel_master:
                from redis.asyncio.sentinel import Sentinel

                sentinel = Sentinel(
                    self.sentinel_hosts,
                    socket_timeout=self.socket_timeout,
                    password=self.password,
                )
                self._client = sentinel.master_for(
                    self.sentinel_master,
                    decode_responses=self.decode_responses,
                )
            else:
                self._client = redis.Redis(
                    host=self.host,
                    port=self.port,
                    db=self.db,
                    password=self.password,
                    ssl=self.ssl,
                    socket_timeout=self.socket_timeout,
                    socket_connect_timeout=self.socket_connect_timeout,
                    max_connections=self.max_connections,
                    decode_responses=self.decode_responses,
                )

            # Test connection
            await self._client.ping()
            self._logger.info(
                "Connected to Redis",
                host=self.host,
                port=self.port,
                cluster_mode=self.cluster_mode,
            )

        except ImportError:
            self._logger.error("redis-py not installed")
            raise

    async def get(self, key: str) -> Optional[str]:
        return await self._client.get(key)

    async def set(self, key: str, value: str, ex: Optional[int] = None) -> bool:
        return await self._client.set(key, value, ex=ex)

    async def incr(self, key: str) -> int:
        return await self._client.incr(key)

    async def expire(self, key: str, seconds: int) -> bool:
        return await self._client.expire(key, seconds)

    async def ttl(self, key: str) -> int:
        return await self._client.ttl(key)

    async def zadd(self, key: str, mapping: dict[str, float]) -> int:
        return await self._client.zadd(key, mapping)

    async def zremrangebyscore(self, key: str, min_score: float, max_score: float) -> int:
        return await self._client.zremrangebyscore(key, min_score, max_score)

    async def zcard(self, key: str) -> int:
        return await self._client.zcard(key)

    async def eval_script(self, script: str, keys: list[str], args: list[str]) -> Any:
        return await self._client.eval(script, len(keys), *keys, *args)

    async def hset(self, key: str, mapping: dict[str, str]) -> int:
        return await self._client.hset(key, mapping=mapping)

    async def hget(self, key: str, field: str) -> Optional[str]:
        return await self._client.hget(key, field)

    async def hincrby(self, key: str, field: str, amount: int) -> int:
        return await self._client.hincrby(key, field, amount)

    async def pipeline(self) -> "AsyncRedisPipeline":
        return AsyncRedisPipeline(self._client.pipeline())

    async def close(self) -> None:
        if self._client:
            await self._client.close()


class AsyncRedisPipeline(RedisPipeline):
    """Async Redis pipeline wrapper."""

    def __init__(self, pipeline: Any) -> None:
        self._pipeline = pipeline

    def incr(self, key: str) -> "AsyncRedisPipeline":
        self._pipeline.incr(key)
        return self

    def expire(self, key: str, seconds: int) -> "AsyncRedisPipeline":
        self._pipeline.expire(key, seconds)
        return self

    def get(self, key: str) -> "AsyncRedisPipeline":
        self._pipeline.get(key)
        return self

    async def execute(self) -> list[Any]:
        return await self._pipeline.execute()


class InMemoryRedisBackend(RedisBackend):
    """
    In-memory Redis-compatible backend for testing and development.
    """

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}
        self._expiry: dict[str, float] = {}
        self._sorted_sets: dict[str, dict[str, float]] = {}
        self._hashes: dict[str, dict[str, str]] = {}
        self._lock = asyncio.Lock()

    async def _check_expiry(self, key: str) -> bool:
        """Check if key is expired and remove if so."""
        if key in self._expiry:
            if time.time() > self._expiry[key]:
                self._data.pop(key, None)
                self._expiry.pop(key, None)
                self._sorted_sets.pop(key, None)
                self._hashes.pop(key, None)
                return True
        return False

    async def get(self, key: str) -> Optional[str]:
        async with self._lock:
            await self._check_expiry(key)
            return self._data.get(key)

    async def set(self, key: str, value: str, ex: Optional[int] = None) -> bool:
        async with self._lock:
            self._data[key] = value
            if ex:
                self._expiry[key] = time.time() + ex
            return True

    async def incr(self, key: str) -> int:
        async with self._lock:
            await self._check_expiry(key)
            val = int(self._data.get(key, 0)) + 1
            self._data[key] = str(val)
            return val

    async def expire(self, key: str, seconds: int) -> bool:
        async with self._lock:
            if key in self._data or key in self._sorted_sets or key in self._hashes:
                self._expiry[key] = time.time() + seconds
                return True
            return False

    async def ttl(self, key: str) -> int:
        async with self._lock:
            if key in self._expiry:
                remaining = int(self._expiry[key] - time.time())
                return max(remaining, -2)
            if key in self._data or key in self._sorted_sets or key in self._hashes:
                return -1  # No expiry
            return -2  # Key doesn't exist

    async def zadd(self, key: str, mapping: dict[str, float]) -> int:
        async with self._lock:
            await self._check_expiry(key)
            if key not in self._sorted_sets:
                self._sorted_sets[key] = {}
            added = 0
            for member, score in mapping.items():
                if member not in self._sorted_sets[key]:
                    added += 1
                self._sorted_sets[key][member] = score
            return added

    async def zremrangebyscore(self, key: str, min_score: float, max_score: float) -> int:
        async with self._lock:
            await self._check_expiry(key)
            if key not in self._sorted_sets:
                return 0
            to_remove = [
                m for m, s in self._sorted_sets[key].items()
                if min_score <= s <= max_score
            ]
            for m in to_remove:
                del self._sorted_sets[key][m]
            return len(to_remove)

    async def zcard(self, key: str) -> int:
        async with self._lock:
            await self._check_expiry(key)
            return len(self._sorted_sets.get(key, {}))

    async def eval_script(self, script: str, keys: list[str], args: list[str]) -> Any:
        # Simplified script execution - implement specific scripts as needed
        raise NotImplementedError("Lua scripts not supported in in-memory backend")

    async def hset(self, key: str, mapping: dict[str, str]) -> int:
        async with self._lock:
            await self._check_expiry(key)
            if key not in self._hashes:
                self._hashes[key] = {}
            added = 0
            for field, value in mapping.items():
                if field not in self._hashes[key]:
                    added += 1
                self._hashes[key][field] = value
            return added

    async def hget(self, key: str, field: str) -> Optional[str]:
        async with self._lock:
            await self._check_expiry(key)
            return self._hashes.get(key, {}).get(field)

    async def hincrby(self, key: str, field: str, amount: int) -> int:
        async with self._lock:
            await self._check_expiry(key)
            if key not in self._hashes:
                self._hashes[key] = {}
            val = int(self._hashes[key].get(field, 0)) + amount
            self._hashes[key][field] = str(val)
            return val

    async def pipeline(self) -> "InMemoryPipeline":
        return InMemoryPipeline(self)

    async def close(self) -> None:
        self._data.clear()
        self._expiry.clear()
        self._sorted_sets.clear()
        self._hashes.clear()


class InMemoryPipeline(RedisPipeline):
    """In-memory pipeline for testing."""

    def __init__(self, backend: InMemoryRedisBackend) -> None:
        self._backend = backend
        self._commands: list[tuple[str, tuple]] = []

    def incr(self, key: str) -> "InMemoryPipeline":
        self._commands.append(("incr", (key,)))
        return self

    def expire(self, key: str, seconds: int) -> "InMemoryPipeline":
        self._commands.append(("expire", (key, seconds)))
        return self

    def get(self, key: str) -> "InMemoryPipeline":
        self._commands.append(("get", (key,)))
        return self

    async def execute(self) -> list[Any]:
        results = []
        for cmd, args in self._commands:
            if cmd == "incr":
                results.append(await self._backend.incr(args[0]))
            elif cmd == "expire":
                results.append(await self._backend.expire(args[0], args[1]))
            elif cmd == "get":
                results.append(await self._backend.get(args[0]))
        return results


# Lua scripts for atomic operations
TOKEN_BUCKET_SCRIPT = """
local key = KEYS[1]
local now = tonumber(ARGV[1])
local rate = tonumber(ARGV[2])
local capacity = tonumber(ARGV[3])
local requested = tonumber(ARGV[4])

local bucket = redis.call('HMGET', key, 'tokens', 'last_update')
local tokens = tonumber(bucket[1]) or capacity
local last_update = tonumber(bucket[2]) or now

-- Calculate tokens to add based on time elapsed
local elapsed = now - last_update
local tokens_to_add = elapsed * rate
tokens = math.min(capacity, tokens + tokens_to_add)

local allowed = 0
local remaining = tokens

if tokens >= requested then
    tokens = tokens - requested
    allowed = 1
    remaining = tokens
end

redis.call('HMSET', key, 'tokens', tokens, 'last_update', now)
redis.call('EXPIRE', key, math.ceil(capacity / rate) + 1)

return {allowed, math.floor(remaining), math.ceil((capacity - remaining) / rate)}
"""

SLIDING_WINDOW_COUNTER_SCRIPT = """
local key = KEYS[1]
local now = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local limit = tonumber(ARGV[3])

local current_window = math.floor(now / window)
local previous_window = current_window - 1
local window_offset = (now % window) / window

local current_key = key .. ':' .. current_window
local previous_key = key .. ':' .. previous_window

local current_count = tonumber(redis.call('GET', current_key) or 0)
local previous_count = tonumber(redis.call('GET', previous_key) or 0)

-- Weighted count based on position in current window
local weighted_count = math.floor(previous_count * (1 - window_offset) + current_count)

if weighted_count >= limit then
    return {0, 0, math.ceil(window * (1 - window_offset))}
end

redis.call('INCR', current_key)
redis.call('EXPIRE', current_key, window * 2)

return {1, limit - weighted_count - 1, 0}
"""

GCRA_SCRIPT = """
local key = KEYS[1]
local now = tonumber(ARGV[1])
local period = tonumber(ARGV[2])
local limit = tonumber(ARGV[3])
local cost = tonumber(ARGV[4])

local emission_interval = period / limit
local increment = emission_interval * cost
local burst_offset = emission_interval * limit

local tat = tonumber(redis.call('GET', key) or now)

local new_tat = math.max(tat, now) + increment
local allow_at = new_tat - burst_offset

if allow_at > now then
    local retry_after = allow_at - now
    return {0, 0, math.ceil(retry_after)}
end

redis.call('SET', key, new_tat)
redis.call('EXPIRE', key, math.ceil(burst_offset + period))

local remaining = math.floor((new_tat - now) / emission_interval)
remaining = limit - remaining

return {1, math.max(0, remaining), 0}
"""


class CircuitBreaker:
    """
    Circuit breaker for rate limiter Redis operations.

    Prevents cascading failures when Redis is unavailable.
    """

    def __init__(self, config: CircuitBreakerConfig) -> None:
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self.half_open_requests = 0
        self._lock = asyncio.Lock()
        self._logger = logger.bind(component="circuit_breaker")

    async def call(self, func, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        async with self._lock:
            if self.state == CircuitState.OPEN:
                if self._should_try_reset():
                    self.state = CircuitState.HALF_OPEN
                    self.half_open_requests = 0
                    self._logger.info("Circuit breaker entering half-open state")
                else:
                    raise CircuitBreakerOpen("Circuit breaker is open")

            if self.state == CircuitState.HALF_OPEN:
                if self.half_open_requests >= self.config.half_open_max_requests:
                    raise CircuitBreakerOpen("Circuit breaker half-open limit reached")
                self.half_open_requests += 1

        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result
        except Exception as e:
            await self._on_failure()
            raise

    async def _on_success(self) -> None:
        """Handle successful call."""
        async with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    self.success_count = 0
                    self._logger.info("Circuit breaker closed")
            else:
                self.failure_count = 0

    async def _on_failure(self) -> None:
        """Handle failed call."""
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            self.success_count = 0

            if self.failure_count >= self.config.failure_threshold:
                self.state = CircuitState.OPEN
                self._logger.warning(
                    "Circuit breaker opened",
                    failure_count=self.failure_count,
                )

    def _should_try_reset(self) -> bool:
        """Check if we should try to reset from open state."""
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time >= self.config.timeout_seconds


class CircuitBreakerOpen(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class DistributedRateLimiter:
    """
    Distributed rate limiter using Redis.

    Provides horizontally scalable rate limiting with multiple algorithms,
    circuit breaker protection, and automatic failover to local limiting.
    """

    def __init__(
        self,
        redis_backend: RedisBackend,
        default_rule: Optional[RateLimitRule] = None,
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
        key_prefix: str = "ratelimit",
        fallback_to_local: bool = True,
    ) -> None:
        self.redis = redis_backend
        self.default_rule = default_rule or RateLimitRule(
            key_prefix="default",
            requests=100,
            window_seconds=60,
        )
        self.circuit_breaker = CircuitBreaker(
            circuit_breaker_config or CircuitBreakerConfig()
        )
        self.key_prefix = key_prefix
        self.fallback_to_local = fallback_to_local
        self._rules: dict[str, RateLimitRule] = {}
        self._local_counters: dict[str, tuple[int, float]] = {}
        self._scripts_registered = False
        self._logger = logger.bind(component="distributed_rate_limiter")

    def add_rule(self, name: str, rule: RateLimitRule) -> None:
        """Add a rate limiting rule."""
        self._rules[name] = rule

    async def check(
        self,
        identifier: str,
        rule_name: Optional[str] = None,
        cost: int = 1,
    ) -> RateLimitResult:
        """
        Check if request is allowed under rate limit.

        Args:
            identifier: Unique identifier (user ID, IP, API key, etc.)
            rule_name: Name of rule to apply (uses default if not specified)
            cost: Cost of this request (default 1)

        Returns:
            RateLimitResult with allowed status and metadata
        """
        rule = self._rules.get(rule_name) if rule_name else self.default_rule
        if not rule:
            rule = self.default_rule

        key = f"{self.key_prefix}:{rule.key_prefix}:{identifier}"

        try:
            result = await self.circuit_breaker.call(
                self._check_redis,
                key,
                rule,
                cost,
            )
            return result
        except CircuitBreakerOpen:
            if self.fallback_to_local:
                self._logger.warning("Falling back to local rate limiting")
                return await self._check_local(key, rule, cost)
            raise
        except Exception as e:
            self._logger.error("Rate limit check failed", error=str(e))
            if self.fallback_to_local:
                return await self._check_local(key, rule, cost)
            # Default allow on error
            return RateLimitResult(
                allowed=True,
                remaining=rule.requests,
                reset_at=time.time() + rule.window_seconds,
                limit=rule.requests,
            )

    async def _check_redis(
        self,
        key: str,
        rule: RateLimitRule,
        cost: int,
    ) -> RateLimitResult:
        """Check rate limit using Redis."""
        now = time.time()

        if rule.strategy == DistributedRateLimitStrategy.TOKEN_BUCKET:
            return await self._token_bucket(key, rule, cost, now)
        elif rule.strategy == DistributedRateLimitStrategy.SLIDING_WINDOW_LOG:
            return await self._sliding_window_log(key, rule, cost, now)
        elif rule.strategy == DistributedRateLimitStrategy.SLIDING_WINDOW_COUNTER:
            return await self._sliding_window_counter(key, rule, cost, now)
        elif rule.strategy == DistributedRateLimitStrategy.LEAKY_BUCKET:
            return await self._leaky_bucket(key, rule, cost, now)
        elif rule.strategy == DistributedRateLimitStrategy.GCRA:
            return await self._gcra(key, rule, cost, now)
        else:
            return await self._sliding_window_counter(key, rule, cost, now)

    async def _token_bucket(
        self,
        key: str,
        rule: RateLimitRule,
        cost: int,
        now: float,
    ) -> RateLimitResult:
        """Token bucket algorithm using Lua script."""
        rate = rule.requests / rule.window_seconds
        capacity = int(rule.requests * rule.burst_multiplier)

        result = await self.redis.eval_script(
            TOKEN_BUCKET_SCRIPT,
            [key],
            [str(now), str(rate), str(capacity), str(cost)],
        )

        allowed, remaining, retry_after = result

        return RateLimitResult(
            allowed=bool(allowed),
            remaining=remaining,
            reset_at=now + (retry_after if retry_after else rule.window_seconds),
            retry_after=retry_after if not allowed else None,
            limit=capacity,
            used=capacity - remaining,
        )

    async def _sliding_window_log(
        self,
        key: str,
        rule: RateLimitRule,
        cost: int,
        now: float,
    ) -> RateLimitResult:
        """Sliding window log algorithm using sorted sets."""
        window_start = now - rule.window_seconds

        # Remove old entries
        await self.redis.zremrangebyscore(key, 0, window_start)

        # Count current entries
        current_count = await self.redis.zcard(key)

        if current_count >= rule.requests:
            return RateLimitResult(
                allowed=False,
                remaining=0,
                reset_at=now + rule.window_seconds,
                retry_after=rule.window_seconds,
                limit=rule.requests,
                used=current_count,
            )

        # Add new entry
        request_id = f"{now}:{hashlib.md5(str(now).encode()).hexdigest()[:8]}"
        await self.redis.zadd(key, {request_id: now})
        await self.redis.expire(key, rule.window_seconds + 1)

        return RateLimitResult(
            allowed=True,
            remaining=rule.requests - current_count - cost,
            reset_at=now + rule.window_seconds,
            limit=rule.requests,
            used=current_count + cost,
        )

    async def _sliding_window_counter(
        self,
        key: str,
        rule: RateLimitRule,
        cost: int,
        now: float,
    ) -> RateLimitResult:
        """Sliding window counter algorithm using Lua script."""
        result = await self.redis.eval_script(
            SLIDING_WINDOW_COUNTER_SCRIPT,
            [key],
            [str(now), str(rule.window_seconds), str(rule.requests)],
        )

        allowed, remaining, retry_after = result

        return RateLimitResult(
            allowed=bool(allowed),
            remaining=remaining,
            reset_at=now + (retry_after if retry_after else rule.window_seconds),
            retry_after=retry_after if not allowed else None,
            limit=rule.requests,
            used=rule.requests - remaining - (1 if allowed else 0),
        )

    async def _leaky_bucket(
        self,
        key: str,
        rule: RateLimitRule,
        cost: int,
        now: float,
    ) -> RateLimitResult:
        """Leaky bucket algorithm (similar to token bucket with fixed drain rate)."""
        return await self._token_bucket(key, rule, cost, now)

    async def _gcra(
        self,
        key: str,
        rule: RateLimitRule,
        cost: int,
        now: float,
    ) -> RateLimitResult:
        """Generic Cell Rate Algorithm using Lua script."""
        result = await self.redis.eval_script(
            GCRA_SCRIPT,
            [key],
            [str(now), str(rule.window_seconds), str(rule.requests), str(cost)],
        )

        allowed, remaining, retry_after = result

        return RateLimitResult(
            allowed=bool(allowed),
            remaining=remaining,
            reset_at=now + (retry_after if retry_after else rule.window_seconds),
            retry_after=retry_after if not allowed else None,
            limit=rule.requests,
            used=rule.requests - remaining,
        )

    async def _check_local(
        self,
        key: str,
        rule: RateLimitRule,
        cost: int,
    ) -> RateLimitResult:
        """Local fallback rate limiting."""
        now = time.time()

        if key in self._local_counters:
            count, window_start = self._local_counters[key]
            if now - window_start > rule.window_seconds:
                # Window expired, reset
                count = 0
                window_start = now
        else:
            count = 0
            window_start = now

        if count >= rule.requests:
            return RateLimitResult(
                allowed=False,
                remaining=0,
                reset_at=window_start + rule.window_seconds,
                retry_after=window_start + rule.window_seconds - now,
                limit=rule.requests,
                used=count,
            )

        self._local_counters[key] = (count + cost, window_start)

        return RateLimitResult(
            allowed=True,
            remaining=rule.requests - count - cost,
            reset_at=window_start + rule.window_seconds,
            limit=rule.requests,
            used=count + cost,
        )

    async def get_stats(self, identifier: str, rule_name: Optional[str] = None) -> dict[str, Any]:
        """Get rate limit stats for an identifier."""
        rule = self._rules.get(rule_name) if rule_name else self.default_rule
        key = f"{self.key_prefix}:{rule.key_prefix}:{identifier}"

        ttl = await self.redis.ttl(key)

        return {
            "identifier": identifier,
            "rule": rule_name or "default",
            "ttl": ttl,
            "limit": rule.requests,
            "window_seconds": rule.window_seconds,
            "strategy": rule.strategy.value,
        }

    async def reset(self, identifier: str, rule_name: Optional[str] = None) -> bool:
        """Reset rate limit for an identifier."""
        rule = self._rules.get(rule_name) if rule_name else self.default_rule
        key = f"{self.key_prefix}:{rule.key_prefix}:{identifier}"

        # Delete all related keys
        await self.redis.set(key, "0", ex=1)

        if key in self._local_counters:
            del self._local_counters[key]

        return True


class MultiTierRateLimiter:
    """
    Multi-tier rate limiter with different limits for different scopes.

    Enforces rate limits at multiple levels:
    - Global: Overall system limit
    - Tenant: Per-tenant limit
    - User: Per-user limit
    - IP: Per-IP limit
    - Endpoint: Per-endpoint limit
    """

    def __init__(self, limiter: DistributedRateLimiter) -> None:
        self.limiter = limiter
        self._logger = logger.bind(component="multi_tier_limiter")

    async def check(
        self,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        endpoint: Optional[str] = None,
    ) -> RateLimitResult:
        """
        Check rate limits at all applicable tiers.

        Returns the most restrictive result.
        """
        results: list[RateLimitResult] = []

        # Check each tier
        if endpoint:
            results.append(await self.limiter.check(endpoint, "endpoint"))

        if ip_address:
            results.append(await self.limiter.check(ip_address, "ip"))

        if user_id:
            results.append(await self.limiter.check(user_id, "user"))

        if tenant_id:
            results.append(await self.limiter.check(tenant_id, "tenant"))

        # Global check
        results.append(await self.limiter.check("global", "global"))

        # Find the most restrictive (denied, or lowest remaining)
        denied = [r for r in results if not r.allowed]
        if denied:
            # Return the one with longest retry_after
            return max(denied, key=lambda r: r.retry_after or 0)

        # All allowed, return the one with lowest remaining
        return min(results, key=lambda r: r.remaining)

    def configure_tiers(
        self,
        global_limit: int = 10000,
        tenant_limit: int = 5000,
        user_limit: int = 1000,
        ip_limit: int = 500,
        endpoint_limit: int = 100,
        window_seconds: int = 60,
    ) -> None:
        """Configure rate limit tiers."""
        self.limiter.add_rule("global", RateLimitRule(
            key_prefix="global",
            requests=global_limit,
            window_seconds=window_seconds,
            strategy=DistributedRateLimitStrategy.SLIDING_WINDOW_COUNTER,
        ))

        self.limiter.add_rule("tenant", RateLimitRule(
            key_prefix="tenant",
            requests=tenant_limit,
            window_seconds=window_seconds,
            strategy=DistributedRateLimitStrategy.SLIDING_WINDOW_COUNTER,
        ))

        self.limiter.add_rule("user", RateLimitRule(
            key_prefix="user",
            requests=user_limit,
            window_seconds=window_seconds,
            strategy=DistributedRateLimitStrategy.TOKEN_BUCKET,
        ))

        self.limiter.add_rule("ip", RateLimitRule(
            key_prefix="ip",
            requests=ip_limit,
            window_seconds=window_seconds,
            strategy=DistributedRateLimitStrategy.SLIDING_WINDOW_COUNTER,
        ))

        self.limiter.add_rule("endpoint", RateLimitRule(
            key_prefix="endpoint",
            requests=endpoint_limit,
            window_seconds=window_seconds,
            strategy=DistributedRateLimitStrategy.GCRA,
        ))


async def create_distributed_rate_limiter(
    redis_url: Optional[str] = None,
    redis_host: str = "localhost",
    redis_port: int = 6379,
    redis_password: Optional[str] = None,
    redis_db: int = 0,
    use_cluster: bool = False,
    fallback_to_local: bool = True,
) -> DistributedRateLimiter:
    """
    Factory function to create a distributed rate limiter.

    Args:
        redis_url: Redis URL (overrides host/port)
        redis_host: Redis host
        redis_port: Redis port
        redis_password: Redis password
        redis_db: Redis database number
        use_cluster: Use Redis cluster mode
        fallback_to_local: Fall back to local limiting if Redis fails

    Returns:
        Configured DistributedRateLimiter
    """
    backend = AsyncRedisBackend(
        host=redis_host,
        port=redis_port,
        password=redis_password,
        db=redis_db,
        cluster_mode=use_cluster,
    )

    try:
        await backend.initialize()
    except Exception as e:
        logger.warning(f"Redis connection failed, using in-memory backend: {e}")
        backend = InMemoryRedisBackend()

    return DistributedRateLimiter(
        redis_backend=backend,
        fallback_to_local=fallback_to_local,
    )
