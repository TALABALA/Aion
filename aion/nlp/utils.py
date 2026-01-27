"""
AION NLP Utilities - Shared infrastructure for the NLP programming system.

Provides:
- Robust JSON parsing from LLM responses
- TTL-based LRU cache for LLM results
- Circuit breaker for external service calls
- Async lock registry for concurrency safety
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections import OrderedDict
from enum import Enum
from typing import Any, Callable, Dict, Hashable, Optional, Tuple, TypeVar

import structlog

logger = structlog.get_logger(__name__)

T = TypeVar("T")


# =============================================================================
# Robust JSON Parsing
# =============================================================================


def parse_json_safe(text: str) -> Dict[str, Any]:
    """
    Safely parse JSON from LLM responses.

    Handles:
    - Raw JSON strings
    - JSON inside markdown code blocks
    - JSON embedded in prose text
    - Malformed JSON with trailing commas
    """
    if not text or not text.strip():
        return {}

    # Try direct parse first
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass

    # Try extracting from markdown code block
    json_block = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if json_block:
        try:
            return json.loads(json_block.group(1))
        except (json.JSONDecodeError, ValueError):
            pass

    # Try to find any JSON object in the text
    brace_match = re.search(r"\{[\s\S]*\}", text)
    if brace_match:
        candidate = brace_match.group()
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            # Try fixing trailing commas
            cleaned = re.sub(r",\s*([}\]])", r"\1", candidate)
            try:
                return json.loads(cleaned)
            except (json.JSONDecodeError, ValueError):
                pass

    return {}


# =============================================================================
# TTL-based LRU Cache
# =============================================================================


class TTLCache:
    """
    Thread-safe TTL-based LRU cache.

    Entries expire after `ttl_seconds` and the cache evicts the least
    recently used entry when `max_size` is exceeded.
    """

    def __init__(self, max_size: int = 256, ttl_seconds: float = 300.0):
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._data: OrderedDict[str, Tuple[Any, float]] = OrderedDict()
        self._lock = asyncio.Lock()
        self._hits = 0
        self._misses = 0

    async def get(self, key: str) -> Optional[Any]:
        """Get a value from cache. Returns None on miss or expiry."""
        async with self._lock:
            if key not in self._data:
                self._misses += 1
                return None

            value, timestamp = self._data[key]
            if time.monotonic() - timestamp > self._ttl:
                # Expired
                del self._data[key]
                self._misses += 1
                return None

            # Move to end (most recently used)
            self._data.move_to_end(key)
            self._hits += 1
            return value

    async def put(self, key: str, value: Any) -> None:
        """Put a value in cache."""
        async with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
            self._data[key] = (value, time.monotonic())

            # Evict if over capacity
            while len(self._data) > self._max_size:
                self._data.popitem(last=False)

    async def invalidate(self, key: str) -> None:
        """Remove a specific key."""
        async with self._lock:
            self._data.pop(key, None)

    async def clear(self) -> None:
        """Clear all entries."""
        async with self._lock:
            self._data.clear()

    @property
    def stats(self) -> Dict[str, Any]:
        total = self._hits + self._misses
        return {
            "size": len(self._data),
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0.0,
        }


# =============================================================================
# Circuit Breaker
# =============================================================================


class CircuitState(Enum):
    CLOSED = "closed"       # Normal operation
    OPEN = "open"           # Failing, reject calls
    HALF_OPEN = "half_open" # Testing recovery


class CircuitBreaker:
    """
    Circuit breaker pattern for external service calls.

    Prevents cascading failures by fast-failing when an external
    service (e.g., LLM) is repeatedly failing.

    States:
    - CLOSED: Normal operation, calls pass through
    - OPEN: Service is down, calls fail immediately
    - HALF_OPEN: Testing if service has recovered
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 1,
    ):
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._half_open_max_calls = half_open_max_calls

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time = 0.0
        self._half_open_calls = 0
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        if self._state == CircuitState.OPEN:
            if time.monotonic() - self._last_failure_time > self._recovery_timeout:
                return CircuitState.HALF_OPEN
        return self._state

    async def call(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        """Execute a function through the circuit breaker."""
        async with self._lock:
            current_state = self.state

            if current_state == CircuitState.OPEN:
                raise CircuitBreakerOpenError(
                    f"Circuit breaker is OPEN (failed {self._failure_count} times). "
                    f"Recovery in {self._recovery_timeout - (time.monotonic() - self._last_failure_time):.1f}s"
                )

            if current_state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self._half_open_max_calls:
                    raise CircuitBreakerOpenError("Circuit breaker is HALF_OPEN, max test calls reached")
                self._half_open_calls += 1

        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result
        except Exception as e:
            await self._on_failure()
            raise

    async def _on_success(self) -> None:
        async with self._lock:
            self._failure_count = 0
            self._half_open_calls = 0
            if self._state != CircuitState.CLOSED:
                logger.info("Circuit breaker recovered", previous_state=self._state.value)
            self._state = CircuitState.CLOSED

    async def _on_failure(self) -> None:
        async with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()

            if self._failure_count >= self._failure_threshold:
                self._state = CircuitState.OPEN
                logger.warning(
                    "Circuit breaker opened",
                    failure_count=self._failure_count,
                    threshold=self._failure_threshold,
                )

    def reset(self) -> None:
        """Manually reset the circuit breaker."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._half_open_calls = 0

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "state": self.state.value,
            "failure_count": self._failure_count,
            "failure_threshold": self._failure_threshold,
        }


class CircuitBreakerOpenError(Exception):
    """Raised when a call is rejected because the circuit breaker is open."""
    pass


# =============================================================================
# Bounded Collection
# =============================================================================


class BoundedList:
    """A list with a maximum size that evicts oldest entries."""

    def __init__(self, max_size: int = 10000):
        self._data: list = []
        self._max_size = max_size

    def append(self, item: Any) -> None:
        self._data.append(item)
        if len(self._data) > self._max_size:
            # Remove oldest 10% to avoid frequent evictions
            trim_count = max(1, self._max_size // 10)
            self._data = self._data[trim_count:]

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    def __getitem__(self, index):
        return self._data[index]
