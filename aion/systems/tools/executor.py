"""
AION Tool Executor

Handles tool execution with:
- Async execution
- Timeout management
- Error handling
- Result caching
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional

import structlog

from aion.systems.tools.registry import Tool, ToolRegistry

logger = structlog.get_logger(__name__)


@dataclass
class ExecutionResult:
    """Result of a tool execution."""
    tool_name: str
    success: bool
    result: Any
    error: Optional[str] = None
    latency_ms: float = 0.0
    cached: bool = False
    execution_id: str = ""
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tool_name": self.tool_name,
            "success": self.success,
            "result": self.result,
            "error": self.error,
            "latency_ms": self.latency_ms,
            "cached": self.cached,
            "execution_id": self.execution_id,
        }


@dataclass
class CacheEntry:
    """A cached tool result."""
    result: ExecutionResult
    created_at: datetime
    expires_at: datetime
    hit_count: int = 0


class ResultCache:
    """LRU cache for tool results."""

    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: dict[str, CacheEntry] = {}
        self._access_order: list[str] = []

    def _make_key(self, tool_name: str, params: dict[str, Any]) -> str:
        """Create cache key from tool name and params."""
        params_str = json.dumps(params, sort_keys=True, default=str)
        key_str = f"{tool_name}:{params_str}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(self, tool_name: str, params: dict[str, Any]) -> Optional[ExecutionResult]:
        """Get a cached result."""
        key = self._make_key(tool_name, params)

        if key not in self._cache:
            return None

        entry = self._cache[key]

        # Check expiration
        if datetime.now() > entry.expires_at:
            del self._cache[key]
            if key in self._access_order:
                self._access_order.remove(key)
            return None

        # Update access order
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)
        entry.hit_count += 1

        # Mark as cached
        result = entry.result
        result.cached = True

        return result

    def set(
        self,
        tool_name: str,
        params: dict[str, Any],
        result: ExecutionResult,
        ttl: Optional[int] = None,
    ) -> None:
        """Cache a result."""
        key = self._make_key(tool_name, params)
        ttl = ttl or self.default_ttl

        # Evict if at capacity
        while len(self._cache) >= self.max_size and self._access_order:
            oldest = self._access_order.pop(0)
            self._cache.pop(oldest, None)

        self._cache[key] = CacheEntry(
            result=result,
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(seconds=ttl),
        )
        self._access_order.append(key)

    def invalidate(self, tool_name: str) -> int:
        """Invalidate all cached results for a tool."""
        to_remove = [
            key for key in self._cache
            if self._cache[key].result.tool_name == tool_name
        ]

        for key in to_remove:
            del self._cache[key]
            if key in self._access_order:
                self._access_order.remove(key)

        return len(to_remove)

    def clear(self) -> None:
        """Clear all cached results."""
        self._cache.clear()
        self._access_order.clear()


class ToolExecutor:
    """
    Executor for running tools.

    Features:
    - Async execution with timeout
    - Result caching
    - Retry logic
    - Parallel execution
    """

    def __init__(
        self,
        registry: ToolRegistry,
        enable_cache: bool = True,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        self.registry = registry
        self.enable_cache = enable_cache
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self._cache = ResultCache() if enable_cache else None
        self._execution_count = 0

    async def execute(
        self,
        tool_name: str,
        params: dict[str, Any],
        timeout: Optional[float] = None,
        use_cache: bool = True,
        context: Optional[dict[str, Any]] = None,
    ) -> ExecutionResult:
        """
        Execute a single tool.

        Args:
            tool_name: Name of the tool
            params: Tool parameters
            timeout: Execution timeout (overrides tool default)
            use_cache: Whether to use cached results
            context: Execution context

        Returns:
            ExecutionResult
        """
        # Get tool
        tool = self.registry.get(tool_name)
        if tool is None:
            return ExecutionResult(
                tool_name=tool_name,
                success=False,
                result=None,
                error=f"Tool not found: {tool_name}",
            )

        # Check if enabled
        if not self.registry.is_enabled(tool_name):
            return ExecutionResult(
                tool_name=tool_name,
                success=False,
                result=None,
                error=f"Tool is disabled: {tool_name}",
            )

        # Check cache
        if self._cache and use_cache:
            cached = self._cache.get(tool_name, params)
            if cached:
                logger.debug("Cache hit", tool=tool_name)
                return cached

        # Validate parameters
        is_valid, errors = tool.validate_params(params)
        if not is_valid:
            return ExecutionResult(
                tool_name=tool_name,
                success=False,
                result=None,
                error=f"Parameter validation failed: {errors}",
            )

        # Wait for rate limit
        await self.registry.wait_for_rate_limit(tool_name)

        # Execute with retry
        result = await self._execute_with_retry(
            tool, params, timeout or tool.timeout, context
        )

        # Cache successful results
        if result.success and self._cache and use_cache:
            self._cache.set(tool_name, params, result)

        return result

    async def _execute_with_retry(
        self,
        tool: Tool,
        params: dict[str, Any],
        timeout: float,
        context: Optional[dict[str, Any]],
    ) -> ExecutionResult:
        """Execute a tool with retry logic."""
        last_error = None

        for attempt in range(self.max_retries):
            try:
                result = await self._execute_once(tool, params, timeout, context)
                if result.success:
                    return result
                last_error = result.error
            except Exception as e:
                last_error = str(e)

            if attempt < self.max_retries - 1:
                delay = self.retry_delay * (2 ** attempt)
                logger.warning(
                    "Tool execution failed, retrying",
                    tool=tool.name,
                    attempt=attempt + 1,
                    delay=delay,
                )
                await asyncio.sleep(delay)

        return ExecutionResult(
            tool_name=tool.name,
            success=False,
            result=None,
            error=f"Failed after {self.max_retries} attempts: {last_error}",
        )

    async def _execute_once(
        self,
        tool: Tool,
        params: dict[str, Any],
        timeout: float,
        context: Optional[dict[str, Any]],
    ) -> ExecutionResult:
        """Execute a tool once."""
        self._execution_count += 1
        execution_id = f"exec_{self._execution_count}"
        started_at = datetime.now()
        start_time = time.monotonic()

        try:
            # Execute handler
            if asyncio.iscoroutinefunction(tool.handler):
                if context:
                    result_value = await asyncio.wait_for(
                        tool.handler(params, context),
                        timeout=timeout,
                    )
                else:
                    result_value = await asyncio.wait_for(
                        tool.handler(params),
                        timeout=timeout,
                    )
            else:
                # Wrap sync function
                loop = asyncio.get_event_loop()
                if context:
                    result_value = await asyncio.wait_for(
                        loop.run_in_executor(None, lambda: tool.handler(params, context)),
                        timeout=timeout,
                    )
                else:
                    result_value = await asyncio.wait_for(
                        loop.run_in_executor(None, lambda: tool.handler(params)),
                        timeout=timeout,
                    )

            latency_ms = (time.monotonic() - start_time) * 1000

            # Record stats
            self.registry.record_call(tool.name, True, latency_ms)

            return ExecutionResult(
                tool_name=tool.name,
                success=True,
                result=result_value,
                latency_ms=latency_ms,
                execution_id=execution_id,
                started_at=started_at,
                completed_at=datetime.now(),
            )

        except asyncio.TimeoutError:
            latency_ms = (time.monotonic() - start_time) * 1000
            error = f"Timeout after {timeout}s"
            self.registry.record_call(tool.name, False, latency_ms, error)

            return ExecutionResult(
                tool_name=tool.name,
                success=False,
                result=None,
                error=error,
                latency_ms=latency_ms,
                execution_id=execution_id,
                started_at=started_at,
                completed_at=datetime.now(),
            )

        except Exception as e:
            latency_ms = (time.monotonic() - start_time) * 1000
            error = str(e)
            self.registry.record_call(tool.name, False, latency_ms, error)

            return ExecutionResult(
                tool_name=tool.name,
                success=False,
                result=None,
                error=error,
                latency_ms=latency_ms,
                execution_id=execution_id,
                started_at=started_at,
                completed_at=datetime.now(),
            )

    async def execute_many(
        self,
        calls: list[tuple[str, dict[str, Any]]],
        max_parallel: int = 10,
    ) -> list[ExecutionResult]:
        """
        Execute multiple tools, potentially in parallel.

        Args:
            calls: List of (tool_name, params) tuples
            max_parallel: Maximum parallel executions

        Returns:
            List of ExecutionResult
        """
        semaphore = asyncio.Semaphore(max_parallel)

        async def execute_with_semaphore(
            tool_name: str, params: dict[str, Any]
        ) -> ExecutionResult:
            async with semaphore:
                return await self.execute(tool_name, params)

        tasks = [
            execute_with_semaphore(tool_name, params)
            for tool_name, params in calls
        ]

        return await asyncio.gather(*tasks)

    def invalidate_cache(self, tool_name: Optional[str] = None) -> int:
        """Invalidate cached results."""
        if self._cache is None:
            return 0

        if tool_name:
            return self._cache.invalidate(tool_name)
        else:
            count = len(self._cache._cache)
            self._cache.clear()
            return count
