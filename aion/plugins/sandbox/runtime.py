"""
AION Sandbox Runtime

Provides sandboxed execution environment for plugins.
"""

from __future__ import annotations

import asyncio
import functools
import resource
import signal
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, TYPE_CHECKING
import threading

import structlog

from aion.plugins.types import (
    PluginPermissions,
    ResourceLimit,
    PermissionLevel,
)
from aion.plugins.sandbox.permissions import PermissionChecker, PermissionViolation

if TYPE_CHECKING:
    from aion.plugins.interfaces.base import BasePlugin

logger = structlog.get_logger(__name__)


@dataclass
class ExecutionContext:
    """Context for sandboxed execution."""

    plugin_id: str
    permissions: PluginPermissions
    start_time: float = field(default_factory=time.time)
    memory_usage: int = 0
    cpu_time: float = 0.0
    is_cancelled: bool = False

    @property
    def elapsed_time(self) -> float:
        return time.time() - self.start_time


@dataclass
class ExecutionResult:
    """Result of sandboxed execution."""

    success: bool
    result: Any = None
    error: Optional[str] = None
    trace: Optional[str] = None
    execution_time_ms: float = 0.0
    memory_used_bytes: int = 0
    permission_violations: List[str] = field(default_factory=list)


class SandboxedPlugin:
    """
    Wrapper that sandboxes plugin method calls.

    Intercepts method calls to enforce permissions and resource limits.
    """

    def __init__(
        self,
        plugin: "BasePlugin",
        permissions: PluginPermissions,
        runtime: "SandboxRuntime",
    ):
        self._plugin = plugin
        self._permissions = permissions
        self._runtime = runtime
        self._checker = PermissionChecker(permissions)

    def __getattr__(self, name: str) -> Any:
        """Intercept attribute access to wrap methods."""
        attr = getattr(self._plugin, name)

        if callable(attr):
            return self._wrap_method(attr, name)

        return attr

    def _wrap_method(self, method: Callable, name: str) -> Callable:
        """Wrap a method with sandbox protection."""

        @functools.wraps(method)
        async def async_wrapper(*args, **kwargs):
            return await self._runtime.execute(
                self._plugin.get_manifest().id,
                method,
                args,
                kwargs,
                self._permissions,
            )

        @functools.wraps(method)
        def sync_wrapper(*args, **kwargs):
            return asyncio.run(async_wrapper(*args, **kwargs))

        if asyncio.iscoroutinefunction(method):
            return async_wrapper
        return sync_wrapper

    @property
    def unwrapped(self) -> "BasePlugin":
        """Get the unwrapped plugin instance."""
        return self._plugin


class SandboxRuntime:
    """
    Sandboxed execution runtime for plugins.

    Features:
    - Resource limiting (CPU, memory, time)
    - Permission enforcement
    - Isolation between plugins
    - Execution monitoring
    - Graceful timeout handling
    """

    def __init__(
        self,
        default_timeout: float = 30.0,
        max_workers: int = 4,
        enable_process_isolation: bool = False,
    ):
        self._default_timeout = default_timeout
        self._max_workers = max_workers
        self._enable_process_isolation = enable_process_isolation

        # Executors
        self._thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        self._process_pool: Optional[ProcessPoolExecutor] = None

        if enable_process_isolation:
            self._process_pool = ProcessPoolExecutor(max_workers=max_workers)

        # Active executions
        self._active_contexts: Dict[str, ExecutionContext] = {}
        self._context_lock = asyncio.Lock()

        # Permission checkers
        self._checkers: Dict[str, PermissionChecker] = {}

        # Stats
        self._stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "timeout_executions": 0,
            "permission_violations": 0,
        }

    async def initialize(self) -> None:
        """Initialize the sandbox runtime."""
        logger.info("Sandbox runtime initialized")

    async def shutdown(self) -> None:
        """Shutdown the sandbox runtime."""
        # Cancel active executions
        async with self._context_lock:
            for ctx in self._active_contexts.values():
                ctx.is_cancelled = True

        # Shutdown executors
        self._thread_pool.shutdown(wait=True)

        if self._process_pool:
            self._process_pool.shutdown(wait=True)

        logger.info("Sandbox runtime shutdown")

    def wrap_plugin(
        self,
        plugin: "BasePlugin",
        permissions: PluginPermissions,
    ) -> SandboxedPlugin:
        """
        Wrap a plugin with sandbox protection.

        Args:
            plugin: Plugin to wrap
            permissions: Plugin permissions

        Returns:
            Sandboxed plugin wrapper
        """
        plugin_id = plugin.get_manifest().id
        self._checkers[plugin_id] = PermissionChecker(permissions)
        return SandboxedPlugin(plugin, permissions, self)

    async def execute(
        self,
        plugin_id: str,
        func: Callable,
        args: tuple,
        kwargs: dict,
        permissions: PluginPermissions,
        timeout: Optional[float] = None,
    ) -> Any:
        """
        Execute a function in the sandbox.

        Args:
            plugin_id: Plugin identifier
            func: Function to execute
            args: Function arguments
            kwargs: Function keyword arguments
            permissions: Plugin permissions
            timeout: Execution timeout

        Returns:
            Function result

        Raises:
            PermissionViolation: If permission check fails
            TimeoutError: If execution times out
            Exception: If execution fails
        """
        timeout = timeout or permissions.resource_limits.max_execution_time_seconds
        timeout = timeout or self._default_timeout

        self._stats["total_executions"] += 1

        # Create execution context
        context = ExecutionContext(
            plugin_id=plugin_id,
            permissions=permissions,
        )

        execution_id = f"{plugin_id}_{time.time()}"

        async with self._context_lock:
            self._active_contexts[execution_id] = context

        try:
            # Execute with timeout
            result = await self._execute_with_limits(
                func, args, kwargs, context, timeout
            )

            self._stats["successful_executions"] += 1
            return result

        except asyncio.TimeoutError:
            self._stats["timeout_executions"] += 1
            logger.warning(f"Plugin {plugin_id} execution timed out")
            raise

        except PermissionViolation as e:
            self._stats["permission_violations"] += 1
            logger.warning(f"Plugin {plugin_id} permission violation: {e}")
            raise

        except Exception as e:
            self._stats["failed_executions"] += 1
            logger.error(f"Plugin {plugin_id} execution failed: {e}")
            raise

        finally:
            async with self._context_lock:
                self._active_contexts.pop(execution_id, None)

    async def _execute_with_limits(
        self,
        func: Callable,
        args: tuple,
        kwargs: dict,
        context: ExecutionContext,
        timeout: float,
    ) -> Any:
        """Execute function with resource limits."""
        # Check if cancelled before starting
        if context.is_cancelled:
            raise asyncio.CancelledError("Execution cancelled")

        # For async functions, execute directly with timeout
        if asyncio.iscoroutinefunction(func):
            return await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=timeout,
            )

        # For sync functions, run in thread pool
        loop = asyncio.get_event_loop()
        return await asyncio.wait_for(
            loop.run_in_executor(
                self._thread_pool,
                functools.partial(func, *args, **kwargs),
            ),
            timeout=timeout,
        )

    @asynccontextmanager
    async def sandbox_context(
        self,
        plugin_id: str,
        permissions: PluginPermissions,
    ):
        """
        Context manager for sandboxed execution.

        Usage:
            async with runtime.sandbox_context(plugin_id, permissions) as ctx:
                # Execute sandboxed code
                result = await some_function()
        """
        context = ExecutionContext(
            plugin_id=plugin_id,
            permissions=permissions,
        )

        execution_id = f"{plugin_id}_{time.time()}"

        async with self._context_lock:
            self._active_contexts[execution_id] = context

        try:
            yield context
        finally:
            async with self._context_lock:
                self._active_contexts.pop(execution_id, None)

    # === Resource Monitoring ===

    def get_resource_usage(self, plugin_id: str) -> Dict[str, Any]:
        """Get resource usage for a plugin."""
        context = next(
            (ctx for ctx in self._active_contexts.values() if ctx.plugin_id == plugin_id),
            None,
        )

        if not context:
            return {"active": False}

        return {
            "active": True,
            "elapsed_time": context.elapsed_time,
            "memory_usage": context.memory_usage,
            "cpu_time": context.cpu_time,
        }

    def cancel_execution(self, plugin_id: str) -> bool:
        """Cancel all executions for a plugin."""
        cancelled = False

        for ctx in self._active_contexts.values():
            if ctx.plugin_id == plugin_id:
                ctx.is_cancelled = True
                cancelled = True

        return cancelled

    # === Permission Checking ===

    def check_permission(
        self,
        plugin_id: str,
        operation: str,
        resource: str,
    ) -> bool:
        """
        Check if plugin has permission for an operation.

        Args:
            plugin_id: Plugin identifier
            operation: Operation type (e.g., "network", "file", "subprocess")
            resource: Resource identifier (e.g., URL, path)

        Returns:
            True if permitted
        """
        checker = self._checkers.get(plugin_id)
        if not checker:
            return False

        return checker.check(operation, resource)

    def require_permission(
        self,
        plugin_id: str,
        operation: str,
        resource: str,
    ) -> None:
        """
        Require permission, raising if not permitted.

        Raises:
            PermissionViolation: If permission denied
        """
        if not self.check_permission(plugin_id, operation, resource):
            raise PermissionViolation(
                plugin_id, operation, resource,
                f"Permission denied: {operation} on {resource}"
            )

    # === Stats ===

    def get_stats(self) -> Dict[str, Any]:
        """Get runtime statistics."""
        return {
            **self._stats,
            "active_executions": len(self._active_contexts),
            "thread_pool_size": self._max_workers,
        }


class ResourceLimiter:
    """
    Utility for setting resource limits on the current process.

    Note: Only works on Unix systems.
    """

    @staticmethod
    def set_memory_limit(max_bytes: int) -> bool:
        """Set maximum memory usage."""
        if sys.platform == "win32":
            return False

        try:
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            resource.setrlimit(resource.RLIMIT_AS, (max_bytes, hard))
            return True
        except Exception:
            return False

    @staticmethod
    def set_cpu_limit(max_seconds: float) -> bool:
        """Set maximum CPU time."""
        if sys.platform == "win32":
            return False

        try:
            soft, hard = resource.getrlimit(resource.RLIMIT_CPU)
            resource.setrlimit(resource.RLIMIT_CPU, (int(max_seconds), hard))
            return True
        except Exception:
            return False

    @staticmethod
    def set_file_limit(max_files: int) -> bool:
        """Set maximum open files."""
        if sys.platform == "win32":
            return False

        try:
            soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            resource.setrlimit(resource.RLIMIT_NOFILE, (max_files, hard))
            return True
        except Exception:
            return False

    @staticmethod
    def get_memory_usage() -> int:
        """Get current memory usage in bytes."""
        if sys.platform == "win32":
            return 0

        try:
            usage = resource.getrusage(resource.RUSAGE_SELF)
            return usage.ru_maxrss * 1024  # Convert to bytes
        except Exception:
            return 0
