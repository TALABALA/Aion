"""
Resource Limits and Quotas

Enforces resource limits on plugins including memory, CPU, file descriptors,
and network connections.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import structlog

logger = structlog.get_logger(__name__)


class ResourceViolation(Exception):
    """Raised when a resource limit is violated."""

    def __init__(
        self,
        resource: str,
        limit: float,
        actual: float,
        message: Optional[str] = None,
    ):
        self.resource = resource
        self.limit = limit
        self.actual = actual
        super().__init__(
            message or f"Resource limit exceeded: {resource} "
            f"(limit={limit}, actual={actual})"
        )


@dataclass
class ResourceQuota:
    """Resource quota configuration for a plugin."""

    # Memory limits
    max_memory_mb: float = 512.0
    memory_warning_threshold: float = 0.8  # Warn at 80%

    # CPU limits
    max_cpu_percent: float = 50.0
    cpu_time_limit_seconds: float = 3600.0  # 1 hour total CPU time

    # I/O limits
    max_file_descriptors: int = 100
    max_open_files: int = 50
    max_file_size_mb: float = 100.0

    # Network limits
    max_connections: int = 10
    max_bandwidth_mbps: float = 10.0

    # Thread/process limits
    max_threads: int = 10
    max_subprocesses: int = 0  # Disabled by default

    # Time limits
    max_execution_time: float = 60.0  # Per operation
    max_idle_time: float = 300.0  # 5 minutes idle

    # Custom limits
    custom_limits: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "max_memory_mb": self.max_memory_mb,
            "memory_warning_threshold": self.memory_warning_threshold,
            "max_cpu_percent": self.max_cpu_percent,
            "cpu_time_limit_seconds": self.cpu_time_limit_seconds,
            "max_file_descriptors": self.max_file_descriptors,
            "max_open_files": self.max_open_files,
            "max_file_size_mb": self.max_file_size_mb,
            "max_connections": self.max_connections,
            "max_bandwidth_mbps": self.max_bandwidth_mbps,
            "max_threads": self.max_threads,
            "max_subprocesses": self.max_subprocesses,
            "max_execution_time": self.max_execution_time,
            "max_idle_time": self.max_idle_time,
            "custom_limits": self.custom_limits,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResourceQuota":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def minimal(cls) -> "ResourceQuota":
        """Create minimal quota for untrusted plugins."""
        return cls(
            max_memory_mb=128.0,
            max_cpu_percent=10.0,
            max_file_descriptors=20,
            max_open_files=10,
            max_connections=2,
            max_threads=2,
            max_subprocesses=0,
            max_execution_time=10.0,
        )

    @classmethod
    def standard(cls) -> "ResourceQuota":
        """Create standard quota for normal plugins."""
        return cls()

    @classmethod
    def elevated(cls) -> "ResourceQuota":
        """Create elevated quota for trusted plugins."""
        return cls(
            max_memory_mb=2048.0,
            max_cpu_percent=80.0,
            max_file_descriptors=500,
            max_open_files=200,
            max_connections=50,
            max_threads=50,
            max_subprocesses=5,
            max_execution_time=300.0,
        )


@dataclass
class ResourceUsage:
    """Current resource usage for a plugin."""

    memory_mb: float = 0.0
    cpu_percent: float = 0.0
    cpu_time_seconds: float = 0.0
    open_files: int = 0
    open_connections: int = 0
    thread_count: int = 0
    subprocess_count: int = 0
    last_activity: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "memory_mb": self.memory_mb,
            "cpu_percent": self.cpu_percent,
            "cpu_time_seconds": self.cpu_time_seconds,
            "open_files": self.open_files,
            "open_connections": self.open_connections,
            "thread_count": self.thread_count,
            "subprocess_count": self.subprocess_count,
            "last_activity": self.last_activity,
            "idle_time": time.time() - self.last_activity,
        }


class ResourceEnforcer:
    """
    Enforces resource limits on plugins.

    Monitors resource usage and raises violations when limits are exceeded.
    """

    def __init__(
        self,
        quota: ResourceQuota,
        plugin_id: str,
        on_warning: Optional[Callable[[str, str, float, float], None]] = None,
        on_violation: Optional[Callable[[ResourceViolation], None]] = None,
    ):
        self.quota = quota
        self.plugin_id = plugin_id
        self.on_warning = on_warning
        self.on_violation = on_violation
        self.usage = ResourceUsage()
        self._start_time = time.time()
        self._warnings_issued: set[str] = set()

    def check_memory(self, memory_mb: float) -> None:
        """Check memory usage against quota."""
        self.usage.memory_mb = memory_mb
        self.usage.last_activity = time.time()

        # Check warning threshold
        warning_limit = self.quota.max_memory_mb * self.quota.memory_warning_threshold
        if memory_mb >= warning_limit and "memory_warning" not in self._warnings_issued:
            self._warnings_issued.add("memory_warning")
            self._warn("memory", memory_mb, self.quota.max_memory_mb)

        # Check hard limit
        if memory_mb > self.quota.max_memory_mb:
            violation = ResourceViolation(
                "memory_mb",
                self.quota.max_memory_mb,
                memory_mb,
            )
            self._handle_violation(violation)
            raise violation

    def check_cpu(self, cpu_percent: float, cpu_time: float) -> None:
        """Check CPU usage against quota."""
        self.usage.cpu_percent = cpu_percent
        self.usage.cpu_time_seconds = cpu_time
        self.usage.last_activity = time.time()

        if cpu_percent > self.quota.max_cpu_percent:
            violation = ResourceViolation(
                "cpu_percent",
                self.quota.max_cpu_percent,
                cpu_percent,
            )
            self._handle_violation(violation)
            raise violation

        if cpu_time > self.quota.cpu_time_limit_seconds:
            violation = ResourceViolation(
                "cpu_time_seconds",
                self.quota.cpu_time_limit_seconds,
                cpu_time,
            )
            self._handle_violation(violation)
            raise violation

    def check_files(self, open_files: int) -> None:
        """Check file descriptor usage against quota."""
        self.usage.open_files = open_files
        self.usage.last_activity = time.time()

        if open_files > self.quota.max_open_files:
            violation = ResourceViolation(
                "open_files",
                self.quota.max_open_files,
                open_files,
            )
            self._handle_violation(violation)
            raise violation

    def check_connections(self, connections: int) -> None:
        """Check network connection count against quota."""
        self.usage.open_connections = connections
        self.usage.last_activity = time.time()

        if connections > self.quota.max_connections:
            violation = ResourceViolation(
                "connections",
                self.quota.max_connections,
                connections,
            )
            self._handle_violation(violation)
            raise violation

    def check_threads(self, threads: int) -> None:
        """Check thread count against quota."""
        self.usage.thread_count = threads
        self.usage.last_activity = time.time()

        if threads > self.quota.max_threads:
            violation = ResourceViolation(
                "threads",
                self.quota.max_threads,
                threads,
            )
            self._handle_violation(violation)
            raise violation

    def check_subprocesses(self, count: int) -> None:
        """Check subprocess count against quota."""
        self.usage.subprocess_count = count
        self.usage.last_activity = time.time()

        if count > self.quota.max_subprocesses:
            violation = ResourceViolation(
                "subprocesses",
                self.quota.max_subprocesses,
                count,
            )
            self._handle_violation(violation)
            raise violation

    def check_idle(self) -> None:
        """Check if plugin has been idle too long."""
        idle_time = time.time() - self.usage.last_activity

        if idle_time > self.quota.max_idle_time:
            violation = ResourceViolation(
                "idle_time",
                self.quota.max_idle_time,
                idle_time,
                "Plugin has been idle too long",
            )
            self._handle_violation(violation)
            raise violation

    def check_execution_time(self, elapsed: float) -> None:
        """Check if execution time exceeds limit."""
        if elapsed > self.quota.max_execution_time:
            violation = ResourceViolation(
                "execution_time",
                self.quota.max_execution_time,
                elapsed,
            )
            self._handle_violation(violation)
            raise violation

    def check_custom(self, name: str, value: float) -> None:
        """Check a custom resource limit."""
        limit = self.quota.custom_limits.get(name)
        if limit is not None and value > limit:
            violation = ResourceViolation(name, limit, value)
            self._handle_violation(violation)
            raise violation

    def record_activity(self) -> None:
        """Record activity to reset idle timer."""
        self.usage.last_activity = time.time()

    def get_usage(self) -> ResourceUsage:
        """Get current resource usage."""
        return self.usage

    def get_stats(self) -> dict[str, Any]:
        """Get resource enforcement statistics."""
        return {
            "plugin_id": self.plugin_id,
            "quota": self.quota.to_dict(),
            "usage": self.usage.to_dict(),
            "uptime": time.time() - self._start_time,
            "warnings_issued": list(self._warnings_issued),
        }

    def _warn(self, resource: str, actual: float, limit: float) -> None:
        """Issue a warning."""
        logger.warning(
            "Resource warning",
            plugin_id=self.plugin_id,
            resource=resource,
            actual=actual,
            limit=limit,
        )
        if self.on_warning:
            self.on_warning(self.plugin_id, resource, actual, limit)

    def _handle_violation(self, violation: ResourceViolation) -> None:
        """Handle a resource violation."""
        logger.error(
            "Resource violation",
            plugin_id=self.plugin_id,
            resource=violation.resource,
            limit=violation.limit,
            actual=violation.actual,
        )
        if self.on_violation:
            self.on_violation(violation)


class ResourceMonitor:
    """
    Monitors resource usage across multiple plugins.

    Provides centralized monitoring and enforcement of resource limits.
    """

    def __init__(self):
        self._enforcers: dict[str, ResourceEnforcer] = {}
        self._monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._monitor_interval = 5.0  # seconds

    def register_plugin(
        self,
        plugin_id: str,
        quota: ResourceQuota,
        on_warning: Optional[Callable] = None,
        on_violation: Optional[Callable] = None,
    ) -> ResourceEnforcer:
        """Register a plugin for resource monitoring."""
        enforcer = ResourceEnforcer(
            quota=quota,
            plugin_id=plugin_id,
            on_warning=on_warning,
            on_violation=on_violation,
        )
        self._enforcers[plugin_id] = enforcer
        return enforcer

    def unregister_plugin(self, plugin_id: str) -> None:
        """Unregister a plugin from monitoring."""
        self._enforcers.pop(plugin_id, None)

    def get_enforcer(self, plugin_id: str) -> Optional[ResourceEnforcer]:
        """Get the enforcer for a plugin."""
        return self._enforcers.get(plugin_id)

    async def start_monitoring(self) -> None:
        """Start the background monitoring task."""
        if self._monitoring:
            return

        self._monitoring = True
        self._monitor_task = asyncio.create_task(self._monitoring_loop())

    async def stop_monitoring(self) -> None:
        """Stop the background monitoring task."""
        self._monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

    async def _monitoring_loop(self) -> None:
        """Background loop that monitors all plugins."""
        while self._monitoring:
            try:
                await asyncio.sleep(self._monitor_interval)

                for plugin_id, enforcer in list(self._enforcers.items()):
                    try:
                        # Check idle time for all plugins
                        enforcer.check_idle()
                    except ResourceViolation as e:
                        logger.warning(
                            "Plugin idle violation",
                            plugin_id=plugin_id,
                            error=str(e),
                        )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Monitoring loop error", error=str(e))

    def get_total_usage(self) -> dict[str, Any]:
        """Get total resource usage across all plugins."""
        total_memory = sum(e.usage.memory_mb for e in self._enforcers.values())
        total_cpu = sum(e.usage.cpu_percent for e in self._enforcers.values())
        total_threads = sum(e.usage.thread_count for e in self._enforcers.values())

        return {
            "total_memory_mb": total_memory,
            "total_cpu_percent": total_cpu,
            "total_threads": total_threads,
            "plugin_count": len(self._enforcers),
            "per_plugin": {
                pid: enforcer.get_stats()
                for pid, enforcer in self._enforcers.items()
            },
        }
