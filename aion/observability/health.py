"""
AION Health Checker

System health monitoring with:
- Configurable health checks
- Dependency health tracking
- Readiness and liveness probes
- Health aggregation
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

import structlog

from aion.observability.types import HealthCheck, HealthStatus, SystemHealth

logger = structlog.get_logger(__name__)


class HealthCheckBase(ABC):
    """Base class for health checks."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the health check."""
        pass

    @abstractmethod
    async def check(self) -> HealthCheck:
        """Perform the health check."""
        pass


class FunctionHealthCheck(HealthCheckBase):
    """Health check from a function."""

    def __init__(
        self,
        name: str,
        check_fn: Callable[[], HealthCheck],
        is_async: bool = False,
    ):
        self._name = name
        self._check_fn = check_fn
        self._is_async = is_async

    @property
    def name(self) -> str:
        return self._name

    async def check(self) -> HealthCheck:
        if self._is_async:
            return await self._check_fn()
        return self._check_fn()


class HealthChecker:
    """
    SOTA System health monitoring.

    Features:
    - Configurable health checks
    - Dependency health tracking
    - Readiness and liveness probes
    - Health history and trends
    """

    def __init__(
        self,
        check_interval: float = 30.0,
        failure_threshold: int = 3,
        success_threshold: int = 1,
    ):
        self.check_interval = check_interval
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold

        # Registered checks
        self._checks: Dict[str, HealthCheckBase] = {}

        # Check results
        self._results: Dict[str, HealthCheck] = {}
        self._result_history: Dict[str, List[HealthCheck]] = {}
        self._max_history = 100

        # Failure counts for threshold-based status
        self._failure_counts: Dict[str, int] = {}
        self._success_counts: Dict[str, int] = {}

        # Service info
        self._service_name = "aion"
        self._service_version = ""
        self._start_time = datetime.utcnow()

        # Readiness/Liveness
        self._ready = False
        self._alive = True

        # Background task
        self._check_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the health checker."""
        if self._initialized:
            return

        logger.info("Initializing Health Checker")

        # Register built-in checks
        self._register_builtin_checks()

        # Start check loop
        self._check_task = asyncio.create_task(self._check_loop())

        # Run initial check
        await self.run_all_checks()

        self._ready = True
        self._initialized = True

    async def shutdown(self) -> None:
        """Shutdown the health checker."""
        logger.info("Shutting down Health Checker")

        self._shutdown_event.set()
        self._ready = False

        if self._check_task:
            self._check_task.cancel()
            try:
                await self._check_task
            except asyncio.CancelledError:
                pass

        self._initialized = False

    def _register_builtin_checks(self) -> None:
        """Register built-in health checks."""

        async def memory_check() -> HealthCheck:
            """Check memory usage."""
            try:
                import psutil
                memory = psutil.virtual_memory()

                status = HealthStatus.HEALTHY
                if memory.percent > 95:
                    status = HealthStatus.UNHEALTHY
                elif memory.percent > 85:
                    status = HealthStatus.DEGRADED

                return HealthCheck(
                    name="memory",
                    status=status,
                    message=f"Memory usage: {memory.percent:.1f}%",
                    details={
                        "total_gb": round(memory.total / (1024**3), 2),
                        "available_gb": round(memory.available / (1024**3), 2),
                        "percent": memory.percent,
                    },
                )
            except ImportError:
                return HealthCheck(
                    name="memory",
                    status=HealthStatus.UNKNOWN,
                    message="psutil not available",
                )

        async def disk_check() -> HealthCheck:
            """Check disk usage."""
            try:
                import psutil
                disk = psutil.disk_usage('/')

                status = HealthStatus.HEALTHY
                if disk.percent > 95:
                    status = HealthStatus.UNHEALTHY
                elif disk.percent > 85:
                    status = HealthStatus.DEGRADED

                return HealthCheck(
                    name="disk",
                    status=status,
                    message=f"Disk usage: {disk.percent:.1f}%",
                    details={
                        "total_gb": round(disk.total / (1024**3), 2),
                        "free_gb": round(disk.free / (1024**3), 2),
                        "percent": disk.percent,
                    },
                )
            except ImportError:
                return HealthCheck(
                    name="disk",
                    status=HealthStatus.UNKNOWN,
                    message="psutil not available",
                )

        async def cpu_check() -> HealthCheck:
            """Check CPU usage."""
            try:
                import psutil
                cpu_percent = psutil.cpu_percent(interval=0.1)

                status = HealthStatus.HEALTHY
                if cpu_percent > 95:
                    status = HealthStatus.UNHEALTHY
                elif cpu_percent > 80:
                    status = HealthStatus.DEGRADED

                return HealthCheck(
                    name="cpu",
                    status=status,
                    message=f"CPU usage: {cpu_percent:.1f}%",
                    details={
                        "percent": cpu_percent,
                        "cores": psutil.cpu_count(),
                    },
                )
            except ImportError:
                return HealthCheck(
                    name="cpu",
                    status=HealthStatus.UNKNOWN,
                    message="psutil not available",
                )

        async def event_loop_check() -> HealthCheck:
            """Check event loop responsiveness."""
            start = time.perf_counter()
            await asyncio.sleep(0.001)
            elapsed_ms = (time.perf_counter() - start) * 1000

            status = HealthStatus.HEALTHY
            if elapsed_ms > 100:
                status = HealthStatus.UNHEALTHY
            elif elapsed_ms > 50:
                status = HealthStatus.DEGRADED

            return HealthCheck(
                name="event_loop",
                status=status,
                message=f"Event loop latency: {elapsed_ms:.2f}ms",
                latency_ms=elapsed_ms,
            )

        self.register_check(FunctionHealthCheck("memory", memory_check, is_async=True))
        self.register_check(FunctionHealthCheck("disk", disk_check, is_async=True))
        self.register_check(FunctionHealthCheck("cpu", cpu_check, is_async=True))
        self.register_check(FunctionHealthCheck("event_loop", event_loop_check, is_async=True))

    def register_check(self, check: HealthCheckBase) -> None:
        """Register a health check."""
        self._checks[check.name] = check
        self._failure_counts[check.name] = 0
        self._success_counts[check.name] = 0
        self._result_history[check.name] = []
        logger.debug(f"Registered health check: {check.name}")

    def register_function_check(
        self,
        name: str,
        check_fn: Callable[[], HealthCheck],
        is_async: bool = False,
    ) -> None:
        """Register a health check from a function."""
        self.register_check(FunctionHealthCheck(name, check_fn, is_async))

    def unregister_check(self, name: str) -> bool:
        """Unregister a health check."""
        if name in self._checks:
            del self._checks[name]
            return True
        return False

    async def _check_loop(self) -> None:
        """Background health check loop."""
        while not self._shutdown_event.is_set():
            try:
                await self.run_all_checks()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")

    async def run_all_checks(self) -> SystemHealth:
        """Run all health checks."""
        results = []

        for name, check in self._checks.items():
            try:
                start = time.perf_counter()
                result = await check.check()
                result.latency_ms = (time.perf_counter() - start) * 1000
                result.checked_at = datetime.utcnow()

                # Update threshold counters
                if result.status == HealthStatus.HEALTHY:
                    self._success_counts[name] += 1
                    self._failure_counts[name] = 0
                elif result.status == HealthStatus.UNHEALTHY:
                    self._failure_counts[name] += 1
                    self._success_counts[name] = 0

                self._results[name] = result
                self._add_to_history(name, result)
                results.append(result)

            except Exception as e:
                result = HealthCheck(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Check failed: {e}",
                )
                self._results[name] = result
                self._failure_counts[name] += 1
                results.append(result)

        # Calculate overall status
        overall = self._calculate_overall_status(results)

        return SystemHealth(
            status=overall,
            checks=results,
            service_name=self._service_name,
            version=self._service_version,
            uptime_seconds=(datetime.utcnow() - self._start_time).total_seconds(),
        )

    def _calculate_overall_status(self, results: List[HealthCheck]) -> HealthStatus:
        """Calculate overall health status."""
        if not results:
            return HealthStatus.UNKNOWN

        statuses = [r.status for r in results]

        # All healthy
        if all(s == HealthStatus.HEALTHY for s in statuses):
            return HealthStatus.HEALTHY

        # Any unhealthy
        if any(s == HealthStatus.UNHEALTHY for s in statuses):
            return HealthStatus.UNHEALTHY

        # Any degraded
        if any(s == HealthStatus.DEGRADED for s in statuses):
            return HealthStatus.DEGRADED

        return HealthStatus.UNKNOWN

    def _add_to_history(self, name: str, result: HealthCheck) -> None:
        """Add result to history."""
        if name not in self._result_history:
            self._result_history[name] = []

        self._result_history[name].append(result)

        # Trim history
        if len(self._result_history[name]) > self._max_history:
            self._result_history[name] = self._result_history[name][-self._max_history:]

    async def run_check(self, name: str) -> Optional[HealthCheck]:
        """Run a specific health check."""
        check = self._checks.get(name)
        if not check:
            return None

        try:
            start = time.perf_counter()
            result = await check.check()
            result.latency_ms = (time.perf_counter() - start) * 1000
            result.checked_at = datetime.utcnow()

            self._results[name] = result
            self._add_to_history(name, result)
            return result
        except Exception as e:
            return HealthCheck(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Check failed: {e}",
            )

    # === Readiness/Liveness ===

    def is_ready(self) -> bool:
        """Check if service is ready to accept traffic."""
        return self._ready and self.get_health().status != HealthStatus.UNHEALTHY

    def is_alive(self) -> bool:
        """Check if service is alive (liveness probe)."""
        return self._alive

    def set_ready(self, ready: bool) -> None:
        """Set readiness status."""
        self._ready = ready

    def set_alive(self, alive: bool) -> None:
        """Set liveness status."""
        self._alive = alive

    # === Query Methods ===

    def get_health(self) -> SystemHealth:
        """Get current health status."""
        results = list(self._results.values())
        overall = self._calculate_overall_status(results)

        return SystemHealth(
            status=overall,
            checks=results,
            service_name=self._service_name,
            version=self._service_version,
            uptime_seconds=(datetime.utcnow() - self._start_time).total_seconds(),
        )

    def get_check(self, name: str) -> Optional[HealthCheck]:
        """Get latest result for a specific check."""
        return self._results.get(name)

    def get_check_history(
        self,
        name: str,
        limit: int = 50,
    ) -> List[HealthCheck]:
        """Get history for a specific check."""
        return self._result_history.get(name, [])[-limit:]

    def get_unhealthy_checks(self) -> List[HealthCheck]:
        """Get all unhealthy checks."""
        return [
            c for c in self._results.values()
            if c.status == HealthStatus.UNHEALTHY
        ]

    def get_degraded_checks(self) -> List[HealthCheck]:
        """Get all degraded checks."""
        return [
            c for c in self._results.values()
            if c.status == HealthStatus.DEGRADED
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Get health checker statistics."""
        results = list(self._results.values())

        return {
            "checks_registered": len(self._checks),
            "healthy_count": sum(1 for r in results if r.status == HealthStatus.HEALTHY),
            "degraded_count": sum(1 for r in results if r.status == HealthStatus.DEGRADED),
            "unhealthy_count": sum(1 for r in results if r.status == HealthStatus.UNHEALTHY),
            "overall_status": self.get_health().status.value,
            "ready": self._ready,
            "alive": self._alive,
            "uptime_seconds": (datetime.utcnow() - self._start_time).total_seconds(),
        }
