"""
AION Performance Profiler

Profile performance of operations:
- CPU profiling
- Memory profiling
- Async operation tracking
- Hot spot detection
"""

from __future__ import annotations

import asyncio
import cProfile
import functools
import io
import pstats
import sys
import time
import traceback
import threading
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Generator, List, Optional, TypeVar

import structlog

from aion.observability.types import ProfileSample, ProfileReport
from aion.observability.context import get_context_manager

logger = structlog.get_logger(__name__)

T = TypeVar('T')


@dataclass
class OperationProfile:
    """Profile data for a single operation."""
    name: str
    call_count: int = 0
    total_time_ns: int = 0
    min_time_ns: int = sys.maxsize
    max_time_ns: int = 0
    error_count: int = 0

    # Recent timings for percentile calculation
    recent_times: List[int] = field(default_factory=list)
    max_recent: int = 1000

    def record(self, time_ns: int, error: bool = False) -> None:
        """Record a timing."""
        self.call_count += 1
        self.total_time_ns += time_ns
        self.min_time_ns = min(self.min_time_ns, time_ns)
        self.max_time_ns = max(self.max_time_ns, time_ns)

        if error:
            self.error_count += 1

        self.recent_times.append(time_ns)
        if len(self.recent_times) > self.max_recent:
            self.recent_times.pop(0)

    @property
    def avg_time_ns(self) -> float:
        """Get average time in nanoseconds."""
        return self.total_time_ns / self.call_count if self.call_count > 0 else 0

    @property
    def avg_time_ms(self) -> float:
        """Get average time in milliseconds."""
        return self.avg_time_ns / 1_000_000

    def get_percentile(self, p: float) -> int:
        """Get percentile timing."""
        if not self.recent_times:
            return 0
        sorted_times = sorted(self.recent_times)
        idx = int(p * len(sorted_times))
        return sorted_times[min(idx, len(sorted_times) - 1)]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "call_count": self.call_count,
            "total_time_ms": self.total_time_ns / 1_000_000,
            "avg_time_ms": self.avg_time_ms,
            "min_time_ms": self.min_time_ns / 1_000_000 if self.min_time_ns != sys.maxsize else 0,
            "max_time_ms": self.max_time_ns / 1_000_000,
            "p50_ms": self.get_percentile(0.5) / 1_000_000,
            "p90_ms": self.get_percentile(0.9) / 1_000_000,
            "p99_ms": self.get_percentile(0.99) / 1_000_000,
            "error_count": self.error_count,
            "error_rate": self.error_count / self.call_count if self.call_count > 0 else 0,
        }


class Profiler:
    """
    SOTA Performance profiler for AION.

    Features:
    - Operation timing
    - CPU profiling
    - Memory tracking
    - Hot spot detection
    - Async operation support
    """

    def __init__(
        self,
        enable_cpu_profiling: bool = False,
        enable_memory_tracking: bool = False,
        hot_spot_threshold_ms: float = 100.0,
    ):
        self.enable_cpu_profiling = enable_cpu_profiling
        self.enable_memory_tracking = enable_memory_tracking
        self.hot_spot_threshold_ms = hot_spot_threshold_ms

        # Operation profiles
        self._operations: Dict[str, OperationProfile] = {}

        # CPU profiler
        self._cpu_profiler: Optional[cProfile.Profile] = None
        self._profiling_active = False

        # Memory baseline
        self._memory_baseline = 0

        # Hot spots
        self._hot_spots: List[Dict[str, Any]] = []
        self._max_hot_spots = 100

        # Active profiles (for nested profiling)
        self._active_profiles: Dict[str, List[int]] = defaultdict(list)

        # Lock for thread safety
        self._lock = threading.Lock()

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the profiler."""
        if self._initialized:
            return

        logger.info("Initializing Profiler")

        if self.enable_memory_tracking:
            self._memory_baseline = self._get_memory_usage()

        self._initialized = True

    async def shutdown(self) -> None:
        """Shutdown the profiler."""
        if self._profiling_active:
            self.stop_cpu_profiling()

        self._initialized = False

    def _get_memory_usage(self) -> int:
        """Get current memory usage in bytes."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss
        except Exception:
            return 0

    # === Operation Profiling ===

    @contextmanager
    def profile_operation(
        self,
        name: str,
        record_trace: bool = True,
    ) -> Generator[OperationProfile, None, None]:
        """
        Context manager to profile an operation.

        Usage:
            with profiler.profile_operation("process_request") as profile:
                do_something()
                profile.record_something()
        """
        start_time = time.perf_counter_ns()

        with self._lock:
            if name not in self._operations:
                self._operations[name] = OperationProfile(name=name)
            profile = self._operations[name]

        error = False
        try:
            yield profile
        except Exception:
            error = True
            raise
        finally:
            elapsed = time.perf_counter_ns() - start_time
            profile.record(elapsed, error)

            # Check for hot spot
            if elapsed / 1_000_000 > self.hot_spot_threshold_ms:
                self._record_hot_spot(name, elapsed, error)

    def time_operation(self, name: str):
        """
        Decorator to profile a function.

        Usage:
            @profiler.time_operation("process_data")
            def process_data(data):
                ...
        """
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            if asyncio.iscoroutinefunction(func):
                @functools.wraps(func)
                async def async_wrapper(*args, **kwargs):
                    with self.profile_operation(name):
                        return await func(*args, **kwargs)
                return async_wrapper
            else:
                @functools.wraps(func)
                def sync_wrapper(*args, **kwargs):
                    with self.profile_operation(name):
                        return func(*args, **kwargs)
                return sync_wrapper
        return decorator

    def record_timing(
        self,
        name: str,
        duration_ms: float,
        error: bool = False,
    ) -> None:
        """Manually record a timing."""
        with self._lock:
            if name not in self._operations:
                self._operations[name] = OperationProfile(name=name)
            self._operations[name].record(int(duration_ms * 1_000_000), error)

    def _record_hot_spot(
        self,
        name: str,
        elapsed_ns: int,
        error: bool,
    ) -> None:
        """Record a hot spot."""
        hot_spot = {
            "name": name,
            "duration_ms": elapsed_ns / 1_000_000,
            "timestamp": datetime.utcnow().isoformat(),
            "error": error,
            "trace_id": get_context_manager().get_trace_id(),
            "stack": self._get_stack_trace(),
        }

        with self._lock:
            self._hot_spots.append(hot_spot)
            if len(self._hot_spots) > self._max_hot_spots:
                self._hot_spots.pop(0)

    def _get_stack_trace(self) -> str:
        """Get current stack trace."""
        return "".join(traceback.format_stack()[:-2])

    # === CPU Profiling ===

    def start_cpu_profiling(self) -> None:
        """Start CPU profiling."""
        if self._profiling_active:
            return

        self._cpu_profiler = cProfile.Profile()
        self._cpu_profiler.enable()
        self._profiling_active = True
        logger.info("CPU profiling started")

    def stop_cpu_profiling(self) -> ProfileReport:
        """Stop CPU profiling and return report."""
        if not self._profiling_active or not self._cpu_profiler:
            return ProfileReport(name="cpu_profile")

        self._cpu_profiler.disable()
        self._profiling_active = False

        # Get stats
        stream = io.StringIO()
        stats = pstats.Stats(self._cpu_profiler, stream=stream)
        stats.sort_stats('cumulative')
        stats.print_stats(50)

        # Parse hot spots
        hot_spots = {}
        self._cpu_profiler.print_stats()

        for func, data in self._cpu_profiler.getstats():
            if hasattr(func, 'co_name'):
                name = f"{func.co_filename}:{func.co_name}"
                hot_spots[name] = int(data.totaltime * 1_000_000_000)

        report = ProfileReport(
            name="cpu_profile",
            end_time=datetime.utcnow(),
            total_cpu_time_ns=sum(hot_spots.values()),
            hot_spots=hot_spots,
        )

        self._cpu_profiler = None
        logger.info("CPU profiling stopped")

        return report

    @contextmanager
    def cpu_profile_block(self, name: str = "block") -> Generator[None, None, None]:
        """Context manager for CPU profiling a block."""
        profiler = cProfile.Profile()
        profiler.enable()
        try:
            yield
        finally:
            profiler.disable()

            # Log top functions
            stats = pstats.Stats(profiler)
            stats.sort_stats('cumulative')

            logger.debug(f"CPU profile for {name}:")
            stream = io.StringIO()
            stats.stream = stream
            stats.print_stats(10)
            logger.debug(stream.getvalue())

    # === Memory Profiling ===

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage."""
        try:
            import psutil

            process = psutil.Process()
            mem_info = process.memory_info()

            return {
                "rss_bytes": mem_info.rss,
                "rss_mb": mem_info.rss / (1024 * 1024),
                "vms_bytes": mem_info.vms,
                "vms_mb": mem_info.vms / (1024 * 1024),
                "percent": process.memory_percent(),
                "baseline_mb": self._memory_baseline / (1024 * 1024),
                "delta_mb": (mem_info.rss - self._memory_baseline) / (1024 * 1024),
            }
        except ImportError:
            return {"error": "psutil not available"}
        except Exception as e:
            return {"error": str(e)}

    @contextmanager
    def track_memory(self, name: str) -> Generator[Dict[str, Any], None, None]:
        """Context manager to track memory usage of a block."""
        result = {"name": name, "start_mb": 0, "end_mb": 0, "delta_mb": 0}

        try:
            import psutil
            process = psutil.Process()
            start_mem = process.memory_info().rss

            result["start_mb"] = start_mem / (1024 * 1024)

            yield result

            end_mem = process.memory_info().rss
            result["end_mb"] = end_mem / (1024 * 1024)
            result["delta_mb"] = (end_mem - start_mem) / (1024 * 1024)

            if result["delta_mb"] > 100:  # More than 100MB
                logger.warning(
                    f"High memory usage in {name}",
                    delta_mb=result["delta_mb"],
                )
        except ImportError:
            yield result

    # === Query Methods ===

    def get_operation(self, name: str) -> Optional[OperationProfile]:
        """Get profile for a specific operation."""
        return self._operations.get(name)

    def get_all_operations(self) -> Dict[str, dict]:
        """Get all operation profiles."""
        return {name: op.to_dict() for name, op in self._operations.items()}

    def get_slowest_operations(self, limit: int = 10) -> List[dict]:
        """Get slowest operations by average time."""
        sorted_ops = sorted(
            self._operations.values(),
            key=lambda op: op.avg_time_ns,
            reverse=True,
        )
        return [op.to_dict() for op in sorted_ops[:limit]]

    def get_most_called_operations(self, limit: int = 10) -> List[dict]:
        """Get most frequently called operations."""
        sorted_ops = sorted(
            self._operations.values(),
            key=lambda op: op.call_count,
            reverse=True,
        )
        return [op.to_dict() for op in sorted_ops[:limit]]

    def get_hot_spots(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent hot spots."""
        return self._hot_spots[-limit:]

    def get_stats(self) -> Dict[str, Any]:
        """Get profiler statistics."""
        return {
            "operations_tracked": len(self._operations),
            "hot_spots_recorded": len(self._hot_spots),
            "cpu_profiling_active": self._profiling_active,
            "total_calls": sum(op.call_count for op in self._operations.values()),
            "total_errors": sum(op.error_count for op in self._operations.values()),
            "memory": self.get_memory_usage() if self.enable_memory_tracking else {},
        }

    def reset(self) -> None:
        """Reset all profiling data."""
        with self._lock:
            self._operations.clear()
            self._hot_spots.clear()


# Convenience decorators
def profile(name: str = None):
    """
    Decorator to profile a function.

    Usage:
        @profile("process_request")
        async def process_request(req):
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        op_name = name or func.__name__

        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                from aion.observability import get_profiler
                profiler = get_profiler()
                if profiler:
                    with profiler.profile_operation(op_name):
                        return await func(*args, **kwargs)
                return await func(*args, **kwargs)
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                from aion.observability import get_profiler
                profiler = get_profiler()
                if profiler:
                    with profiler.profile_operation(op_name):
                        return func(*args, **kwargs)
                return func(*args, **kwargs)
            return sync_wrapper

    return decorator


def profile_method():
    """
    Decorator to profile a method (includes class name).

    Usage:
        class MyService:
            @profile_method()
            async def process(self):
                ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(self, *args, **kwargs):
                from aion.observability import get_profiler
                profiler = get_profiler()
                op_name = f"{self.__class__.__name__}.{func.__name__}"
                if profiler:
                    with profiler.profile_operation(op_name):
                        return await func(self, *args, **kwargs)
                return await func(self, *args, **kwargs)
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(self, *args, **kwargs):
                from aion.observability import get_profiler
                profiler = get_profiler()
                op_name = f"{self.__class__.__name__}.{func.__name__}"
                if profiler:
                    with profiler.profile_operation(op_name):
                        return func(self, *args, **kwargs)
                return func(self, *args, **kwargs)
            return sync_wrapper

    return decorator
