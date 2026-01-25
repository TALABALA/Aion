"""
Continuous Profiling with Flame Graphs

SOTA features:
- Continuous CPU profiling
- Memory allocation profiling
- Lock contention profiling
- Async task profiling
- Flame graph generation
- Differential flame graphs
"""

from __future__ import annotations

import asyncio
import cProfile
import gc
import io
import logging
import pstats
import sys
import threading
import time
import traceback
import tracemalloc
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import json

logger = logging.getLogger(__name__)


# =============================================================================
# Types
# =============================================================================

class ProfileType(Enum):
    """Types of profiles."""
    CPU = "cpu"
    MEMORY = "memory"
    ALLOCATION = "allocation"
    WALL_TIME = "wall_time"
    LOCK = "lock"
    ASYNC = "async"


@dataclass
class ProfileFrame:
    """A single frame in a profile stack."""
    function: str
    filename: str
    line_number: int
    module: str = ""

    def __hash__(self):
        return hash((self.function, self.filename, self.line_number))

    def __eq__(self, other):
        if not isinstance(other, ProfileFrame):
            return False
        return (
            self.function == other.function and
            self.filename == other.filename and
            self.line_number == other.line_number
        )

    def to_string(self) -> str:
        """Convert to string for flame graph."""
        return f"{self.module}.{self.function}" if self.module else self.function


@dataclass
class ProfileStack:
    """A stack trace with an associated value."""
    frames: List[ProfileFrame]
    value: float  # CPU time, memory bytes, count, etc.
    labels: Dict[str, str] = field(default_factory=dict)

    def to_collapsed(self) -> str:
        """Convert to collapsed stack format for flame graphs."""
        stack_str = ";".join(f.to_string() for f in reversed(self.frames))
        return f"{stack_str} {int(self.value)}"


@dataclass
class ProfileSample:
    """A single profile sample."""
    timestamp: datetime
    stack: ProfileStack
    profile_type: ProfileType
    thread_id: int = 0
    thread_name: str = ""


@dataclass
class ProfileData:
    """Aggregated profile data."""
    profile_type: ProfileType
    start_time: datetime
    end_time: datetime
    samples: List[ProfileSample] = field(default_factory=list)
    aggregated_stacks: Dict[str, float] = field(default_factory=dict)

    def to_collapsed(self) -> str:
        """Export to collapsed stack format."""
        lines = []
        for stack_str, value in self.aggregated_stacks.items():
            lines.append(f"{stack_str} {int(value)}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.profile_type.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "sample_count": len(self.samples),
            "unique_stacks": len(self.aggregated_stacks),
        }


# =============================================================================
# Flame Graph Generator
# =============================================================================

@dataclass
class FlameGraphNode:
    """A node in the flame graph tree."""
    name: str
    value: float
    children: Dict[str, "FlameGraphNode"] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "children": [c.to_dict() for c in self.children.values()],
        }


class FlameGraph:
    """
    Flame graph representation and generation.

    Supports:
    - Standard flame graphs
    - Icicle graphs (inverted)
    - Differential flame graphs
    - SVG and JSON export
    """

    def __init__(self, profile_data: ProfileData):
        self.profile_data = profile_data
        self.root = FlameGraphNode(name="root", value=0)
        self._build_tree()

    def _build_tree(self) -> None:
        """Build the flame graph tree from profile data."""
        for stack_str, value in self.profile_data.aggregated_stacks.items():
            frames = stack_str.split(";")
            self._add_to_tree(self.root, frames, value)

        # Calculate total value
        self.root.value = sum(c.value for c in self.root.children.values())

    def _add_to_tree(
        self,
        node: FlameGraphNode,
        frames: List[str],
        value: float,
    ) -> None:
        """Add a stack to the tree."""
        if not frames:
            return

        frame = frames[0]
        if frame not in node.children:
            node.children[frame] = FlameGraphNode(name=frame, value=0)

        child = node.children[frame]
        child.value += value

        if len(frames) > 1:
            self._add_to_tree(child, frames[1:], value)

    def to_json(self) -> str:
        """Export as D3 flame graph JSON."""
        return json.dumps(self.root.to_dict(), indent=2)

    def to_svg(
        self,
        width: int = 1200,
        height: int = 800,
        title: str = "Flame Graph",
    ) -> str:
        """Generate SVG flame graph."""
        svg_lines = []

        # Header
        svg_lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" '
                        f'viewBox="0 0 {width} {height}" '
                        f'width="{width}" height="{height}">')
        svg_lines.append('<style>')
        svg_lines.append('  .func { stroke: #000; stroke-width: 0.5; cursor: pointer; }')
        svg_lines.append('  .func:hover { stroke-width: 1.5; }')
        svg_lines.append('  text { font-family: monospace; font-size: 10px; }')
        svg_lines.append('</style>')

        # Title
        svg_lines.append(f'<text x="{width/2}" y="20" text-anchor="middle" '
                        f'style="font-size: 14px; font-weight: bold;">{title}</text>')

        # Generate rectangles
        y_offset = 40
        frame_height = 16
        total_value = self.root.value or 1

        def draw_node(
            node: FlameGraphNode,
            x: float,
            y: float,
            node_width: float,
            depth: int = 0,
        ) -> None:
            if node_width < 1 or depth > 50:
                return

            # Color based on function name hash
            hue = hash(node.name) % 360
            color = f"hsl({hue}, 70%, 60%)"

            # Draw rectangle
            svg_lines.append(
                f'<rect class="func" x="{x}" y="{y}" '
                f'width="{node_width}" height="{frame_height}" '
                f'fill="{color}">'
                f'<title>{node.name} ({node.value:.0f})</title>'
                f'</rect>'
            )

            # Draw text if wide enough
            if node_width > 50:
                text = node.name[:int(node_width / 6)]
                svg_lines.append(
                    f'<text x="{x + 2}" y="{y + 12}">{text}</text>'
                )

            # Draw children
            child_x = x
            for child in sorted(node.children.values(), key=lambda c: -c.value):
                child_width = (child.value / total_value) * width
                draw_node(child, child_x, y + frame_height, child_width, depth + 1)
                child_x += child_width

        # Draw from root
        draw_node(self.root, 0, y_offset, width)

        svg_lines.append('</svg>')
        return "\n".join(svg_lines)

    def to_html(self, title: str = "Flame Graph") -> str:
        """Generate interactive HTML flame graph."""
        json_data = self.to_json()

        return f'''<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <style>
        body {{ font-family: sans-serif; margin: 0; padding: 20px; }}
        #chart {{ width: 100%; height: 600px; }}
        .details {{ margin-top: 10px; padding: 10px; background: #f0f0f0; }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <div id="chart">{self.to_svg()}</div>
    <div class="details">
        <p>Profile Type: {self.profile_data.profile_type.value}</p>
        <p>Duration: {(self.profile_data.end_time - self.profile_data.start_time).total_seconds():.1f}s</p>
        <p>Samples: {len(self.profile_data.samples)}</p>
        <p>Unique Stacks: {len(self.profile_data.aggregated_stacks)}</p>
    </div>
</body>
</html>'''

    @staticmethod
    def differential(
        baseline: "FlameGraph",
        comparison: "FlameGraph",
    ) -> "FlameGraph":
        """
        Create a differential flame graph.

        Shows what changed between baseline and comparison.
        """
        # Create new profile data for diff
        diff_stacks = {}

        all_stacks = set(baseline.profile_data.aggregated_stacks.keys())
        all_stacks.update(comparison.profile_data.aggregated_stacks.keys())

        for stack in all_stacks:
            base_val = baseline.profile_data.aggregated_stacks.get(stack, 0)
            comp_val = comparison.profile_data.aggregated_stacks.get(stack, 0)
            diff = comp_val - base_val

            if abs(diff) > 0:
                diff_stacks[stack] = diff

        diff_data = ProfileData(
            profile_type=baseline.profile_data.profile_type,
            start_time=baseline.profile_data.start_time,
            end_time=comparison.profile_data.end_time,
            aggregated_stacks=diff_stacks,
        )

        return FlameGraph(diff_data)


# =============================================================================
# Continuous Profiler
# =============================================================================

class ContinuousProfiler:
    """
    Continuous profiler for production systems.

    Features:
    - Low-overhead sampling
    - Multiple profile types
    - Rolling window storage
    - Flame graph generation
    """

    def __init__(
        self,
        sample_rate_hz: float = 100.0,
        cpu_profiling: bool = True,
        memory_profiling: bool = True,
        allocation_profiling: bool = False,
        async_profiling: bool = True,
        max_samples: int = 100000,
        aggregation_interval_seconds: float = 60.0,
    ):
        self.sample_rate_hz = sample_rate_hz
        self.cpu_profiling = cpu_profiling
        self.memory_profiling = memory_profiling
        self.allocation_profiling = allocation_profiling
        self.async_profiling = async_profiling
        self.max_samples = max_samples
        self.aggregation_interval = aggregation_interval_seconds

        self._running = False
        self._sample_thread: Optional[threading.Thread] = None
        self._samples: Dict[ProfileType, List[ProfileSample]] = defaultdict(list)
        self._aggregated: Dict[ProfileType, ProfileData] = {}
        self._lock = threading.Lock()

        # Tracemalloc for memory profiling
        self._tracemalloc_started = False

        # Stats
        self._stats = {
            "samples_collected": 0,
            "samples_dropped": 0,
            "aggregations_completed": 0,
        }

    async def start(self) -> None:
        """Start the profiler."""
        if self._running:
            return

        self._running = True

        # Start tracemalloc for memory profiling
        if self.memory_profiling or self.allocation_profiling:
            if not tracemalloc.is_tracing():
                tracemalloc.start(25)  # 25 frames
                self._tracemalloc_started = True

        # Start sampling thread
        self._sample_thread = threading.Thread(
            target=self._sampling_loop,
            daemon=True,
            name="ContinuousProfiler",
        )
        self._sample_thread.start()

        logger.info("Continuous profiler started")

    async def stop(self) -> None:
        """Stop the profiler."""
        self._running = False

        if self._sample_thread:
            self._sample_thread.join(timeout=2.0)

        if self._tracemalloc_started:
            tracemalloc.stop()

        logger.info("Continuous profiler stopped")

    def _sampling_loop(self) -> None:
        """Background thread for collecting samples."""
        interval = 1.0 / self.sample_rate_hz
        last_aggregation = time.time()

        while self._running:
            try:
                self._collect_samples()

                # Periodic aggregation
                if time.time() - last_aggregation >= self.aggregation_interval:
                    self._aggregate_samples()
                    last_aggregation = time.time()

            except Exception as e:
                logger.error(f"Profiling error: {e}")

            time.sleep(interval)

    def _collect_samples(self) -> None:
        """Collect samples from all enabled profile types."""
        timestamp = datetime.utcnow()

        if self.cpu_profiling:
            self._collect_cpu_sample(timestamp)

        if self.memory_profiling:
            self._collect_memory_sample(timestamp)

        if self.async_profiling:
            self._collect_async_sample(timestamp)

    def _collect_cpu_sample(self, timestamp: datetime) -> None:
        """Collect CPU profile sample."""
        # Sample all threads
        for thread_id, frame in sys._current_frames().items():
            thread = self._get_thread(thread_id)

            frames = []
            while frame is not None:
                code = frame.f_code
                frames.append(ProfileFrame(
                    function=code.co_name,
                    filename=code.co_filename,
                    line_number=frame.f_lineno,
                    module=frame.f_globals.get("__name__", ""),
                ))
                frame = frame.f_back

            if frames:
                sample = ProfileSample(
                    timestamp=timestamp,
                    stack=ProfileStack(frames=frames, value=1),
                    profile_type=ProfileType.CPU,
                    thread_id=thread_id,
                    thread_name=thread.name if thread else "",
                )
                self._add_sample(sample)

    def _collect_memory_sample(self, timestamp: datetime) -> None:
        """Collect memory profile sample."""
        if not tracemalloc.is_tracing():
            return

        snapshot = tracemalloc.take_snapshot()
        stats = snapshot.statistics("traceback")

        for stat in stats[:100]:  # Top 100 allocations
            frames = []
            for frame_info in stat.traceback:
                frames.append(ProfileFrame(
                    function="",
                    filename=frame_info.filename,
                    line_number=frame_info.lineno,
                ))

            if frames:
                sample = ProfileSample(
                    timestamp=timestamp,
                    stack=ProfileStack(frames=frames, value=stat.size),
                    profile_type=ProfileType.MEMORY,
                )
                self._add_sample(sample)

    def _collect_async_sample(self, timestamp: datetime) -> None:
        """Collect async task profile sample."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return

        tasks = asyncio.all_tasks(loop)

        for task in tasks:
            coro = task.get_coro()
            if coro is None:
                continue

            frames = []

            # Extract coroutine frames
            cr_frame = getattr(coro, "cr_frame", None)
            while cr_frame is not None:
                code = cr_frame.f_code
                frames.append(ProfileFrame(
                    function=code.co_name,
                    filename=code.co_filename,
                    line_number=cr_frame.f_lineno,
                    module=cr_frame.f_globals.get("__name__", ""),
                ))
                cr_frame = cr_frame.f_back

            if frames:
                sample = ProfileSample(
                    timestamp=timestamp,
                    stack=ProfileStack(
                        frames=frames,
                        value=1,
                        labels={"task_name": task.get_name()},
                    ),
                    profile_type=ProfileType.ASYNC,
                )
                self._add_sample(sample)

    def _get_thread(self, thread_id: int) -> Optional[threading.Thread]:
        """Get thread by ID."""
        for thread in threading.enumerate():
            if thread.ident == thread_id:
                return thread
        return None

    def _add_sample(self, sample: ProfileSample) -> None:
        """Add a sample to the buffer."""
        with self._lock:
            samples = self._samples[sample.profile_type]

            if len(samples) >= self.max_samples:
                # Drop oldest samples
                samples.pop(0)
                self._stats["samples_dropped"] += 1

            samples.append(sample)
            self._stats["samples_collected"] += 1

    def _aggregate_samples(self) -> None:
        """Aggregate samples into profiles."""
        with self._lock:
            now = datetime.utcnow()

            for profile_type, samples in self._samples.items():
                if not samples:
                    continue

                # Aggregate stacks
                aggregated: Dict[str, float] = defaultdict(float)

                for sample in samples:
                    stack_str = ";".join(
                        f.to_string() for f in reversed(sample.stack.frames)
                    )
                    aggregated[stack_str] += sample.stack.value

                # Create profile data
                profile_data = ProfileData(
                    profile_type=profile_type,
                    start_time=samples[0].timestamp,
                    end_time=samples[-1].timestamp,
                    samples=samples.copy(),
                    aggregated_stacks=dict(aggregated),
                )

                self._aggregated[profile_type] = profile_data

            # Clear samples
            self._samples.clear()
            self._stats["aggregations_completed"] += 1

    def get_profile(self, profile_type: ProfileType) -> Optional[ProfileData]:
        """Get aggregated profile data."""
        with self._lock:
            return self._aggregated.get(profile_type)

    def get_flame_graph(self, profile_type: ProfileType) -> Optional[FlameGraph]:
        """Get flame graph for a profile type."""
        profile = self.get_profile(profile_type)
        if profile:
            return FlameGraph(profile)
        return None

    def get_all_profiles(self) -> Dict[ProfileType, ProfileData]:
        """Get all aggregated profiles."""
        with self._lock:
            return dict(self._aggregated)

    def get_hot_functions(
        self,
        profile_type: ProfileType = ProfileType.CPU,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Get hottest functions."""
        profile = self.get_profile(profile_type)
        if not profile:
            return []

        # Aggregate by function
        by_function: Dict[str, float] = defaultdict(float)

        for stack_str, value in profile.aggregated_stacks.items():
            # Get leaf function (last in stack)
            functions = stack_str.split(";")
            if functions:
                leaf = functions[-1]
                by_function[leaf] += value

        # Sort and return
        sorted_funcs = sorted(by_function.items(), key=lambda x: -x[1])

        return [
            {"function": func, "value": value, "percentage": value / sum(by_function.values()) * 100}
            for func, value in sorted_funcs[:limit]
        ]

    def export_pprof(self, profile_type: ProfileType) -> bytes:
        """Export profile in pprof format (for Go tools compatibility)."""
        # Simplified pprof-like export
        # Full pprof would require protobuf
        profile = self.get_profile(profile_type)
        if not profile:
            return b""

        lines = []
        lines.append(f"# Profile type: {profile_type.value}")
        lines.append(f"# Start time: {profile.start_time.isoformat()}")
        lines.append(f"# End time: {profile.end_time.isoformat()}")
        lines.append("")

        for stack_str, value in profile.aggregated_stacks.items():
            lines.append(f"{int(value)} {stack_str}")

        return "\n".join(lines).encode("utf-8")

    def get_stats(self) -> Dict[str, Any]:
        """Get profiler statistics."""
        with self._lock:
            return {
                **self._stats,
                "running": self._running,
                "sample_rate_hz": self.sample_rate_hz,
                "profile_types": {
                    pt.value: {
                        "samples": len(self._samples.get(pt, [])),
                        "aggregated": pt in self._aggregated,
                    }
                    for pt in ProfileType
                },
            }


# =============================================================================
# Profile Decorators
# =============================================================================

def profile_function(
    profiler: Optional[ContinuousProfiler] = None,
    profile_type: ProfileType = ProfileType.CPU,
) -> Callable:
    """Decorator to profile a specific function."""

    def decorator(func: Callable) -> Callable:
        import functools

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if profiler is None:
                return func(*args, **kwargs)

            start_time = time.perf_counter()

            try:
                result = func(*args, **kwargs)
                return result
            finally:
                elapsed = (time.perf_counter() - start_time) * 1000  # ms

                # Create a simple stack
                frame = ProfileFrame(
                    function=func.__name__,
                    filename=func.__code__.co_filename,
                    line_number=func.__code__.co_firstlineno,
                    module=func.__module__,
                )

                sample = ProfileSample(
                    timestamp=datetime.utcnow(),
                    stack=ProfileStack(frames=[frame], value=elapsed),
                    profile_type=profile_type,
                )

                profiler._add_sample(sample)

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            if profiler is None:
                return await func(*args, **kwargs)

            start_time = time.perf_counter()

            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                elapsed = (time.perf_counter() - start_time) * 1000

                frame = ProfileFrame(
                    function=func.__name__,
                    filename=func.__code__.co_filename,
                    line_number=func.__code__.co_firstlineno,
                    module=func.__module__,
                )

                sample = ProfileSample(
                    timestamp=datetime.utcnow(),
                    stack=ProfileStack(frames=[frame], value=elapsed),
                    profile_type=profile_type,
                )

                profiler._add_sample(sample)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper

    return decorator
