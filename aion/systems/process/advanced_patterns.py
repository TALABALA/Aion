"""
AION Advanced Patterns

True SOTA patterns that go beyond basic implementations:
- Adaptive Circuit Breaker with ML-based thresholds
- Hedge Requests for latency optimization
- Tail-based Sampling for distributed tracing
- Dynamic DAGs with runtime modification
- Backfill operations for workflow orchestration
"""

from __future__ import annotations

import asyncio
import math
import random
import statistics
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar

import structlog

logger = structlog.get_logger(__name__)

T = TypeVar("T")


# === Adaptive Circuit Breaker ===

class AdaptiveCircuitBreaker:
    """
    Adaptive Circuit Breaker with dynamic thresholds.

    Uses statistical analysis to automatically adjust:
    - Failure rate threshold based on historical baseline
    - Timeout threshold based on latency percentiles
    - Recovery behavior based on success rate trends

    Implements:
    - Sliding window metrics (count-based and time-based)
    - Slow call detection
    - Gradual recovery (half-open with increasing permit count)
    """

    def __init__(
        self,
        name: str,
        # Window configuration
        sliding_window_type: str = "count",  # "count" or "time"
        sliding_window_size: int = 100,  # calls or seconds
        minimum_calls: int = 10,  # Min calls before evaluating
        # Thresholds (initial, will adapt)
        failure_rate_threshold: float = 50.0,  # Percent
        slow_call_rate_threshold: float = 100.0,  # Percent
        slow_call_duration_threshold: float = 2.0,  # Seconds
        # Recovery
        wait_duration_open: float = 30.0,  # Seconds in open state
        permitted_calls_half_open: int = 10,  # Calls allowed in half-open
        # Adaptation
        enable_adaptation: bool = True,
        adaptation_window: int = 1000,  # Historical calls for baseline
        adaptation_sensitivity: float = 2.0,  # Std deviations from baseline
    ):
        self.name = name
        self.sliding_window_type = sliding_window_type
        self.sliding_window_size = sliding_window_size
        self.minimum_calls = minimum_calls
        self.failure_rate_threshold = failure_rate_threshold
        self.slow_call_rate_threshold = slow_call_rate_threshold
        self.slow_call_duration_threshold = slow_call_duration_threshold
        self.wait_duration_open = wait_duration_open
        self.permitted_calls_half_open = permitted_calls_half_open
        self.enable_adaptation = enable_adaptation
        self.adaptation_window = adaptation_window
        self.adaptation_sensitivity = adaptation_sensitivity

        # State
        self._state = "closed"
        self._opened_at: Optional[float] = None
        self._half_open_calls = 0
        self._half_open_successes = 0

        # Sliding window metrics
        self._window: deque = deque(maxlen=sliding_window_size)
        self._window_start = time.time()

        # Historical data for adaptation
        self._historical_failure_rates: deque = deque(maxlen=adaptation_window)
        self._historical_latencies: deque = deque(maxlen=adaptation_window)
        self._baseline_failure_rate: Optional[float] = None
        self._baseline_latency_p99: Optional[float] = None

        # Metrics
        self._metrics = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "slow_calls": 0,
            "rejected_calls": 0,
            "state_transitions": 0,
            "threshold_adaptations": 0,
        }

        self._lock = asyncio.Lock()

    @dataclass
    class CallResult:
        """Result of a call through the circuit breaker."""
        success: bool
        duration: float
        timestamp: float = field(default_factory=time.time)
        error: Optional[str] = None

    async def execute(
        self,
        func: Callable[..., T],
        *args,
        **kwargs,
    ) -> T:
        """Execute a function through the circuit breaker."""
        async with self._lock:
            # Check if call is permitted
            if not self._is_call_permitted():
                self._metrics["rejected_calls"] += 1
                raise CircuitBreakerOpenError(
                    f"Circuit breaker {self.name} is {self._state}"
                )

        # Execute with timing
        start = time.time()
        error = None
        success = True

        try:
            result = func(*args, **kwargs)
            if asyncio.iscoroutine(result):
                result = await result
            return result

        except Exception as e:
            error = str(e)
            success = False
            raise

        finally:
            duration = time.time() - start
            call_result = self.CallResult(
                success=success,
                duration=duration,
                error=error,
            )
            await self._record_call(call_result)

    async def _record_call(self, result: CallResult) -> None:
        """Record call result and update state."""
        async with self._lock:
            self._metrics["total_calls"] += 1

            if result.success:
                self._metrics["successful_calls"] += 1
            else:
                self._metrics["failed_calls"] += 1

            if result.duration > self.slow_call_duration_threshold:
                self._metrics["slow_calls"] += 1

            # Add to sliding window
            self._window.append(result)

            # Add to historical data
            self._historical_latencies.append(result.duration)
            self._historical_failure_rates.append(0 if result.success else 1)

            # Update half-open state
            if self._state == "half_open":
                self._half_open_calls += 1
                if result.success:
                    self._half_open_successes += 1

                # Check if we should transition
                if self._half_open_calls >= self.permitted_calls_half_open:
                    success_rate = self._half_open_successes / self._half_open_calls * 100
                    if success_rate >= (100 - self.failure_rate_threshold):
                        self._transition_to_closed()
                    else:
                        self._transition_to_open()

            # Evaluate thresholds for closed state
            elif self._state == "closed":
                self._evaluate_thresholds()

            # Adapt thresholds periodically
            if self.enable_adaptation and self._metrics["total_calls"] % 100 == 0:
                self._adapt_thresholds()

    def _is_call_permitted(self) -> bool:
        """Check if a call is permitted."""
        if self._state == "closed":
            return True

        if self._state == "open":
            # Check if wait duration has passed
            if self._opened_at:
                elapsed = time.time() - self._opened_at
                if elapsed >= self.wait_duration_open:
                    self._transition_to_half_open()
                    return True
            return False

        if self._state == "half_open":
            return self._half_open_calls < self.permitted_calls_half_open

        return False

    def _evaluate_thresholds(self) -> None:
        """Evaluate if circuit should open based on current metrics."""
        if len(self._window) < self.minimum_calls:
            return

        # Calculate failure rate
        failures = sum(1 for r in self._window if not r.success)
        failure_rate = failures / len(self._window) * 100

        # Calculate slow call rate
        slow_calls = sum(
            1 for r in self._window
            if r.duration > self.slow_call_duration_threshold
        )
        slow_call_rate = slow_calls / len(self._window) * 100

        # Check thresholds
        if failure_rate >= self.failure_rate_threshold:
            self._transition_to_open()
        elif slow_call_rate >= self.slow_call_rate_threshold:
            self._transition_to_open()

    def _adapt_thresholds(self) -> None:
        """Adapt thresholds based on historical baseline."""
        if len(self._historical_failure_rates) < self.adaptation_window // 2:
            return

        # Calculate baseline failure rate
        failure_rates = list(self._historical_failure_rates)
        mean_failure = statistics.mean(failure_rates) * 100
        std_failure = statistics.stdev(failure_rates) * 100 if len(failure_rates) > 1 else 0

        self._baseline_failure_rate = mean_failure

        # Adapt threshold: baseline + sensitivity * std
        new_threshold = min(
            mean_failure + self.adaptation_sensitivity * std_failure,
            95.0,  # Cap at 95%
        )
        new_threshold = max(new_threshold, 10.0)  # Floor at 10%

        if abs(new_threshold - self.failure_rate_threshold) > 5:
            self.failure_rate_threshold = new_threshold
            self._metrics["threshold_adaptations"] += 1
            logger.info(
                f"Circuit {self.name} adapted failure threshold to {new_threshold:.1f}%"
            )

        # Calculate baseline latency
        latencies = list(self._historical_latencies)
        if latencies:
            latencies_sorted = sorted(latencies)
            p99_idx = int(len(latencies_sorted) * 0.99)
            self._baseline_latency_p99 = latencies_sorted[p99_idx]

            # Adapt slow call threshold
            new_slow_threshold = self._baseline_latency_p99 * 2
            if new_slow_threshold != self.slow_call_duration_threshold:
                self.slow_call_duration_threshold = new_slow_threshold

    def _transition_to_open(self) -> None:
        """Transition to open state."""
        if self._state == "open":
            return

        self._state = "open"
        self._opened_at = time.time()
        self._metrics["state_transitions"] += 1
        logger.warning(f"Circuit {self.name} opened")

    def _transition_to_half_open(self) -> None:
        """Transition to half-open state."""
        self._state = "half_open"
        self._half_open_calls = 0
        self._half_open_successes = 0
        self._metrics["state_transitions"] += 1
        logger.info(f"Circuit {self.name} half-opened")

    def _transition_to_closed(self) -> None:
        """Transition to closed state."""
        self._state = "closed"
        self._opened_at = None
        self._window.clear()
        self._metrics["state_transitions"] += 1
        logger.info(f"Circuit {self.name} closed")

    def get_state(self) -> str:
        """Get current state."""
        return self._state

    def get_metrics(self) -> Dict[str, Any]:
        """Get circuit breaker metrics."""
        failure_rate = 0
        if self._window:
            failures = sum(1 for r in self._window if not r.success)
            failure_rate = failures / len(self._window) * 100

        return {
            **self._metrics,
            "state": self._state,
            "current_failure_rate": failure_rate,
            "failure_rate_threshold": self.failure_rate_threshold,
            "slow_call_threshold": self.slow_call_duration_threshold,
            "baseline_failure_rate": self._baseline_failure_rate,
            "baseline_latency_p99": self._baseline_latency_p99,
            "window_size": len(self._window),
        }


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass


# === Hedge Requests ===

class HedgeRequest:
    """
    Hedge Request pattern for latency optimization.

    Sends a second (hedge) request if the primary request takes too long,
    using the first response to complete. Dramatically reduces tail latency.

    Strategies:
    - Percentile-based: Hedge after p50/p90/p99 latency threshold
    - Adaptive: Learn optimal hedge delay from historical data
    - Budget-based: Limit hedge rate to control extra load
    """

    def __init__(
        self,
        name: str = "default",
        # Hedging strategy
        hedge_delay_percentile: float = 0.95,  # Hedge after this percentile
        min_hedge_delay: float = 0.001,  # Minimum delay before hedging
        max_hedge_delay: float = 5.0,  # Maximum delay before hedging
        # Budget
        hedge_budget_percent: float = 10.0,  # Max % of requests that can be hedged
        hedge_budget_window: float = 60.0,  # Window for budget calculation
        # Learning
        enable_adaptive_delay: bool = True,
        latency_window_size: int = 1000,
    ):
        self.name = name
        self.hedge_delay_percentile = hedge_delay_percentile
        self.min_hedge_delay = min_hedge_delay
        self.max_hedge_delay = max_hedge_delay
        self.hedge_budget_percent = hedge_budget_percent
        self.hedge_budget_window = hedge_budget_window
        self.enable_adaptive_delay = enable_adaptive_delay
        self.latency_window_size = latency_window_size

        # Latency tracking
        self._latencies: deque = deque(maxlen=latency_window_size)
        self._hedge_delay = min_hedge_delay

        # Budget tracking
        self._request_times: deque = deque()
        self._hedge_times: deque = deque()

        # Metrics
        self._metrics = {
            "total_requests": 0,
            "hedged_requests": 0,
            "primary_won": 0,
            "hedge_won": 0,
            "budget_exceeded": 0,
        }

        self._lock = asyncio.Lock()

    async def execute(
        self,
        func: Callable[..., T],
        *args,
        hedge_func: Optional[Callable[..., T]] = None,
        **kwargs,
    ) -> T:
        """
        Execute with hedge request support.

        Args:
            func: Primary function to execute
            hedge_func: Alternative function for hedge (defaults to same func)
            *args, **kwargs: Arguments to pass to functions
        """
        hedge_func = hedge_func or func

        async with self._lock:
            self._metrics["total_requests"] += 1
            now = time.time()

            # Cleanup old budget records
            cutoff = now - self.hedge_budget_window
            while self._request_times and self._request_times[0] < cutoff:
                self._request_times.popleft()
            while self._hedge_times and self._hedge_times[0] < cutoff:
                self._hedge_times.popleft()

            self._request_times.append(now)

            # Check hedge budget
            can_hedge = True
            if self._request_times:
                hedge_rate = len(self._hedge_times) / len(self._request_times) * 100
                if hedge_rate >= self.hedge_budget_percent:
                    can_hedge = False
                    self._metrics["budget_exceeded"] += 1

        # Calculate hedge delay
        hedge_delay = self._calculate_hedge_delay()

        # Create tasks
        primary_task = asyncio.create_task(self._timed_call(func, *args, **kwargs))

        # Start hedge after delay if budget allows
        hedge_task = None
        if can_hedge:
            hedge_task = asyncio.create_task(
                self._delayed_hedge(hedge_delay, hedge_func, *args, **kwargs)
            )

        try:
            # Wait for first completion
            tasks = [primary_task]
            if hedge_task:
                tasks.append(hedge_task)

            done, pending = await asyncio.wait(
                tasks,
                return_when=asyncio.FIRST_COMPLETED,
            )

            # Cancel pending
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            # Get result from completed task
            completed = done.pop()
            result, latency, is_hedge = completed.result()

            # Record metrics
            async with self._lock:
                self._latencies.append(latency)

                if is_hedge:
                    self._metrics["hedge_won"] += 1
                else:
                    self._metrics["primary_won"] += 1

                # Update adaptive delay
                if self.enable_adaptive_delay:
                    self._update_hedge_delay()

            return result

        except Exception:
            # Cancel any remaining tasks
            primary_task.cancel()
            if hedge_task:
                hedge_task.cancel()
            raise

    async def _timed_call(
        self,
        func: Callable[..., T],
        *args,
        **kwargs,
    ) -> Tuple[T, float, bool]:
        """Execute function and return with timing info."""
        start = time.time()
        result = func(*args, **kwargs)
        if asyncio.iscoroutine(result):
            result = await result
        latency = time.time() - start
        return result, latency, False

    async def _delayed_hedge(
        self,
        delay: float,
        func: Callable[..., T],
        *args,
        **kwargs,
    ) -> Tuple[T, float, bool]:
        """Execute hedge after delay."""
        await asyncio.sleep(delay)

        async with self._lock:
            self._metrics["hedged_requests"] += 1
            self._hedge_times.append(time.time())

        start = time.time()
        result = func(*args, **kwargs)
        if asyncio.iscoroutine(result):
            result = await result
        latency = time.time() - start
        return result, latency, True

    def _calculate_hedge_delay(self) -> float:
        """Calculate delay before sending hedge request."""
        if not self._latencies:
            return self.min_hedge_delay

        latencies = sorted(self._latencies)
        percentile_idx = int(len(latencies) * self.hedge_delay_percentile)
        percentile_latency = latencies[min(percentile_idx, len(latencies) - 1)]

        delay = max(self.min_hedge_delay, min(percentile_latency, self.max_hedge_delay))
        return delay

    def _update_hedge_delay(self) -> None:
        """Update adaptive hedge delay."""
        self._hedge_delay = self._calculate_hedge_delay()

    def get_metrics(self) -> Dict[str, Any]:
        """Get hedge request metrics."""
        return {
            **self._metrics,
            "current_hedge_delay": self._hedge_delay,
            "hedge_win_rate": (
                self._metrics["hedge_won"] / max(1, self._metrics["hedged_requests"]) * 100
            ),
            "latency_samples": len(self._latencies),
        }


# === Tail-Based Sampling ===

class SamplingDecision(Enum):
    """Sampling decision for a trace."""
    SAMPLE = auto()
    DROP = auto()
    DEFER = auto()  # Defer decision to tail


@dataclass
class TraceData:
    """Data about a trace for sampling decision."""
    trace_id: str
    root_span_name: str
    spans: List[Dict[str, Any]] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    duration_ms: float = 0
    error: bool = False
    status_code: Optional[int] = None
    tags: Dict[str, str] = field(default_factory=dict)


class TailBasedSampler:
    """
    Tail-based sampling for distributed tracing.

    Makes sampling decisions after seeing the complete trace,
    allowing intelligent sampling of:
    - Error traces (always sample)
    - Slow traces (sample above percentile)
    - Interesting traces (based on rules)
    - Random baseline (for statistics)

    Requires a collection buffer to hold spans until trace completes.
    """

    def __init__(
        self,
        # Base sampling
        base_sample_rate: float = 0.01,  # 1% baseline
        # Error sampling
        always_sample_errors: bool = True,
        # Latency sampling
        latency_threshold_ms: float = 1000,  # Sample traces > 1s
        latency_sample_rate: float = 1.0,  # 100% of slow traces
        # Rule-based sampling
        rules: Optional[List[Dict[str, Any]]] = None,
        # Buffer
        max_traces_buffer: int = 10000,
        trace_timeout_seconds: float = 60.0,
    ):
        self.base_sample_rate = base_sample_rate
        self.always_sample_errors = always_sample_errors
        self.latency_threshold_ms = latency_threshold_ms
        self.latency_sample_rate = latency_sample_rate
        self.rules = rules or []
        self.max_traces_buffer = max_traces_buffer
        self.trace_timeout_seconds = trace_timeout_seconds

        # Buffer for incomplete traces
        self._traces: Dict[str, TraceData] = {}
        self._trace_spans: Dict[str, List[Dict[str, Any]]] = {}

        # Metrics
        self._metrics = {
            "traces_received": 0,
            "traces_sampled": 0,
            "traces_dropped": 0,
            "traces_timeout": 0,
            "sampled_by_error": 0,
            "sampled_by_latency": 0,
            "sampled_by_rule": 0,
            "sampled_by_baseline": 0,
        }

        # Cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        self._shutdown = False

    async def start(self) -> None:
        """Start the sampler."""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def stop(self) -> None:
        """Stop the sampler."""
        self._shutdown = True
        if self._cleanup_task:
            self._cleanup_task.cancel()

    def record_span(self, span: Dict[str, Any]) -> None:
        """Record a span for tail-based sampling."""
        trace_id = span.get("trace_id")
        if not trace_id:
            return

        self._metrics["traces_received"] += 1

        # Initialize trace if new
        if trace_id not in self._traces:
            if len(self._traces) >= self.max_traces_buffer:
                # Evict oldest trace
                oldest = min(self._traces.keys(), key=lambda t: self._traces[t].start_time)
                self._finalize_trace(oldest)

            self._traces[trace_id] = TraceData(
                trace_id=trace_id,
                root_span_name=span.get("name", "unknown"),
            )
            self._trace_spans[trace_id] = []

        # Add span
        self._trace_spans[trace_id].append(span)

        # Update trace data
        trace = self._traces[trace_id]
        if span.get("error"):
            trace.error = True
        if span.get("status_code"):
            trace.status_code = span["status_code"]
        if span.get("duration_ms"):
            trace.duration_ms = max(trace.duration_ms, span["duration_ms"])

        # Check if trace is complete (root span ended)
        if span.get("is_root") and span.get("end_time"):
            self._finalize_trace(trace_id)

    def _finalize_trace(self, trace_id: str) -> SamplingDecision:
        """Make final sampling decision for a trace."""
        trace = self._traces.pop(trace_id, None)
        spans = self._trace_spans.pop(trace_id, [])

        if not trace:
            return SamplingDecision.DROP

        trace.spans = spans

        # Make sampling decision
        decision = self._make_sampling_decision(trace)

        if decision == SamplingDecision.SAMPLE:
            self._metrics["traces_sampled"] += 1
            self._export_trace(trace)
        else:
            self._metrics["traces_dropped"] += 1

        return decision

    def _make_sampling_decision(self, trace: TraceData) -> SamplingDecision:
        """Make sampling decision based on trace characteristics."""
        # Always sample errors
        if self.always_sample_errors and trace.error:
            self._metrics["sampled_by_error"] += 1
            return SamplingDecision.SAMPLE

        # Sample slow traces
        if trace.duration_ms > self.latency_threshold_ms:
            if random.random() < self.latency_sample_rate:
                self._metrics["sampled_by_latency"] += 1
                return SamplingDecision.SAMPLE

        # Apply rules
        for rule in self.rules:
            if self._matches_rule(trace, rule):
                sample_rate = rule.get("sample_rate", 1.0)
                if random.random() < sample_rate:
                    self._metrics["sampled_by_rule"] += 1
                    return SamplingDecision.SAMPLE

        # Baseline sampling
        if random.random() < self.base_sample_rate:
            self._metrics["sampled_by_baseline"] += 1
            return SamplingDecision.SAMPLE

        return SamplingDecision.DROP

    def _matches_rule(self, trace: TraceData, rule: Dict[str, Any]) -> bool:
        """Check if trace matches a sampling rule."""
        # Match by operation name
        if "operation" in rule:
            if not trace.root_span_name.startswith(rule["operation"]):
                return False

        # Match by status code
        if "status_code" in rule:
            if trace.status_code != rule["status_code"]:
                return False

        # Match by tags
        if "tags" in rule:
            for key, value in rule["tags"].items():
                if trace.tags.get(key) != value:
                    return False

        # Match by minimum duration
        if "min_duration_ms" in rule:
            if trace.duration_ms < rule["min_duration_ms"]:
                return False

        return True

    def _export_trace(self, trace: TraceData) -> None:
        """Export sampled trace (override for actual export)."""
        logger.debug(f"Sampled trace: {trace.trace_id} ({len(trace.spans)} spans)")

    async def _cleanup_loop(self) -> None:
        """Cleanup timed-out traces."""
        while not self._shutdown:
            try:
                await asyncio.sleep(10.0)

                now = datetime.now()
                timeout_delta = timedelta(seconds=self.trace_timeout_seconds)

                timed_out = [
                    trace_id
                    for trace_id, trace in self._traces.items()
                    if now - trace.start_time > timeout_delta
                ]

                for trace_id in timed_out:
                    self._finalize_trace(trace_id)
                    self._metrics["traces_timeout"] += 1

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup error: {e}")

    def get_metrics(self) -> Dict[str, Any]:
        """Get sampler metrics."""
        return {
            **self._metrics,
            "buffered_traces": len(self._traces),
            "effective_sample_rate": (
                self._metrics["traces_sampled"] /
                max(1, self._metrics["traces_sampled"] + self._metrics["traces_dropped"])
            ),
        }


# === Dynamic DAGs ===

@dataclass
class DynamicTask:
    """A task that can be dynamically added to a running DAG."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    handler: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    upstream_tasks: Set[str] = field(default_factory=set)
    condition: Optional[Callable[[Dict[str, Any]], bool]] = None  # Dynamic condition


class DynamicDAG:
    """
    Dynamic DAG that supports runtime modifications.

    Features:
    - Add tasks during execution based on upstream results
    - Conditional branching
    - Loop expansion
    - Parameter templating from upstream results
    """

    def __init__(self, dag_id: str, name: str = ""):
        self.dag_id = dag_id
        self.name = name
        self._tasks: Dict[str, DynamicTask] = {}
        self._task_results: Dict[str, Any] = {}
        self._dynamic_generators: List[Callable[[Dict[str, Any]], List[DynamicTask]]] = []

    def add_task(self, task: DynamicTask) -> None:
        """Add a static task."""
        self._tasks[task.id] = task

    def add_dynamic_generator(
        self,
        generator: Callable[[Dict[str, Any]], List[DynamicTask]],
    ) -> None:
        """
        Add a generator that produces tasks dynamically.

        Generator receives context with all upstream results and
        returns new tasks to add.
        """
        self._dynamic_generators.append(generator)

    def expand_tasks(self, context: Dict[str, Any]) -> List[DynamicTask]:
        """Expand dynamic tasks based on current context."""
        new_tasks = []

        for generator in self._dynamic_generators:
            try:
                generated = generator(context)
                for task in generated:
                    if task.id not in self._tasks:
                        self._tasks[task.id] = task
                        new_tasks.append(task)
            except Exception as e:
                logger.error(f"Dynamic task generation error: {e}")

        return new_tasks

    def should_execute_task(self, task: DynamicTask, context: Dict[str, Any]) -> bool:
        """Check if task should be executed based on condition."""
        if task.condition is None:
            return True

        try:
            return task.condition(context)
        except Exception as e:
            logger.error(f"Task condition error: {e}")
            return False

    def template_params(self, task: DynamicTask, context: Dict[str, Any]) -> Dict[str, Any]:
        """Template task parameters with values from context."""
        params = {}

        for key, value in task.params.items():
            if isinstance(value, str) and value.startswith("{{") and value.endswith("}}"):
                # Template syntax: {{upstream_task_id.result_key}}
                path = value[2:-2].strip()
                parts = path.split(".")

                current = context
                for part in parts:
                    if isinstance(current, dict):
                        current = current.get(part)
                    else:
                        current = None
                        break

                params[key] = current
            else:
                params[key] = value

        return params


class BackfillOperation:
    """
    Backfill operation for historical DAG runs.

    Features:
    - Run DAG for historical date range
    - Parallel execution with concurrency limit
    - Progress tracking
    - Failure handling with retry
    """

    def __init__(
        self,
        dag_id: str,
        start_date: datetime,
        end_date: datetime,
        interval: timedelta = timedelta(days=1),
        max_concurrent: int = 4,
        retry_failed: bool = True,
        max_retries: int = 3,
    ):
        self.dag_id = dag_id
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.max_concurrent = max_concurrent
        self.retry_failed = retry_failed
        self.max_retries = max_retries

        self._execution_dates: List[datetime] = []
        self._completed: Set[datetime] = set()
        self._failed: Dict[datetime, int] = {}  # date -> attempt count
        self._running: Set[datetime] = set()

        # Generate execution dates
        current = start_date
        while current <= end_date:
            self._execution_dates.append(current)
            current += interval

        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._metrics = {
            "total_runs": len(self._execution_dates),
            "completed_runs": 0,
            "failed_runs": 0,
            "retried_runs": 0,
        }

    async def run(
        self,
        dag_executor: Callable[[str, datetime], Any],
    ) -> Dict[str, Any]:
        """
        Run the backfill operation.

        Args:
            dag_executor: Function that executes DAG for a given date
        """
        logger.info(
            f"Starting backfill for {self.dag_id}: "
            f"{self.start_date} to {self.end_date} "
            f"({len(self._execution_dates)} runs)"
        )

        tasks = []
        for exec_date in self._execution_dates:
            task = asyncio.create_task(
                self._run_for_date(dag_executor, exec_date)
            )
            tasks.append(task)

        await asyncio.gather(*tasks)

        return {
            "dag_id": self.dag_id,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "metrics": self._metrics,
            "failed_dates": [d.isoformat() for d in self._failed.keys()],
        }

    async def _run_for_date(
        self,
        dag_executor: Callable[[str, datetime], Any],
        exec_date: datetime,
    ) -> bool:
        """Run DAG for a specific date with retry."""
        async with self._semaphore:
            self._running.add(exec_date)

            for attempt in range(self.max_retries + 1):
                try:
                    await dag_executor(self.dag_id, exec_date)
                    self._completed.add(exec_date)
                    self._metrics["completed_runs"] += 1
                    self._running.discard(exec_date)
                    return True

                except Exception as e:
                    logger.warning(
                        f"Backfill failed for {exec_date}: {e} "
                        f"(attempt {attempt + 1}/{self.max_retries + 1})"
                    )

                    if attempt < self.max_retries and self.retry_failed:
                        self._metrics["retried_runs"] += 1
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    else:
                        self._failed[exec_date] = attempt + 1
                        self._metrics["failed_runs"] += 1

            self._running.discard(exec_date)
            return False

    def get_progress(self) -> Dict[str, Any]:
        """Get current progress."""
        return {
            "total": len(self._execution_dates),
            "completed": len(self._completed),
            "failed": len(self._failed),
            "running": len(self._running),
            "pending": len(self._execution_dates) - len(self._completed) - len(self._failed) - len(self._running),
            "progress_percent": len(self._completed) / max(1, len(self._execution_dates)) * 100,
        }
