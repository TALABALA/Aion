"""
Advanced Tracing Features

SOTA tracing capabilities:
- Tail-based sampling (decisions after seeing full trace)
- Automatic instrumentation for httpx, aiohttp, databases
- Exemplars (linking metrics to traces)
- Trace analytics and pattern detection
"""

from __future__ import annotations

import asyncio
import functools
import logging
import time
import weakref
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)
from enum import Enum

from aion.observability.types import Span, SpanKind, SpanContext, Trace
from aion.observability.context import (
    get_current_span,
    get_current_trace_id,
    get_request_id,
)

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


# =============================================================================
# Tail-Based Sampling
# =============================================================================

class TailSamplingDecision(Enum):
    """Tail sampling decision types."""
    SAMPLE = "sample"
    DROP = "drop"
    DEFER = "defer"  # Wait for more spans


@dataclass
class TailSamplingPolicy:
    """Policy for tail-based sampling."""
    name: str
    description: str = ""
    enabled: bool = True
    priority: int = 0  # Higher priority evaluated first

    @abstractmethod
    def evaluate(self, trace: Trace) -> TailSamplingDecision:
        """Evaluate if trace should be sampled."""
        pass


@dataclass
class ErrorBasedPolicy(TailSamplingPolicy):
    """Sample traces that contain errors."""
    name: str = "error_based"
    sample_all_errors: bool = True
    error_status_codes: Set[int] = field(default_factory=lambda: {500, 502, 503, 504})

    def evaluate(self, trace: Trace) -> TailSamplingDecision:
        if not self.enabled:
            return TailSamplingDecision.DEFER

        if trace.has_errors:
            return TailSamplingDecision.SAMPLE

        # Check for error status codes in HTTP spans
        for span in trace.spans:
            status_code = span.attributes.get("http.status_code")
            if status_code and int(status_code) in self.error_status_codes:
                return TailSamplingDecision.SAMPLE

        return TailSamplingDecision.DEFER


@dataclass
class LatencyBasedPolicy(TailSamplingPolicy):
    """Sample traces that exceed latency threshold."""
    name: str = "latency_based"
    latency_threshold_ms: float = 1000.0
    percentile_threshold: float = 0.0  # 0 = use absolute, >0 = use percentile

    def evaluate(self, trace: Trace) -> TailSamplingDecision:
        if not self.enabled:
            return TailSamplingDecision.DEFER

        if trace.duration_ms and trace.duration_ms > self.latency_threshold_ms:
            return TailSamplingDecision.SAMPLE

        return TailSamplingDecision.DEFER


@dataclass
class StatusCodePolicy(TailSamplingPolicy):
    """Sample based on HTTP status codes."""
    name: str = "status_code"
    sample_codes: Set[int] = field(default_factory=lambda: {400, 401, 403, 404, 429, 500, 502, 503})

    def evaluate(self, trace: Trace) -> TailSamplingDecision:
        if not self.enabled:
            return TailSamplingDecision.DEFER

        for span in trace.spans:
            status_code = span.attributes.get("http.status_code")
            if status_code and int(status_code) in self.sample_codes:
                return TailSamplingDecision.SAMPLE

        return TailSamplingDecision.DEFER


@dataclass
class AttributeBasedPolicy(TailSamplingPolicy):
    """Sample based on span attributes."""
    name: str = "attribute_based"
    required_attributes: Dict[str, Any] = field(default_factory=dict)
    forbidden_attributes: Dict[str, Any] = field(default_factory=dict)

    def evaluate(self, trace: Trace) -> TailSamplingDecision:
        if not self.enabled:
            return TailSamplingDecision.DEFER

        for span in trace.spans:
            # Check required attributes
            if self.required_attributes:
                for key, value in self.required_attributes.items():
                    if span.attributes.get(key) == value:
                        return TailSamplingDecision.SAMPLE

            # Check forbidden attributes (inverse)
            if self.forbidden_attributes:
                for key, value in self.forbidden_attributes.items():
                    if span.attributes.get(key) == value:
                        return TailSamplingDecision.DROP

        return TailSamplingDecision.DEFER


@dataclass
class RateLimitingPolicy(TailSamplingPolicy):
    """Rate-limit sampling to N traces per second."""
    name: str = "rate_limiting"
    max_traces_per_second: float = 100.0

    _count: int = field(default=0, repr=False)
    _window_start: float = field(default_factory=time.time, repr=False)

    def evaluate(self, trace: Trace) -> TailSamplingDecision:
        if not self.enabled:
            return TailSamplingDecision.DEFER

        now = time.time()

        # Reset window
        if now - self._window_start >= 1.0:
            self._count = 0
            self._window_start = now

        if self._count < self.max_traces_per_second:
            self._count += 1
            return TailSamplingDecision.SAMPLE

        return TailSamplingDecision.DROP


@dataclass
class ProbabilisticPolicy(TailSamplingPolicy):
    """Probabilistic sampling as fallback."""
    name: str = "probabilistic"
    sampling_rate: float = 0.1  # 10% default

    def evaluate(self, trace: Trace) -> TailSamplingDecision:
        if not self.enabled:
            return TailSamplingDecision.DEFER

        import random
        if random.random() < self.sampling_rate:
            return TailSamplingDecision.SAMPLE

        return TailSamplingDecision.DROP


@dataclass
class CompositePolicy(TailSamplingPolicy):
    """Combine multiple policies."""
    name: str = "composite"
    policies: List[TailSamplingPolicy] = field(default_factory=list)
    require_all: bool = False  # AND vs OR logic

    def evaluate(self, trace: Trace) -> TailSamplingDecision:
        if not self.enabled or not self.policies:
            return TailSamplingDecision.DEFER

        decisions = [p.evaluate(trace) for p in self.policies]

        if self.require_all:
            # AND: All must sample
            if all(d == TailSamplingDecision.SAMPLE for d in decisions):
                return TailSamplingDecision.SAMPLE
            if any(d == TailSamplingDecision.DROP for d in decisions):
                return TailSamplingDecision.DROP
        else:
            # OR: Any can sample
            if any(d == TailSamplingDecision.SAMPLE for d in decisions):
                return TailSamplingDecision.SAMPLE
            if all(d == TailSamplingDecision.DROP for d in decisions):
                return TailSamplingDecision.DROP

        return TailSamplingDecision.DEFER


class TailBasedSampler:
    """
    Tail-based sampler that makes sampling decisions after seeing complete traces.

    Unlike head-based sampling which decides at trace start, tail-based sampling
    can consider the entire trace including errors, latency, and span attributes.
    """

    def __init__(
        self,
        policies: Optional[List[TailSamplingPolicy]] = None,
        decision_wait_time: float = 30.0,  # seconds to wait for trace completion
        max_pending_traces: int = 10000,
        default_decision: TailSamplingDecision = TailSamplingDecision.DROP,
    ):
        if policies is None:
            # Default policies
            policies = [
                ErrorBasedPolicy(priority=100),
                LatencyBasedPolicy(priority=90, latency_threshold_ms=1000),
                StatusCodePolicy(priority=80),
                RateLimitingPolicy(priority=10, max_traces_per_second=100),
                ProbabilisticPolicy(priority=0, sampling_rate=0.01),
            ]

        # Sort by priority (descending)
        self.policies = sorted(policies, key=lambda p: p.priority, reverse=True)
        self.decision_wait_time = decision_wait_time
        self.max_pending_traces = max_pending_traces
        self.default_decision = default_decision

        self._pending_traces: Dict[str, Tuple[Trace, float]] = {}
        self._sampled_traces: Set[str] = set()
        self._dropped_traces: Set[str] = set()
        self._decision_callbacks: Dict[str, List[Callable]] = defaultdict(list)

        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False

        # Stats
        self._stats = {
            "traces_evaluated": 0,
            "traces_sampled": 0,
            "traces_dropped": 0,
            "policy_hits": defaultdict(int),
        }

    async def start(self) -> None:
        """Start the sampler background tasks."""
        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def stop(self) -> None:
        """Stop the sampler."""
        self._running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

    async def _cleanup_loop(self) -> None:
        """Periodically clean up old pending traces."""
        while self._running:
            await asyncio.sleep(5.0)
            await self._cleanup_expired()

    async def _cleanup_expired(self) -> None:
        """Clean up traces that have exceeded wait time."""
        now = time.time()
        expired = []

        for trace_id, (trace, timestamp) in self._pending_traces.items():
            if now - timestamp > self.decision_wait_time:
                expired.append(trace_id)

        for trace_id in expired:
            trace, _ = self._pending_traces.pop(trace_id)
            await self._make_final_decision(trace)

    def add_span(self, span: Span) -> None:
        """Add a span to its trace for evaluation."""
        trace_id = span.trace_id

        # Already decided?
        if trace_id in self._sampled_traces or trace_id in self._dropped_traces:
            return

        # Get or create pending trace
        if trace_id not in self._pending_traces:
            if len(self._pending_traces) >= self.max_pending_traces:
                # Evict oldest
                oldest = min(self._pending_traces.items(), key=lambda x: x[1][1])
                del self._pending_traces[oldest[0]]

            trace = Trace(trace_id=trace_id)
            self._pending_traces[trace_id] = (trace, time.time())
        else:
            trace, timestamp = self._pending_traces[trace_id]

        trace.spans.append(span)

    async def on_trace_complete(self, trace_id: str) -> TailSamplingDecision:
        """Called when a trace is complete to make final decision."""
        if trace_id in self._sampled_traces:
            return TailSamplingDecision.SAMPLE
        if trace_id in self._dropped_traces:
            return TailSamplingDecision.DROP

        if trace_id in self._pending_traces:
            trace, _ = self._pending_traces.pop(trace_id)
            return await self._make_final_decision(trace)

        return self.default_decision

    async def _make_final_decision(self, trace: Trace) -> TailSamplingDecision:
        """Make final sampling decision for a trace."""
        self._stats["traces_evaluated"] += 1

        for policy in self.policies:
            decision = policy.evaluate(trace)

            if decision == TailSamplingDecision.SAMPLE:
                self._sampled_traces.add(trace.trace_id)
                self._stats["traces_sampled"] += 1
                self._stats["policy_hits"][policy.name] += 1

                # Notify callbacks
                await self._notify_decision(trace.trace_id, decision, trace)
                return decision

            elif decision == TailSamplingDecision.DROP:
                self._dropped_traces.add(trace.trace_id)
                self._stats["traces_dropped"] += 1
                self._stats["policy_hits"][policy.name] += 1

                await self._notify_decision(trace.trace_id, decision, trace)
                return decision

        # No policy made a decision, use default
        if self.default_decision == TailSamplingDecision.SAMPLE:
            self._sampled_traces.add(trace.trace_id)
            self._stats["traces_sampled"] += 1
        else:
            self._dropped_traces.add(trace.trace_id)
            self._stats["traces_dropped"] += 1

        await self._notify_decision(trace.trace_id, self.default_decision, trace)
        return self.default_decision

    async def _notify_decision(
        self,
        trace_id: str,
        decision: TailSamplingDecision,
        trace: Trace,
    ) -> None:
        """Notify registered callbacks of sampling decision."""
        callbacks = self._decision_callbacks.get(trace_id, [])
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(decision, trace)
                else:
                    callback(decision, trace)
            except Exception as e:
                logger.error(f"Error in sampling callback: {e}")

        # Clean up callbacks
        self._decision_callbacks.pop(trace_id, None)

    def register_callback(
        self,
        trace_id: str,
        callback: Callable[[TailSamplingDecision, Trace], Any],
    ) -> None:
        """Register a callback for when sampling decision is made."""
        self._decision_callbacks[trace_id].append(callback)

    def is_sampled(self, trace_id: str) -> Optional[bool]:
        """Check if a trace has been sampled (None if pending)."""
        if trace_id in self._sampled_traces:
            return True
        if trace_id in self._dropped_traces:
            return False
        return None

    def get_stats(self) -> Dict[str, Any]:
        """Get sampler statistics."""
        return {
            **self._stats,
            "policy_hits": dict(self._stats["policy_hits"]),
            "pending_traces": len(self._pending_traces),
            "policies": [p.name for p in self.policies],
        }


# =============================================================================
# Exemplars (Metrics → Traces Linking)
# =============================================================================

@dataclass
class Exemplar:
    """
    An exemplar links a metric observation to a trace.

    This enables drill-down from metrics to specific traces
    that contributed to the metric value.
    """
    value: float
    timestamp: datetime
    trace_id: str
    span_id: str
    labels: Dict[str, str] = field(default_factory=dict)
    filtered_labels: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "labels": self.labels,
            "filtered_labels": self.filtered_labels,
        }


class ExemplarReservoir:
    """
    Reservoir sampling for exemplars.

    Maintains a fixed-size sample of exemplars that is
    representative of the underlying distribution.
    """

    def __init__(
        self,
        max_size: int = 10,
        min_interval_ms: float = 100.0,
    ):
        self.max_size = max_size
        self.min_interval_ms = min_interval_ms

        self._exemplars: List[Exemplar] = []
        self._count: int = 0
        self._last_added: float = 0

    def offer(
        self,
        value: float,
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None,
        labels: Optional[Dict[str, str]] = None,
    ) -> bool:
        """
        Offer a new exemplar to the reservoir.

        Returns True if the exemplar was added.
        """
        now = time.time() * 1000

        # Rate limiting
        if now - self._last_added < self.min_interval_ms:
            return False

        # Need trace context
        if trace_id is None:
            trace_id = get_current_trace_id()
        if span_id is None:
            span = get_current_span()
            span_id = span.span_id if span else None

        if not trace_id or not span_id:
            return False

        exemplar = Exemplar(
            value=value,
            timestamp=datetime.utcnow(),
            trace_id=trace_id,
            span_id=span_id,
            labels=labels or {},
        )

        self._count += 1

        if len(self._exemplars) < self.max_size:
            self._exemplars.append(exemplar)
            self._last_added = now
            return True

        # Reservoir sampling
        import random
        j = random.randint(0, self._count - 1)
        if j < self.max_size:
            self._exemplars[j] = exemplar
            self._last_added = now
            return True

        return False

    def collect(self) -> List[Exemplar]:
        """Collect and reset exemplars."""
        exemplars = self._exemplars
        self._exemplars = []
        self._count = 0
        return exemplars

    def get_exemplars(self) -> List[Exemplar]:
        """Get current exemplars without resetting."""
        return list(self._exemplars)


class ExemplarStore:
    """
    Store for exemplars associated with metrics.

    Provides efficient storage and retrieval of exemplars
    by metric name, labels, and time range.
    """

    def __init__(self, max_exemplars_per_metric: int = 100):
        self.max_exemplars_per_metric = max_exemplars_per_metric
        self._reservoirs: Dict[str, ExemplarReservoir] = {}
        self._stored_exemplars: Dict[str, List[Exemplar]] = defaultdict(list)

    def get_reservoir(self, metric_key: str) -> ExemplarReservoir:
        """Get or create reservoir for a metric."""
        if metric_key not in self._reservoirs:
            self._reservoirs[metric_key] = ExemplarReservoir()
        return self._reservoirs[metric_key]

    def add_exemplar(
        self,
        metric_name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> bool:
        """Add an exemplar for a metric observation."""
        key = self._make_key(metric_name, labels or {})
        reservoir = self.get_reservoir(key)
        return reservoir.offer(value, labels=labels)

    def collect_exemplars(self, metric_name: str) -> List[Exemplar]:
        """Collect exemplars for a metric (used during scrape)."""
        exemplars = []
        prefix = f"{metric_name}:"

        for key, reservoir in self._reservoirs.items():
            if key.startswith(prefix) or key == metric_name:
                collected = reservoir.collect()
                exemplars.extend(collected)

                # Store for later retrieval
                self._stored_exemplars[key].extend(collected)
                # Trim old exemplars
                self._stored_exemplars[key] = \
                    self._stored_exemplars[key][-self.max_exemplars_per_metric:]

        return exemplars

    def get_exemplars(
        self,
        metric_name: str,
        labels: Optional[Dict[str, str]] = None,
        limit: int = 10,
    ) -> List[Exemplar]:
        """Get stored exemplars for a metric."""
        key = self._make_key(metric_name, labels or {})

        if key in self._stored_exemplars:
            return self._stored_exemplars[key][-limit:]

        # Try prefix match
        result = []
        for k, exemplars in self._stored_exemplars.items():
            if k.startswith(f"{metric_name}:"):
                result.extend(exemplars)

        return sorted(result, key=lambda e: e.timestamp)[-limit:]

    def get_trace_ids_for_metric(
        self,
        metric_name: str,
        labels: Optional[Dict[str, str]] = None,
    ) -> List[str]:
        """Get trace IDs associated with a metric."""
        exemplars = self.get_exemplars(metric_name, labels)
        return list(set(e.trace_id for e in exemplars))

    def _make_key(self, metric_name: str, labels: Dict[str, str]) -> str:
        """Create a unique key for metric + labels."""
        if not labels:
            return metric_name
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{metric_name}:{label_str}"


# =============================================================================
# Automatic Instrumentation
# =============================================================================

class InstrumentationConfig:
    """Configuration for automatic instrumentation."""

    def __init__(
        self,
        enabled: bool = True,
        capture_headers: bool = True,
        capture_body: bool = False,
        max_body_size: int = 1024,
        excluded_urls: Optional[List[str]] = None,
        excluded_methods: Optional[List[str]] = None,
    ):
        self.enabled = enabled
        self.capture_headers = capture_headers
        self.capture_body = capture_body
        self.max_body_size = max_body_size
        self.excluded_urls = excluded_urls or []
        self.excluded_methods = excluded_methods or []


class HTTPClientInstrumentor:
    """
    Automatic instrumentation for HTTP clients (httpx, aiohttp, requests).

    Wraps HTTP client methods to automatically create spans for requests.
    """

    def __init__(
        self,
        tracing_engine: Any,
        config: Optional[InstrumentationConfig] = None,
    ):
        self.tracing_engine = tracing_engine
        self.config = config or InstrumentationConfig()
        self._original_methods: Dict[str, Any] = {}
        self._instrumented = False

    def instrument_httpx(self) -> bool:
        """Instrument httpx library."""
        try:
            import httpx

            original_send = httpx.AsyncClient.send

            async def instrumented_send(self_client, request, *args, **kwargs):
                if not self.config.enabled:
                    return await original_send(self_client, request, *args, **kwargs)

                url = str(request.url)
                method = request.method

                # Check exclusions
                for pattern in self.config.excluded_urls:
                    if pattern in url:
                        return await original_send(self_client, request, *args, **kwargs)

                if method in self.config.excluded_methods:
                    return await original_send(self_client, request, *args, **kwargs)

                span_name = f"HTTP {method} {request.url.host}{request.url.path}"

                with self.tracing_engine.trace(span_name, kind=SpanKind.CLIENT) as span:
                    # Set HTTP attributes
                    span.set_attribute("http.method", method)
                    span.set_attribute("http.url", url)
                    span.set_attribute("http.host", str(request.url.host))
                    span.set_attribute("http.scheme", request.url.scheme)
                    span.set_attribute("http.target", str(request.url.path))

                    # Inject trace context into headers
                    if hasattr(self.tracing_engine, "propagator"):
                        ctx = SpanContext(
                            trace_id=span.trace_id,
                            span_id=span.span_id,
                            sampled=True,
                        )
                        headers = dict(request.headers)
                        self.tracing_engine.propagator.inject(ctx, headers)
                        request.headers.update(headers)

                    # Capture request headers
                    if self.config.capture_headers:
                        for key, value in request.headers.items():
                            if key.lower() not in ("authorization", "cookie"):
                                span.set_attribute(f"http.request.header.{key}", value)

                    try:
                        response = await original_send(self_client, request, *args, **kwargs)

                        span.set_attribute("http.status_code", response.status_code)
                        span.set_attribute(
                            "http.response_content_length",
                            len(response.content) if response.content else 0
                        )

                        if response.status_code >= 400:
                            span.set_error(Exception(f"HTTP {response.status_code}"))

                        return response

                    except Exception as e:
                        span.set_error(e)
                        raise

            httpx.AsyncClient.send = instrumented_send
            self._original_methods["httpx.AsyncClient.send"] = original_send

            # Also instrument sync client
            original_sync_send = httpx.Client.send

            def instrumented_sync_send(self_client, request, *args, **kwargs):
                if not self.config.enabled:
                    return original_sync_send(self_client, request, *args, **kwargs)

                # Similar instrumentation for sync...
                return original_sync_send(self_client, request, *args, **kwargs)

            httpx.Client.send = instrumented_sync_send
            self._original_methods["httpx.Client.send"] = original_sync_send

            self._instrumented = True
            logger.info("Instrumented httpx")
            return True

        except ImportError:
            return False

    def instrument_aiohttp(self) -> bool:
        """Instrument aiohttp library."""
        try:
            import aiohttp

            original_request = aiohttp.ClientSession._request

            async def instrumented_request(
                self_session, method, url, *args, **kwargs
            ):
                if not self.config.enabled:
                    return await original_request(
                        self_session, method, url, *args, **kwargs
                    )

                span_name = f"HTTP {method} {url}"

                with self.tracing_engine.trace(span_name, kind=SpanKind.CLIENT) as span:
                    span.set_attribute("http.method", method)
                    span.set_attribute("http.url", str(url))

                    # Inject trace context
                    headers = kwargs.get("headers", {}) or {}
                    if hasattr(self.tracing_engine, "propagator"):
                        ctx = SpanContext(
                            trace_id=span.trace_id,
                            span_id=span.span_id,
                            sampled=True,
                        )
                        self.tracing_engine.propagator.inject(ctx, headers)
                        kwargs["headers"] = headers

                    try:
                        response = await original_request(
                            self_session, method, url, *args, **kwargs
                        )

                        span.set_attribute("http.status_code", response.status)

                        if response.status >= 400:
                            span.set_error(Exception(f"HTTP {response.status}"))

                        return response

                    except Exception as e:
                        span.set_error(e)
                        raise

            aiohttp.ClientSession._request = instrumented_request
            self._original_methods["aiohttp.ClientSession._request"] = original_request

            self._instrumented = True
            logger.info("Instrumented aiohttp")
            return True

        except ImportError:
            return False

    def uninstrument(self) -> None:
        """Remove instrumentation."""
        for key, original in self._original_methods.items():
            parts = key.rsplit(".", 1)
            if len(parts) == 2:
                module_path, method = parts
                # Restore original method
                try:
                    obj = eval(module_path)
                    setattr(obj, method.split(".")[-1], original)
                except Exception:
                    pass

        self._original_methods.clear()
        self._instrumented = False


class DatabaseInstrumentor:
    """
    Automatic instrumentation for database clients.

    Supports: asyncpg, psycopg2, sqlite3, pymongo, redis, etc.
    """

    def __init__(
        self,
        tracing_engine: Any,
        config: Optional[InstrumentationConfig] = None,
    ):
        self.tracing_engine = tracing_engine
        self.config = config or InstrumentationConfig()
        self._original_methods: Dict[str, Any] = {}
        self._instrumented = False

    def instrument_asyncpg(self) -> bool:
        """Instrument asyncpg (PostgreSQL async driver)."""
        try:
            import asyncpg

            original_execute = asyncpg.Connection.execute
            original_fetch = asyncpg.Connection.fetch
            original_fetchrow = asyncpg.Connection.fetchrow

            async def instrumented_execute(conn, query, *args, **kwargs):
                span_name = self._get_span_name("execute", query)

                with self.tracing_engine.trace(span_name, kind=SpanKind.CLIENT) as span:
                    span.set_attribute("db.system", "postgresql")
                    span.set_attribute("db.operation", "execute")
                    span.set_attribute("db.statement", self._sanitize_query(query))

                    try:
                        result = await original_execute(conn, query, *args, **kwargs)
                        return result
                    except Exception as e:
                        span.set_error(e)
                        raise

            async def instrumented_fetch(conn, query, *args, **kwargs):
                span_name = self._get_span_name("fetch", query)

                with self.tracing_engine.trace(span_name, kind=SpanKind.CLIENT) as span:
                    span.set_attribute("db.system", "postgresql")
                    span.set_attribute("db.operation", "fetch")
                    span.set_attribute("db.statement", self._sanitize_query(query))

                    try:
                        result = await original_fetch(conn, query, *args, **kwargs)
                        span.set_attribute("db.rows_affected", len(result))
                        return result
                    except Exception as e:
                        span.set_error(e)
                        raise

            asyncpg.Connection.execute = instrumented_execute
            asyncpg.Connection.fetch = instrumented_fetch

            self._original_methods["asyncpg.Connection.execute"] = original_execute
            self._original_methods["asyncpg.Connection.fetch"] = original_fetch

            self._instrumented = True
            logger.info("Instrumented asyncpg")
            return True

        except ImportError:
            return False

    def instrument_redis(self) -> bool:
        """Instrument redis-py."""
        try:
            import redis.asyncio as aioredis

            original_execute = aioredis.Redis.execute_command

            async def instrumented_execute(self_redis, *args, **kwargs):
                command = args[0] if args else "UNKNOWN"
                span_name = f"Redis {command}"

                with self.tracing_engine.trace(span_name, kind=SpanKind.CLIENT) as span:
                    span.set_attribute("db.system", "redis")
                    span.set_attribute("db.operation", command)
                    span.set_attribute(
                        "db.statement",
                        " ".join(str(a) for a in args[:3])  # First 3 args only
                    )

                    try:
                        result = await original_execute(self_redis, *args, **kwargs)
                        return result
                    except Exception as e:
                        span.set_error(e)
                        raise

            aioredis.Redis.execute_command = instrumented_execute
            self._original_methods["redis.asyncio.Redis.execute_command"] = original_execute

            self._instrumented = True
            logger.info("Instrumented redis")
            return True

        except ImportError:
            return False

    def instrument_pymongo(self) -> bool:
        """Instrument pymongo."""
        try:
            from pymongo import monitoring

            class TracingCommandListener(monitoring.CommandListener):
                def __init__(self_listener, tracing_engine):
                    self_listener.tracing_engine = tracing_engine
                    self_listener.spans: Dict[int, Span] = {}

                def started(self_listener, event):
                    span_name = f"MongoDB {event.command_name}"
                    span = self_listener.tracing_engine.start_span(
                        span_name,
                        kind=SpanKind.CLIENT,
                    )
                    span.set_attribute("db.system", "mongodb")
                    span.set_attribute("db.operation", event.command_name)
                    span.set_attribute("db.name", event.database_name)
                    self_listener.spans[event.request_id] = span

                def succeeded(self_listener, event):
                    span = self_listener.spans.pop(event.request_id, None)
                    if span:
                        span.set_attribute("db.duration_ms", event.duration_micros / 1000)
                        span.end()

                def failed(self_listener, event):
                    span = self_listener.spans.pop(event.request_id, None)
                    if span:
                        span.set_error(Exception(str(event.failure)))
                        span.end()

            listener = TracingCommandListener(self.tracing_engine)
            monitoring.register(listener)

            self._instrumented = True
            logger.info("Instrumented pymongo")
            return True

        except ImportError:
            return False

    def _get_span_name(self, operation: str, query: str) -> str:
        """Generate span name from operation and query."""
        # Extract table name from query
        query_upper = query.upper().strip()

        for keyword in ["FROM", "INTO", "UPDATE", "TABLE"]:
            if keyword in query_upper:
                idx = query_upper.index(keyword)
                remaining = query[idx + len(keyword):].strip()
                table = remaining.split()[0] if remaining else "unknown"
                return f"DB {operation} {table}"

        return f"DB {operation}"

    def _sanitize_query(self, query: str) -> str:
        """Sanitize query to remove sensitive data."""
        # Simple sanitization - replace literals
        import re

        # Replace string literals
        sanitized = re.sub(r"'[^']*'", "'?'", query)
        # Replace numeric literals
        sanitized = re.sub(r"\b\d+\b", "?", sanitized)

        return sanitized[:500]  # Limit length

    def uninstrument(self) -> None:
        """Remove instrumentation."""
        for key, original in self._original_methods.items():
            # Restore original methods
            pass
        self._original_methods.clear()
        self._instrumented = False


class AutoInstrumentor:
    """
    Unified automatic instrumentation manager.

    Instruments multiple libraries automatically.
    """

    def __init__(
        self,
        tracing_engine: Any,
        config: Optional[InstrumentationConfig] = None,
    ):
        self.tracing_engine = tracing_engine
        self.config = config or InstrumentationConfig()

        self.http_instrumentor = HTTPClientInstrumentor(tracing_engine, config)
        self.db_instrumentor = DatabaseInstrumentor(tracing_engine, config)

        self._instrumented_libraries: List[str] = []

    def instrument_all(self) -> List[str]:
        """Instrument all available libraries."""
        instrumented = []

        # HTTP clients
        if self.http_instrumentor.instrument_httpx():
            instrumented.append("httpx")
        if self.http_instrumentor.instrument_aiohttp():
            instrumented.append("aiohttp")

        # Databases
        if self.db_instrumentor.instrument_asyncpg():
            instrumented.append("asyncpg")
        if self.db_instrumentor.instrument_redis():
            instrumented.append("redis")
        if self.db_instrumentor.instrument_pymongo():
            instrumented.append("pymongo")

        self._instrumented_libraries = instrumented
        return instrumented

    def uninstrument_all(self) -> None:
        """Remove all instrumentation."""
        self.http_instrumentor.uninstrument()
        self.db_instrumentor.uninstrument()
        self._instrumented_libraries.clear()

    def get_instrumented_libraries(self) -> List[str]:
        """Get list of instrumented libraries."""
        return self._instrumented_libraries


# =============================================================================
# Trace Analytics
# =============================================================================

@dataclass
class TracePattern:
    """A detected pattern in traces."""
    name: str
    description: str
    occurrence_count: int
    avg_duration_ms: float
    services: List[str]
    operations: List[str]
    example_trace_ids: List[str]


class TraceAnalyzer:
    """
    Analyze traces to find patterns, bottlenecks, and anomalies.
    """

    def __init__(self, max_traces: int = 10000):
        self.max_traces = max_traces
        self._traces: Dict[str, Trace] = {}
        self._operation_stats: Dict[str, List[float]] = defaultdict(list)
        self._service_edges: Dict[Tuple[str, str], int] = defaultdict(int)

    def add_trace(self, trace: Trace) -> None:
        """Add a trace for analysis."""
        if len(self._traces) >= self.max_traces:
            # Remove oldest
            oldest = min(self._traces.values(), key=lambda t: t.start_time or datetime.min)
            del self._traces[oldest.trace_id]

        self._traces[trace.trace_id] = trace
        self._analyze_trace(trace)

    def _analyze_trace(self, trace: Trace) -> None:
        """Analyze a single trace."""
        services_in_trace = set()

        for span in trace.spans:
            # Track operation durations
            if span.duration_ms:
                key = f"{span.service_name}:{span.operation_name}"
                self._operation_stats[key].append(span.duration_ms)

            services_in_trace.add(span.service_name)

            # Track service-to-service calls
            if span.parent_span_id:
                parent = self._find_span(trace, span.parent_span_id)
                if parent and parent.service_name != span.service_name:
                    edge = (parent.service_name, span.service_name)
                    self._service_edges[edge] += 1

    def _find_span(self, trace: Trace, span_id: str) -> Optional[Span]:
        """Find a span by ID in a trace."""
        for span in trace.spans:
            if span.span_id == span_id:
                return span
        return None

    def get_slow_operations(
        self,
        percentile: float = 95.0,
        min_samples: int = 10,
    ) -> List[Dict[str, Any]]:
        """Get operations with high latency."""
        import numpy as np

        slow_ops = []

        for key, durations in self._operation_stats.items():
            if len(durations) < min_samples:
                continue

            p50 = np.percentile(durations, 50)
            p95 = np.percentile(durations, 95)
            p99 = np.percentile(durations, 99)

            service, operation = key.split(":", 1)

            slow_ops.append({
                "service": service,
                "operation": operation,
                "p50_ms": p50,
                "p95_ms": p95,
                "p99_ms": p99,
                "sample_count": len(durations),
            })

        # Sort by p95
        slow_ops.sort(key=lambda x: x["p95_ms"], reverse=True)
        return slow_ops[:20]

    def get_service_dependencies(self) -> List[Dict[str, Any]]:
        """Get service-to-service dependencies."""
        deps = []
        for (source, target), count in self._service_edges.items():
            deps.append({
                "source": source,
                "target": target,
                "call_count": count,
            })

        deps.sort(key=lambda x: x["call_count"], reverse=True)
        return deps

    def find_patterns(self, min_occurrences: int = 5) -> List[TracePattern]:
        """Find common patterns in traces."""
        # Group traces by operation sequence
        pattern_groups: Dict[str, List[Trace]] = defaultdict(list)

        for trace in self._traces.values():
            # Create signature from root operation + child operations
            if trace.root_span:
                sig_parts = [trace.root_span.operation_name]
                for span in trace.spans[:10]:  # Limit for signature
                    if span.span_id != trace.root_span.span_id:
                        sig_parts.append(span.operation_name)
                signature = "->".join(sig_parts)
                pattern_groups[signature].append(trace)

        patterns = []
        for signature, traces in pattern_groups.items():
            if len(traces) >= min_occurrences:
                durations = [t.duration_ms or 0 for t in traces]
                services = list(set(
                    s.service_name for t in traces for s in t.spans
                ))
                operations = signature.split("->")

                patterns.append(TracePattern(
                    name=signature[:50],
                    description=f"Pattern with {len(operations)} operations",
                    occurrence_count=len(traces),
                    avg_duration_ms=sum(durations) / len(durations) if durations else 0,
                    services=services,
                    operations=operations,
                    example_trace_ids=[t.trace_id for t in traces[:3]],
                ))

        patterns.sort(key=lambda p: p.occurrence_count, reverse=True)
        return patterns[:20]
