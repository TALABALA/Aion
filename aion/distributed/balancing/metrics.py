"""
AION Distributed Computing - Load Metrics Collection

Production-grade metrics aggregation for cluster-wide load monitoring.
Implements SOTA patterns including:
- Per-node latency histograms with p50 / p95 / p99 percentile computation
- Sliding-window based error rate and throughput tracking
- Exponentially weighted moving average (EWMA) for smooth metric transitions
- Hotspot detection for nodes exceeding utilisation thresholds
- Cluster-wide aggregation (average, min, max, percentiles)
"""

from __future__ import annotations

import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Deque, Dict, List, Optional, Sequence, Tuple

import structlog

from aion.distributed.types import ClusterMetrics, NodeStatus

if TYPE_CHECKING:
    from aion.distributed.cluster.manager import ClusterManager

logger = structlog.get_logger(__name__)

# Default configuration constants
_DEFAULT_WINDOW_SIZE: int = 300  # 5 minutes of per-second slots
_DEFAULT_HOTSPOT_THRESHOLD: float = 0.80
_DEFAULT_EWMA_ALPHA: float = 0.3
_DEFAULT_MAX_LATENCY_SAMPLES: int = 1000


# =============================================================================
# Per-Node Metrics Container
# =============================================================================


@dataclass
class _NodeMetrics:
    """Internal per-node metrics store.

    Maintains sliding-window collections for latency samples, error counts,
    and throughput counters together with EWMA-smoothed summary values.
    """

    node_id: str = ""

    # Latency histogram (raw samples for percentile computation)
    latency_samples: Deque[float] = field(
        default_factory=lambda: deque(maxlen=_DEFAULT_MAX_LATENCY_SAMPLES),
    )
    latency_ewma: float = 0.0
    _latency_initialised: bool = False

    # Error tracking (sliding window of booleans: True = error)
    error_window: Deque[bool] = field(
        default_factory=lambda: deque(maxlen=_DEFAULT_WINDOW_SIZE),
    )

    # Throughput tracking (timestamps of completed requests)
    request_timestamps: Deque[float] = field(
        default_factory=lambda: deque(maxlen=_DEFAULT_WINDOW_SIZE * 10),
    )

    # Aggregated counters
    total_requests: int = 0
    total_errors: int = 0
    last_updated: float = field(default_factory=time.monotonic)

    # -----------------------------------------------------------------
    # Latency helpers
    # -----------------------------------------------------------------

    def record_latency(self, latency_ms: float, ewma_alpha: float) -> None:
        """Record an observed request latency in milliseconds."""
        self.latency_samples.append(latency_ms)
        if not self._latency_initialised:
            self.latency_ewma = latency_ms
            self._latency_initialised = True
        else:
            self.latency_ewma = (
                ewma_alpha * latency_ms + (1.0 - ewma_alpha) * self.latency_ewma
            )
        self.total_requests += 1
        self.request_timestamps.append(time.monotonic())
        self.last_updated = time.monotonic()

    def latency_percentile(self, p: float) -> float:
        """Return the *p*-th percentile of latency samples (0-100 scale).

        Uses the nearest-rank method.  Returns ``0.0`` when no samples are
        available.
        """
        if not self.latency_samples:
            return 0.0
        sorted_samples = sorted(self.latency_samples)
        k = max(0, min(int(math.ceil(p / 100.0 * len(sorted_samples))) - 1, len(sorted_samples) - 1))
        return sorted_samples[k]

    # -----------------------------------------------------------------
    # Error helpers
    # -----------------------------------------------------------------

    def record_error(self) -> None:
        """Record a request error."""
        self.error_window.append(True)
        self.total_errors += 1
        self.last_updated = time.monotonic()

    def record_success(self) -> None:
        """Record a successful request (for sliding window accuracy)."""
        self.error_window.append(False)
        self.last_updated = time.monotonic()

    @property
    def error_rate(self) -> float:
        """Current error rate in the sliding window (0.0-1.0)."""
        if not self.error_window:
            return 0.0
        return sum(1 for e in self.error_window if e) / len(self.error_window)

    # -----------------------------------------------------------------
    # Throughput helpers
    # -----------------------------------------------------------------

    def throughput_rps(self, window_seconds: float = 60.0) -> float:
        """Compute requests-per-second over the last *window_seconds*."""
        if not self.request_timestamps:
            return 0.0
        now = time.monotonic()
        cutoff = now - window_seconds
        recent = [ts for ts in self.request_timestamps if ts >= cutoff]
        if not recent or window_seconds <= 0:
            return 0.0
        elapsed = now - recent[0]
        if elapsed <= 0:
            return float(len(recent))
        return len(recent) / elapsed

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the per-node metrics to a dictionary."""
        return {
            "node_id": self.node_id,
            "latency_ewma_ms": round(self.latency_ewma, 2),
            "latency_p50_ms": round(self.latency_percentile(50), 2),
            "latency_p95_ms": round(self.latency_percentile(95), 2),
            "latency_p99_ms": round(self.latency_percentile(99), 2),
            "error_rate": round(self.error_rate, 4),
            "total_requests": self.total_requests,
            "total_errors": self.total_errors,
            "throughput_rps": round(self.throughput_rps(), 2),
            "latency_sample_count": len(self.latency_samples),
        }


# =============================================================================
# LoadMetrics - Cluster-Wide Aggregation
# =============================================================================


class LoadMetrics:
    """Collects and aggregates load metrics across the AION cluster.

    Designed to be driven by the :class:`LoadBalancer` which forwards
    latency / error observations after each request, and periodically by
    a background collector that queries the :class:`ClusterManager` for
    node resource utilisation.

    Features:
        * Per-node latency histograms with p50 / p95 / p99 computation.
        * Sliding-window error rate and throughput tracking.
        * EWMA for smooth metric transitions.
        * Hotspot detection (nodes above a configurable utilisation threshold).
        * Cluster-wide aggregation: avg, min, max across all node metrics.

    Args:
        cluster_manager: Reference to the cluster manager for node discovery.
        window_size: Sliding window size for error / throughput tracking.
        hotspot_threshold: Load score above which a node is a hotspot.
        ewma_alpha: Smoothing factor for EWMA latency (0 < alpha <= 1).
    """

    def __init__(
        self,
        cluster_manager: ClusterManager,
        *,
        window_size: int = _DEFAULT_WINDOW_SIZE,
        hotspot_threshold: float = _DEFAULT_HOTSPOT_THRESHOLD,
        ewma_alpha: float = _DEFAULT_EWMA_ALPHA,
    ) -> None:
        self._cluster_manager = cluster_manager
        self._window_size = window_size
        self._hotspot_threshold = hotspot_threshold
        self._ewma_alpha = ewma_alpha

        # Per-node metrics keyed by node_id
        self._node_metrics: Dict[str, _NodeMetrics] = {}

        logger.info(
            "load_metrics_init",
            window_size=window_size,
            hotspot_threshold=hotspot_threshold,
            ewma_alpha=ewma_alpha,
        )

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _ensure_node(self, node_id: str) -> _NodeMetrics:
        """Lazily create per-node metrics container."""
        if node_id not in self._node_metrics:
            self._node_metrics[node_id] = _NodeMetrics(
                node_id=node_id,
                latency_samples=deque(maxlen=_DEFAULT_MAX_LATENCY_SAMPLES),
                error_window=deque(maxlen=self._window_size),
                request_timestamps=deque(maxlen=self._window_size * 10),
            )
        return self._node_metrics[node_id]

    # -----------------------------------------------------------------
    # Recording API (called by LoadBalancer / external code)
    # -----------------------------------------------------------------

    def record_request_latency(self, node_id: str, latency_ms: float) -> None:
        """Record an observed request latency for *node_id*.

        Updates the latency histogram, EWMA estimate, and throughput counter
        for the given node.

        Args:
            node_id: The identifier of the node that handled the request.
            latency_ms: Observed latency in milliseconds.
        """
        nm = self._ensure_node(node_id)
        nm.record_latency(latency_ms, self._ewma_alpha)
        nm.record_success()
        logger.debug(
            "latency_recorded",
            node_id=node_id,
            latency_ms=round(latency_ms, 2),
            ewma=round(nm.latency_ewma, 2),
        )

    def record_request_error(self, node_id: str) -> None:
        """Record a request error for *node_id*.

        Increments the error counter and appends to the sliding window.

        Args:
            node_id: The identifier of the node that produced the error.
        """
        nm = self._ensure_node(node_id)
        nm.record_error()
        logger.debug(
            "error_recorded",
            node_id=node_id,
            error_rate=round(nm.error_rate, 4),
            total_errors=nm.total_errors,
        )

    # -----------------------------------------------------------------
    # Query API
    # -----------------------------------------------------------------

    async def collect(self) -> ClusterMetrics:
        """Collect and return cluster-wide aggregated metrics.

        Queries the cluster manager for current node state and combines it
        with locally tracked latency / throughput data.

        Returns:
            A :class:`ClusterMetrics` snapshot.
        """
        nodes = await self._get_nodes()

        total = len(nodes)
        healthy = sum(
            1 for n in nodes if n.status == NodeStatus.HEALTHY
        )
        total_tasks_running = sum(n.current_tasks for n in nodes)
        avg_cpu = (
            sum(n.cpu_usage for n in nodes) / total if total > 0 else 0.0
        )
        avg_mem = (
            sum(n.memory_usage for n in nodes) / total if total > 0 else 0.0
        )

        # Aggregate latency
        all_ewma: List[float] = [
            nm.latency_ewma
            for nm in self._node_metrics.values()
            if nm._latency_initialised
        ]
        avg_latency = sum(all_ewma) / len(all_ewma) if all_ewma else 0.0
        p99_latency = self._cluster_latency_percentile(99)

        # Aggregate throughput
        total_rps = sum(
            nm.throughput_rps() for nm in self._node_metrics.values()
        )

        return ClusterMetrics(
            timestamp=datetime.now(),
            total_nodes=total,
            healthy_nodes=healthy,
            has_quorum=healthy >= (total // 2 + 1) if total > 0 else False,
            avg_cpu_usage=avg_cpu,
            avg_memory_usage=avg_mem,
            total_tasks_running=total_tasks_running,
            tasks_per_second=total_rps,
            avg_rpc_latency_ms=avg_latency,
            p99_rpc_latency_ms=p99_latency,
        )

    async def get_node_metrics(self, node_id: str) -> Dict[str, Any]:
        """Return detailed metrics for a specific node.

        Args:
            node_id: The identifier of the node.

        Returns:
            Dictionary with latency histogram, error rate, throughput, and
            resource utilisation.
        """
        nm = self._ensure_node(node_id)
        result = nm.to_dict()

        # Enrich with live resource data from the cluster manager
        node = await self._get_node(node_id)
        if node is not None:
            result.update({
                "cpu_usage": round(node.cpu_usage, 4),
                "memory_usage": round(node.memory_usage, 4),
                "current_tasks": node.current_tasks,
                "max_concurrent_tasks": node.max_concurrent_tasks,
                "load_score": round(node.load_score, 4),
                "status": node.status.value,
            })

        return result

    async def get_hotspots(self) -> List[str]:
        """Identify nodes whose load score exceeds the hotspot threshold.

        Returns:
            List of node IDs that are considered hotspots.
        """
        nodes = await self._get_nodes()
        hotspots: List[str] = []
        for node in nodes:
            if node.load_score >= self._hotspot_threshold:
                hotspots.append(node.id)
                logger.warning(
                    "hotspot_detected",
                    node_id=node.id,
                    load_score=round(node.load_score, 4),
                    threshold=self._hotspot_threshold,
                )
        return hotspots

    async def get_utilization(self) -> float:
        """Return overall cluster utilisation as a float (0.0 - 1.0).

        Computed as the average ``load_score`` across all known nodes.
        """
        nodes = await self._get_nodes()
        if not nodes:
            return 0.0
        return sum(n.load_score for n in nodes) / len(nodes)

    # -----------------------------------------------------------------
    # Advanced aggregation
    # -----------------------------------------------------------------

    def get_cluster_latency_summary(self) -> Dict[str, float]:
        """Return cluster-level latency summary across all nodes.

        Returns:
            Dictionary with ``avg``, ``min``, ``max``, ``p50``, ``p95``,
            ``p99`` latency values in milliseconds.
        """
        all_samples: List[float] = []
        for nm in self._node_metrics.values():
            all_samples.extend(nm.latency_samples)

        if not all_samples:
            return {"avg": 0.0, "min": 0.0, "max": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}

        sorted_s = sorted(all_samples)
        n = len(sorted_s)
        return {
            "avg": round(sum(sorted_s) / n, 2),
            "min": round(sorted_s[0], 2),
            "max": round(sorted_s[-1], 2),
            "p50": round(sorted_s[max(0, int(n * 0.50) - 1)], 2),
            "p95": round(sorted_s[max(0, int(n * 0.95) - 1)], 2),
            "p99": round(sorted_s[max(0, int(n * 0.99) - 1)], 2),
        }

    def get_cluster_error_summary(self) -> Dict[str, Any]:
        """Return cluster-level error summary across all nodes.

        Returns:
            Dictionary with ``avg_error_rate``, ``max_error_rate``,
            ``total_errors``, and ``nodes_with_errors``.
        """
        if not self._node_metrics:
            return {
                "avg_error_rate": 0.0,
                "max_error_rate": 0.0,
                "total_errors": 0,
                "nodes_with_errors": 0,
            }

        rates = [nm.error_rate for nm in self._node_metrics.values()]
        total_errors = sum(nm.total_errors for nm in self._node_metrics.values())
        nodes_with_errors = sum(1 for nm in self._node_metrics.values() if nm.total_errors > 0)

        return {
            "avg_error_rate": round(sum(rates) / len(rates), 4) if rates else 0.0,
            "max_error_rate": round(max(rates), 4) if rates else 0.0,
            "total_errors": total_errors,
            "nodes_with_errors": nodes_with_errors,
        }

    # -----------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------

    def _cluster_latency_percentile(self, p: float) -> float:
        """Compute a percentile across all node latency samples."""
        all_samples: List[float] = []
        for nm in self._node_metrics.values():
            all_samples.extend(nm.latency_samples)
        if not all_samples:
            return 0.0
        sorted_s = sorted(all_samples)
        k = max(0, min(int(math.ceil(p / 100.0 * len(sorted_s))) - 1, len(sorted_s) - 1))
        return sorted_s[k]

    async def _get_nodes(self) -> List[Any]:
        """Retrieve the current node list from the cluster manager."""
        try:
            if hasattr(self._cluster_manager, "get_nodes"):
                nodes = await self._cluster_manager.get_nodes()
                return list(nodes) if nodes else []
            if hasattr(self._cluster_manager, "state"):
                state = self._cluster_manager.state
                return list(state.nodes.values()) if state and state.nodes else []
        except Exception:
            logger.exception("failed_to_get_nodes")
        return []

    async def _get_node(self, node_id: str) -> Optional[Any]:
        """Retrieve a single node from the cluster manager."""
        try:
            if hasattr(self._cluster_manager, "get_node"):
                return await self._cluster_manager.get_node(node_id)
            if hasattr(self._cluster_manager, "state"):
                state = self._cluster_manager.state
                if state and state.nodes:
                    return state.nodes.get(node_id)
        except Exception:
            logger.exception("failed_to_get_node", node_id=node_id)
        return None
