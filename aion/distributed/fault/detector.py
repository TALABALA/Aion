"""
AION Failure Detector

Production-grade failure detection implementing:
- Phi accrual failure detector (Hayashibara et al.) for adaptive suspicion
- SWIM-style protocol with indirect probing for reduced false positives
- Sliding window of heartbeat inter-arrival times per node
- Normal distribution CDF-based phi calculation
- Configurable suspicion thresholds with multi-level status mapping
"""

from __future__ import annotations

import asyncio
import math
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Deque, Dict, List, Optional, Set

import structlog

from aion.distributed.types import NodeStatus

if TYPE_CHECKING:
    from aion.distributed.cluster.manager import ClusterManager

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Internal data structures
# ---------------------------------------------------------------------------


@dataclass
class _HeartbeatWindow:
    """Sliding window of heartbeat inter-arrival intervals for a single node.

    Maintains a bounded deque of time deltas between consecutive heartbeat
    arrivals, along with running statistics for efficient phi computation.
    """

    window_size: int = 100
    intervals: Deque[float] = field(default_factory=deque)
    last_arrival: Optional[float] = None
    total_heartbeats: int = 0

    # Running sums for incremental mean / variance
    _sum: float = 0.0
    _sum_sq: float = 0.0

    def record(self, now: float) -> None:
        """Record a new heartbeat arrival timestamp."""
        if self.last_arrival is not None:
            interval = now - self.last_arrival
            if interval > 0:
                self._add_interval(interval)
        self.last_arrival = now
        self.total_heartbeats += 1

    # -- internal helpers --------------------------------------------------

    def _add_interval(self, interval: float) -> None:
        """Append an interval to the sliding window, evicting oldest if full."""
        if len(self.intervals) >= self.window_size:
            evicted = self.intervals.popleft()
            self._sum -= evicted
            self._sum_sq -= evicted * evicted
        self.intervals.append(interval)
        self._sum += interval
        self._sum_sq += interval * interval

    @property
    def mean(self) -> float:
        """Mean inter-arrival time."""
        n = len(self.intervals)
        if n == 0:
            return 0.0
        return self._sum / n

    @property
    def variance(self) -> float:
        """Variance of inter-arrival times (population variance)."""
        n = len(self.intervals)
        if n < 2:
            return 0.0
        mean = self.mean
        return max(0.0, (self._sum_sq / n) - (mean * mean))

    @property
    def stddev(self) -> float:
        """Standard deviation of inter-arrival times."""
        return math.sqrt(self.variance)


@dataclass
class _IndirectProbeState:
    """Tracks in-flight indirect probes for the SWIM protocol layer."""

    target_node_id: str
    requested_at: float
    responders: Set[str] = field(default_factory=set)
    ack_received: bool = False
    timeout: float = 2.0


# ---------------------------------------------------------------------------
# FailureDetector
# ---------------------------------------------------------------------------


class FailureDetector:
    """Phi accrual failure detector with SWIM-style indirect probing.

    The detector maintains a sliding window of heartbeat inter-arrival times
    for every monitored node and derives a continuous *phi* suspicion value
    using the CDF of the fitted normal distribution.  A higher phi value
    indicates stronger suspicion that the node has failed.

    When a node's phi exceeds the configured *threshold*, the node is
    classified as *suspected*.  An additional multiplier promotes suspected
    nodes to *offline* status.

    To mitigate false positives caused by transient network issues, the
    detector also implements SWIM-style indirect probing: when a direct
    heartbeat times out, *K* randomly chosen peers are asked to ping the
    suspect on behalf of the detector.

    Args:
        threshold: Phi value at which a node is considered suspected.
                   Default ``8.0`` (probability ~1e-8 of a false positive
                   assuming a stable heartbeat rate).
        window_size: Maximum number of inter-arrival samples to retain per
                     node.  Default ``100``.
        cluster_manager: Optional reference to the :class:`ClusterManager`
                         used for indirect probing.
        indirect_probe_count: Number of random peers (K) asked to perform
                              an indirect ping.  Default ``3``.
    """

    # Multiplier over *threshold* at which a suspected node is declared dead
    _DEAD_MULTIPLIER: float = 1.5

    # Minimum standard deviation floor to avoid divide-by-zero
    _MIN_STDDEV: float = 0.01

    def __init__(
        self,
        threshold: float = 8.0,
        window_size: int = 100,
        cluster_manager: Optional[ClusterManager] = None,
        indirect_probe_count: int = 3,
    ) -> None:
        self.threshold = threshold
        self._window_size = window_size
        self._cluster_manager = cluster_manager
        self._indirect_probe_count = indirect_probe_count

        # Per-node heartbeat windows
        self._windows: Dict[str, _HeartbeatWindow] = {}

        # Indirect probe bookkeeping
        self._active_probes: Dict[str, _IndirectProbeState] = {}

        # Configurable intervals
        self._default_interval: float = 1.0  # expected heartbeat interval (s)
        self._probe_timeout: float = 2.0     # indirect probe timeout (s)

        logger.info(
            "failure_detector.init",
            threshold=threshold,
            window_size=window_size,
            indirect_probe_count=indirect_probe_count,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_heartbeat(self, node_id: str) -> None:
        """Record receipt of a heartbeat from *node_id*.

        This updates the sliding window of inter-arrival times, which in
        turn affects the phi calculation for that node.
        """
        now = time.monotonic()
        window = self._windows.get(node_id)
        if window is None:
            window = _HeartbeatWindow(window_size=self._window_size)
            self._windows[node_id] = window
        window.record(now)

        # If we had an active indirect probe, mark it as resolved
        probe = self._active_probes.pop(node_id, None)
        if probe is not None:
            logger.debug(
                "failure_detector.indirect_probe_resolved",
                node_id=node_id,
            )

    def get_phi(self, node_id: str) -> float:
        """Compute the current phi value for *node_id*.

        Phi is defined as ``-log10(1 - CDF(t_now - t_last))`` where CDF is
        the cumulative distribution function of the normal distribution
        fitted to the observed heartbeat inter-arrival times.

        Returns:
            The phi suspicion level.  Returns ``float('inf')`` if no
            heartbeat has ever been recorded.
        """
        window = self._windows.get(node_id)
        if window is None or window.last_arrival is None:
            return float("inf")

        now = time.monotonic()
        elapsed = now - window.last_arrival

        mean = window.mean if window.mean > 0 else self._default_interval
        stddev = max(window.stddev, self._MIN_STDDEV)

        # Compute CDF of the normal distribution at *elapsed*
        p = self._normal_cdf(elapsed, mean, stddev)

        # Phi = -log10(1 - p)
        if p >= 1.0:
            return float("inf")
        try:
            phi = -math.log10(1.0 - p)
        except (ValueError, ZeroDivisionError):
            phi = float("inf")

        return phi

    def is_alive(self, node_id: str) -> bool:
        """Return ``True`` if *node_id* is considered alive.

        A node is alive if its phi value is strictly below the configured
        threshold.
        """
        return self.get_phi(node_id) < self.threshold

    def get_status(self, node_id: str) -> NodeStatus:
        """Map the current phi value of *node_id* to a :class:`NodeStatus`.

        * ``phi < threshold`` -- :attr:`NodeStatus.HEALTHY`
        * ``threshold <= phi < threshold * 1.5`` -- :attr:`NodeStatus.SUSPECTED`
        * ``phi >= threshold * 1.5`` -- :attr:`NodeStatus.OFFLINE`
        """
        phi = self.get_phi(node_id)
        if phi < self.threshold:
            return NodeStatus.HEALTHY
        if phi < self.threshold * self._DEAD_MULTIPLIER:
            return NodeStatus.SUSPECTED
        return NodeStatus.OFFLINE

    def get_all_statuses(self) -> Dict[str, Dict[str, Any]]:
        """Return phi values and derived statuses for all monitored nodes.

        Returns:
            Mapping from node ID to a dict with ``phi``, ``status``, and
            ``total_heartbeats`` keys.
        """
        result: Dict[str, Dict[str, Any]] = {}
        for node_id, window in self._windows.items():
            phi = self.get_phi(node_id)
            result[node_id] = {
                "phi": round(phi, 4) if phi != float("inf") else None,
                "status": self.get_status(node_id).value,
                "total_heartbeats": window.total_heartbeats,
                "mean_interval": round(window.mean, 6),
                "stddev_interval": round(window.stddev, 6),
                "is_alive": self.is_alive(node_id),
            }
        return result

    def remove_node(self, node_id: str) -> None:
        """Stop tracking *node_id* and release its heartbeat window."""
        self._windows.pop(node_id, None)
        self._active_probes.pop(node_id, None)
        logger.debug("failure_detector.node_removed", node_id=node_id)

    # ------------------------------------------------------------------
    # SWIM-style indirect probing
    # ------------------------------------------------------------------

    async def initiate_indirect_probe(self, suspect_node_id: str) -> bool:
        """Perform a SWIM-style indirect probe for *suspect_node_id*.

        Selects up to *K* random peers and asks each to ping the suspect.
        Returns ``True`` if at least one peer received an acknowledgement
        from the suspect within the probe timeout.

        This method is a coroutine and should only be used when the
        cluster manager is available.
        """
        if self._cluster_manager is None:
            logger.warning(
                "failure_detector.indirect_probe_no_manager",
                node_id=suspect_node_id,
            )
            return False

        # Already probing this node?
        if suspect_node_id in self._active_probes:
            return False

        # Gather eligible proxies (healthy peers excluding suspect and self)
        cluster_state = self._cluster_manager.state
        own_id = self._cluster_manager.node_id
        candidates = [
            nid
            for nid, node in cluster_state.nodes.items()
            if nid != own_id
            and nid != suspect_node_id
            and node.status == NodeStatus.HEALTHY
        ]

        if not candidates:
            logger.debug(
                "failure_detector.no_proxy_candidates",
                suspect=suspect_node_id,
            )
            return False

        k = min(self._indirect_probe_count, len(candidates))
        proxies = random.sample(candidates, k)

        probe = _IndirectProbeState(
            target_node_id=suspect_node_id,
            requested_at=time.monotonic(),
            timeout=self._probe_timeout,
        )
        self._active_probes[suspect_node_id] = probe

        logger.info(
            "failure_detector.indirect_probe_started",
            suspect=suspect_node_id,
            proxies=proxies,
        )

        # Fan-out probe requests
        tasks: List[asyncio.Task[bool]] = []
        for proxy_id in proxies:
            tasks.append(
                asyncio.create_task(
                    self._send_indirect_ping(proxy_id, suspect_node_id)
                )
            )

        # Wait for at least one success within the timeout window
        try:
            done, pending = await asyncio.wait(
                tasks,
                timeout=self._probe_timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )
            # Cancel remaining tasks
            for t in pending:
                t.cancel()

            for t in done:
                if t.result():
                    probe.ack_received = True
                    self.record_heartbeat(suspect_node_id)
                    logger.info(
                        "failure_detector.indirect_probe_success",
                        suspect=suspect_node_id,
                    )
                    return True
        except Exception:
            logger.exception(
                "failure_detector.indirect_probe_error",
                suspect=suspect_node_id,
            )
        finally:
            self._active_probes.pop(suspect_node_id, None)

        logger.warning(
            "failure_detector.indirect_probe_failed",
            suspect=suspect_node_id,
        )
        return False

    async def _send_indirect_ping(
        self, proxy_node_id: str, target_node_id: str
    ) -> bool:
        """Ask *proxy_node_id* to ping *target_node_id* and report back.

        Returns ``True`` if the proxy successfully reached the target.
        """
        if self._cluster_manager is None:
            return False

        rpc_client = getattr(self._cluster_manager, "_rpc_client", None)
        if rpc_client is None:
            return False

        proxy_node = self._cluster_manager.state.nodes.get(proxy_node_id)
        if proxy_node is None:
            return False

        try:
            result = await rpc_client.send_indirect_ping(
                proxy_node.address,
                target_node_id,
            )
            return bool(result)
        except Exception:
            logger.debug(
                "failure_detector.indirect_ping_failed",
                proxy=proxy_node_id,
                target=target_node_id,
            )
            return False

    # ------------------------------------------------------------------
    # Mathematical helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normal_cdf(x: float, mean: float, stddev: float) -> float:
        """Evaluate the CDF of a normal distribution at *x*.

        Uses the complementary error function for numerical stability.
        """
        if stddev <= 0:
            return 1.0 if x >= mean else 0.0
        z = (x - mean) / stddev
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_diagnostics(self) -> Dict[str, Any]:
        """Return detailed diagnostic information for debugging."""
        return {
            "threshold": self.threshold,
            "window_size": self._window_size,
            "indirect_probe_count": self._indirect_probe_count,
            "monitored_nodes": len(self._windows),
            "active_probes": len(self._active_probes),
            "node_details": self.get_all_statuses(),
        }
