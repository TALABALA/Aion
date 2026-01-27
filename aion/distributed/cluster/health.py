"""
AION Health Checker -- Phi Accrual Failure Detector

Production-grade implementation of the phi accrual failure detection
algorithm (Hayashibara et al., 2004).  Unlike simple heartbeat-timeout
detectors, the phi accrual approach:

1. Maintains a sliding window of inter-arrival times for each node.
2. Fits an exponential distribution to those intervals.
3. Computes a *suspicion level* (phi) that expresses the probability
   that the monitored node has crashed, on a continuous scale.
4. Compares phi against a configurable threshold to decide liveness.

This yields adaptive, self-tuning failure detection that automatically
adjusts for network jitter and variable heartbeat latencies.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Deque, Dict, Optional

import structlog

if TYPE_CHECKING:
    from aion.distributed.cluster.manager import ClusterManager

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------
_DEFAULT_THRESHOLD = 8.0
_DEFAULT_WINDOW_SIZE = 100
_DEFAULT_MIN_SAMPLES = 3
_DEFAULT_INITIAL_HEARTBEAT_INTERVAL = 1.0  # seconds
_MIN_STD_DEV = 0.1  # floor for standard deviation to avoid division issues


# ---------------------------------------------------------------------------
# Per-node heartbeat arrival state
# ---------------------------------------------------------------------------

@dataclass
class _HeartbeatWindow:
    """Sliding window of heartbeat inter-arrival times for a single node."""

    intervals: Deque[float] = field(default_factory=deque)
    max_size: int = _DEFAULT_WINDOW_SIZE
    last_arrival: Optional[float] = None

    # Pre-computed statistics (updated on every append)
    _mean: float = _DEFAULT_INITIAL_HEARTBEAT_INTERVAL
    _variance: float = 0.0

    # ------------------------------------------------------------------

    def record(self, now: Optional[float] = None) -> None:
        """Record a heartbeat arrival.  Call once per received heartbeat."""
        now = now if now is not None else time.monotonic()
        if self.last_arrival is not None:
            interval = now - self.last_arrival
            if interval > 0:
                self.intervals.append(interval)
                if len(self.intervals) > self.max_size:
                    self.intervals.popleft()
                self._recompute()
        self.last_arrival = now

    def _recompute(self) -> None:
        """Re-derive mean and variance from the window."""
        n = len(self.intervals)
        if n == 0:
            self._mean = _DEFAULT_INITIAL_HEARTBEAT_INTERVAL
            self._variance = 0.0
            return

        total = 0.0
        for v in self.intervals:
            total += v
        self._mean = total / n

        if n < 2:
            self._variance = 0.0
            return

        ss = 0.0
        for v in self.intervals:
            diff = v - self._mean
            ss += diff * diff
        self._variance = ss / (n - 1)

    @property
    def mean(self) -> float:
        return self._mean

    @property
    def std_dev(self) -> float:
        return max(math.sqrt(self._variance), _MIN_STD_DEV)

    @property
    def sample_count(self) -> int:
        return len(self.intervals)

    def time_since_last(self, now: Optional[float] = None) -> float:
        """Seconds since the last heartbeat was recorded."""
        now = now if now is not None else time.monotonic()
        if self.last_arrival is None:
            return float("inf")
        return now - self.last_arrival


# ---------------------------------------------------------------------------
# HealthChecker
# ---------------------------------------------------------------------------

class HealthChecker:
    """
    Phi accrual failure detector for cluster nodes.

    Usage::

        hc = HealthChecker(cluster_manager)

        # On every heartbeat received from node X:
        hc.record_heartbeat("node-x")

        # To query liveness:
        alive = hc.is_alive("node-x")
        phi   = hc.get_phi("node-x")
    """

    def __init__(
        self,
        cluster_manager: ClusterManager,
        *,
        threshold: float = _DEFAULT_THRESHOLD,
        window_size: int = _DEFAULT_WINDOW_SIZE,
        min_samples: int = _DEFAULT_MIN_SAMPLES,
    ) -> None:
        self._manager = cluster_manager
        self._threshold = threshold
        self._window_size = window_size
        self._min_samples = min_samples

        # Per-node heartbeat windows
        self._windows: Dict[str, _HeartbeatWindow] = {}

        logger.info(
            "health_checker.init",
            threshold=self._threshold,
            window_size=self._window_size,
            min_samples=self._min_samples,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def threshold(self) -> float:
        """Phi threshold above which a node is considered failed."""
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        self._threshold = max(0.1, value)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_heartbeat(self, node_id: str) -> None:
        """
        Record a heartbeat arrival from *node_id*.

        Should be called every time a heartbeat (or any proof-of-life
        message) is received from the node.
        """
        window = self._windows.get(node_id)
        if window is None:
            window = _HeartbeatWindow(max_size=self._window_size)
            self._windows[node_id] = window
        window.record()
        logger.debug(
            "health_checker.heartbeat_recorded",
            node_id=node_id,
            samples=window.sample_count,
        )

    def check_health(self, node_id: str) -> bool:
        """
        Return ``True`` if *node_id* is considered healthy.

        This is a convenience wrapper around :meth:`get_phi` that
        compares the computed phi value against the threshold.
        """
        phi = self.get_phi(node_id)
        healthy = phi < self._threshold
        if not healthy:
            logger.warning(
                "health_checker.node_unhealthy",
                node_id=node_id,
                phi=round(phi, 3),
                threshold=self._threshold,
            )
        return healthy

    def get_phi(self, node_id: str) -> float:
        """
        Compute the phi (suspicion) value for *node_id*.

        Phi is defined as::

            phi = -log10(P_later(t_now - t_last))

        where ``P_later`` is the probability that the next heartbeat has
        not yet arrived given the empirical distribution of inter-arrival
        times.

        Higher phi indicates higher suspicion of failure.  Typical
        interpretation:

        * phi < 1  : very likely alive
        * phi ~ 3  : somewhat suspicious
        * phi > 8  : almost certainly dead (default threshold)

        Returns ``0.0`` if there is insufficient data.
        """
        window = self._windows.get(node_id)
        if window is None:
            return 0.0

        if window.sample_count < self._min_samples:
            # Not enough data -- assume alive
            return 0.0

        elapsed = window.time_since_last()
        return self._phi_from_distribution(elapsed, window.mean, window.std_dev)

    def is_alive(self, node_id: str) -> bool:
        """
        Return ``True`` if *node_id* is considered alive
        (phi below threshold).
        """
        return self.get_phi(node_id) < self._threshold

    # ------------------------------------------------------------------
    # Bulk operations
    # ------------------------------------------------------------------

    def get_all_phi(self) -> Dict[str, float]:
        """Return phi values for all tracked nodes."""
        return {nid: self.get_phi(nid) for nid in self._windows}

    def get_suspected_nodes(self) -> list[str]:
        """Return node IDs whose phi exceeds the threshold."""
        return [
            nid
            for nid, phi in self.get_all_phi().items()
            if phi >= self._threshold
        ]

    def remove_node(self, node_id: str) -> None:
        """Stop tracking heartbeats for a departed node."""
        self._windows.pop(node_id, None)
        logger.info("health_checker.node_removed", node_id=node_id)

    def reset(self) -> None:
        """Clear all heartbeat history."""
        self._windows.clear()
        logger.info("health_checker.reset")

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_stats(self, node_id: str) -> Dict[str, Any]:
        """Return detailed heartbeat statistics for a single node."""
        window = self._windows.get(node_id)
        if window is None:
            return {"node_id": node_id, "tracked": False}
        return {
            "node_id": node_id,
            "tracked": True,
            "phi": round(self.get_phi(node_id), 4),
            "alive": self.is_alive(node_id),
            "mean_interval": round(window.mean, 4),
            "std_dev": round(window.std_dev, 4),
            "sample_count": window.sample_count,
            "time_since_last": round(window.time_since_last(), 4),
        }

    def get_summary(self) -> Dict[str, Any]:
        """Return a summary across all tracked nodes."""
        all_phi = self.get_all_phi()
        alive_count = sum(1 for p in all_phi.values() if p < self._threshold)
        return {
            "tracked_nodes": len(self._windows),
            "alive": alive_count,
            "suspected": len(self._windows) - alive_count,
            "threshold": self._threshold,
            "nodes": {
                nid: round(phi, 4) for nid, phi in all_phi.items()
            },
        }

    # ------------------------------------------------------------------
    # Internal: phi accrual computation
    # ------------------------------------------------------------------

    @staticmethod
    def _phi_from_distribution(
        elapsed: float,
        mean: float,
        std_dev: float,
    ) -> float:
        """
        Compute phi using a normal distribution approximation.

        ``P_later(t) = 1 - CDF_normal(t; mean, std_dev)``

        Using the complementary error function for numerical stability::

            P_later = 0.5 * erfc((t - mean) / (std_dev * sqrt(2)))
            phi     = -log10(P_later)
        """
        if std_dev < _MIN_STD_DEV:
            std_dev = _MIN_STD_DEV

        # Standardised value
        y = (elapsed - mean) / std_dev

        # Complementary CDF of the standard normal
        # P_later = 0.5 * erfc(y / sqrt(2))
        p_later = 0.5 * math.erfc(y / math.sqrt(2.0))

        # Clamp to avoid log(0)
        if p_later < 1e-15:
            p_later = 1e-15

        phi = -math.log10(p_later)
        return max(0.0, phi)
