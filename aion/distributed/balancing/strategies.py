"""
AION Distributed Computing - Load Balancing Strategies

Production-grade load balancing strategies for distributing tasks across
cluster nodes. Implements SOTA patterns including:
- Round-robin with index tracking
- Least-connections routing
- Weighted random distribution
- Capability-aware filtering
- Locality-aware placement (zone / region preference)
- Adaptive multi-signal scoring with EWMA latency tracking
- Power-of-two-choices (proven near-optimal randomised selection)
"""

from __future__ import annotations

import abc
import math
import random
import time
from collections import defaultdict, deque
from typing import TYPE_CHECKING, Any, Deque, Dict, List, Optional, Sequence

import structlog

if TYPE_CHECKING:
    from aion.distributed.types import DistributedTask, NodeInfo

logger = structlog.get_logger(__name__)


# =============================================================================
# Abstract Base Strategy
# =============================================================================


class LoadBalancingStrategy(abc.ABC):
    """Abstract base class for all load balancing strategies.

    Every concrete strategy must implement :meth:`select_node` which receives
    a list of *available* (healthy, capacity-remaining) nodes and an optional
    task descriptor used for affinity / capability filtering.
    """

    @abc.abstractmethod
    async def select_node(
        self,
        nodes: Sequence[NodeInfo],
        task: Optional[DistributedTask] = None,
    ) -> Optional[NodeInfo]:
        """Select the best node from *nodes* for the given *task*.

        Args:
            nodes: Pre-filtered list of available cluster nodes.
            task: Optional task descriptor for capability / affinity matching.

        Returns:
            The selected :class:`NodeInfo`, or ``None`` when no suitable node
            can be found.
        """

    def __repr__(self) -> str:  # pragma: no cover
        return f"<{self.__class__.__name__}>"


# =============================================================================
# Round-Robin Strategy
# =============================================================================


class RoundRobinStrategy(LoadBalancingStrategy):
    """Simple round-robin strategy with wrapping index.

    Distributes requests evenly by cycling through the available node list in
    order. The internal counter wraps automatically and is resilient to node
    list size changes between invocations.
    """

    def __init__(self) -> None:
        self._index: int = 0

    async def select_node(
        self,
        nodes: Sequence[NodeInfo],
        task: Optional[DistributedTask] = None,
    ) -> Optional[NodeInfo]:
        if not nodes:
            return None
        idx = self._index % len(nodes)
        self._index = (self._index + 1) % max(len(nodes), 1)
        selected = nodes[idx]
        logger.debug(
            "round_robin_selected",
            node_id=selected.id,
            index=idx,
            total=len(nodes),
        )
        return selected


# =============================================================================
# Least-Connections Strategy
# =============================================================================


class LeastConnectionsStrategy(LoadBalancingStrategy):
    """Select the node with the fewest active tasks.

    Ties are broken by choosing the node with the lower ``load_score`` which
    blends CPU and memory utilisation, providing a more nuanced secondary
    signal than simple task count alone.
    """

    async def select_node(
        self,
        nodes: Sequence[NodeInfo],
        task: Optional[DistributedTask] = None,
    ) -> Optional[NodeInfo]:
        if not nodes:
            return None
        selected = min(nodes, key=lambda n: (n.current_tasks, n.load_score))
        logger.debug(
            "least_connections_selected",
            node_id=selected.id,
            current_tasks=selected.current_tasks,
            load_score=round(selected.load_score, 4),
        )
        return selected


# =============================================================================
# Weighted Random Strategy
# =============================================================================


class WeightedStrategy(LoadBalancingStrategy):
    """Weighted random strategy based on spare capacity.

    The probability of selecting a node is proportional to::

        weight = (1.0 - load_score) * max_concurrent_tasks

    Nodes under lighter load and with larger task capacity are therefore more
    likely to be chosen, providing a natural self-balancing effect.
    """

    async def select_node(
        self,
        nodes: Sequence[NodeInfo],
        task: Optional[DistributedTask] = None,
    ) -> Optional[NodeInfo]:
        if not nodes:
            return None

        weights: List[float] = []
        for node in nodes:
            w = (1.0 - node.load_score) * max(node.max_concurrent_tasks, 1)
            weights.append(max(w, 0.01))  # floor to avoid zero weights

        total = sum(weights)
        if total <= 0:
            return nodes[0] if nodes else None

        # Weighted random selection using cumulative distribution
        r = random.random() * total
        cumulative = 0.0
        for node, weight in zip(nodes, weights):
            cumulative += weight
            if r <= cumulative:
                logger.debug(
                    "weighted_selected",
                    node_id=node.id,
                    weight=round(weight, 4),
                    total_weight=round(total, 4),
                )
                return node

        # Fallback (should not happen due to float precision)
        return nodes[-1]


# =============================================================================
# Capability-Aware Strategy
# =============================================================================


class CapabilityAwareStrategy(LoadBalancingStrategy):
    """Filters nodes by task-required capabilities, then delegates.

    If the task specifies ``required_capabilities``, only nodes whose
    capability set is a superset will be considered. The filtered list is
    passed to a configurable *fallback* strategy (defaulting to
    :class:`LeastConnectionsStrategy`).
    """

    def __init__(
        self,
        fallback: Optional[LoadBalancingStrategy] = None,
    ) -> None:
        self._fallback = fallback or LeastConnectionsStrategy()

    async def select_node(
        self,
        nodes: Sequence[NodeInfo],
        task: Optional[DistributedTask] = None,
    ) -> Optional[NodeInfo]:
        if not nodes:
            return None

        if task is not None and task.required_capabilities:
            required = task.required_capabilities
            capable = [
                n for n in nodes
                if required.issubset(n.capabilities)
            ]
            logger.debug(
                "capability_filter",
                required=sorted(required),
                candidates_before=len(nodes),
                candidates_after=len(capable),
            )
            if not capable:
                logger.warning(
                    "no_capable_nodes",
                    required=sorted(required),
                    available_capabilities=[
                        sorted(n.capabilities) for n in nodes
                    ],
                )
                return None
            nodes = capable

        return await self._fallback.select_node(nodes, task)


# =============================================================================
# Locality-Aware Strategy
# =============================================================================


class LocalityAwareStrategy(LoadBalancingStrategy):
    """Prefer nodes in the same zone, then same region, then any.

    Locality tiers:
    1. Same *zone* (lowest latency, e.g. same rack / availability zone).
    2. Same *region* (moderate latency, e.g. same data-centre region).
    3. Any available node (cross-region fallback).

    Within each tier, a *fallback* strategy (default
    :class:`LeastConnectionsStrategy`) determines the final pick.
    """

    def __init__(
        self,
        local_zone: str = "",
        local_region: str = "",
        fallback: Optional[LoadBalancingStrategy] = None,
    ) -> None:
        self._local_zone = local_zone
        self._local_region = local_region
        self._fallback = fallback or LeastConnectionsStrategy()

    def set_locality(self, zone: str, region: str) -> None:
        """Update the local topology identifiers at runtime."""
        self._local_zone = zone
        self._local_region = region

    async def select_node(
        self,
        nodes: Sequence[NodeInfo],
        task: Optional[DistributedTask] = None,
    ) -> Optional[NodeInfo]:
        if not nodes:
            return None

        # Tier 1: same zone
        if self._local_zone:
            same_zone = [n for n in nodes if n.zone == self._local_zone]
            if same_zone:
                logger.debug(
                    "locality_same_zone",
                    zone=self._local_zone,
                    count=len(same_zone),
                )
                return await self._fallback.select_node(same_zone, task)

        # Tier 2: same region
        if self._local_region:
            same_region = [n for n in nodes if n.region == self._local_region]
            if same_region:
                logger.debug(
                    "locality_same_region",
                    region=self._local_region,
                    count=len(same_region),
                )
                return await self._fallback.select_node(same_region, task)

        # Tier 3: any node
        logger.debug("locality_fallback_any", count=len(nodes))
        return await self._fallback.select_node(nodes, task)


# =============================================================================
# Adaptive Strategy (SOTA)
# =============================================================================


class AdaptiveStrategy(LoadBalancingStrategy):
    """Adaptive multi-signal load balancing with EWMA smoothing.

    Combines several real-time signals into a composite score for every node:

    * **Health score** -- binary healthy / unhealthy from the node's own
      ``load_score`` (inverted so lower load = higher score).
    * **Load signal** -- ``1 - load_score`` giving preference to idle nodes.
    * **Latency signal** -- smoothed via an exponentially weighted moving
      average (EWMA) with a configurable *decay factor*.  Normalised against
      the observed maximum latency.
    * **Error rate signal** -- sliding-window error rate per node.

    Final score::

        score = (health_weight * health)
              + (load_weight   * (1 - load))
              + (latency_weight * (1 - normalised_latency))

    The node with the **highest** composite score is selected.

    Parameters:
        health_weight:  Weight for health signal.  Default ``0.3``.
        load_weight:    Weight for load signal.    Default ``0.4``.
        latency_weight: Weight for latency signal. Default ``0.3``.
        ewma_decay:     EWMA decay factor (alpha). Default ``0.3``.
        error_window:   Size of the per-node sliding error window.
    """

    def __init__(
        self,
        health_weight: float = 0.3,
        load_weight: float = 0.4,
        latency_weight: float = 0.3,
        ewma_decay: float = 0.3,
        error_window: int = 100,
    ) -> None:
        self._health_weight = health_weight
        self._load_weight = load_weight
        self._latency_weight = latency_weight
        self._ewma_decay = ewma_decay
        self._error_window = error_window

        # Per-node EWMA latency (node_id -> smoothed latency in ms)
        self._latency_ewma: Dict[str, float] = {}
        # Per-node sliding window of success/failure booleans
        self._error_windows: Dict[str, Deque[bool]] = defaultdict(
            lambda: deque(maxlen=error_window),
        )

    # -- Public helpers used by LoadMetrics / LoadBalancer --------------------

    def record_latency(self, node_id: str, latency_ms: float) -> None:
        """Record an observed request latency for *node_id*."""
        prev = self._latency_ewma.get(node_id)
        alpha = self._ewma_decay
        if prev is None:
            self._latency_ewma[node_id] = latency_ms
        else:
            self._latency_ewma[node_id] = alpha * latency_ms + (1 - alpha) * prev

    def record_error(self, node_id: str, is_error: bool = True) -> None:
        """Record a success (``False``) or error (``True``) for *node_id*."""
        self._error_windows[node_id].append(is_error)

    def get_error_rate(self, node_id: str) -> float:
        """Return the current error rate for *node_id* (0.0 - 1.0)."""
        window = self._error_windows.get(node_id)
        if not window:
            return 0.0
        return sum(1 for e in window if e) / len(window)

    def get_ewma_latency(self, node_id: str) -> float:
        """Return the current EWMA latency for *node_id* in milliseconds."""
        return self._latency_ewma.get(node_id, 0.0)

    # -- Strategy implementation ---------------------------------------------

    def _score_node(self, node: NodeInfo, max_latency: float) -> float:
        """Compute composite score for a single node (higher is better)."""
        # Health signal: healthy nodes get 1.0, unhealthy get 0.0
        health = 1.0 if node.load_score < 0.95 else 0.0

        # Load signal: invert load_score so idle nodes score higher
        load_signal = 1.0 - node.load_score

        # Latency signal: normalise by max observed EWMA latency
        node_latency = self._latency_ewma.get(node.id, 0.0)
        if max_latency > 0:
            normalised_latency = min(node_latency / max_latency, 1.0)
        else:
            normalised_latency = 0.0

        # Error penalty: reduce score proportionally to error rate
        error_rate = self.get_error_rate(node.id)
        error_penalty = 1.0 - error_rate

        score = (
            self._health_weight * health
            + self._load_weight * load_signal
            + self._latency_weight * (1.0 - normalised_latency)
        )
        # Apply error penalty as a multiplicative factor
        score *= error_penalty

        return score

    async def select_node(
        self,
        nodes: Sequence[NodeInfo],
        task: Optional[DistributedTask] = None,
    ) -> Optional[NodeInfo]:
        if not nodes:
            return None

        # Determine max EWMA latency across candidates for normalisation
        max_latency = max(
            (self._latency_ewma.get(n.id, 0.0) for n in nodes),
            default=0.0,
        )

        best_node: Optional[NodeInfo] = None
        best_score: float = -1.0

        for node in nodes:
            score = self._score_node(node, max_latency)
            if score > best_score:
                best_score = score
                best_node = node

        if best_node is not None:
            logger.debug(
                "adaptive_selected",
                node_id=best_node.id,
                score=round(best_score, 4),
                ewma_latency=round(self._latency_ewma.get(best_node.id, 0.0), 2),
                error_rate=round(self.get_error_rate(best_node.id), 4),
            )

        return best_node


# =============================================================================
# Power-of-Two-Choices Strategy
# =============================================================================


class PowerOfTwoChoicesStrategy(LoadBalancingStrategy):
    """Pick two random nodes, select the one with lower load.

    This deceptively simple algorithm is proven to achieve near-optimal load
    distribution (the "power of two choices" phenomenon).  By only sampling
    two candidates the overhead is O(1) regardless of cluster size, while
    avoiding the herd-behaviour problems of pure least-connections.

    Reference:
        Mitzenmacher, M. (2001). *The Power of Two Choices in Randomized
        Load Balancing*. IEEE TPDS.
    """

    async def select_node(
        self,
        nodes: Sequence[NodeInfo],
        task: Optional[DistributedTask] = None,
    ) -> Optional[NodeInfo]:
        if not nodes:
            return None
        if len(nodes) == 1:
            return nodes[0]

        # Sample two distinct candidates at random
        a, b = random.sample(list(nodes), k=min(2, len(nodes)))

        # Select the less loaded one (prefer lower current_tasks, then load_score)
        if (a.current_tasks, a.load_score) <= (b.current_tasks, b.load_score):
            selected, rejected = a, b
        else:
            selected, rejected = b, a

        logger.debug(
            "p2c_selected",
            selected_id=selected.id,
            selected_tasks=selected.current_tasks,
            rejected_id=rejected.id,
            rejected_tasks=rejected.current_tasks,
        )
        return selected
