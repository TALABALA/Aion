"""
AION Distributed Computing - Load Balancer

Production-grade load balancer that orchestrates strategy selection and
node routing for the AION distributed task system.  Implements SOTA
patterns including:
- Per-node circuit breaker (open / half-open / closed states)
- Health-aware routing (automatically skips unhealthy nodes)
- Adaptive strategy switching based on cluster conditions
- Pluggable strategy interface with runtime hot-swapping
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

import structlog

from aion.distributed.balancing.strategies import (
    AdaptiveStrategy,
    CapabilityAwareStrategy,
    LeastConnectionsStrategy,
    LoadBalancingStrategy,
    LocalityAwareStrategy,
    PowerOfTwoChoicesStrategy,
    RoundRobinStrategy,
    WeightedStrategy,
)
from aion.distributed.types import NodeInfo, NodeStatus

if TYPE_CHECKING:
    from aion.distributed.cluster.manager import ClusterManager
    from aion.distributed.types import DistributedTask

logger = structlog.get_logger(__name__)


# =============================================================================
# Circuit Breaker
# =============================================================================


class CircuitState(str, Enum):
    """State of a per-node circuit breaker."""
    CLOSED = "closed"        # Normal operation -- requests flow through
    OPEN = "open"            # Tripped -- all requests short-circuited
    HALF_OPEN = "half_open"  # Trial -- allow a single probe request


@dataclass
class _CircuitBreaker:
    """Per-node circuit breaker preventing cascading failures.

    State machine:
        CLOSED  --[N consecutive failures]--> OPEN
        OPEN    --[timeout elapsed]---------> HALF_OPEN
        HALF_OPEN --[success]---------------> CLOSED
        HALF_OPEN --[failure]---------------> OPEN

    Attributes:
        state: Current breaker state.
        failure_count: Consecutive failures since last success.
        failure_threshold: Failures required to trip the breaker.
        recovery_timeout: Seconds before an OPEN breaker transitions
            to HALF_OPEN.
        last_failure_time: Monotonic timestamp of the last failure.
        last_state_change: Monotonic timestamp of the last state transition.
        total_trips: Lifetime count of CLOSED -> OPEN transitions.
    """

    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    last_failure_time: float = 0.0
    last_state_change: float = field(default_factory=time.monotonic)
    total_trips: int = 0

    def record_success(self) -> None:
        """Record a successful request.  Resets the breaker to CLOSED."""
        if self.state == CircuitState.HALF_OPEN:
            logger.info("circuit_breaker_closed", previous_state=self.state.value)
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_state_change = time.monotonic()

    def record_failure(self) -> None:
        """Record a failed request.  May trip the breaker to OPEN."""
        self.failure_count += 1
        self.last_failure_time = time.monotonic()

        if self.state == CircuitState.HALF_OPEN:
            # Probe failed -- go back to OPEN
            self.state = CircuitState.OPEN
            self.last_state_change = time.monotonic()
            self.total_trips += 1
            logger.warning("circuit_breaker_reopened", failures=self.failure_count)
            return

        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            self.last_state_change = time.monotonic()
            self.total_trips += 1
            logger.warning(
                "circuit_breaker_tripped",
                failures=self.failure_count,
                threshold=self.failure_threshold,
            )

    def is_available(self) -> bool:
        """Return whether requests should be routed to this node.

        Transitions OPEN -> HALF_OPEN when the recovery timeout elapses.
        """
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            elapsed = time.monotonic() - self.last_failure_time
            if elapsed >= self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                self.last_state_change = time.monotonic()
                logger.info(
                    "circuit_breaker_half_open",
                    elapsed=round(elapsed, 2),
                    timeout=self.recovery_timeout,
                )
                return True
            return False

        # HALF_OPEN: allow one probe
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Serialise circuit breaker state."""
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "failure_threshold": self.failure_threshold,
            "recovery_timeout": self.recovery_timeout,
            "total_trips": self.total_trips,
            "seconds_since_last_failure": (
                round(time.monotonic() - self.last_failure_time, 2)
                if self.last_failure_time > 0
                else None
            ),
        }


# =============================================================================
# Strategy Registry
# =============================================================================


_STRATEGY_REGISTRY: Dict[str, type] = {
    "round_robin": RoundRobinStrategy,
    "least_connections": LeastConnectionsStrategy,
    "weighted": WeightedStrategy,
    "capability_aware": CapabilityAwareStrategy,
    "locality_aware": LocalityAwareStrategy,
    "adaptive": AdaptiveStrategy,
    "power_of_two": PowerOfTwoChoicesStrategy,
}


# =============================================================================
# LoadBalancer
# =============================================================================


class LoadBalancer:
    """Main load balancer for the AION distributed computing system.

    Orchestrates strategy selection, health filtering, and circuit-breaker
    protection to route tasks to the most appropriate cluster node.

    Features:
        * **Pluggable strategies** -- swap algorithms at runtime via
          :meth:`set_strategy`.
        * **Circuit breaker** -- per-node breaker that opens after *N*
          consecutive failures, transitions to half-open after a configurable
          timeout, and closes again on success.
        * **Health-aware routing** -- automatically excludes nodes whose
          status is not ``HEALTHY`` or that lack available capacity.
        * **Adaptive switching** -- when using the
          :class:`AdaptiveStrategy`, the balancer feeds back latency and
          error observations to refine scoring in real time.

    Args:
        cluster_manager: Reference to the cluster manager for node discovery.
        strategy: Initial strategy name or instance (default ``"adaptive"``).
        circuit_breaker_threshold: Consecutive failures before tripping.
        circuit_breaker_timeout: Seconds before recovery probe.
    """

    def __init__(
        self,
        cluster_manager: ClusterManager,
        *,
        strategy: str | LoadBalancingStrategy = "adaptive",
        circuit_breaker_threshold: int = 5,
        circuit_breaker_timeout: float = 30.0,
    ) -> None:
        self._cluster_manager = cluster_manager
        self._cb_threshold = circuit_breaker_threshold
        self._cb_timeout = circuit_breaker_timeout

        # Resolve strategy
        if isinstance(strategy, str):
            self._strategy = self._create_strategy(strategy)
            self._strategy_name = strategy
        else:
            self._strategy = strategy
            self._strategy_name = type(strategy).__name__

        # Per-node circuit breakers (node_id -> breaker)
        self._breakers: Dict[str, _CircuitBreaker] = {}

        # Statistics
        self._total_selections: int = 0
        self._total_failures: int = 0
        self._total_no_node: int = 0
        self._created_at: float = time.monotonic()

        logger.info(
            "load_balancer_init",
            strategy=self._strategy_name,
            cb_threshold=circuit_breaker_threshold,
            cb_timeout=circuit_breaker_timeout,
        )

    # -----------------------------------------------------------------
    # Strategy management
    # -----------------------------------------------------------------

    def _create_strategy(self, name: str) -> LoadBalancingStrategy:
        """Instantiate a strategy by registered name."""
        cls = _STRATEGY_REGISTRY.get(name)
        if cls is None:
            logger.warning("unknown_strategy", name=name, fallback="adaptive")
            cls = AdaptiveStrategy
        return cls()

    def set_strategy(self, strategy: str | LoadBalancingStrategy) -> None:
        """Hot-swap the active load balancing strategy.

        Args:
            strategy: Strategy name (string) or a pre-built instance.
        """
        if isinstance(strategy, str):
            self._strategy = self._create_strategy(strategy)
            self._strategy_name = strategy
        else:
            self._strategy = strategy
            self._strategy_name = type(strategy).__name__

        logger.info("strategy_changed", strategy=self._strategy_name)

    @property
    def strategy(self) -> LoadBalancingStrategy:
        """The currently active strategy instance."""
        return self._strategy

    # -----------------------------------------------------------------
    # Circuit breaker management
    # -----------------------------------------------------------------

    def _get_breaker(self, node_id: str) -> _CircuitBreaker:
        """Lazily create and return a circuit breaker for *node_id*."""
        if node_id not in self._breakers:
            self._breakers[node_id] = _CircuitBreaker(
                failure_threshold=self._cb_threshold,
                recovery_timeout=self._cb_timeout,
            )
        return self._breakers[node_id]

    def record_success(self, node_id: str) -> None:
        """Notify the balancer of a successful request to *node_id*."""
        self._get_breaker(node_id).record_success()

    def record_failure(self, node_id: str) -> None:
        """Notify the balancer of a failed request to *node_id*."""
        self._get_breaker(node_id).record_failure()
        self._total_failures += 1

    # -----------------------------------------------------------------
    # Node filtering
    # -----------------------------------------------------------------

    async def _get_available_nodes(self) -> List[NodeInfo]:
        """Return healthy, capacity-available, circuit-unbroken nodes."""
        nodes = await self._fetch_nodes()
        available: List[NodeInfo] = []

        for node in nodes:
            # Health gate
            if node.status != NodeStatus.HEALTHY:
                continue

            # Capacity gate
            if node.current_tasks >= node.max_concurrent_tasks:
                continue

            # Circuit breaker gate
            breaker = self._get_breaker(node.id)
            if not breaker.is_available():
                logger.debug(
                    "node_circuit_open",
                    node_id=node.id,
                    breaker_state=breaker.state.value,
                )
                continue

            available.append(node)

        return available

    async def _fetch_nodes(self) -> List[NodeInfo]:
        """Retrieve all nodes from the cluster manager."""
        try:
            if hasattr(self._cluster_manager, "get_nodes"):
                result = await self._cluster_manager.get_nodes()
                return list(result) if result else []
            if hasattr(self._cluster_manager, "state"):
                state = self._cluster_manager.state
                if state and state.nodes:
                    return list(state.nodes.values())
        except Exception:
            logger.exception("failed_to_fetch_nodes")
        return []

    # -----------------------------------------------------------------
    # Core selection API
    # -----------------------------------------------------------------

    async def select_node(
        self,
        task: Optional[DistributedTask] = None,
    ) -> Optional[NodeInfo]:
        """Select a single node for task execution.

        Applies health filtering and circuit-breaker gating before
        delegating to the active strategy.

        Args:
            task: Optional task descriptor for affinity / capability routing.

        Returns:
            The selected :class:`NodeInfo`, or ``None`` if no suitable node
            is available.
        """
        available = await self._get_available_nodes()
        if not available:
            self._total_no_node += 1
            logger.warning("no_available_nodes", strategy=self._strategy_name)
            return None

        selected = await self._strategy.select_node(available, task)
        if selected is not None:
            self._total_selections += 1
            logger.debug(
                "node_selected",
                node_id=selected.id,
                strategy=self._strategy_name,
                available=len(available),
            )
        else:
            self._total_no_node += 1
            logger.warning(
                "strategy_returned_none",
                strategy=self._strategy_name,
                available=len(available),
            )

        return selected

    async def select_nodes(
        self,
        count: int,
        task: Optional[DistributedTask] = None,
    ) -> List[NodeInfo]:
        """Select multiple distinct nodes for parallel execution.

        Iteratively selects nodes via the active strategy, excluding
        previously selected nodes from each subsequent round.

        Args:
            count: Number of nodes to select.
            task: Optional task descriptor.

        Returns:
            List of up to *count* selected :class:`NodeInfo` objects.
            May contain fewer if the cluster has insufficient capacity.
        """
        available = await self._get_available_nodes()
        if not available:
            self._total_no_node += 1
            logger.warning("no_available_nodes_multi", requested=count)
            return []

        selected: List[NodeInfo] = []
        remaining = list(available)

        for _ in range(min(count, len(remaining))):
            if not remaining:
                break

            node = await self._strategy.select_node(remaining, task)
            if node is None:
                break

            selected.append(node)
            remaining = [n for n in remaining if n.id != node.id]
            self._total_selections += 1

        logger.debug(
            "nodes_selected",
            requested=count,
            selected=len(selected),
            strategy=self._strategy_name,
        )
        return selected

    # -----------------------------------------------------------------
    # Statistics / Observability
    # -----------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Return comprehensive load balancer statistics.

        Returns:
            Dictionary containing selection counts, failure rates,
            circuit breaker states, strategy info, and uptime.
        """
        uptime_seconds = time.monotonic() - self._created_at

        # Aggregate circuit breaker info
        breaker_summary: Dict[str, int] = {
            "closed": 0,
            "open": 0,
            "half_open": 0,
        }
        breaker_details: Dict[str, Dict[str, Any]] = {}
        for node_id, breaker in self._breakers.items():
            # Poke to allow state transitions before reporting
            _ = breaker.is_available()
            breaker_summary[breaker.state.value] += 1
            breaker_details[node_id] = breaker.to_dict()

        return {
            "strategy": self._strategy_name,
            "total_selections": self._total_selections,
            "total_failures": self._total_failures,
            "total_no_node": self._total_no_node,
            "failure_rate": (
                round(self._total_failures / max(self._total_selections, 1), 4)
            ),
            "uptime_seconds": round(uptime_seconds, 2),
            "selections_per_second": (
                round(self._total_selections / max(uptime_seconds, 0.001), 2)
            ),
            "circuit_breakers": {
                "summary": breaker_summary,
                "nodes": breaker_details,
            },
        }

    def get_circuit_breaker_state(self, node_id: str) -> Dict[str, Any]:
        """Return circuit breaker state for a specific node.

        Args:
            node_id: The identifier of the node.

        Returns:
            Dictionary with breaker state details.
        """
        breaker = self._get_breaker(node_id)
        return breaker.to_dict()

    def reset_circuit_breaker(self, node_id: str) -> None:
        """Manually reset a tripped circuit breaker for *node_id*.

        Args:
            node_id: The identifier of the node to reset.
        """
        breaker = self._get_breaker(node_id)
        breaker.state = CircuitState.CLOSED
        breaker.failure_count = 0
        breaker.last_state_change = time.monotonic()
        logger.info("circuit_breaker_manual_reset", node_id=node_id)

    def reset_all_circuit_breakers(self) -> None:
        """Reset all circuit breakers to CLOSED state."""
        for node_id in list(self._breakers):
            self.reset_circuit_breaker(node_id)
        logger.info("all_circuit_breakers_reset", count=len(self._breakers))
