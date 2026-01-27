"""
AION Distributed Computing - Load Balancing

Production-grade load balancing infrastructure providing:
- Pluggable strategy-based node selection (round-robin, least-connections,
  weighted, capability-aware, locality-aware, adaptive, power-of-two)
- Per-node circuit breakers with open / half-open / closed state machine
- Health-aware routing with automatic unhealthy-node exclusion
- Cluster-wide load metrics with latency histograms and hotspot detection
"""

from aion.distributed.balancing.load_balancer import LoadBalancer
from aion.distributed.balancing.metrics import LoadMetrics
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

__all__ = [
    "LoadBalancer",
    "LoadMetrics",
    "LoadBalancingStrategy",
    "RoundRobinStrategy",
    "LeastConnectionsStrategy",
    "WeightedStrategy",
    "CapabilityAwareStrategy",
    "LocalityAwareStrategy",
    "AdaptiveStrategy",
    "PowerOfTwoChoicesStrategy",
]
