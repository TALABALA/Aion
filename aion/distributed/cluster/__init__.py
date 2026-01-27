"""
AION Distributed Cluster Management

Public API surface for the cluster sub-package:

- :class:`ClusterManager`  -- central coordinator (node lifecycle, task
  distribution, background loops)
- :class:`NodeDiscovery`   -- pluggable peer discovery (static, DNS,
  multicast)
- :class:`HealthChecker`   -- phi accrual failure detector
- :class:`ClusterTopology` -- topology-aware placement and queries
"""

from __future__ import annotations

from aion.distributed.cluster.discovery import NodeDiscovery
from aion.distributed.cluster.health import HealthChecker
from aion.distributed.cluster.manager import ClusterManager
from aion.distributed.cluster.topology import ClusterTopology, PlacementGroup, TopologySummary

__all__ = [
    "ClusterManager",
    "ClusterTopology",
    "HealthChecker",
    "NodeDiscovery",
    "PlacementGroup",
    "TopologySummary",
]
