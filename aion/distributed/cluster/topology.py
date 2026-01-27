"""
AION Cluster Topology

Manages the physical and logical topology of the AION cluster:

- **Regions, Zones, and Racks**: three-level failure-domain hierarchy for
  replica placement.
- **Placement groups**: sets of nodes that share replicas, constructed to
  maximise spread across failure domains.
- **Spread scoring**: quantitative measure of how well replicas are
  distributed across the topology.
- **Topology-aware queries**: retrieve nodes filtered by region, zone, or
  label selectors.

The topology information is derived entirely from ``NodeInfo`` metadata
stored in the ``ClusterState``.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    FrozenSet,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

import structlog

from aion.distributed.types import (
    ClusterState,
    NodeInfo,
    NodeStatus,
)

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class PlacementGroup:
    """
    A placement group is a logical set of nodes chosen to host replicas
    of the same data partition.  Good placement maximises the number of
    distinct failure domains covered.
    """
    group_id: str = ""
    node_ids: List[str] = field(default_factory=list)
    regions: Set[str] = field(default_factory=set)
    zones: Set[str] = field(default_factory=set)
    racks: Set[str] = field(default_factory=set)

    @property
    def domain_count(self) -> int:
        """Number of distinct failure domains (racks) covered."""
        return len(self.racks) if self.racks else len(self.zones) or len(self.regions)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "group_id": self.group_id,
            "node_ids": self.node_ids,
            "regions": sorted(self.regions),
            "zones": sorted(self.zones),
            "racks": sorted(self.racks),
            "domain_count": self.domain_count,
        }


@dataclass
class TopologySummary:
    """Human-readable summary of the topology."""
    total_nodes: int = 0
    regions: Dict[str, int] = field(default_factory=dict)
    zones: Dict[str, int] = field(default_factory=dict)
    racks: Dict[str, int] = field(default_factory=dict)
    spread_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_nodes": self.total_nodes,
            "regions": self.regions,
            "zones": self.zones,
            "racks": self.racks,
            "spread_score": round(self.spread_score, 4),
        }


# ---------------------------------------------------------------------------
# ClusterTopology
# ---------------------------------------------------------------------------

class ClusterTopology:
    """
    Read-only view of the cluster's physical / logical topology.

    Constructed from a ``ClusterState`` reference and re-derived on
    every query so that callers always see a consistent snapshot.

    Usage::

        topo = ClusterTopology(cluster_state)
        nodes = topo.get_nodes_in_region("us-east-1")
        score = topo.calculate_spread_score()
    """

    def __init__(self, cluster_state: ClusterState) -> None:
        self._state = cluster_state
        logger.info(
            "cluster_topology.init",
            nodes=len(self._state.nodes),
        )

    # ------------------------------------------------------------------
    # Node queries
    # ------------------------------------------------------------------

    def get_all_nodes(self, healthy_only: bool = False) -> List[NodeInfo]:
        """Return all nodes, optionally filtering to healthy ones."""
        nodes = list(self._state.nodes.values())
        if healthy_only:
            nodes = [n for n in nodes if n.status == NodeStatus.HEALTHY]
        return nodes

    def get_nodes_in_region(self, region: str) -> List[NodeInfo]:
        """Return all nodes whose ``region`` matches."""
        return [
            n for n in self._state.nodes.values()
            if n.region == region
        ]

    def get_nodes_in_zone(self, zone: str) -> List[NodeInfo]:
        """Return all nodes whose ``zone`` matches."""
        return [
            n for n in self._state.nodes.values()
            if n.zone == zone
        ]

    def get_nodes_in_rack(self, rack: str) -> List[NodeInfo]:
        """Return all nodes whose ``rack`` matches."""
        return [
            n for n in self._state.nodes.values()
            if n.rack == rack
        ]

    def get_nodes_by_label(self, key: str, value: str) -> List[NodeInfo]:
        """Return nodes whose labels contain ``key=value``."""
        return [
            n for n in self._state.nodes.values()
            if n.labels.get(key) == value
        ]

    def get_nodes_with_capability(self, capability: str) -> List[NodeInfo]:
        """Return nodes that advertise a specific capability."""
        return [
            n for n in self._state.nodes.values()
            if capability in n.capabilities
        ]

    # ------------------------------------------------------------------
    # Region / zone / rack enumeration
    # ------------------------------------------------------------------

    def get_regions(self) -> List[str]:
        """Return the distinct regions present in the cluster."""
        return sorted({n.region for n in self._state.nodes.values() if n.region})

    def get_zones(self, region: Optional[str] = None) -> List[str]:
        """Return distinct zones, optionally filtered by region."""
        nodes = self._state.nodes.values()
        if region is not None:
            nodes = [n for n in nodes if n.region == region]
        return sorted({n.zone for n in nodes if n.zone})

    def get_racks(
        self,
        region: Optional[str] = None,
        zone: Optional[str] = None,
    ) -> List[str]:
        """Return distinct racks, optionally filtered by region and zone."""
        nodes = list(self._state.nodes.values())
        if region is not None:
            nodes = [n for n in nodes if n.region == region]
        if zone is not None:
            nodes = [n for n in nodes if n.zone == zone]
        return sorted({n.rack for n in nodes if n.rack})

    # ------------------------------------------------------------------
    # Placement groups
    # ------------------------------------------------------------------

    def get_placement_groups(
        self,
        replication_factor: Optional[int] = None,
    ) -> List[PlacementGroup]:
        """
        Build placement groups that maximise failure-domain spread.

        Each group contains up to ``replication_factor`` nodes (defaults
        to the cluster state's ``replication_factor``).  The algorithm
        greedily selects nodes from the least-used failure domain first.
        """
        rf = replication_factor or self._state.replication_factor
        healthy = [
            n for n in self._state.nodes.values()
            if n.status == NodeStatus.HEALTHY
        ]
        if not healthy:
            return []

        # Bucket nodes by their finest-grained failure domain
        domain_buckets: Dict[str, List[NodeInfo]] = defaultdict(list)
        for node in healthy:
            domain = self._failure_domain(node)
            domain_buckets[domain].append(node)

        # Round-robin across domains to build groups
        groups: List[PlacementGroup] = []
        used: Set[str] = set()
        group_idx = 0

        # Sort domains by their size (ascending) so smaller domains get
        # represented early, improving spread.
        domain_order = sorted(domain_buckets.keys(), key=lambda d: len(domain_buckets[d]))

        while len(used) < len(healthy):
            pg = PlacementGroup(group_id=f"pg-{group_idx}")
            visited_domains: Set[str] = set()

            for domain in domain_order:
                if len(pg.node_ids) >= rf:
                    break
                for node in domain_buckets[domain]:
                    if node.id in used and len(used) < len(healthy):
                        continue
                    if node.id in {nid for nid in pg.node_ids}:
                        continue
                    if domain in visited_domains and len(domain_order) > len(pg.node_ids):
                        # Try to pick from a different domain first
                        continue
                    pg.node_ids.append(node.id)
                    used.add(node.id)
                    visited_domains.add(domain)
                    if node.region:
                        pg.regions.add(node.region)
                    if node.zone:
                        pg.zones.add(node.zone)
                    if node.rack:
                        pg.racks.add(node.rack)
                    if len(pg.node_ids) >= rf:
                        break

            # If we still need more members, relax the domain constraint
            if len(pg.node_ids) < rf:
                for node in healthy:
                    if node.id in {nid for nid in pg.node_ids}:
                        continue
                    pg.node_ids.append(node.id)
                    used.add(node.id)
                    if node.region:
                        pg.regions.add(node.region)
                    if node.zone:
                        pg.zones.add(node.zone)
                    if node.rack:
                        pg.racks.add(node.rack)
                    if len(pg.node_ids) >= rf:
                        break

            if pg.node_ids:
                groups.append(pg)
            else:
                break  # safety: no progress
            group_idx += 1

        logger.debug(
            "cluster_topology.placement_groups_built",
            groups=len(groups),
            replication_factor=rf,
        )
        return groups

    # ------------------------------------------------------------------
    # Spread scoring
    # ------------------------------------------------------------------

    def calculate_spread_score(self) -> float:
        """
        Compute a spread score in the range ``[0.0, 1.0]``.

        The score measures how evenly nodes are distributed across
        failure domains.  A perfectly balanced cluster scores ``1.0``;
        all nodes in a single domain scores close to ``0.0``.

        The metric is based on the normalised entropy of the domain
        distribution::

            H = -sum(p_i * log2(p_i))   for each domain i
            score = H / log2(k)          where k = number of domains

        If there is only one domain the score is ``0.0``.
        """
        domain_counts: Dict[str, int] = defaultdict(int)
        for node in self._state.nodes.values():
            domain = self._failure_domain(node)
            domain_counts[domain] += 1

        k = len(domain_counts)
        if k <= 1:
            return 0.0

        total = sum(domain_counts.values())
        entropy = 0.0
        for count in domain_counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)

        max_entropy = math.log2(k)
        if max_entropy == 0:
            return 0.0

        return entropy / max_entropy

    # ------------------------------------------------------------------
    # Topology summary
    # ------------------------------------------------------------------

    def get_summary(self) -> TopologySummary:
        """Build a ``TopologySummary`` of the current cluster."""
        region_counts: Dict[str, int] = defaultdict(int)
        zone_counts: Dict[str, int] = defaultdict(int)
        rack_counts: Dict[str, int] = defaultdict(int)

        for node in self._state.nodes.values():
            if node.region:
                region_counts[node.region] += 1
            if node.zone:
                zone_counts[node.zone] += 1
            if node.rack:
                rack_counts[node.rack] += 1

        return TopologySummary(
            total_nodes=len(self._state.nodes),
            regions=dict(region_counts),
            zones=dict(zone_counts),
            racks=dict(rack_counts),
            spread_score=self.calculate_spread_score(),
        )

    # ------------------------------------------------------------------
    # Topology-aware node selection
    # ------------------------------------------------------------------

    def select_diverse_nodes(
        self,
        count: int,
        exclude: Optional[Set[str]] = None,
        region: Optional[str] = None,
    ) -> List[NodeInfo]:
        """
        Select up to *count* healthy nodes maximising failure-domain
        diversity.

        Useful for choosing replica targets or scatter-gather
        destinations.
        """
        exclude = exclude or set()
        candidates = [
            n for n in self._state.nodes.values()
            if n.status == NodeStatus.HEALTHY
            and n.id not in exclude
            and (region is None or n.region == region)
        ]

        if not candidates:
            return []

        # Greedily pick from least-represented domains
        selected: List[NodeInfo] = []
        used_domains: Dict[str, int] = defaultdict(int)

        # Sort candidates so that nodes in rarer domains come first
        domain_sizes: Dict[str, int] = defaultdict(int)
        for c in candidates:
            domain_sizes[self._failure_domain(c)] += 1

        candidates.sort(key=lambda n: (
            used_domains.get(self._failure_domain(n), 0),
            domain_sizes.get(self._failure_domain(n), 0),
            n.load_score,
        ))

        for node in candidates:
            if len(selected) >= count:
                break
            domain = self._failure_domain(node)
            selected.append(node)
            used_domains[domain] += 1
            # Re-sort remaining candidates after each pick
            candidates.sort(key=lambda n: (
                used_domains.get(self._failure_domain(n), 0),
                domain_sizes.get(self._failure_domain(n), 0),
                n.load_score,
            ))

        return selected[:count]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _failure_domain(node: NodeInfo) -> str:
        """
        Return the finest-grained failure domain label for a node.

        Preference order: rack > zone > region > ``"default"``.
        """
        if node.rack:
            return f"{node.region}/{node.zone}/{node.rack}"
        if node.zone:
            return f"{node.region}/{node.zone}"
        if node.region:
            return node.region
        return "default"
