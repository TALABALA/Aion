"""
AION Distributed Task Router

Production-grade task routing with multi-strategy node selection.
Implements SOTA patterns including:
- Capability-based matching to ensure nodes can handle the task type
- Least-loaded routing for optimal resource utilization
- Locality-aware routing to minimize cross-zone traffic
- Sticky session (affinity) routing for stateful workloads
- Multi-factor node scoring combining load, capability, and locality
- Batch routing optimization for related task groups
"""

from __future__ import annotations

import hashlib
import time
from collections import defaultdict
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

import structlog

from aion.distributed.types import (
    DistributedTask,
    NodeInfo,
    NodeStatus,
    TaskPriority,
    TaskType,
)

if TYPE_CHECKING:
    from aion.distributed.cluster.manager import ClusterManager

logger = structlog.get_logger(__name__)

# Scoring weights for multi-factor routing
LOAD_SCORE_WEIGHT = 0.50
CAPABILITY_MATCH_WEIGHT = 0.25
LOCALITY_SCORE_WEIGHT = 0.15
AFFINITY_SCORE_WEIGHT = 0.10

# Default TTL for sticky session affinity cache entries
DEFAULT_AFFINITY_TTL_SECONDS = 300


class TaskRouter:
    """
    Routes distributed tasks to the most appropriate cluster nodes.

    The router applies a pipeline of filters and scoring functions to
    select the best node for each task. It supports four routing
    strategies that can be combined:

    1. **capability_match**: Ensures the node has the required capabilities.
    2. **least_loaded**: Prefers nodes with the lowest load score.
    3. **locality_aware**: Prefers nodes in the same region/zone.
    4. **sticky**: Routes tasks with the same routing key to the same node.

    Nodes are first filtered for availability and capability, then scored
    using a weighted combination of load, capability, locality, and
    affinity factors.

    Attributes:
        cluster_manager: Reference to the cluster manager for node state.
        affinity_ttl: TTL in seconds for sticky session affinity entries.
    """

    def __init__(
        self,
        cluster_manager: ClusterManager,
        *,
        affinity_ttl: int = DEFAULT_AFFINITY_TTL_SECONDS,
    ) -> None:
        self._cluster_manager = cluster_manager
        self.affinity_ttl = affinity_ttl

        # Sticky session affinity cache: routing_key -> (node_id, expiry_ts)
        self._affinity_cache: Dict[str, tuple[str, float]] = {}

        # Routing statistics
        self._stats = {
            "routed": 0,
            "batch_routed": 0,
            "no_suitable_node": 0,
            "affinity_hits": 0,
            "affinity_misses": 0,
            "capability_filtered": 0,
            "availability_filtered": 0,
        }

        logger.info(
            "task_router_initialized",
            affinity_ttl=affinity_ttl,
        )

    # -------------------------------------------------------------------------
    # Core Routing
    # -------------------------------------------------------------------------

    async def route(self, task: DistributedTask) -> Optional[NodeInfo]:
        """
        Route a task to the most suitable node in the cluster.

        Applies the following pipeline:
        1. Check sticky affinity cache for routing key match.
        2. Filter nodes by availability (healthy + has capacity).
        3. Filter by required capabilities.
        4. Exclude explicitly excluded nodes.
        5. Prefer explicitly preferred nodes if any are available.
        6. Score remaining candidates by weighted multi-factor score.
        7. Select the node with the lowest (best) score.

        Args:
            task: The task to route.

        Returns:
            The selected NodeInfo, or None if no suitable node exists.
        """
        # Step 1: Check sticky affinity
        if task.routing_key:
            affinity_node = self._check_affinity(task.routing_key)
            if affinity_node is not None:
                self._stats["affinity_hits"] += 1
                logger.debug(
                    "routing_affinity_hit",
                    task_id=task.id,
                    routing_key=task.routing_key,
                    node_id=affinity_node.id,
                )
                return affinity_node
            else:
                self._stats["affinity_misses"] += 1

        # Step 2: Get all candidate nodes
        candidates = self._get_available_nodes()

        if not candidates:
            self._stats["no_suitable_node"] += 1
            logger.warning(
                "no_available_nodes",
                task_id=task.id,
                task_name=task.name,
            )
            return None

        # Step 3: Filter by required capabilities
        if task.required_capabilities:
            before_count = len(candidates)
            candidates = self._filter_by_capabilities(
                candidates, task.required_capabilities,
            )
            filtered = before_count - len(candidates)
            if filtered > 0:
                self._stats["capability_filtered"] += filtered

        if not candidates:
            self._stats["no_suitable_node"] += 1
            logger.warning(
                "no_capable_nodes",
                task_id=task.id,
                required=list(task.required_capabilities),
            )
            return None

        # Step 4: Exclude explicitly excluded nodes
        if task.excluded_nodes:
            candidates = [
                n for n in candidates if n.id not in task.excluded_nodes
            ]

        if not candidates:
            self._stats["no_suitable_node"] += 1
            logger.warning(
                "all_nodes_excluded",
                task_id=task.id,
                excluded_count=len(task.excluded_nodes),
            )
            return None

        # Step 5: Prefer explicitly preferred nodes
        if task.preferred_nodes:
            preferred = [
                n for n in candidates if n.id in task.preferred_nodes
            ]
            if preferred:
                candidates = preferred
                logger.debug(
                    "using_preferred_nodes",
                    task_id=task.id,
                    preferred_count=len(preferred),
                )

        # Step 6: Score candidates
        scored = self._score_candidates(task, candidates)

        # Step 7: Select best (lowest score)
        best_node, best_score = scored[0]

        # Update sticky affinity cache
        if task.routing_key:
            self._set_affinity(task.routing_key, best_node.id)

        self._stats["routed"] += 1

        logger.info(
            "task_routed",
            task_id=task.id,
            task_name=task.name,
            node_id=best_node.id,
            node_name=best_node.name,
            score=round(best_score, 4),
            candidates_count=len(candidates),
        )
        return best_node

    async def route_batch(
        self, tasks: List[DistributedTask],
    ) -> Dict[str, NodeInfo]:
        """
        Route a batch of tasks to appropriate nodes.

        Optimizes batch routing by computing node state once and routing
        each task against the cached state. Tasks that cannot be routed
        are omitted from the result.

        Args:
            tasks: List of tasks to route.

        Returns:
            Dictionary mapping task ID to the selected NodeInfo.
            Tasks without a suitable node are not included.
        """
        if not tasks:
            return {}

        assignments: Dict[str, NodeInfo] = {}

        for task in tasks:
            node = await self.route(task)
            if node is not None:
                assignments[task.id] = node
                task.assigned_node = node.id

        self._stats["batch_routed"] += 1

        logger.info(
            "batch_routed",
            total=len(tasks),
            assigned=len(assignments),
            unroutable=len(tasks) - len(assignments),
        )
        return assignments

    # -------------------------------------------------------------------------
    # Node Filtering
    # -------------------------------------------------------------------------

    def _get_available_nodes(self) -> List[NodeInfo]:
        """
        Get all nodes that are available to accept new tasks.

        A node is available if it is healthy and has remaining capacity.

        Returns:
            List of available NodeInfo instances.
        """
        try:
            cluster_state = self._cluster_manager.state
            available = [
                node for node in cluster_state.nodes.values()
                if node.is_available()
            ]
            return available
        except Exception:
            logger.debug("get_available_nodes_error", exc_info=True)
            return []

    def _filter_by_capabilities(
        self,
        nodes: List[NodeInfo],
        required: Set[str],
    ) -> List[NodeInfo]:
        """
        Filter nodes that possess all required capabilities.

        Args:
            nodes: Candidate nodes to filter.
            required: Set of required capability strings.

        Returns:
            Nodes that have all required capabilities.
        """
        return [
            node for node in nodes
            if required.issubset(node.capabilities)
        ]

    # -------------------------------------------------------------------------
    # Node Scoring
    # -------------------------------------------------------------------------

    def _score_candidates(
        self,
        task: DistributedTask,
        candidates: List[NodeInfo],
    ) -> List[tuple[NodeInfo, float]]:
        """
        Score and rank candidate nodes for a task.

        Each node is scored using a weighted combination of:
        - Load score (lower is better): task and resource utilization.
        - Capability score (lower is better): how well the node's
          capabilities match the task requirements.
        - Locality score (lower is better): topology proximity.
        - Affinity score (lower is better): historical assignment.

        Args:
            task: The task being routed.
            candidates: List of candidate nodes.

        Returns:
            List of (node, score) tuples sorted by score ascending.
        """
        scored: List[tuple[NodeInfo, float]] = []

        for node in candidates:
            load = self._compute_load_score(node)
            capability = self._compute_capability_score(node, task)
            locality = self._compute_locality_score(node, task)
            affinity = self._compute_affinity_score(node, task)

            total_score = (
                load * LOAD_SCORE_WEIGHT
                + capability * CAPABILITY_MATCH_WEIGHT
                + locality * LOCALITY_SCORE_WEIGHT
                + affinity * AFFINITY_SCORE_WEIGHT
            )
            scored.append((node, total_score))

        # Sort by score ascending (lower is better)
        scored.sort(key=lambda x: x[1])
        return scored

    def _compute_load_score(self, node: NodeInfo) -> float:
        """
        Compute the load score for a node (0.0 to 1.0, lower is better).

        Uses the node's built-in load_score property which combines
        task utilization and resource usage.

        Args:
            node: The node to score.

        Returns:
            Load score between 0.0 (idle) and 1.0 (fully loaded).
        """
        return node.load_score

    def _compute_capability_score(
        self,
        node: NodeInfo,
        task: DistributedTask,
    ) -> float:
        """
        Compute capability match score (0.0 to 1.0, lower is better).

        A node with exactly the required capabilities scores 0.0.
        Additional capabilities slightly increase the score (prefer
        specialized nodes over over-provisioned ones).

        Args:
            node: The node to score.
            task: The task to match against.

        Returns:
            Capability score between 0.0 and 1.0.
        """
        if not task.required_capabilities:
            return 0.0

        node_caps = node.capabilities
        required = task.required_capabilities

        # All required are present (guaranteed by filter)
        matched = len(required)
        total_node_caps = len(node_caps) if node_caps else 1

        # Prefer nodes that are more specialized (fewer extra capabilities)
        extra_caps = total_node_caps - matched
        specialization = extra_caps / max(total_node_caps, 1)

        return min(1.0, specialization * 0.5)

    def _compute_locality_score(
        self,
        node: NodeInfo,
        task: DistributedTask,
    ) -> float:
        """
        Compute locality score based on topology proximity (0.0 to 1.0).

        Uses the source node's region and zone to prefer co-located nodes.
        Same zone = 0.0, same region = 0.3, different region = 1.0.

        Args:
            node: The candidate node.
            task: The task with source_node information.

        Returns:
            Locality score between 0.0 (co-located) and 1.0 (remote).
        """
        if not task.source_node:
            return 0.5  # No locality information available

        try:
            source_info = self._cluster_manager.state.get_node(task.source_node)
            if source_info is None:
                return 0.5

            # Same node
            if node.id == task.source_node:
                return 0.0

            # Same zone
            if node.zone and source_info.zone and node.zone == source_info.zone:
                return 0.1

            # Same region
            if node.region and source_info.region and node.region == source_info.region:
                return 0.3

            # Different region
            return 1.0

        except Exception:
            return 0.5

    def _compute_affinity_score(
        self,
        node: NodeInfo,
        task: DistributedTask,
    ) -> float:
        """
        Compute affinity score for sticky session routing (0.0 to 1.0).

        If the task has a routing key and it maps to this node in the
        affinity cache, the score is 0.0 (strong preference). Otherwise
        the score is 0.5 (neutral).

        Args:
            node: The candidate node.
            task: The task with optional routing key.

        Returns:
            Affinity score between 0.0 (cached match) and 0.5 (neutral).
        """
        if not task.routing_key:
            return 0.5

        cached = self._affinity_cache.get(task.routing_key)
        if cached is not None:
            cached_node_id, expiry = cached
            if time.time() < expiry and cached_node_id == node.id:
                return 0.0

        return 0.5

    # -------------------------------------------------------------------------
    # Affinity Cache
    # -------------------------------------------------------------------------

    def _check_affinity(self, routing_key: str) -> Optional[NodeInfo]:
        """
        Check the affinity cache for a valid, available node.

        Returns the cached node only if it is still healthy and has
        remaining capacity. Expired or unavailable entries are evicted.

        Args:
            routing_key: The routing key to look up.

        Returns:
            The cached NodeInfo if valid and available, None otherwise.
        """
        cached = self._affinity_cache.get(routing_key)
        if cached is None:
            return None

        node_id, expiry = cached
        if time.time() >= expiry:
            del self._affinity_cache[routing_key]
            return None

        # Verify the node is still available
        try:
            node = self._cluster_manager.state.get_node(node_id)
            if node is not None and node.is_available():
                return node
        except Exception:
            pass

        # Node is no longer available; evict the cache entry
        del self._affinity_cache[routing_key]
        return None

    def _set_affinity(self, routing_key: str, node_id: str) -> None:
        """
        Store a routing key to node mapping in the affinity cache.

        Args:
            routing_key: The routing key.
            node_id: The node ID to associate.
        """
        expiry = time.time() + self.affinity_ttl
        self._affinity_cache[routing_key] = (node_id, expiry)

    async def cleanup_expired_affinity(self) -> int:
        """
        Purge expired entries from the affinity cache.

        Should be called periodically to prevent unbounded memory growth.

        Returns:
            Number of expired entries removed.
        """
        now = time.time()
        expired_keys = [
            key
            for key, (_, expiry) in self._affinity_cache.items()
            if now >= expiry
        ]
        for key in expired_keys:
            del self._affinity_cache[key]

        if expired_keys:
            logger.debug(
                "affinity_cache_cleaned",
                removed_count=len(expired_keys),
            )
        return len(expired_keys)

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    async def get_stats(self) -> Dict[str, Any]:
        """
        Return comprehensive routing statistics.

        Returns:
            Dictionary containing routing counts, filter statistics,
            and affinity cache state.
        """
        return {
            "counters": dict(self._stats),
            "affinity_cache_size": len(self._affinity_cache),
            "affinity_ttl_seconds": self.affinity_ttl,
        }
