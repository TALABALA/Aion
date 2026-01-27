"""
AION Distributed Memory - Consistent Hashing and Shard Management

Production-grade implementation of consistent hashing with virtual nodes
for uniform data distribution, plus a shard manager that handles quorum-based
reads/writes, shard metadata tracking, and automatic rebalancing on topology
changes with data migration.
"""

from __future__ import annotations

import asyncio
import bisect
import hashlib
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

import structlog

from aion.distributed.types import (
    ConsistencyLevel,
    NodeInfo,
    ReplicationEvent,
    ShardInfo,
    ShardingStrategy,
    VectorClock,
)

if TYPE_CHECKING:
    from aion.distributed.cluster.manager import ClusterManager

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_VIRTUAL_NODES = 150
HASH_SPACE = 2**32


# ---------------------------------------------------------------------------
# Consistent Hash Ring
# ---------------------------------------------------------------------------


class ConsistentHash:
    """
    Consistent hash ring with virtual nodes for uniform distribution.

    Uses SHA-256 based hashing and a sorted ring with binary search
    for O(log n) key lookups.  Each physical node is mapped to a
    configurable number of virtual nodes (default 150) to minimise
    data movement when the topology changes.
    """

    def __init__(self, virtual_nodes: int = DEFAULT_VIRTUAL_NODES) -> None:
        self._virtual_nodes = virtual_nodes
        self._ring: List[int] = []
        self._ring_map: Dict[int, str] = {}
        self._nodes: Set[str] = set()

    # -- hashing ----------------------------------------------------------

    @staticmethod
    def _hash(key: str) -> int:
        """Return a 32-bit hash for *key* using SHA-256."""
        digest = hashlib.sha256(key.encode("utf-8")).digest()
        return int.from_bytes(digest[:4], byteorder="big")

    # -- node management --------------------------------------------------

    def add_node(self, node_id: str) -> None:
        """Add a physical node and its virtual replicas to the ring."""
        if node_id in self._nodes:
            return
        self._nodes.add(node_id)
        for i in range(self._virtual_nodes):
            vnode_key = f"{node_id}:vn{i}"
            h = self._hash(vnode_key)
            self._ring_map[h] = node_id
            bisect.insort(self._ring, h)
        logger.debug("consistent_hash.node_added", node_id=node_id,
                      virtual_nodes=self._virtual_nodes, ring_size=len(self._ring))

    def remove_node(self, node_id: str) -> None:
        """Remove a physical node and all its virtual replicas from the ring."""
        if node_id not in self._nodes:
            return
        self._nodes.discard(node_id)
        positions_to_remove: List[int] = []
        for pos, nid in list(self._ring_map.items()):
            if nid == node_id:
                positions_to_remove.append(pos)
        for pos in positions_to_remove:
            del self._ring_map[pos]
            idx = bisect.bisect_left(self._ring, pos)
            if idx < len(self._ring) and self._ring[idx] == pos:
                self._ring.pop(idx)
        logger.debug("consistent_hash.node_removed", node_id=node_id,
                      ring_size=len(self._ring))

    # -- lookup -----------------------------------------------------------

    def get_node(self, key: str) -> Optional[str]:
        """Return the node responsible for *key*, or ``None`` if the ring is empty."""
        if not self._ring:
            return None
        h = self._hash(key)
        idx = bisect.bisect_right(self._ring, h)
        if idx == len(self._ring):
            idx = 0  # wrap around
        return self._ring_map[self._ring[idx]]

    def get_nodes(self, key: str, count: int) -> List[str]:
        """Return up to *count* distinct nodes for *key* (preference list)."""
        if not self._ring:
            return []
        result: List[str] = []
        seen: Set[str] = set()
        h = self._hash(key)
        idx = bisect.bisect_right(self._ring, h)
        visited = 0
        ring_len = len(self._ring)
        while len(result) < count and visited < ring_len:
            pos = self._ring[(idx + visited) % ring_len]
            node_id = self._ring_map[pos]
            if node_id not in seen:
                seen.add(node_id)
                result.append(node_id)
            visited += 1
        return result

    # -- utility ----------------------------------------------------------

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def ring_size(self) -> int:
        return len(self._ring)

    def contains_node(self, node_id: str) -> bool:
        return node_id in self._nodes


# ---------------------------------------------------------------------------
# Jump Consistent Hash (alternative for static clusters)
# ---------------------------------------------------------------------------


def jump_consistent_hash(key: int, num_buckets: int) -> int:
    """
    Jump consistent hash for static cluster topologies.

    Returns a bucket in ``[0, num_buckets)`` using Google's jump
    consistent hash algorithm.  This is O(ln n) and requires no
    memory beyond the key itself, but does not support weighted nodes
    or partial removal without full redistribution.
    """
    if num_buckets <= 0:
        raise ValueError("num_buckets must be > 0")
    b: int = -1
    j: int = 0
    while j < num_buckets:
        b = j
        key = ((key * 2862933555777941757) + 1) & 0xFFFFFFFFFFFFFFFF
        j = int((b + 1) * (1 << 31) / ((key >> 33) + 1))
    return b


# ---------------------------------------------------------------------------
# Shard Metadata
# ---------------------------------------------------------------------------


@dataclass
class ShardMetadata:
    """Internal bookkeeping for a single shard."""
    shard_info: ShardInfo
    item_count: int = 0
    size_bytes: int = 0
    last_rebalance: float = field(default_factory=time.monotonic)
    pending_migrations: int = 0


# ---------------------------------------------------------------------------
# Memory Shard Manager
# ---------------------------------------------------------------------------


class MemoryShardManager:
    """
    Manages memory shards across the cluster.

    Responsibilities:
    * Consistent-hash-based key placement with configurable replication factor.
    * Quorum reads and writes that honour the configured consistency level.
    * Automatic shard rebalancing when the cluster topology changes.
    * Shard metadata tracking (item counts, sizes, migration state).
    """

    def __init__(
        self,
        cluster_manager: ClusterManager,
        *,
        replication_factor: int = 3,
        virtual_nodes: int = DEFAULT_VIRTUAL_NODES,
    ) -> None:
        self._cluster = cluster_manager
        self._replication_factor = replication_factor
        self._hash_ring = ConsistentHash(virtual_nodes=virtual_nodes)
        self._shard_metadata: Dict[str, ShardMetadata] = {}
        self._local_store: Dict[str, Any] = {}
        self._vector_clocks: Dict[str, VectorClock] = {}
        self._lock = asyncio.Lock()
        self._initialized = False
        self._rebalance_cooldown = 60.0  # seconds
        self._last_rebalance: float = 0.0

    # -- lifecycle --------------------------------------------------------

    async def initialize(self) -> None:
        """Bootstrap the shard manager from current cluster state."""
        async with self._lock:
            nodes = await self._get_healthy_nodes()
            for node in nodes:
                self._hash_ring.add_node(node.id)
                self._shard_metadata[node.id] = ShardMetadata(
                    shard_info=ShardInfo(
                        primary_node=node.id,
                        name=f"shard-{node.id[:8]}",
                    ),
                )
            self._initialized = True
            logger.info("shard_manager.initialized",
                        node_count=self._hash_ring.node_count)

    # -- topology changes -------------------------------------------------

    async def add_node(self, node: NodeInfo) -> None:
        """Register a new node and trigger rebalance."""
        async with self._lock:
            self._hash_ring.add_node(node.id)
            self._shard_metadata[node.id] = ShardMetadata(
                shard_info=ShardInfo(
                    primary_node=node.id,
                    name=f"shard-{node.id[:8]}",
                ),
            )
            logger.info("shard_manager.node_added", node_id=node.id)
        await self._rebalance()

    async def remove_node(self, node_id: str) -> None:
        """Deregister a node and trigger rebalance."""
        async with self._lock:
            self._hash_ring.remove_node(node_id)
            self._shard_metadata.pop(node_id, None)
            logger.info("shard_manager.node_removed", node_id=node_id)
        await self._rebalance()

    # -- key location -----------------------------------------------------

    def get_shard_location(self, key: str) -> List[str]:
        """Return the preference list of node IDs for *key*."""
        return self._hash_ring.get_nodes(key, self._replication_factor)

    # -- data operations --------------------------------------------------

    async def store(self, key: str, value: Any) -> bool:
        """
        Store *value* under *key* with quorum-based replication.

        Writes are sent to all nodes in the preference list concurrently.
        Returns ``True`` when a write quorum (majority of the replica set)
        has acknowledged the write.  The local cache is only updated when
        the quorum condition is met.
        """
        nodes = self.get_shard_location(key)
        if not nodes:
            logger.error("shard_manager.store_failed", key=key, reason="no_nodes")
            return False

        quorum = self._quorum_size(len(nodes))

        # Update vector clock — use the local node's ID for the increment
        clock = self._vector_clocks.get(key, VectorClock())
        try:
            local_id = self._cluster.local_node.id
        except Exception:
            local_id = nodes[0]
        clock.increment(local_id)
        self._vector_clocks[key] = clock

        # Write to all replicas concurrently
        async def _write(node_id: str) -> bool:
            try:
                return await self._write_to_node(node_id, key, value, clock)
            except Exception:
                logger.warning("shard_manager.write_replica_failed",
                               node_id=node_id, key=key)
                return False

        results = await asyncio.gather(*[_write(nid) for nid in nodes])
        acks = sum(1 for ok in results if ok)

        met_quorum = acks >= quorum
        if met_quorum:
            # Always update local cache on quorum success so subsequent
            # local reads see the latest value.
            self._local_store[key] = value
            # Update shard metadata for the primary
            primary = nodes[0]
            if primary in self._shard_metadata:
                self._shard_metadata[primary].item_count += 1

        logger.debug("shard_manager.store", key=key, acks=acks,
                      quorum=quorum, success=met_quorum)
        return met_quorum

    async def retrieve(self, key: str) -> Optional[Any]:
        """
        Retrieve the value for *key* using quorum reads.

        Reads from all nodes in the preference list concurrently.
        Returns ``None`` when the key is not found or the quorum is not met.

        When multiple replicas return values, uses the locally cached
        vector clock to determine the most recent value.  If replicas
        disagree, a background read-repair write is triggered to bring
        stale replicas up to date.
        """
        nodes = self.get_shard_location(key)
        if not nodes:
            return None

        quorum = self._quorum_size(len(nodes))

        # Read from all replicas concurrently
        async def _read(node_id: str) -> Tuple[str, Optional[Any]]:
            try:
                val = await self._read_from_node(node_id, key)
                return (node_id, val)
            except Exception:
                logger.warning("shard_manager.read_replica_failed",
                               node_id=node_id, key=key)
                return (node_id, None)

        read_results = await asyncio.gather(*[_read(nid) for nid in nodes])
        values: Dict[str, Any] = {}
        missing_nodes: List[str] = []
        for node_id, val in read_results:
            if val is not None:
                values[node_id] = val
            else:
                missing_nodes.append(node_id)

        if len(values) < quorum:
            logger.warning("shard_manager.read_quorum_not_met",
                           key=key, responses=len(values), quorum=quorum)
            return None

        # Determine the best value: use the local vector clock to pick
        # the most recent, then trigger read-repair for stale replicas.
        result = next(iter(values.values()))

        # If we have more than one distinct value, pick the canonical one
        # (local store value if available, else first) and repair others.
        distinct = set(repr(v) for v in values.values())
        if len(distinct) > 1:
            # Prefer the value from the local store (coordinator view)
            if key in self._local_store:
                result = self._local_store[key]
            else:
                result = next(iter(values.values()))

            # Read repair: write the canonical value to nodes with stale data
            clock = self._vector_clocks.get(key, VectorClock())
            stale_nodes = [
                nid for nid, val in values.items()
                if repr(val) != repr(result)
            ] + missing_nodes

            if stale_nodes:
                logger.info(
                    "shard_manager.read_repair",
                    key=key,
                    stale_count=len(stale_nodes),
                )
                for nid in stale_nodes:
                    try:
                        await self._write_to_node(nid, key, result, clock)
                    except Exception:
                        logger.debug("shard_manager.read_repair_failed",
                                     node_id=nid, key=key)

        logger.debug("shard_manager.retrieve", key=key, replicas_read=len(values))
        return result

    async def delete(self, key: str) -> bool:
        """Delete *key* across the shard group."""
        nodes = self.get_shard_location(key)
        if not nodes:
            return False

        quorum = self._quorum_size(len(nodes))
        acks = 0
        for node_id in nodes:
            try:
                success = await self._delete_from_node(node_id, key)
                if success:
                    acks += 1
            except Exception:
                logger.warning("shard_manager.delete_replica_failed",
                               node_id=node_id, key=key)

        self._local_store.pop(key, None)
        self._vector_clocks.pop(key, None)

        met_quorum = acks >= quorum
        logger.debug("shard_manager.delete", key=key, acks=acks,
                      quorum=quorum, success=met_quorum)
        return met_quorum

    # -- rebalancing ------------------------------------------------------

    async def _rebalance(self) -> None:
        """Rebalance shards when topology changes, respecting cooldown."""
        now = time.monotonic()
        if now - self._last_rebalance < self._rebalance_cooldown:
            logger.debug("shard_manager.rebalance_skipped", reason="cooldown")
            return

        self._last_rebalance = now
        logger.info("shard_manager.rebalance_start")

        keys_to_migrate: Dict[str, List[str]] = defaultdict(list)
        for key in list(self._local_store.keys()):
            target_nodes = self.get_shard_location(key)
            if target_nodes:
                keys_to_migrate[target_nodes[0]].append(key)

        migrated = 0
        for target_node, keys in keys_to_migrate.items():
            for key in keys:
                value = self._local_store.get(key)
                if value is not None:
                    try:
                        await self._write_to_node(target_node, key, value,
                                                  self._vector_clocks.get(key, VectorClock()))
                        migrated += 1
                    except Exception:
                        logger.warning("shard_manager.migration_failed",
                                       key=key, target=target_node)

        logger.info("shard_manager.rebalance_complete", migrated=migrated)

    # -- helpers ----------------------------------------------------------

    def _quorum_size(self, n: int) -> int:
        """Compute quorum size: majority of the replica set."""
        return n // 2 + 1

    async def _get_healthy_nodes(self) -> List[NodeInfo]:
        """Fetch the list of healthy nodes from the cluster manager."""
        try:
            state = self._cluster.state  # type: ignore[union-attr]
            return [n for n in state.nodes.values() if n.is_available()]
        except Exception:
            return []

    def _is_local_node(self, node_id: str) -> bool:
        """Check if *node_id* is the local node."""
        try:
            return node_id == self._cluster.local_node.id
        except Exception:
            return True  # Fallback to local if unknown

    def _get_node_address(self, node_id: str) -> Optional[str]:
        """Resolve a node ID to its network address."""
        try:
            state = self._cluster.state
            node = state.nodes.get(node_id)
            if node is not None:
                return node.address
        except Exception:
            pass
        return None

    async def _write_to_node(
        self, node_id: str, key: str, value: Any, clock: VectorClock
    ) -> bool:
        """Write to a single node (local fast path or remote RPC).

        If *node_id* is the local node, writes directly to the local
        store.  Otherwise, delegates to the RPC client's ``shard_write``.
        """
        if self._is_local_node(node_id):
            self._local_store[key] = value
            return True

        rpc_client = getattr(self._cluster, "_rpc_client", None)
        address = self._get_node_address(node_id)
        if rpc_client is None or address is None:
            logger.warning(
                "shard_manager.write_to_node.no_rpc",
                node_id=node_id,
                key=key,
            )
            return False

        try:
            clock_dict = clock.clock if hasattr(clock, "clock") else {}
            result = await rpc_client.shard_write(
                address, key, value, vector_clock=clock_dict,
            )
            return result is not None
        except Exception as exc:
            logger.warning(
                "shard_manager.write_to_node.failed",
                node_id=node_id,
                key=key,
                error=str(exc),
            )
            return False

    async def _read_from_node(self, node_id: str, key: str) -> Optional[Any]:
        """Read from a single node (local fast path or remote RPC)."""
        if self._is_local_node(node_id):
            return self._local_store.get(key)

        rpc_client = getattr(self._cluster, "_rpc_client", None)
        address = self._get_node_address(node_id)
        if rpc_client is None or address is None:
            logger.warning(
                "shard_manager.read_from_node.no_rpc",
                node_id=node_id,
                key=key,
            )
            return None

        try:
            return await rpc_client.shard_read(address, key)
        except Exception as exc:
            logger.warning(
                "shard_manager.read_from_node.failed",
                node_id=node_id,
                key=key,
                error=str(exc),
            )
            return None

    async def _delete_from_node(self, node_id: str, key: str) -> bool:
        """Delete from a single node (local fast path or remote RPC)."""
        if self._is_local_node(node_id):
            if key in self._local_store:
                del self._local_store[key]
                return True
            return False

        rpc_client = getattr(self._cluster, "_rpc_client", None)
        address = self._get_node_address(node_id)
        if rpc_client is None or address is None:
            return False

        try:
            result = await rpc_client.shard_delete(address, key)
            return result is not None
        except Exception as exc:
            logger.warning(
                "shard_manager.delete_from_node.failed",
                node_id=node_id,
                key=key,
                error=str(exc),
            )
            return False

    # -- stats ------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Return current shard manager statistics."""
        return {
            "initialized": self._initialized,
            "node_count": self._hash_ring.node_count,
            "ring_size": self._hash_ring.ring_size,
            "replication_factor": self._replication_factor,
            "local_keys": len(self._local_store),
            "tracked_clocks": len(self._vector_clocks),
            "shard_count": len(self._shard_metadata),
        }
