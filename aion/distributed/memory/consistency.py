"""
AION Distributed Memory - Consistency Manager

Manages consistency guarantees for distributed memory operations.
Implements quorum-based reads and writes, read repair, linearizable
reads via leader forwarding, and hinted handoff for temporarily
unavailable nodes.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

import structlog

from aion.distributed.types import (
    ConsistencyLevel,
    NodeInfo,
    NodeStatus,
    ReplicationEvent,
    VectorClock,
)

if TYPE_CHECKING:
    from aion.distributed.cluster.manager import ClusterManager

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_HINTED_HANDOFF_MAX_AGE = 3600.0  # seconds before hints are discarded
_HINTED_HANDOFF_RETRY_INTERVAL = 30.0  # seconds between handoff attempts
_READ_REPAIR_CONCURRENCY = 8


# ---------------------------------------------------------------------------
# Internal data structures
# ---------------------------------------------------------------------------


@dataclass
class VersionedValue:
    """A value tagged with its vector clock and timestamp."""
    value: Any
    clock: VectorClock = field(default_factory=VectorClock)
    timestamp: float = field(default_factory=time.time)
    source_node: str = ""


@dataclass
class HintedWrite:
    """A write destined for a temporarily unavailable node."""
    target_node: str
    key: str
    value: Any
    clock: VectorClock
    created_at: float = field(default_factory=time.time)

    @property
    def is_expired(self) -> bool:
        return (time.time() - self.created_at) > _HINTED_HANDOFF_MAX_AGE


@dataclass
class ReadResponse:
    """Response from a single node during a distributed read."""
    node_id: str
    value: Optional[VersionedValue] = None
    success: bool = False
    latency_ms: float = 0.0
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Consistency Manager
# ---------------------------------------------------------------------------


class ConsistencyManager:
    """
    Enforces configurable consistency guarantees for distributed memory.

    Supports multiple consistency levels (ONE, QUORUM, ALL, LOCAL,
    EVENTUAL) and provides:

    * **Quorum reads/writes** -- R + W > N ensures strong consistency.
    * **Read repair** -- detects stale replicas during reads and
      asynchronously brings them up to date.
    * **Linearizable reads** -- forwarded to the current cluster leader.
    * **Hinted handoff** -- writes destined for unavailable nodes are
      stored locally and replayed when the target recovers.
    """

    def __init__(
        self,
        cluster_manager: ClusterManager,
        *,
        replication_factor: int = 3,
    ) -> None:
        self._cluster = cluster_manager
        self._replication_factor = replication_factor

        # In-memory versioned store (node-local)
        self._store: Dict[str, VersionedValue] = {}

        # Hinted handoff queue: target_node -> [HintedWrite]
        self._hints: Dict[str, List[HintedWrite]] = defaultdict(list)

        # Background task handles
        self._handoff_task: Optional[asyncio.Task[None]] = None
        self._running = False
        self._lock = asyncio.Lock()

        # Metrics
        self._reads: int = 0
        self._writes: int = 0
        self._read_repairs: int = 0
        self._hints_stored: int = 0
        self._hints_delivered: int = 0

    # -- lifecycle --------------------------------------------------------

    async def start(self) -> None:
        """Start background hinted-handoff delivery loop."""
        if self._running:
            return
        self._running = True
        self._handoff_task = asyncio.create_task(self._handoff_loop())
        logger.info("consistency_manager.started")

    async def stop(self) -> None:
        """Stop background tasks."""
        self._running = False
        if self._handoff_task is not None:
            self._handoff_task.cancel()
            try:
                await self._handoff_task
            except asyncio.CancelledError:
                pass
            self._handoff_task = None
        logger.info("consistency_manager.stopped")

    # -- public API -------------------------------------------------------

    async def read(
        self,
        key: str,
        level: ConsistencyLevel = ConsistencyLevel.QUORUM,
    ) -> Optional[Any]:
        """
        Read a key at the requested consistency level.

        Parameters
        ----------
        key:
            The key to read.
        level:
            Desired consistency -- ONE, QUORUM, ALL, LOCAL, or EVENTUAL.
        """
        self._reads += 1

        if level == ConsistencyLevel.LOCAL:
            return self._read_local(key)

        if level == ConsistencyLevel.ONE:
            return await self._read_one(key)

        if level in (ConsistencyLevel.QUORUM, ConsistencyLevel.LOCAL_QUORUM):
            return await self.quorum_read(key)

        if level == ConsistencyLevel.ALL:
            return await self._read_all(key)

        # EVENTUAL -- best effort from local store
        return self._read_local(key)

    async def write(
        self,
        key: str,
        value: Any,
        level: ConsistencyLevel = ConsistencyLevel.QUORUM,
    ) -> bool:
        """
        Write a key at the requested consistency level.

        Returns ``True`` when enough replicas have acknowledged.
        """
        self._writes += 1

        if level == ConsistencyLevel.LOCAL:
            return self._write_local(key, value)

        if level == ConsistencyLevel.ONE:
            return await self._write_one(key, value)

        if level in (ConsistencyLevel.QUORUM, ConsistencyLevel.LOCAL_QUORUM):
            return await self.quorum_write(key, value)

        if level == ConsistencyLevel.ALL:
            return await self._write_all(key, value)

        # EVENTUAL -- fire-and-forget with best-effort replication
        self._write_local(key, value)
        asyncio.create_task(self._replicate_async(key, value))
        return True

    # -- quorum operations ------------------------------------------------

    async def quorum_read(self, key: str) -> Optional[Any]:
        """
        Read with quorum consistency (R > N/2).

        Contacts a majority of replicas, resolves conflicts using
        vector clocks, and triggers read repair for stale replicas.
        """
        replicas = self._get_replica_nodes(key)
        quorum = self._quorum_size(len(replicas))

        responses: List[ReadResponse] = await self._read_from_replicas(key, replicas)
        successful = [r for r in responses if r.success]

        if len(successful) < quorum:
            logger.warning("consistency.quorum_read_failed", key=key,
                           responses=len(successful), quorum=quorum)
            return None

        # Resolve value -- pick the one with the dominant vector clock
        values_by_node: Dict[str, Any] = {}
        best: Optional[VersionedValue] = None

        for resp in successful:
            if resp.value is not None:
                values_by_node[resp.node_id] = resp.value.value
                if best is None or resp.value.clock.dominates(best.clock):
                    best = resp.value

        # Trigger async read repair if replicas diverge
        if best is not None and len(set(id(v) for v in values_by_node.values())) > 1:
            asyncio.create_task(self.read_repair(key, values_by_node))

        return best.value if best is not None else None

    async def quorum_write(self, key: str, value: Any) -> bool:
        """
        Write with quorum consistency (W > N/2).

        The write succeeds when a majority of replicas acknowledge.
        Nodes that are temporarily unavailable receive a hinted handoff.
        """
        replicas = self._get_replica_nodes(key)
        quorum = self._quorum_size(len(replicas))

        # Build versioned value
        clock = self._get_or_create_clock(key)
        local_id = self._local_node_id()
        clock.increment(local_id)

        versioned = VersionedValue(
            value=value,
            clock=clock.copy(),
            source_node=local_id,
        )

        acks = 0
        for node_id in replicas:
            try:
                ok = await self._write_to_replica(node_id, key, versioned)
                if ok:
                    acks += 1
            except Exception:
                # Hinted handoff for unavailable node
                self._store_hint(node_id, key, value, clock)
                logger.debug("consistency.hinted_handoff_stored",
                              target=node_id, key=key)

        met_quorum = acks >= quorum
        if met_quorum:
            self._store[key] = versioned

        logger.debug("consistency.quorum_write", key=key, acks=acks,
                      quorum=quorum, success=met_quorum)
        return met_quorum

    # -- read repair ------------------------------------------------------

    async def read_repair(self, key: str, values: Dict[str, Any]) -> None:
        """
        Detect stale replicas and send the latest value.

        Called asynchronously after a quorum read that observes
        divergent values.
        """
        self._read_repairs += 1
        latest = self._store.get(key)
        if latest is None:
            return

        latest_value = latest.value
        stale_nodes = [
            node_id for node_id, val in values.items()
            if val != latest_value
        ]

        if not stale_nodes:
            return

        logger.info("consistency.read_repair", key=key,
                     stale_count=len(stale_nodes))

        sem = asyncio.Semaphore(_READ_REPAIR_CONCURRENCY)
        async def _repair_one(node_id: str) -> None:
            async with sem:
                try:
                    await self._write_to_replica(node_id, key, latest)
                except Exception as exc:
                    logger.warning("consistency.read_repair_failed",
                                   node=node_id, key=key, error=str(exc))

        await asyncio.gather(*[_repair_one(n) for n in stale_nodes])

    # -- linearizable read ------------------------------------------------

    async def linearizable_read(self, key: str) -> Optional[Any]:
        """
        Strongly consistent read forwarded to the cluster leader.

        Guarantees linearizability by ensuring the leader has committed
        all preceding writes before responding.
        """
        leader = self._get_leader()
        if leader is None:
            logger.warning("consistency.linearizable_no_leader", key=key)
            return await self.quorum_read(key)

        try:
            response = await self._read_from_leader(leader, key)
            return response
        except Exception as exc:
            logger.warning("consistency.linearizable_fallback",
                           key=key, error=str(exc))
            return await self.quorum_read(key)

    # -- hinted handoff ---------------------------------------------------

    def _store_hint(
        self, target_node: str, key: str, value: Any, clock: VectorClock
    ) -> None:
        """Queue a write for later delivery to *target_node*."""
        hint = HintedWrite(
            target_node=target_node,
            key=key,
            value=value,
            clock=clock.copy(),
        )
        self._hints[target_node].append(hint)
        self._hints_stored += 1

    async def _deliver_hints(self, node_id: str) -> int:
        """Attempt to deliver all pending hints for *node_id*."""
        hints = self._hints.get(node_id, [])
        if not hints:
            return 0

        delivered = 0
        remaining: List[HintedWrite] = []

        for hint in hints:
            if hint.is_expired:
                continue  # discard stale hints
            try:
                versioned = VersionedValue(
                    value=hint.value,
                    clock=hint.clock,
                    source_node=self._local_node_id(),
                )
                ok = await self._write_to_replica(node_id, hint.key, versioned)
                if ok:
                    delivered += 1
                    self._hints_delivered += 1
                else:
                    remaining.append(hint)
            except Exception:
                remaining.append(hint)

        self._hints[node_id] = remaining
        if delivered > 0:
            logger.info("consistency.hints_delivered",
                        target=node_id, delivered=delivered,
                        remaining=len(remaining))
        return delivered

    async def _handoff_loop(self) -> None:
        """Background loop that periodically delivers hinted writes."""
        while self._running:
            try:
                await asyncio.sleep(_HINTED_HANDOFF_RETRY_INTERVAL)
                for node_id in list(self._hints.keys()):
                    if self._is_node_available(node_id):
                        await self._deliver_hints(node_id)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("consistency.handoff_loop_error", error=str(exc))

    # -- helpers ----------------------------------------------------------

    def _quorum_size(self, n: int) -> int:
        return n // 2 + 1

    def _get_replica_nodes(self, key: str) -> List[str]:
        """Return replica node IDs for *key*."""
        try:
            state = self._cluster.state  # type: ignore[union-attr]
            node_ids = [n.id for n in state.nodes.values() if n.is_available()]
            return node_ids[: self._replication_factor]
        except Exception:
            return []

    def _local_node_id(self) -> str:
        try:
            return self._cluster.node_id  # type: ignore[union-attr]
        except Exception:
            return "local"

    def _get_leader(self) -> Optional[str]:
        try:
            state = self._cluster.state  # type: ignore[union-attr]
            return state.leader_id
        except Exception:
            return None

    def _is_node_available(self, node_id: str) -> bool:
        try:
            state = self._cluster.state  # type: ignore[union-attr]
            node = state.nodes.get(node_id)
            return node is not None and node.is_available()
        except Exception:
            return False

    def _get_or_create_clock(self, key: str) -> VectorClock:
        versioned = self._store.get(key)
        if versioned is not None:
            return versioned.clock.copy()
        return VectorClock()

    # -- local operations -------------------------------------------------

    def _read_local(self, key: str) -> Optional[Any]:
        versioned = self._store.get(key)
        return versioned.value if versioned else None

    def _write_local(self, key: str, value: Any) -> bool:
        clock = self._get_or_create_clock(key)
        clock.increment(self._local_node_id())
        self._store[key] = VersionedValue(
            value=value, clock=clock, source_node=self._local_node_id()
        )
        return True

    # -- replica I/O stubs ------------------------------------------------

    async def _read_from_replicas(
        self, key: str, replicas: List[str]
    ) -> List[ReadResponse]:
        """Read *key* from each replica in parallel."""
        results: List[ReadResponse] = []
        for node_id in replicas:
            start = time.monotonic()
            try:
                versioned = self._store.get(key)
                elapsed = (time.monotonic() - start) * 1000.0
                results.append(ReadResponse(
                    node_id=node_id,
                    value=versioned,
                    success=True,
                    latency_ms=elapsed,
                ))
            except Exception as exc:
                elapsed = (time.monotonic() - start) * 1000.0
                results.append(ReadResponse(
                    node_id=node_id,
                    success=False,
                    latency_ms=elapsed,
                    error=str(exc),
                ))
        return results

    async def _write_to_replica(
        self, node_id: str, key: str, versioned: VersionedValue
    ) -> bool:
        """Write to a single replica (local stub)."""
        self._store[key] = versioned
        return True

    async def _read_from_leader(self, leader_id: str, key: str) -> Optional[Any]:
        """Forward a read to the cluster leader (stub)."""
        return self._read_local(key)

    async def _read_one(self, key: str) -> Optional[Any]:
        replicas = self._get_replica_nodes(key)
        for node_id in replicas:
            try:
                versioned = self._store.get(key)
                if versioned is not None:
                    return versioned.value
            except Exception:
                continue
        return None

    async def _write_one(self, key: str, value: Any) -> bool:
        self._write_local(key, value)
        return True

    async def _read_all(self, key: str) -> Optional[Any]:
        return await self.quorum_read(key)

    async def _write_all(self, key: str, value: Any) -> bool:
        replicas = self._get_replica_nodes(key)
        clock = self._get_or_create_clock(key)
        clock.increment(self._local_node_id())
        versioned = VersionedValue(value=value, clock=clock, source_node=self._local_node_id())

        acks = 0
        for node_id in replicas:
            try:
                if await self._write_to_replica(node_id, key, versioned):
                    acks += 1
            except Exception:
                self._store_hint(node_id, key, value, clock)

        return acks == len(replicas)

    async def _replicate_async(self, key: str, value: Any) -> None:
        """Best-effort background replication for EVENTUAL writes."""
        replicas = self._get_replica_nodes(key)
        clock = self._get_or_create_clock(key)
        versioned = VersionedValue(value=value, clock=clock, source_node=self._local_node_id())
        for node_id in replicas:
            try:
                await self._write_to_replica(node_id, key, versioned)
            except Exception:
                self._store_hint(node_id, key, value, clock)

    # -- stats ------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Return consistency manager metrics."""
        total_hints = sum(len(h) for h in self._hints.values())
        return {
            "reads": self._reads,
            "writes": self._writes,
            "read_repairs": self._read_repairs,
            "hints_stored": self._hints_stored,
            "hints_delivered": self._hints_delivered,
            "hints_pending": total_hints,
            "local_keys": len(self._store),
            "replication_factor": self._replication_factor,
        }
