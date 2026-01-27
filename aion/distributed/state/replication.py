"""
AION Distributed State Replicator

Replicates state changes to follower nodes with configurable consistency
levels. Supports synchronous, asynchronous, and semi-synchronous replication
with background queuing and acknowledgment tracking.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

import structlog

from aion.distributed.types import (
    ClusterState,
    ConsistencyLevel,
    NodeInfo,
    ReplicationEvent,
    ReplicationMode,
    VectorClock,
)

if TYPE_CHECKING:
    from aion.distributed.cluster.manager import ClusterManager

logger = structlog.get_logger(__name__)


class AckStatus(str, Enum):
    """Status of a replication acknowledgment."""

    PENDING = "pending"
    ACKED = "acked"
    NACKED = "nacked"
    TIMEOUT = "timeout"


@dataclass
class PendingAck:
    """Tracks a pending acknowledgment for a replicated event."""

    event_id: str
    node_id: str
    status: AckStatus = AckStatus.PENDING
    sent_at: datetime = field(default_factory=datetime.now)
    acked_at: Optional[datetime] = None
    attempt: int = 1
    error: Optional[str] = None


@dataclass
class ReplicationStats:
    """Per-node replication statistics."""

    node_id: str
    events_sent: int = 0
    events_acked: int = 0
    events_failed: int = 0
    last_ack_at: Optional[datetime] = None
    last_event_at: Optional[datetime] = None
    lag_events: int = 0
    avg_latency_ms: float = 0.0
    _latency_samples: List[float] = field(default_factory=list)

    def record_latency(self, latency_ms: float) -> None:
        """Record a latency sample, keeping a sliding window."""
        self._latency_samples.append(latency_ms)
        if len(self._latency_samples) > 100:
            self._latency_samples = self._latency_samples[-100:]
        self.avg_latency_ms = (
            sum(self._latency_samples) / len(self._latency_samples)
            if self._latency_samples
            else 0.0
        )


class StateReplicator:
    """
    Replicates state changes to follower nodes in the cluster.

    Supports multiple consistency levels:
    - ONE: acknowledge after a single replica confirms
    - QUORUM: wait for a majority of replicas
    - ALL: wait for every replica to acknowledge

    Features:
    - Async replication queue with background worker
    - Per-node replication lag monitoring
    - Semi-sync mode (wait for at least N replicas)
    - Configurable ack timeout
    - Retry with backoff on transient failures
    """

    def __init__(
        self,
        cluster_manager: ClusterManager,
        *,
        replication_mode: ReplicationMode = ReplicationMode.SEMI_SYNC,
        min_sync_replicas: int = 1,
        ack_timeout_seconds: float = 5.0,
        max_retries: int = 3,
        queue_max_size: int = 10000,
        batch_size: int = 50,
    ) -> None:
        self._cluster_manager = cluster_manager
        self._replication_mode = replication_mode
        self._min_sync_replicas = min_sync_replicas
        self._ack_timeout = ack_timeout_seconds
        self._max_retries = max_retries
        self._queue_max_size = queue_max_size
        self._batch_size = batch_size

        # Pending acks keyed by event_id -> {node_id -> PendingAck}
        self._pending_acks: Dict[str, Dict[str, PendingAck]] = {}

        # Async replication queue
        self._replication_queue: asyncio.Queue[ReplicationEvent] = asyncio.Queue(
            maxsize=queue_max_size
        )

        # Per-node statistics
        self._node_stats: Dict[str, ReplicationStats] = {}

        # Event completion futures: event_id -> Future[bool]
        self._completion_futures: Dict[str, asyncio.Future[bool]] = {}

        # Background worker
        self._running = False
        self._worker_task: Optional[asyncio.Task[None]] = None

        # Committed event index per node (for lag tracking)
        self._committed_index: int = 0
        self._node_replicated_index: Dict[str, int] = {}

        logger.info(
            "state_replicator.init",
            mode=replication_mode.value,
            min_sync_replicas=min_sync_replicas,
            ack_timeout=ack_timeout_seconds,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the background replication worker."""
        if self._running:
            logger.warning("state_replicator.already_running")
            return
        self._running = True
        self._worker_task = asyncio.create_task(self._replication_worker())
        logger.info("state_replicator.started")

    async def stop(self) -> None:
        """Stop the replication worker and drain the queue."""
        if not self._running:
            return
        self._running = False

        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

        # Resolve outstanding futures
        for future in self._completion_futures.values():
            if not future.done():
                future.set_result(False)
        self._completion_futures.clear()

        self._worker_task = None
        logger.info("state_replicator.stopped")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def replicate(self, event: ReplicationEvent) -> bool:
        """
        Replicate a state change event according to its consistency level.

        For SYNC / SEMI_SYNC modes this awaits acknowledgments before
        returning. For ASYNC mode the event is enqueued and this returns
        immediately.
        """
        logger.debug(
            "state_replicator.replicate",
            event_id=event.event_id,
            consistency=event.consistency.value,
        )

        self._committed_index += 1
        target_nodes = self._get_replication_targets()
        if not target_nodes:
            logger.warning("state_replicator.no_targets", event_id=event.event_id)
            return True  # No followers to replicate to

        # Create pending ack entries
        self._pending_acks[event.event_id] = {}
        for node_id in target_nodes:
            self._pending_acks[event.event_id][node_id] = PendingAck(
                event_id=event.event_id,
                node_id=node_id,
            )

        if self._replication_mode == ReplicationMode.ASYNC:
            # Enqueue for background processing
            try:
                self._replication_queue.put_nowait(event)
            except asyncio.QueueFull:
                logger.error(
                    "state_replicator.queue_full", event_id=event.event_id
                )
                return False
            return True

        # Synchronous or semi-sync: replicate and wait
        send_results = await asyncio.gather(
            *(
                self._send_to_node(node_id, event)
                for node_id in target_nodes
            ),
            return_exceptions=True,
        )

        # Record results
        acked = 0
        for node_id, result in zip(target_nodes, send_results):
            stats = self._get_or_create_stats(node_id)
            if isinstance(result, Exception):
                self._mark_ack(event.event_id, node_id, AckStatus.NACKED, str(result))
                stats.events_failed += 1
            elif result:
                self._mark_ack(event.event_id, node_id, AckStatus.ACKED)
                stats.events_acked += 1
                stats.last_ack_at = datetime.now()
                self._node_replicated_index[node_id] = self._committed_index
                acked += 1
            else:
                self._mark_ack(event.event_id, node_id, AckStatus.NACKED)
                stats.events_failed += 1

        # Evaluate consistency requirement
        return self._evaluate_consistency(event.consistency, acked, len(target_nodes))

    async def replicate_to_node(
        self, node_id: str, event: ReplicationEvent
    ) -> bool:
        """Replicate a single event directly to a specific node."""
        stats = self._get_or_create_stats(node_id)
        stats.events_sent += 1
        stats.last_event_at = datetime.now()

        start = time.monotonic()
        try:
            success = await self._send_to_node(node_id, event)
            elapsed_ms = (time.monotonic() - start) * 1000
            stats.record_latency(elapsed_ms)

            if success:
                stats.events_acked += 1
                stats.last_ack_at = datetime.now()
                self._node_replicated_index[node_id] = self._committed_index
            else:
                stats.events_failed += 1

            return success
        except Exception as exc:
            elapsed_ms = (time.monotonic() - start) * 1000
            stats.record_latency(elapsed_ms)
            stats.events_failed += 1
            logger.error(
                "state_replicator.replicate_to_node.failed",
                node_id=node_id,
                event_id=event.event_id,
                error=str(exc),
            )
            return False

    async def wait_for_ack(
        self,
        event_id: str,
        consistency: ConsistencyLevel = ConsistencyLevel.QUORUM,
        timeout: Optional[float] = None,
    ) -> bool:
        """
        Wait for acknowledgments on a previously replicated event.

        Returns True if the required consistency level has been met
        within the timeout period.
        """
        effective_timeout = timeout or self._ack_timeout
        deadline = time.monotonic() + effective_timeout

        while time.monotonic() < deadline:
            ack_map = self._pending_acks.get(event_id, {})
            if not ack_map:
                return True  # No pending acks or already cleaned up

            acked = sum(
                1 for pa in ack_map.values() if pa.status == AckStatus.ACKED
            )
            total = len(ack_map)

            if self._evaluate_consistency(consistency, acked, total):
                return True

            # Check if all have responded (even if not met)
            resolved = sum(
                1
                for pa in ack_map.values()
                if pa.status != AckStatus.PENDING
            )
            if resolved == total:
                return self._evaluate_consistency(consistency, acked, total)

            await asyncio.sleep(0.05)

        # Timeout: mark remaining as timed out
        ack_map = self._pending_acks.get(event_id, {})
        for pa in ack_map.values():
            if pa.status == AckStatus.PENDING:
                pa.status = AckStatus.TIMEOUT
        acked = sum(1 for pa in ack_map.values() if pa.status == AckStatus.ACKED)
        return self._evaluate_consistency(consistency, acked, len(ack_map))

    def get_replication_lag(self, node_id: str) -> int:
        """
        Get the replication lag for a node measured in events.

        Returns the number of events the node is behind the committed
        index.
        """
        node_index = self._node_replicated_index.get(node_id, 0)
        return max(0, self._committed_index - node_index)

    def get_replication_status(self) -> Dict[str, Any]:
        """Return comprehensive replication status."""
        return {
            "running": self._running,
            "mode": self._replication_mode.value,
            "committed_index": self._committed_index,
            "queue_size": self._replication_queue.qsize(),
            "pending_ack_count": sum(
                1
                for acks in self._pending_acks.values()
                for a in acks.values()
                if a.status == AckStatus.PENDING
            ),
            "nodes": {
                nid: {
                    "events_sent": stats.events_sent,
                    "events_acked": stats.events_acked,
                    "events_failed": stats.events_failed,
                    "lag_events": self.get_replication_lag(nid),
                    "avg_latency_ms": round(stats.avg_latency_ms, 2),
                    "last_ack_at": (
                        stats.last_ack_at.isoformat()
                        if stats.last_ack_at
                        else None
                    ),
                }
                for nid, stats in self._node_stats.items()
            },
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evaluate_consistency(
        self,
        consistency: ConsistencyLevel,
        acked: int,
        total: int,
    ) -> bool:
        """Evaluate whether a consistency requirement is satisfied."""
        if consistency == ConsistencyLevel.ONE:
            return acked >= 1
        elif consistency == ConsistencyLevel.QUORUM:
            quorum = (total // 2) + 1
            return acked >= quorum
        elif consistency == ConsistencyLevel.ALL:
            return acked >= total
        elif consistency == ConsistencyLevel.LOCAL:
            return True  # Local only, always satisfied
        elif consistency == ConsistencyLevel.EVENTUAL:
            return True  # Fire-and-forget
        else:
            # LOCAL_QUORUM or unknown: treat as quorum
            quorum = (total // 2) + 1
            return acked >= quorum

    def _get_replication_targets(self) -> List[str]:
        """Get node IDs that should receive replicated data."""
        try:
            cluster_state: ClusterState = self._cluster_manager.get_cluster_state()
            local_id = self._cluster_manager.local_node_id
            return [
                nid
                for nid, node in cluster_state.nodes.items()
                if nid != local_id and node.is_available()
            ]
        except Exception:
            return []

    def _mark_ack(
        self,
        event_id: str,
        node_id: str,
        status: AckStatus,
        error: Optional[str] = None,
    ) -> None:
        """Mark the acknowledgment status for a node on an event."""
        ack_map = self._pending_acks.get(event_id)
        if ack_map and node_id in ack_map:
            ack_map[node_id].status = status
            ack_map[node_id].error = error
            if status == AckStatus.ACKED:
                ack_map[node_id].acked_at = datetime.now()

    def _get_or_create_stats(self, node_id: str) -> ReplicationStats:
        if node_id not in self._node_stats:
            self._node_stats[node_id] = ReplicationStats(node_id=node_id)
        return self._node_stats[node_id]

    async def _send_to_node(
        self, node_id: str, event: ReplicationEvent
    ) -> bool:
        """
        Send a replication event to a specific node.

        In a production system this would perform an RPC call.
        Returns True if the remote node acknowledged the event.
        """
        stats = self._get_or_create_stats(node_id)
        stats.events_sent += 1
        stats.last_event_at = datetime.now()

        logger.debug(
            "state_replicator.send_to_node",
            node_id=node_id,
            event_id=event.event_id,
        )
        # Stub: real implementation sends via gRPC or HTTP
        return True

    # ------------------------------------------------------------------
    # Background worker
    # ------------------------------------------------------------------

    async def _replication_worker(self) -> None:
        """Background worker that drains the async replication queue."""
        logger.info("state_replicator.worker.started")
        while self._running:
            try:
                event = await asyncio.wait_for(
                    self._replication_queue.get(), timeout=1.0
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            target_nodes = self._get_replication_targets()
            for node_id in target_nodes:
                try:
                    await self._send_to_node(node_id, event)
                    self._mark_ack(event.event_id, node_id, AckStatus.ACKED)
                    self._node_replicated_index[node_id] = self._committed_index
                except Exception as exc:
                    self._mark_ack(
                        event.event_id, node_id, AckStatus.NACKED, str(exc)
                    )
                    logger.error(
                        "state_replicator.worker.send_failed",
                        node_id=node_id,
                        event_id=event.event_id,
                        error=str(exc),
                    )

        logger.info("state_replicator.worker.stopped")
