"""
AION Distributed State Synchronizer

Synchronizes state across cluster nodes using anti-entropy protocols.
Implements Merkle tree digest comparison for efficient incremental sync,
with push/pull strategies selected based on data volume.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

import structlog

from aion.distributed.types import (
    ClusterState,
    NodeInfo,
    ReplicationEvent,
    VectorClock,
)

if TYPE_CHECKING:
    from aion.distributed.cluster.manager import ClusterManager

logger = structlog.get_logger(__name__)


class SyncDirection(str, Enum):
    """Direction of state synchronization."""

    PUSH = "push"
    PULL = "pull"
    PUSH_PULL = "push_pull"


class SyncStatus(str, Enum):
    """Status of a sync operation for a particular node."""

    IDLE = "idle"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class MerkleNode:
    """Node in a simplified Merkle tree for state digest comparison."""

    hash_value: str = ""
    children: Dict[str, MerkleNode] = field(default_factory=dict)
    key: str = ""
    level: int = 0


@dataclass
class SyncProgress:
    """Tracks synchronization progress for a single peer node."""

    node_id: str = ""
    status: SyncStatus = SyncStatus.IDLE
    last_sync_at: Optional[datetime] = None
    last_success_at: Optional[datetime] = None
    keys_synced: int = 0
    keys_divergent: int = 0
    bytes_transferred: int = 0
    consecutive_failures: int = 0
    error: Optional[str] = None
    direction: SyncDirection = SyncDirection.PUSH_PULL
    duration_ms: float = 0.0


class StateSynchronizer:
    """
    Synchronizes distributed state across cluster nodes.

    Uses an anti-entropy protocol built on Merkle tree digest comparison
    to detect divergence efficiently. Supports push, pull, and push-pull
    synchronization modes with automatic direction selection based on
    estimated data volume delta.

    Features:
    - Background sync loop with configurable interval
    - Per-node sync progress tracking
    - Merkle tree based efficient diff detection
    - Adaptive push/pull strategy
    - Full and incremental sync modes
    """

    def __init__(
        self,
        cluster_manager: ClusterManager,
        *,
        sync_interval_seconds: float = 5.0,
        full_sync_interval_seconds: float = 300.0,
        max_keys_per_batch: int = 1000,
        push_threshold_keys: int = 100,
    ) -> None:
        self._cluster_manager = cluster_manager
        self._sync_interval = sync_interval_seconds
        self._full_sync_interval = full_sync_interval_seconds
        self._max_keys_per_batch = max_keys_per_batch
        self._push_threshold = push_threshold_keys

        # Internal state store (key -> value with metadata)
        self._state: Dict[str, Any] = {}
        self._state_versions: Dict[str, VectorClock] = {}
        self._state_timestamps: Dict[str, float] = {}

        # Sync tracking
        self._sync_progress: Dict[str, SyncProgress] = {}
        self._running = False
        self._sync_task: Optional[asyncio.Task[None]] = None
        self._full_sync_task: Optional[asyncio.Task[None]] = None
        self._last_full_sync: Optional[datetime] = None

        # Merkle tree root for quick digest comparison
        self._merkle_root_hash: str = ""
        self._merkle_dirty = True

        logger.info(
            "state_synchronizer.init",
            sync_interval=sync_interval_seconds,
            full_sync_interval=full_sync_interval_seconds,
        )

    async def start(self) -> None:
        """Start the background synchronization loops."""
        if self._running:
            logger.warning("state_synchronizer.already_running")
            return

        self._running = True
        self._sync_task = asyncio.create_task(self._sync_loop())
        self._full_sync_task = asyncio.create_task(self._full_sync_loop())
        logger.info("state_synchronizer.started")

    async def stop(self) -> None:
        """Stop all synchronization activity gracefully."""
        if not self._running:
            return

        self._running = False

        for task in (self._sync_task, self._full_sync_task):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        self._sync_task = None
        self._full_sync_task = None
        logger.info("state_synchronizer.stopped")

    # ------------------------------------------------------------------
    # Public sync methods
    # ------------------------------------------------------------------

    async def sync_to_node(self, node_id: str) -> bool:
        """Push local state to a specific remote node."""
        progress = self._get_or_create_progress(node_id)
        progress.status = SyncStatus.IN_PROGRESS
        progress.direction = SyncDirection.PUSH
        start = time.monotonic()

        try:
            local_digest = self._compute_merkle_root()
            remote_digest = await self._fetch_remote_digest(node_id)

            if local_digest == remote_digest:
                logger.debug("state_synchronizer.sync_to_node.no_diff", node_id=node_id)
                progress.status = SyncStatus.COMPLETED
                progress.keys_divergent = 0
                return True

            divergent_keys = await self._find_divergent_keys(node_id)
            pushed = 0
            for batch_start in range(0, len(divergent_keys), self._max_keys_per_batch):
                batch = divergent_keys[batch_start : batch_start + self._max_keys_per_batch]
                payload = {k: self._state[k] for k in batch if k in self._state}
                await self._push_state_batch(node_id, payload)
                pushed += len(payload)

            progress.keys_synced = pushed
            progress.keys_divergent = len(divergent_keys)
            progress.status = SyncStatus.COMPLETED
            progress.last_success_at = datetime.now()
            progress.consecutive_failures = 0
            logger.info(
                "state_synchronizer.sync_to_node.complete",
                node_id=node_id,
                keys_pushed=pushed,
            )
            return True

        except Exception as exc:
            progress.status = SyncStatus.FAILED
            progress.consecutive_failures += 1
            progress.error = str(exc)
            logger.error(
                "state_synchronizer.sync_to_node.failed",
                node_id=node_id,
                error=str(exc),
            )
            return False
        finally:
            elapsed = (time.monotonic() - start) * 1000
            progress.duration_ms = elapsed
            progress.last_sync_at = datetime.now()

    async def sync_from_node(self, node_id: str) -> bool:
        """Pull state from a specific remote node."""
        progress = self._get_or_create_progress(node_id)
        progress.status = SyncStatus.IN_PROGRESS
        progress.direction = SyncDirection.PULL
        start = time.monotonic()

        try:
            remote_state = await self._fetch_remote_state(node_id)
            merged = 0
            for key, value in remote_state.items():
                if key not in self._state or self._is_remote_newer(key, node_id):
                    self._state[key] = value
                    self._state_timestamps[key] = time.time()
                    merged += 1

            self._merkle_dirty = True
            progress.keys_synced = merged
            progress.status = SyncStatus.COMPLETED
            progress.last_success_at = datetime.now()
            progress.consecutive_failures = 0
            logger.info(
                "state_synchronizer.sync_from_node.complete",
                node_id=node_id,
                keys_merged=merged,
            )
            return True

        except Exception as exc:
            progress.status = SyncStatus.FAILED
            progress.consecutive_failures += 1
            progress.error = str(exc)
            logger.error(
                "state_synchronizer.sync_from_node.failed",
                node_id=node_id,
                error=str(exc),
            )
            return False
        finally:
            elapsed = (time.monotonic() - start) * 1000
            progress.duration_ms = elapsed
            progress.last_sync_at = datetime.now()

    async def full_sync(self) -> Dict[str, bool]:
        """Perform a full synchronization with all known peer nodes."""
        results: Dict[str, bool] = {}
        peer_ids = await self._get_peer_node_ids()

        logger.info("state_synchronizer.full_sync.start", peer_count=len(peer_ids))

        for node_id in peer_ids:
            push_ok = await self.sync_to_node(node_id)
            pull_ok = await self.sync_from_node(node_id)
            results[node_id] = push_ok and pull_ok

        self._last_full_sync = datetime.now()
        succeeded = sum(1 for v in results.values() if v)
        logger.info(
            "state_synchronizer.full_sync.complete",
            total=len(results),
            succeeded=succeeded,
        )
        return results

    async def incremental_sync(self) -> Dict[str, bool]:
        """
        Perform incremental sync using Merkle tree digest comparison.

        Only exchanges data with nodes whose digest differs from ours.
        Chooses push or pull direction based on estimated delta size.
        """
        results: Dict[str, bool] = {}
        peer_ids = await self._get_peer_node_ids()
        local_digest = self._compute_merkle_root()

        for node_id in peer_ids:
            try:
                remote_digest = await self._fetch_remote_digest(node_id)
                if remote_digest == local_digest:
                    results[node_id] = True
                    continue

                # Decide direction based on estimated divergence
                divergent_keys = await self._find_divergent_keys(node_id)
                if len(divergent_keys) <= self._push_threshold:
                    ok = await self.sync_to_node(node_id)
                else:
                    ok = await self.sync_to_node(node_id)
                    pull_ok = await self.sync_from_node(node_id)
                    ok = ok and pull_ok

                results[node_id] = ok
            except Exception as exc:
                logger.error(
                    "state_synchronizer.incremental_sync.node_error",
                    node_id=node_id,
                    error=str(exc),
                )
                results[node_id] = False

        return results

    def get_sync_status(self) -> Dict[str, Any]:
        """Return current synchronization status for all tracked nodes."""
        return {
            "running": self._running,
            "state_key_count": len(self._state),
            "merkle_root": self._merkle_root_hash,
            "last_full_sync": (
                self._last_full_sync.isoformat() if self._last_full_sync else None
            ),
            "nodes": {
                nid: {
                    "status": progress.status.value,
                    "last_sync_at": (
                        progress.last_sync_at.isoformat()
                        if progress.last_sync_at
                        else None
                    ),
                    "last_success_at": (
                        progress.last_success_at.isoformat()
                        if progress.last_success_at
                        else None
                    ),
                    "keys_synced": progress.keys_synced,
                    "keys_divergent": progress.keys_divergent,
                    "consecutive_failures": progress.consecutive_failures,
                    "direction": progress.direction.value,
                    "duration_ms": round(progress.duration_ms, 2),
                    "error": progress.error,
                }
                for nid, progress in self._sync_progress.items()
            },
        }

    # ------------------------------------------------------------------
    # Merkle tree helpers
    # ------------------------------------------------------------------

    def _compute_merkle_root(self) -> str:
        """
        Compute a simplified Merkle tree root hash over the local state.

        Keys are sorted and hashed pairwise up to a single root digest.
        The result is cached until state mutations mark the tree dirty.
        """
        if not self._merkle_dirty and self._merkle_root_hash:
            return self._merkle_root_hash

        if not self._state:
            self._merkle_root_hash = hashlib.sha256(b"empty").hexdigest()
            self._merkle_dirty = False
            return self._merkle_root_hash

        # Leaf hashes: sorted key-value pairs
        sorted_keys = sorted(self._state.keys())
        leaf_hashes: List[str] = []
        for key in sorted_keys:
            value_repr = repr(self._state[key]).encode("utf-8")
            h = hashlib.sha256(key.encode("utf-8") + b":" + value_repr).hexdigest()
            leaf_hashes.append(h)

        # Build tree bottom-up
        level = leaf_hashes
        while len(level) > 1:
            next_level: List[str] = []
            for i in range(0, len(level), 2):
                left = level[i]
                right = level[i + 1] if i + 1 < len(level) else left
                combined = hashlib.sha256((left + right).encode("utf-8")).hexdigest()
                next_level.append(combined)
            level = next_level

        self._merkle_root_hash = level[0]
        self._merkle_dirty = False
        return self._merkle_root_hash

    def _compute_key_hash(self, key: str) -> str:
        """Compute the hash for a single key-value pair."""
        if key not in self._state:
            return ""
        value_repr = repr(self._state[key]).encode("utf-8")
        return hashlib.sha256(key.encode("utf-8") + b":" + value_repr).hexdigest()

    # ------------------------------------------------------------------
    # Remote communication stubs
    # ------------------------------------------------------------------

    async def _fetch_remote_digest(self, node_id: str) -> str:
        """Fetch the Merkle root digest from a remote node."""
        # In a real implementation this would perform an RPC call.
        logger.debug("state_synchronizer.fetch_remote_digest", node_id=node_id)
        return ""

    async def _fetch_remote_state(self, node_id: str) -> Dict[str, Any]:
        """Fetch the full state from a remote node."""
        logger.debug("state_synchronizer.fetch_remote_state", node_id=node_id)
        return {}

    async def _push_state_batch(
        self, node_id: str, payload: Dict[str, Any]
    ) -> None:
        """Push a batch of state entries to a remote node."""
        logger.debug(
            "state_synchronizer.push_state_batch",
            node_id=node_id,
            batch_size=len(payload),
        )

    async def _find_divergent_keys(self, node_id: str) -> List[str]:
        """
        Identify keys that differ between local state and a remote node.

        In a production system this would walk the Merkle tree exchanging
        intermediate hashes to narrow down divergent subtrees.
        """
        logger.debug("state_synchronizer.find_divergent_keys", node_id=node_id)
        return list(self._state.keys())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_remote_newer(self, key: str, node_id: str) -> bool:
        """Check whether the remote version of a key is newer than local."""
        if key in self._state_versions:
            # Placeholder: in production compare vector clocks
            return False
        return True

    def _get_or_create_progress(self, node_id: str) -> SyncProgress:
        if node_id not in self._sync_progress:
            self._sync_progress[node_id] = SyncProgress(node_id=node_id)
        return self._sync_progress[node_id]

    async def _get_peer_node_ids(self) -> List[str]:
        """Retrieve list of peer node IDs from the cluster manager."""
        try:
            cluster_state: ClusterState = self._cluster_manager.get_cluster_state()
            local_id = self._cluster_manager.local_node_id
            return [nid for nid in cluster_state.nodes if nid != local_id]
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Background loops
    # ------------------------------------------------------------------

    async def _sync_loop(self) -> None:
        """Background loop performing periodic incremental syncs."""
        logger.info(
            "state_synchronizer.sync_loop.started",
            interval=self._sync_interval,
        )
        while self._running:
            try:
                await asyncio.sleep(self._sync_interval)
                if self._running:
                    await self.incremental_sync()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("state_synchronizer.sync_loop.error", error=str(exc))
                await asyncio.sleep(1.0)

    async def _full_sync_loop(self) -> None:
        """Background loop performing periodic full syncs."""
        logger.info(
            "state_synchronizer.full_sync_loop.started",
            interval=self._full_sync_interval,
        )
        while self._running:
            try:
                await asyncio.sleep(self._full_sync_interval)
                if self._running:
                    await self.full_sync()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("state_synchronizer.full_sync_loop.error", error=str(exc))
                await asyncio.sleep(5.0)

    # ------------------------------------------------------------------
    # Local state mutation helpers (used by other subsystems)
    # ------------------------------------------------------------------

    def put(self, key: str, value: Any, vclock: Optional[VectorClock] = None) -> None:
        """Insert or update a key in the local state store."""
        self._state[key] = value
        self._state_timestamps[key] = time.time()
        if vclock:
            self._state_versions[key] = vclock
        self._merkle_dirty = True

    def get(self, key: str) -> Optional[Any]:
        """Retrieve a value from the local state store."""
        return self._state.get(key)

    def delete(self, key: str) -> bool:
        """Remove a key from the local state store."""
        removed = key in self._state
        self._state.pop(key, None)
        self._state_versions.pop(key, None)
        self._state_timestamps.pop(key, None)
        if removed:
            self._merkle_dirty = True
        return removed

    def keys(self) -> List[str]:
        """Return all keys in the local state store."""
        return list(self._state.keys())
