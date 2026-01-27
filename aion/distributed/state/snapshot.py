"""
AION Distributed Snapshot Manager

Manages state snapshots for Raft log compaction and node recovery.
Snapshots are stored as compressed JSON files with SHA-256 checksum
verification, atomic creation via temp-file-then-rename, and
configurable automatic scheduling and retention policies.
"""

from __future__ import annotations

import asyncio
import gzip
import hashlib
import json
import os
import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import structlog

from aion.distributed.types import (
    ClusterState,
    NodeInfo,
    RaftLogEntry,
    SnapshotMetadata,
)

if TYPE_CHECKING:
    from aion.distributed.cluster.manager import ClusterManager

logger = structlog.get_logger(__name__)


# Filename conventions
_SNAPSHOT_PREFIX = "snapshot_"
_SNAPSHOT_SUFFIX = ".json.gz"
_META_SUFFIX = ".meta.json"


class SnapshotError(Exception):
    """Raised when a snapshot operation fails."""


class SnapshotManager:
    """
    Manages state snapshots for log compaction and node recovery.

    Snapshots are compressed JSON files accompanied by a metadata
    sidecar with checksum information. Creation is atomic: data is
    written to a temporary file in the same directory and renamed into
    place only on success.

    Features:
    - Atomic snapshot creation (write-to-temp, rename)
    - SHA-256 checksum verification on restore
    - Compressed storage (gzip)
    - Automatic periodic snapshot creation
    - Configurable retention policy
    - Snapshot transfer to peer nodes
    """

    def __init__(
        self,
        cluster_manager: ClusterManager,
        snapshot_dir: str = "data/distributed/snapshots",
        *,
        auto_snapshot_interval_seconds: float = 3600.0,
        max_snapshots_retained: int = 5,
        compression_level: int = 6,
    ) -> None:
        self._cluster_manager = cluster_manager
        self._snapshot_dir = Path(snapshot_dir)
        self._auto_interval = auto_snapshot_interval_seconds
        self._max_retained = max_snapshots_retained
        self._compression_level = compression_level

        # Internal caches
        self._metadata_cache: Dict[str, SnapshotMetadata] = {}
        self._latest_snapshot_id: Optional[str] = None

        # Background task
        self._running = False
        self._auto_task: Optional[asyncio.Task[None]] = None

        # Ensure the snapshot directory exists
        self._snapshot_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "snapshot_manager.init",
            snapshot_dir=str(self._snapshot_dir),
            auto_interval=auto_snapshot_interval_seconds,
            max_retained=max_snapshots_retained,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the automatic snapshot background task."""
        if self._running:
            logger.warning("snapshot_manager.already_running")
            return
        self._running = True
        self._auto_task = asyncio.create_task(self._auto_snapshot_loop())
        # Load existing metadata on start
        await self._load_metadata_cache()
        logger.info("snapshot_manager.started")

    async def stop(self) -> None:
        """Stop the automatic snapshot background task."""
        if not self._running:
            return
        self._running = False
        if self._auto_task and not self._auto_task.done():
            self._auto_task.cancel()
            try:
                await self._auto_task
            except asyncio.CancelledError:
                pass
        self._auto_task = None
        logger.info("snapshot_manager.stopped")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def create_snapshot(self) -> SnapshotMetadata:
        """
        Create a new state snapshot.

        The cluster state is serialised to compressed JSON and written
        atomically. A metadata sidecar is persisted alongside the data
        file.
        """
        snapshot_id = str(uuid.uuid4())
        start = time.monotonic()

        logger.info("snapshot_manager.create.start", snapshot_id=snapshot_id)

        try:
            cluster_state = self._cluster_manager.get_cluster_state()
        except Exception as exc:
            raise SnapshotError(f"Failed to read cluster state: {exc}") from exc

        # Serialise state
        state_dict = self._serialise_cluster_state(cluster_state)
        json_bytes = json.dumps(state_dict, default=str, sort_keys=True).encode("utf-8")

        # Compress
        compressed = gzip.compress(json_bytes, compresslevel=self._compression_level)

        # Compute checksum over the *compressed* data
        checksum = hashlib.sha256(compressed).hexdigest()

        # Write atomically
        data_path = self._snapshot_dir / f"{_SNAPSHOT_PREFIX}{snapshot_id}{_SNAPSHOT_SUFFIX}"
        meta_path = self._snapshot_dir / f"{_SNAPSHOT_PREFIX}{snapshot_id}{_META_SUFFIX}"

        await self._atomic_write(data_path, compressed)

        # Build metadata
        metadata = SnapshotMetadata(
            snapshot_id=snapshot_id,
            last_included_index=cluster_state.commit_index,
            last_included_term=cluster_state.term,
            node_count=len(cluster_state.nodes),
            size_bytes=len(compressed),
            checksum=checksum,
            created_at=datetime.now(),
        )

        # Write metadata sidecar
        meta_bytes = json.dumps(metadata.to_dict(), default=str).encode("utf-8")
        await self._atomic_write(meta_path, meta_bytes)

        # Update caches
        self._metadata_cache[snapshot_id] = metadata
        self._latest_snapshot_id = snapshot_id

        elapsed_ms = (time.monotonic() - start) * 1000
        logger.info(
            "snapshot_manager.create.complete",
            snapshot_id=snapshot_id,
            size_bytes=len(compressed),
            checksum=checksum,
            duration_ms=round(elapsed_ms, 2),
        )

        # Enforce retention
        await self._enforce_retention()

        return metadata

    async def restore_snapshot(self, snapshot_id: str) -> Dict[str, Any]:
        """
        Restore a snapshot by its ID.

        Reads the compressed data file, verifies the SHA-256 checksum,
        and returns the deserialised state dictionary.
        """
        data_path = self._snapshot_dir / f"{_SNAPSHOT_PREFIX}{snapshot_id}{_SNAPSHOT_SUFFIX}"
        meta_path = self._snapshot_dir / f"{_SNAPSHOT_PREFIX}{snapshot_id}{_META_SUFFIX}"

        if not data_path.exists():
            raise SnapshotError(f"Snapshot data file not found: {data_path}")

        logger.info("snapshot_manager.restore.start", snapshot_id=snapshot_id)

        # Read compressed data
        compressed = await asyncio.to_thread(data_path.read_bytes)

        # Verify checksum
        actual_checksum = hashlib.sha256(compressed).hexdigest()
        expected_checksum = await self._read_expected_checksum(snapshot_id, meta_path)

        if expected_checksum and actual_checksum != expected_checksum:
            raise SnapshotError(
                f"Checksum mismatch for snapshot {snapshot_id}: "
                f"expected {expected_checksum}, got {actual_checksum}"
            )

        # Decompress and parse
        json_bytes = gzip.decompress(compressed)
        state_dict: Dict[str, Any] = json.loads(json_bytes)

        logger.info(
            "snapshot_manager.restore.complete",
            snapshot_id=snapshot_id,
            checksum_verified=expected_checksum is not None,
        )
        return state_dict

    async def list_snapshots(self) -> List[SnapshotMetadata]:
        """List all available snapshots, sorted by creation time (newest first)."""
        await self._load_metadata_cache()
        snapshots = sorted(
            self._metadata_cache.values(),
            key=lambda m: m.created_at,
            reverse=True,
        )
        return snapshots

    async def delete_snapshot(self, snapshot_id: str) -> bool:
        """
        Delete a snapshot and its metadata sidecar.

        Returns True if the snapshot was deleted, False if not found.
        """
        data_path = self._snapshot_dir / f"{_SNAPSHOT_PREFIX}{snapshot_id}{_SNAPSHOT_SUFFIX}"
        meta_path = self._snapshot_dir / f"{_SNAPSHOT_PREFIX}{snapshot_id}{_META_SUFFIX}"

        deleted = False
        for path in (data_path, meta_path):
            if path.exists():
                await asyncio.to_thread(path.unlink)
                deleted = True

        self._metadata_cache.pop(snapshot_id, None)
        if self._latest_snapshot_id == snapshot_id:
            self._latest_snapshot_id = None

        if deleted:
            logger.info("snapshot_manager.delete", snapshot_id=snapshot_id)
        else:
            logger.warning(
                "snapshot_manager.delete.not_found", snapshot_id=snapshot_id
            )
        return deleted

    async def get_latest_snapshot(self) -> Optional[SnapshotMetadata]:
        """Return metadata for the most recent snapshot, or None."""
        snapshots = await self.list_snapshots()
        return snapshots[0] if snapshots else None

    async def transfer_to_node(
        self, node_id: str, snapshot_id: str
    ) -> bool:
        """
        Transfer a snapshot to a remote peer node.

        Reads the compressed data and sends it via the cluster manager's
        transport layer. In a production system this would stream chunks
        via gRPC InstallSnapshot RPCs.
        """
        data_path = self._snapshot_dir / f"{_SNAPSHOT_PREFIX}{snapshot_id}{_SNAPSHOT_SUFFIX}"
        if not data_path.exists():
            raise SnapshotError(f"Snapshot not found: {snapshot_id}")

        compressed = await asyncio.to_thread(data_path.read_bytes)
        metadata = self._metadata_cache.get(snapshot_id)

        logger.info(
            "snapshot_manager.transfer.start",
            node_id=node_id,
            snapshot_id=snapshot_id,
            size_bytes=len(compressed),
        )

        try:
            # Stub: in production this would use gRPC streaming
            await self._send_snapshot_to_node(node_id, snapshot_id, compressed, metadata)
            logger.info(
                "snapshot_manager.transfer.complete",
                node_id=node_id,
                snapshot_id=snapshot_id,
            )
            return True
        except Exception as exc:
            logger.error(
                "snapshot_manager.transfer.failed",
                node_id=node_id,
                snapshot_id=snapshot_id,
                error=str(exc),
            )
            return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _serialise_cluster_state(self, state: ClusterState) -> Dict[str, Any]:
        """Convert cluster state to a serialisable dictionary."""
        return {
            "cluster_id": state.cluster_id,
            "name": state.name,
            "leader_id": state.leader_id,
            "term": state.term,
            "commit_index": state.commit_index,
            "epoch": state.epoch,
            "config_version": state.config_version,
            "replication_factor": state.replication_factor,
            "nodes": {
                nid: node.to_dict() for nid, node in state.nodes.items()
            },
            "created_at": state.created_at.isoformat(),
            "updated_at": state.updated_at.isoformat(),
        }

    async def _atomic_write(self, target_path: Path, data: bytes) -> None:
        """
        Write data atomically by creating a temp file then renaming.

        The temp file is created in the same directory as the target to
        ensure the rename is atomic on the same filesystem.
        """
        target_dir = target_path.parent

        def _write() -> None:
            fd, tmp_path = tempfile.mkstemp(
                dir=str(target_dir), suffix=".tmp"
            )
            try:
                os.write(fd, data)
                os.fsync(fd)
                os.close(fd)
                os.rename(tmp_path, str(target_path))
            except Exception:
                os.close(fd) if not os.get_inheritable(fd) else None
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                raise

        await asyncio.to_thread(_write)

    async def _read_expected_checksum(
        self, snapshot_id: str, meta_path: Path
    ) -> Optional[str]:
        """Read the expected checksum from the metadata sidecar."""
        # Try in-memory cache first
        if snapshot_id in self._metadata_cache:
            return self._metadata_cache[snapshot_id].checksum

        if not meta_path.exists():
            return None

        try:
            meta_bytes = await asyncio.to_thread(meta_path.read_bytes)
            meta_dict = json.loads(meta_bytes)
            return meta_dict.get("checksum")
        except Exception as exc:
            logger.warning(
                "snapshot_manager.read_checksum.error",
                snapshot_id=snapshot_id,
                error=str(exc),
            )
            return None

    async def _load_metadata_cache(self) -> None:
        """Scan the snapshot directory and load all metadata into cache."""
        if not self._snapshot_dir.exists():
            return

        meta_files = sorted(self._snapshot_dir.glob(f"{_SNAPSHOT_PREFIX}*{_META_SUFFIX}"))
        for meta_path in meta_files:
            try:
                meta_bytes = await asyncio.to_thread(meta_path.read_bytes)
                meta_dict = json.loads(meta_bytes)
                sid = meta_dict.get("snapshot_id", "")
                if sid and sid not in self._metadata_cache:
                    self._metadata_cache[sid] = SnapshotMetadata(
                        snapshot_id=sid,
                        last_included_index=meta_dict.get("last_included_index", 0),
                        last_included_term=meta_dict.get("last_included_term", 0),
                        size_bytes=meta_dict.get("size_bytes", 0),
                        checksum=meta_dict.get("checksum", ""),
                        created_at=datetime.fromisoformat(
                            meta_dict["created_at"]
                        )
                        if "created_at" in meta_dict
                        else datetime.now(),
                    )
            except Exception as exc:
                logger.warning(
                    "snapshot_manager.load_meta.error",
                    path=str(meta_path),
                    error=str(exc),
                )

        # Determine latest
        if self._metadata_cache:
            latest = max(
                self._metadata_cache.values(), key=lambda m: m.created_at
            )
            self._latest_snapshot_id = latest.snapshot_id

    async def _enforce_retention(self) -> None:
        """Delete old snapshots beyond the retention limit."""
        snapshots = await self.list_snapshots()
        if len(snapshots) <= self._max_retained:
            return

        to_delete = snapshots[self._max_retained :]
        for meta in to_delete:
            await self.delete_snapshot(meta.snapshot_id)
            logger.info(
                "snapshot_manager.retention.deleted",
                snapshot_id=meta.snapshot_id,
                created_at=meta.created_at.isoformat(),
            )

    async def _send_snapshot_to_node(
        self,
        node_id: str,
        snapshot_id: str,
        data: bytes,
        metadata: Optional[SnapshotMetadata],
    ) -> None:
        """
        Send snapshot data to a remote node.

        Stub for the transport layer. In production this streams chunks
        via gRPC InstallSnapshot RPCs.
        """
        logger.debug(
            "snapshot_manager.send_to_node",
            node_id=node_id,
            snapshot_id=snapshot_id,
            size_bytes=len(data),
        )

    # ------------------------------------------------------------------
    # Background loop
    # ------------------------------------------------------------------

    async def _auto_snapshot_loop(self) -> None:
        """Background loop that creates snapshots at regular intervals."""
        logger.info(
            "snapshot_manager.auto_loop.started",
            interval=self._auto_interval,
        )
        while self._running:
            try:
                await asyncio.sleep(self._auto_interval)
                if self._running:
                    await self.create_snapshot()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(
                    "snapshot_manager.auto_loop.error", error=str(exc)
                )
                await asyncio.sleep(10.0)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Return current snapshot manager status."""
        return {
            "running": self._running,
            "snapshot_dir": str(self._snapshot_dir),
            "snapshot_count": len(self._metadata_cache),
            "max_retained": self._max_retained,
            "auto_interval_seconds": self._auto_interval,
            "latest_snapshot_id": self._latest_snapshot_id,
            "latest_snapshot": (
                self._metadata_cache[self._latest_snapshot_id].to_dict()
                if self._latest_snapshot_id
                and self._latest_snapshot_id in self._metadata_cache
                else None
            ),
        }
