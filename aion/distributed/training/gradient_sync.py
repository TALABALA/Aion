"""
AION Distributed Gradient Synchronization

State-of-the-art gradient synchronization primitives for distributed
reinforcement learning. Provides all-reduce, ring all-reduce, gradient
compression (top-K sparsification, random sparsification, quantization),
error-feedback accumulation, gradient clipping, and mixed-precision
communication support with bandwidth tracking.
"""

from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import structlog

try:
    import numpy as np

    _HAS_NUMPY = True
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]
    _HAS_NUMPY = False

from aion.distributed.types import DistributedTask, NodeInfo, TaskPriority, TaskType

if TYPE_CHECKING:
    from aion.distributed.cluster.manager import ClusterManager

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class GradientSyncConfig:
    """Configuration for gradient synchronisation."""

    # Compression
    compression_enabled: bool = True
    compression_method: str = "topk"  # "topk", "random", "quantize"
    topk_ratio: float = 0.1  # Keep top 10% of gradients by magnitude
    random_sparsity: float = 0.1  # Keep 10% of gradients randomly
    quantize_bits: int = 16  # 16 for float16, 8 for int8

    # Error feedback
    error_feedback_enabled: bool = True

    # Clipping
    gradient_clip_norm: float = 1.0

    # Mixed precision
    mixed_precision: bool = True  # float16 for comms, float32 for compute

    # Bandwidth
    max_bandwidth_bytes_per_sec: int = 0  # 0 = unlimited


# ---------------------------------------------------------------------------
# GradientSynchronizer
# ---------------------------------------------------------------------------


class GradientSynchronizer:
    """
    SOTA gradient synchronization with compression for distributed training.

    Implements multiple communication patterns and compression strategies
    to minimise network overhead while preserving gradient quality.

    Features:
        - All-reduce: every node contributes and receives averaged gradients
        - Ring all-reduce: bandwidth-optimal O(2(N-1)/N * data_size) pattern
        - Top-K sparsification with error feedback accumulation
        - Random sparsification for stochastic communication
        - Quantization (float32 -> float16 / int8)
        - Gradient clipping by global L2 norm
        - Mixed precision: float16 over the wire, float32 for computation
        - Bandwidth tracking and optimisation
    """

    def __init__(
        self,
        cluster_manager: ClusterManager,
        config: Optional[GradientSyncConfig] = None,
    ) -> None:
        self._cluster = cluster_manager
        self._config = config or GradientSyncConfig()

        # Error feedback residuals (per parameter key)
        self._error_residuals: Dict[str, float] = {}

        # Bandwidth tracking
        self._total_bytes_sent: int = 0
        self._total_bytes_received: int = 0
        self._bytes_saved_by_compression: int = 0
        self._sync_count: int = 0
        self._last_sync_time: float = 0.0

        # Round tracking
        self._round: int = 0

        logger.info(
            "gradient_synchronizer_created",
            compression=self._config.compression_method if self._config.compression_enabled else "none",
            mixed_precision=self._config.mixed_precision,
        )

    # ------------------------------------------------------------------
    # All-Reduce
    # ------------------------------------------------------------------

    async def all_reduce(self, gradients: Dict[str, Any]) -> Dict[str, Any]:
        """Perform an all-reduce operation across all cluster nodes.

        Every node contributes its local gradients and receives the
        element-wise average of all contributions.

        Args:
            gradients: Local gradient dict mapping parameter names to values.

        Returns:
            Averaged gradient dict after reducing across all nodes.
        """
        self._round += 1
        start = time.monotonic()

        # Optionally compress before sending
        to_send = gradients
        if self._config.compression_enabled:
            to_send = self.compress(gradients)

        # Build a distributed task for all-reduce
        task = DistributedTask(
            name="gradient_all_reduce",
            task_type=TaskType.TRAINING_STEP.value,
            priority=TaskPriority.HIGH,
            payload={
                "action": "all_reduce",
                "round": self._round,
                "node_id": self._get_node_id(),
                "gradients": to_send,
                "compressed": self._config.compression_enabled,
                "timestamp": time.time(),
            },
        )

        # Track bandwidth
        payload_size = self._estimate_size(to_send)
        self._total_bytes_sent += payload_size

        try:
            await self._cluster.submit_task(task)
        except Exception:
            logger.exception("all_reduce_submit_failed", round=self._round)
            return gradients  # Fall back to local gradients

        # Collect results from all nodes
        all_node_gradients = await self._collect_node_gradients(to_send)

        # Decompress if needed
        decompressed: List[Dict[str, Any]] = []
        for node_grads in all_node_gradients:
            if self._config.compression_enabled:
                decompressed.append(self.decompress(node_grads))
            else:
                decompressed.append(node_grads)

        # Average
        result = self._element_wise_average(decompressed)

        elapsed_ms = (time.monotonic() - start) * 1000.0
        self._sync_count += 1
        self._last_sync_time = time.monotonic()

        logger.debug(
            "all_reduce_completed",
            round=self._round,
            num_nodes=len(all_node_gradients),
            elapsed_ms=round(elapsed_ms, 2),
            payload_bytes=payload_size,
        )

        return result

    # ------------------------------------------------------------------
    # Ring All-Reduce
    # ------------------------------------------------------------------

    async def ring_all_reduce(self, gradients: Dict[str, Any]) -> Dict[str, Any]:
        """Perform a ring all-reduce for bandwidth-optimal communication.

        Achieves O(2(N-1)/N * data_size) total communication, which is
        near-optimal for large gradient tensors across many nodes.

        The algorithm proceeds in two phases:
        1. Scatter-reduce: each node sends a chunk to its right neighbour,
           accumulating partial sums around the ring.
        2. All-gather: the fully-reduced chunks are circulated so every
           node obtains the complete averaged result.

        Args:
            gradients: Local gradient dict mapping parameter names to values.

        Returns:
            Averaged gradient dict after ring-reducing across all nodes.
        """
        self._round += 1
        start = time.monotonic()

        nodes = self._get_sorted_nodes()
        num_nodes = len(nodes)

        if num_nodes <= 1:
            return gradients

        # Partition gradient keys into N chunks
        keys = sorted(gradients.keys())
        chunks = self._partition_keys(keys, num_nodes)

        node_id = self._get_node_id()
        my_rank = self._get_rank(node_id, nodes)

        # Initialise chunk accumulators with local values
        chunk_data: List[Dict[str, float]] = []
        for chunk_keys in chunks:
            chunk_data.append({
                k: self._safe_float(gradients.get(k, 0.0))
                for k in chunk_keys
            })

        rpc_client = getattr(self._cluster, "_rpc_client", None)

        def _resolve_address(nid: str) -> Optional[str]:
            try:
                state = self._cluster.get_state()
                if state and nid in state.nodes:
                    return state.nodes[nid].address
            except Exception:
                pass
            return None

        right_id = nodes[(my_rank + 1) % num_nodes]
        left_id = nodes[(my_rank - 1) % num_nodes]
        right_addr = _resolve_address(right_id)

        # Phase 1 - Scatter-reduce: N-1 steps
        for step in range(num_nodes - 1):
            send_chunk_idx = (my_rank - step) % num_nodes
            recv_chunk_idx = (my_rank - step - 1) % num_nodes

            outgoing = chunk_data[send_chunk_idx]
            if self._config.compression_enabled:
                outgoing = self.compress(outgoing)

            payload_size = self._estimate_size(outgoing)
            self._total_bytes_sent += payload_size

            # Send chunk to right neighbour via RPC
            if rpc_client and right_addr:
                try:
                    resp = await rpc_client.send_ring_chunk(
                        right_addr,
                        outgoing,
                        phase="scatter_reduce",
                        step=step,
                        chunk_idx=send_chunk_idx,
                        round_id=self._round,
                    )
                    # The response contains the chunk from our left neighbour
                    if resp and "chunk_data" in resp:
                        incoming = resp["chunk_data"]
                    else:
                        incoming = outgoing  # Fallback: echo local
                except Exception:
                    logger.debug(
                        "ring_scatter_reduce_rpc_failed",
                        step=step,
                        round=self._round,
                    )
                    incoming = outgoing
            else:
                incoming = outgoing

            if self._config.compression_enabled:
                incoming = self.decompress(incoming)

            for k, v in incoming.items():
                if k in chunk_data[recv_chunk_idx]:
                    chunk_data[recv_chunk_idx][k] += self._safe_float(v)

        # Phase 2 - All-gather: N-1 steps
        for step in range(num_nodes - 1):
            send_chunk_idx = (my_rank - step + 1) % num_nodes
            recv_chunk_idx = (my_rank - step) % num_nodes

            outgoing = chunk_data[send_chunk_idx]
            payload_size = self._estimate_size(outgoing)
            self._total_bytes_sent += payload_size

            # Send fully-reduced chunk to right neighbour via RPC
            if rpc_client and right_addr:
                try:
                    resp = await rpc_client.send_ring_chunk(
                        right_addr,
                        outgoing,
                        phase="all_gather",
                        step=step,
                        chunk_idx=send_chunk_idx,
                        round_id=self._round,
                    )
                    if resp and "chunk_data" in resp:
                        chunk_data[recv_chunk_idx] = resp["chunk_data"]
                    else:
                        chunk_data[recv_chunk_idx] = outgoing
                except Exception:
                    logger.debug(
                        "ring_all_gather_rpc_failed",
                        step=step,
                        round=self._round,
                    )
                    chunk_data[recv_chunk_idx] = outgoing
            else:
                chunk_data[recv_chunk_idx] = outgoing

        # Reassemble full gradient from chunks and divide by N
        result: Dict[str, Any] = {}
        for chunk in chunk_data:
            for k, v in chunk.items():
                result[k] = v / num_nodes

        elapsed_ms = (time.monotonic() - start) * 1000.0
        self._sync_count += 1

        logger.debug(
            "ring_all_reduce_completed",
            round=self._round,
            num_nodes=num_nodes,
            elapsed_ms=round(elapsed_ms, 2),
        )

        return result

    # ------------------------------------------------------------------
    # Compression / Decompression
    # ------------------------------------------------------------------

    def compress(self, gradients: Dict[str, Any]) -> Dict[str, Any]:
        """Compress gradients using the configured compression strategy.

        Supports:
        - Top-K sparsification: retain only the largest K% by magnitude.
        - Random sparsification: randomly retain a fraction of entries.
        - Quantization: reduce float32 to float16 or int8.

        When error feedback is enabled the compression residual is
        accumulated and added back in the next round.

        Args:
            gradients: Full gradient dict.

        Returns:
            Compressed gradient dict with metadata.
        """
        method = self._config.compression_method

        # Apply error feedback residuals
        if self._config.error_feedback_enabled:
            gradients = self._add_error_feedback(gradients)

        original_size = len(gradients)

        if method == "topk":
            compressed = self._topk_sparsify(gradients, self._config.topk_ratio)
        elif method == "random":
            compressed = self._random_sparsify(gradients, self._config.random_sparsity)
        elif method == "quantize":
            compressed = self._quantize(gradients, self._config.quantize_bits)
        else:
            compressed = dict(gradients)

        # Accumulate error feedback
        if self._config.error_feedback_enabled and method in ("topk", "random"):
            self._accumulate_error(gradients, compressed)

        compressed_size = len(compressed)
        saved = max(0, original_size - compressed_size)
        self._bytes_saved_by_compression += saved * 4  # rough estimate

        compressed["__meta__"] = {
            "method": method,
            "original_keys": original_size,
            "compressed_keys": compressed_size,
            "round": self._round,
        }

        return compressed

    def decompress(self, compressed: Dict[str, Any]) -> Dict[str, Any]:
        """Decompress a previously compressed gradient dict.

        Reverses quantization if applicable and strips compression
        metadata.  Sparsified gradients are returned as-is (missing
        keys are implicitly zero).

        Args:
            compressed: Compressed gradient dict with ``__meta__`` key.

        Returns:
            Decompressed gradient dict.
        """
        meta = compressed.pop("__meta__", {})
        method = meta.get("method", "none")

        if method == "quantize":
            return self._dequantize(compressed, meta)

        # For sparsification, the dict already contains the non-zero entries
        return {k: v for k, v in compressed.items() if k != "__meta__"}

    # ------------------------------------------------------------------
    # Gradient Clipping
    # ------------------------------------------------------------------

    def apply_gradient_clipping(
        self, gradients: Dict[str, Any], max_norm: float
    ) -> Dict[str, Any]:
        """Clip gradients by global L2 norm.

        If the L2 norm of all gradient values exceeds ``max_norm``,
        every value is scaled down proportionally.

        Args:
            gradients: Gradient dict.
            max_norm: Maximum allowed L2 norm.

        Returns:
            Clipped gradient dict (original is not mutated).
        """
        total_norm_sq = 0.0
        numeric_keys: List[str] = []

        for key, value in gradients.items():
            fval = self._safe_float(value)
            total_norm_sq += fval * fval
            numeric_keys.append(key)

        total_norm = math.sqrt(total_norm_sq)

        if total_norm <= max_norm or total_norm == 0.0:
            return dict(gradients)

        scale = max_norm / total_norm
        clipped: Dict[str, Any] = {}
        for key in gradients:
            val = gradients[key]
            if isinstance(val, (int, float)):
                clipped[key] = float(val) * scale
            else:
                clipped[key] = val

        logger.debug(
            "gradient_clipped",
            original_norm=round(total_norm, 4),
            max_norm=max_norm,
            scale=round(scale, 6),
        )

        return clipped

    # ------------------------------------------------------------------
    # Top-K Sparsification
    # ------------------------------------------------------------------

    def _topk_sparsify(
        self, gradients: Dict[str, Any], ratio: float
    ) -> Dict[str, Any]:
        """Keep only the top K% of gradient entries by magnitude."""
        items = [
            (k, self._safe_float(v))
            for k, v in gradients.items()
            if not k.startswith("__")
        ]

        if not items:
            return {}

        k = max(1, int(len(items) * ratio))

        # Sort by absolute value descending
        items.sort(key=lambda x: abs(x[1]), reverse=True)
        top_items = items[:k]

        return {key: val for key, val in top_items}

    # ------------------------------------------------------------------
    # Random Sparsification
    # ------------------------------------------------------------------

    def _random_sparsify(
        self, gradients: Dict[str, Any], ratio: float
    ) -> Dict[str, Any]:
        """Randomly keep a fraction of gradient entries."""
        import random

        keys = [k for k in gradients if not k.startswith("__")]
        if not keys:
            return {}

        k = max(1, int(len(keys) * ratio))
        selected = random.sample(keys, min(k, len(keys)))

        # Scale up to preserve expected value
        scale = 1.0 / ratio if ratio > 0 else 1.0
        return {
            key: self._safe_float(gradients[key]) * scale for key in selected
        }

    # ------------------------------------------------------------------
    # Quantization
    # ------------------------------------------------------------------

    def _quantize(
        self, gradients: Dict[str, Any], bits: int
    ) -> Dict[str, Any]:
        """Quantize gradient values from float32 to lower precision.

        Supports 16-bit (float16) and 8-bit (int8) quantization.
        For int8, values are scaled to [-127, 127] and a scale factor
        is stored in metadata.
        """
        result: Dict[str, Any] = {}

        if bits == 16:
            # float32 -> float16 (half precision)
            for k, v in gradients.items():
                if k.startswith("__"):
                    continue
                fval = self._safe_float(v)
                if _HAS_NUMPY:
                    result[k] = float(np.float16(fval))
                else:
                    # Manual half-precision approximation
                    result[k] = round(fval, 3)

        elif bits == 8:
            # int8 quantization with min-max scaling
            values = {
                k: self._safe_float(v)
                for k, v in gradients.items()
                if not k.startswith("__")
            }
            if not values:
                return result

            abs_max = max(abs(v) for v in values.values()) or 1.0
            scale = abs_max / 127.0

            for k, v in values.items():
                quantized = int(round(v / scale))
                quantized = max(-127, min(127, quantized))
                result[k] = quantized

            result["__quantize_scale__"] = scale
        else:
            result = {
                k: v for k, v in gradients.items() if not k.startswith("__")
            }

        return result

    def _dequantize(
        self, compressed: Dict[str, Any], meta: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Reverse quantization back to float32."""
        bits = meta.get("bits", self._config.quantize_bits)
        scale = compressed.pop("__quantize_scale__", 1.0)

        result: Dict[str, Any] = {}
        for k, v in compressed.items():
            if k.startswith("__"):
                continue
            if bits == 8:
                result[k] = float(v) * scale
            else:
                result[k] = float(v)

        return result

    # ------------------------------------------------------------------
    # Error Feedback
    # ------------------------------------------------------------------

    def _add_error_feedback(self, gradients: Dict[str, Any]) -> Dict[str, Any]:
        """Add accumulated error residuals to current gradients."""
        result: Dict[str, Any] = {}
        for k, v in gradients.items():
            if k.startswith("__"):
                result[k] = v
                continue
            fval = self._safe_float(v)
            residual = self._error_residuals.get(k, 0.0)
            result[k] = fval + residual
        return result

    def _accumulate_error(
        self,
        original: Dict[str, Any],
        compressed: Dict[str, Any],
    ) -> None:
        """Accumulate the compression error for the next round."""
        for k in original:
            if k.startswith("__"):
                continue
            orig_val = self._safe_float(original.get(k, 0.0))
            comp_val = self._safe_float(compressed.get(k, 0.0))
            self._error_residuals[k] = orig_val - comp_val

    # ------------------------------------------------------------------
    # Mixed Precision Helpers
    # ------------------------------------------------------------------

    def to_half_precision(self, gradients: Dict[str, Any]) -> Dict[str, Any]:
        """Convert gradients to float16 for network transfer."""
        result: Dict[str, Any] = {}
        for k, v in gradients.items():
            fval = self._safe_float(v)
            if _HAS_NUMPY:
                result[k] = float(np.float16(fval))
            else:
                result[k] = round(fval, 3)
        return result

    def to_full_precision(self, gradients: Dict[str, Any]) -> Dict[str, Any]:
        """Convert gradients back to float32 for computation."""
        return {k: float(v) for k, v in gradients.items()}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _collect_node_gradients(
        self, local_gradients: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Collect gradient contributions from all cluster nodes via RPC.

        Sends a ``collect_gradients`` RPC to each peer and gathers their
        gradient dicts.  The local node's gradients are always included.
        Falls back to local-only if RPC is unavailable.
        """
        results: List[Dict[str, Any]] = [local_gradients]

        rpc_client = getattr(self._cluster, "_rpc_client", None)
        peer_addresses = self._get_peer_addresses()

        if rpc_client is None or not peer_addresses:
            return results

        async def _fetch(address: str) -> Optional[Dict[str, Any]]:
            try:
                resp = await rpc_client.collect_gradients(address, self._round)
                if resp is not None:
                    payload_size = self._estimate_size(resp)
                    self._total_bytes_received += payload_size
                return resp
            except Exception:
                logger.debug(
                    "gradient_collect_failed",
                    address=address,
                    round=self._round,
                )
                return None

        import asyncio
        fetches = [_fetch(addr) for addr in peer_addresses]
        responses = await asyncio.gather(*fetches, return_exceptions=True)

        for resp in responses:
            if isinstance(resp, dict) and resp:
                results.append(resp)

        return results

    def _get_peer_addresses(self) -> List[str]:
        """Return network addresses of all peer nodes (excluding self)."""
        try:
            state = self._cluster.get_state()
            if state is None:
                return []
            local_id = self._get_node_id()
            return [
                node.address
                for nid, node in state.nodes.items()
                if nid != local_id and hasattr(node, "address") and node.address
            ]
        except Exception:
            return []

    def _element_wise_average(
        self, all_gradients: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compute the element-wise average across gradient dicts."""
        if not all_gradients:
            return {}

        accumulated: Dict[str, float] = {}
        counts: Dict[str, int] = {}

        for gdict in all_gradients:
            for k, v in gdict.items():
                if k.startswith("__"):
                    continue
                fval = self._safe_float(v)
                accumulated[k] = accumulated.get(k, 0.0) + fval
                counts[k] = counts.get(k, 0) + 1

        return {
            k: accumulated[k] / counts[k] for k in accumulated if counts.get(k, 0) > 0
        }

    def _partition_keys(
        self, keys: List[str], n: int
    ) -> List[List[str]]:
        """Split a sorted key list into n roughly equal chunks."""
        chunks: List[List[str]] = [[] for _ in range(n)]
        for i, key in enumerate(keys):
            chunks[i % n].append(key)
        return chunks

    def _get_sorted_nodes(self) -> List[str]:
        """Return a deterministically sorted list of node IDs."""
        try:
            state = self._cluster.get_state()
            if state is not None:
                return sorted(state.nodes.keys())
        except Exception:
            pass
        return [self._get_node_id()]

    def _get_rank(self, node_id: str, sorted_nodes: List[str]) -> int:
        """Return this node's rank (index) within the sorted node list."""
        try:
            return sorted_nodes.index(node_id)
        except ValueError:
            return 0

    def _get_node_id(self) -> str:
        try:
            return self._cluster.node_id
        except Exception:
            return "unknown"

    def _get_cluster_size(self) -> int:
        try:
            state = self._cluster.get_state()
            if state is not None:
                return len(state.nodes)
        except Exception:
            pass
        return 1

    @staticmethod
    def _safe_float(value: Any) -> float:
        """Convert value to float, defaulting to 0.0."""
        if isinstance(value, (int, float)):
            return float(value)
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _estimate_size(data: Dict[str, Any]) -> int:
        """Rough byte-size estimate for a gradient dict."""
        # Each entry is roughly key-length + 8 bytes (float64)
        return sum(len(str(k)) + 8 for k in data)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Return synchronization statistics for monitoring."""
        return {
            "round": self._round,
            "sync_count": self._sync_count,
            "total_bytes_sent": self._total_bytes_sent,
            "total_bytes_received": self._total_bytes_received,
            "bytes_saved_by_compression": self._bytes_saved_by_compression,
            "compression_method": self._config.compression_method,
            "compression_enabled": self._config.compression_enabled,
            "mixed_precision": self._config.mixed_precision,
            "error_residual_keys": len(self._error_residuals),
        }
