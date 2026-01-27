"""
AION Distributed Memory - Distributed FAISS Vector Search

Production-grade distributed vector similarity search built on top of
the memory shard manager.  Vectors are partitioned across cluster nodes
using consistent hashing on vector IDs.  Queries are fanned out to all
shards in parallel, and partial results are merged with a top-k merge
sort to produce the final ranked list.
"""

from __future__ import annotations

import asyncio
import heapq
import math
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

import structlog

if TYPE_CHECKING:
    from aion.distributed.memory.sharding import MemoryShardManager

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Internal data structures
# ---------------------------------------------------------------------------

_VECTOR_PREFIX = "__vec:"
_INDEX_META_KEY = "__faiss_index_meta"


@dataclass
class VectorEntry:
    """A stored vector with its metadata."""
    id: str
    vector: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    shard_id: Optional[str] = None


@dataclass
class SearchResult:
    """A single search result with distance score."""
    id: str
    distance: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __lt__(self, other: SearchResult) -> bool:
        """Lower distance means higher relevance."""
        return self.distance < other.distance


@dataclass
class ShardSearchResult:
    """Results collected from one shard."""
    shard_id: str
    results: List[SearchResult] = field(default_factory=list)
    latency_ms: float = 0.0
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Distance functions
# ---------------------------------------------------------------------------


def _euclidean_distance(a: List[float], b: List[float]) -> float:
    """Compute L2 (Euclidean) distance between two vectors."""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _cosine_distance(a: List[float], b: List[float]) -> float:
    """Compute cosine distance (1 - cosine similarity)."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 1.0
    return 1.0 - dot / (norm_a * norm_b)


def _inner_product(a: List[float], b: List[float]) -> float:
    """Compute negative inner product (for max-IP search as min-distance)."""
    return -sum(x * y for x, y in zip(a, b))


_DISTANCE_FNS = {
    "l2": _euclidean_distance,
    "cosine": _cosine_distance,
    "ip": _inner_product,
}


# ---------------------------------------------------------------------------
# Distributed Vector Search
# ---------------------------------------------------------------------------


class DistributedVectorSearch:
    """
    Distributed FAISS-style vector similarity search.

    Vectors are shard-placed via consistent hashing on their string IDs.
    Search fans out to every shard in parallel, each shard performs a
    brute-force (or indexed) scan, and results are merged with a k-way
    merge sort so only the global top-k are returned.
    """

    def __init__(
        self,
        shard_manager: MemoryShardManager,
        *,
        distance_metric: str = "cosine",
        default_k: int = 10,
    ) -> None:
        self._shards = shard_manager
        self._distance_fn = _DISTANCE_FNS.get(distance_metric, _cosine_distance)
        self._distance_metric = distance_metric
        self._default_k = default_k

        # In-memory index per shard (shard_id -> {vec_id -> VectorEntry})
        self._local_index: Dict[str, Dict[str, VectorEntry]] = {}
        self._total_vectors: int = 0
        self._total_searches: int = 0
        self._total_search_time_ms: float = 0.0
        self._lock = asyncio.Lock()

    # -- public API -------------------------------------------------------

    async def add_vector(self, id: str, vector: Any, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Add a single vector to the distributed index.

        The vector is placed on the shard determined by consistent
        hashing of *id*.  Returns ``True`` on success.
        """
        entry = VectorEntry(
            id=id,
            vector=list(vector),
            metadata=metadata or {},
        )

        # Determine shard placement
        nodes = self._shards.get_shard_location(f"{_VECTOR_PREFIX}{id}")
        if not nodes:
            logger.error("vector_search.add_failed", id=id, reason="no_shards")
            return False

        shard_id = nodes[0]
        entry.shard_id = shard_id

        # Store via shard manager for replication
        stored = await self._shards.store(f"{_VECTOR_PREFIX}{id}", entry.__dict__)
        if not stored:
            logger.error("vector_search.add_failed", id=id, reason="shard_store_failed")
            return False

        # Update local index
        async with self._lock:
            if shard_id not in self._local_index:
                self._local_index[shard_id] = {}
            self._local_index[shard_id][id] = entry
            self._total_vectors += 1

        logger.debug("vector_search.added", id=id, shard=shard_id,
                      dim=len(entry.vector))
        return True

    async def remove_vector(self, id: str) -> bool:
        """Remove a vector from the distributed index."""
        key = f"{_VECTOR_PREFIX}{id}"
        deleted = await self._shards.delete(key)

        async with self._lock:
            for shard_id, index in self._local_index.items():
                if id in index:
                    del index[id]
                    self._total_vectors = max(0, self._total_vectors - 1)
                    logger.debug("vector_search.removed", id=id, shard=shard_id)
                    return True

        return deleted

    async def batch_add(self, vectors: Dict[str, Any]) -> Dict[str, bool]:
        """
        Add multiple vectors in batch.

        Parameters
        ----------
        vectors:
            Mapping of ``{id: vector_or_dict}``.  If the value is a dict
            it must contain a ``"vector"`` key and may contain ``"metadata"``.

        Returns
        -------
        Dict mapping each id to its success status.
        """
        results: Dict[str, bool] = {}
        tasks = []

        for vec_id, vec_data in vectors.items():
            if isinstance(vec_data, dict):
                vector = vec_data.get("vector", vec_data)
                metadata = vec_data.get("metadata")
            else:
                vector = vec_data
                metadata = None
            tasks.append((vec_id, vector, metadata))

        # Execute adds concurrently in controlled batches
        batch_size = 64
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i : i + batch_size]
            coros = [
                self.add_vector(vid, vec, meta) for vid, vec, meta in batch
            ]
            batch_results = await asyncio.gather(*coros, return_exceptions=True)
            for (vid, _, _), res in zip(batch, batch_results):
                if isinstance(res, Exception):
                    logger.warning("vector_search.batch_add_error",
                                   id=vid, error=str(res))
                    results[vid] = False
                else:
                    results[vid] = bool(res)

        logger.info("vector_search.batch_add_complete",
                     total=len(vectors), success=sum(results.values()))
        return results

    async def search(
        self,
        query_vector: Any,
        k: Optional[int] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Search for the *k* nearest vectors across all shards.

        Returns a list of ``(id, distance, metadata)`` tuples sorted
        by ascending distance.
        """
        k = k or self._default_k
        query = list(query_vector)
        start = time.monotonic()

        # Fan out to all shards in parallel
        shard_ids = list(self._local_index.keys())
        if not shard_ids:
            return []

        coros = [
            self._search_shard(sid, query, k, filter_metadata)
            for sid in shard_ids
        ]
        shard_results: List[ShardSearchResult] = await asyncio.gather(*coros)

        # Top-k merge sort across shard results
        merged = self._merge_results(shard_results, k)

        elapsed_ms = (time.monotonic() - start) * 1000.0
        self._total_searches += 1
        self._total_search_time_ms += elapsed_ms

        logger.debug("vector_search.search", k=k, shards=len(shard_ids),
                      results=len(merged), latency_ms=round(elapsed_ms, 2))

        return [
            (r.id, r.distance, r.metadata) for r in merged
        ]

    async def get_index_stats(self) -> Dict[str, Any]:
        """Return aggregated statistics across all shards."""
        shard_stats: List[Dict[str, Any]] = []
        total_vectors = 0

        async with self._lock:
            for shard_id, index in self._local_index.items():
                count = len(index)
                total_vectors += count
                dims: Set[int] = set()
                for entry in index.values():
                    dims.add(len(entry.vector))
                shard_stats.append({
                    "shard_id": shard_id,
                    "vector_count": count,
                    "dimensions": sorted(dims),
                })

        avg_search_ms = 0.0
        if self._total_searches > 0:
            avg_search_ms = self._total_search_time_ms / self._total_searches

        return {
            "total_vectors": total_vectors,
            "shard_count": len(self._local_index),
            "distance_metric": self._distance_metric,
            "total_searches": self._total_searches,
            "avg_search_latency_ms": round(avg_search_ms, 3),
            "shards": shard_stats,
        }

    # -- internal ---------------------------------------------------------

    async def _search_shard(
        self,
        shard_id: str,
        query: List[float],
        k: int,
        filter_metadata: Optional[Dict[str, Any]],
    ) -> ShardSearchResult:
        """Search a single shard for the top-k nearest vectors."""
        start = time.monotonic()
        result = ShardSearchResult(shard_id=shard_id)

        try:
            index = self._local_index.get(shard_id, {})
            heap: List[Tuple[float, str, Dict[str, Any]]] = []

            for entry in index.values():
                # Apply metadata filter if provided
                if filter_metadata and not self._matches_filter(entry.metadata, filter_metadata):
                    continue

                dist = self._distance_fn(query, entry.vector)
                if len(heap) < k:
                    # Use negative distance for max-heap behaviour
                    heapq.heappush(heap, (-dist, entry.id, entry.metadata))
                elif dist < -heap[0][0]:
                    heapq.heapreplace(heap, (-dist, entry.id, entry.metadata))

            result.results = sorted(
                [SearchResult(id=vid, distance=-d, metadata=meta) for d, vid, meta in heap],
                key=lambda r: r.distance,
            )
        except Exception as exc:
            result.error = str(exc)
            logger.warning("vector_search.shard_error", shard_id=shard_id, error=str(exc))

        result.latency_ms = (time.monotonic() - start) * 1000.0
        return result

    @staticmethod
    def _merge_results(
        shard_results: List[ShardSearchResult],
        k: int,
    ) -> List[SearchResult]:
        """
        Merge top-k results from multiple shards using k-way merge.

        Uses a min-heap to efficiently combine pre-sorted per-shard
        lists into a single globally sorted top-k list.
        """
        iterators = []
        for sr in shard_results:
            if sr.error is None and sr.results:
                iterators.append(iter(sr.results))

        merged: List[SearchResult] = list(heapq.merge(*iterators))
        return merged[:k]

    @staticmethod
    def _matches_filter(metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check whether *metadata* satisfies all *filters*."""
        for key, expected in filters.items():
            actual = metadata.get(key)
            if actual != expected:
                return False
        return True
