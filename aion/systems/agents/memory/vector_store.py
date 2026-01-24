"""
Vector Store for Semantic Memory

High-performance vector storage with multiple similarity metrics,
approximate nearest neighbor search, and clustering support.
"""

import asyncio
import hashlib
import json
import math
import struct
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional
import heapq
import random

import structlog

logger = structlog.get_logger()


class SimilarityMetric(Enum):
    """Similarity metrics for vector comparison."""

    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"
    MANHATTAN = "manhattan"


@dataclass
class VectorEntry:
    """A vector entry in the store."""

    id: str
    vector: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)
    text: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    last_accessed: Optional[datetime] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "vector": self.vector,
            "metadata": self.metadata,
            "text": self.text,
            "timestamp": self.timestamp.isoformat(),
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VectorEntry":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            vector=data["vector"],
            metadata=data.get("metadata", {}),
            text=data.get("text", ""),
            timestamp=datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else datetime.now(),
            access_count=data.get("access_count", 0),
            last_accessed=datetime.fromisoformat(data["last_accessed"]) if data.get("last_accessed") else None,
        )


@dataclass
class SearchResult:
    """Result from a vector search."""

    entry: VectorEntry
    score: float
    rank: int


@dataclass
class HNSWNode:
    """Node in HNSW graph for approximate nearest neighbor search."""

    entry_id: str
    level: int
    neighbors: dict[int, list[str]] = field(default_factory=dict)  # level -> neighbor ids


class VectorStore:
    """
    High-performance vector store with HNSW indexing.

    Features:
    - Multiple similarity metrics
    - HNSW approximate nearest neighbor search
    - Metadata filtering
    - Automatic persistence
    - LRU-based memory management
    """

    def __init__(
        self,
        dimension: int = 1536,  # OpenAI embedding dimension
        metric: SimilarityMetric = SimilarityMetric.COSINE,
        max_elements: int = 100000,
        ef_construction: int = 200,  # HNSW construction parameter
        M: int = 16,  # HNSW max neighbors per node
        storage_path: Optional[Path] = None,
        embedding_fn: Optional[Callable[[str], list[float]]] = None,
    ):
        self.dimension = dimension
        self.metric = metric
        self.max_elements = max_elements
        self.ef_construction = ef_construction
        self.M = M
        self.storage_path = storage_path
        self.embedding_fn = embedding_fn

        # Storage
        self._entries: dict[str, VectorEntry] = {}
        self._hnsw_nodes: dict[str, HNSWNode] = {}
        self._entry_point: Optional[str] = None
        self._max_level: int = 0

        # Caching
        self._similarity_cache: dict[tuple[str, str], float] = {}
        self._cache_max_size = 10000

        self._initialized = False
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize the vector store."""
        if self._initialized:
            return

        if self.storage_path and self.storage_path.exists():
            await self._load_from_disk()

        self._initialized = True
        logger.info("vector_store_initialized", dimension=self.dimension, metric=self.metric.value)

    async def shutdown(self) -> None:
        """Shutdown and persist."""
        if self.storage_path:
            await self._save_to_disk()

        self._initialized = False
        logger.info("vector_store_shutdown")

    def _compute_similarity(self, v1: list[float], v2: list[float]) -> float:
        """Compute similarity between two vectors."""
        if len(v1) != len(v2):
            raise ValueError(f"Vector dimension mismatch: {len(v1)} vs {len(v2)}")

        if self.metric == SimilarityMetric.COSINE:
            dot = sum(a * b for a, b in zip(v1, v2))
            norm1 = math.sqrt(sum(a * a for a in v1))
            norm2 = math.sqrt(sum(b * b for b in v2))
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return dot / (norm1 * norm2)

        elif self.metric == SimilarityMetric.DOT_PRODUCT:
            return sum(a * b for a, b in zip(v1, v2))

        elif self.metric == SimilarityMetric.EUCLIDEAN:
            # Convert distance to similarity
            dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))
            return 1.0 / (1.0 + dist)

        elif self.metric == SimilarityMetric.MANHATTAN:
            dist = sum(abs(a - b) for a, b in zip(v1, v2))
            return 1.0 / (1.0 + dist)

        return 0.0

    def _get_random_level(self) -> int:
        """Generate random level for HNSW insertion."""
        ml = 1.0 / math.log(self.M)
        level = int(-math.log(random.random()) * ml)
        return min(level, 16)  # Cap at 16 levels

    async def add(
        self,
        text: str,
        vector: Optional[list[float]] = None,
        metadata: Optional[dict[str, Any]] = None,
        entry_id: Optional[str] = None,
    ) -> str:
        """Add a vector entry to the store."""
        async with self._lock:
            # Generate embedding if not provided
            if vector is None:
                if self.embedding_fn is None:
                    # Use simple hash-based embedding for testing
                    vector = self._simple_embedding(text)
                else:
                    vector = self.embedding_fn(text)

            if len(vector) != self.dimension:
                raise ValueError(f"Vector dimension {len(vector)} does not match store dimension {self.dimension}")

            # Generate ID if not provided
            if entry_id is None:
                entry_id = hashlib.sha256(f"{text}{datetime.now().isoformat()}".encode()).hexdigest()[:16]

            # Create entry
            entry = VectorEntry(
                id=entry_id,
                vector=vector,
                metadata=metadata or {},
                text=text,
            )

            # Check capacity
            if len(self._entries) >= self.max_elements:
                await self._evict_lru()

            # Store entry
            self._entries[entry_id] = entry

            # Add to HNSW index
            await self._hnsw_insert(entry)

            logger.debug("vector_added", entry_id=entry_id, text_length=len(text))

            return entry_id

    async def add_batch(
        self,
        texts: list[str],
        vectors: Optional[list[list[float]]] = None,
        metadata_list: Optional[list[dict[str, Any]]] = None,
    ) -> list[str]:
        """Add multiple entries in batch."""
        ids = []
        for i, text in enumerate(texts):
            vector = vectors[i] if vectors else None
            metadata = metadata_list[i] if metadata_list else None
            entry_id = await self.add(text, vector, metadata)
            ids.append(entry_id)
        return ids

    async def search(
        self,
        query: str | list[float],
        k: int = 10,
        filter_fn: Optional[Callable[[VectorEntry], bool]] = None,
        metadata_filter: Optional[dict[str, Any]] = None,
        ef_search: int = 50,
    ) -> list[SearchResult]:
        """Search for similar vectors."""
        # Get query vector
        if isinstance(query, str):
            if self.embedding_fn:
                query_vector = self.embedding_fn(query)
            else:
                query_vector = self._simple_embedding(query)
        else:
            query_vector = query

        # Use HNSW search
        candidates = await self._hnsw_search(query_vector, k * 2, ef_search)

        # Apply filters and compute final scores
        results = []
        for entry_id, score in candidates:
            entry = self._entries.get(entry_id)
            if entry is None:
                continue

            # Apply metadata filter
            if metadata_filter:
                match = all(
                    entry.metadata.get(key) == value
                    for key, value in metadata_filter.items()
                )
                if not match:
                    continue

            # Apply custom filter
            if filter_fn and not filter_fn(entry):
                continue

            # Update access stats
            entry.access_count += 1
            entry.last_accessed = datetime.now()

            results.append(SearchResult(entry=entry, score=score, rank=0))

        # Sort by score and assign ranks
        results.sort(key=lambda x: x.score, reverse=True)
        for i, result in enumerate(results[:k]):
            result.rank = i + 1

        return results[:k]

    async def _hnsw_insert(self, entry: VectorEntry) -> None:
        """Insert entry into HNSW index."""
        level = self._get_random_level()

        node = HNSWNode(
            entry_id=entry.id,
            level=level,
            neighbors={l: [] for l in range(level + 1)},
        )
        self._hnsw_nodes[entry.id] = node

        if self._entry_point is None:
            self._entry_point = entry.id
            self._max_level = level
            return

        # Find entry point for insertion
        current = self._entry_point

        # Traverse from top level
        for lc in range(self._max_level, level, -1):
            current = await self._greedy_search(entry.vector, current, lc)

        # Insert at each level
        for lc in range(min(level, self._max_level), -1, -1):
            neighbors = await self._search_layer(entry.vector, current, self.ef_construction, lc)

            # Select M best neighbors
            neighbors = neighbors[:self.M]
            node.neighbors[lc] = [n[0] for n in neighbors]

            # Add bidirectional connections
            for neighbor_id, _ in neighbors:
                neighbor_node = self._hnsw_nodes.get(neighbor_id)
                if neighbor_node and lc in neighbor_node.neighbors:
                    if len(neighbor_node.neighbors[lc]) < self.M:
                        neighbor_node.neighbors[lc].append(entry.id)
                    else:
                        # Prune weakest connection
                        await self._prune_connections(neighbor_node, lc, entry.id)

            if neighbors:
                current = neighbors[0][0]

        # Update entry point if needed
        if level > self._max_level:
            self._entry_point = entry.id
            self._max_level = level

    async def _greedy_search(self, query: list[float], start: str, level: int) -> str:
        """Greedy search at a single level."""
        current = start
        current_dist = self._compute_similarity(query, self._entries[current].vector)

        while True:
            node = self._hnsw_nodes.get(current)
            if not node or level not in node.neighbors:
                break

            changed = False
            for neighbor_id in node.neighbors.get(level, []):
                if neighbor_id not in self._entries:
                    continue
                dist = self._compute_similarity(query, self._entries[neighbor_id].vector)
                if dist > current_dist:
                    current = neighbor_id
                    current_dist = dist
                    changed = True

            if not changed:
                break

        return current

    async def _search_layer(
        self,
        query: list[float],
        entry_point: str,
        ef: int,
        level: int,
    ) -> list[tuple[str, float]]:
        """Search within a single HNSW layer."""
        visited = {entry_point}
        candidates = []
        results = []

        ep_dist = self._compute_similarity(query, self._entries[entry_point].vector)
        heapq.heappush(candidates, (-ep_dist, entry_point))
        heapq.heappush(results, (ep_dist, entry_point))

        while candidates:
            neg_dist, current = heapq.heappop(candidates)
            current_dist = -neg_dist

            # Check if we've found enough
            if results and current_dist < results[0][0]:
                break

            node = self._hnsw_nodes.get(current)
            if not node:
                continue

            for neighbor_id in node.neighbors.get(level, []):
                if neighbor_id in visited:
                    continue
                visited.add(neighbor_id)

                if neighbor_id not in self._entries:
                    continue

                dist = self._compute_similarity(query, self._entries[neighbor_id].vector)

                if len(results) < ef or dist > results[0][0]:
                    heapq.heappush(candidates, (-dist, neighbor_id))
                    heapq.heappush(results, (dist, neighbor_id))

                    if len(results) > ef:
                        heapq.heappop(results)

        # Return sorted by similarity (descending)
        return sorted([(id, dist) for dist, id in results], key=lambda x: x[1], reverse=True)

    async def _hnsw_search(
        self,
        query: list[float],
        k: int,
        ef: int,
    ) -> list[tuple[str, float]]:
        """Search the HNSW index."""
        if not self._entry_point:
            return []

        current = self._entry_point

        # Traverse from top level
        for lc in range(self._max_level, 0, -1):
            current = await self._greedy_search(query, current, lc)

        # Search at level 0
        results = await self._search_layer(query, current, max(ef, k), 0)

        return results[:k]

    async def _prune_connections(self, node: HNSWNode, level: int, new_id: str) -> None:
        """Prune connections to maintain M limit."""
        if new_id not in self._entries:
            return

        new_vector = self._entries[new_id].vector

        # Compute all distances
        candidates = []
        for neighbor_id in node.neighbors[level]:
            if neighbor_id in self._entries:
                dist = self._compute_similarity(new_vector, self._entries[neighbor_id].vector)
                candidates.append((neighbor_id, dist))

        candidates.append((new_id, 1.0))  # New connection

        # Sort and keep top M
        candidates.sort(key=lambda x: x[1], reverse=True)
        node.neighbors[level] = [c[0] for c in candidates[:self.M]]

    async def _evict_lru(self) -> None:
        """Evict least recently used entries."""
        if not self._entries:
            return

        # Find LRU entry
        lru_entry = min(
            self._entries.values(),
            key=lambda e: e.last_accessed or e.timestamp,
        )

        await self.delete(lru_entry.id)
        logger.debug("vector_evicted", entry_id=lru_entry.id)

    async def delete(self, entry_id: str) -> bool:
        """Delete an entry from the store."""
        async with self._lock:
            if entry_id not in self._entries:
                return False

            del self._entries[entry_id]

            # Remove from HNSW index
            if entry_id in self._hnsw_nodes:
                node = self._hnsw_nodes[entry_id]

                # Remove references from neighbors
                for level, neighbors in node.neighbors.items():
                    for neighbor_id in neighbors:
                        neighbor_node = self._hnsw_nodes.get(neighbor_id)
                        if neighbor_node and level in neighbor_node.neighbors:
                            neighbor_node.neighbors[level] = [
                                n for n in neighbor_node.neighbors[level] if n != entry_id
                            ]

                del self._hnsw_nodes[entry_id]

                # Update entry point if needed
                if self._entry_point == entry_id:
                    self._entry_point = next(iter(self._hnsw_nodes), None)
                    if self._entry_point:
                        self._max_level = self._hnsw_nodes[self._entry_point].level
                    else:
                        self._max_level = 0

            return True

    async def get(self, entry_id: str) -> Optional[VectorEntry]:
        """Get an entry by ID."""
        entry = self._entries.get(entry_id)
        if entry:
            entry.access_count += 1
            entry.last_accessed = datetime.now()
        return entry

    async def update_metadata(self, entry_id: str, metadata: dict[str, Any]) -> bool:
        """Update entry metadata."""
        entry = self._entries.get(entry_id)
        if not entry:
            return False
        entry.metadata.update(metadata)
        return True

    def _simple_embedding(self, text: str) -> list[float]:
        """Simple hash-based embedding for testing."""
        # Create a deterministic embedding from text
        vector = [0.0] * self.dimension

        # Use character-level features
        for i, char in enumerate(text):
            idx = (ord(char) + i) % self.dimension
            vector[idx] += 1.0

        # Normalize
        norm = math.sqrt(sum(v * v for v in vector))
        if norm > 0:
            vector = [v / norm for v in vector]

        return vector

    async def _save_to_disk(self) -> None:
        """Save store to disk."""
        if not self.storage_path:
            return

        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Save entries
        entries_data = {
            entry_id: entry.to_dict()
            for entry_id, entry in self._entries.items()
        }

        with open(self.storage_path / "entries.json", "w") as f:
            json.dump(entries_data, f)

        # Save HNSW index
        hnsw_data = {
            "entry_point": self._entry_point,
            "max_level": self._max_level,
            "nodes": {
                node_id: {
                    "entry_id": node.entry_id,
                    "level": node.level,
                    "neighbors": node.neighbors,
                }
                for node_id, node in self._hnsw_nodes.items()
            },
        }

        with open(self.storage_path / "hnsw.json", "w") as f:
            json.dump(hnsw_data, f)

        logger.info("vector_store_saved", path=str(self.storage_path), entries=len(self._entries))

    async def _load_from_disk(self) -> None:
        """Load store from disk."""
        if not self.storage_path:
            return

        entries_path = self.storage_path / "entries.json"
        hnsw_path = self.storage_path / "hnsw.json"

        if entries_path.exists():
            with open(entries_path) as f:
                entries_data = json.load(f)
            self._entries = {
                entry_id: VectorEntry.from_dict(data)
                for entry_id, data in entries_data.items()
            }

        if hnsw_path.exists():
            with open(hnsw_path) as f:
                hnsw_data = json.load(f)

            self._entry_point = hnsw_data.get("entry_point")
            self._max_level = hnsw_data.get("max_level", 0)

            for node_id, node_data in hnsw_data.get("nodes", {}).items():
                self._hnsw_nodes[node_id] = HNSWNode(
                    entry_id=node_data["entry_id"],
                    level=node_data["level"],
                    neighbors={int(k): v for k, v in node_data["neighbors"].items()},
                )

        logger.info("vector_store_loaded", path=str(self.storage_path), entries=len(self._entries))

    def get_stats(self) -> dict[str, Any]:
        """Get store statistics."""
        return {
            "total_entries": len(self._entries),
            "max_elements": self.max_elements,
            "dimension": self.dimension,
            "metric": self.metric.value,
            "hnsw_max_level": self._max_level,
            "hnsw_nodes": len(self._hnsw_nodes),
            "cache_size": len(self._similarity_cache),
        }
