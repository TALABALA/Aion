"""
AION Vector Index

FAISS-based vector index with:
- Multiple index types (Flat, IVF, HNSW)
- Efficient similarity search
- Persistence support
- Metadata filtering
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class IndexType(str, Enum):
    """Types of vector indices."""
    FLAT = "flat"  # Exact search, O(n)
    IVF = "ivf"    # Inverted file index, faster for large datasets
    HNSW = "hnsw"  # Hierarchical navigable small world, fastest approx


@dataclass
class SearchResult:
    """A single search result."""
    id: str
    score: float
    metadata: dict[str, Any]
    vector: Optional[np.ndarray] = None


@dataclass
class VectorEntry:
    """A vector entry in the index."""
    id: str
    vector: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    last_accessed: Optional[datetime] = None


class VectorIndex:
    """
    FAISS-based vector index with metadata support.

    Features:
    - Multiple index types for different scale/accuracy tradeoffs
    - Metadata filtering
    - Persistence to disk
    - Automatic index optimization
    """

    def __init__(
        self,
        dimension: int,
        index_type: IndexType = IndexType.FLAT,
        nlist: int = 100,  # For IVF
        nprobe: int = 10,  # For IVF search
        ef_construction: int = 200,  # For HNSW
        ef_search: int = 50,  # For HNSW
        max_size: int = 1_000_000,
    ):
        self.dimension = dimension
        self.index_type = index_type
        self.nlist = nlist
        self.nprobe = nprobe
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.max_size = max_size

        self._index = None
        self._id_to_idx: dict[str, int] = {}
        self._idx_to_id: dict[int, str] = {}
        self._metadata: dict[str, dict[str, Any]] = {}
        self._vectors: list[np.ndarray] = []
        self._next_idx = 0
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the index."""
        if self._initialized:
            return

        logger.info(
            "Initializing vector index",
            type=self.index_type.value,
            dimension=self.dimension,
        )

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._create_index)

        self._initialized = True

    def _create_index(self) -> None:
        """Create the FAISS index (blocking)."""
        try:
            import faiss

            if self.index_type == IndexType.FLAT:
                self._index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine
            elif self.index_type == IndexType.IVF:
                quantizer = faiss.IndexFlatIP(self.dimension)
                self._index = faiss.IndexIVFFlat(
                    quantizer, self.dimension, self.nlist, faiss.METRIC_INNER_PRODUCT
                )
            elif self.index_type == IndexType.HNSW:
                self._index = faiss.IndexHNSWFlat(self.dimension, 32, faiss.METRIC_INNER_PRODUCT)
                self._index.hnsw.efConstruction = self.ef_construction
                self._index.hnsw.efSearch = self.ef_search
            else:
                self._index = faiss.IndexFlatIP(self.dimension)

            logger.info("FAISS index created", type=self.index_type.value)

        except ImportError:
            logger.warning("FAISS not available, using numpy fallback")
            self._index = None

    async def add(
        self,
        vector_id: str,
        vector: np.ndarray,
        metadata: Optional[dict[str, Any]] = None,
    ) -> bool:
        """
        Add a vector to the index.

        Args:
            vector_id: Unique identifier
            vector: Vector to add
            metadata: Optional metadata

        Returns:
            True if added successfully
        """
        if not self._initialized:
            await self.initialize()

        # Check size limit
        if len(self._id_to_idx) >= self.max_size:
            logger.warning("Index at maximum capacity", max_size=self.max_size)
            return False

        # Handle duplicates
        if vector_id in self._id_to_idx:
            await self.remove(vector_id)

        # Ensure correct shape
        vector = np.array(vector, dtype=np.float32)
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)

        # Add to index
        idx = self._next_idx
        self._next_idx += 1

        self._id_to_idx[vector_id] = idx
        self._idx_to_id[idx] = vector_id
        self._metadata[vector_id] = metadata or {}
        self._vectors.append(vector.flatten())

        if self._index is not None:
            import faiss
            self._index.add(vector)

        return True

    async def add_batch(
        self,
        vectors: list[tuple[str, np.ndarray, Optional[dict[str, Any]]]],
    ) -> int:
        """
        Add multiple vectors in batch.

        Args:
            vectors: List of (id, vector, metadata) tuples

        Returns:
            Number of vectors added
        """
        if not self._initialized:
            await self.initialize()

        added = 0
        for vector_id, vector, metadata in vectors:
            if await self.add(vector_id, vector, metadata):
                added += 1

        return added

    async def remove(self, vector_id: str) -> bool:
        """
        Remove a vector from the index.

        Note: FAISS doesn't support efficient removal, so this marks for removal
        and a rebuild may be needed.

        Args:
            vector_id: ID to remove

        Returns:
            True if removed
        """
        if vector_id not in self._id_to_idx:
            return False

        idx = self._id_to_idx.pop(vector_id)
        self._idx_to_id.pop(idx, None)
        self._metadata.pop(vector_id, None)

        # Note: FAISS removal requires index rebuild for efficiency
        return True

    async def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        filter_fn: Optional[callable] = None,
    ) -> list[SearchResult]:
        """
        Search for similar vectors.

        Args:
            query_vector: Query vector
            k: Number of results
            filter_fn: Optional filter function for metadata

        Returns:
            List of SearchResult
        """
        if not self._initialized:
            await self.initialize()

        if not self._id_to_idx:
            return []

        query_vector = np.array(query_vector, dtype=np.float32)
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            lambda: self._search_sync(query_vector, k, filter_fn),
        )

        return results

    def _search_sync(
        self,
        query_vector: np.ndarray,
        k: int,
        filter_fn: Optional[callable] = None,
    ) -> list[SearchResult]:
        """Synchronous search implementation."""
        if self._index is not None:
            # Use FAISS search
            try:
                import faiss

                # Set search parameters
                if self.index_type == IndexType.IVF and hasattr(self._index, 'nprobe'):
                    self._index.nprobe = self.nprobe

                # Search for more results if filtering
                search_k = k * 3 if filter_fn else k
                search_k = min(search_k, len(self._id_to_idx))

                if search_k == 0:
                    return []

                scores, indices = self._index.search(query_vector, search_k)

                results = []
                for score, idx in zip(scores[0], indices[0]):
                    if idx < 0:  # Invalid index
                        continue

                    vector_id = self._idx_to_id.get(idx)
                    if vector_id is None:
                        continue

                    metadata = self._metadata.get(vector_id, {})

                    # Apply filter
                    if filter_fn and not filter_fn(metadata):
                        continue

                    results.append(SearchResult(
                        id=vector_id,
                        score=float(score),
                        metadata=metadata,
                    ))

                    if len(results) >= k:
                        break

                return results

            except Exception as e:
                logger.warning(f"FAISS search failed: {e}, falling back to numpy")

        # Numpy fallback
        if not self._vectors:
            return []

        vectors_matrix = np.vstack(self._vectors)
        scores = np.dot(vectors_matrix, query_vector.flatten())

        # Sort by score (descending)
        indices = np.argsort(-scores)

        results = []
        for idx in indices:
            vector_id = self._idx_to_id.get(idx)
            if vector_id is None:
                continue

            metadata = self._metadata.get(vector_id, {})

            if filter_fn and not filter_fn(metadata):
                continue

            results.append(SearchResult(
                id=vector_id,
                score=float(scores[idx]),
                metadata=metadata,
            ))

            if len(results) >= k:
                break

        return results

    async def get(self, vector_id: str) -> Optional[VectorEntry]:
        """
        Get a vector by ID.

        Args:
            vector_id: Vector ID

        Returns:
            VectorEntry if found
        """
        if vector_id not in self._id_to_idx:
            return None

        idx = self._id_to_idx[vector_id]
        vector = self._vectors[idx] if idx < len(self._vectors) else None

        return VectorEntry(
            id=vector_id,
            vector=vector,
            metadata=self._metadata.get(vector_id, {}),
        )

    def count(self) -> int:
        """Get number of vectors in index."""
        return len(self._id_to_idx)

    async def save(self, path: Union[str, Path]) -> None:
        """
        Save index to disk.

        Args:
            path: Directory path to save to
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        if self._index is not None:
            try:
                import faiss
                faiss.write_index(self._index, str(path / "index.faiss"))
            except Exception as e:
                logger.warning(f"Failed to save FAISS index: {e}")

        # Save metadata
        data = {
            "dimension": self.dimension,
            "index_type": self.index_type.value,
            "id_to_idx": self._id_to_idx,
            "idx_to_id": {str(k): v for k, v in self._idx_to_id.items()},
            "metadata": self._metadata,
            "next_idx": self._next_idx,
        }

        with open(path / "metadata.json", "w") as f:
            json.dump(data, f)

        # Save vectors
        if self._vectors:
            vectors_array = np.vstack(self._vectors)
            np.save(path / "vectors.npy", vectors_array)

        logger.info("Index saved", path=str(path), count=len(self._id_to_idx))

    async def load(self, path: Union[str, Path]) -> None:
        """
        Load index from disk.

        Args:
            path: Directory path to load from
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Index path not found: {path}")

        # Load metadata
        with open(path / "metadata.json") as f:
            data = json.load(f)

        self.dimension = data["dimension"]
        self.index_type = IndexType(data["index_type"])
        self._id_to_idx = data["id_to_idx"]
        self._idx_to_id = {int(k): v for k, v in data["idx_to_id"].items()}
        self._metadata = data["metadata"]
        self._next_idx = data["next_idx"]

        # Load vectors
        vectors_path = path / "vectors.npy"
        if vectors_path.exists():
            vectors_array = np.load(vectors_path)
            self._vectors = [v for v in vectors_array]

        # Load FAISS index
        faiss_path = path / "index.faiss"
        if faiss_path.exists():
            try:
                import faiss
                self._index = faiss.read_index(str(faiss_path))
            except Exception as e:
                logger.warning(f"Failed to load FAISS index: {e}")
                await self.initialize()
                # Rebuild index
                if self._vectors:
                    vectors_array = np.vstack(self._vectors).astype(np.float32)
                    self._index.add(vectors_array)

        self._initialized = True
        logger.info("Index loaded", path=str(path), count=len(self._id_to_idx))

    async def shutdown(self) -> None:
        """Shutdown the index."""
        self._index = None
        self._initialized = False
