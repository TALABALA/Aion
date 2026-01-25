"""
AION Vector Store

True SOTA vector database support with:
- pgvector integration for PostgreSQL
- FAISS for high-performance local search
- Hybrid search (vector + keyword)
- Automatic index optimization
- Quantization for memory efficiency
- Multi-index support (HNSW, IVF, Flat)
- Metadata filtering
- Batch operations
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import struct
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Protocol, Sequence
import numpy as np

logger = logging.getLogger(__name__)


class IndexType(str, Enum):
    """Vector index types."""
    FLAT = "flat"           # Exact search, slow but accurate
    HNSW = "hnsw"           # Hierarchical Navigable Small World, fast approximate
    IVF = "ivf"             # Inverted File Index, good for large datasets
    IVF_PQ = "ivf_pq"       # IVF with Product Quantization, memory efficient
    SCALAR_QUANTIZER = "sq" # Scalar quantization


class DistanceMetric(str, Enum):
    """Distance metrics for vector similarity."""
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"
    MANHATTAN = "manhattan"


@dataclass
class VectorDocument:
    """A document with its vector embedding."""
    id: str
    embedding: np.ndarray
    content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        if isinstance(self.embedding, list):
            self.embedding = np.array(self.embedding, dtype=np.float32)


@dataclass
class SearchResult:
    """Result of a vector search."""
    id: str
    score: float
    content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    distance: float = 0.0


@dataclass
class VectorIndexConfig:
    """Configuration for vector index."""
    dimension: int
    index_type: IndexType = IndexType.HNSW
    metric: DistanceMetric = DistanceMetric.COSINE
    # HNSW parameters
    hnsw_m: int = 16                    # Number of connections per layer
    hnsw_ef_construction: int = 200     # Size of dynamic list during construction
    hnsw_ef_search: int = 50            # Size of dynamic list during search
    # IVF parameters
    ivf_nlist: int = 100                # Number of clusters
    ivf_nprobe: int = 10                # Number of clusters to search
    # Quantization
    pq_m: int = 8                       # Number of sub-quantizers
    pq_nbits: int = 8                   # Bits per sub-quantizer


class VectorStore(ABC):
    """Abstract base class for vector stores."""

    @abstractmethod
    async def create_index(self, name: str, config: VectorIndexConfig) -> bool:
        """Create a new vector index."""
        pass

    @abstractmethod
    async def drop_index(self, name: str) -> bool:
        """Drop an existing index."""
        pass

    @abstractmethod
    async def insert(self, index_name: str, documents: Sequence[VectorDocument]) -> int:
        """Insert documents into the index."""
        pass

    @abstractmethod
    async def search(
        self,
        index_name: str,
        query_vector: np.ndarray,
        k: int = 10,
        filter: Optional[dict[str, Any]] = None,
    ) -> list[SearchResult]:
        """Search for similar vectors."""
        pass

    @abstractmethod
    async def delete(self, index_name: str, ids: Sequence[str]) -> int:
        """Delete documents by ID."""
        pass

    @abstractmethod
    async def get(self, index_name: str, id: str) -> Optional[VectorDocument]:
        """Get a document by ID."""
        pass


# ==================== FAISS Implementation ====================

class FAISSVectorStore(VectorStore):
    """
    FAISS-based vector store for high-performance local search.

    Features:
    - Multiple index types (Flat, HNSW, IVF)
    - GPU acceleration support
    - Quantization for memory efficiency
    - Metadata filtering
    - Persistence to disk
    """

    def __init__(
        self,
        base_path: Optional[Path] = None,
        use_gpu: bool = False,
    ):
        self.base_path = base_path or Path("./vector_indices")
        self.use_gpu = use_gpu
        self._indices: dict[str, Any] = {}
        self._id_maps: dict[str, dict[str, int]] = {}  # id -> internal index
        self._reverse_id_maps: dict[str, dict[int, str]] = {}  # internal index -> id
        self._metadata: dict[str, dict[str, dict]] = {}  # index -> id -> metadata
        self._configs: dict[str, VectorIndexConfig] = {}
        self._faiss = None

    def _ensure_faiss(self):
        """Ensure FAISS is available."""
        if self._faiss is None:
            try:
                import faiss
                self._faiss = faiss
            except ImportError:
                raise RuntimeError("FAISS is required for FAISSVectorStore")
        return self._faiss

    async def create_index(self, name: str, config: VectorIndexConfig) -> bool:
        """Create a new FAISS index."""
        faiss = self._ensure_faiss()

        if name in self._indices:
            logger.warning(f"Index {name} already exists")
            return False

        # Create the appropriate index type
        if config.index_type == IndexType.FLAT:
            if config.metric == DistanceMetric.COSINE:
                index = faiss.IndexFlatIP(config.dimension)  # Inner product after normalization
            elif config.metric == DistanceMetric.EUCLIDEAN:
                index = faiss.IndexFlatL2(config.dimension)
            else:
                index = faiss.IndexFlatIP(config.dimension)

        elif config.index_type == IndexType.HNSW:
            index = faiss.IndexHNSWFlat(config.dimension, config.hnsw_m)
            index.hnsw.efConstruction = config.hnsw_ef_construction
            index.hnsw.efSearch = config.hnsw_ef_search

        elif config.index_type == IndexType.IVF:
            quantizer = faiss.IndexFlatL2(config.dimension)
            index = faiss.IndexIVFFlat(quantizer, config.dimension, config.ivf_nlist)
            index.nprobe = config.ivf_nprobe

        elif config.index_type == IndexType.IVF_PQ:
            quantizer = faiss.IndexFlatL2(config.dimension)
            index = faiss.IndexIVFPQ(
                quantizer, config.dimension, config.ivf_nlist,
                config.pq_m, config.pq_nbits
            )
            index.nprobe = config.ivf_nprobe

        else:
            raise ValueError(f"Unsupported index type: {config.index_type}")

        # Move to GPU if requested and available
        if self.use_gpu:
            try:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
            except Exception as e:
                logger.warning(f"GPU not available, using CPU: {e}")

        self._indices[name] = index
        self._id_maps[name] = {}
        self._reverse_id_maps[name] = {}
        self._metadata[name] = {}
        self._configs[name] = config

        logger.info(f"Created FAISS index: {name} ({config.index_type.value})")
        return True

    async def drop_index(self, name: str) -> bool:
        """Drop a FAISS index."""
        if name not in self._indices:
            return False

        del self._indices[name]
        del self._id_maps[name]
        del self._reverse_id_maps[name]
        del self._metadata[name]
        del self._configs[name]

        # Remove persisted files
        index_path = self.base_path / f"{name}.faiss"
        if index_path.exists():
            index_path.unlink()

        return True

    async def insert(self, index_name: str, documents: Sequence[VectorDocument]) -> int:
        """Insert documents into FAISS index."""
        faiss = self._ensure_faiss()

        if index_name not in self._indices:
            raise ValueError(f"Index {index_name} does not exist")

        index = self._indices[index_name]
        config = self._configs[index_name]
        id_map = self._id_maps[index_name]
        reverse_map = self._reverse_id_maps[index_name]
        metadata_store = self._metadata[index_name]

        # Prepare vectors
        vectors = np.stack([doc.embedding for doc in documents]).astype(np.float32)

        # Normalize for cosine similarity
        if config.metric == DistanceMetric.COSINE:
            faiss.normalize_L2(vectors)

        # Train IVF indices if needed
        if config.index_type in (IndexType.IVF, IndexType.IVF_PQ) and not index.is_trained:
            index.train(vectors)

        # Get starting index
        start_idx = len(id_map)

        # Add vectors
        index.add(vectors)

        # Update mappings
        for i, doc in enumerate(documents):
            idx = start_idx + i
            id_map[doc.id] = idx
            reverse_map[idx] = doc.id
            metadata_store[doc.id] = {
                "content": doc.content,
                "metadata": doc.metadata,
                "created_at": doc.created_at.isoformat(),
            }

        return len(documents)

    async def search(
        self,
        index_name: str,
        query_vector: np.ndarray,
        k: int = 10,
        filter: Optional[dict[str, Any]] = None,
    ) -> list[SearchResult]:
        """Search for similar vectors."""
        faiss = self._ensure_faiss()

        if index_name not in self._indices:
            raise ValueError(f"Index {index_name} does not exist")

        index = self._indices[index_name]
        config = self._configs[index_name]
        reverse_map = self._reverse_id_maps[index_name]
        metadata_store = self._metadata[index_name]

        # Prepare query
        query = query_vector.astype(np.float32).reshape(1, -1)

        # Normalize for cosine similarity
        if config.metric == DistanceMetric.COSINE:
            faiss.normalize_L2(query)

        # Search with extra results for filtering
        search_k = k * 3 if filter else k
        distances, indices = index.search(query, search_k)

        # Build results with filtering
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:  # No result
                continue

            doc_id = reverse_map.get(idx)
            if not doc_id:
                continue

            meta = metadata_store.get(doc_id, {})

            # Apply filter
            if filter:
                doc_meta = meta.get("metadata", {})
                if not self._matches_filter(doc_meta, filter):
                    continue

            # Convert distance to score
            if config.metric == DistanceMetric.COSINE:
                score = float(dist)  # Inner product is already similarity
            else:
                score = 1.0 / (1.0 + float(dist))  # Convert distance to similarity

            results.append(SearchResult(
                id=doc_id,
                score=score,
                distance=float(dist),
                content=meta.get("content", ""),
                metadata=meta.get("metadata", {}),
            ))

            if len(results) >= k:
                break

        return results

    def _matches_filter(self, metadata: dict, filter: dict) -> bool:
        """Check if metadata matches filter."""
        for key, value in filter.items():
            if key not in metadata:
                return False

            if isinstance(value, dict):
                # Operator-based filter
                for op, val in value.items():
                    meta_val = metadata[key]
                    if op == "$eq" and meta_val != val:
                        return False
                    elif op == "$ne" and meta_val == val:
                        return False
                    elif op == "$gt" and not meta_val > val:
                        return False
                    elif op == "$gte" and not meta_val >= val:
                        return False
                    elif op == "$lt" and not meta_val < val:
                        return False
                    elif op == "$lte" and not meta_val <= val:
                        return False
                    elif op == "$in" and meta_val not in val:
                        return False
            else:
                # Direct equality
                if metadata[key] != value:
                    return False

        return True

    async def delete(self, index_name: str, ids: Sequence[str]) -> int:
        """Delete documents by ID (marks as deleted, requires rebuild for FAISS)."""
        if index_name not in self._indices:
            raise ValueError(f"Index {index_name} does not exist")

        deleted = 0
        metadata_store = self._metadata[index_name]

        for doc_id in ids:
            if doc_id in metadata_store:
                del metadata_store[doc_id]
                deleted += 1

        # Note: FAISS doesn't support true deletion, would need rebuild
        logger.warning("FAISS deletion is lazy - rebuild required for actual removal")

        return deleted

    async def get(self, index_name: str, id: str) -> Optional[VectorDocument]:
        """Get a document by ID."""
        if index_name not in self._indices:
            raise ValueError(f"Index {index_name} does not exist")

        id_map = self._id_maps[index_name]
        metadata_store = self._metadata[index_name]

        if id not in id_map:
            return None

        idx = id_map[id]
        meta = metadata_store.get(id, {})

        # Reconstruct vector from index
        index = self._indices[index_name]
        vector = index.reconstruct(idx)

        return VectorDocument(
            id=id,
            embedding=vector,
            content=meta.get("content", ""),
            metadata=meta.get("metadata", {}),
            created_at=datetime.fromisoformat(meta.get("created_at", datetime.utcnow().isoformat())),
        )

    async def save(self, index_name: str) -> bool:
        """Save index to disk."""
        faiss = self._ensure_faiss()

        if index_name not in self._indices:
            return False

        self.base_path.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        index_path = self.base_path / f"{index_name}.faiss"
        faiss.write_index(self._indices[index_name], str(index_path))

        # Save metadata
        meta_path = self.base_path / f"{index_name}.meta.json"
        with open(meta_path, 'w') as f:
            json.dump({
                "id_map": self._id_maps[index_name],
                "reverse_map": {str(k): v for k, v in self._reverse_id_maps[index_name].items()},
                "metadata": self._metadata[index_name],
                "config": {
                    "dimension": self._configs[index_name].dimension,
                    "index_type": self._configs[index_name].index_type.value,
                    "metric": self._configs[index_name].metric.value,
                },
            }, f)

        return True

    async def load(self, index_name: str) -> bool:
        """Load index from disk."""
        faiss = self._ensure_faiss()

        index_path = self.base_path / f"{index_name}.faiss"
        meta_path = self.base_path / f"{index_name}.meta.json"

        if not index_path.exists() or not meta_path.exists():
            return False

        # Load FAISS index
        self._indices[index_name] = faiss.read_index(str(index_path))

        # Load metadata
        with open(meta_path, 'r') as f:
            data = json.load(f)
            self._id_maps[index_name] = data["id_map"]
            self._reverse_id_maps[index_name] = {int(k): v for k, v in data["reverse_map"].items()}
            self._metadata[index_name] = data["metadata"]
            self._configs[index_name] = VectorIndexConfig(
                dimension=data["config"]["dimension"],
                index_type=IndexType(data["config"]["index_type"]),
                metric=DistanceMetric(data["config"]["metric"]),
            )

        return True


# ==================== pgvector Implementation ====================

class PgVectorStore(VectorStore):
    """
    PostgreSQL pgvector-based vector store.

    Features:
    - Native PostgreSQL integration
    - HNSW and IVFFlat indexes
    - SQL-based filtering
    - ACID transactions
    - Hybrid search with full-text
    """

    def __init__(self, connection: Any):
        self.connection = connection
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize pgvector extension."""
        await self.connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
        self._initialized = True

    async def create_index(self, name: str, config: VectorIndexConfig) -> bool:
        """Create a new pgvector index."""
        # Create table
        await self.connection.execute(f"""
            CREATE TABLE IF NOT EXISTS vector_{name} (
                id TEXT PRIMARY KEY,
                embedding vector({config.dimension}),
                content TEXT,
                metadata JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create vector index
        if config.index_type == IndexType.HNSW:
            op_class = self._get_op_class(config.metric)
            await self.connection.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{name}_embedding
                ON vector_{name}
                USING hnsw (embedding {op_class})
                WITH (m = {config.hnsw_m}, ef_construction = {config.hnsw_ef_construction})
            """)
        elif config.index_type == IndexType.IVF:
            op_class = self._get_op_class(config.metric)
            await self.connection.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{name}_embedding
                ON vector_{name}
                USING ivfflat (embedding {op_class})
                WITH (lists = {config.ivf_nlist})
            """)

        # Create GIN index for metadata
        await self.connection.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{name}_metadata
            ON vector_{name} USING GIN (metadata)
        """)

        return True

    def _get_op_class(self, metric: DistanceMetric) -> str:
        """Get pgvector operator class for distance metric."""
        if metric == DistanceMetric.COSINE:
            return "vector_cosine_ops"
        elif metric == DistanceMetric.EUCLIDEAN:
            return "vector_l2_ops"
        elif metric == DistanceMetric.DOT_PRODUCT:
            return "vector_ip_ops"
        return "vector_cosine_ops"

    def _get_distance_op(self, metric: DistanceMetric) -> str:
        """Get pgvector distance operator."""
        if metric == DistanceMetric.COSINE:
            return "<=>"  # Cosine distance
        elif metric == DistanceMetric.EUCLIDEAN:
            return "<->"  # L2 distance
        elif metric == DistanceMetric.DOT_PRODUCT:
            return "<#>"  # Negative inner product
        return "<=>"

    async def drop_index(self, name: str) -> bool:
        """Drop a pgvector table."""
        await self.connection.execute(f"DROP TABLE IF EXISTS vector_{name}")
        return True

    async def insert(self, index_name: str, documents: Sequence[VectorDocument]) -> int:
        """Insert documents into pgvector."""
        for doc in documents:
            await self.connection.execute(
                f"""
                INSERT INTO vector_{index_name} (id, embedding, content, metadata, created_at)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (id) DO UPDATE SET
                    embedding = EXCLUDED.embedding,
                    content = EXCLUDED.content,
                    metadata = EXCLUDED.metadata
                """,
                (
                    doc.id,
                    doc.embedding.tolist(),
                    doc.content,
                    json.dumps(doc.metadata),
                    doc.created_at,
                ),
            )
        return len(documents)

    async def search(
        self,
        index_name: str,
        query_vector: np.ndarray,
        k: int = 10,
        filter: Optional[dict[str, Any]] = None,
        metric: DistanceMetric = DistanceMetric.COSINE,
    ) -> list[SearchResult]:
        """Search for similar vectors using pgvector."""
        op = self._get_distance_op(metric)

        # Build filter clause
        filter_clause = ""
        params = [query_vector.tolist(), k]

        if filter:
            conditions = []
            for key, value in filter.items():
                param_idx = len(params) + 1
                if isinstance(value, dict):
                    for op_name, val in value.items():
                        sql_op = {"$eq": "=", "$ne": "!=", "$gt": ">", "$gte": ">=", "$lt": "<", "$lte": "<="}.get(op_name, "=")
                        conditions.append(f"metadata->>'{key}' {sql_op} ${param_idx}")
                        params.append(str(val))
                else:
                    conditions.append(f"metadata->>'{key}' = ${param_idx}")
                    params.append(str(value))

            if conditions:
                filter_clause = "WHERE " + " AND ".join(conditions)

        query = f"""
            SELECT id, content, metadata,
                   embedding {op} $1 as distance
            FROM vector_{index_name}
            {filter_clause}
            ORDER BY embedding {op} $1
            LIMIT $2
        """

        rows = await self.connection.fetch_all(query, tuple(params))

        results = []
        for row in rows:
            distance = float(row["distance"])
            # Convert distance to score
            if metric == DistanceMetric.COSINE:
                score = 1.0 - distance  # Cosine similarity
            else:
                score = 1.0 / (1.0 + distance)

            results.append(SearchResult(
                id=row["id"],
                score=score,
                distance=distance,
                content=row["content"] or "",
                metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            ))

        return results

    async def delete(self, index_name: str, ids: Sequence[str]) -> int:
        """Delete documents by ID."""
        placeholders = ", ".join(f"${i+1}" for i in range(len(ids)))
        result = await self.connection.execute(
            f"DELETE FROM vector_{index_name} WHERE id IN ({placeholders})",
            tuple(ids),
        )
        return len(ids)

    async def get(self, index_name: str, id: str) -> Optional[VectorDocument]:
        """Get a document by ID."""
        row = await self.connection.fetch_one(
            f"SELECT * FROM vector_{index_name} WHERE id = $1",
            (id,),
        )

        if not row:
            return None

        return VectorDocument(
            id=row["id"],
            embedding=np.array(row["embedding"], dtype=np.float32),
            content=row["content"] or "",
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            created_at=row["created_at"],
        )

    async def hybrid_search(
        self,
        index_name: str,
        query_vector: np.ndarray,
        query_text: str,
        k: int = 10,
        vector_weight: float = 0.7,
        text_weight: float = 0.3,
    ) -> list[SearchResult]:
        """
        Perform hybrid search combining vector similarity and full-text search.

        Args:
            index_name: Name of the index
            query_vector: Query embedding
            query_text: Query text for full-text search
            k: Number of results
            vector_weight: Weight for vector similarity (0-1)
            text_weight: Weight for text similarity (0-1)
        """
        query = f"""
            WITH vector_results AS (
                SELECT id, content, metadata,
                       1 - (embedding <=> $1) as vector_score
                FROM vector_{index_name}
                ORDER BY embedding <=> $1
                LIMIT $3
            ),
            text_results AS (
                SELECT id, content, metadata,
                       ts_rank(to_tsvector('english', content), plainto_tsquery('english', $2)) as text_score
                FROM vector_{index_name}
                WHERE to_tsvector('english', content) @@ plainto_tsquery('english', $2)
                LIMIT $3
            )
            SELECT
                COALESCE(v.id, t.id) as id,
                COALESCE(v.content, t.content) as content,
                COALESCE(v.metadata, t.metadata) as metadata,
                COALESCE(v.vector_score, 0) * $4 + COALESCE(t.text_score, 0) * $5 as combined_score
            FROM vector_results v
            FULL OUTER JOIN text_results t ON v.id = t.id
            ORDER BY combined_score DESC
            LIMIT $3
        """

        rows = await self.connection.fetch_all(
            query,
            (query_vector.tolist(), query_text, k, vector_weight, text_weight),
        )

        return [
            SearchResult(
                id=row["id"],
                score=float(row["combined_score"]),
                content=row["content"] or "",
                metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            )
            for row in rows
        ]


# ==================== Vector Store Factory ====================

class VectorStoreFactory:
    """Factory for creating vector stores."""

    @staticmethod
    def create(
        backend: str = "faiss",
        **kwargs,
    ) -> VectorStore:
        """
        Create a vector store.

        Args:
            backend: "faiss" or "pgvector"
            **kwargs: Backend-specific arguments

        Returns:
            VectorStore instance
        """
        if backend == "faiss":
            return FAISSVectorStore(**kwargs)
        elif backend == "pgvector":
            if "connection" not in kwargs:
                raise ValueError("pgvector requires a connection")
            return PgVectorStore(**kwargs)
        else:
            raise ValueError(f"Unknown vector store backend: {backend}")
