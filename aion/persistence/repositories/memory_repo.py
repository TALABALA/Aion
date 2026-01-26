"""
AION Memory Repository

Persistence for the cognitive memory system including:
- Memory entries with full text search
- Embeddings storage with compression
- Memory relations (graph structure)
- FAISS index snapshots
"""

from __future__ import annotations

import io
import zlib
from datetime import datetime
from typing import Any, Optional
import uuid

import numpy as np
import structlog

from aion.persistence.repositories.base import BaseRepository, QueryOptions
from aion.persistence.database import DatabaseManager
from aion.persistence.backends.redis_cache import CacheManager

logger = structlog.get_logger(__name__)

# Import Memory type from cognitive system
try:
    from aion.systems.memory.cognitive import Memory, MemoryType
except ImportError:
    # Fallback for when memory system isn't available
    from enum import Enum

    class MemoryType(str, Enum):
        EPISODIC = "episodic"
        SEMANTIC = "semantic"
        PROCEDURAL = "procedural"
        WORKING = "working"

    from dataclasses import dataclass, field

    @dataclass
    class Memory:
        id: str
        content: str
        memory_type: MemoryType
        embedding: Optional[np.ndarray] = None
        metadata: dict[str, Any] = field(default_factory=dict)
        importance: float = 0.5
        access_count: int = 0
        created_at: datetime = field(default_factory=datetime.now)
        last_accessed: Optional[datetime] = None
        decay_rate: float = 0.01
        linked_memories: list[str] = field(default_factory=list)


class MemoryRepository(BaseRepository[Memory]):
    """
    Repository for memory persistence.

    Features:
    - Full text search on content
    - Embedding storage with compression
    - Memory relations as graph edges
    - FAISS index persistence
    """

    def __init__(
        self,
        db: DatabaseManager,
        cache: Optional[CacheManager] = None,
        compress_embeddings: bool = True,
    ):
        super().__init__(db, cache)
        self._table_name = "memories"
        self._soft_delete_column = "deleted_at"
        self._compress_embeddings = compress_embeddings

    def _serialize(self, memory: Memory) -> dict[str, Any]:
        """Serialize Memory to database row."""
        return {
            "id": memory.id,
            "content": memory.content,
            "memory_type": memory.memory_type.value if isinstance(memory.memory_type, MemoryType) else memory.memory_type,
            "importance": memory.importance,
            "decay_rate": memory.decay_rate,
            "access_count": memory.access_count,
            "created_at": self._from_datetime(memory.created_at),
            "last_accessed_at": self._from_datetime(memory.last_accessed) if hasattr(memory, 'last_accessed') else None,
            "metadata": self._to_json(memory.metadata),
            "tags": self._to_json(memory.metadata.get("tags", [])),
            "source_id": memory.metadata.get("source_id"),
        }

    def _deserialize(self, row: dict[str, Any]) -> Memory:
        """Deserialize database row to Memory."""
        metadata = self._from_json(row.get("metadata")) or {}

        return Memory(
            id=row["id"],
            content=row["content"],
            memory_type=MemoryType(row["memory_type"]),
            importance=row.get("importance", 0.5),
            decay_rate=row.get("decay_rate", 0.01),
            access_count=row.get("access_count", 0),
            created_at=self._to_datetime(row.get("created_at")) or datetime.now(),
            last_accessed=self._to_datetime(row.get("last_accessed_at")),
            metadata=metadata,
            linked_memories=[],
        )

    # === Embedding Operations ===

    async def save_embedding(
        self,
        memory_id: str,
        embedding: np.ndarray,
        model: str,
    ) -> None:
        """
        Save memory embedding with optional compression.

        Args:
            memory_id: Memory ID
            embedding: Embedding vector as numpy array
            model: Name of the embedding model used
        """
        # Serialize numpy array
        buffer = io.BytesIO()
        np.save(buffer, embedding.astype(np.float32))
        embedding_bytes = buffer.getvalue()

        # Compress if enabled and large enough
        if self._compress_embeddings and len(embedding_bytes) > 1024:
            embedding_bytes = zlib.compress(embedding_bytes, level=6)
            compressed = True
        else:
            compressed = False

        query = """
            INSERT INTO memory_embeddings (memory_id, embedding, embedding_model, dimension, compressed, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(memory_id) DO UPDATE SET
                embedding = excluded.embedding,
                embedding_model = excluded.embedding_model,
                dimension = excluded.dimension,
                compressed = excluded.compressed,
                created_at = excluded.created_at
        """

        await self.db.execute(query, (
            memory_id,
            embedding_bytes,
            model,
            embedding.shape[-1],
            1 if compressed else 0,
            datetime.now().isoformat(),
        ))

    async def get_embedding(self, memory_id: str) -> Optional[np.ndarray]:
        """Get memory embedding."""
        query = "SELECT embedding, compressed FROM memory_embeddings WHERE memory_id = ?"
        row = await self.db.fetch_one(query, (memory_id,))

        if not row or not row.get("embedding"):
            return None

        embedding_bytes = row["embedding"]

        # Decompress if needed
        if row.get("compressed"):
            embedding_bytes = zlib.decompress(embedding_bytes)

        # Deserialize numpy array
        buffer = io.BytesIO(embedding_bytes)
        return np.load(buffer)

    async def get_embeddings_batch(
        self,
        memory_ids: list[str],
    ) -> dict[str, np.ndarray]:
        """Get multiple embeddings efficiently."""
        if not memory_ids:
            return {}

        placeholders = ", ".join(["?" for _ in memory_ids])
        query = f"""
            SELECT memory_id, embedding, compressed
            FROM memory_embeddings
            WHERE memory_id IN ({placeholders})
        """

        rows = await self.db.fetch_all(query, tuple(memory_ids))

        result = {}
        for row in rows:
            if row.get("embedding"):
                embedding_bytes = row["embedding"]

                if row.get("compressed"):
                    embedding_bytes = zlib.decompress(embedding_bytes)

                buffer = io.BytesIO(embedding_bytes)
                result[row["memory_id"]] = np.load(buffer)

        return result

    async def get_all_embeddings(self) -> dict[str, np.ndarray]:
        """Get all embeddings as {memory_id: embedding}."""
        query = "SELECT memory_id, embedding, compressed FROM memory_embeddings"
        rows = await self.db.fetch_all(query)

        embeddings = {}
        for row in rows:
            if row.get("embedding"):
                embedding_bytes = row["embedding"]

                if row.get("compressed"):
                    embedding_bytes = zlib.decompress(embedding_bytes)

                buffer = io.BytesIO(embedding_bytes)
                embeddings[row["memory_id"]] = np.load(buffer)

        return embeddings

    async def delete_embedding(self, memory_id: str) -> bool:
        """Delete memory embedding."""
        query = "DELETE FROM memory_embeddings WHERE memory_id = ?"
        await self.db.execute(query, (memory_id,))
        return True

    # === Relation Operations ===

    async def add_relation(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        strength: float = 1.0,
        metadata: Optional[dict] = None,
    ) -> None:
        """Add a relation between memories."""
        query = """
            INSERT INTO memory_relations (source_id, target_id, relation_type, strength, metadata, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_id, target_id, relation_type) DO UPDATE SET
                strength = excluded.strength,
                metadata = excluded.metadata
        """

        await self.db.execute(query, (
            source_id,
            target_id,
            relation_type,
            strength,
            self._to_json(metadata),
            datetime.now().isoformat(),
        ))

    async def get_relations(
        self,
        memory_id: str,
        relation_type: Optional[str] = None,
        direction: str = "both",  # "outgoing", "incoming", "both"
    ) -> list[dict[str, Any]]:
        """Get memory relations."""
        conditions = []
        params = []

        if direction == "outgoing":
            conditions.append("source_id = ?")
            params.append(memory_id)
        elif direction == "incoming":
            conditions.append("target_id = ?")
            params.append(memory_id)
        else:  # both
            conditions.append("(source_id = ? OR target_id = ?)")
            params.extend([memory_id, memory_id])

        if relation_type:
            conditions.append("relation_type = ?")
            params.append(relation_type)

        query = f"""
            SELECT * FROM memory_relations
            WHERE {" AND ".join(conditions)}
            ORDER BY strength DESC
        """

        return await self.db.fetch_all(query, tuple(params))

    async def remove_relation(
        self,
        source_id: str,
        target_id: str,
        relation_type: Optional[str] = None,
    ) -> bool:
        """Remove a relation between memories."""
        if relation_type:
            query = """
                DELETE FROM memory_relations
                WHERE source_id = ? AND target_id = ? AND relation_type = ?
            """
            await self.db.execute(query, (source_id, target_id, relation_type))
        else:
            query = """
                DELETE FROM memory_relations
                WHERE source_id = ? AND target_id = ?
            """
            await self.db.execute(query, (source_id, target_id))

        return True

    async def get_related_memories(
        self,
        memory_id: str,
        depth: int = 1,
        min_strength: float = 0.0,
    ) -> list[Memory]:
        """Get memories related to a given memory, traversing the graph."""
        visited = {memory_id}
        to_visit = [memory_id]
        related_ids = []

        for _ in range(depth):
            next_visit = []
            for mid in to_visit:
                relations = await self.get_relations(mid)
                for rel in relations:
                    if rel["strength"] < min_strength:
                        continue

                    related_id = (
                        rel["target_id"]
                        if rel["source_id"] == mid
                        else rel["source_id"]
                    )

                    if related_id not in visited:
                        visited.add(related_id)
                        related_ids.append(related_id)
                        next_visit.append(related_id)

            to_visit = next_visit

        if not related_ids:
            return []

        return list((await self.get_many(related_ids)).values())

    # === FAISS Index Operations ===

    async def save_faiss_index(
        self,
        name: str,
        index,  # faiss.Index
        id_mapping: dict[int, str],
        index_type: str,
    ) -> None:
        """Save a FAISS index."""
        try:
            import faiss

            # Serialize FAISS index
            index_bytes = faiss.serialize_index(index).tobytes()

            # Compress
            compressed_bytes = zlib.compress(index_bytes, level=6)

            query = """
                INSERT INTO faiss_indices (id, name, index_type, dimension, num_vectors, index_data, id_mapping, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    index_type = excluded.index_type,
                    dimension = excluded.dimension,
                    num_vectors = excluded.num_vectors,
                    index_data = excluded.index_data,
                    id_mapping = excluded.id_mapping,
                    created_at = excluded.created_at
            """

            await self.db.execute(query, (
                str(uuid.uuid4()),
                name,
                index_type,
                index.d,
                index.ntotal,
                compressed_bytes,
                self._to_json(id_mapping),
                datetime.now().isoformat(),
            ))

            logger.info(
                "FAISS index saved",
                name=name,
                vectors=index.ntotal,
                dimension=index.d,
            )

        except ImportError:
            logger.warning("FAISS not available, index not saved")

    async def load_faiss_index(self, name: str) -> Optional[tuple]:
        """
        Load a FAISS index.

        Returns:
            Tuple of (index, id_mapping) or None if not found
        """
        try:
            import faiss

            query = "SELECT index_data, id_mapping FROM faiss_indices WHERE name = ?"
            row = await self.db.fetch_one(query, (name,))

            if not row:
                return None

            # Decompress
            index_bytes = zlib.decompress(row["index_data"])

            # Deserialize FAISS index
            index = faiss.deserialize_index(
                np.frombuffer(index_bytes, dtype=np.uint8)
            )

            # Parse ID mapping
            id_mapping = self._from_json(row["id_mapping"])
            if id_mapping:
                id_mapping = {int(k): v for k, v in id_mapping.items()}
            else:
                id_mapping = {}

            logger.info(
                "FAISS index loaded",
                name=name,
                vectors=index.ntotal,
            )

            return index, id_mapping

        except ImportError:
            logger.warning("FAISS not available")
            return None

    async def delete_faiss_index(self, name: str) -> bool:
        """Delete a FAISS index."""
        query = "DELETE FROM faiss_indices WHERE name = ?"
        await self.db.execute(query, (name,))
        return True

    async def list_faiss_indices(self) -> list[dict[str, Any]]:
        """List all saved FAISS indices."""
        query = """
            SELECT id, name, index_type, dimension, num_vectors, created_at
            FROM faiss_indices
            ORDER BY created_at DESC
        """
        return await self.db.fetch_all(query)

    # === Query Helpers ===

    async def find_by_type(
        self,
        memory_type: MemoryType,
        options: Optional[QueryOptions] = None,
    ) -> list[Memory]:
        """Find memories by type."""
        type_value = memory_type.value if isinstance(memory_type, MemoryType) else memory_type
        return await self.find_where(
            "memory_type = ?",
            (type_value,),
            options=options,
        )

    async def find_important(
        self,
        min_importance: float = 0.7,
        options: Optional[QueryOptions] = None,
    ) -> list[Memory]:
        """Find important memories."""
        options = options or QueryOptions()
        options.order_by = "importance DESC"
        return await self.find_where(
            "importance >= ?",
            (min_importance,),
            options=options,
        )

    async def find_recent(
        self,
        hours: int = 24,
        options: Optional[QueryOptions] = None,
    ) -> list[Memory]:
        """Find recently accessed memories."""
        options = options or QueryOptions()
        options.order_by = "last_accessed_at DESC"
        return await self.find_where(
            "last_accessed_at >= datetime('now', '-' || ? || ' hours')",
            (hours,),
            options=options,
        )

    async def search_content(
        self,
        query: str,
        options: Optional[QueryOptions] = None,
    ) -> list[Memory]:
        """Search memories by content (basic LIKE search)."""
        return await self.find_where(
            "content LIKE ?",
            (f"%{query}%",),
            options=options,
        )

    async def update_access(self, id: str) -> None:
        """Update access time and count for a memory."""
        query = """
            UPDATE memories
            SET last_accessed_at = ?, access_count = access_count + 1
            WHERE id = ?
        """
        await self.db.execute(query, (datetime.now().isoformat(), id))

        # Invalidate cache
        if self.cache:
            await self.cache.delete(self._cache_key(id))

    async def decay_memories(
        self,
        threshold: float = 0.1,
    ) -> int:
        """
        Apply decay to memories and soft-delete those below threshold.

        Returns number of memories decayed/deleted.
        """
        # Get memories to potentially decay
        query = """
            SELECT id, importance, decay_rate, last_accessed_at, created_at
            FROM memories
            WHERE deleted_at IS NULL
        """
        rows = await self.db.fetch_all(query)

        decayed = 0
        for row in rows:
            last_access = self._to_datetime(row.get("last_accessed_at"))
            created = self._to_datetime(row.get("created_at"))
            base_time = last_access or created or datetime.now()

            elapsed_hours = (datetime.now() - base_time).total_seconds() / 3600
            decay = row.get("decay_rate", 0.01)
            new_importance = row.get("importance", 0.5) * np.exp(-decay * elapsed_hours)

            if new_importance < threshold:
                # Soft delete
                await self.delete(row["id"], hard=False)
                decayed += 1
            elif new_importance < row.get("importance", 0.5) - 0.01:
                # Update importance
                await self.update_fields(row["id"], {"importance": new_importance})

        return decayed

    async def get_statistics(self) -> dict[str, Any]:
        """Get memory system statistics."""
        stats_query = """
            SELECT
                COUNT(*) as total,
                COUNT(CASE WHEN memory_type = 'episodic' THEN 1 END) as episodic,
                COUNT(CASE WHEN memory_type = 'semantic' THEN 1 END) as semantic,
                COUNT(CASE WHEN memory_type = 'procedural' THEN 1 END) as procedural,
                COUNT(CASE WHEN memory_type = 'working' THEN 1 END) as working,
                AVG(importance) as avg_importance,
                SUM(access_count) as total_accesses
            FROM memories
            WHERE deleted_at IS NULL
        """

        embedding_query = """
            SELECT COUNT(*) as count, AVG(dimension) as avg_dimension
            FROM memory_embeddings
        """

        relation_query = """
            SELECT COUNT(*) as count
            FROM memory_relations
        """

        stats = await self.db.fetch_one(stats_query)
        embeddings = await self.db.fetch_one(embedding_query)
        relations = await self.db.fetch_one(relation_query)

        return {
            "memories": {
                "total": stats["total"] if stats else 0,
                "by_type": {
                    "episodic": stats["episodic"] if stats else 0,
                    "semantic": stats["semantic"] if stats else 0,
                    "procedural": stats["procedural"] if stats else 0,
                    "working": stats["working"] if stats else 0,
                },
                "avg_importance": stats["avg_importance"] if stats else 0,
                "total_accesses": stats["total_accesses"] if stats else 0,
            },
            "embeddings": {
                "count": embeddings["count"] if embeddings else 0,
                "avg_dimension": embeddings["avg_dimension"] if embeddings else 0,
            },
            "relations": {
                "count": relations["count"] if relations else 0,
            },
        }
