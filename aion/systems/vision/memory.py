"""
AION Visual Memory

Visual memory system for storing and retrieving visual experiences:
- Image similarity search
- Scene graph matching
- Visual concept learning
"""

from __future__ import annotations

import asyncio
import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, Union
from pathlib import Path

import numpy as np
import structlog

from aion.systems.vision.perception import (
    DetectedObject,
    SceneGraph,
)

logger = structlog.get_logger(__name__)


@dataclass
class VisualMemoryEntry:
    """An entry in visual memory."""
    id: str
    image_hash: str
    embedding: np.ndarray
    scene_graph: SceneGraph
    caption: str
    metadata: dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    importance: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "image_hash": self.image_hash,
            "scene_graph": self.scene_graph.to_dict(),
            "caption": self.caption,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "access_count": self.access_count,
            "importance": self.importance,
        }


@dataclass
class VisualSearchResult:
    """Result of a visual memory search."""
    entry: VisualMemoryEntry
    similarity: float


class VisualMemory:
    """
    AION Visual Memory System

    Stores visual experiences and enables retrieval by:
    - Visual similarity (embedding-based)
    - Object presence
    - Scene description
    - Concept matching
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        max_entries: int = 10000,
    ):
        self.embedding_dim = embedding_dim
        self.max_entries = max_entries

        # Storage
        self._entries: dict[str, VisualMemoryEntry] = {}
        self._embeddings: list[np.ndarray] = []
        self._id_to_idx: dict[str, int] = {}

        # Object-based index
        self._object_index: dict[str, set[str]] = {}  # label -> memory IDs

        # Statistics
        self._stats = {
            "entries_stored": 0,
            "searches_performed": 0,
        }

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the visual memory system."""
        if self._initialized:
            return

        logger.info("Initializing Visual Memory System")
        self._initialized = True

    async def shutdown(self) -> None:
        """Shutdown the visual memory system."""
        self._initialized = False

    def _compute_image_hash(
        self,
        image_data: Union[bytes, np.ndarray],
    ) -> str:
        """Compute a hash for image deduplication."""
        if isinstance(image_data, np.ndarray):
            image_bytes = image_data.tobytes()
        else:
            image_bytes = image_data

        return hashlib.sha256(image_bytes).hexdigest()[:16]

    def _compute_embedding(
        self,
        scene_graph: SceneGraph,
    ) -> np.ndarray:
        """
        Compute an embedding for a scene.

        Uses a simple bag-of-objects approach. In production,
        this would use a learned visual encoder.
        """
        # Create a fixed-size embedding based on object presence
        embedding = np.zeros(self.embedding_dim, dtype=np.float32)

        # Use hash of object labels for consistent indexing
        for obj in scene_graph.objects:
            label_hash = hash(obj.label) % (self.embedding_dim // 2)
            embedding[label_hash] += obj.confidence

            # Add positional information
            pos_idx = self.embedding_dim // 2 + hash(f"{obj.label}_{int(obj.bounding_box.center[0] * 10)}") % (self.embedding_dim // 2)
            embedding[pos_idx] += 0.5

        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    async def store(
        self,
        scene_graph: SceneGraph,
        image_data: Optional[Union[bytes, np.ndarray]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> VisualMemoryEntry:
        """
        Store a visual memory.

        Args:
            scene_graph: Scene graph from perception
            image_data: Optional raw image data
            metadata: Additional metadata

        Returns:
            Created VisualMemoryEntry
        """
        if not self._initialized:
            await self.initialize()

        # Generate ID and hash
        entry_id = str(uuid.uuid4())
        image_hash = ""
        if image_data is not None:
            image_hash = self._compute_image_hash(image_data)

            # Check for duplicate
            for existing in self._entries.values():
                if existing.image_hash == image_hash:
                    logger.debug("Duplicate image detected", hash=image_hash)
                    existing.access_count += 1
                    existing.last_accessed = datetime.now()
                    return existing

        # Compute embedding
        embedding = self._compute_embedding(scene_graph)

        # Create entry
        entry = VisualMemoryEntry(
            id=entry_id,
            image_hash=image_hash,
            embedding=embedding,
            scene_graph=scene_graph,
            caption=scene_graph.global_features.get("caption", ""),
            metadata=metadata or {},
        )

        # Store
        self._entries[entry_id] = entry
        idx = len(self._embeddings)
        self._embeddings.append(embedding)
        self._id_to_idx[entry_id] = idx

        # Update object index
        for obj in scene_graph.objects:
            if obj.label not in self._object_index:
                self._object_index[obj.label] = set()
            self._object_index[obj.label].add(entry_id)

        self._stats["entries_stored"] += 1

        # Enforce max entries
        if len(self._entries) > self.max_entries:
            await self._evict_oldest()

        logger.debug("Visual memory stored", entry_id=entry_id)
        return entry

    async def _evict_oldest(self) -> None:
        """Evict oldest/least important entries."""
        # Sort by access count and recency
        sorted_entries = sorted(
            self._entries.values(),
            key=lambda e: (e.importance, e.access_count, e.created_at),
        )

        # Remove oldest 10%
        to_remove = sorted_entries[:len(sorted_entries) // 10]

        for entry in to_remove:
            del self._entries[entry.id]
            # Note: embeddings array is not compacted for efficiency
            # In production, would periodically rebuild

            for label in self._object_index:
                self._object_index[label].discard(entry.id)

    async def search_by_similarity(
        self,
        query_scene: SceneGraph,
        limit: int = 10,
    ) -> list[VisualSearchResult]:
        """
        Search for similar visual memories.

        Args:
            query_scene: Scene to match
            limit: Maximum results

        Returns:
            List of VisualSearchResult sorted by similarity
        """
        if not self._entries:
            return []

        query_embedding = self._compute_embedding(query_scene)

        # Compute similarities
        results = []
        for entry_id, entry in self._entries.items():
            similarity = float(np.dot(query_embedding, entry.embedding))
            results.append(VisualSearchResult(
                entry=entry,
                similarity=similarity,
            ))

        # Sort by similarity
        results.sort(key=lambda r: r.similarity, reverse=True)

        # Update access tracking
        for result in results[:limit]:
            result.entry.access_count += 1
            result.entry.last_accessed = datetime.now()

        self._stats["searches_performed"] += 1

        return results[:limit]

    async def search_by_objects(
        self,
        object_labels: list[str],
        require_all: bool = False,
        limit: int = 10,
    ) -> list[VisualSearchResult]:
        """
        Search for memories containing specific objects.

        Args:
            object_labels: Objects to search for
            require_all: If True, all objects must be present
            limit: Maximum results

        Returns:
            List of VisualSearchResult
        """
        if not object_labels:
            return []

        # Find matching entries
        if require_all:
            matching = None
            for label in object_labels:
                label_matches = self._object_index.get(label, set())
                if matching is None:
                    matching = label_matches.copy()
                else:
                    matching &= label_matches
        else:
            matching = set()
            for label in object_labels:
                matching |= self._object_index.get(label, set())

        # Score by object match quality
        results = []
        for entry_id in matching:
            if entry_id not in self._entries:
                continue

            entry = self._entries[entry_id]
            entry_labels = {obj.label for obj in entry.scene_graph.objects}

            # Score by overlap
            overlap = len(entry_labels & set(object_labels))
            total = len(set(object_labels))
            similarity = overlap / total if total > 0 else 0

            results.append(VisualSearchResult(
                entry=entry,
                similarity=similarity,
            ))

        results.sort(key=lambda r: r.similarity, reverse=True)

        self._stats["searches_performed"] += 1

        return results[:limit]

    async def search_by_caption(
        self,
        query: str,
        limit: int = 10,
    ) -> list[VisualSearchResult]:
        """
        Search for memories by caption text.

        Args:
            query: Text query
            limit: Maximum results

        Returns:
            List of VisualSearchResult
        """
        query_words = set(query.lower().split())

        results = []
        for entry in self._entries.values():
            caption_words = set(entry.caption.lower().split())

            # Simple word overlap scoring
            overlap = len(query_words & caption_words)
            total = len(query_words)
            similarity = overlap / total if total > 0 else 0

            if similarity > 0:
                results.append(VisualSearchResult(
                    entry=entry,
                    similarity=similarity,
                ))

        results.sort(key=lambda r: r.similarity, reverse=True)

        self._stats["searches_performed"] += 1

        return results[:limit]

    async def get(self, entry_id: str) -> Optional[VisualMemoryEntry]:
        """Get a specific memory entry."""
        entry = self._entries.get(entry_id)
        if entry:
            entry.access_count += 1
            entry.last_accessed = datetime.now()
        return entry

    def count(self) -> int:
        """Get number of stored memories."""
        return len(self._entries)

    def get_object_labels(self) -> list[str]:
        """Get all unique object labels in memory."""
        return list(self._object_index.keys())

    def get_stats(self) -> dict[str, Any]:
        """Get memory statistics."""
        return {
            **self._stats,
            "total_entries": len(self._entries),
            "unique_objects": len(self._object_index),
        }
