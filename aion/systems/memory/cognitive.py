"""
AION Cognitive Memory System

A comprehensive memory system inspired by human cognition:
- Episodic memory (experiences and events)
- Semantic memory (facts and knowledge)
- Working memory (current context)
- Memory consolidation and forgetting
- Importance-based retention
"""

from __future__ import annotations

import asyncio
import hashlib
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional, Union

import numpy as np
import structlog

from aion.systems.memory.embeddings import EmbeddingEngine
from aion.systems.memory.index import VectorIndex, IndexType, SearchResult

logger = structlog.get_logger(__name__)


class MemoryType(str, Enum):
    """Types of memories."""
    EPISODIC = "episodic"    # Events and experiences
    SEMANTIC = "semantic"    # Facts and knowledge
    PROCEDURAL = "procedural"  # Skills and procedures
    WORKING = "working"      # Current context


@dataclass
class Memory:
    """A single memory entry."""
    id: str
    content: str
    memory_type: MemoryType
    embedding: Optional[np.ndarray] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    importance: float = 0.5  # 0-1 scale
    access_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: Optional[datetime] = None
    decay_rate: float = 0.01
    linked_memories: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type.value,
            "metadata": self.metadata,
            "importance": self.importance,
            "access_count": self.access_count,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
            "decay_rate": self.decay_rate,
            "linked_memories": self.linked_memories,
        }

    def current_strength(self) -> float:
        """Calculate current memory strength based on decay."""
        if self.last_accessed is None:
            elapsed = (datetime.now() - self.created_at).total_seconds()
        else:
            elapsed = (datetime.now() - self.last_accessed).total_seconds()

        # Exponential decay with access boost
        base_strength = self.importance * np.exp(-self.decay_rate * elapsed / 3600)
        access_boost = min(0.3, self.access_count * 0.02)

        return min(1.0, base_strength + access_boost)


@dataclass
class MemorySearchResult:
    """A memory search result."""
    memory: Memory
    relevance: float
    combined_score: float  # Relevance + strength


class WorkingMemory:
    """
    Working memory for current context.

    Maintains a limited capacity buffer of active memories
    that are immediately relevant to the current task.
    """

    def __init__(self, capacity: int = 7):  # Miller's Law
        self.capacity = capacity
        self._items: list[Memory] = []
        self._context: dict[str, Any] = {}

    def add(self, memory: Memory) -> None:
        """Add a memory to working memory."""
        # Remove if already present
        self._items = [m for m in self._items if m.id != memory.id]

        # Add to front
        self._items.insert(0, memory)

        # Enforce capacity
        if len(self._items) > self.capacity:
            self._items = self._items[:self.capacity]

    def get_all(self) -> list[Memory]:
        """Get all items in working memory."""
        return self._items.copy()

    def clear(self) -> None:
        """Clear working memory."""
        self._items.clear()
        self._context.clear()

    def set_context(self, key: str, value: Any) -> None:
        """Set context variable."""
        self._context[key] = value

    def get_context(self, key: str) -> Optional[Any]:
        """Get context variable."""
        return self._context.get(key)

    def get_summary(self) -> str:
        """Get a summary of working memory contents."""
        if not self._items:
            return "Working memory is empty."

        items = [f"- {m.content[:100]}..." for m in self._items[:5]]
        return "Working memory:\n" + "\n".join(items)


class CognitiveMemorySystem:
    """
    AION Cognitive Memory System

    A sophisticated memory system that mimics human memory processes:
    - Multiple memory types (episodic, semantic, procedural)
    - Working memory for active context
    - Memory consolidation (transferring important memories)
    - Natural forgetting (decay over time)
    - Associative retrieval (linking related memories)
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        embedding_dim: int = 384,
        max_memories: int = 100_000,
        index_type: IndexType = IndexType.FLAT,
        consolidation_interval: int = 3600,  # seconds
        forgetting_threshold: float = 0.1,
    ):
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim
        self.max_memories = max_memories
        self.consolidation_interval = consolidation_interval
        self.forgetting_threshold = forgetting_threshold

        # Components
        self._embeddings = EmbeddingEngine(model_name=embedding_model)
        self._index = VectorIndex(
            dimension=embedding_dim,
            index_type=index_type,
            max_size=max_memories,
        )
        self._working_memory = WorkingMemory()

        # Memory storage
        self._memories: dict[str, Memory] = {}

        # Statistics
        self._stats = {
            "memories_stored": 0,
            "memories_retrieved": 0,
            "memories_forgotten": 0,
            "consolidations": 0,
        }

        # Background tasks
        self._consolidation_task: Optional[asyncio.Task] = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the memory system."""
        if self._initialized:
            return

        logger.info("Initializing Cognitive Memory System")

        await self._embeddings.initialize()
        await self._index.initialize()

        # Update embedding dimension from model
        self.embedding_dim = self._embeddings.dimension

        # Start consolidation loop
        self._consolidation_task = asyncio.create_task(
            self._consolidation_loop()
        )

        self._initialized = True
        logger.info("Cognitive Memory System initialized")

    async def shutdown(self) -> None:
        """Shutdown the memory system."""
        if self._consolidation_task:
            self._consolidation_task.cancel()
            try:
                await self._consolidation_task
            except asyncio.CancelledError:
                pass

        await self._embeddings.shutdown()
        await self._index.shutdown()

        self._initialized = False
        logger.info("Cognitive Memory System shutdown complete")

    async def store(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.EPISODIC,
        metadata: Optional[dict[str, Any]] = None,
        importance: float = 0.5,
        linked_to: Optional[list[str]] = None,
    ) -> Memory:
        """
        Store a new memory.

        Args:
            content: Memory content
            memory_type: Type of memory
            metadata: Additional metadata
            importance: Importance score (0-1)
            linked_to: IDs of related memories

        Returns:
            Created Memory
        """
        if not self._initialized:
            await self.initialize()

        # Generate ID
        memory_id = str(uuid.uuid4())

        # Generate embedding
        embedding = await self._embeddings.embed(content)
        if embedding.ndim > 1:
            embedding = embedding[0]

        # Create memory
        memory = Memory(
            id=memory_id,
            content=content,
            memory_type=memory_type,
            embedding=embedding,
            metadata=metadata or {},
            importance=importance,
            linked_memories=linked_to or [],
        )

        # Store in index
        await self._index.add(
            memory_id,
            embedding,
            {
                "content": content,
                "type": memory_type.value,
                "importance": importance,
                **memory.metadata,
            },
        )

        # Store in memory dict
        self._memories[memory_id] = memory
        self._stats["memories_stored"] += 1

        # Add to working memory if important
        if importance > 0.7:
            self._working_memory.add(memory)

        logger.debug(
            "Memory stored",
            memory_id=memory_id,
            type=memory_type.value,
            importance=importance,
        )

        return memory

    async def search(
        self,
        query: str,
        limit: int = 10,
        memory_type: Optional[MemoryType] = None,
        min_importance: float = 0.0,
        include_strength: bool = True,
    ) -> list[MemorySearchResult]:
        """
        Search for relevant memories.

        Args:
            query: Search query
            limit: Maximum results
            memory_type: Filter by type
            min_importance: Minimum importance threshold
            include_strength: Include memory strength in scoring

        Returns:
            List of MemorySearchResult
        """
        if not self._initialized:
            await self.initialize()

        # Generate query embedding
        query_embedding = await self._embeddings.embed(query)
        if query_embedding.ndim > 1:
            query_embedding = query_embedding[0]

        # Build filter
        def filter_fn(metadata: dict) -> bool:
            if memory_type and metadata.get("type") != memory_type.value:
                return False
            if metadata.get("importance", 0) < min_importance:
                return False
            return True

        # Search index
        search_results = await self._index.search(
            query_embedding,
            k=limit * 2,  # Get more for filtering
            filter_fn=filter_fn,
        )

        # Build results with memory objects
        results = []
        for sr in search_results:
            memory = self._memories.get(sr.id)
            if memory is None:
                continue

            # Update access tracking
            memory.access_count += 1
            memory.last_accessed = datetime.now()

            # Calculate combined score
            relevance = sr.score
            strength = memory.current_strength() if include_strength else 1.0
            combined = relevance * 0.7 + strength * 0.3

            results.append(MemorySearchResult(
                memory=memory,
                relevance=relevance,
                combined_score=combined,
            ))

            if len(results) >= limit:
                break

        # Sort by combined score
        results.sort(key=lambda x: x.combined_score, reverse=True)

        self._stats["memories_retrieved"] += len(results)

        return results

    async def recall(
        self,
        query: str,
        context: Optional[str] = None,
    ) -> str:
        """
        Recall relevant information as a synthesized response.

        Args:
            query: What to recall
            context: Additional context

        Returns:
            Synthesized response from memories
        """
        results = await self.search(query, limit=5)

        if not results:
            return "No relevant memories found."

        # Synthesize response
        memories_text = "\n".join([
            f"- {r.memory.content} (relevance: {r.relevance:.2f})"
            for r in results
        ])

        return f"Recalled {len(results)} relevant memories:\n{memories_text}"

    async def associate(
        self,
        memory_id: str,
        related_id: str,
        bidirectional: bool = True,
    ) -> bool:
        """
        Create an association between memories.

        Args:
            memory_id: First memory ID
            related_id: Related memory ID
            bidirectional: Create link in both directions

        Returns:
            True if association created
        """
        if memory_id not in self._memories or related_id not in self._memories:
            return False

        memory = self._memories[memory_id]
        if related_id not in memory.linked_memories:
            memory.linked_memories.append(related_id)

        if bidirectional:
            related = self._memories[related_id]
            if memory_id not in related.linked_memories:
                related.linked_memories.append(memory_id)

        return True

    async def get_associated(
        self,
        memory_id: str,
        depth: int = 1,
    ) -> list[Memory]:
        """
        Get memories associated with a given memory.

        Args:
            memory_id: Memory ID
            depth: How many levels of association to follow

        Returns:
            List of associated memories
        """
        if memory_id not in self._memories:
            return []

        seen = {memory_id}
        to_visit = [memory_id]
        results = []

        for _ in range(depth):
            next_visit = []
            for mid in to_visit:
                memory = self._memories.get(mid)
                if memory is None:
                    continue

                for linked_id in memory.linked_memories:
                    if linked_id not in seen:
                        seen.add(linked_id)
                        linked = self._memories.get(linked_id)
                        if linked:
                            results.append(linked)
                            next_visit.append(linked_id)

            to_visit = next_visit

        return results

    async def forget(
        self,
        memory_id: str,
        permanent: bool = False,
    ) -> bool:
        """
        Forget a memory.

        Args:
            memory_id: Memory to forget
            permanent: If True, completely remove. Otherwise, just reduce importance

        Returns:
            True if forgotten
        """
        if memory_id not in self._memories:
            return False

        if permanent:
            del self._memories[memory_id]
            await self._index.remove(memory_id)
            self._stats["memories_forgotten"] += 1
        else:
            memory = self._memories[memory_id]
            memory.importance *= 0.5
            memory.decay_rate *= 2

        return True

    async def consolidate(self) -> dict[str, int]:
        """
        Run memory consolidation.

        This process:
        1. Removes weak memories below threshold
        2. Strengthens important memories
        3. Promotes episodic to semantic memories

        Returns:
            Statistics about the consolidation
        """
        stats = {
            "evaluated": 0,
            "forgotten": 0,
            "strengthened": 0,
            "promoted": 0,
        }

        memories_to_forget = []

        for memory_id, memory in self._memories.items():
            stats["evaluated"] += 1
            strength = memory.current_strength()

            if strength < self.forgetting_threshold:
                memories_to_forget.append(memory_id)
                stats["forgotten"] += 1

            elif strength > 0.8:
                # Strengthen important memories
                memory.importance = min(1.0, memory.importance * 1.1)
                memory.decay_rate *= 0.9
                stats["strengthened"] += 1

                # Promote frequently accessed episodic to semantic
                if (memory.memory_type == MemoryType.EPISODIC and
                    memory.access_count > 5):
                    memory.memory_type = MemoryType.SEMANTIC
                    stats["promoted"] += 1

        # Forget weak memories
        for memory_id in memories_to_forget:
            await self.forget(memory_id, permanent=True)

        self._stats["consolidations"] += 1
        logger.info("Memory consolidation complete", **stats)

        return stats

    async def _consolidation_loop(self) -> None:
        """Background consolidation loop."""
        while True:
            await asyncio.sleep(self.consolidation_interval)
            try:
                await self.consolidate()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Consolidation error", error=str(e))

    def get_working_memory(self) -> WorkingMemory:
        """Get working memory."""
        return self._working_memory

    def get_stats(self) -> dict[str, Any]:
        """Get memory system statistics."""
        return {
            **self._stats,
            "total_memories": len(self._memories),
            "working_memory_items": len(self._working_memory.get_all()),
            "index_size": self._index.count(),
        }

    async def export_memories(
        self,
        memory_type: Optional[MemoryType] = None,
    ) -> list[dict[str, Any]]:
        """
        Export memories for persistence.

        Args:
            memory_type: Filter by type

        Returns:
            List of memory dictionaries
        """
        memories = []
        for memory in self._memories.values():
            if memory_type and memory.memory_type != memory_type:
                continue
            memories.append(memory.to_dict())
        return memories

    async def import_memories(
        self,
        memories: list[dict[str, Any]],
    ) -> int:
        """
        Import memories from persistence.

        Args:
            memories: List of memory dictionaries

        Returns:
            Number of memories imported
        """
        imported = 0
        for data in memories:
            try:
                await self.store(
                    content=data["content"],
                    memory_type=MemoryType(data.get("memory_type", "episodic")),
                    metadata=data.get("metadata", {}),
                    importance=data.get("importance", 0.5),
                    linked_to=data.get("linked_memories", []),
                )
                imported += 1
            except Exception as e:
                logger.warning(f"Failed to import memory: {e}")

        return imported
