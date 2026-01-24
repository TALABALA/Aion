"""
Agent Memory Manager

Unified interface for managing all memory systems for an agent.
Provides integrated memory operations across vector, episodic,
semantic, and working memory.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional, Callable

import structlog

from .vector_store import VectorStore, VectorEntry, SimilarityMetric
from .episodic import EpisodicMemory, Episode, EpisodeType
from .semantic import SemanticMemory, Concept, Relation, RelationType, KnowledgeTriple
from .working import WorkingMemory, MemorySlot, SlotType
from .rag import RAGEngine, RAGConfig, RAGContext
from .consolidation import MemoryConsolidator, ConsolidationStrategy

logger = structlog.get_logger()


@dataclass
class MemoryConfig:
    """Configuration for agent memory systems."""

    # Storage
    storage_path: Optional[Path] = None

    # Vector store
    embedding_dimension: int = 1536
    max_vectors: int = 100000

    # Episodic memory
    max_episodes: int = 10000
    episode_decay_rate: float = 0.01

    # Semantic memory
    max_concepts: int = 100000
    max_relations: int = 500000

    # Working memory
    working_memory_capacity: int = 7
    attention_decay_rate: float = 0.1

    # RAG
    rag_top_k: int = 10
    rag_similarity_threshold: float = 0.5

    # Consolidation
    consolidation_interval_hours: float = 1.0
    min_pattern_support: int = 3

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "storage_path": str(self.storage_path) if self.storage_path else None,
            "embedding_dimension": self.embedding_dimension,
            "max_vectors": self.max_vectors,
            "max_episodes": self.max_episodes,
            "episode_decay_rate": self.episode_decay_rate,
            "max_concepts": self.max_concepts,
            "max_relations": self.max_relations,
            "working_memory_capacity": self.working_memory_capacity,
            "attention_decay_rate": self.attention_decay_rate,
            "rag_top_k": self.rag_top_k,
            "rag_similarity_threshold": self.rag_similarity_threshold,
            "consolidation_interval_hours": self.consolidation_interval_hours,
            "min_pattern_support": self.min_pattern_support,
        }


class AgentMemoryManager:
    """
    Unified memory manager for agents.

    Features:
    - Integrated access to all memory systems
    - Automatic memory routing
    - Cross-memory search
    - Context-aware retrieval
    - Memory lifecycle management
    - Statistics and monitoring
    """

    def __init__(
        self,
        agent_id: str,
        config: Optional[MemoryConfig] = None,
        embedding_fn: Optional[Callable[[str], list[float]]] = None,
    ):
        self.agent_id = agent_id
        self.config = config or MemoryConfig()
        self.embedding_fn = embedding_fn

        # Determine storage paths
        base_path = self.config.storage_path or Path(f"./memory/{agent_id}")

        # Initialize memory systems
        self.vector_store = VectorStore(
            dimension=self.config.embedding_dimension,
            max_elements=self.config.max_vectors,
            storage_path=base_path / "vectors",
            embedding_fn=embedding_fn,
        )

        self.episodic_memory = EpisodicMemory(
            max_episodes=self.config.max_episodes,
            embedding_dimension=self.config.embedding_dimension,
            storage_path=base_path / "episodic",
            decay_rate=self.config.episode_decay_rate,
        )

        self.semantic_memory = SemanticMemory(
            storage_path=base_path / "semantic",
            max_concepts=self.config.max_concepts,
            max_relations=self.config.max_relations,
        )

        self.working_memory = WorkingMemory(
            capacity=self.config.working_memory_capacity,
            decay_rate=self.config.attention_decay_rate,
            context_id=f"{agent_id}-working",
        )

        # RAG engine
        self.rag = RAGEngine(
            vector_store=self.vector_store,
            episodic_memory=self.episodic_memory,
            semantic_memory=self.semantic_memory,
            config=RAGConfig(
                top_k=self.config.rag_top_k,
                similarity_threshold=self.config.rag_similarity_threshold,
            ),
            embedding_fn=embedding_fn,
        )

        # Consolidator
        self.consolidator = MemoryConsolidator(
            episodic_memory=self.episodic_memory,
            semantic_memory=self.semantic_memory,
            vector_store=self.vector_store,
            consolidation_interval=timedelta(hours=self.config.consolidation_interval_hours),
            min_pattern_support=self.config.min_pattern_support,
        )

        # Current episode being recorded
        self._current_episode: Optional[Episode] = None

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize all memory systems."""
        if self._initialized:
            return

        await asyncio.gather(
            self.vector_store.initialize(),
            self.episodic_memory.initialize(),
            self.semantic_memory.initialize(),
            self.working_memory.initialize(),
            self.rag.initialize(),
            self.consolidator.initialize(),
        )

        self._initialized = True
        logger.info("agent_memory_manager_initialized", agent_id=self.agent_id)

    async def shutdown(self) -> None:
        """Shutdown all memory systems."""
        await asyncio.gather(
            self.vector_store.shutdown(),
            self.episodic_memory.shutdown(),
            self.semantic_memory.shutdown(),
            self.working_memory.shutdown(),
            self.rag.shutdown(),
            self.consolidator.shutdown(),
        )

        self._initialized = False
        logger.info("agent_memory_manager_shutdown", agent_id=self.agent_id)

    # ========== Unified Memory Operations ==========

    async def remember(
        self,
        content: str,
        memory_type: str = "auto",  # "auto", "vector", "episodic", "semantic", "working"
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """
        Store something in memory.

        Automatically routes to appropriate memory system based on content.
        """
        metadata = metadata or {}

        if memory_type == "auto":
            memory_type = self._classify_memory_type(content, metadata)

        if memory_type == "vector":
            entry_id = await self.vector_store.add(
                text=content,
                metadata=metadata,
            )
            return entry_id

        elif memory_type == "episodic":
            if self._current_episode:
                self._current_episode.add_step(
                    action=metadata.get("action", "observe"),
                    observation=content,
                    thought=metadata.get("thought", ""),
                    reward=metadata.get("reward", 0.0),
                )
                return self._current_episode.id
            else:
                # Create single-step episode
                episode = Episode(
                    agent_id=self.agent_id,
                    title=content[:50],
                    description=content,
                    episode_type=EpisodeType(metadata.get("episode_type", "task_execution")),
                )
                episode.add_step(
                    action=metadata.get("action", "observe"),
                    observation=content,
                )
                episode.complete(outcome=content[:100], success=True)
                await self.episodic_memory.store(episode)
                return episode.id

        elif memory_type == "semantic":
            # Parse as knowledge triple if possible
            if "subject" in metadata and "predicate" in metadata and "object" in metadata:
                triple = KnowledgeTriple(
                    subject=metadata["subject"],
                    predicate=metadata["predicate"],
                    object=metadata["object"],
                )
                subj, rel, obj = await self.semantic_memory.add_triple(triple)
                return f"{subj.id}:{rel.id}:{obj.id}"
            else:
                # Add as concept
                concept = await self.semantic_memory.add_concept(
                    name=metadata.get("name", content[:50]),
                    description=content,
                )
                return concept.id

        elif memory_type == "working":
            slot = await self.working_memory.store(
                content=content,
                slot_type=SlotType(metadata.get("slot_type", "intermediate")),
                metadata=metadata,
            )
            return slot.id

        return ""

    async def recall(
        self,
        query: str,
        k: int = 5,
        memory_types: Optional[list[str]] = None,
    ) -> RAGContext:
        """
        Recall relevant information for a query.

        Uses RAG to retrieve from all memory systems.
        """
        return await self.rag.retrieve(query)

    async def forget(
        self,
        memory_id: str,
        memory_type: str = "auto",
    ) -> bool:
        """Remove something from memory."""
        if memory_type == "auto":
            # Try each memory type
            if await self.vector_store.delete(memory_id):
                return True
            if memory_id in self.working_memory._slots:
                return await self.working_memory.delete(memory_id)
            return False

        if memory_type == "vector":
            return await self.vector_store.delete(memory_id)
        elif memory_type == "working":
            return await self.working_memory.delete(memory_id)

        return False

    # ========== Episode Recording ==========

    async def start_episode(
        self,
        title: str,
        episode_type: EpisodeType = EpisodeType.TASK_EXECUTION,
        context: Optional[dict[str, Any]] = None,
    ) -> Episode:
        """Start recording a new episode."""
        if self._current_episode:
            # Auto-complete previous episode
            await self.end_episode(outcome="Interrupted", success=False)

        self._current_episode = Episode(
            agent_id=self.agent_id,
            title=title,
            episode_type=episode_type,
            context=context or {},
        )

        logger.debug("episode_started", episode_id=self._current_episode.id, title=title)

        return self._current_episode

    async def record_step(
        self,
        action: str,
        observation: str,
        thought: str = "",
        reward: float = 0.0,
    ) -> None:
        """Record a step in the current episode."""
        if self._current_episode:
            self._current_episode.add_step(
                action=action,
                observation=observation,
                thought=thought,
                reward=reward,
            )

    async def end_episode(
        self,
        outcome: str,
        success: bool,
        lessons: Optional[list[str]] = None,
    ) -> Optional[Episode]:
        """End the current episode and store it."""
        if not self._current_episode:
            return None

        self._current_episode.complete(
            outcome=outcome,
            success=success,
            lessons=lessons,
        )

        await self.episodic_memory.store(self._current_episode)

        episode = self._current_episode
        self._current_episode = None

        logger.debug(
            "episode_ended",
            episode_id=episode.id,
            success=success,
            steps=len(episode.steps),
        )

        return episode

    # ========== Working Memory Operations ==========

    async def focus(self, content: Any, slot_type: SlotType = SlotType.ATTENTION) -> MemorySlot:
        """Put something in focused attention."""
        slot = await self.working_memory.store(
            content=content,
            slot_type=slot_type,
            lock=True,
        )
        await self.working_memory.focus(slot.id)
        return slot

    async def get_focus(self) -> Optional[MemorySlot]:
        """Get the currently focused item."""
        return await self.working_memory.get_focus()

    async def get_working_context(self) -> dict[str, Any]:
        """Get current working memory context."""
        return self.working_memory.get_context_snapshot()

    # ========== Knowledge Operations ==========

    async def learn(
        self,
        subject: str,
        predicate: str,
        obj: str,
        confidence: float = 1.0,
    ) -> tuple[Concept, Relation, Concept]:
        """Learn a new fact (add to semantic memory)."""
        triple = KnowledgeTriple(
            subject=subject,
            predicate=predicate,
            object=obj,
            confidence=confidence,
            source=self.agent_id,
        )
        return await self.semantic_memory.add_triple(triple)

    async def query_knowledge(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        obj: Optional[str] = None,
    ) -> list[tuple[Concept, Relation, Concept]]:
        """Query the knowledge graph."""
        return await self.semantic_memory.query(
            subject=subject,
            predicate=predicate,
            object=obj,
        )

    async def get_related_concepts(
        self,
        concept: str,
        relation_type: Optional[RelationType] = None,
    ) -> list[Concept]:
        """Get concepts related to the given concept."""
        relations = await self.semantic_memory.get_relations(
            concept,
            relation_type=relation_type,
        )

        concepts = []
        for rel in relations:
            target = await self.semantic_memory.get_concept(rel.target_id)
            if target:
                concepts.append(target)

        return concepts

    # ========== Experience Operations ==========

    async def get_similar_experiences(
        self,
        query: str,
        k: int = 5,
        success_only: bool = False,
    ) -> list[Episode]:
        """Get similar past experiences."""
        return await self.episodic_memory.search_similar(
            query=query,
            k=k,
            success_only=success_only,
        )

    async def get_lessons_learned(self, context: str, k: int = 10) -> list[str]:
        """Get relevant lessons from past experiences."""
        return await self.episodic_memory.get_lessons_for_context(context, k=k)

    async def replay_experiences(
        self,
        n: int = 5,
        prioritized: bool = True,
    ) -> list[Episode]:
        """Sample experiences for replay/learning."""
        return await self.episodic_memory.sample_for_replay(
            n=n,
            prioritized=prioritized,
            agent_id=self.agent_id,
        )

    # ========== Consolidation ==========

    async def consolidate(
        self,
        strategy: ConsolidationStrategy = ConsolidationStrategy.REPLAY,
    ) -> dict[str, Any]:
        """Run memory consolidation."""
        result = await self.consolidator.consolidate(strategy)
        return result.to_dict()

    def get_patterns(self) -> list[dict[str, Any]]:
        """Get patterns extracted during consolidation."""
        return [p.to_dict() for p in self.consolidator.get_patterns()]

    # ========== Utility ==========

    def _classify_memory_type(
        self,
        content: str,
        metadata: dict[str, Any],
    ) -> str:
        """Classify what type of memory to use."""
        # Check metadata hints
        if "subject" in metadata and "predicate" in metadata:
            return "semantic"
        if "episode_type" in metadata or "action" in metadata:
            return "episodic"
        if "slot_type" in metadata:
            return "working"

        # Content-based classification
        content_lower = content.lower()

        # Knowledge patterns
        knowledge_patterns = ["is a", "is an", "has", "can", "means", "causes"]
        if any(p in content_lower for p in knowledge_patterns):
            return "semantic"

        # Experience patterns
        experience_patterns = ["i did", "we completed", "task", "step", "action"]
        if any(p in content_lower for p in experience_patterns):
            return "episodic"

        # Default to vector for general content
        return "vector"

    def get_stats(self) -> dict[str, Any]:
        """Get comprehensive memory statistics."""
        return {
            "agent_id": self.agent_id,
            "initialized": self._initialized,
            "current_episode": self._current_episode.id if self._current_episode else None,
            "vector_store": self.vector_store.get_stats(),
            "episodic_memory": self.episodic_memory.get_stats(),
            "semantic_memory": self.semantic_memory.get_stats(),
            "working_memory": self.working_memory.get_stats(),
            "rag": self.rag.get_stats(),
            "consolidator": self.consolidator.get_stats(),
        }
