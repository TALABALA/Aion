"""
AION SOTA Memory Integration System

State-of-the-art memory system featuring:
- RAG (Retrieval Augmented Generation) with semantic embeddings
- Hierarchical memory architecture (Working → Short-term → Long-term)
- Memory types: Episodic, Semantic, Procedural
- Graph-based knowledge representation
- Memory consolidation (inspired by human sleep consolidation)
- Importance-weighted retention and forgetting
- Temporal decay with reinforcement learning
"""

from __future__ import annotations

import asyncio
import hashlib
import heapq
import json
import math
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, Generic, List, Optional, Protocol, Set, Tuple, TypeVar

import numpy as np
import structlog

from aion.conversation.types import Message, MessageRole

logger = structlog.get_logger(__name__)


# =============================================================================
# Memory Types and Data Structures
# =============================================================================

class MemoryType(str, Enum):
    """Types of memory based on cognitive science."""
    WORKING = "working"        # Immediate, limited capacity, fast decay
    EPISODIC = "episodic"      # Personal experiences, autobiographical
    SEMANTIC = "semantic"      # Facts, concepts, general knowledge
    PROCEDURAL = "procedural"  # How-to knowledge, skills


class MemoryTier(str, Enum):
    """Memory hierarchy tiers."""
    WORKING = "working"       # ~7 items, seconds to minutes
    SHORT_TERM = "short_term"  # Hours to days
    LONG_TERM = "long_term"    # Days to permanent


class ConsolidationState(str, Enum):
    """Memory consolidation states."""
    NEW = "new"
    PROCESSING = "processing"
    CONSOLIDATED = "consolidated"
    ARCHIVED = "archived"


@dataclass
class MemoryEmbedding:
    """Embedding representation of memory content."""
    vector: List[float]
    model: str
    dimension: int
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class MemoryNode:
    """
    A single memory unit with full metadata.
    Represents a node in the memory graph.
    """
    id: str
    content: str
    memory_type: MemoryType
    tier: MemoryTier

    # Embeddings for semantic search
    embedding: Optional[MemoryEmbedding] = None

    # Importance and relevance
    importance: float = 0.5
    access_count: int = 0
    last_accessed: Optional[datetime] = None

    # Temporal information
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None

    # Consolidation state
    consolidation_state: ConsolidationState = ConsolidationState.NEW
    consolidated_at: Optional[datetime] = None

    # Graph relationships
    connections: Dict[str, float] = field(default_factory=dict)  # memory_id -> strength

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = "conversation"
    tags: Set[str] = field(default_factory=set)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type.value,
            "tier": self.tier.value,
            "importance": self.importance,
            "access_count": self.access_count,
            "created_at": self.created_at.isoformat(),
            "consolidation_state": self.consolidation_state.value,
            "connections": dict(self.connections),
            "metadata": self.metadata,
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryNode":
        return cls(
            id=data["id"],
            content=data["content"],
            memory_type=MemoryType(data["memory_type"]),
            tier=MemoryTier(data["tier"]),
            importance=data.get("importance", 0.5),
            access_count=data.get("access_count", 0),
            created_at=datetime.fromisoformat(data["created_at"]),
            consolidation_state=ConsolidationState(data.get("consolidation_state", "new")),
            connections=data.get("connections", {}),
            metadata=data.get("metadata", {}),
            tags=set(data.get("tags", [])),
        )

    def compute_recall_probability(self, current_time: Optional[datetime] = None) -> float:
        """
        Compute recall probability using the forgetting curve.
        Based on Ebbinghaus forgetting curve with modifications for importance and access.
        """
        if current_time is None:
            current_time = datetime.now()

        # Time since creation (in hours)
        age_hours = (current_time - self.created_at).total_seconds() / 3600

        # Base decay rate (higher = faster forgetting)
        base_decay = 0.1

        # Modify decay based on importance (higher importance = slower decay)
        decay_rate = base_decay * (1 - self.importance * 0.5)

        # Modify based on access count (more access = better retention)
        retention_boost = min(self.access_count * 0.1, 0.5)

        # Forgetting curve: R = e^(-t/S) where S is stability
        stability = (1 + retention_boost) / decay_rate
        recall_prob = math.exp(-age_hours / stability)

        # Boost for recently accessed memories
        if self.last_accessed:
            hours_since_access = (current_time - self.last_accessed).total_seconds() / 3600
            recency_boost = math.exp(-hours_since_access / 24) * 0.3
            recall_prob = min(1.0, recall_prob + recency_boost)

        return recall_prob


@dataclass
class MemoryEdge:
    """
    Edge in the memory graph representing relationships.
    """
    source_id: str
    target_id: str
    relationship: str
    strength: float = 1.0
    bidirectional: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """Result from memory retrieval."""
    memory: MemoryNode
    score: float
    retrieval_method: str
    explanation: str = ""


# =============================================================================
# Embedding Provider
# =============================================================================

class EmbeddingProvider(Protocol):
    """Protocol for embedding providers."""

    async def embed(self, text: str) -> List[float]:
        """Generate embedding for text."""
        ...

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        ...


class DefaultEmbeddingProvider:
    """Default embedding provider with fallback support."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self._dimension = 384
        self._initialized = False

    async def initialize(self) -> None:
        if self._initialized:
            return

        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            self._dimension = self._model.get_sentence_embedding_dimension()
            logger.info(f"Loaded embedding model: {self.model_name}, dim={self._dimension}")
        except ImportError:
            logger.warning("sentence-transformers not available, using hash-based fallback")

        self._initialized = True

    async def embed(self, text: str) -> List[float]:
        if not self._initialized:
            await self.initialize()

        if self._model is not None:
            embedding = self._model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        else:
            return self._hash_embedding(text)

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not self._initialized:
            await self.initialize()

        if self._model is not None:
            embeddings = self._model.encode(texts, convert_to_numpy=True)
            return embeddings.tolist()
        else:
            return [self._hash_embedding(t) for t in texts]

    def _hash_embedding(self, text: str) -> List[float]:
        """Generate pseudo-embedding using hashing."""
        embedding = []
        for i in range(self._dimension):
            h = hashlib.md5(f"{text}_{i}".encode()).hexdigest()
            value = (int(h[:8], 16) / (2**32)) * 2 - 1
            embedding.append(value)
        return embedding

    @property
    def dimension(self) -> int:
        return self._dimension


# =============================================================================
# Vector Index for Semantic Search
# =============================================================================

class VectorIndex:
    """
    HNSW-based vector index for efficient similarity search.
    Falls back to brute force if hnswlib not available.
    """

    def __init__(self, dimension: int, max_elements: int = 100000):
        self.dimension = dimension
        self.max_elements = max_elements
        self._index = None
        self._id_to_idx: Dict[str, int] = {}
        self._idx_to_id: Dict[int, str] = {}
        self._next_idx = 0
        self._vectors: Dict[str, np.ndarray] = {}  # Fallback storage
        self._use_hnsw = False

    async def initialize(self) -> None:
        try:
            import hnswlib
            self._index = hnswlib.Index(space='cosine', dim=self.dimension)
            self._index.init_index(max_elements=self.max_elements, ef_construction=200, M=16)
            self._index.set_ef(50)
            self._use_hnsw = True
            logger.info(f"Initialized HNSW index with dimension {self.dimension}")
        except ImportError:
            logger.warning("hnswlib not available, using brute-force search")
            self._use_hnsw = False

    async def add(self, memory_id: str, vector: List[float]) -> None:
        """Add a vector to the index."""
        np_vector = np.array(vector, dtype=np.float32)

        if self._use_hnsw and self._index is not None:
            idx = self._next_idx
            self._id_to_idx[memory_id] = idx
            self._idx_to_id[idx] = memory_id
            self._index.add_items(np_vector.reshape(1, -1), np.array([idx]))
            self._next_idx += 1
        else:
            self._vectors[memory_id] = np_vector

    async def search(
        self,
        query_vector: List[float],
        k: int = 10,
    ) -> List[Tuple[str, float]]:
        """Search for nearest neighbors."""
        np_query = np.array(query_vector, dtype=np.float32)

        if self._use_hnsw and self._index is not None and self._next_idx > 0:
            labels, distances = self._index.knn_query(np_query.reshape(1, -1), k=min(k, self._next_idx))
            results = []
            for idx, dist in zip(labels[0], distances[0]):
                if idx in self._idx_to_id:
                    # Convert cosine distance to similarity
                    similarity = 1 - dist
                    results.append((self._idx_to_id[idx], similarity))
            return results
        else:
            # Brute force search
            results = []
            for memory_id, vector in self._vectors.items():
                similarity = self._cosine_similarity(np_query, vector)
                results.append((memory_id, similarity))
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:k]

    async def remove(self, memory_id: str) -> None:
        """Remove a vector from the index."""
        if memory_id in self._vectors:
            del self._vectors[memory_id]
        # Note: HNSW doesn't support removal, would need to rebuild

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))


# =============================================================================
# Knowledge Graph
# =============================================================================

class KnowledgeGraph:
    """
    Graph-based knowledge representation for memory relationships.
    Supports semantic relationships, temporal connections, and causal links.
    """

    RELATIONSHIP_TYPES = {
        "related_to": 1.0,
        "caused_by": 0.9,
        "leads_to": 0.9,
        "similar_to": 0.8,
        "contradicts": 0.7,
        "extends": 0.8,
        "references": 0.6,
        "temporal_before": 0.5,
        "temporal_after": 0.5,
        "part_of": 0.8,
        "instance_of": 0.7,
    }

    def __init__(self):
        self._nodes: Dict[str, MemoryNode] = {}
        self._edges: Dict[str, List[MemoryEdge]] = defaultdict(list)
        self._reverse_edges: Dict[str, List[MemoryEdge]] = defaultdict(list)
        self._type_index: Dict[MemoryType, Set[str]] = defaultdict(set)
        self._tag_index: Dict[str, Set[str]] = defaultdict(set)

    def add_node(self, node: MemoryNode) -> None:
        """Add a memory node to the graph."""
        self._nodes[node.id] = node
        self._type_index[node.memory_type].add(node.id)
        for tag in node.tags:
            self._tag_index[tag].add(node.id)

    def get_node(self, node_id: str) -> Optional[MemoryNode]:
        """Get a node by ID."""
        return self._nodes.get(node_id)

    def remove_node(self, node_id: str) -> None:
        """Remove a node and its edges."""
        if node_id in self._nodes:
            node = self._nodes[node_id]
            self._type_index[node.memory_type].discard(node_id)
            for tag in node.tags:
                self._tag_index[tag].discard(node_id)
            del self._nodes[node_id]

        # Remove edges
        if node_id in self._edges:
            del self._edges[node_id]
        if node_id in self._reverse_edges:
            del self._reverse_edges[node_id]

    def add_edge(self, edge: MemoryEdge) -> None:
        """Add an edge between nodes."""
        self._edges[edge.source_id].append(edge)

        if edge.bidirectional:
            reverse_edge = MemoryEdge(
                source_id=edge.target_id,
                target_id=edge.source_id,
                relationship=edge.relationship,
                strength=edge.strength,
                bidirectional=False,
            )
            self._reverse_edges[edge.target_id].append(reverse_edge)

    def get_neighbors(
        self,
        node_id: str,
        relationship: Optional[str] = None,
        max_depth: int = 1,
    ) -> List[Tuple[MemoryNode, float, int]]:
        """
        Get neighboring nodes with optional relationship filter.
        Returns: List of (node, strength, depth)
        """
        if node_id not in self._nodes:
            return []

        visited: Set[str] = {node_id}
        result: List[Tuple[MemoryNode, float, int]] = []
        queue: List[Tuple[str, float, int]] = [(node_id, 1.0, 0)]

        while queue:
            current_id, current_strength, depth = queue.pop(0)

            if depth >= max_depth:
                continue

            edges = self._edges.get(current_id, []) + self._reverse_edges.get(current_id, [])

            for edge in edges:
                target_id = edge.target_id if edge.source_id == current_id else edge.source_id

                if target_id in visited:
                    continue

                if relationship and edge.relationship != relationship:
                    continue

                visited.add(target_id)

                if target_id in self._nodes:
                    combined_strength = current_strength * edge.strength
                    result.append((self._nodes[target_id], combined_strength, depth + 1))
                    queue.append((target_id, combined_strength, depth + 1))

        return result

    def find_path(
        self,
        source_id: str,
        target_id: str,
        max_depth: int = 5,
    ) -> Optional[List[str]]:
        """Find shortest path between two nodes."""
        if source_id not in self._nodes or target_id not in self._nodes:
            return None

        visited: Set[str] = {source_id}
        queue: List[Tuple[str, List[str]]] = [(source_id, [source_id])]

        while queue:
            current_id, path = queue.pop(0)

            if len(path) > max_depth:
                continue

            if current_id == target_id:
                return path

            edges = self._edges.get(current_id, []) + self._reverse_edges.get(current_id, [])

            for edge in edges:
                target = edge.target_id if edge.source_id == current_id else edge.source_id

                if target not in visited:
                    visited.add(target)
                    queue.append((target, path + [target]))

        return None

    def get_by_type(self, memory_type: MemoryType) -> List[MemoryNode]:
        """Get all nodes of a specific type."""
        return [self._nodes[nid] for nid in self._type_index[memory_type] if nid in self._nodes]

    def get_by_tag(self, tag: str) -> List[MemoryNode]:
        """Get all nodes with a specific tag."""
        return [self._nodes[nid] for nid in self._tag_index[tag] if nid in self._nodes]

    def compute_centrality(self, node_id: str) -> float:
        """Compute node centrality (PageRank-like)."""
        if node_id not in self._nodes:
            return 0.0

        # Simple degree centrality
        outgoing = len(self._edges.get(node_id, []))
        incoming = len(self._reverse_edges.get(node_id, []))
        total_edges = sum(len(edges) for edges in self._edges.values())

        if total_edges == 0:
            return 0.0

        return (outgoing + incoming) / total_edges

    def get_subgraph(
        self,
        center_id: str,
        radius: int = 2,
    ) -> Tuple[List[MemoryNode], List[MemoryEdge]]:
        """Extract a subgraph around a center node."""
        neighbors = self.get_neighbors(center_id, max_depth=radius)
        node_ids = {center_id} | {n.id for n, _, _ in neighbors}

        nodes = [self._nodes[nid] for nid in node_ids if nid in self._nodes]
        edges = []

        for nid in node_ids:
            for edge in self._edges.get(nid, []):
                if edge.target_id in node_ids:
                    edges.append(edge)

        return nodes, edges


# =============================================================================
# Memory Consolidation Engine
# =============================================================================

class MemoryConsolidator:
    """
    Consolidates memories from working/short-term to long-term storage.
    Inspired by human memory consolidation during sleep.

    Process:
    1. Identify important memories for consolidation
    2. Extract semantic patterns and create abstractions
    3. Strengthen connections between related memories
    4. Prune low-importance, redundant memories
    """

    def __init__(
        self,
        graph: KnowledgeGraph,
        embedding_provider: EmbeddingProvider,
        consolidation_threshold: float = 0.6,
        prune_threshold: float = 0.2,
    ):
        self.graph = graph
        self.embedding_provider = embedding_provider
        self.consolidation_threshold = consolidation_threshold
        self.prune_threshold = prune_threshold
        self._consolidation_task: Optional[asyncio.Task] = None
        self._running = False

    async def start_background_consolidation(
        self,
        interval_seconds: int = 3600,  # 1 hour
    ) -> None:
        """Start background consolidation process."""
        self._running = True
        self._consolidation_task = asyncio.create_task(
            self._consolidation_loop(interval_seconds)
        )
        logger.info("Started background memory consolidation")

    async def stop(self) -> None:
        """Stop background consolidation."""
        self._running = False
        if self._consolidation_task:
            self._consolidation_task.cancel()
            try:
                await self._consolidation_task
            except asyncio.CancelledError:
                pass

    async def _consolidation_loop(self, interval: int) -> None:
        """Background consolidation loop."""
        while self._running:
            try:
                await asyncio.sleep(interval)
                await self.consolidate()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Consolidation error: {e}")

    async def consolidate(self) -> Dict[str, Any]:
        """
        Run full consolidation cycle.
        Returns statistics about the consolidation.
        """
        stats = {
            "promoted": 0,
            "pruned": 0,
            "connections_created": 0,
            "abstractions_created": 0,
        }

        logger.info("Starting memory consolidation cycle")
        start_time = time.time()

        # 1. Promote important short-term memories to long-term
        stats["promoted"] = await self._promote_memories()

        # 2. Strengthen connections between related memories
        stats["connections_created"] = await self._strengthen_connections()

        # 3. Create abstractions from similar memories
        stats["abstractions_created"] = await self._create_abstractions()

        # 4. Prune low-importance memories
        stats["pruned"] = await self._prune_memories()

        elapsed = time.time() - start_time
        logger.info(
            "Memory consolidation complete",
            elapsed_seconds=f"{elapsed:.2f}",
            **stats,
        )

        return stats

    async def _promote_memories(self) -> int:
        """Promote important short-term memories to long-term."""
        promoted = 0

        for memory_type in [MemoryType.EPISODIC, MemoryType.SEMANTIC]:
            memories = self.graph.get_by_type(memory_type)

            for memory in memories:
                if memory.tier != MemoryTier.SHORT_TERM:
                    continue

                recall_prob = memory.compute_recall_probability()
                should_promote = (
                    memory.importance >= self.consolidation_threshold or
                    (recall_prob >= 0.7 and memory.access_count >= 3) or
                    memory.access_count >= 5
                )

                if should_promote:
                    memory.tier = MemoryTier.LONG_TERM
                    memory.consolidation_state = ConsolidationState.CONSOLIDATED
                    memory.consolidated_at = datetime.now()
                    promoted += 1

        return promoted

    async def _strengthen_connections(self) -> int:
        """Find and strengthen connections between related memories."""
        connections_created = 0

        # Get all long-term memories with embeddings
        memories = [
            m for m in self.graph._nodes.values()
            if m.tier == MemoryTier.LONG_TERM and m.embedding
        ]

        if len(memories) < 2:
            return 0

        # Find similar pairs
        for i, mem_a in enumerate(memories):
            for mem_b in memories[i+1:]:
                if mem_b.id in mem_a.connections:
                    continue

                similarity = self._cosine_similarity(
                    mem_a.embedding.vector,
                    mem_b.embedding.vector,
                )

                if similarity >= 0.7:
                    # Create bidirectional connection
                    edge = MemoryEdge(
                        source_id=mem_a.id,
                        target_id=mem_b.id,
                        relationship="similar_to",
                        strength=similarity,
                        bidirectional=True,
                    )
                    self.graph.add_edge(edge)

                    mem_a.connections[mem_b.id] = similarity
                    mem_b.connections[mem_a.id] = similarity
                    connections_created += 1

        return connections_created

    async def _create_abstractions(self) -> int:
        """Create abstract semantic memories from clusters of similar episodic memories."""
        abstractions_created = 0

        # Group episodic memories by similarity
        episodic = [
            m for m in self.graph.get_by_type(MemoryType.EPISODIC)
            if m.embedding and m.tier == MemoryTier.LONG_TERM
        ]

        if len(episodic) < 3:
            return 0

        # Simple clustering: find groups with high mutual similarity
        clusters: List[List[MemoryNode]] = []
        used = set()

        for memory in episodic:
            if memory.id in used:
                continue

            cluster = [memory]
            used.add(memory.id)

            for other in episodic:
                if other.id in used:
                    continue

                avg_similarity = np.mean([
                    self._cosine_similarity(m.embedding.vector, other.embedding.vector)
                    for m in cluster
                ])

                if avg_similarity >= 0.75:
                    cluster.append(other)
                    used.add(other.id)

            if len(cluster) >= 3:
                clusters.append(cluster)

        # Create abstractions for each cluster
        for cluster in clusters:
            # Combine content into abstraction
            contents = [m.content for m in cluster]
            abstract_content = f"[Abstracted from {len(cluster)} memories]\n" + \
                             "\n---\n".join(contents[:3])  # Limit for size

            # Average embedding
            avg_embedding = np.mean(
                [m.embedding.vector for m in cluster],
                axis=0,
            ).tolist()

            # Create semantic memory
            abstraction = MemoryNode(
                id=f"abstract_{uuid.uuid4().hex[:12]}",
                content=abstract_content,
                memory_type=MemoryType.SEMANTIC,
                tier=MemoryTier.LONG_TERM,
                embedding=MemoryEmbedding(
                    vector=avg_embedding,
                    model="averaged",
                    dimension=len(avg_embedding),
                ),
                importance=np.mean([m.importance for m in cluster]),
                consolidation_state=ConsolidationState.CONSOLIDATED,
                consolidated_at=datetime.now(),
                metadata={
                    "abstracted_from": [m.id for m in cluster],
                    "cluster_size": len(cluster),
                },
                tags=set().union(*[m.tags for m in cluster]),
            )

            self.graph.add_node(abstraction)

            # Link abstraction to source memories
            for memory in cluster:
                edge = MemoryEdge(
                    source_id=abstraction.id,
                    target_id=memory.id,
                    relationship="instance_of",
                    strength=0.9,
                )
                self.graph.add_edge(edge)

            abstractions_created += 1

        return abstractions_created

    async def _prune_memories(self) -> int:
        """Prune low-importance, decayed memories."""
        pruned = 0
        to_remove = []

        for memory_id, memory in self.graph._nodes.items():
            # Don't prune long-term or recently accessed
            if memory.tier == MemoryTier.LONG_TERM:
                continue

            recall_prob = memory.compute_recall_probability()

            # Prune if:
            # - Low importance AND low recall probability
            # - Never accessed and old
            should_prune = (
                (memory.importance < self.prune_threshold and recall_prob < 0.3) or
                (memory.access_count == 0 and recall_prob < 0.1)
            )

            if should_prune:
                to_remove.append(memory_id)

        for memory_id in to_remove:
            self.graph.remove_node(memory_id)
            pruned += 1

        return pruned

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        a_np = np.array(a)
        b_np = np.array(b)
        norm_a = np.linalg.norm(a_np)
        norm_b = np.linalg.norm(b_np)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a_np, b_np) / (norm_a * norm_b))


# =============================================================================
# Hierarchical Memory Store
# =============================================================================

class HierarchicalMemoryStore:
    """
    Three-tier hierarchical memory store:
    - Working Memory: Limited capacity (~7 items), fast access, rapid decay
    - Short-term Memory: Larger capacity, moderate decay
    - Long-term Memory: Persistent, consolidated memories
    """

    def __init__(
        self,
        working_capacity: int = 7,
        short_term_capacity: int = 100,
        embedding_provider: Optional[EmbeddingProvider] = None,
    ):
        self.working_capacity = working_capacity
        self.short_term_capacity = short_term_capacity

        # Memory stores by tier
        self._working: List[MemoryNode] = []  # Stack-like, FIFO eviction
        self._short_term: Dict[str, MemoryNode] = {}
        self._long_term: Dict[str, MemoryNode] = {}

        # Knowledge graph for all memories
        self.graph = KnowledgeGraph()

        # Embedding provider
        self.embedding_provider = embedding_provider or DefaultEmbeddingProvider()

        # Vector index for semantic search
        self._vector_index: Optional[VectorIndex] = None

        # Consolidator
        self._consolidator: Optional[MemoryConsolidator] = None

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the memory store."""
        if self._initialized:
            return

        if hasattr(self.embedding_provider, 'initialize'):
            await self.embedding_provider.initialize()

        dim = getattr(self.embedding_provider, 'dimension', 384)
        self._vector_index = VectorIndex(dimension=dim)
        await self._vector_index.initialize()

        self._consolidator = MemoryConsolidator(
            graph=self.graph,
            embedding_provider=self.embedding_provider,
        )

        self._initialized = True
        logger.info("Hierarchical memory store initialized")

    async def shutdown(self) -> None:
        """Shutdown the memory store."""
        if self._consolidator:
            await self._consolidator.stop()
        self._initialized = False

    async def store(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.EPISODIC,
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[Set[str]] = None,
        skip_working: bool = False,
    ) -> MemoryNode:
        """
        Store a new memory.

        New memories enter through working memory (unless skip_working=True),
        then automatically flow to short-term based on importance.
        """
        if not self._initialized:
            await self.initialize()

        # Generate embedding
        embedding_vector = await self.embedding_provider.embed(content)
        embedding = MemoryEmbedding(
            vector=embedding_vector,
            model=getattr(self.embedding_provider, 'model_name', 'unknown'),
            dimension=len(embedding_vector),
        )

        # Create memory node
        memory = MemoryNode(
            id=f"mem_{uuid.uuid4().hex[:12]}",
            content=content,
            memory_type=memory_type,
            tier=MemoryTier.SHORT_TERM if skip_working else MemoryTier.WORKING,
            embedding=embedding,
            importance=importance,
            metadata=metadata or {},
            tags=tags or set(),
        )

        # Add to graph
        self.graph.add_node(memory)

        # Add to vector index
        await self._vector_index.add(memory.id, embedding_vector)

        # Add to appropriate tier
        if skip_working:
            self._short_term[memory.id] = memory
        else:
            self._add_to_working(memory)

        logger.debug(
            "Stored memory",
            memory_id=memory.id[:8],
            type=memory_type.value,
            importance=f"{importance:.2f}",
        )

        return memory

    def _add_to_working(self, memory: MemoryNode) -> None:
        """Add memory to working memory with capacity management."""
        self._working.append(memory)

        # Evict oldest if over capacity
        while len(self._working) > self.working_capacity:
            evicted = self._working.pop(0)

            # Move to short-term if important enough
            if evicted.importance >= 0.4 or evicted.access_count > 0:
                evicted.tier = MemoryTier.SHORT_TERM
                self._short_term[evicted.id] = evicted
            else:
                # Just remove from graph
                self.graph.remove_node(evicted.id)

    async def retrieve(
        self,
        query: str,
        limit: int = 5,
        memory_types: Optional[List[MemoryType]] = None,
        tiers: Optional[List[MemoryTier]] = None,
        min_similarity: float = 0.5,
        include_graph_neighbors: bool = True,
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant memories using RAG.

        Combines:
        1. Semantic vector search
        2. Graph-based expansion
        3. Recency/importance reranking
        """
        if not self._initialized:
            await self.initialize()

        results: List[RetrievalResult] = []

        # 1. Semantic search
        query_embedding = await self.embedding_provider.embed(query)
        vector_results = await self._vector_index.search(query_embedding, k=limit * 2)

        for memory_id, similarity in vector_results:
            if similarity < min_similarity:
                continue

            memory = self.graph.get_node(memory_id)
            if not memory:
                continue

            # Filter by type and tier
            if memory_types and memory.memory_type not in memory_types:
                continue
            if tiers and memory.tier not in tiers:
                continue

            results.append(RetrievalResult(
                memory=memory,
                score=similarity,
                retrieval_method="semantic_search",
            ))

        # 2. Graph-based expansion
        if include_graph_neighbors and results:
            top_result = results[0]
            neighbors = self.graph.get_neighbors(top_result.memory.id, max_depth=2)

            existing_ids = {r.memory.id for r in results}

            for neighbor, strength, depth in neighbors:
                if neighbor.id in existing_ids:
                    continue

                # Filter by type and tier
                if memory_types and neighbor.memory_type not in memory_types:
                    continue
                if tiers and neighbor.tier not in tiers:
                    continue

                # Compute score based on connection strength and depth
                neighbor_score = top_result.score * strength * (0.7 ** depth)

                if neighbor_score >= min_similarity:
                    results.append(RetrievalResult(
                        memory=neighbor,
                        score=neighbor_score,
                        retrieval_method="graph_expansion",
                        explanation=f"Connected via {depth}-hop path",
                    ))

        # 3. Rerank by importance and recency
        for result in results:
            memory = result.memory

            # Update access
            memory.access_count += 1
            memory.last_accessed = datetime.now()

            # Compute final score
            recall_prob = memory.compute_recall_probability()
            importance_boost = memory.importance * 0.2
            centrality = self.graph.compute_centrality(memory.id)

            result.score = (
                result.score * 0.5 +
                recall_prob * 0.2 +
                importance_boost +
                centrality * 0.1
            )

        # Sort by final score and limit
        results.sort(key=lambda r: r.score, reverse=True)

        return results[:limit]

    async def retrieve_by_type(
        self,
        memory_type: MemoryType,
        limit: int = 10,
    ) -> List[MemoryNode]:
        """Retrieve memories by type."""
        memories = self.graph.get_by_type(memory_type)
        memories.sort(key=lambda m: m.importance, reverse=True)
        return memories[:limit]

    async def get_working_memory(self) -> List[MemoryNode]:
        """Get current working memory contents."""
        return list(self._working)

    async def clear_working_memory(self) -> None:
        """Clear working memory, optionally saving to short-term."""
        for memory in self._working:
            if memory.importance >= 0.4:
                memory.tier = MemoryTier.SHORT_TERM
                self._short_term[memory.id] = memory
            else:
                self.graph.remove_node(memory.id)

        self._working.clear()

    async def forget(self, memory_id: str) -> bool:
        """Forget a specific memory."""
        memory = self.graph.get_node(memory_id)
        if not memory:
            return False

        # Remove from tier-specific storage
        if memory.tier == MemoryTier.WORKING:
            self._working = [m for m in self._working if m.id != memory_id]
        elif memory.tier == MemoryTier.SHORT_TERM:
            self._short_term.pop(memory_id, None)
        else:
            self._long_term.pop(memory_id, None)

        # Remove from graph and index
        self.graph.remove_node(memory_id)
        await self._vector_index.remove(memory_id)

        return True

    async def consolidate(self) -> Dict[str, Any]:
        """Trigger manual consolidation."""
        if self._consolidator:
            return await self._consolidator.consolidate()
        return {}

    def get_stats(self) -> Dict[str, Any]:
        """Get memory store statistics."""
        return {
            "working_count": len(self._working),
            "working_capacity": self.working_capacity,
            "short_term_count": len(self._short_term),
            "short_term_capacity": self.short_term_capacity,
            "long_term_count": len(self._long_term),
            "total_nodes": len(self.graph._nodes),
            "total_edges": sum(len(edges) for edges in self.graph._edges.values()),
            "type_distribution": {
                t.value: len(self.graph._type_index[t])
                for t in MemoryType
            },
        }


# =============================================================================
# SOTA Memory Integrator
# =============================================================================

class MemoryIntegrator:
    """
    SOTA Memory Integrator for conversation system.

    Features:
    - RAG (Retrieval Augmented Generation)
    - Hierarchical memory (Working → Short-term → Long-term)
    - Graph-based knowledge representation
    - Memory consolidation
    - Importance-weighted storage and retrieval
    """

    def __init__(
        self,
        memory_system: Optional[Any] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
        default_importance: float = 0.5,
        min_importance_threshold: float = 0.3,
    ):
        # Legacy memory system integration
        self._legacy_memory = memory_system

        # SOTA hierarchical memory
        self._hierarchical_store = HierarchicalMemoryStore(
            embedding_provider=embedding_provider,
        )

        self.default_importance = default_importance
        self.min_importance_threshold = min_importance_threshold

        # Deduplication
        self._recent_hashes: Set[str] = set()
        self._max_recent_hashes = 1000

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the memory integrator."""
        if self._initialized:
            return

        await self._hierarchical_store.initialize()

        if self._legacy_memory and hasattr(self._legacy_memory, "initialize"):
            await self._legacy_memory.initialize()

        self._initialized = True
        logger.info("SOTA Memory integrator initialized")

    async def shutdown(self) -> None:
        """Shutdown the memory integrator."""
        await self._hierarchical_store.shutdown()
        self._recent_hashes.clear()
        self._initialized = False

    async def retrieve_relevant(
        self,
        query: str,
        limit: int = 5,
        memory_types: Optional[List[str]] = None,
        min_similarity: float = 0.5,
        include_graph_context: bool = True,
    ) -> List[RetrievalResult]:
        """
        Retrieve memories relevant to a query using RAG.

        Args:
            query: Search query
            limit: Maximum memories to return
            memory_types: Filter by memory types
            min_similarity: Minimum similarity threshold
            include_graph_context: Include graph-connected memories

        Returns:
            List of RetrievalResult with memories and scores
        """
        if not self._initialized:
            await self.initialize()

        # Convert string types to enum
        types = None
        if memory_types:
            types = [MemoryType(t) for t in memory_types if t in [m.value for m in MemoryType]]

        # Retrieve from hierarchical store
        results = await self._hierarchical_store.retrieve(
            query=query,
            limit=limit,
            memory_types=types,
            min_similarity=min_similarity,
            include_graph_neighbors=include_graph_context,
        )

        # Also check legacy memory system
        if self._legacy_memory:
            try:
                if hasattr(self._legacy_memory, "search"):
                    legacy_results = await self._legacy_memory.search(
                        query=query,
                        limit=limit,
                        memory_types=memory_types,
                    )
                    # Convert legacy results (simplified)
                    for lr in legacy_results:
                        if hasattr(lr, "content"):
                            # Check for duplicates
                            is_dup = any(r.memory.content == lr.content for r in results)
                            if not is_dup:
                                # Create a pseudo MemoryNode
                                memory = MemoryNode(
                                    id=getattr(lr, "id", f"legacy_{uuid.uuid4().hex[:8]}"),
                                    content=lr.content,
                                    memory_type=MemoryType.EPISODIC,
                                    tier=MemoryTier.LONG_TERM,
                                    importance=getattr(lr, "importance", 0.5),
                                )
                                results.append(RetrievalResult(
                                    memory=memory,
                                    score=getattr(lr, "similarity", 0.6),
                                    retrieval_method="legacy_system",
                                ))
            except Exception as e:
                logger.warning(f"Legacy memory retrieval failed: {e}")

        # Sort by score and limit
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]

    async def store_interaction(
        self,
        user_message: Message,
        assistant_message: Message,
        importance: Optional[float] = None,
        conversation_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        Store a conversation interaction as episodic memory.

        Returns memory ID if stored, None if skipped.
        """
        if not self._initialized:
            await self.initialize()

        try:
            user_text = user_message.get_text()
            assistant_text = assistant_message.get_text()

            content = f"User: {user_text}\nAssistant: {assistant_text[:1000]}"

            # Deduplication check
            content_hash = hashlib.md5(content.encode()).hexdigest()
            if content_hash in self._recent_hashes:
                logger.debug("Skipping duplicate memory")
                return None

            # Calculate importance if not provided
            if importance is None:
                importance = self._calculate_importance(user_message, assistant_message)

            # Skip low-importance
            if importance < self.min_importance_threshold:
                logger.debug(f"Skipping low-importance memory: {importance:.2f}")
                return None

            # Build metadata
            metadata = {
                "type": "conversation",
                "user_message_id": user_message.id,
                "assistant_message_id": assistant_message.id,
                "timestamp": datetime.now().isoformat(),
            }

            if conversation_id:
                metadata["conversation_id"] = conversation_id

            if assistant_message.has_tool_use():
                metadata["has_tool_use"] = True
                tool_names = [t.name for t in assistant_message.get_tool_uses()]
                metadata["tools_used"] = tool_names

            # Build tags
            tags = {"conversation", "interaction"}
            if assistant_message.has_tool_use():
                tags.add("tool_use")

            # Store in hierarchical memory
            memory = await self._hierarchical_store.store(
                content=content,
                memory_type=MemoryType.EPISODIC,
                importance=importance,
                metadata=metadata,
                tags=tags,
            )

            # Update deduplication cache
            self._recent_hashes.add(content_hash)
            if len(self._recent_hashes) > self._max_recent_hashes:
                self._recent_hashes.pop()

            # Also store in legacy system if available
            if self._legacy_memory and hasattr(self._legacy_memory, "store"):
                try:
                    await self._legacy_memory.store(
                        content=content,
                        memory_type="episodic",
                        importance=importance,
                        metadata=metadata,
                    )
                except Exception as e:
                    logger.warning(f"Legacy memory storage failed: {e}")

            return memory.id

        except Exception as e:
            logger.error(f"Memory storage error: {e}")
            return None

    async def store_fact(
        self,
        fact: str,
        source: str = "conversation",
        importance: float = 0.7,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Store a semantic fact."""
        if not self._initialized:
            await self.initialize()

        try:
            memory = await self._hierarchical_store.store(
                content=fact,
                memory_type=MemoryType.SEMANTIC,
                importance=importance,
                metadata={
                    "source": source,
                    "timestamp": datetime.now().isoformat(),
                    **(metadata or {}),
                },
                tags={"fact", "semantic", source},
                skip_working=True,  # Facts go directly to short-term
            )

            return memory.id

        except Exception as e:
            logger.error(f"Fact storage error: {e}")
            return None

    async def store_procedure(
        self,
        name: str,
        steps: List[str],
        description: Optional[str] = None,
        importance: float = 0.8,
    ) -> Optional[str]:
        """Store a procedural memory."""
        if not self._initialized:
            await self.initialize()

        try:
            steps_text = "\n".join(f"{i+1}. {step}" for i, step in enumerate(steps))
            content = f"Procedure: {name}\n"
            if description:
                content += f"Description: {description}\n"
            content += f"Steps:\n{steps_text}"

            memory = await self._hierarchical_store.store(
                content=content,
                memory_type=MemoryType.PROCEDURAL,
                importance=importance,
                metadata={
                    "procedure_name": name,
                    "step_count": len(steps),
                    "timestamp": datetime.now().isoformat(),
                },
                tags={"procedure", "how-to", name.lower().replace(" ", "_")},
                skip_working=True,
            )

            return memory.id

        except Exception as e:
            logger.error(f"Procedure storage error: {e}")
            return None

    async def get_working_memory_context(self) -> List[Dict[str, Any]]:
        """
        Get current working memory contents for context injection.
        Returns a list of memory summaries.
        """
        working = await self._hierarchical_store.get_working_memory()

        return [
            {
                "content": m.content[:500],
                "type": m.memory_type.value,
                "importance": m.importance,
                "age_seconds": (datetime.now() - m.created_at).total_seconds(),
            }
            for m in working
        ]

    async def get_conversation_history(
        self,
        conversation_id: str,
        limit: int = 50,
    ) -> List[RetrievalResult]:
        """Retrieve memories from a specific conversation."""
        results = await self.retrieve_relevant(
            query=f"conversation:{conversation_id}",
            limit=limit,
        )

        # Filter by conversation ID
        return [
            r for r in results
            if r.memory.metadata.get("conversation_id") == conversation_id
        ]

    async def forget(self, memory_id: str) -> bool:
        """Forget a specific memory."""
        success = await self._hierarchical_store.forget(memory_id)

        # Also forget in legacy system
        if self._legacy_memory:
            try:
                if hasattr(self._legacy_memory, "forget"):
                    await self._legacy_memory.forget(memory_id)
                elif hasattr(self._legacy_memory, "delete"):
                    await self._legacy_memory.delete(memory_id)
            except Exception as e:
                logger.warning(f"Legacy forget failed: {e}")

        return success

    async def consolidate_memories(self) -> Dict[str, Any]:
        """Trigger memory consolidation."""
        return await self._hierarchical_store.consolidate()

    async def get_knowledge_graph_context(
        self,
        memory_id: str,
        radius: int = 2,
    ) -> Dict[str, Any]:
        """
        Get knowledge graph context around a memory.
        Useful for understanding relationships and context.
        """
        nodes, edges = self._hierarchical_store.graph.get_subgraph(memory_id, radius)

        return {
            "center_memory": memory_id,
            "related_memories": [
                {
                    "id": n.id,
                    "content": n.content[:200],
                    "type": n.memory_type.value,
                    "importance": n.importance,
                }
                for n in nodes if n.id != memory_id
            ],
            "relationships": [
                {
                    "source": e.source_id,
                    "target": e.target_id,
                    "type": e.relationship,
                    "strength": e.strength,
                }
                for e in edges
            ],
            "node_count": len(nodes),
            "edge_count": len(edges),
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get memory system statistics."""
        hierarchical_stats = self._hierarchical_store.get_stats()

        return {
            **hierarchical_stats,
            "dedup_cache_size": len(self._recent_hashes),
            "legacy_system_active": self._legacy_memory is not None,
        }

    def _calculate_importance(
        self,
        user_message: Message,
        assistant_message: Message,
    ) -> float:
        """
        Calculate importance score using multiple signals.
        """
        importance = self.default_importance

        user_text = user_message.get_text()
        assistant_text = assistant_message.get_text()

        # Length signals
        if len(user_text) > 200:
            importance += 0.1
        if len(assistant_text) > 500:
            importance += 0.1

        # Tool usage (indicates task completion)
        if assistant_message.has_tool_use():
            importance += 0.15

        # Question (indicates information need)
        if "?" in user_text:
            importance += 0.05

        # Explicit importance markers
        important_keywords = [
            "important", "remember", "critical", "key", "essential",
            "don't forget", "make sure", "always", "never", "must",
        ]
        if any(kw in user_text.lower() for kw in important_keywords):
            importance += 0.15

        # Learning/teaching signals
        learning_signals = ["learn", "understand", "teach", "explain", "show me how"]
        if any(sig in user_text.lower() for sig in learning_signals):
            importance += 0.1

        # Trivial patterns (reduce importance)
        trivial_patterns = [
            "hello", "hi", "thanks", "thank you", "bye", "goodbye",
            "ok", "okay", "yes", "no", "sure",
        ]
        if user_text.lower().strip() in trivial_patterns:
            importance -= 0.2

        return max(0.0, min(1.0, importance))


# =============================================================================
# Memory Context Enricher
# =============================================================================

class MemoryContextEnricher:
    """
    Enriches conversation context with relevant memories.
    Uses RAG to provide context-aware memory injection.
    """

    def __init__(self, integrator: MemoryIntegrator):
        self.integrator = integrator

    async def enrich_context(
        self,
        user_message: str,
        conversation_history: List[Message],
        max_memories: int = 5,
        include_working_memory: bool = True,
    ) -> Dict[str, Any]:
        """
        Enrich context with relevant memories.

        Returns:
            Dict with memories, working memory, and context summary
        """
        result: Dict[str, Any] = {
            "memories": [],
            "working_memory": [],
            "context_summary": "",
            "memory_count": 0,
            "retrieval_stats": {},
        }

        # Retrieve relevant memories via RAG
        retrieval_results = await self.integrator.retrieve_relevant(
            query=user_message,
            limit=max_memories,
            include_graph_context=True,
        )

        result["memories"] = retrieval_results
        result["memory_count"] = len(retrieval_results)

        # Include working memory
        if include_working_memory:
            result["working_memory"] = await self.integrator.get_working_memory_context()

        # Generate context summary
        if retrieval_results:
            memory_types = set(r.memory.memory_type.value for r in retrieval_results)
            result["context_summary"] = (
                f"Found {len(retrieval_results)} relevant memories "
                f"({', '.join(memory_types)}) from previous interactions."
            )

            # Add retrieval statistics
            result["retrieval_stats"] = {
                "avg_similarity": np.mean([r.score for r in retrieval_results]),
                "max_similarity": max(r.score for r in retrieval_results),
                "retrieval_methods": list(set(r.retrieval_method for r in retrieval_results)),
                "types_found": list(memory_types),
            }

        return result

    def format_memories_for_prompt(
        self,
        retrieval_results: List[RetrievalResult],
        max_tokens: int = 1000,
    ) -> str:
        """
        Format retrieved memories for inclusion in prompt.

        Returns formatted string suitable for system prompt injection.
        """
        if not retrieval_results:
            return ""

        lines = ["## Relevant Memory Context\n"]
        total_chars = 0
        char_limit = max_tokens * 4  # Rough estimate

        for i, result in enumerate(retrieval_results, 1):
            memory = result.memory

            memory_text = (
                f"### Memory {i} ({memory.memory_type.value}, relevance: {result.score:.2f})\n"
                f"{memory.content[:500]}\n"
            )

            if total_chars + len(memory_text) > char_limit:
                break

            lines.append(memory_text)
            total_chars += len(memory_text)

        return "\n".join(lines)
