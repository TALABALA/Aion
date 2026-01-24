"""
AION Multi-Agent Memory Systems

State-of-the-art memory architecture for multi-agent systems including:
- Vector memory with semantic search
- Episodic memory for experience replay
- Semantic memory for knowledge graphs
- Working memory with attention mechanisms
- RAG (Retrieval-Augmented Generation) integration
"""

from .vector_store import VectorStore, VectorEntry, SimilarityMetric
from .episodic import EpisodicMemory, Episode, EpisodeType
from .semantic import SemanticMemory, Concept, Relation, KnowledgeTriple
from .working import WorkingMemory, MemorySlot, AttentionWeight
from .rag import RAGEngine, RAGConfig, RetrievalResult
from .consolidation import MemoryConsolidator, ConsolidationStrategy
from .manager import AgentMemoryManager

__all__ = [
    # Vector Store
    "VectorStore",
    "VectorEntry",
    "SimilarityMetric",
    # Episodic Memory
    "EpisodicMemory",
    "Episode",
    "EpisodeType",
    # Semantic Memory
    "SemanticMemory",
    "Concept",
    "Relation",
    "KnowledgeTriple",
    # Working Memory
    "WorkingMemory",
    "MemorySlot",
    "AttentionWeight",
    # RAG
    "RAGEngine",
    "RAGConfig",
    "RetrievalResult",
    # Consolidation
    "MemoryConsolidator",
    "ConsolidationStrategy",
    # Manager
    "AgentMemoryManager",
]
