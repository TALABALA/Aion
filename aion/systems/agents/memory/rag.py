"""
RAG (Retrieval-Augmented Generation) Engine

Advanced retrieval system for augmenting agent responses with
relevant context from memory systems.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional, Callable
import json

import structlog

from .vector_store import VectorStore, SearchResult, SimilarityMetric
from .episodic import EpisodicMemory, Episode
from .semantic import SemanticMemory, Concept, Relation

logger = structlog.get_logger()


class RetrievalStrategy(Enum):
    """Retrieval strategies for RAG."""

    DENSE = "dense"  # Vector similarity only
    SPARSE = "sparse"  # Keyword/BM25 based
    HYBRID = "hybrid"  # Combination of dense and sparse
    MULTI_HOP = "multi_hop"  # Follow relations in knowledge graph
    HIERARCHICAL = "hierarchical"  # Query at multiple granularities


class RerankingMethod(Enum):
    """Methods for reranking retrieved results."""

    NONE = "none"
    CROSS_ENCODER = "cross_encoder"
    RECIPROCAL_RANK_FUSION = "rrf"
    LEARNED = "learned"


@dataclass
class RAGConfig:
    """Configuration for RAG engine."""

    # Retrieval settings
    retrieval_strategy: RetrievalStrategy = RetrievalStrategy.HYBRID
    top_k: int = 10
    similarity_threshold: float = 0.5

    # Source weights
    vector_weight: float = 0.4
    episodic_weight: float = 0.3
    semantic_weight: float = 0.3

    # Reranking
    reranking_method: RerankingMethod = RerankingMethod.RECIPROCAL_RANK_FUSION
    rerank_top_k: int = 5

    # Multi-hop settings
    max_hops: int = 2
    hop_decay: float = 0.7  # Relevance decay per hop

    # Context window
    max_context_length: int = 4000
    include_metadata: bool = True

    # Caching
    cache_enabled: bool = True
    cache_ttl_seconds: int = 300

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "retrieval_strategy": self.retrieval_strategy.value,
            "top_k": self.top_k,
            "similarity_threshold": self.similarity_threshold,
            "vector_weight": self.vector_weight,
            "episodic_weight": self.episodic_weight,
            "semantic_weight": self.semantic_weight,
            "reranking_method": self.reranking_method.value,
            "rerank_top_k": self.rerank_top_k,
            "max_hops": self.max_hops,
            "hop_decay": self.hop_decay,
            "max_context_length": self.max_context_length,
            "include_metadata": self.include_metadata,
            "cache_enabled": self.cache_enabled,
            "cache_ttl_seconds": self.cache_ttl_seconds,
        }


@dataclass
class RetrievalResult:
    """A single retrieval result."""

    id: str
    content: str
    source: str  # "vector", "episodic", "semantic"
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    hop_distance: int = 0  # For multi-hop retrieval

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "source": self.source,
            "score": self.score,
            "metadata": self.metadata,
            "hop_distance": self.hop_distance,
        }


@dataclass
class RAGContext:
    """Context assembled by RAG for generation."""

    query: str
    results: list[RetrievalResult]
    formatted_context: str
    total_tokens_estimate: int
    retrieval_time_ms: float
    sources_used: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query": self.query,
            "results": [r.to_dict() for r in self.results],
            "formatted_context": self.formatted_context,
            "total_tokens_estimate": self.total_tokens_estimate,
            "retrieval_time_ms": self.retrieval_time_ms,
            "sources_used": self.sources_used,
        }


class RAGEngine:
    """
    Retrieval-Augmented Generation engine.

    Features:
    - Multi-source retrieval (vector, episodic, semantic)
    - Hybrid retrieval strategies
    - Multi-hop reasoning through knowledge graphs
    - Reciprocal rank fusion for result combination
    - Context formatting for LLM consumption
    - Query expansion and decomposition
    - Result caching
    """

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        episodic_memory: Optional[EpisodicMemory] = None,
        semantic_memory: Optional[SemanticMemory] = None,
        config: Optional[RAGConfig] = None,
        embedding_fn: Optional[Callable[[str], list[float]]] = None,
    ):
        self.vector_store = vector_store
        self.episodic_memory = episodic_memory
        self.semantic_memory = semantic_memory
        self.config = config or RAGConfig()
        self.embedding_fn = embedding_fn

        # Cache
        self._cache: dict[str, tuple[RAGContext, datetime]] = {}

        # Statistics
        self._total_queries = 0
        self._cache_hits = 0
        self._avg_retrieval_time_ms = 0.0

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize RAG engine."""
        self._initialized = True
        logger.info("rag_engine_initialized", config=self.config.to_dict())

    async def shutdown(self) -> None:
        """Shutdown RAG engine."""
        self._cache.clear()
        self._initialized = False
        logger.info("rag_engine_shutdown")

    async def retrieve(
        self,
        query: str,
        config_override: Optional[RAGConfig] = None,
    ) -> RAGContext:
        """
        Retrieve relevant context for a query.

        Args:
            query: The query to retrieve context for
            config_override: Optional config override for this query

        Returns:
            RAGContext with retrieved and formatted results
        """
        config = config_override or self.config
        start_time = datetime.now()

        # Check cache
        cache_key = f"{query}:{hash(json.dumps(config.to_dict()))}"
        if config.cache_enabled and cache_key in self._cache:
            cached, cache_time = self._cache[cache_key]
            if (datetime.now() - cache_time).total_seconds() < config.cache_ttl_seconds:
                self._cache_hits += 1
                return cached

        self._total_queries += 1

        # Retrieve from all sources
        all_results: list[RetrievalResult] = []

        if config.retrieval_strategy in (RetrievalStrategy.DENSE, RetrievalStrategy.HYBRID):
            vector_results = await self._retrieve_from_vector(query, config)
            all_results.extend(vector_results)

        if config.retrieval_strategy in (RetrievalStrategy.SPARSE, RetrievalStrategy.HYBRID):
            episodic_results = await self._retrieve_from_episodic(query, config)
            all_results.extend(episodic_results)

        # Semantic/knowledge graph retrieval
        semantic_results = await self._retrieve_from_semantic(query, config)
        all_results.extend(semantic_results)

        # Multi-hop retrieval if enabled
        if config.retrieval_strategy == RetrievalStrategy.MULTI_HOP:
            multi_hop_results = await self._multi_hop_retrieve(query, all_results, config)
            all_results.extend(multi_hop_results)

        # Rerank results
        reranked = await self._rerank(query, all_results, config)

        # Format context
        formatted_context, token_estimate = self._format_context(reranked, config)

        elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000

        # Update stats
        self._avg_retrieval_time_ms = (
            self._avg_retrieval_time_ms * 0.9 + elapsed_ms * 0.1
        )

        # Build result
        sources_used = list(set(r.source for r in reranked))

        context = RAGContext(
            query=query,
            results=reranked,
            formatted_context=formatted_context,
            total_tokens_estimate=token_estimate,
            retrieval_time_ms=elapsed_ms,
            sources_used=sources_used,
        )

        # Cache result
        if config.cache_enabled:
            self._cache[cache_key] = (context, datetime.now())

            # Limit cache size
            if len(self._cache) > 1000:
                oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][1])
                del self._cache[oldest_key]

        logger.debug(
            "rag_retrieval_complete",
            query_length=len(query),
            results=len(reranked),
            elapsed_ms=elapsed_ms,
        )

        return context

    async def _retrieve_from_vector(
        self,
        query: str,
        config: RAGConfig,
    ) -> list[RetrievalResult]:
        """Retrieve from vector store."""
        if not self.vector_store:
            return []

        try:
            results = await self.vector_store.search(
                query=query,
                k=config.top_k,
            )

            return [
                RetrievalResult(
                    id=r.entry.id,
                    content=r.entry.text,
                    source="vector",
                    score=r.score * config.vector_weight,
                    metadata=r.entry.metadata,
                )
                for r in results
                if r.score >= config.similarity_threshold
            ]

        except Exception as e:
            logger.error("vector_retrieval_error", error=str(e))
            return []

    async def _retrieve_from_episodic(
        self,
        query: str,
        config: RAGConfig,
    ) -> list[RetrievalResult]:
        """Retrieve from episodic memory."""
        if not self.episodic_memory:
            return []

        try:
            episodes = await self.episodic_memory.search_similar(
                query=query,
                k=config.top_k,
            )

            results = []
            for episode in episodes:
                content = episode.get_summary()
                results.append(
                    RetrievalResult(
                        id=episode.id,
                        content=content,
                        source="episodic",
                        score=episode.importance * config.episodic_weight,
                        metadata={
                            "type": episode.episode_type.value,
                            "success": episode.success,
                            "lessons": episode.lessons_learned,
                        },
                    )
                )

            return results

        except Exception as e:
            logger.error("episodic_retrieval_error", error=str(e))
            return []

    async def _retrieve_from_semantic(
        self,
        query: str,
        config: RAGConfig,
    ) -> list[RetrievalResult]:
        """Retrieve from semantic memory (knowledge graph)."""
        if not self.semantic_memory:
            return []

        try:
            results = []

            # Extract key terms from query (simple tokenization)
            terms = [t.strip().lower() for t in query.split() if len(t) > 3]

            for term in terms[:5]:  # Limit terms
                concept = await self.semantic_memory.get_concept(term)
                if concept:
                    # Get related concepts
                    relations = await self.semantic_memory.get_relations(concept)

                    for rel in relations[:3]:
                        target = await self.semantic_memory.get_concept(rel.target_id)
                        if target:
                            content = f"{concept.name} {rel.relation_type.value} {target.name}"
                            if target.description:
                                content += f": {target.description}"

                            results.append(
                                RetrievalResult(
                                    id=f"{concept.id}-{rel.id}",
                                    content=content,
                                    source="semantic",
                                    score=rel.confidence * config.semantic_weight,
                                    metadata={
                                        "relation_type": rel.relation_type.value,
                                        "source_concept": concept.name,
                                        "target_concept": target.name,
                                    },
                                )
                            )

            return results

        except Exception as e:
            logger.error("semantic_retrieval_error", error=str(e))
            return []

    async def _multi_hop_retrieve(
        self,
        query: str,
        initial_results: list[RetrievalResult],
        config: RAGConfig,
    ) -> list[RetrievalResult]:
        """Perform multi-hop retrieval following knowledge graph edges."""
        if not self.semantic_memory:
            return []

        multi_hop_results = []

        # Use initial results as starting points
        for result in initial_results[:3]:  # Limit starting points
            if result.source == "semantic":
                # Extract concept from result
                source_concept_name = result.metadata.get("target_concept")
                if not source_concept_name:
                    continue

                concept = await self.semantic_memory.get_concept(source_concept_name)
                if not concept:
                    continue

                # Follow relations
                for hop in range(1, config.max_hops + 1):
                    relations = await self.semantic_memory.get_relations(concept)

                    for rel in relations[:2]:
                        target = await self.semantic_memory.get_concept(rel.target_id)
                        if target:
                            # Apply hop decay
                            hop_score = result.score * (config.hop_decay ** hop)

                            content = f"[Hop {hop}] {concept.name} -> {rel.relation_type.value} -> {target.name}"

                            multi_hop_results.append(
                                RetrievalResult(
                                    id=f"hop-{hop}-{rel.id}",
                                    content=content,
                                    source="semantic",
                                    score=hop_score,
                                    metadata={
                                        "hop_distance": hop,
                                        "path": f"{concept.name} -> {target.name}",
                                    },
                                    hop_distance=hop,
                                )
                            )

                            # Use target as next concept
                            concept = target

        return multi_hop_results

    async def _rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        config: RAGConfig,
    ) -> list[RetrievalResult]:
        """Rerank results using specified method."""
        if not results:
            return []

        if config.reranking_method == RerankingMethod.NONE:
            # Just sort by score
            results.sort(key=lambda r: r.score, reverse=True)
            return results[:config.rerank_top_k]

        elif config.reranking_method == RerankingMethod.RECIPROCAL_RANK_FUSION:
            # RRF combines rankings from different sources
            k = 60  # RRF constant

            # Group by source and rank within each source
            source_rankings: dict[str, dict[str, int]] = {}
            for source in ["vector", "episodic", "semantic"]:
                source_results = [r for r in results if r.source == source]
                source_results.sort(key=lambda r: r.score, reverse=True)
                source_rankings[source] = {
                    r.id: rank for rank, r in enumerate(source_results)
                }

            # Compute RRF scores
            rrf_scores: dict[str, float] = {}
            for result in results:
                rrf_score = 0.0
                for source, rankings in source_rankings.items():
                    if result.id in rankings:
                        rank = rankings[result.id]
                        rrf_score += 1.0 / (k + rank + 1)
                rrf_scores[result.id] = rrf_score

            # Update scores and sort
            for result in results:
                result.score = rrf_scores.get(result.id, 0.0)

            results.sort(key=lambda r: r.score, reverse=True)
            return results[:config.rerank_top_k]

        else:
            # Default to score-based ranking
            results.sort(key=lambda r: r.score, reverse=True)
            return results[:config.rerank_top_k]

    def _format_context(
        self,
        results: list[RetrievalResult],
        config: RAGConfig,
    ) -> tuple[str, int]:
        """Format results into context string for LLM."""
        if not results:
            return "", 0

        context_parts = []
        total_length = 0

        for i, result in enumerate(results):
            # Format each result
            part = f"[{i + 1}] {result.content}"

            if config.include_metadata and result.metadata:
                # Add relevant metadata
                meta_str = ", ".join(
                    f"{k}: {v}" for k, v in list(result.metadata.items())[:3]
                )
                part += f"\n   ({meta_str})"

            part += "\n"

            # Check length limit
            if total_length + len(part) > config.max_context_length * 4:  # Rough char estimate
                break

            context_parts.append(part)
            total_length += len(part)

        formatted = "\n".join(context_parts)

        # Rough token estimate (4 chars per token)
        token_estimate = len(formatted) // 4

        return formatted, token_estimate

    async def expand_query(self, query: str) -> list[str]:
        """
        Expand query into multiple sub-queries for better retrieval.

        This enables query decomposition for complex questions.
        """
        expanded = [query]  # Original query

        # Simple expansion: extract key phrases
        words = query.split()
        if len(words) > 5:
            # Create sub-queries from different parts
            mid = len(words) // 2
            expanded.append(" ".join(words[:mid]))
            expanded.append(" ".join(words[mid:]))

        # Add question reformulations
        if query.lower().startswith("how"):
            expanded.append(query.replace("How", "What is the process for", 1))
        elif query.lower().startswith("why"):
            expanded.append(query.replace("Why", "What is the reason for", 1))
        elif query.lower().startswith("what"):
            expanded.append(query.replace("What", "Describe", 1))

        return expanded[:5]  # Limit expansions

    async def retrieve_with_expansion(
        self,
        query: str,
        config_override: Optional[RAGConfig] = None,
    ) -> RAGContext:
        """Retrieve with query expansion for better coverage."""
        config = config_override or self.config

        expanded_queries = await self.expand_query(query)

        all_results = []
        for eq in expanded_queries:
            context = await self.retrieve(eq, config)
            all_results.extend(context.results)

        # Deduplicate and rerank
        seen_ids = set()
        unique_results = []
        for result in all_results:
            if result.id not in seen_ids:
                seen_ids.add(result.id)
                unique_results.append(result)

        # Rerank combined results
        reranked = await self._rerank(query, unique_results, config)

        # Format
        formatted_context, token_estimate = self._format_context(reranked, config)

        return RAGContext(
            query=query,
            results=reranked,
            formatted_context=formatted_context,
            total_tokens_estimate=token_estimate,
            retrieval_time_ms=0.0,  # Not tracked for expansion
            sources_used=list(set(r.source for r in reranked)),
        )

    def get_stats(self) -> dict[str, Any]:
        """Get RAG engine statistics."""
        cache_hit_rate = (
            self._cache_hits / max(1, self._total_queries)
        ) if self._total_queries > 0 else 0.0

        return {
            "total_queries": self._total_queries,
            "cache_hits": self._cache_hits,
            "cache_hit_rate": cache_hit_rate,
            "cache_size": len(self._cache),
            "avg_retrieval_time_ms": self._avg_retrieval_time_ms,
            "config": self.config.to_dict(),
        }
