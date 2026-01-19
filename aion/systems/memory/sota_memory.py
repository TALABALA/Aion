"""
AION SOTA Memory System

Advanced retrieval-augmented memory with:
- Hypothetical Document Embeddings (HyDE)
- Hybrid dense-sparse retrieval (ColBERT-style)
- Cross-encoder reranking
- Multi-hop reasoning retrieval
- Episodic memory with temporal context
- Memory compression and summarization
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import re
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


# ============================================================================
# Hypothetical Document Embeddings (HyDE)
# ============================================================================

class HyDEGenerator:
    """
    Hypothetical Document Embeddings generator.

    Given a query, generates a hypothetical answer document,
    then embeds that document for retrieval. This captures
    the semantic space of answers rather than questions.
    """

    def __init__(self, llm_adapter, num_hypotheses: int = 3):
        self.llm = llm_adapter
        self.num_hypotheses = num_hypotheses

    async def generate_hypothetical_documents(
        self,
        query: str,
        context: Optional[str] = None,
    ) -> list[str]:
        """
        Generate hypothetical documents that would answer the query.

        Args:
            query: The search query
            context: Optional additional context

        Returns:
            List of hypothetical document texts
        """
        from aion.core.llm import Message

        context_text = f"\nContext: {context}" if context else ""

        prompt = f"""Given this question, write {self.num_hypotheses} different hypothetical passages that would perfectly answer it.

Question: {query}{context_text}

Write diverse passages with different perspectives, styles, and information.
Each passage should be self-contained and directly answer the question.

Format each passage on a new paragraph, separated by "---"
"""

        try:
            response = await self.llm.complete([
                Message(role="system", content="You generate hypothetical document passages for semantic search."),
                Message(role="user", content=prompt),
            ], temperature=0.8)

            # Split into separate documents
            docs = [
                doc.strip()
                for doc in response.content.split("---")
                if doc.strip()
            ]

            # Ensure we have enough
            while len(docs) < self.num_hypotheses:
                docs.append(query)

            return docs[:self.num_hypotheses]

        except Exception as e:
            logger.warning("HyDE generation failed", error=str(e))
            return [query]  # Fallback to original query


# ============================================================================
# BM25 Sparse Retrieval
# ============================================================================

class BM25Index:
    """
    BM25 sparse retrieval index.

    Implements Okapi BM25 scoring for keyword-based retrieval
    to complement dense embeddings.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

        self.documents: dict[str, str] = {}
        self.doc_lengths: dict[str, int] = {}
        self.avg_doc_length: float = 0.0
        self.doc_freqs: dict[str, int] = defaultdict(int)  # term -> doc count
        self.inverted_index: dict[str, set[str]] = defaultdict(set)  # term -> doc_ids
        self.term_freqs: dict[str, dict[str, int]] = {}  # doc_id -> {term: count}

    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization."""
        # Lowercase and split on non-alphanumeric
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens

    def add_document(self, doc_id: str, text: str) -> None:
        """Add a document to the index."""
        tokens = self._tokenize(text)

        self.documents[doc_id] = text
        self.doc_lengths[doc_id] = len(tokens)

        # Update term frequencies
        term_freq = defaultdict(int)
        seen_terms = set()

        for token in tokens:
            term_freq[token] += 1
            if token not in seen_terms:
                self.doc_freqs[token] += 1
                seen_terms.add(token)
            self.inverted_index[token].add(doc_id)

        self.term_freqs[doc_id] = dict(term_freq)

        # Update average doc length
        self.avg_doc_length = sum(self.doc_lengths.values()) / len(self.doc_lengths)

    def remove_document(self, doc_id: str) -> None:
        """Remove a document from the index."""
        if doc_id not in self.documents:
            return

        # Update inverted index and doc freqs
        for term in self.term_freqs.get(doc_id, {}):
            self.inverted_index[term].discard(doc_id)
            self.doc_freqs[term] -= 1

        del self.documents[doc_id]
        del self.doc_lengths[doc_id]
        del self.term_freqs[doc_id]

        if self.doc_lengths:
            self.avg_doc_length = sum(self.doc_lengths.values()) / len(self.doc_lengths)

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """
        Search for documents matching the query.

        Returns:
            List of (doc_id, score) tuples
        """
        query_tokens = self._tokenize(query)
        scores: dict[str, float] = defaultdict(float)

        N = len(self.documents)

        for token in query_tokens:
            if token not in self.inverted_index:
                continue

            # IDF
            df = self.doc_freqs[token]
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1)

            for doc_id in self.inverted_index[token]:
                tf = self.term_freqs[doc_id].get(token, 0)
                doc_len = self.doc_lengths[doc_id]

                # BM25 scoring
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_length)

                scores[doc_id] += idf * (numerator / denominator)

        # Sort by score
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        return sorted_scores[:top_k]


# ============================================================================
# Cross-Encoder Reranker
# ============================================================================

class CrossEncoderReranker:
    """
    Cross-encoder reranking using LLM.

    More accurate than bi-encoder but slower.
    Used as second-stage reranking.
    """

    def __init__(self, llm_adapter, batch_size: int = 5):
        self.llm = llm_adapter
        self.batch_size = batch_size

    async def rerank(
        self,
        query: str,
        documents: list[tuple[str, str, float]],  # (id, text, initial_score)
        top_k: int = 5,
    ) -> list[tuple[str, str, float]]:
        """
        Rerank documents using cross-encoder.

        Args:
            query: Search query
            documents: List of (doc_id, doc_text, initial_score)
            top_k: Number of documents to return

        Returns:
            Reranked list of (doc_id, doc_text, new_score)
        """
        if not documents:
            return []

        from aion.core.llm import Message

        reranked = []

        # Process in batches
        for i in range(0, len(documents), self.batch_size):
            batch = documents[i:i + self.batch_size]

            # Score each document
            for doc_id, doc_text, initial_score in batch:
                score = await self._score_pair(query, doc_text)
                reranked.append((doc_id, doc_text, score))

        # Sort by new score
        reranked.sort(key=lambda x: x[2], reverse=True)

        return reranked[:top_k]

    async def _score_pair(self, query: str, document: str) -> float:
        """Score a query-document pair."""
        from aion.core.llm import Message

        prompt = f"""Rate how well this document answers the question on a scale of 0-10.

Question: {query}

Document: {document[:1000]}

Consider:
- Relevance: Does it address the question?
- Completeness: Does it fully answer the question?
- Quality: Is the information accurate and well-presented?

Respond with only a number from 0-10.
"""

        try:
            response = await self.llm.complete([
                Message(role="system", content="You rate document relevance. Respond with only a number 0-10."),
                Message(role="user", content=prompt),
            ], temperature=0.1)

            score = float(response.content.strip().split()[0])
            return min(10, max(0, score)) / 10

        except:
            return 0.5


# ============================================================================
# Multi-Hop Retrieval
# ============================================================================

class MultiHopRetriever:
    """
    Multi-hop retrieval for complex queries.

    Decomposes complex queries into sub-queries,
    retrieves relevant documents for each,
    and synthesizes the results.
    """

    def __init__(
        self,
        llm_adapter,
        retriever_fn,  # Function to retrieve documents
        max_hops: int = 3,
    ):
        self.llm = llm_adapter
        self.retriever = retriever_fn
        self.max_hops = max_hops

    async def retrieve(
        self,
        query: str,
        top_k_per_hop: int = 3,
    ) -> tuple[list[tuple[str, str, float]], list[str]]:
        """
        Perform multi-hop retrieval.

        Args:
            query: Original complex query
            top_k_per_hop: Documents to retrieve per hop

        Returns:
            Tuple of (all_documents, sub_queries)
        """
        from aion.core.llm import Message

        all_docs = []
        sub_queries = [query]
        seen_doc_ids = set()

        current_query = query
        context_so_far = ""

        for hop in range(self.max_hops):
            # Retrieve documents for current query
            docs = await self.retriever(current_query, top_k=top_k_per_hop)

            new_docs = []
            for doc_id, doc_text, score in docs:
                if doc_id not in seen_doc_ids:
                    seen_doc_ids.add(doc_id)
                    new_docs.append((doc_id, doc_text, score))
                    all_docs.append((doc_id, doc_text, score))

            if not new_docs:
                break

            # Update context
            context_so_far += "\n".join([d[1][:200] for d in new_docs[:2]]) + "\n"

            # Generate follow-up query if needed
            if hop < self.max_hops - 1:
                next_query = await self._generate_followup_query(
                    query, context_so_far
                )

                if next_query and next_query != current_query:
                    current_query = next_query
                    sub_queries.append(next_query)
                else:
                    break  # No more queries needed

        return all_docs, sub_queries

    async def _generate_followup_query(
        self,
        original_query: str,
        context: str,
    ) -> Optional[str]:
        """Generate a follow-up query based on retrieved context."""
        from aion.core.llm import Message

        prompt = f"""Original question: {original_query}

Information found so far:
{context[:1000]}

Is more information needed to fully answer the original question?
If yes, generate a follow-up search query to find the missing information.
If the question is fully answered, respond with "COMPLETE".

Follow-up query:"""

        try:
            response = await self.llm.complete([
                Message(role="system", content="Generate follow-up queries for information retrieval."),
                Message(role="user", content=prompt),
            ], temperature=0.3)

            result = response.content.strip()

            if "COMPLETE" in result.upper():
                return None

            return result

        except:
            return None


# ============================================================================
# Episodic Memory with Temporal Context
# ============================================================================

@dataclass
class EpisodicMemoryEntry:
    """An episode in memory."""
    id: str
    content: str
    timestamp: datetime
    context: dict[str, Any]
    embeddings: Optional[np.ndarray] = None

    # Temporal links
    before_ids: list[str] = field(default_factory=list)
    after_ids: list[str] = field(default_factory=list)

    # Importance and access
    importance: float = 0.5
    access_count: int = 0
    emotional_valence: float = 0.0  # -1 to 1

    # Associations
    linked_concepts: list[str] = field(default_factory=list)
    causal_links: list[tuple[str, str]] = field(default_factory=list)  # (linked_id, relationship)


class EpisodicMemoryStore:
    """
    Episodic memory with temporal awareness.

    Features:
    - Temporal ordering and context
    - Causal relationship tracking
    - Memory consolidation over time
    - Emotional salience
    """

    def __init__(
        self,
        embedding_fn,
        consolidation_threshold: float = 0.3,
    ):
        self.embedding_fn = embedding_fn
        self.consolidation_threshold = consolidation_threshold

        self.episodes: dict[str, EpisodicMemoryEntry] = {}
        self.temporal_index: list[str] = []  # Ordered by time
        self.concept_index: dict[str, set[str]] = defaultdict(set)

    async def store_episode(
        self,
        content: str,
        context: Optional[dict] = None,
        importance: float = 0.5,
        emotional_valence: float = 0.0,
    ) -> EpisodicMemoryEntry:
        """Store a new episode."""
        episode_id = str(uuid.uuid4())

        # Generate embedding
        embedding = await self.embedding_fn(content)

        # Extract concepts
        concepts = self._extract_concepts(content)

        episode = EpisodicMemoryEntry(
            id=episode_id,
            content=content,
            timestamp=datetime.now(),
            context=context or {},
            embeddings=embedding,
            importance=importance,
            emotional_valence=emotional_valence,
            linked_concepts=concepts,
        )

        # Link to previous episode
        if self.temporal_index:
            prev_id = self.temporal_index[-1]
            episode.before_ids.append(prev_id)
            self.episodes[prev_id].after_ids.append(episode_id)

        # Store
        self.episodes[episode_id] = episode
        self.temporal_index.append(episode_id)

        # Update concept index
        for concept in concepts:
            self.concept_index[concept].add(episode_id)

        return episode

    def _extract_concepts(self, text: str) -> list[str]:
        """Extract key concepts from text."""
        # Simple extraction - in production use NER/keyword extraction
        words = re.findall(r'\b[A-Z][a-z]+\b', text)
        return list(set(words))[:10]

    async def recall_temporal(
        self,
        reference_time: Optional[datetime] = None,
        window_hours: float = 24,
        limit: int = 10,
    ) -> list[EpisodicMemoryEntry]:
        """Recall episodes around a time window."""
        reference_time = reference_time or datetime.now()

        window_start = reference_time - timedelta(hours=window_hours)
        window_end = reference_time + timedelta(hours=window_hours)

        matching = [
            self.episodes[eid]
            for eid in self.temporal_index
            if window_start <= self.episodes[eid].timestamp <= window_end
        ]

        # Sort by closeness to reference time
        matching.sort(
            key=lambda e: abs((e.timestamp - reference_time).total_seconds())
        )

        return matching[:limit]

    async def recall_causal_chain(
        self,
        episode_id: str,
        direction: str = "both",  # "before", "after", "both"
        max_depth: int = 5,
    ) -> list[EpisodicMemoryEntry]:
        """Recall causally linked episodes."""
        if episode_id not in self.episodes:
            return []

        visited = set()
        chain = []

        def traverse(eid: str, depth: int):
            if depth > max_depth or eid in visited:
                return
            visited.add(eid)

            episode = self.episodes[eid]
            chain.append(episode)

            if direction in ("before", "both"):
                for prev_id in episode.before_ids:
                    traverse(prev_id, depth + 1)

            if direction in ("after", "both"):
                for next_id in episode.after_ids:
                    traverse(next_id, depth + 1)

        traverse(episode_id, 0)

        return chain

    async def consolidate(self) -> dict[str, int]:
        """
        Consolidate episodic memories.

        Merges similar episodes, strengthens important ones,
        and forgets unimportant ones.
        """
        stats = {"merged": 0, "strengthened": 0, "forgotten": 0}

        to_forget = []

        for episode_id, episode in self.episodes.items():
            # Calculate current strength
            age_hours = (datetime.now() - episode.timestamp).total_seconds() / 3600
            decay = math.exp(-0.01 * age_hours)
            strength = episode.importance * decay + episode.access_count * 0.1

            if strength < self.consolidation_threshold:
                to_forget.append(episode_id)
                stats["forgotten"] += 1
            elif episode.access_count > 5:
                episode.importance = min(1.0, episode.importance * 1.1)
                stats["strengthened"] += 1

        # Remove forgotten episodes
        for eid in to_forget:
            self._remove_episode(eid)

        return stats

    def _remove_episode(self, episode_id: str) -> None:
        """Remove an episode."""
        if episode_id not in self.episodes:
            return

        episode = self.episodes[episode_id]

        # Update links
        for prev_id in episode.before_ids:
            if prev_id in self.episodes:
                self.episodes[prev_id].after_ids.remove(episode_id)
                # Link to our after
                self.episodes[prev_id].after_ids.extend(episode.after_ids)

        for next_id in episode.after_ids:
            if next_id in self.episodes:
                self.episodes[next_id].before_ids.remove(episode_id)
                self.episodes[next_id].before_ids.extend(episode.before_ids)

        # Update indices
        self.temporal_index.remove(episode_id)
        for concept in episode.linked_concepts:
            self.concept_index[concept].discard(episode_id)

        del self.episodes[episode_id]


# ============================================================================
# Memory Compression & Summarization
# ============================================================================

class MemoryCompressor:
    """
    Compress and summarize memories for efficient storage.

    Uses hierarchical summarization to create multi-scale
    representations of memory content.
    """

    def __init__(self, llm_adapter):
        self.llm = llm_adapter

    async def compress_memories(
        self,
        memories: list[str],
        target_ratio: float = 0.3,
    ) -> str:
        """
        Compress multiple memories into a summary.

        Args:
            memories: List of memory contents
            target_ratio: Target compression ratio

        Returns:
            Compressed summary
        """
        from aion.core.llm import Message

        all_content = "\n\n".join(memories)
        target_length = int(len(all_content) * target_ratio)

        prompt = f"""Summarize the following memories into approximately {target_length} characters.
Preserve the most important information and key facts.

Memories:
{all_content}

Summary:"""

        response = await self.llm.complete([
            Message(role="system", content="You create concise, informative summaries."),
            Message(role="user", content=prompt),
        ])

        return response.content

    async def hierarchical_summarize(
        self,
        memories: list[str],
        levels: int = 3,
    ) -> dict[str, str]:
        """
        Create hierarchical summaries at different levels.

        Returns:
            Dict mapping level to summary
        """
        summaries = {}

        current_content = memories

        for level in range(levels):
            ratio = 1.0 / (2 ** (level + 1))  # 0.5, 0.25, 0.125

            if isinstance(current_content, list):
                text = "\n\n".join(current_content)
            else:
                text = current_content

            summary = await self.compress_memories([text], ratio)
            summaries[f"level_{level}"] = summary
            current_content = summary

        return summaries


# ============================================================================
# SOTA Memory System
# ============================================================================

class SOTAMemorySystem:
    """
    State-of-the-art memory system combining:
    - HyDE for query expansion
    - Hybrid dense-sparse retrieval
    - Cross-encoder reranking
    - Multi-hop retrieval
    - Episodic memory with temporal context
    - Memory compression
    """

    def __init__(
        self,
        llm_adapter,
        embedding_fn,
        max_memories: int = 100000,
    ):
        self.llm = llm_adapter
        self.embedding_fn = embedding_fn
        self.max_memories = max_memories

        # Components
        self.hyde = HyDEGenerator(llm_adapter)
        self.bm25 = BM25Index()
        self.reranker = CrossEncoderReranker(llm_adapter)
        self.compressor = MemoryCompressor(llm_adapter)

        # Storage
        self.memories: dict[str, dict] = {}  # id -> {content, embedding, metadata}
        self.episodic_store = EpisodicMemoryStore(embedding_fn)

        # Indices
        self._embeddings_matrix: Optional[np.ndarray] = None
        self._id_to_idx: dict[str, int] = {}
        self._idx_to_id: dict[int, str] = {}

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the memory system."""
        logger.info("Initializing SOTA Memory System")
        self._initialized = True

    async def store(
        self,
        content: str,
        metadata: Optional[dict] = None,
        importance: float = 0.5,
        is_episodic: bool = True,
    ) -> str:
        """
        Store a memory.

        Args:
            content: Memory content
            metadata: Additional metadata
            importance: Importance score
            is_episodic: Whether to store in episodic memory

        Returns:
            Memory ID
        """
        memory_id = str(uuid.uuid4())

        # Generate embedding
        embedding = await self.embedding_fn(content)
        if hasattr(embedding, 'numpy'):
            embedding = embedding.numpy()
        if embedding.ndim > 1:
            embedding = embedding[0]

        # Store in main memory
        self.memories[memory_id] = {
            "content": content,
            "embedding": embedding,
            "metadata": metadata or {},
            "importance": importance,
            "created_at": datetime.now(),
            "access_count": 0,
        }

        # Add to BM25 index
        self.bm25.add_document(memory_id, content)

        # Update embeddings matrix
        self._update_embeddings_index(memory_id, embedding)

        # Store in episodic memory if needed
        if is_episodic:
            await self.episodic_store.store_episode(
                content=content,
                context=metadata,
                importance=importance,
            )

        return memory_id

    def _update_embeddings_index(self, memory_id: str, embedding: np.ndarray) -> None:
        """Update the embeddings index."""
        idx = len(self._id_to_idx)
        self._id_to_idx[memory_id] = idx
        self._idx_to_id[idx] = memory_id

        if self._embeddings_matrix is None:
            self._embeddings_matrix = embedding.reshape(1, -1)
        else:
            self._embeddings_matrix = np.vstack([
                self._embeddings_matrix,
                embedding.reshape(1, -1)
            ])

    async def search(
        self,
        query: str,
        top_k: int = 10,
        use_hyde: bool = True,
        use_reranking: bool = True,
        use_multihop: bool = False,
    ) -> list[dict]:
        """
        Search for relevant memories.

        Args:
            query: Search query
            top_k: Number of results
            use_hyde: Use hypothetical document embeddings
            use_reranking: Use cross-encoder reranking
            use_multihop: Use multi-hop retrieval

        Returns:
            List of search results
        """
        if not self.memories:
            return []

        # Step 1: Query expansion with HyDE
        if use_hyde:
            hypothetical_docs = await self.hyde.generate_hypothetical_documents(query)
        else:
            hypothetical_docs = [query]

        # Step 2: Dense retrieval with expanded queries
        dense_results = await self._dense_search(hypothetical_docs, top_k * 3)

        # Step 3: Sparse (BM25) retrieval
        sparse_results = self.bm25.search(query, top_k * 3)

        # Step 4: Combine results (hybrid)
        combined = self._combine_results(dense_results, sparse_results)

        # Step 5: Multi-hop if needed
        if use_multihop:
            async def retrieve_fn(q, top_k):
                docs = await self._dense_search([q], top_k)
                return [(d["id"], d["content"], d["score"]) for d in docs]

            multi_hop = MultiHopRetriever(self.llm, retrieve_fn)
            multi_docs, _ = await multi_hop.retrieve(query, top_k_per_hop=3)
            combined.extend([
                {"id": d[0], "content": d[1], "score": d[2]}
                for d in multi_docs
            ])

        # Deduplicate
        seen = set()
        unique = []
        for r in combined:
            if r["id"] not in seen:
                seen.add(r["id"])
                unique.append(r)

        # Step 6: Rerank with cross-encoder
        if use_reranking and len(unique) > top_k:
            to_rerank = [
                (r["id"], r["content"], r["score"])
                for r in unique[:top_k * 2]
            ]
            reranked = await self.reranker.rerank(query, to_rerank, top_k)
            return [
                {
                    "id": r[0],
                    "content": r[1],
                    "score": r[2],
                    "metadata": self.memories.get(r[0], {}).get("metadata", {}),
                }
                for r in reranked
            ]

        # Update access counts
        for r in unique[:top_k]:
            if r["id"] in self.memories:
                self.memories[r["id"]]["access_count"] += 1

        return unique[:top_k]

    async def _dense_search(
        self,
        queries: list[str],
        top_k: int,
    ) -> list[dict]:
        """Perform dense vector search."""
        if self._embeddings_matrix is None or len(self._embeddings_matrix) == 0:
            return []

        all_scores = np.zeros(len(self._embeddings_matrix))

        for query in queries:
            query_embedding = await self.embedding_fn(query)
            if hasattr(query_embedding, 'numpy'):
                query_embedding = query_embedding.numpy()
            if query_embedding.ndim > 1:
                query_embedding = query_embedding[0]

            # Cosine similarity
            query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
            matrix_norms = self._embeddings_matrix / (
                np.linalg.norm(self._embeddings_matrix, axis=1, keepdims=True) + 1e-8
            )
            scores = np.dot(matrix_norms, query_norm)
            all_scores += scores

        # Average scores
        all_scores /= len(queries)

        # Get top-k
        top_indices = np.argsort(-all_scores)[:top_k]

        results = []
        for idx in top_indices:
            memory_id = self._idx_to_id.get(int(idx))
            if memory_id and memory_id in self.memories:
                results.append({
                    "id": memory_id,
                    "content": self.memories[memory_id]["content"],
                    "score": float(all_scores[idx]),
                    "metadata": self.memories[memory_id].get("metadata", {}),
                })

        return results

    def _combine_results(
        self,
        dense_results: list[dict],
        sparse_results: list[tuple[str, float]],
        dense_weight: float = 0.7,
    ) -> list[dict]:
        """Combine dense and sparse results using RRF."""
        # Reciprocal Rank Fusion
        rrf_scores: dict[str, float] = defaultdict(float)
        k = 60  # RRF constant

        # Dense results
        for rank, r in enumerate(dense_results):
            rrf_scores[r["id"]] += dense_weight / (k + rank + 1)

        # Sparse results
        for rank, (doc_id, score) in enumerate(sparse_results):
            rrf_scores[doc_id] += (1 - dense_weight) / (k + rank + 1)

        # Sort by RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        results = []
        for doc_id in sorted_ids:
            if doc_id in self.memories:
                results.append({
                    "id": doc_id,
                    "content": self.memories[doc_id]["content"],
                    "score": rrf_scores[doc_id],
                    "metadata": self.memories[doc_id].get("metadata", {}),
                })

        return results

    async def recall(
        self,
        query: str,
        context: Optional[str] = None,
    ) -> str:
        """
        Recall and synthesize information.

        Args:
            query: What to recall
            context: Additional context

        Returns:
            Synthesized response
        """
        results = await self.search(query, top_k=5, use_reranking=True)

        if not results:
            return "No relevant memories found."

        from aion.core.llm import Message

        context_text = "\n\n".join([
            f"Memory {i+1}: {r['content']}"
            for i, r in enumerate(results)
        ])

        prompt = f"""Based on the following memories, answer the question.

Question: {query}

Relevant memories:
{context_text}

Synthesize the information to provide a comprehensive answer.
If the memories don't fully answer the question, say what's missing.
"""

        response = await self.llm.complete([
            Message(role="system", content="You synthesize information from memories to answer questions."),
            Message(role="user", content=prompt),
        ])

        return response.content

    async def consolidate(self) -> dict[str, Any]:
        """Consolidate memories."""
        stats = {"compressed": 0}

        # Consolidate episodic store
        episodic_stats = await self.episodic_store.consolidate()

        # Compress old memories
        old_memories = [
            (mid, m) for mid, m in self.memories.items()
            if (datetime.now() - m["created_at"]).days > 7
            and m["access_count"] < 3
        ]

        if len(old_memories) > 100:
            # Compress into summaries
            contents = [m[1]["content"] for m in old_memories[:50]]
            summary = await self.compressor.compress_memories(contents)

            # Store summary as new memory
            await self.store(
                content=f"[Compressed memories]: {summary}",
                metadata={"type": "compressed", "source_count": len(contents)},
                importance=0.5,
                is_episodic=False,
            )

            # Remove old memories
            for mid, _ in old_memories[:50]:
                del self.memories[mid]
                self.bm25.remove_document(mid)

            stats["compressed"] = 50

        return {**stats, **episodic_stats}

    def get_stats(self) -> dict[str, Any]:
        """Get memory system statistics."""
        return {
            "total_memories": len(self.memories),
            "episodic_memories": len(self.episodic_store.episodes),
            "bm25_documents": len(self.bm25.documents),
            "embedding_matrix_shape": (
                self._embeddings_matrix.shape if self._embeddings_matrix is not None else None
            ),
        }

    async def shutdown(self) -> None:
        """Shutdown the memory system."""
        self._initialized = False
