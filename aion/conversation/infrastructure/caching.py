"""
AION SOTA Semantic Caching

State-of-the-art semantic caching featuring:
- Embedding-based similarity lookup
- LRU eviction with importance weighting
- Time-based expiration
- Cache warming and preloading
- Statistics and hit rate tracking
"""

from __future__ import annotations

import asyncio
import hashlib
import heapq
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Generic, List, Optional, Protocol, Tuple, TypeVar

import numpy as np
import structlog

logger = structlog.get_logger(__name__)

T = TypeVar("T")


# =============================================================================
# Embedding Provider Protocol
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
    """Default embedding provider using sentence-transformers or fallback."""

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
            logger.info(f"Loaded embedding model: {self.model_name}")
        except ImportError:
            logger.warning("sentence-transformers not available, using hash-based embeddings")

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
# Cache Entry
# =============================================================================

@dataclass
class CacheEntry:
    """A cached entry with embedding and metadata."""
    key: str
    value: Any
    embedding: List[float]

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    access_count: int = 0

    # For prioritization
    importance: float = 1.0

    # Original query (for debugging)
    original_query: str = ""

    @property
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at

    def touch(self) -> None:
        """Update access time and count."""
        self.last_accessed = datetime.now()
        self.access_count += 1

    @property
    def priority_score(self) -> float:
        """
        Calculate priority score for eviction.
        Higher score = more important = evict later.
        """
        # Base on access count and recency
        age_hours = (datetime.now() - self.last_accessed).total_seconds() / 3600
        recency_score = 1.0 / (1.0 + age_hours)  # Decays over time

        access_score = min(self.access_count / 10.0, 1.0)  # Caps at 10 accesses

        return (recency_score * 0.4 + access_score * 0.3 + self.importance * 0.3)


@dataclass
class CacheStats:
    """Cache statistics."""
    hits: int = 0
    misses: int = 0
    semantic_hits: int = 0
    exact_hits: int = 0
    evictions: int = 0
    expirations: int = 0
    total_queries: int = 0
    avg_similarity: float = 0.0

    @property
    def hit_rate(self) -> float:
        if self.total_queries == 0:
            return 0.0
        return self.hits / self.total_queries


# =============================================================================
# Semantic Cache
# =============================================================================

class SemanticCache:
    """
    Semantic cache using embedding-based similarity lookup.

    Features:
    - Exact key matching for identical queries
    - Semantic similarity matching for similar queries
    - Configurable similarity threshold
    - LRU eviction with importance weighting
    - Background cleanup of expired entries
    """

    def __init__(
        self,
        embedding_provider: Optional[EmbeddingProvider] = None,
        max_size: int = 1000,
        similarity_threshold: float = 0.85,
        default_ttl_seconds: int = 3600,
        cleanup_interval_seconds: int = 300,
    ):
        self.embedding_provider = embedding_provider or DefaultEmbeddingProvider()
        self.max_size = max_size
        self.similarity_threshold = similarity_threshold
        self.default_ttl_seconds = default_ttl_seconds
        self.cleanup_interval_seconds = cleanup_interval_seconds

        # Storage
        self._exact_cache: Dict[str, CacheEntry] = {}  # hash -> entry
        self._semantic_entries: List[CacheEntry] = []  # For semantic search
        self._embeddings: np.ndarray = np.array([])  # Embedding matrix

        # Statistics
        self._stats = CacheStats()

        # Background cleanup
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False

        # Lock for thread safety
        self._lock = asyncio.Lock()

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the cache."""
        if self._initialized:
            return

        if hasattr(self.embedding_provider, 'initialize'):
            await self.embedding_provider.initialize()

        self._initialized = True
        logger.info("Semantic cache initialized")

    async def start_cleanup(self) -> None:
        """Start background cleanup task."""
        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def stop(self) -> None:
        """Stop the cache and cleanup."""
        self._running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

    async def _cleanup_loop(self) -> None:
        """Background cleanup of expired entries."""
        while self._running:
            try:
                await asyncio.sleep(self.cleanup_interval_seconds)
                await self._cleanup_expired()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")

    async def _cleanup_expired(self) -> int:
        """Remove expired entries."""
        async with self._lock:
            expired_keys = [
                key for key, entry in self._exact_cache.items()
                if entry.is_expired
            ]

            for key in expired_keys:
                del self._exact_cache[key]
                self._stats.expirations += 1

            # Rebuild semantic entries list
            self._semantic_entries = [
                e for e in self._semantic_entries if not e.is_expired
            ]
            self._rebuild_embedding_matrix()

            return len(expired_keys)

    def _rebuild_embedding_matrix(self) -> None:
        """Rebuild the embedding matrix for semantic search."""
        if self._semantic_entries:
            self._embeddings = np.array([e.embedding for e in self._semantic_entries])
        else:
            self._embeddings = np.array([])

    def _compute_key_hash(self, query: str) -> str:
        """Compute hash key for exact matching."""
        normalized = query.lower().strip()
        return hashlib.sha256(normalized.encode()).hexdigest()

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    async def get(
        self,
        query: str,
        use_semantic: bool = True,
    ) -> Optional[Tuple[Any, float]]:
        """
        Get a cached value for a query.

        Args:
            query: The query string
            use_semantic: Whether to use semantic similarity matching

        Returns:
            Tuple of (value, similarity_score) or None if not found
        """
        if not self._initialized:
            await self.initialize()

        self._stats.total_queries += 1

        # Try exact match first
        key_hash = self._compute_key_hash(query)

        async with self._lock:
            if key_hash in self._exact_cache:
                entry = self._exact_cache[key_hash]
                if not entry.is_expired:
                    entry.touch()
                    self._stats.hits += 1
                    self._stats.exact_hits += 1
                    return (entry.value, 1.0)
                else:
                    del self._exact_cache[key_hash]
                    self._stats.expirations += 1

        # Try semantic match
        if use_semantic and len(self._semantic_entries) > 0:
            return await self._semantic_lookup(query)

        self._stats.misses += 1
        return None

    async def _semantic_lookup(self, query: str) -> Optional[Tuple[Any, float]]:
        """Look up using semantic similarity."""
        # Get query embedding
        query_embedding = await self.embedding_provider.embed(query)
        query_vector = np.array(query_embedding)

        async with self._lock:
            if len(self._embeddings) == 0:
                self._stats.misses += 1
                return None

            # Compute similarities
            similarities = np.dot(self._embeddings, query_vector) / (
                np.linalg.norm(self._embeddings, axis=1) * np.linalg.norm(query_vector)
            )

            # Find best match
            best_idx = np.argmax(similarities)
            best_similarity = similarities[best_idx]

            # Update average similarity stat
            self._stats.avg_similarity = (
                self._stats.avg_similarity * 0.9 + best_similarity * 0.1
            )

            if best_similarity >= self.similarity_threshold:
                entry = self._semantic_entries[best_idx]
                if not entry.is_expired:
                    entry.touch()
                    self._stats.hits += 1
                    self._stats.semantic_hits += 1
                    logger.debug(
                        f"Semantic cache hit",
                        similarity=f"{best_similarity:.3f}",
                        original_query=entry.original_query[:50],
                    )
                    return (entry.value, float(best_similarity))

        self._stats.misses += 1
        return None

    async def set(
        self,
        query: str,
        value: Any,
        ttl_seconds: Optional[int] = None,
        importance: float = 1.0,
    ) -> None:
        """
        Cache a value for a query.

        Args:
            query: The query string
            value: The value to cache
            ttl_seconds: Time-to-live in seconds
            importance: Importance score for eviction priority
        """
        if not self._initialized:
            await self.initialize()

        # Get embedding
        embedding = await self.embedding_provider.embed(query)

        # Compute expiration
        ttl = ttl_seconds or self.default_ttl_seconds
        expires_at = datetime.now() + timedelta(seconds=ttl) if ttl else None

        # Create entry
        entry = CacheEntry(
            key=self._compute_key_hash(query),
            value=value,
            embedding=embedding,
            expires_at=expires_at,
            importance=importance,
            original_query=query,
        )

        async with self._lock:
            # Check capacity and evict if needed
            while len(self._exact_cache) >= self.max_size:
                await self._evict_one()

            # Add to exact cache
            self._exact_cache[entry.key] = entry

            # Add to semantic entries
            self._semantic_entries.append(entry)
            self._rebuild_embedding_matrix()

        logger.debug(f"Cached response for query: {query[:50]}...")

    async def _evict_one(self) -> None:
        """Evict one entry based on priority."""
        if not self._semantic_entries:
            return

        # Find entry with lowest priority
        min_priority = float('inf')
        min_entry = None
        min_idx = 0

        for idx, entry in enumerate(self._semantic_entries):
            if entry.is_expired:
                min_entry = entry
                min_idx = idx
                break

            priority = entry.priority_score
            if priority < min_priority:
                min_priority = priority
                min_entry = entry
                min_idx = idx

        if min_entry:
            # Remove from both caches
            self._exact_cache.pop(min_entry.key, None)
            self._semantic_entries.pop(min_idx)
            self._stats.evictions += 1

    async def invalidate(self, query: str) -> bool:
        """Invalidate a cached entry."""
        key_hash = self._compute_key_hash(query)

        async with self._lock:
            if key_hash in self._exact_cache:
                entry = self._exact_cache[key_hash]
                del self._exact_cache[key_hash]

                # Remove from semantic entries
                self._semantic_entries = [
                    e for e in self._semantic_entries if e.key != key_hash
                ]
                self._rebuild_embedding_matrix()
                return True

        return False

    async def invalidate_similar(
        self,
        query: str,
        threshold: float = 0.9,
    ) -> int:
        """Invalidate entries similar to the query."""
        query_embedding = await self.embedding_provider.embed(query)
        query_vector = np.array(query_embedding)

        async with self._lock:
            if len(self._embeddings) == 0:
                return 0

            # Find similar entries
            similarities = np.dot(self._embeddings, query_vector) / (
                np.linalg.norm(self._embeddings, axis=1) * np.linalg.norm(query_vector)
            )

            # Get indices of similar entries
            similar_indices = np.where(similarities >= threshold)[0]

            if len(similar_indices) == 0:
                return 0

            # Get keys to remove
            keys_to_remove = [
                self._semantic_entries[idx].key for idx in similar_indices
            ]

            # Remove from caches
            for key in keys_to_remove:
                self._exact_cache.pop(key, None)

            self._semantic_entries = [
                e for e in self._semantic_entries if e.key not in keys_to_remove
            ]
            self._rebuild_embedding_matrix()

            return len(keys_to_remove)

    async def clear(self) -> None:
        """Clear all cache entries."""
        async with self._lock:
            self._exact_cache.clear()
            self._semantic_entries.clear()
            self._embeddings = np.array([])

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "hits": self._stats.hits,
            "misses": self._stats.misses,
            "hit_rate": self._stats.hit_rate,
            "exact_hits": self._stats.exact_hits,
            "semantic_hits": self._stats.semantic_hits,
            "evictions": self._stats.evictions,
            "expirations": self._stats.expirations,
            "total_queries": self._stats.total_queries,
            "avg_similarity": self._stats.avg_similarity,
            "size": len(self._exact_cache),
            "max_size": self.max_size,
        }


# =============================================================================
# Response Cache (Specialized for LLM responses)
# =============================================================================

class LLMResponseCache:
    """
    Specialized cache for LLM responses.

    Optimized for caching model completions with:
    - Query normalization
    - Model-aware caching
    - Temperature-based cache invalidation
    - Token cost tracking
    """

    def __init__(
        self,
        embedding_provider: Optional[EmbeddingProvider] = None,
        max_size: int = 500,
        similarity_threshold: float = 0.9,  # Higher for LLM responses
        default_ttl_seconds: int = 7200,  # 2 hours
    ):
        self._cache = SemanticCache(
            embedding_provider=embedding_provider,
            max_size=max_size,
            similarity_threshold=similarity_threshold,
            default_ttl_seconds=default_ttl_seconds,
        )

        # Token savings tracking
        self._tokens_saved = 0
        self._tokens_served = 0

    async def initialize(self) -> None:
        await self._cache.initialize()

    async def get(
        self,
        query: str,
        model: str,
        temperature: float = 0.0,
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached LLM response.

        Note: Only exact matches are returned for temperature > 0
        since responses should vary.
        """
        # Build cache key including model
        cache_key = f"{model}:{query}"

        # For non-deterministic requests, only use exact match
        use_semantic = temperature == 0.0

        result = await self._cache.get(cache_key, use_semantic=use_semantic)

        if result:
            response, similarity = result
            self._tokens_served += response.get("total_tokens", 0)
            return response

        return None

    async def set(
        self,
        query: str,
        model: str,
        response: Dict[str, Any],
        temperature: float = 0.0,
        ttl_seconds: Optional[int] = None,
    ) -> None:
        """
        Cache an LLM response.

        Args:
            query: The query/prompt
            model: Model name
            response: Response dict (should include content and token counts)
            temperature: Temperature used (affects caching strategy)
            ttl_seconds: Cache TTL
        """
        # Don't cache high-temperature responses (they should be varied)
        if temperature > 0.5:
            return

        cache_key = f"{model}:{query}"

        # Higher importance for expensive responses
        importance = 1.0
        if response.get("total_tokens", 0) > 1000:
            importance = 1.5
        if response.get("total_tokens", 0) > 5000:
            importance = 2.0

        await self._cache.set(
            query=cache_key,
            value=response,
            ttl_seconds=ttl_seconds,
            importance=importance,
        )

        self._tokens_saved += response.get("total_tokens", 0)

    async def invalidate_for_model(self, model: str) -> int:
        """Invalidate all cached responses for a model."""
        # This would require iterating through the cache
        # For now, just clear the whole cache
        await self._cache.clear()
        return 0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics including token savings."""
        base_stats = self._cache.get_stats()
        return {
            **base_stats,
            "tokens_saved": self._tokens_saved,
            "tokens_served_from_cache": self._tokens_served,
            "estimated_cost_savings": self._tokens_saved * 0.00001,  # Rough estimate
        }
