"""
AION Hybrid Search Reranker

Learned reranking for hybrid search results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class RerankerConfig:
    """Configuration for reranker."""
    use_llm: bool = False
    cross_encoder_model: Optional[str] = None
    max_candidates: int = 50
    query_relevance_weight: float = 0.4
    entity_quality_weight: float = 0.3
    diversity_weight: float = 0.3


class HybridReranker:
    """
    Reranker for hybrid search results.

    Reranking strategies:
    1. Cross-encoder scoring (if available)
    2. LLM-based relevance scoring
    3. Feature-based reranking
    4. Maximal Marginal Relevance (MMR) for diversity
    """

    def __init__(self, config: Optional[RerankerConfig] = None):
        self.config = config or RerankerConfig()
        self._cross_encoder = None
        self._llm = None

    async def _load_cross_encoder(self) -> None:
        """Lazy load cross-encoder model."""
        if self._cross_encoder is not None:
            return

        if not self.config.cross_encoder_model:
            return

        try:
            from sentence_transformers import CrossEncoder
            self._cross_encoder = CrossEncoder(self.config.cross_encoder_model)
        except ImportError:
            logger.warning("sentence-transformers not available")

    async def rerank(
        self,
        results: List[Any],  # List[HybridResult]
        query: str,
        diversity: float = 0.3,
    ) -> List[Any]:
        """
        Rerank search results.

        Args:
            results: Initial search results
            query: Original search query
            diversity: Weight for diversity (0 = no diversity, 1 = max diversity)

        Returns:
            Reranked results
        """
        if not results:
            return results

        # Truncate to max candidates
        candidates = results[:self.config.max_candidates]

        # 1. Compute relevance scores
        relevance_scores = await self._compute_relevance(candidates, query)

        # 2. Compute quality scores
        quality_scores = self._compute_quality(candidates)

        # 3. Apply MMR for diversity
        reranked = self._mmr_rerank(
            candidates,
            relevance_scores,
            quality_scores,
            diversity,
        )

        return reranked

    async def _compute_relevance(
        self,
        results: List[Any],
        query: str,
    ) -> List[float]:
        """Compute relevance scores for each result."""
        scores = []

        # Try cross-encoder first
        await self._load_cross_encoder()

        if self._cross_encoder:
            pairs = [(query, r.entity.name + " " + r.entity.description) for r in results]
            try:
                ce_scores = self._cross_encoder.predict(pairs)
                return [float(s) for s in ce_scores]
            except Exception as e:
                logger.warning(f"Cross-encoder failed: {e}")

        # Fall back to feature-based scoring
        query_lower = query.lower()
        query_words = set(query_lower.split())

        for result in results:
            entity = result.entity

            # Name overlap
            name_lower = entity.name.lower()
            name_words = set(name_lower.split())
            name_overlap = len(query_words & name_words) / max(len(query_words), 1)

            # Description overlap
            desc_lower = entity.description.lower()
            desc_words = set(desc_lower.split())
            desc_overlap = len(query_words & desc_words) / max(len(query_words), 1)

            # Exact match boost
            exact_match = 1.0 if query_lower in name_lower else 0.0

            # Combine
            score = (
                0.4 * exact_match +
                0.4 * name_overlap +
                0.2 * desc_overlap
            )

            scores.append(score)

        return scores

    def _compute_quality(self, results: List[Any]) -> List[float]:
        """Compute entity quality scores."""
        scores = []

        for result in results:
            entity = result.entity

            # Factors:
            # - Confidence
            # - Importance/PageRank
            # - Description length (proxy for completeness)
            # - Number of properties

            confidence = entity.confidence
            importance = entity.importance
            completeness = min(len(entity.description) / 100, 1.0)
            richness = min(len(entity.properties) / 10, 1.0)

            score = (
                0.3 * confidence +
                0.3 * importance +
                0.2 * completeness +
                0.2 * richness
            )

            scores.append(score)

        return scores

    def _mmr_rerank(
        self,
        results: List[Any],
        relevance_scores: List[float],
        quality_scores: List[float],
        diversity_weight: float,
    ) -> List[Any]:
        """
        Maximal Marginal Relevance reranking.

        Balances relevance with diversity to avoid redundant results.
        """
        if not results:
            return results

        n = len(results)

        # Combine relevance and quality
        combined_scores = [
            self.config.query_relevance_weight * relevance_scores[i] +
            self.config.entity_quality_weight * quality_scores[i]
            for i in range(n)
        ]

        # Compute pairwise similarity (for diversity)
        similarity_matrix = self._compute_similarity_matrix(results)

        # Greedy MMR selection
        selected_indices = []
        remaining_indices = list(range(n))

        while remaining_indices and len(selected_indices) < n:
            best_idx = None
            best_score = float('-inf')

            for idx in remaining_indices:
                # Relevance component
                relevance = combined_scores[idx]

                # Diversity component (max similarity to already selected)
                if selected_indices:
                    max_sim = max(
                        similarity_matrix[idx][sel_idx]
                        for sel_idx in selected_indices
                    )
                else:
                    max_sim = 0.0

                # MMR score
                mmr_score = (1 - diversity_weight) * relevance - diversity_weight * max_sim

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx

            if best_idx is not None:
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)

                # Update combined score for display
                results[best_idx].combined_score = combined_scores[best_idx]

        # Return reranked results
        return [results[i] for i in selected_indices]

    def _compute_similarity_matrix(
        self,
        results: List[Any],
    ) -> List[List[float]]:
        """Compute pairwise similarity between results."""
        n = len(results)
        matrix = [[0.0] * n for _ in range(n)]

        for i in range(n):
            for j in range(i + 1, n):
                sim = self._entity_similarity(
                    results[i].entity,
                    results[j].entity,
                )
                matrix[i][j] = sim
                matrix[j][i] = sim

        return matrix

    def _entity_similarity(self, e1: Any, e2: Any) -> float:
        """Compute similarity between two entities."""
        # Type similarity
        type_sim = 1.0 if e1.entity_type == e2.entity_type else 0.0

        # Name similarity (Jaccard)
        words1 = set(e1.name.lower().split())
        words2 = set(e2.name.lower().split())
        if words1 or words2:
            name_sim = len(words1 & words2) / len(words1 | words2)
        else:
            name_sim = 0.0

        # Description similarity
        desc1 = set(e1.description.lower().split())
        desc2 = set(e2.description.lower().split())
        if desc1 or desc2:
            desc_sim = len(desc1 & desc2) / len(desc1 | desc2)
        else:
            desc_sim = 0.0

        return 0.4 * type_sim + 0.4 * name_sim + 0.2 * desc_sim
