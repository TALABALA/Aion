"""
AION Hybrid Search

State-of-the-art hybrid search combining:
- Vector similarity search (semantic)
- Graph traversal (structural)
- Learned reranking
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import structlog

from aion.systems.knowledge.types import (
    Entity,
    EntityType,
    Relationship,
    Path,
    Subgraph,
)
from aion.systems.knowledge.store.base import GraphStore
from aion.systems.knowledge.hybrid.reranker import HybridReranker

logger = structlog.get_logger(__name__)


@dataclass
class HybridResult:
    """Result from hybrid search."""
    entity: Entity
    vector_score: float = 0.0
    graph_score: float = 0.0
    text_score: float = 0.0
    combined_score: float = 0.0
    path_from_seed: Optional[Path] = None
    explanation: str = ""

    def to_dict(self) -> dict:
        return {
            "entity": self.entity.to_dict(),
            "vector_score": self.vector_score,
            "graph_score": self.graph_score,
            "text_score": self.text_score,
            "combined_score": self.combined_score,
            "path": self.path_from_seed.to_string() if self.path_from_seed else None,
            "explanation": self.explanation,
        }


@dataclass
class SearchConfig:
    """Configuration for hybrid search."""
    vector_weight: float = 0.4
    graph_weight: float = 0.3
    text_weight: float = 0.3
    max_graph_depth: int = 2
    enable_reranking: bool = True
    boost_connected: float = 1.2
    decay_factor: float = 0.8


class HybridSearch:
    """
    Hybrid search combining vector similarity and graph structure.

    Strategy:
    1. Vector search finds semantically similar entities
    2. Graph traversal finds structurally related entities
    3. Text search for exact/fuzzy matches
    4. Results are combined and reranked using learned weights
    """

    def __init__(
        self,
        store: GraphStore,
        memory_system: Optional[Any] = None,
        config: Optional[SearchConfig] = None,
    ):
        self.store = store
        self.memory = memory_system
        self.config = config or SearchConfig()
        self.reranker = HybridReranker()

        # Embedding function (lazy loaded)
        self._embed_fn = None

    async def _ensure_embedding(self) -> None:
        """Lazy load embedding function."""
        if self._embed_fn is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
            self._embed_fn = lambda text: model.encode(text).tolist()
        except ImportError:
            logger.warning("sentence-transformers not available, vector search disabled")
            self._embed_fn = None

    async def search(
        self,
        query: str,
        entity_types: Optional[List[EntityType]] = None,
        seed_entity_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 20,
    ) -> List[HybridResult]:
        """
        Perform hybrid search.

        Args:
            query: Search query (natural language)
            entity_types: Filter by entity types
            seed_entity_id: Start graph traversal from this entity
            filters: Additional property filters
            limit: Maximum results

        Returns:
            Ranked list of hybrid results
        """
        results_map: Dict[str, HybridResult] = {}

        # 1. Vector search
        vector_results = await self._vector_search(query, entity_types, limit * 3)
        for entity, score in vector_results:
            if entity.id not in results_map:
                results_map[entity.id] = HybridResult(entity=entity)
            results_map[entity.id].vector_score = score

        # 2. Text search
        text_results = await self._text_search(query, entity_types, limit * 3)
        for entity, score in text_results:
            if entity.id not in results_map:
                results_map[entity.id] = HybridResult(entity=entity)
            results_map[entity.id].text_score = score

        # 3. Graph traversal from seed
        if seed_entity_id:
            graph_results = await self._graph_search(
                seed_entity_id,
                entity_types,
                self.config.max_graph_depth,
                limit * 3,
            )
            for entity, score, path in graph_results:
                if entity.id not in results_map:
                    results_map[entity.id] = HybridResult(entity=entity)
                results_map[entity.id].graph_score = score
                results_map[entity.id].path_from_seed = path

        # 4. Apply filters
        if filters:
            results_map = {
                eid: r for eid, r in results_map.items()
                if self._matches_filters(r.entity, filters)
            }

        # 5. Combine scores
        for result in results_map.values():
            result.combined_score = self._combine_scores(result)
            result.explanation = self._generate_explanation(result)

        # 6. Rerank if enabled
        results = list(results_map.values())
        if self.config.enable_reranking and results:
            results = await self.reranker.rerank(results, query)

        # 7. Sort and return
        results.sort(key=lambda r: r.combined_score, reverse=True)

        return results[:limit]

    async def _vector_search(
        self,
        query: str,
        entity_types: Optional[List[EntityType]],
        limit: int,
    ) -> List[Tuple[Entity, float]]:
        """Perform vector similarity search."""
        await self._ensure_embedding()

        if not self._embed_fn:
            return []

        try:
            # Embed query
            query_embedding = self._embed_fn(query)

            # Search store
            results = await self.store.find_similar_by_embedding(
                query_embedding,
                embedding_type="text",
                limit=limit * 2,
                threshold=0.5,
            )

            # Filter by type
            if entity_types:
                results = [
                    (e, s) for e, s in results
                    if e.entity_type in entity_types
                ]

            return results[:limit]

        except Exception as e:
            logger.warning(f"Vector search failed: {e}")
            return []

    async def _text_search(
        self,
        query: str,
        entity_types: Optional[List[EntityType]],
        limit: int,
    ) -> List[Tuple[Entity, float]]:
        """Perform text-based search."""
        entity_type = entity_types[0] if entity_types and len(entity_types) == 1 else None

        entities = await self.store.search_entities(
            query=query,
            entity_type=entity_type,
            limit=limit,
        )

        # Score based on match quality
        results = []
        query_lower = query.lower()

        for i, entity in enumerate(entities):
            # Base score from position
            position_score = 1.0 - (i / max(len(entities), 1))

            # Boost for exact matches
            name_lower = entity.name.lower()
            if name_lower == query_lower:
                score = 1.0
            elif query_lower in name_lower:
                score = 0.9
            elif any(query_lower in alias.lower() for alias in entity.aliases):
                score = 0.8
            else:
                score = position_score * 0.7

            if not entity_types or entity.entity_type in entity_types:
                results.append((entity, score))

        return results[:limit]

    async def _graph_search(
        self,
        seed_id: str,
        entity_types: Optional[List[EntityType]],
        max_depth: int,
        limit: int,
    ) -> List[Tuple[Entity, float, Optional[Path]]]:
        """Perform graph traversal from seed entity."""
        results = []
        visited = {seed_id}

        # BFS with distance tracking
        from collections import deque
        queue = deque([(seed_id, 0, [])])  # (id, depth, path_rels)

        seed_entity = await self.store.get_entity(seed_id)

        while queue and len(results) < limit:
            current_id, depth, path_rels = queue.popleft()

            if depth > max_depth:
                continue

            if depth > 0:  # Don't include seed
                entity = await self.store.get_entity(current_id)
                if entity:
                    if not entity_types or entity.entity_type in entity_types:
                        # Score decays with distance
                        score = self.config.decay_factor ** depth

                        # Boost for importance
                        score *= (0.5 + 0.5 * entity.importance)

                        # Build path
                        path = None
                        if seed_entity and path_rels:
                            path_entities = [seed_entity]
                            for rel in path_rels:
                                e = await self.store.get_entity(rel.target_id)
                                if e:
                                    path_entities.append(e)
                            path = Path(entities=path_entities, relationships=path_rels)
                            path.compute_metrics()

                        results.append((entity, score, path))

            # Get neighbors
            rels = await self.store.get_relationships(current_id, direction="both")
            for rel in rels:
                next_id = rel.target_id if rel.source_id == current_id else rel.source_id

                if next_id not in visited:
                    visited.add(next_id)
                    queue.append((next_id, depth + 1, path_rels + [rel]))

        return results

    def _matches_filters(self, entity: Entity, filters: Dict[str, Any]) -> bool:
        """Check if entity matches property filters."""
        for key, value in filters.items():
            entity_value = entity.properties.get(key)

            if isinstance(value, dict):
                # Range filter
                if "min" in value and (entity_value is None or entity_value < value["min"]):
                    return False
                if "max" in value and (entity_value is None or entity_value > value["max"]):
                    return False
                if "contains" in value and (entity_value is None or value["contains"] not in str(entity_value)):
                    return False
            else:
                if entity_value != value:
                    return False

        return True

    def _combine_scores(self, result: HybridResult) -> float:
        """Combine different scores into final score."""
        weighted_sum = (
            self.config.vector_weight * result.vector_score +
            self.config.graph_weight * result.graph_score +
            self.config.text_weight * result.text_score
        )

        # Boost if entity has multiple signal types
        signals = sum([
            result.vector_score > 0,
            result.graph_score > 0,
            result.text_score > 0,
        ])

        if signals >= 2:
            weighted_sum *= self.config.boost_connected

        # Factor in entity importance
        weighted_sum *= (0.8 + 0.2 * result.entity.importance)

        return weighted_sum

    def _generate_explanation(self, result: HybridResult) -> str:
        """Generate explanation for why this result was returned."""
        parts = []

        if result.text_score > 0.8:
            parts.append("exact name match")
        elif result.text_score > 0.5:
            parts.append("name contains query")

        if result.vector_score > 0.8:
            parts.append("highly semantically similar")
        elif result.vector_score > 0.5:
            parts.append("semantically related")

        if result.graph_score > 0.5:
            if result.path_from_seed:
                parts.append(f"connected via {result.path_from_seed.length} hop(s)")
            else:
                parts.append("structurally related")

        if not parts:
            parts.append("partial match")

        return "; ".join(parts)

    async def expand_entity(
        self,
        entity_id: str,
        max_depth: int = 1,
    ) -> Dict[str, Any]:
        """
        Expand an entity to show its neighborhood.

        Returns structured data about the entity and its connections.
        """
        entity = await self.store.get_entity(entity_id)
        if not entity:
            return {"error": "Entity not found"}

        # Get subgraph
        subgraph = await self.store.get_neighbors(entity_id, depth=max_depth)

        # Organize by relationship type
        connections: Dict[str, Dict[str, List[dict]]] = {}

        for rel in subgraph.relationships.values():
            rel_type = rel.relation_type.value

            if rel_type not in connections:
                connections[rel_type] = {"outgoing": [], "incoming": []}

            if rel.source_id == entity_id:
                target = subgraph.entities.get(rel.target_id)
                if target:
                    connections[rel_type]["outgoing"].append({
                        "entity": target.to_dict(),
                        "confidence": rel.confidence,
                    })
            else:
                source = subgraph.entities.get(rel.source_id)
                if source:
                    connections[rel_type]["incoming"].append({
                        "entity": source.to_dict(),
                        "confidence": rel.confidence,
                    })

        return {
            "entity": entity.to_dict(),
            "connections": connections,
            "total_connections": len(subgraph.relationships),
            "neighbor_count": len(subgraph.entities) - 1,
        }

    async def find_similar_entities(
        self,
        entity_id: str,
        limit: int = 10,
    ) -> List[HybridResult]:
        """Find entities similar to a given entity."""
        entity = await self.store.get_entity(entity_id)
        if not entity:
            return []

        # Use entity name and description as query
        query = f"{entity.name} {entity.description}"

        results = await self.search(
            query=query,
            entity_types=[entity.entity_type],
            seed_entity_id=entity_id,
            limit=limit + 1,  # +1 to exclude self
        )

        # Remove the entity itself
        return [r for r in results if r.entity.id != entity_id][:limit]
