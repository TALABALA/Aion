"""
AION In-Memory Graph Store

Fast in-memory graph storage for testing and small graphs.
"""

from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import structlog

from aion.systems.knowledge.types import (
    Entity,
    EntityType,
    Relationship,
    RelationType,
    Path as GraphPath,
    Subgraph,
    GraphStatistics,
    GraphEmbedding,
)
from aion.systems.knowledge.store.base import GraphStore

logger = structlog.get_logger(__name__)


class InMemoryGraphStore(GraphStore):
    """
    Fast in-memory graph storage.

    Best for:
    - Unit testing
    - Small graphs (< 100K entities)
    - Temporary/session graphs
    - Prototyping

    Features:
    - O(1) entity/relationship lookup
    - O(k) neighbor traversal (k = degree)
    - No persistence (data lost on shutdown)
    """

    def __init__(self):
        # Primary storage
        self._entities: Dict[str, Entity] = {}
        self._relationships: Dict[str, Relationship] = {}

        # Indices for fast lookup
        self._entities_by_name: Dict[str, str] = {}  # name (lower) -> id
        self._aliases: Dict[str, str] = {}  # alias (lower) -> entity_id
        self._entities_by_type: Dict[EntityType, set] = defaultdict(set)

        # Adjacency lists for graph traversal
        self._outgoing: Dict[str, set] = defaultdict(set)  # entity_id -> set of rel_ids
        self._incoming: Dict[str, set] = defaultdict(set)  # entity_id -> set of rel_ids

        # Embeddings
        self._embeddings: Dict[str, Dict[str, List[float]]] = defaultdict(dict)  # entity_id -> type -> vector

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the store."""
        self._initialized = True
        logger.info("In-memory graph store initialized")

    async def shutdown(self) -> None:
        """Clear all data."""
        self._entities.clear()
        self._relationships.clear()
        self._entities_by_name.clear()
        self._aliases.clear()
        self._entities_by_type.clear()
        self._outgoing.clear()
        self._incoming.clear()
        self._embeddings.clear()
        self._initialized = False
        logger.info("In-memory graph store shutdown")

    async def health_check(self) -> Dict[str, Any]:
        """Check store health."""
        return {
            "status": "ok",
            "type": "in_memory",
            "entity_count": len(self._entities),
            "relationship_count": len(self._relationships),
        }

    # ==========================================================================
    # Entity Operations
    # ==========================================================================

    async def create_entity(self, entity: Entity) -> str:
        """Create a new entity."""
        self._entities[entity.id] = entity

        # Update indices
        self._entities_by_name[entity.name.lower()] = entity.id
        self._entities_by_type[entity.entity_type].add(entity.id)

        for alias in entity.aliases:
            self._aliases[alias.lower()] = entity.id

        # Store embeddings
        if entity.text_embedding and entity.text_embedding.vector:
            self._embeddings[entity.id]["text"] = entity.text_embedding.vector
        if entity.graph_embedding and entity.graph_embedding.vector:
            self._embeddings[entity.id]["graph"] = entity.graph_embedding.vector

        return entity.id

    async def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get an entity by ID."""
        entity = self._entities.get(entity_id)
        if entity and entity.deleted_at:
            return None
        return entity

    async def get_entity_by_name(
        self,
        name: str,
        entity_type: Optional[EntityType] = None,
    ) -> Optional[Entity]:
        """Get an entity by name or alias."""
        # Try name
        entity_id = self._entities_by_name.get(name.lower())
        if entity_id:
            entity = self._entities.get(entity_id)
            if entity and not entity.deleted_at:
                if not entity_type or entity.entity_type == entity_type:
                    return entity

        # Try alias
        entity_id = self._aliases.get(name.lower())
        if entity_id:
            entity = self._entities.get(entity_id)
            if entity and not entity.deleted_at:
                if not entity_type or entity.entity_type == entity_type:
                    return entity

        return None

    async def update_entity(self, entity: Entity) -> bool:
        """Update an existing entity."""
        if entity.id not in self._entities:
            return False

        old_entity = self._entities[entity.id]

        # Update indices if name changed
        if old_entity.name.lower() != entity.name.lower():
            del self._entities_by_name[old_entity.name.lower()]
            self._entities_by_name[entity.name.lower()] = entity.id

        # Update type index if changed
        if old_entity.entity_type != entity.entity_type:
            self._entities_by_type[old_entity.entity_type].discard(entity.id)
            self._entities_by_type[entity.entity_type].add(entity.id)

        # Update aliases
        for alias in old_entity.aliases:
            if alias.lower() in self._aliases:
                del self._aliases[alias.lower()]
        for alias in entity.aliases:
            self._aliases[alias.lower()] = entity.id

        entity.updated_at = datetime.now()
        self._entities[entity.id] = entity

        return True

    async def delete_entity(self, entity_id: str, soft: bool = True) -> bool:
        """Delete an entity."""
        if entity_id not in self._entities:
            return False

        if soft:
            self._entities[entity_id].deleted_at = datetime.now()
        else:
            entity = self._entities[entity_id]

            # Remove from indices
            if entity.name.lower() in self._entities_by_name:
                del self._entities_by_name[entity.name.lower()]
            self._entities_by_type[entity.entity_type].discard(entity_id)
            for alias in entity.aliases:
                if alias.lower() in self._aliases:
                    del self._aliases[alias.lower()]

            # Remove relationships
            for rel_id in list(self._outgoing.get(entity_id, [])):
                await self.delete_relationship(rel_id)
            for rel_id in list(self._incoming.get(entity_id, [])):
                await self.delete_relationship(rel_id)

            # Remove entity
            del self._entities[entity_id]

            # Remove embeddings
            if entity_id in self._embeddings:
                del self._embeddings[entity_id]

        return True

    async def search_entities(
        self,
        query: Optional[str] = None,
        entity_type: Optional[EntityType] = None,
        properties: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Entity]:
        """Search for entities."""
        results = []

        # Start with type filter if provided
        if entity_type:
            candidate_ids = self._entities_by_type.get(entity_type, set())
        else:
            candidate_ids = set(self._entities.keys())

        for entity_id in candidate_ids:
            entity = self._entities.get(entity_id)
            if not entity or entity.deleted_at:
                continue

            # Text search
            if query:
                query_lower = query.lower()
                if not (
                    query_lower in entity.name.lower() or
                    query_lower in entity.description.lower() or
                    any(query_lower in a.lower() for a in entity.aliases)
                ):
                    continue

            # Property filter
            if properties:
                match = True
                for key, value in properties.items():
                    if entity.properties.get(key) != value:
                        match = False
                        break
                if not match:
                    continue

            results.append(entity)

        # Sort by importance
        results.sort(key=lambda e: (e.importance, e.name), reverse=True)

        return results[offset:offset + limit]

    async def count_entities(self, entity_type: Optional[EntityType] = None) -> int:
        """Count entities."""
        if entity_type:
            return sum(1 for eid in self._entities_by_type.get(entity_type, [])
                      if not self._entities[eid].deleted_at)
        return sum(1 for e in self._entities.values() if not e.deleted_at)

    # ==========================================================================
    # Relationship Operations
    # ==========================================================================

    async def create_relationship(self, rel: Relationship) -> str:
        """Create a new relationship."""
        self._relationships[rel.id] = rel

        # Update adjacency lists
        self._outgoing[rel.source_id].add(rel.id)
        self._incoming[rel.target_id].add(rel.id)

        # Handle bidirectional
        if rel.bidirectional:
            self._outgoing[rel.target_id].add(rel.id)
            self._incoming[rel.source_id].add(rel.id)

        return rel.id

    async def get_relationship(self, rel_id: str) -> Optional[Relationship]:
        """Get a relationship by ID."""
        return self._relationships.get(rel_id)

    async def get_relationships(
        self,
        entity_id: str,
        direction: str = "both",
        relation_type: Optional[RelationType] = None,
    ) -> List[Relationship]:
        """Get relationships for an entity."""
        rel_ids = set()

        if direction in ("outgoing", "both"):
            rel_ids.update(self._outgoing.get(entity_id, set()))

        if direction in ("incoming", "both"):
            rel_ids.update(self._incoming.get(entity_id, set()))

        relationships = []
        for rel_id in rel_ids:
            rel = self._relationships.get(rel_id)
            if rel:
                if relation_type and rel.relation_type != relation_type:
                    continue
                relationships.append(rel)

        return relationships

    async def delete_relationship(self, rel_id: str) -> bool:
        """Delete a relationship."""
        if rel_id not in self._relationships:
            return False

        rel = self._relationships[rel_id]

        # Remove from adjacency lists
        self._outgoing[rel.source_id].discard(rel_id)
        self._incoming[rel.target_id].discard(rel_id)

        if rel.bidirectional:
            self._outgoing[rel.target_id].discard(rel_id)
            self._incoming[rel.source_id].discard(rel_id)

        del self._relationships[rel_id]

        return True

    # ==========================================================================
    # Graph Traversal
    # ==========================================================================

    async def get_neighbors(
        self,
        entity_id: str,
        relation_types: Optional[List[RelationType]] = None,
        depth: int = 1,
    ) -> Subgraph:
        """Get neighboring entities using BFS."""
        subgraph = Subgraph()
        visited = set()

        queue = deque([(entity_id, 0)])

        while queue:
            current_id, current_depth = queue.popleft()

            if current_id in visited or current_depth > depth:
                continue

            visited.add(current_id)

            entity = await self.get_entity(current_id)
            if entity:
                subgraph.add_entity(entity)

            if current_depth < depth:
                relationships = await self.get_relationships(current_id, direction="both")

                for rel in relationships:
                    if relation_types and rel.relation_type not in relation_types:
                        continue

                    subgraph.add_relationship(rel)

                    next_id = rel.target_id if rel.source_id == current_id else rel.source_id
                    if next_id not in visited:
                        queue.append((next_id, current_depth + 1))

        return subgraph

    async def find_path(
        self,
        start_id: str,
        end_id: str,
        max_depth: int = 5,
        relation_types: Optional[List[RelationType]] = None,
    ) -> Optional[GraphPath]:
        """Find shortest path using BFS."""
        if start_id == end_id:
            entity = await self.get_entity(start_id)
            return GraphPath(entities=[entity] if entity else [], relationships=[])

        # BFS
        visited = {start_id}
        queue = deque([(start_id, [], [])])

        while queue:
            current_id, path_entities, path_rels = queue.popleft()

            if len(path_rels) >= max_depth:
                continue

            relationships = await self.get_relationships(current_id, direction="both")

            for rel in relationships:
                if relation_types and rel.relation_type not in relation_types:
                    continue

                next_id = rel.target_id if rel.source_id == current_id else rel.source_id

                if next_id in visited:
                    continue

                visited.add(next_id)

                entity = await self.get_entity(next_id)
                if not entity:
                    continue

                new_entities = path_entities + [entity]
                new_rels = path_rels + [rel]

                if next_id == end_id:
                    start_entity = await self.get_entity(start_id)
                    path = GraphPath(
                        entities=[start_entity] + new_entities,
                        relationships=new_rels,
                    )
                    path.compute_metrics()
                    return path

                queue.append((next_id, new_entities, new_rels))

        return None

    # ==========================================================================
    # Embedding Operations
    # ==========================================================================

    async def save_embedding(
        self,
        entity_id: str,
        embedding: List[float],
        embedding_type: str = "text",
        model_id: Optional[str] = None,
    ) -> None:
        """Save an embedding."""
        self._embeddings[entity_id][embedding_type] = embedding

        # Also update entity if it exists
        if entity_id in self._entities:
            emb = GraphEmbedding(
                embedding_type=embedding_type,
                vector=embedding,
                model_id=model_id,
            )
            if embedding_type == "text":
                self._entities[entity_id].text_embedding = emb
            elif embedding_type == "graph":
                self._entities[entity_id].graph_embedding = emb

    async def get_embedding(
        self,
        entity_id: str,
        embedding_type: str = "text",
    ) -> Optional[List[float]]:
        """Get an embedding."""
        return self._embeddings.get(entity_id, {}).get(embedding_type)

    async def find_similar_by_embedding(
        self,
        embedding: List[float],
        embedding_type: str = "text",
        limit: int = 10,
        threshold: float = 0.7,
    ) -> List[Tuple[Entity, float]]:
        """Find similar entities by embedding."""
        import numpy as np

        query_vec = np.array(embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query_vec)

        if query_norm == 0:
            return []

        results = []

        for entity_id, embeddings in self._embeddings.items():
            entity_emb = embeddings.get(embedding_type)
            if not entity_emb:
                continue

            entity_vec = np.array(entity_emb, dtype=np.float32)
            entity_norm = np.linalg.norm(entity_vec)

            if entity_norm > 0:
                similarity = float(np.dot(query_vec, entity_vec) / (query_norm * entity_norm))
                if similarity >= threshold:
                    entity = await self.get_entity(entity_id)
                    if entity:
                        results.append((entity, similarity))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    # ==========================================================================
    # Statistics
    # ==========================================================================

    async def get_stats(self) -> GraphStatistics:
        """Get graph statistics."""
        stats = GraphStatistics()

        stats.total_entities = await self.count_entities()
        stats.total_relationships = len(self._relationships)

        # By type
        for entity_type, ids in self._entities_by_type.items():
            count = sum(1 for eid in ids if not self._entities[eid].deleted_at)
            if count > 0:
                stats.entities_by_type[entity_type.value] = count

        rel_type_counts: Dict[str, int] = {}
        for rel in self._relationships.values():
            rel_type = rel.relation_type.value
            rel_type_counts[rel_type] = rel_type_counts.get(rel_type, 0) + 1
        stats.relationships_by_type = rel_type_counts

        # Degree statistics
        if stats.total_entities > 0:
            degrees = []
            for entity_id in self._entities:
                if not self._entities[entity_id].deleted_at:
                    degree = len(self._outgoing.get(entity_id, set())) + len(self._incoming.get(entity_id, set()))
                    degrees.append(degree)

            if degrees:
                stats.avg_degree = sum(degrees) / len(degrees)
                stats.max_degree = max(degrees)

            n = stats.total_entities
            if n > 1:
                max_edges = n * (n - 1)
                stats.density = stats.total_relationships / max_edges

        stats.computed_at = datetime.now()

        return stats
