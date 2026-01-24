"""
AION Knowledge Graph Store - Abstract Base

Defines the interface for graph storage backends.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from aion.systems.knowledge.types import (
    Entity,
    EntityType,
    Relationship,
    RelationType,
    Path,
    Subgraph,
    GraphStatistics,
)


class GraphStore(ABC):
    """
    Abstract base class for graph storage backends.

    Implementations must provide:
    - Entity CRUD operations
    - Relationship CRUD operations
    - Graph traversal
    - Path finding
    - Statistics

    Optional (with default implementations):
    - Embedding storage
    - Full-text search
    - Transactions
    """

    # ==========================================================================
    # Lifecycle
    # ==========================================================================

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the store (create tables, indices, etc.)."""
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """Gracefully shutdown the store."""
        pass

    async def health_check(self) -> Dict[str, Any]:
        """Check store health."""
        return {"status": "ok"}

    # ==========================================================================
    # Entity Operations
    # ==========================================================================

    @abstractmethod
    async def create_entity(self, entity: Entity) -> str:
        """
        Create a new entity.

        Args:
            entity: The entity to create

        Returns:
            The entity ID
        """
        pass

    @abstractmethod
    async def get_entity(self, entity_id: str) -> Optional[Entity]:
        """
        Get an entity by ID.

        Args:
            entity_id: The entity ID

        Returns:
            The entity or None if not found
        """
        pass

    @abstractmethod
    async def get_entity_by_name(
        self,
        name: str,
        entity_type: Optional[EntityType] = None,
    ) -> Optional[Entity]:
        """
        Get an entity by name (exact match or alias).

        Args:
            name: Entity name or alias
            entity_type: Optional type filter

        Returns:
            The entity or None if not found
        """
        pass

    @abstractmethod
    async def update_entity(self, entity: Entity) -> bool:
        """
        Update an existing entity.

        Args:
            entity: The updated entity

        Returns:
            True if updated successfully
        """
        pass

    @abstractmethod
    async def delete_entity(self, entity_id: str, soft: bool = True) -> bool:
        """
        Delete an entity.

        Args:
            entity_id: The entity ID
            soft: If True, mark as deleted. If False, hard delete.

        Returns:
            True if deleted successfully
        """
        pass

    @abstractmethod
    async def search_entities(
        self,
        query: Optional[str] = None,
        entity_type: Optional[EntityType] = None,
        properties: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Entity]:
        """
        Search for entities.

        Args:
            query: Text search query (name, description)
            entity_type: Filter by type
            properties: Filter by properties
            limit: Maximum results
            offset: Pagination offset

        Returns:
            List of matching entities
        """
        pass

    async def get_entities_by_ids(self, entity_ids: List[str]) -> List[Entity]:
        """Get multiple entities by IDs."""
        entities = []
        for eid in entity_ids:
            entity = await self.get_entity(eid)
            if entity:
                entities.append(entity)
        return entities

    async def entity_exists(self, entity_id: str) -> bool:
        """Check if an entity exists."""
        entity = await self.get_entity(entity_id)
        return entity is not None

    async def count_entities(
        self,
        entity_type: Optional[EntityType] = None,
    ) -> int:
        """Count entities, optionally by type."""
        entities = await self.search_entities(
            entity_type=entity_type,
            limit=1000000,  # High limit for counting
        )
        return len(entities)

    # ==========================================================================
    # Relationship Operations
    # ==========================================================================

    @abstractmethod
    async def create_relationship(self, rel: Relationship) -> str:
        """
        Create a new relationship.

        Args:
            rel: The relationship to create

        Returns:
            The relationship ID
        """
        pass

    @abstractmethod
    async def get_relationship(self, rel_id: str) -> Optional[Relationship]:
        """
        Get a relationship by ID.

        Args:
            rel_id: The relationship ID

        Returns:
            The relationship or None if not found
        """
        pass

    @abstractmethod
    async def get_relationships(
        self,
        entity_id: str,
        direction: str = "both",
        relation_type: Optional[RelationType] = None,
    ) -> List[Relationship]:
        """
        Get relationships for an entity.

        Args:
            entity_id: The entity ID
            direction: "outgoing", "incoming", or "both"
            relation_type: Optional type filter

        Returns:
            List of relationships
        """
        pass

    @abstractmethod
    async def delete_relationship(self, rel_id: str) -> bool:
        """
        Delete a relationship.

        Args:
            rel_id: The relationship ID

        Returns:
            True if deleted successfully
        """
        pass

    async def update_relationship(self, rel: Relationship) -> bool:
        """Update an existing relationship."""
        # Default implementation: delete and recreate
        await self.delete_relationship(rel.id)
        await self.create_relationship(rel)
        return True

    async def get_relationship_between(
        self,
        source_id: str,
        target_id: str,
        relation_type: Optional[RelationType] = None,
    ) -> Optional[Relationship]:
        """Get relationship between two specific entities."""
        rels = await self.get_relationships(source_id, direction="outgoing", relation_type=relation_type)
        for rel in rels:
            if rel.target_id == target_id:
                return rel
        return None

    async def relationship_exists(
        self,
        source_id: str,
        target_id: str,
        relation_type: Optional[RelationType] = None,
    ) -> bool:
        """Check if a relationship exists between two entities."""
        rel = await self.get_relationship_between(source_id, target_id, relation_type)
        return rel is not None

    async def count_relationships(
        self,
        relation_type: Optional[RelationType] = None,
    ) -> int:
        """Count relationships, optionally by type."""
        # Default implementation - subclasses should override for efficiency
        stats = await self.get_stats()
        if relation_type:
            return stats.relationships_by_type.get(relation_type.value, 0)
        return stats.total_relationships

    # ==========================================================================
    # Graph Traversal
    # ==========================================================================

    @abstractmethod
    async def get_neighbors(
        self,
        entity_id: str,
        relation_types: Optional[List[RelationType]] = None,
        depth: int = 1,
    ) -> Subgraph:
        """
        Get neighboring entities up to a certain depth.

        Args:
            entity_id: Starting entity
            relation_types: Filter by relationship types
            depth: Maximum traversal depth

        Returns:
            Subgraph containing neighbors and relationships
        """
        pass

    @abstractmethod
    async def find_path(
        self,
        start_id: str,
        end_id: str,
        max_depth: int = 5,
        relation_types: Optional[List[RelationType]] = None,
    ) -> Optional[Path]:
        """
        Find shortest path between two entities.

        Args:
            start_id: Starting entity ID
            end_id: Target entity ID
            max_depth: Maximum path length
            relation_types: Filter by relationship types

        Returns:
            Path if found, None otherwise
        """
        pass

    async def find_all_paths(
        self,
        start_id: str,
        end_id: str,
        max_depth: int = 5,
        max_paths: int = 10,
        relation_types: Optional[List[RelationType]] = None,
    ) -> List[Path]:
        """
        Find all paths between two entities (up to limit).

        Default implementation uses DFS. Subclasses can override.
        """
        paths = []

        async def dfs(current: str, target: str, visited: set, path_entities: list, path_rels: list):
            if len(paths) >= max_paths:
                return
            if len(path_rels) >= max_depth:
                return
            if current == target:
                # Found a path
                path = Path(
                    entities=path_entities.copy(),
                    relationships=path_rels.copy(),
                )
                path.compute_metrics()
                paths.append(path)
                return

            visited.add(current)

            rels = await self.get_relationships(current, direction="both")
            for rel in rels:
                if relation_types and rel.relation_type not in relation_types:
                    continue

                next_id = rel.target_id if rel.source_id == current else rel.source_id
                if next_id in visited:
                    continue

                next_entity = await self.get_entity(next_id)
                if next_entity:
                    await dfs(
                        next_id,
                        target,
                        visited,
                        path_entities + [next_entity],
                        path_rels + [rel],
                    )

            visited.remove(current)

        start_entity = await self.get_entity(start_id)
        if start_entity:
            await dfs(start_id, end_id, set(), [start_entity], [])

        return paths

    async def get_connected_component(self, entity_id: str) -> Subgraph:
        """Get the connected component containing an entity."""
        return await self.get_neighbors(entity_id, depth=1000)

    # ==========================================================================
    # Embedding Operations (Optional)
    # ==========================================================================

    async def save_embedding(
        self,
        entity_id: str,
        embedding: List[float],
        embedding_type: str = "text",
        model_id: Optional[str] = None,
    ) -> None:
        """
        Save an embedding for an entity.

        Default implementation stores in entity properties.
        """
        entity = await self.get_entity(entity_id)
        if entity:
            entity.properties[f"embedding_{embedding_type}"] = embedding
            entity.properties[f"embedding_{embedding_type}_model"] = model_id
            await self.update_entity(entity)

    async def get_embedding(
        self,
        entity_id: str,
        embedding_type: str = "text",
    ) -> Optional[List[float]]:
        """Get an embedding for an entity."""
        entity = await self.get_entity(entity_id)
        if entity:
            return entity.properties.get(f"embedding_{embedding_type}")
        return None

    async def find_similar_by_embedding(
        self,
        embedding: List[float],
        embedding_type: str = "text",
        limit: int = 10,
        threshold: float = 0.7,
    ) -> List[Tuple[Entity, float]]:
        """
        Find entities similar to a given embedding.

        Default implementation is O(n) - subclasses should override with
        proper vector index.
        """
        import numpy as np

        results = []
        entities = await self.search_entities(limit=10000)

        query_vec = np.array(embedding)
        query_norm = np.linalg.norm(query_vec)

        for entity in entities:
            entity_emb = entity.properties.get(f"embedding_{embedding_type}")
            if entity_emb:
                entity_vec = np.array(entity_emb)
                entity_norm = np.linalg.norm(entity_vec)

                if query_norm > 0 and entity_norm > 0:
                    similarity = np.dot(query_vec, entity_vec) / (query_norm * entity_norm)
                    if similarity >= threshold:
                        results.append((entity, float(similarity)))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    # ==========================================================================
    # Statistics
    # ==========================================================================

    @abstractmethod
    async def get_stats(self) -> GraphStatistics:
        """
        Get comprehensive graph statistics.

        Returns:
            GraphStatistics object
        """
        pass

    async def get_entity_degree(self, entity_id: str) -> int:
        """Get the degree (number of connections) of an entity."""
        rels = await self.get_relationships(entity_id, direction="both")
        return len(rels)

    async def get_top_entities_by_degree(
        self,
        limit: int = 10,
        entity_type: Optional[EntityType] = None,
    ) -> List[Tuple[Entity, int]]:
        """Get entities with highest degree."""
        entities = await self.search_entities(entity_type=entity_type, limit=10000)
        degrees = []

        for entity in entities:
            degree = await self.get_entity_degree(entity.id)
            degrees.append((entity, degree))

        degrees.sort(key=lambda x: x[1], reverse=True)
        return degrees[:limit]

    # ==========================================================================
    # Transactions (Optional)
    # ==========================================================================

    async def begin_transaction(self) -> str:
        """Begin a transaction. Returns transaction ID."""
        raise NotImplementedError("Transactions not supported by this store")

    async def commit_transaction(self, tx_id: str) -> None:
        """Commit a transaction."""
        raise NotImplementedError("Transactions not supported by this store")

    async def rollback_transaction(self, tx_id: str) -> None:
        """Rollback a transaction."""
        raise NotImplementedError("Transactions not supported by this store")

    # ==========================================================================
    # Bulk Operations
    # ==========================================================================

    async def bulk_create_entities(self, entities: List[Entity]) -> List[str]:
        """Create multiple entities in bulk."""
        ids = []
        for entity in entities:
            eid = await self.create_entity(entity)
            ids.append(eid)
        return ids

    async def bulk_create_relationships(self, relationships: List[Relationship]) -> List[str]:
        """Create multiple relationships in bulk."""
        ids = []
        for rel in relationships:
            rid = await self.create_relationship(rel)
            ids.append(rid)
        return ids

    async def bulk_delete_entities(self, entity_ids: List[str], soft: bool = True) -> int:
        """Delete multiple entities in bulk."""
        count = 0
        for eid in entity_ids:
            if await self.delete_entity(eid, soft=soft):
                count += 1
        return count

    # ==========================================================================
    # Import/Export
    # ==========================================================================

    async def export_subgraph(self, entity_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Export a subgraph as JSON.

        Args:
            entity_ids: If provided, export only these entities and their relationships.
                       If None, export entire graph.
        """
        if entity_ids:
            entities = await self.get_entities_by_ids(entity_ids)
        else:
            entities = await self.search_entities(limit=1000000)

        # Get all relationships between these entities
        entity_id_set = {e.id for e in entities}
        relationships = []

        for entity in entities:
            rels = await self.get_relationships(entity.id, direction="outgoing")
            for rel in rels:
                if rel.target_id in entity_id_set:
                    relationships.append(rel)

        return {
            "entities": [e.to_dict() for e in entities],
            "relationships": [r.to_dict() for r in relationships],
            "exported_at": datetime.now().isoformat(),
        }

    async def import_subgraph(
        self,
        data: Dict[str, Any],
        merge: bool = True,
    ) -> Dict[str, int]:
        """
        Import a subgraph from JSON.

        Args:
            data: JSON data with entities and relationships
            merge: If True, merge with existing. If False, fail on conflicts.

        Returns:
            Counts of imported entities and relationships
        """
        entities_imported = 0
        relationships_imported = 0

        # Map old IDs to new IDs
        id_map = {}

        # Import entities
        for e_data in data.get("entities", []):
            entity = Entity(
                name=e_data["name"],
                entity_type=EntityType(e_data["type"]),
                description=e_data.get("description", ""),
                properties=e_data.get("properties", {}),
            )

            if merge:
                existing = await self.get_entity_by_name(entity.name, entity.entity_type)
                if existing:
                    id_map[e_data["id"]] = existing.id
                    existing.merge_from(entity)
                    await self.update_entity(existing)
                    entities_imported += 1
                    continue

            await self.create_entity(entity)
            id_map[e_data["id"]] = entity.id
            entities_imported += 1

        # Import relationships
        for r_data in data.get("relationships", []):
            source_id = id_map.get(r_data["source_id"], r_data["source_id"])
            target_id = id_map.get(r_data["target_id"], r_data["target_id"])

            if not await self.entity_exists(source_id) or not await self.entity_exists(target_id):
                continue

            rel = Relationship(
                source_id=source_id,
                target_id=target_id,
                relation_type=RelationType(r_data["type"]),
                properties=r_data.get("properties", {}),
                weight=r_data.get("weight", 1.0),
                confidence=r_data.get("confidence", 1.0),
            )

            if merge:
                existing = await self.get_relationship_between(source_id, target_id, rel.relation_type)
                if existing:
                    continue

            await self.create_relationship(rel)
            relationships_imported += 1

        return {
            "entities_imported": entities_imported,
            "relationships_imported": relationships_imported,
        }
