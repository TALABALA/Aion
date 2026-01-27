"""AION Entity Manager - ECS entity management.

Provides:
- EntityManager: Manages entity lifecycle with ECS patterns.
- Spatial and component-based queries.
- Entity archetype system for rapid instantiation.
- Relationship graph between entities.
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Set

import structlog

from aion.simulation.types import Entity, EntityType

logger = structlog.get_logger(__name__)


class EntityManager:
    """Manages entities with ECS-style component queries.

    Features:
    - O(1) entity lookup by ID.
    - Component-indexed queries for fast archetype matching.
    - Relationship graph traversal.
    - Entity archetypes for templated creation.
    - Lifecycle hooks (on_create, on_destroy, on_component_add).
    """

    def __init__(self) -> None:
        self._entities: Dict[str, Entity] = {}

        # Component index: component_name -> set of entity IDs
        self._component_index: Dict[str, Set[str]] = defaultdict(set)

        # Type index: EntityType -> set of entity IDs
        self._type_index: Dict[EntityType, Set[str]] = defaultdict(set)

        # Tag index: tag -> set of entity IDs
        self._tag_index: Dict[str, Set[str]] = defaultdict(set)

        # Archetypes: name -> entity template dict
        self._archetypes: Dict[str, Dict[str, Any]] = {}

        # Lifecycle hooks
        self._on_create: List[Callable[[Entity], None]] = []
        self._on_destroy: List[Callable[[Entity], None]] = []

    # -- Lifecycle --

    def create(
        self,
        entity_type: EntityType,
        name: str = "",
        properties: Optional[Dict[str, Any]] = None,
        components: Optional[Dict[str, Dict[str, Any]]] = None,
        behaviors: Optional[List[str]] = None,
        tags: Optional[Set[str]] = None,
        entity_id: Optional[str] = None,
    ) -> Entity:
        """Create a new entity."""
        entity = Entity(
            id=entity_id or str(uuid.uuid4()),
            type=entity_type,
            name=name,
            properties=properties or {},
            components=components or {},
            behaviors=behaviors or [],
            tags=tags or set(),
        )
        return self.add(entity)

    def add(self, entity: Entity) -> Entity:
        """Add an existing entity to the manager."""
        self._entities[entity.id] = entity
        self._type_index[entity.type].add(entity.id)

        for tag in entity.tags:
            self._tag_index[tag].add(entity.id)

        for comp_name in entity.components:
            self._component_index[comp_name].add(entity.id)

        for hook in self._on_create:
            try:
                hook(entity)
            except Exception as exc:
                logger.error("entity_create_hook_error", entity_id=entity.id, error=str(exc))

        return entity

    def remove(self, entity_id: str) -> Optional[Entity]:
        """Remove an entity."""
        entity = self._entities.pop(entity_id, None)
        if entity is None:
            return None

        self._type_index[entity.type].discard(entity_id)

        for tag in entity.tags:
            self._tag_index[tag].discard(entity_id)

        for comp_name in entity.components:
            self._component_index[comp_name].discard(entity_id)

        for hook in self._on_destroy:
            try:
                hook(entity)
            except Exception as exc:
                logger.error("entity_destroy_hook_error", entity_id=entity.id, error=str(exc))

        return entity

    def get(self, entity_id: str) -> Optional[Entity]:
        """Get entity by ID."""
        return self._entities.get(entity_id)

    def exists(self, entity_id: str) -> bool:
        return entity_id in self._entities

    @property
    def count(self) -> int:
        return len(self._entities)

    @property
    def all_entities(self) -> Dict[str, Entity]:
        return self._entities

    # -- Component Operations --

    def add_component(
        self,
        entity_id: str,
        component_name: str,
        data: Dict[str, Any],
    ) -> bool:
        """Add a component to an entity."""
        entity = self._entities.get(entity_id)
        if entity is None:
            return False
        entity.add_component(component_name, data)
        self._component_index[component_name].add(entity_id)
        return True

    def remove_component(self, entity_id: str, component_name: str) -> bool:
        """Remove a component from an entity."""
        entity = self._entities.get(entity_id)
        if entity is None:
            return False
        entity.remove_component(component_name)
        self._component_index[component_name].discard(entity_id)
        return True

    # -- Queries --

    def query_by_type(self, entity_type: EntityType) -> List[Entity]:
        """Get all entities of a given type."""
        return [
            self._entities[eid]
            for eid in self._type_index.get(entity_type, set())
            if eid in self._entities
        ]

    def query_by_tag(self, tag: str) -> List[Entity]:
        """Get all entities with a given tag."""
        return [
            self._entities[eid]
            for eid in self._tag_index.get(tag, set())
            if eid in self._entities
        ]

    def query_by_component(self, component_name: str) -> List[Entity]:
        """Get all entities having a specific component."""
        return [
            self._entities[eid]
            for eid in self._component_index.get(component_name, set())
            if eid in self._entities
        ]

    def query_by_components(self, *component_names: str) -> List[Entity]:
        """Get entities having ALL specified components."""
        if not component_names:
            return list(self._entities.values())
        sets = [
            self._component_index.get(name, set()) for name in component_names
        ]
        intersection = set.intersection(*sets) if sets else set()
        return [self._entities[eid] for eid in intersection if eid in self._entities]

    def query(
        self,
        entity_type: Optional[EntityType] = None,
        tags: Optional[Set[str]] = None,
        components: Optional[List[str]] = None,
        properties: Optional[Dict[str, Any]] = None,
        predicate: Optional[Callable[[Entity], bool]] = None,
    ) -> List[Entity]:
        """Flexible entity query with multiple filters."""
        # Start with candidate sets based on indexed criteria
        candidate_ids: Optional[Set[str]] = None

        if entity_type is not None:
            type_ids = self._type_index.get(entity_type, set())
            candidate_ids = set(type_ids)

        if tags:
            for tag in tags:
                tag_ids = self._tag_index.get(tag, set())
                if candidate_ids is None:
                    candidate_ids = set(tag_ids)
                else:
                    candidate_ids &= tag_ids

        if components:
            for comp in components:
                comp_ids = self._component_index.get(comp, set())
                if candidate_ids is None:
                    candidate_ids = set(comp_ids)
                else:
                    candidate_ids &= comp_ids

        if candidate_ids is None:
            candidate_ids = set(self._entities.keys())

        results: List[Entity] = []
        for eid in candidate_ids:
            entity = self._entities.get(eid)
            if entity is None:
                continue

            if properties:
                if not all(entity.properties.get(k) == v for k, v in properties.items()):
                    continue

            if predicate and not predicate(entity):
                continue

            results.append(entity)

        return results

    # -- Archetypes --

    def register_archetype(self, name: str, template: Dict[str, Any]) -> None:
        """Register an entity archetype for rapid creation."""
        self._archetypes[name] = template

    def create_from_archetype(
        self,
        archetype_name: str,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> Optional[Entity]:
        """Create an entity from an archetype template."""
        template = self._archetypes.get(archetype_name)
        if template is None:
            logger.warning("archetype_not_found", name=archetype_name)
            return None

        merged = {**template, **(overrides or {})}
        return self.create(
            entity_type=EntityType(merged.get("type", "resource")),
            name=merged.get("name", archetype_name),
            properties=merged.get("properties", {}),
            components=merged.get("components", {}),
            behaviors=merged.get("behaviors", []),
            tags=set(merged.get("tags", [])),
        )

    # -- Relationship Graph --

    def get_children(self, entity_id: str) -> List[Entity]:
        """Get child entities."""
        entity = self._entities.get(entity_id)
        if entity is None:
            return []
        return [
            self._entities[cid]
            for cid in entity.children_ids
            if cid in self._entities
        ]

    def get_parent(self, entity_id: str) -> Optional[Entity]:
        """Get parent entity."""
        entity = self._entities.get(entity_id)
        if entity is None or entity.parent_id is None:
            return None
        return self._entities.get(entity.parent_id)

    def get_related(self, entity_id: str, relationship: str) -> List[Entity]:
        """Get entities related by a specific relationship."""
        entity = self._entities.get(entity_id)
        if entity is None:
            return []
        related_ids = entity.relationships.get(relationship, [])
        return [self._entities[rid] for rid in related_ids if rid in self._entities]

    def add_relationship(
        self,
        from_id: str,
        to_id: str,
        relationship: str,
        bidirectional: bool = False,
    ) -> bool:
        """Add a relationship between entities."""
        from_entity = self._entities.get(from_id)
        to_entity = self._entities.get(to_id)
        if from_entity is None or to_entity is None:
            return False

        if relationship not in from_entity.relationships:
            from_entity.relationships[relationship] = []
        if to_id not in from_entity.relationships[relationship]:
            from_entity.relationships[relationship].append(to_id)

        if bidirectional:
            inverse = f"inverse_{relationship}"
            if inverse not in to_entity.relationships:
                to_entity.relationships[inverse] = []
            if from_id not in to_entity.relationships[inverse]:
                to_entity.relationships[inverse].append(from_id)

        return True

    # -- Lifecycle Hooks --

    def on_create(self, hook: Callable[[Entity], None]) -> None:
        self._on_create.append(hook)

    def on_destroy(self, hook: Callable[[Entity], None]) -> None:
        self._on_destroy.append(hook)

    # -- Bulk Operations --

    def clear(self) -> None:
        """Remove all entities."""
        self._entities.clear()
        self._component_index.clear()
        self._type_index.clear()
        self._tag_index.clear()

    def load_entities(self, entities: Dict[str, Entity]) -> None:
        """Bulk load entities (for snapshot restore)."""
        self.clear()
        for entity in entities.values():
            self.add(entity)

    def create_from_data(self, data: Dict[str, Any]) -> Entity:
        """Create entity from dictionary data."""
        return self.create(
            entity_type=EntityType(data.get("type", "resource")),
            name=data.get("name", ""),
            properties=data.get("properties", {}),
            components=data.get("components", {}),
            behaviors=data.get("behaviors", []),
            tags=set(data.get("tags", [])),
            entity_id=data.get("id"),
        )
