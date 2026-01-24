"""
AION Knowledge Graph Ontology Schema

Schema definition and management for the knowledge graph.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import structlog

from aion.systems.knowledge.types import EntityType, RelationType

logger = structlog.get_logger(__name__)


@dataclass
class PropertyDefinition:
    """Definition of an entity/relationship property."""
    name: str
    data_type: str = "string"  # string, number, boolean, date, json, embedding
    required: bool = False
    indexed: bool = False
    searchable: bool = True
    default: Optional[Any] = None
    description: str = ""
    constraints: Dict[str, Any] = field(default_factory=dict)  # min, max, pattern, enum

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "data_type": self.data_type,
            "required": self.required,
            "indexed": self.indexed,
            "description": self.description,
            "constraints": self.constraints,
        }


@dataclass
class EntityTypeDefinition:
    """Definition of an entity type."""
    name: str
    parent: Optional[str] = None  # For inheritance
    description: str = ""
    properties: List[PropertyDefinition] = field(default_factory=list)
    required_properties: List[str] = field(default_factory=list)
    allowed_relationships: List[str] = field(default_factory=list)  # Outgoing
    icon: str = ""
    color: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "parent": self.parent,
            "description": self.description,
            "properties": [p.to_dict() for p in self.properties],
            "required_properties": self.required_properties,
            "allowed_relationships": self.allowed_relationships,
        }


@dataclass
class RelationTypeDefinition:
    """Definition of a relationship type."""
    name: str
    description: str = ""
    source_types: List[str] = field(default_factory=list)  # Allowed source entity types
    target_types: List[str] = field(default_factory=list)  # Allowed target entity types
    properties: List[PropertyDefinition] = field(default_factory=list)
    symmetric: bool = False
    transitive: bool = False
    inverse: Optional[str] = None
    cardinality: str = "many_to_many"  # one_to_one, one_to_many, many_to_one, many_to_many

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "source_types": self.source_types,
            "target_types": self.target_types,
            "symmetric": self.symmetric,
            "transitive": self.transitive,
            "inverse": self.inverse,
            "cardinality": self.cardinality,
        }


@dataclass
class OntologySchema:
    """
    Complete ontology schema for the knowledge graph.

    Defines:
    - Entity types and their properties
    - Relationship types and constraints
    - Inheritance hierarchies
    - Validation rules
    """
    name: str = "default"
    version: str = "1.0"
    description: str = ""

    entity_types: Dict[str, EntityTypeDefinition] = field(default_factory=dict)
    relation_types: Dict[str, RelationTypeDefinition] = field(default_factory=dict)

    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Initialize with default types if empty."""
        if not self.entity_types:
            self._add_default_entity_types()
        if not self.relation_types:
            self._add_default_relation_types()

    def _add_default_entity_types(self) -> None:
        """Add default entity type definitions."""
        defaults = [
            EntityTypeDefinition(
                name="thing",
                description="Root type for all entities",
                properties=[
                    PropertyDefinition("name", "string", required=True, indexed=True),
                    PropertyDefinition("description", "string", searchable=True),
                    PropertyDefinition("created_at", "date"),
                ],
            ),
            EntityTypeDefinition(
                name="person",
                parent="thing",
                description="A human being",
                properties=[
                    PropertyDefinition("email", "string", indexed=True),
                    PropertyDefinition("role", "string"),
                    PropertyDefinition("department", "string"),
                ],
                allowed_relationships=["works_for", "manages", "knows", "created", "member_of"],
            ),
            EntityTypeDefinition(
                name="organization",
                parent="thing",
                description="A company, team, or group",
                properties=[
                    PropertyDefinition("industry", "string"),
                    PropertyDefinition("size", "string"),
                    PropertyDefinition("founded", "date"),
                ],
            ),
            EntityTypeDefinition(
                name="project",
                parent="thing",
                description="A work project or initiative",
                properties=[
                    PropertyDefinition("status", "string", constraints={"enum": ["active", "completed", "archived"]}),
                    PropertyDefinition("start_date", "date"),
                    PropertyDefinition("end_date", "date"),
                ],
            ),
            EntityTypeDefinition(
                name="concept",
                parent="thing",
                description="An abstract idea or notion",
                properties=[
                    PropertyDefinition("domain", "string"),
                ],
            ),
            EntityTypeDefinition(
                name="document",
                parent="thing",
                description="A document or file",
                properties=[
                    PropertyDefinition("file_type", "string"),
                    PropertyDefinition("url", "string"),
                ],
            ),
            EntityTypeDefinition(
                name="event",
                parent="thing",
                description="Something that happens",
                properties=[
                    PropertyDefinition("date", "date"),
                    PropertyDefinition("location", "string"),
                ],
            ),
            EntityTypeDefinition(
                name="location",
                parent="thing",
                description="A physical or virtual place",
                properties=[
                    PropertyDefinition("address", "string"),
                    PropertyDefinition("coordinates", "json"),
                ],
            ),
        ]

        for et in defaults:
            self.entity_types[et.name] = et

    def _add_default_relation_types(self) -> None:
        """Add default relationship type definitions."""
        defaults = [
            RelationTypeDefinition(
                name="works_for",
                description="Employment relationship",
                source_types=["person"],
                target_types=["organization"],
                inverse="employs",
            ),
            RelationTypeDefinition(
                name="manages",
                description="Management relationship",
                source_types=["person"],
                target_types=["person", "project", "team"],
                inverse="managed_by",
            ),
            RelationTypeDefinition(
                name="knows",
                description="Acquaintance relationship",
                source_types=["person"],
                target_types=["person"],
                symmetric=True,
            ),
            RelationTypeDefinition(
                name="member_of",
                description="Membership in a group",
                source_types=["person"],
                target_types=["organization", "team"],
                inverse="has_member",
            ),
            RelationTypeDefinition(
                name="created",
                description="Creation relationship",
                source_types=["person"],
                target_types=["project", "document", "code"],
                inverse="created_by",
            ),
            RelationTypeDefinition(
                name="part_of",
                description="Part-whole relationship",
                source_types=["thing"],
                target_types=["thing"],
                transitive=True,
                inverse="has_part",
            ),
            RelationTypeDefinition(
                name="is_a",
                description="Type hierarchy",
                source_types=["thing"],
                target_types=["thing"],
                transitive=True,
            ),
            RelationTypeDefinition(
                name="related_to",
                description="General relationship",
                source_types=["thing"],
                target_types=["thing"],
                symmetric=True,
            ),
            RelationTypeDefinition(
                name="depends_on",
                description="Dependency relationship",
                source_types=["project", "task"],
                target_types=["project", "task"],
                inverse="dependency_of",
            ),
            RelationTypeDefinition(
                name="located_in",
                description="Location relationship",
                source_types=["thing"],
                target_types=["location"],
                transitive=True,
            ),
        ]

        for rt in defaults:
            self.relation_types[rt.name] = rt

    def add_entity_type(self, definition: EntityTypeDefinition) -> None:
        """Add or update an entity type."""
        self.entity_types[definition.name] = definition
        self.updated_at = datetime.now()

    def add_relation_type(self, definition: RelationTypeDefinition) -> None:
        """Add or update a relation type."""
        self.relation_types[definition.name] = definition
        self.updated_at = datetime.now()

    def get_entity_type(self, name: str) -> Optional[EntityTypeDefinition]:
        """Get entity type definition by name."""
        return self.entity_types.get(name.lower())

    def get_relation_type(self, name: str) -> Optional[RelationTypeDefinition]:
        """Get relation type definition by name."""
        return self.relation_types.get(name.lower())

    def get_all_properties(self, entity_type: str) -> List[PropertyDefinition]:
        """Get all properties for an entity type including inherited ones."""
        properties = []
        visited = set()

        def collect(type_name: str):
            if type_name in visited:
                return
            visited.add(type_name)

            type_def = self.entity_types.get(type_name)
            if type_def:
                properties.extend(type_def.properties)
                if type_def.parent:
                    collect(type_def.parent)

        collect(entity_type.lower())
        return properties

    def get_type_hierarchy(self, entity_type: str) -> List[str]:
        """Get type hierarchy (ancestors) for an entity type."""
        hierarchy = []
        current = entity_type.lower()

        while current:
            type_def = self.entity_types.get(current)
            if type_def:
                hierarchy.append(current)
                current = type_def.parent
            else:
                break

        return hierarchy

    def is_subtype_of(self, entity_type: str, potential_parent: str) -> bool:
        """Check if entity_type is a subtype of potential_parent."""
        hierarchy = self.get_type_hierarchy(entity_type)
        return potential_parent.lower() in hierarchy

    def get_allowed_relationships(
        self,
        source_type: str,
        target_type: str,
    ) -> List[str]:
        """Get allowed relationship types between two entity types."""
        allowed = []

        for rel_name, rel_def in self.relation_types.items():
            # Check source type
            source_ok = not rel_def.source_types or any(
                self.is_subtype_of(source_type, st)
                for st in rel_def.source_types
            )

            # Check target type
            target_ok = not rel_def.target_types or any(
                self.is_subtype_of(target_type, tt)
                for tt in rel_def.target_types
            )

            if source_ok and target_ok:
                allowed.append(rel_name)

        return allowed

    def to_dict(self) -> dict:
        """Convert schema to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "entity_types": {k: v.to_dict() for k, v in self.entity_types.items()},
            "relation_types": {k: v.to_dict() for k, v in self.relation_types.items()},
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    def save(self, path: str) -> None:
        """Save schema to file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> OntologySchema:
        """Load schema from file."""
        with open(path) as f:
            data = json.load(f)

        schema = cls(
            name=data.get("name", "default"),
            version=data.get("version", "1.0"),
            description=data.get("description", ""),
        )

        # Clear defaults
        schema.entity_types.clear()
        schema.relation_types.clear()

        # Load entity types
        for name, et_data in data.get("entity_types", {}).items():
            properties = [
                PropertyDefinition(**p)
                for p in et_data.get("properties", [])
            ]
            et = EntityTypeDefinition(
                name=name,
                parent=et_data.get("parent"),
                description=et_data.get("description", ""),
                properties=properties,
                required_properties=et_data.get("required_properties", []),
                allowed_relationships=et_data.get("allowed_relationships", []),
            )
            schema.entity_types[name] = et

        # Load relation types
        for name, rt_data in data.get("relation_types", {}).items():
            properties = [
                PropertyDefinition(**p)
                for p in rt_data.get("properties", [])
            ]
            rt = RelationTypeDefinition(
                name=name,
                description=rt_data.get("description", ""),
                source_types=rt_data.get("source_types", []),
                target_types=rt_data.get("target_types", []),
                properties=properties,
                symmetric=rt_data.get("symmetric", False),
                transitive=rt_data.get("transitive", False),
                inverse=rt_data.get("inverse"),
                cardinality=rt_data.get("cardinality", "many_to_many"),
            )
            schema.relation_types[name] = rt

        return schema
