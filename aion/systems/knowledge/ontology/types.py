"""
AION Knowledge Graph Type Registry

Dynamic type registration and management.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

import structlog

from aion.systems.knowledge.types import EntityType, RelationType

logger = structlog.get_logger(__name__)


@dataclass
class CustomType:
    """A custom entity or relation type."""
    name: str
    base_type: str  # Built-in type this extends
    description: str = ""
    properties: Dict[str, str] = field(default_factory=dict)  # name -> data_type
    validators: List[Callable] = field(default_factory=list)
    created_by: str = "user"


class TypeRegistry:
    """
    Registry for entity and relationship types.

    Supports:
    - Built-in types from EntityType and RelationType enums
    - Custom user-defined types
    - Type inheritance
    - Dynamic type registration
    """

    def __init__(self):
        self._entity_types: Dict[str, CustomType] = {}
        self._relation_types: Dict[str, CustomType] = {}

        # Register built-in types
        self._register_builtins()

    def _register_builtins(self) -> None:
        """Register built-in types."""
        for et in EntityType:
            self._entity_types[et.value] = CustomType(
                name=et.value,
                base_type="thing",
                description=f"Built-in entity type: {et.value}",
                created_by="system",
            )

        for rt in RelationType:
            props = RelationType.get_properties().get(rt.value, {})
            self._relation_types[rt.value] = CustomType(
                name=rt.value,
                base_type="relationship",
                description=f"Built-in relation type: {rt.value}",
                properties={
                    "symmetric": str(props.get("symmetric", False)),
                    "transitive": str(props.get("transitive", False)),
                },
                created_by="system",
            )

    def register_entity_type(
        self,
        name: str,
        base_type: str = "thing",
        description: str = "",
        properties: Optional[Dict[str, str]] = None,
    ) -> bool:
        """Register a custom entity type."""
        if name in self._entity_types:
            logger.warning(f"Entity type {name} already exists")
            return False

        # Validate base type exists
        if base_type not in self._entity_types:
            logger.error(f"Base type {base_type} does not exist")
            return False

        self._entity_types[name] = CustomType(
            name=name,
            base_type=base_type,
            description=description,
            properties=properties or {},
        )

        logger.info(f"Registered entity type: {name}")
        return True

    def register_relation_type(
        self,
        name: str,
        description: str = "",
        symmetric: bool = False,
        transitive: bool = False,
        inverse: Optional[str] = None,
    ) -> bool:
        """Register a custom relation type."""
        if name in self._relation_types:
            logger.warning(f"Relation type {name} already exists")
            return False

        self._relation_types[name] = CustomType(
            name=name,
            base_type="relationship",
            description=description,
            properties={
                "symmetric": str(symmetric),
                "transitive": str(transitive),
                "inverse": inverse or "",
            },
        )

        logger.info(f"Registered relation type: {name}")
        return True

    def get_entity_type(self, name: str) -> Optional[CustomType]:
        """Get an entity type by name."""
        return self._entity_types.get(name.lower())

    def get_relation_type(self, name: str) -> Optional[CustomType]:
        """Get a relation type by name."""
        return self._relation_types.get(name.lower())

    def list_entity_types(self) -> List[str]:
        """List all entity types."""
        return list(self._entity_types.keys())

    def list_relation_types(self) -> List[str]:
        """List all relation types."""
        return list(self._relation_types.keys())

    def is_entity_type(self, name: str) -> bool:
        """Check if name is a valid entity type."""
        return name.lower() in self._entity_types

    def is_relation_type(self, name: str) -> bool:
        """Check if name is a valid relation type."""
        return name.lower() in self._relation_types

    def get_type_hierarchy(self, entity_type: str) -> List[str]:
        """Get the type hierarchy for an entity type."""
        hierarchy = []
        current = entity_type.lower()

        while current:
            if current not in self._entity_types:
                break
            hierarchy.append(current)
            ct = self._entity_types[current]
            current = ct.base_type if ct.base_type != current else None

        return hierarchy

    def is_subtype_of(self, child: str, parent: str) -> bool:
        """Check if child is a subtype of parent."""
        hierarchy = self.get_type_hierarchy(child)
        return parent.lower() in hierarchy

    def unregister_entity_type(self, name: str) -> bool:
        """Unregister a custom entity type."""
        ct = self._entity_types.get(name.lower())
        if not ct:
            return False

        if ct.created_by == "system":
            logger.warning(f"Cannot unregister built-in type: {name}")
            return False

        del self._entity_types[name.lower()]
        return True

    def unregister_relation_type(self, name: str) -> bool:
        """Unregister a custom relation type."""
        ct = self._relation_types.get(name.lower())
        if not ct:
            return False

        if ct.created_by == "system":
            logger.warning(f"Cannot unregister built-in type: {name}")
            return False

        del self._relation_types[name.lower()]
        return True

    def get_all_properties(self, entity_type: str) -> Dict[str, str]:
        """Get all properties for a type including inherited."""
        properties = {}

        for type_name in reversed(self.get_type_hierarchy(entity_type)):
            ct = self._entity_types.get(type_name)
            if ct:
                properties.update(ct.properties)

        return properties
