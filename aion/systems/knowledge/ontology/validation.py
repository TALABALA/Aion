"""
AION Knowledge Graph Schema Validation

Validate entities and relationships against schema.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import structlog

from aion.systems.knowledge.types import Entity, Relationship
from aion.systems.knowledge.ontology.schema import OntologySchema

logger = structlog.get_logger(__name__)


@dataclass
class ValidationError:
    """A validation error."""
    field: str
    message: str
    severity: str = "error"  # error, warning, info
    code: str = ""

    def to_dict(self) -> dict:
        return {
            "field": self.field,
            "message": self.message,
            "severity": self.severity,
            "code": self.code,
        }


@dataclass
class ValidationResult:
    """Result of validation."""
    valid: bool = True
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationError] = field(default_factory=list)

    def add_error(self, field: str, message: str, code: str = "") -> None:
        self.errors.append(ValidationError(field, message, "error", code))
        self.valid = False

    def add_warning(self, field: str, message: str, code: str = "") -> None:
        self.warnings.append(ValidationError(field, message, "warning", code))

    def to_dict(self) -> dict:
        return {
            "valid": self.valid,
            "errors": [e.to_dict() for e in self.errors],
            "warnings": [w.to_dict() for w in self.warnings],
        }


class SchemaValidator:
    """
    Validate entities and relationships against ontology schema.

    Validates:
    - Entity types exist in schema
    - Required properties are present
    - Property types are correct
    - Property constraints are satisfied
    - Relationship types are valid
    - Relationship endpoints match allowed types
    """

    def __init__(self, schema: OntologySchema):
        self.schema = schema

    def validate_entity(self, entity: Entity) -> ValidationResult:
        """Validate an entity against the schema."""
        result = ValidationResult()

        # Check entity type exists
        type_def = self.schema.get_entity_type(entity.entity_type.value)
        if not type_def:
            if entity.entity_type.value != "custom":
                result.add_warning(
                    "entity_type",
                    f"Entity type '{entity.entity_type.value}' not in schema",
                    "unknown_type",
                )
            type_def = self.schema.get_entity_type("thing")

        if not type_def:
            return result

        # Get all properties including inherited
        all_properties = self.schema.get_all_properties(entity.entity_type.value)
        property_map = {p.name: p for p in all_properties}

        # Check required properties
        for prop in all_properties:
            if prop.required and prop.name not in entity.properties:
                if prop.name == "name" and entity.name:
                    continue  # Name is a special case
                result.add_error(
                    prop.name,
                    f"Required property '{prop.name}' is missing",
                    "missing_required",
                )

        # Validate property values
        for key, value in entity.properties.items():
            prop_def = property_map.get(key)
            if prop_def:
                self._validate_property_value(result, key, value, prop_def)
            else:
                result.add_warning(
                    key,
                    f"Property '{key}' is not defined in schema",
                    "unknown_property",
                )

        # Validate name
        if not entity.name or not entity.name.strip():
            result.add_error("name", "Entity name cannot be empty", "empty_name")

        return result

    def validate_relationship(
        self,
        relationship: Relationship,
        source_entity: Optional[Entity] = None,
        target_entity: Optional[Entity] = None,
    ) -> ValidationResult:
        """Validate a relationship against the schema."""
        result = ValidationResult()

        # Check relation type exists
        rel_def = self.schema.get_relation_type(relationship.relation_type.value)
        if not rel_def:
            if relationship.relation_type.value != "custom":
                result.add_warning(
                    "relation_type",
                    f"Relation type '{relationship.relation_type.value}' not in schema",
                    "unknown_type",
                )
            return result

        # Check source type
        if source_entity and rel_def.source_types:
            source_type = source_entity.entity_type.value
            if not any(
                self.schema.is_subtype_of(source_type, allowed)
                for allowed in rel_def.source_types
            ):
                result.add_error(
                    "source_id",
                    f"Source entity type '{source_type}' not allowed for relation '{rel_def.name}'",
                    "invalid_source_type",
                )

        # Check target type
        if target_entity and rel_def.target_types:
            target_type = target_entity.entity_type.value
            if not any(
                self.schema.is_subtype_of(target_type, allowed)
                for allowed in rel_def.target_types
            ):
                result.add_error(
                    "target_id",
                    f"Target entity type '{target_type}' not allowed for relation '{rel_def.name}'",
                    "invalid_target_type",
                )

        # Check IDs are present
        if not relationship.source_id:
            result.add_error("source_id", "Source ID is required", "missing_source")
        if not relationship.target_id:
            result.add_error("target_id", "Target ID is required", "missing_target")

        # Check not self-loop (unless allowed)
        if relationship.source_id == relationship.target_id:
            # Some relations allow self-loops (e.g., similar_to)
            if not rel_def.symmetric:
                result.add_warning(
                    "target_id",
                    "Self-loop relationship detected",
                    "self_loop",
                )

        # Validate temporal bounds
        if relationship.valid_from and relationship.valid_until:
            if relationship.valid_from > relationship.valid_until:
                result.add_error(
                    "valid_until",
                    "valid_until must be after valid_from",
                    "invalid_temporal_range",
                )

        # Validate confidence
        if relationship.confidence < 0 or relationship.confidence > 1:
            result.add_error(
                "confidence",
                "Confidence must be between 0 and 1",
                "invalid_confidence",
            )

        # Validate weight
        if relationship.weight < 0:
            result.add_error(
                "weight",
                "Weight must be non-negative",
                "invalid_weight",
            )

        return result

    def _validate_property_value(
        self,
        result: ValidationResult,
        key: str,
        value: Any,
        prop_def: Any,
    ) -> None:
        """Validate a property value against its definition."""
        # Type validation
        expected_type = prop_def.data_type

        if expected_type == "string":
            if not isinstance(value, str):
                result.add_error(
                    key,
                    f"Property '{key}' must be a string",
                    "invalid_type",
                )
                return

        elif expected_type == "number":
            if not isinstance(value, (int, float)):
                result.add_error(
                    key,
                    f"Property '{key}' must be a number",
                    "invalid_type",
                )
                return

        elif expected_type == "boolean":
            if not isinstance(value, bool):
                result.add_error(
                    key,
                    f"Property '{key}' must be a boolean",
                    "invalid_type",
                )
                return

        elif expected_type == "date":
            if not isinstance(value, (str, datetime)):
                result.add_error(
                    key,
                    f"Property '{key}' must be a date",
                    "invalid_type",
                )
                return

            # Try to parse if string
            if isinstance(value, str):
                try:
                    datetime.fromisoformat(value.replace("Z", "+00:00"))
                except ValueError:
                    result.add_error(
                        key,
                        f"Property '{key}' has invalid date format",
                        "invalid_date_format",
                    )
                    return

        # Constraint validation
        constraints = prop_def.constraints

        if "min" in constraints and isinstance(value, (int, float)):
            if value < constraints["min"]:
                result.add_error(
                    key,
                    f"Property '{key}' must be >= {constraints['min']}",
                    "constraint_min",
                )

        if "max" in constraints and isinstance(value, (int, float)):
            if value > constraints["max"]:
                result.add_error(
                    key,
                    f"Property '{key}' must be <= {constraints['max']}",
                    "constraint_max",
                )

        if "enum" in constraints:
            if value not in constraints["enum"]:
                result.add_error(
                    key,
                    f"Property '{key}' must be one of: {constraints['enum']}",
                    "constraint_enum",
                )

        if "pattern" in constraints and isinstance(value, str):
            import re
            if not re.match(constraints["pattern"], value):
                result.add_error(
                    key,
                    f"Property '{key}' does not match pattern",
                    "constraint_pattern",
                )

        if "min_length" in constraints and isinstance(value, str):
            if len(value) < constraints["min_length"]:
                result.add_error(
                    key,
                    f"Property '{key}' must be at least {constraints['min_length']} characters",
                    "constraint_min_length",
                )

        if "max_length" in constraints and isinstance(value, str):
            if len(value) > constraints["max_length"]:
                result.add_error(
                    key,
                    f"Property '{key}' must be at most {constraints['max_length']} characters",
                    "constraint_max_length",
                )

    def validate_batch(
        self,
        entities: List[Entity],
        relationships: List[Relationship],
    ) -> Dict[str, ValidationResult]:
        """Validate a batch of entities and relationships."""
        results = {}

        # Build entity map
        entity_map = {e.id: e for e in entities}

        # Validate entities
        for entity in entities:
            results[f"entity:{entity.id}"] = self.validate_entity(entity)

        # Validate relationships
        for rel in relationships:
            source = entity_map.get(rel.source_id)
            target = entity_map.get(rel.target_id)
            results[f"relationship:{rel.id}"] = self.validate_relationship(
                rel, source, target
            )

        return results
