"""
AION Knowledge Graph Ontology

Schema definition, validation, and evolution.
"""

from aion.systems.knowledge.ontology.schema import OntologySchema
from aion.systems.knowledge.ontology.types import TypeRegistry
from aion.systems.knowledge.ontology.validation import SchemaValidator

__all__ = [
    "OntologySchema",
    "TypeRegistry",
    "SchemaValidator",
]
