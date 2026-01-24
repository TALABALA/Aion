"""
AION Knowledge Graph System

A state-of-the-art knowledge graph for entity-relationship reasoning,
combining structured graph queries with vector semantic search.

Features:
- Entity and relationship CRUD with temporal scoping
- Multi-hop graph traversal and path finding
- Hybrid vector + graph search with learned reranking
- LLM-powered entity extraction and linking
- Probabilistic inference with uncertainty quantification
- Graph embeddings (TransE-style) for relation prediction
- Natural language to graph query translation
- Ontology schema validation and evolution

Usage:
    from aion.systems.knowledge import KnowledgeManager, Entity, EntityType

    kg = KnowledgeManager()
    await kg.initialize()

    # Add entities
    alice = await kg.add_entity("Alice", EntityType.PERSON)
    acme = await kg.add_entity("Acme Corp", EntityType.ORGANIZATION)

    # Add relationships
    await kg.add_relationship(alice.id, acme.id, RelationType.WORKS_FOR)

    # Query
    result = await kg.query(natural_language="Who works at Acme?")
    path = await kg.find_path("Alice", "Bob")
"""

from aion.systems.knowledge.types import (
    # Enums
    EntityType,
    RelationType,

    # Core dataclasses
    Entity,
    EntityProperty,
    Relationship,
    Triple,
    Path,
    Subgraph,

    # Query types
    GraphQuery,
    QueryResult,

    # Inference
    InferenceRule,
    InferenceResult,

    # Embeddings
    GraphEmbedding,
)

from aion.systems.knowledge.manager import KnowledgeManager

__all__ = [
    # Manager
    "KnowledgeManager",

    # Enums
    "EntityType",
    "RelationType",

    # Core types
    "Entity",
    "EntityProperty",
    "Relationship",
    "Triple",
    "Path",
    "Subgraph",

    # Query
    "GraphQuery",
    "QueryResult",

    # Inference
    "InferenceRule",
    "InferenceResult",

    # Embeddings
    "GraphEmbedding",
]
