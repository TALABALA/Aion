"""
AION Knowledge Manager

Central coordinator for the knowledge graph system.
Provides unified API for all knowledge graph operations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import structlog

from aion.systems.knowledge.types import (
    Entity,
    EntityType,
    Relationship,
    RelationType,
    GraphQuery,
    QueryResult,
    Path as GraphPath,
    Triple,
    Subgraph,
    InferenceRule,
    GraphStatistics,
)
from aion.systems.knowledge.store.sqlite_store import SQLiteGraphStore
from aion.systems.knowledge.store.memory_store import InMemoryGraphStore
from aion.systems.knowledge.query.engine import QueryEngine
from aion.systems.knowledge.inference.engine import InferenceEngine
from aion.systems.knowledge.inference.rules import get_default_rules
from aion.systems.knowledge.hybrid.search import HybridSearch, HybridResult, SearchConfig
from aion.systems.knowledge.extraction.extractor import EntityExtractor
from aion.systems.knowledge.extraction.linker import EntityLinker
from aion.systems.knowledge.ontology.schema import OntologySchema
from aion.systems.knowledge.ontology.validation import SchemaValidator

logger = structlog.get_logger(__name__)


@dataclass
class KnowledgeConfig:
    """Configuration for knowledge manager."""
    db_path: str = "./data/knowledge_graph.db"
    use_memory_store: bool = False
    enable_fts: bool = True
    enable_inference: bool = True
    enable_extraction: bool = True
    enable_validation: bool = True
    auto_embed: bool = True
    embedding_model: str = "all-MiniLM-L6-v2"
    schema_path: Optional[str] = None
    search_config: Optional[SearchConfig] = None


class KnowledgeManager:
    """
    Central manager for the knowledge graph.

    Provides unified API for:
    - Entity and relationship CRUD
    - Graph queries (structured and natural language)
    - Hybrid search (vector + graph)
    - Knowledge extraction from text
    - Inference and reasoning
    - Schema validation

    Usage:
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

        # Search
        results = await kg.search("machine learning expert")

        # Extract from text
        await kg.extract_and_add("Alice works at Acme Corp as a data scientist.")
    """

    def __init__(
        self,
        config: Optional[KnowledgeConfig] = None,
        memory_system: Optional[Any] = None,
    ):
        self.config = config or KnowledgeConfig()

        # Initialize store
        if self.config.use_memory_store:
            self.store = InMemoryGraphStore()
        else:
            self.store = SQLiteGraphStore(
                db_path=self.config.db_path,
                enable_fts=self.config.enable_fts,
            )

        # Initialize components
        self.query_engine = QueryEngine(self.store)
        self.inference_engine = InferenceEngine(self.store)
        self.hybrid_search = HybridSearch(
            self.store,
            memory_system,
            self.config.search_config,
        )

        # Optional components
        self.extractor = EntityExtractor() if self.config.enable_extraction else None
        self.linker = EntityLinker()

        # Schema
        self.schema = OntologySchema()
        self.validator = SchemaValidator(self.schema) if self.config.enable_validation else None

        # Embedding function
        self._embed_fn = None

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the knowledge manager."""
        if self._initialized:
            return

        logger.info("Initializing Knowledge Manager")

        # Initialize store
        await self.store.initialize()

        # Initialize extractor
        if self.extractor:
            await self.extractor.initialize()

        # Load schema
        if self.config.schema_path and Path(self.config.schema_path).exists():
            self.schema = OntologySchema.load(self.config.schema_path)
            self.validator = SchemaValidator(self.schema)

        # Add default inference rules
        if self.config.enable_inference:
            for rule in get_default_rules():
                self.inference_engine.add_rule(rule)

        # Initialize embedding function
        if self.config.auto_embed:
            await self._init_embedding()

        self._initialized = True
        logger.info("Knowledge Manager initialized")

    async def shutdown(self) -> None:
        """Shutdown the knowledge manager."""
        await self.store.shutdown()
        self._initialized = False
        logger.info("Knowledge Manager shutdown")

    async def _init_embedding(self) -> None:
        """Initialize embedding function."""
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(self.config.embedding_model)
            self._embed_fn = lambda text: model.encode(text).tolist()
            logger.info(f"Embedding model loaded: {self.config.embedding_model}")
        except ImportError:
            logger.warning("sentence-transformers not available, auto-embedding disabled")

    async def _embed_text(self, text: str) -> Optional[List[float]]:
        """Embed text if embedding function available."""
        if self._embed_fn:
            try:
                return self._embed_fn(text)
            except Exception as e:
                logger.warning(f"Embedding failed: {e}")
        return None

    # ==========================================================================
    # Entity Operations
    # ==========================================================================

    async def add_entity(
        self,
        name: str,
        entity_type: Union[EntityType, str],
        description: str = "",
        properties: Optional[Dict[str, Any]] = None,
        aliases: Optional[List[str]] = None,
        auto_embed: bool = True,
    ) -> Entity:
        """
        Add a new entity to the knowledge graph.

        Args:
            name: Entity name
            entity_type: Type of entity
            description: Optional description
            properties: Optional properties dict
            aliases: Optional list of aliases
            auto_embed: Whether to auto-generate embedding

        Returns:
            Created entity
        """
        # Convert string to EntityType
        if isinstance(entity_type, str):
            try:
                entity_type = EntityType(entity_type.lower())
            except ValueError:
                entity_type = EntityType.CUSTOM

        entity = Entity(
            name=name,
            entity_type=entity_type,
            description=description,
            properties=properties or {},
            aliases=aliases or [],
        )

        # Validate if enabled
        if self.validator:
            result = self.validator.validate_entity(entity)
            if not result.valid:
                logger.warning(f"Entity validation errors: {result.errors}")

        # Generate embedding
        if auto_embed and self.config.auto_embed:
            embed_text = f"{name} {description}"
            embedding = await self._embed_text(embed_text)
            if embedding:
                from aion.systems.knowledge.types import GraphEmbedding
                entity.text_embedding = GraphEmbedding(
                    embedding_type="text",
                    vector=embedding,
                    model_id=self.config.embedding_model,
                )

        await self.store.create_entity(entity)

        logger.info(f"Added entity: {name} ({entity_type.value})")

        return entity

    async def get_entity(
        self,
        entity_id: Optional[str] = None,
        name: Optional[str] = None,
    ) -> Optional[Entity]:
        """Get an entity by ID or name."""
        if entity_id:
            return await self.store.get_entity(entity_id)
        elif name:
            return await self.store.get_entity_by_name(name)
        return None

    async def update_entity(self, entity: Entity) -> bool:
        """Update an entity."""
        if self.validator:
            result = self.validator.validate_entity(entity)
            if not result.valid:
                logger.warning(f"Entity validation errors: {result.errors}")

        return await self.store.update_entity(entity)

    async def delete_entity(self, entity_id: str, hard: bool = False) -> bool:
        """Delete an entity."""
        return await self.store.delete_entity(entity_id, soft=not hard)

    async def list_entities(
        self,
        entity_type: Optional[EntityType] = None,
        query: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Entity]:
        """List entities with optional filters."""
        return await self.store.search_entities(
            query=query,
            entity_type=entity_type,
            limit=limit,
            offset=offset,
        )

    # ==========================================================================
    # Relationship Operations
    # ==========================================================================

    async def add_relationship(
        self,
        source: str,
        target: str,
        relation_type: Union[RelationType, str],
        properties: Optional[Dict[str, Any]] = None,
        bidirectional: bool = False,
        confidence: float = 1.0,
        valid_from: Optional[datetime] = None,
        valid_until: Optional[datetime] = None,
    ) -> Relationship:
        """
        Add a relationship between entities.

        Args:
            source: Source entity ID or name
            target: Target entity ID or name
            relation_type: Type of relationship
            properties: Optional properties
            bidirectional: Whether relationship is bidirectional
            confidence: Confidence score (0-1)
            valid_from: Start of validity period
            valid_until: End of validity period

        Returns:
            Created relationship
        """
        # Resolve entity names to IDs
        source_entity = await self.get_entity(entity_id=source) or await self.get_entity(name=source)
        target_entity = await self.get_entity(entity_id=target) or await self.get_entity(name=target)

        if not source_entity:
            raise ValueError(f"Source entity not found: {source}")
        if not target_entity:
            raise ValueError(f"Target entity not found: {target}")

        # Convert string to RelationType
        if isinstance(relation_type, str):
            try:
                relation_type = RelationType(relation_type.lower())
            except ValueError:
                relation_type = RelationType.CUSTOM

        rel = Relationship(
            source_id=source_entity.id,
            target_id=target_entity.id,
            relation_type=relation_type,
            properties=properties or {},
            bidirectional=bidirectional or relation_type.is_symmetric(),
            confidence=confidence,
            valid_from=valid_from,
            valid_until=valid_until,
        )

        # Validate if enabled
        if self.validator:
            result = self.validator.validate_relationship(
                rel, source_entity, target_entity
            )
            if not result.valid:
                logger.warning(f"Relationship validation errors: {result.errors}")

        await self.store.create_relationship(rel)

        logger.info(
            f"Added relationship: {source_entity.name} -[{relation_type.value}]-> {target_entity.name}"
        )

        return rel

    async def get_relationships(
        self,
        entity: str,
        direction: str = "both",
        relation_type: Optional[RelationType] = None,
    ) -> List[Relationship]:
        """Get relationships for an entity."""
        entity_obj = await self.get_entity(entity_id=entity) or await self.get_entity(name=entity)

        if not entity_obj:
            return []

        return await self.store.get_relationships(
            entity_obj.id,
            direction=direction,
            relation_type=relation_type,
        )

    async def delete_relationship(self, relationship_id: str) -> bool:
        """Delete a relationship."""
        return await self.store.delete_relationship(relationship_id)

    # ==========================================================================
    # Query Operations
    # ==========================================================================

    async def query(
        self,
        natural_language: Optional[str] = None,
        entity_types: Optional[List[EntityType]] = None,
        relation_types: Optional[List[RelationType]] = None,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        **kwargs,
    ) -> QueryResult:
        """
        Execute a query against the knowledge graph.

        Args:
            natural_language: Natural language query
            entity_types: Filter by entity types
            relation_types: Filter by relationship types
            filters: Property filters
            limit: Maximum results

        Returns:
            QueryResult with matching entities and relationships
        """
        graph_query = GraphQuery(
            natural_language=natural_language,
            entity_types=entity_types or [],
            relation_types=relation_types or [],
            entity_filters=filters or {},
            limit=limit,
            **kwargs,
        )

        return await self.query_engine.execute(graph_query)

    async def find_path(
        self,
        start: str,
        end: str,
        max_depth: int = 5,
        relation_types: Optional[List[RelationType]] = None,
    ) -> Optional[GraphPath]:
        """
        Find shortest path between two entities.

        Args:
            start: Start entity ID or name
            end: End entity ID or name
            max_depth: Maximum path length
            relation_types: Optional relationship type filter

        Returns:
            Path if found, None otherwise
        """
        # Resolve names
        start_entity = await self.get_entity(entity_id=start) or await self.get_entity(name=start)
        end_entity = await self.get_entity(entity_id=end) or await self.get_entity(name=end)

        if not start_entity or not end_entity:
            return None

        return await self.store.find_path(
            start_entity.id,
            end_entity.id,
            max_depth=max_depth,
            relation_types=relation_types,
        )

    async def get_subgraph(
        self,
        entity: str,
        depth: int = 1,
        relation_types: Optional[List[RelationType]] = None,
    ) -> Subgraph:
        """Get subgraph around an entity."""
        entity_obj = await self.get_entity(entity_id=entity) or await self.get_entity(name=entity)

        if not entity_obj:
            return Subgraph()

        return await self.store.get_neighbors(
            entity_obj.id,
            relation_types=relation_types,
            depth=depth,
        )

    # ==========================================================================
    # Search Operations
    # ==========================================================================

    async def search(
        self,
        query: str,
        entity_types: Optional[List[EntityType]] = None,
        seed_entity: Optional[str] = None,
        limit: int = 20,
    ) -> List[HybridResult]:
        """
        Perform hybrid search (vector + graph).

        Args:
            query: Search query
            entity_types: Optional type filter
            seed_entity: Optional entity to start graph traversal from
            limit: Maximum results

        Returns:
            List of hybrid results with scores
        """
        seed_id = None
        if seed_entity:
            entity = await self.get_entity(entity_id=seed_entity) or await self.get_entity(name=seed_entity)
            if entity:
                seed_id = entity.id

        return await self.hybrid_search.search(
            query=query,
            entity_types=entity_types,
            seed_entity_id=seed_id,
            limit=limit,
        )

    async def search_similar(
        self,
        entity: str,
        limit: int = 10,
    ) -> List[HybridResult]:
        """Find entities similar to a given entity."""
        entity_obj = await self.get_entity(entity_id=entity) or await self.get_entity(name=entity)

        if not entity_obj:
            return []

        return await self.hybrid_search.find_similar_entities(entity_obj.id, limit)

    async def expand(self, entity: str, depth: int = 1) -> Dict[str, Any]:
        """Expand an entity to show its neighborhood."""
        entity_obj = await self.get_entity(entity_id=entity) or await self.get_entity(name=entity)

        if not entity_obj:
            return {"error": "Entity not found"}

        return await self.hybrid_search.expand_entity(entity_obj.id, depth)

    # ==========================================================================
    # Extraction Operations
    # ==========================================================================

    async def extract_and_add(
        self,
        text: str,
        context: Optional[str] = None,
        merge_existing: bool = True,
    ) -> Dict[str, Any]:
        """
        Extract entities and relationships from text and add to graph.

        Args:
            text: Text to extract from
            context: Optional context
            merge_existing: Whether to merge with existing entities

        Returns:
            Summary of extracted knowledge
        """
        if not self.extractor:
            return {"error": "Extraction not enabled"}

        # Get existing entities for linking
        existing = await self.store.search_entities(limit=1000) if merge_existing else []

        # Extract
        result = await self.extractor.extract(text, context, existing)

        added_entities = []
        added_relationships = []

        # Add entities
        for entity in result.entities:
            if merge_existing:
                match = self.linker.find_best_match(entity, existing)
                if match:
                    match.merge_from(entity)
                    await self.store.update_entity(match)
                    entity = match
                else:
                    # Generate embedding
                    if self.config.auto_embed:
                        embed_text = f"{entity.name} {entity.description}"
                        embedding = await self._embed_text(embed_text)
                        if embedding:
                            from aion.systems.knowledge.types import GraphEmbedding
                            entity.text_embedding = GraphEmbedding(
                                embedding_type="text",
                                vector=embedding,
                            )
                    await self.store.create_entity(entity)
            else:
                await self.store.create_entity(entity)

            added_entities.append(entity)

        # Add relationships
        for rel in result.relationships:
            await self.store.create_relationship(rel)
            added_relationships.append(rel)

        return {
            "entities_added": len(added_entities),
            "relationships_added": len(added_relationships),
            "entities": [e.to_dict() for e in added_entities],
            "relationships": [r.to_dict() for r in added_relationships],
            "triples": [t.to_string() for t in result.triples],
        }

    # ==========================================================================
    # Inference Operations
    # ==========================================================================

    async def run_inference(
        self,
        entity_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run inference to discover new relationships."""
        if not self.config.enable_inference:
            return {"error": "Inference not enabled"}

        summary = await self.inference_engine.run_inference(entity_ids)
        return summary.to_dict()

    async def compute_centrality(
        self,
        algorithm: str = "pagerank",
    ) -> Dict[str, float]:
        """Compute centrality scores for entities."""
        return await self.inference_engine.compute_centrality(algorithm)

    async def update_scores(self) -> int:
        """Update importance scores for all entities."""
        return await self.inference_engine.update_entity_scores()

    def add_inference_rule(self, rule: InferenceRule) -> None:
        """Add a custom inference rule."""
        self.inference_engine.add_rule(rule)

    # ==========================================================================
    # Statistics
    # ==========================================================================

    async def get_stats(self) -> GraphStatistics:
        """Get knowledge graph statistics."""
        return await self.store.get_stats()

    async def health_check(self) -> Dict[str, Any]:
        """Check system health."""
        store_health = await self.store.health_check()
        stats = await self.get_stats()

        return {
            "status": "ok" if store_health.get("status") == "ok" else "degraded",
            "store": store_health,
            "entities": stats.total_entities,
            "relationships": stats.total_relationships,
            "initialized": self._initialized,
        }

    # ==========================================================================
    # Import/Export
    # ==========================================================================

    async def export_graph(
        self,
        entity_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Export graph (or subgraph) as JSON."""
        return await self.store.export_subgraph(entity_ids)

    async def import_graph(
        self,
        data: Dict[str, Any],
        merge: bool = True,
    ) -> Dict[str, int]:
        """Import graph from JSON."""
        return await self.store.import_subgraph(data, merge)
