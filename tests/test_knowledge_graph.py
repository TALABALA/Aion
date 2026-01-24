"""
AION Knowledge Graph Tests

Comprehensive test suite for the Knowledge Graph System.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
import tempfile


class TestKnowledgeTypes:
    """Test knowledge graph types and enums."""

    def test_entity_type_enum(self):
        """Test EntityType enum."""
        from aion.systems.knowledge.types import EntityType

        assert EntityType.PERSON.value == "person"
        assert EntityType.ORGANIZATION.value == "organization"
        assert EntityType.CONCEPT.value == "concept"

    def test_entity_type_hierarchy(self):
        """Test entity type hierarchy."""
        from aion.systems.knowledge.types import EntityType

        # Test subtypes
        assert EntityType.PERSON.is_subtype_of(EntityType.THING)
        assert EntityType.TEAM.is_subtype_of(EntityType.ORGANIZATION)
        assert not EntityType.PERSON.is_subtype_of(EntityType.ORGANIZATION)

    def test_entity_type_categories(self):
        """Test entity type categories."""
        from aion.systems.knowledge.types import EntityType

        assert EntityType.PERSON.get_category() == "agent"
        assert EntityType.CONCEPT.get_category() == "abstract"
        assert EntityType.LOCATION.get_category() == "physical"

    def test_relation_type_properties(self):
        """Test relation type properties."""
        from aion.systems.knowledge.types import RelationType

        props = RelationType.get_properties()

        # Test symmetric relations
        assert props["knows"]["symmetric"] is True
        assert props["works_for"]["symmetric"] is False

        # Test transitive relations
        assert props["is_a"]["transitive"] is True
        assert props["knows"]["transitive"] is False

    def test_entity_creation(self):
        """Test entity creation with all fields."""
        from aion.systems.knowledge.types import Entity, EntityType

        entity = Entity(
            name="Test Entity",
            entity_type=EntityType.PERSON,
            description="A test person",
            properties={"role": "developer"},
            aliases=["Test", "TE"],
            confidence=0.9,
            importance=0.8,
        )

        assert entity.name == "Test Entity"
        assert entity.entity_type == EntityType.PERSON
        assert entity.properties["role"] == "developer"
        assert "Test" in entity.aliases
        assert entity.confidence == 0.9
        assert entity.id is not None

    def test_entity_to_dict(self):
        """Test entity serialization."""
        from aion.systems.knowledge.types import Entity, EntityType

        entity = Entity(
            name="Test",
            entity_type=EntityType.CONCEPT,
        )

        data = entity.to_dict()
        assert data["name"] == "Test"
        assert data["entity_type"] == "concept"
        assert "id" in data

    def test_relationship_creation(self):
        """Test relationship creation."""
        from aion.systems.knowledge.types import Relationship, RelationType

        rel = Relationship(
            source_id="entity1",
            target_id="entity2",
            relation_type=RelationType.WORKS_FOR,
            confidence=0.95,
            weight=1.0,
        )

        assert rel.source_id == "entity1"
        assert rel.target_id == "entity2"
        assert rel.relation_type == RelationType.WORKS_FOR
        assert rel.id is not None

    def test_relationship_temporal(self):
        """Test temporal relationship."""
        from aion.systems.knowledge.types import Relationship, RelationType

        now = datetime.now()
        future = now + timedelta(days=365)

        rel = Relationship(
            source_id="e1",
            target_id="e2",
            relation_type=RelationType.MEMBER_OF,
            valid_from=now,
            valid_until=future,
        )

        assert rel.is_currently_valid()
        assert rel.valid_from == now
        assert rel.valid_until == future

    def test_triple_creation(self):
        """Test triple creation."""
        from aion.systems.knowledge.types import Entity, EntityType, RelationType, Triple

        subject = Entity(name="Alice", entity_type=EntityType.PERSON)
        obj = Entity(name="Acme Corp", entity_type=EntityType.ORGANIZATION)

        triple = Triple(
            subject=subject,
            predicate=RelationType.WORKS_FOR,
            object=obj,
            confidence=0.9,
        )

        assert triple.subject.name == "Alice"
        assert triple.predicate == RelationType.WORKS_FOR
        assert triple.object.name == "Acme Corp"
        assert "Alice" in triple.to_string()

    def test_path_creation(self):
        """Test path creation."""
        from aion.systems.knowledge.types import Entity, EntityType, Relationship, RelationType, Path

        e1 = Entity(name="A", entity_type=EntityType.PERSON)
        e2 = Entity(name="B", entity_type=EntityType.PERSON)
        e3 = Entity(name="C", entity_type=EntityType.PERSON)

        r1 = Relationship(source_id=e1.id, target_id=e2.id, relation_type=RelationType.KNOWS)
        r2 = Relationship(source_id=e2.id, target_id=e3.id, relation_type=RelationType.KNOWS)

        path = Path(
            entities=[e1, e2, e3],
            relationships=[r1, r2],
        )

        assert path.length == 2
        assert path.start_entity.name == "A"
        assert path.end_entity.name == "C"


class TestMemoryStore:
    """Test in-memory graph store."""

    @pytest.fixture
    def memory_store(self):
        """Create a memory store for testing."""
        from aion.systems.knowledge.store.memory_store import InMemoryGraphStore

        return InMemoryGraphStore()

    @pytest.mark.asyncio
    async def test_entity_crud(self, memory_store):
        """Test entity CRUD operations."""
        from aion.systems.knowledge.types import Entity, EntityType

        await memory_store.initialize()

        # Create
        entity = Entity(
            name="Test Entity",
            entity_type=EntityType.PERSON,
            description="A test entity",
        )
        created = await memory_store.create_entity(entity)
        assert created.id == entity.id

        # Read
        retrieved = await memory_store.get_entity(entity.id)
        assert retrieved is not None
        assert retrieved.name == "Test Entity"

        # Update
        entity.description = "Updated description"
        updated = await memory_store.update_entity(entity)
        assert updated.description == "Updated description"

        # Delete
        success = await memory_store.delete_entity(entity.id)
        assert success

        # Verify deleted
        deleted = await memory_store.get_entity(entity.id)
        assert deleted is None

    @pytest.mark.asyncio
    async def test_relationship_crud(self, memory_store):
        """Test relationship CRUD operations."""
        from aion.systems.knowledge.types import Entity, EntityType, Relationship, RelationType

        await memory_store.initialize()

        # Create entities first
        e1 = Entity(name="Alice", entity_type=EntityType.PERSON)
        e2 = Entity(name="Bob", entity_type=EntityType.PERSON)
        await memory_store.create_entity(e1)
        await memory_store.create_entity(e2)

        # Create relationship
        rel = Relationship(
            source_id=e1.id,
            target_id=e2.id,
            relation_type=RelationType.KNOWS,
        )
        created = await memory_store.create_relationship(rel)
        assert created.id == rel.id

        # Get relationships
        outgoing = await memory_store.get_relationships(e1.id, direction="outgoing")
        assert len(outgoing) == 1
        assert outgoing[0].target_id == e2.id

        incoming = await memory_store.get_relationships(e2.id, direction="incoming")
        assert len(incoming) == 1
        assert incoming[0].source_id == e1.id

        # Delete relationship
        success = await memory_store.delete_relationship(rel.id)
        assert success

    @pytest.mark.asyncio
    async def test_search_entities(self, memory_store):
        """Test entity search."""
        from aion.systems.knowledge.types import Entity, EntityType

        await memory_store.initialize()

        # Create test entities
        entities = [
            Entity(name="Python Programming", entity_type=EntityType.CONCEPT),
            Entity(name="Python Snake", entity_type=EntityType.CONCEPT),
            Entity(name="Java Programming", entity_type=EntityType.CONCEPT),
        ]

        for e in entities:
            await memory_store.create_entity(e)

        # Search
        results = await memory_store.search_entities("Python", limit=10)
        assert len(results) >= 2
        assert any("Python" in e.name for e in results)

    @pytest.mark.asyncio
    async def test_get_neighbors(self, memory_store):
        """Test getting neighbors."""
        from aion.systems.knowledge.types import Entity, EntityType, Relationship, RelationType

        await memory_store.initialize()

        # Create a small graph: A -> B -> C
        a = Entity(name="A", entity_type=EntityType.CONCEPT)
        b = Entity(name="B", entity_type=EntityType.CONCEPT)
        c = Entity(name="C", entity_type=EntityType.CONCEPT)

        for e in [a, b, c]:
            await memory_store.create_entity(e)

        await memory_store.create_relationship(
            Relationship(source_id=a.id, target_id=b.id, relation_type=RelationType.RELATED_TO)
        )
        await memory_store.create_relationship(
            Relationship(source_id=b.id, target_id=c.id, relation_type=RelationType.RELATED_TO)
        )

        # Get direct neighbors
        neighbors = await memory_store.get_neighbors(a.id, depth=1)
        assert len(neighbors) == 1
        assert neighbors[0].name == "B"

        # Get 2-hop neighbors
        neighbors_2 = await memory_store.get_neighbors(a.id, depth=2)
        assert len(neighbors_2) == 2

    @pytest.mark.asyncio
    async def test_find_path(self, memory_store):
        """Test path finding."""
        from aion.systems.knowledge.types import Entity, EntityType, Relationship, RelationType

        await memory_store.initialize()

        # Create entities
        entities = [Entity(name=f"Node{i}", entity_type=EntityType.CONCEPT) for i in range(5)]
        for e in entities:
            await memory_store.create_entity(e)

        # Create chain: 0 -> 1 -> 2 -> 3 -> 4
        for i in range(4):
            await memory_store.create_relationship(
                Relationship(
                    source_id=entities[i].id,
                    target_id=entities[i+1].id,
                    relation_type=RelationType.RELATED_TO,
                )
            )

        # Find path from 0 to 4
        paths = await memory_store.find_path(
            entities[0].id,
            entities[4].id,
            max_depth=5,
        )

        assert len(paths) > 0
        assert paths[0].length == 4


class TestSQLiteStore:
    """Test SQLite graph store."""

    @pytest.fixture
    def sqlite_store(self):
        """Create a SQLite store for testing."""
        from aion.systems.knowledge.store.sqlite_store import SQLiteGraphStore

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_knowledge.db"
            yield SQLiteGraphStore(str(db_path))

    @pytest.mark.asyncio
    async def test_initialization(self, sqlite_store):
        """Test SQLite store initialization."""
        await sqlite_store.initialize()
        await sqlite_store.shutdown()

    @pytest.mark.asyncio
    async def test_entity_persistence(self, sqlite_store):
        """Test entity persistence."""
        from aion.systems.knowledge.types import Entity, EntityType

        await sqlite_store.initialize()

        entity = Entity(
            name="Persistent Entity",
            entity_type=EntityType.PROJECT,
            description="This should persist",
            properties={"key": "value"},
        )

        await sqlite_store.create_entity(entity)

        # Retrieve
        retrieved = await sqlite_store.get_entity(entity.id)
        assert retrieved is not None
        assert retrieved.name == "Persistent Entity"
        assert retrieved.properties.get("key") == "value"

        await sqlite_store.shutdown()

    @pytest.mark.asyncio
    async def test_full_text_search(self, sqlite_store):
        """Test full-text search with FTS5."""
        from aion.systems.knowledge.types import Entity, EntityType

        await sqlite_store.initialize()

        # Create entities with descriptive text
        entities = [
            Entity(
                name="Machine Learning",
                entity_type=EntityType.CONCEPT,
                description="A subset of artificial intelligence focused on learning from data",
            ),
            Entity(
                name="Deep Learning",
                entity_type=EntityType.CONCEPT,
                description="Machine learning with neural networks having many layers",
            ),
            Entity(
                name="Natural Language Processing",
                entity_type=EntityType.CONCEPT,
                description="Processing and understanding human language with computers",
            ),
        ]

        for e in entities:
            await sqlite_store.create_entity(e)

        # Search
        results = await sqlite_store.search_entities("machine learning", limit=10)
        assert len(results) >= 2

        await sqlite_store.shutdown()


class TestQueryEngine:
    """Test query engine."""

    @pytest.fixture
    def query_engine(self):
        """Create a query engine for testing."""
        from aion.systems.knowledge.query.engine import QueryEngine
        from aion.systems.knowledge.store.memory_store import InMemoryGraphStore

        store = InMemoryGraphStore()
        return QueryEngine(store)

    @pytest.mark.asyncio
    async def test_simple_query(self, query_engine):
        """Test simple query execution."""
        from aion.systems.knowledge.types import Entity, EntityType

        await query_engine.store.initialize()
        await query_engine.initialize()

        # Add test data
        entity = Entity(name="Test", entity_type=EntityType.CONCEPT)
        await query_engine.store.create_entity(entity)

        # Query
        result = await query_engine.execute("MATCH (e:concept) RETURN e LIMIT 10")
        assert result.total_count >= 1

    @pytest.mark.asyncio
    async def test_query_caching(self, query_engine):
        """Test query result caching."""
        from aion.systems.knowledge.types import Entity, EntityType

        await query_engine.store.initialize()
        await query_engine.initialize()

        # Add test data
        for i in range(5):
            entity = Entity(name=f"Entity{i}", entity_type=EntityType.CONCEPT)
            await query_engine.store.create_entity(entity)

        # Execute query twice
        query = "MATCH (e:concept) RETURN e"
        result1 = await query_engine.execute(query)
        result2 = await query_engine.execute(query)

        # Second should be from cache (faster)
        assert result1.total_count == result2.total_count


class TestInferenceEngine:
    """Test inference engine."""

    @pytest.fixture
    def inference_engine(self):
        """Create an inference engine for testing."""
        from aion.systems.knowledge.inference.engine import InferenceEngine
        from aion.systems.knowledge.store.memory_store import InMemoryGraphStore

        store = InMemoryGraphStore()
        return InferenceEngine(store)

    @pytest.mark.asyncio
    async def test_transitive_inference(self, inference_engine):
        """Test transitive relationship inference."""
        from aion.systems.knowledge.types import Entity, EntityType, Relationship, RelationType

        await inference_engine.store.initialize()

        # Create hierarchy: A is_a B, B is_a C => A is_a C
        a = Entity(name="Dog", entity_type=EntityType.CONCEPT)
        b = Entity(name="Animal", entity_type=EntityType.CONCEPT)
        c = Entity(name="LivingThing", entity_type=EntityType.CONCEPT)

        for e in [a, b, c]:
            await inference_engine.store.create_entity(e)

        await inference_engine.store.create_relationship(
            Relationship(source_id=a.id, target_id=b.id, relation_type=RelationType.IS_A)
        )
        await inference_engine.store.create_relationship(
            Relationship(source_id=b.id, target_id=c.id, relation_type=RelationType.IS_A)
        )

        # Run inference
        inferred = await inference_engine.infer_transitive(RelationType.IS_A)

        # Should infer A is_a C
        assert len(inferred) >= 1

    @pytest.mark.asyncio
    async def test_centrality_computation(self, inference_engine):
        """Test centrality computation."""
        from aion.systems.knowledge.types import Entity, EntityType, Relationship, RelationType

        await inference_engine.store.initialize()

        # Create star graph with center node
        center = Entity(name="Center", entity_type=EntityType.CONCEPT)
        await inference_engine.store.create_entity(center)

        for i in range(5):
            leaf = Entity(name=f"Leaf{i}", entity_type=EntityType.CONCEPT)
            await inference_engine.store.create_entity(leaf)
            await inference_engine.store.create_relationship(
                Relationship(source_id=center.id, target_id=leaf.id, relation_type=RelationType.RELATED_TO)
            )

        # Compute centrality
        centrality = await inference_engine.compute_centrality()

        # Center should have highest centrality
        assert center.id in centrality
        center_score = centrality[center.id]
        assert center_score > 0


class TestHybridSearch:
    """Test hybrid search functionality."""

    @pytest.fixture
    def hybrid_search(self):
        """Create a hybrid search instance for testing."""
        from aion.systems.knowledge.hybrid.search import HybridSearch
        from aion.systems.knowledge.store.memory_store import InMemoryGraphStore

        store = InMemoryGraphStore()
        return HybridSearch(store)

    @pytest.mark.asyncio
    async def test_basic_search(self, hybrid_search):
        """Test basic hybrid search."""
        from aion.systems.knowledge.types import Entity, EntityType

        await hybrid_search.store.initialize()

        # Add test entities
        entities = [
            Entity(name="Python", entity_type=EntityType.TOOL, description="Programming language"),
            Entity(name="JavaScript", entity_type=EntityType.TOOL, description="Web programming language"),
            Entity(name="Machine Learning", entity_type=EntityType.CONCEPT, description="AI technique"),
        ]

        for e in entities:
            await hybrid_search.store.create_entity(e)

        # Search
        results = await hybrid_search.search("programming language", limit=5)
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_weighted_search(self, hybrid_search):
        """Test search with different weights."""
        from aion.systems.knowledge.types import Entity, EntityType

        await hybrid_search.store.initialize()

        entity = Entity(
            name="Test Entity",
            entity_type=EntityType.CONCEPT,
            description="A test entity for search",
        )
        await hybrid_search.store.create_entity(entity)

        # Search with text weight emphasized
        results = await hybrid_search.search(
            "test entity",
            limit=5,
            vector_weight=0.2,
            graph_weight=0.2,
            text_weight=0.6,
        )
        assert len(results) > 0


class TestEntityExtraction:
    """Test entity extraction."""

    @pytest.fixture
    def extractor(self):
        """Create an entity extractor for testing."""
        from aion.systems.knowledge.extraction.extractor import EntityExtractor

        return EntityExtractor()

    @pytest.mark.asyncio
    async def test_fallback_extraction(self, extractor):
        """Test fallback regex-based extraction."""
        text = "John Smith works at Microsoft in Seattle. He collaborates with Jane Doe on the Azure project."

        result = extractor._fallback_extract(text)

        # Should extract capitalized names
        entity_names = {e.name for e in result.entities}
        assert "John Smith" in entity_names or "John" in entity_names


class TestEntityLinker:
    """Test entity linker."""

    @pytest.fixture
    def linker(self):
        """Create an entity linker for testing."""
        from aion.systems.knowledge.extraction.linker import EntityLinker

        return EntityLinker()

    def test_exact_match(self, linker):
        """Test exact name matching."""
        from aion.systems.knowledge.types import Entity, EntityType

        entity = Entity(name="Microsoft", entity_type=EntityType.ORGANIZATION)
        candidates = [
            Entity(name="Microsoft", entity_type=EntityType.ORGANIZATION),
            Entity(name="Apple", entity_type=EntityType.ORGANIZATION),
        ]

        match = linker.find_best_match(entity, candidates)
        assert match is not None
        assert match.name == "Microsoft"

    def test_alias_match(self, linker):
        """Test alias matching."""
        from aion.systems.knowledge.types import Entity, EntityType

        entity = Entity(name="MS", entity_type=EntityType.ORGANIZATION)
        candidates = [
            Entity(name="Microsoft", entity_type=EntityType.ORGANIZATION, aliases=["MS", "MSFT"]),
        ]

        match = linker.find_best_match(entity, candidates)
        assert match is not None
        assert match.name == "Microsoft"

    def test_fuzzy_match(self, linker):
        """Test fuzzy string matching."""
        from aion.systems.knowledge.types import Entity, EntityType

        entity = Entity(name="Microsft", entity_type=EntityType.ORGANIZATION)  # Typo
        candidates = [
            Entity(name="Microsoft", entity_type=EntityType.ORGANIZATION),
        ]

        match = linker.find_best_match(entity, candidates)
        assert match is not None  # Should match despite typo

    def test_entity_merge(self, linker):
        """Test entity merging."""
        from aion.systems.knowledge.types import Entity, EntityType

        primary = Entity(
            name="Microsoft",
            entity_type=EntityType.ORGANIZATION,
            properties={"founded": "1975"},
        )

        secondary = Entity(
            name="MS",
            entity_type=EntityType.ORGANIZATION,
            properties={"headquarters": "Redmond"},
            description="Technology company",
        )

        merged = linker.merge_entities(primary, secondary)

        assert merged.name == "Microsoft"
        assert "MS" in merged.aliases
        assert merged.properties.get("founded") == "1975"
        assert merged.properties.get("headquarters") == "Redmond"
        assert merged.description == "Technology company"


class TestOntologySchema:
    """Test ontology schema."""

    @pytest.fixture
    def schema(self):
        """Create an ontology schema for testing."""
        from aion.systems.knowledge.ontology.schema import OntologySchema

        return OntologySchema()

    def test_default_types(self, schema):
        """Test that default types are registered."""
        assert "person" in schema.entity_types
        assert "organization" in schema.entity_types
        assert "works_for" in schema.relation_types
        assert "knows" in schema.relation_types

    def test_type_hierarchy(self, schema):
        """Test type hierarchy."""
        hierarchy = schema.get_type_hierarchy("person")
        assert "person" in hierarchy
        assert "thing" in hierarchy

    def test_subtype_check(self, schema):
        """Test subtype checking."""
        assert schema.is_subtype_of("person", "thing")
        assert not schema.is_subtype_of("thing", "person")

    def test_allowed_relationships(self, schema):
        """Test getting allowed relationships."""
        allowed = schema.get_allowed_relationships("person", "organization")
        assert "works_for" in allowed
        assert "member_of" in allowed


class TestSchemaValidation:
    """Test schema validation."""

    @pytest.fixture
    def validator(self):
        """Create a schema validator for testing."""
        from aion.systems.knowledge.ontology.validation import SchemaValidator

        return SchemaValidator()

    def test_validate_entity(self, validator):
        """Test entity validation."""
        from aion.systems.knowledge.types import Entity, EntityType

        # Valid entity
        entity = Entity(
            name="Test Person",
            entity_type=EntityType.PERSON,
        )
        result = validator.validate_entity(entity)
        assert result.valid

        # Invalid entity (empty name)
        invalid = Entity(name="", entity_type=EntityType.PERSON)
        result = validator.validate_entity(invalid)
        assert not result.valid

    def test_validate_relationship(self, validator):
        """Test relationship validation."""
        from aion.systems.knowledge.types import Entity, EntityType, Relationship, RelationType

        person = Entity(name="Alice", entity_type=EntityType.PERSON)
        org = Entity(name="Acme", entity_type=EntityType.ORGANIZATION)

        # Valid relationship
        rel = Relationship(
            source_id=person.id,
            target_id=org.id,
            relation_type=RelationType.WORKS_FOR,
        )
        result = validator.validate_relationship(rel, person, org)
        assert result.valid


class TestKnowledgeManager:
    """Test the main KnowledgeManager coordinator."""

    @pytest.fixture
    def knowledge_manager(self):
        """Create a KnowledgeManager for testing."""
        from aion.systems.knowledge.manager import KnowledgeManager, KnowledgeManagerConfig

        config = KnowledgeManagerConfig(store_type="memory")
        return KnowledgeManager(config)

    @pytest.mark.asyncio
    async def test_initialization(self, knowledge_manager):
        """Test manager initialization."""
        await knowledge_manager.initialize()
        assert knowledge_manager._initialized
        await knowledge_manager.shutdown()

    @pytest.mark.asyncio
    async def test_add_and_get_entity(self, knowledge_manager):
        """Test adding and retrieving entities."""
        from aion.systems.knowledge.types import Entity, EntityType

        await knowledge_manager.initialize()

        entity = Entity(
            name="Test Entity",
            entity_type=EntityType.CONCEPT,
        )

        added = await knowledge_manager.add_entity(entity)
        assert added.id == entity.id

        retrieved = await knowledge_manager.get_entity(entity.id)
        assert retrieved is not None
        assert retrieved.name == "Test Entity"

        await knowledge_manager.shutdown()

    @pytest.mark.asyncio
    async def test_add_relationship(self, knowledge_manager):
        """Test adding relationships."""
        from aion.systems.knowledge.types import Entity, EntityType, Relationship, RelationType

        await knowledge_manager.initialize()

        e1 = Entity(name="A", entity_type=EntityType.CONCEPT)
        e2 = Entity(name="B", entity_type=EntityType.CONCEPT)

        await knowledge_manager.add_entity(e1)
        await knowledge_manager.add_entity(e2)

        rel = Relationship(
            source_id=e1.id,
            target_id=e2.id,
            relation_type=RelationType.RELATED_TO,
        )

        added = await knowledge_manager.add_relationship(rel)
        assert added.id == rel.id

        await knowledge_manager.shutdown()

    @pytest.mark.asyncio
    async def test_search(self, knowledge_manager):
        """Test hybrid search through manager."""
        from aion.systems.knowledge.types import Entity, EntityType

        await knowledge_manager.initialize()

        entities = [
            Entity(name="Machine Learning", entity_type=EntityType.CONCEPT),
            Entity(name="Deep Learning", entity_type=EntityType.CONCEPT),
            Entity(name="Natural Language Processing", entity_type=EntityType.CONCEPT),
        ]

        for e in entities:
            await knowledge_manager.add_entity(e)

        results = await knowledge_manager.search("machine learning", limit=5)
        assert len(results) > 0

        await knowledge_manager.shutdown()

    @pytest.mark.asyncio
    async def test_get_stats(self, knowledge_manager):
        """Test getting statistics."""
        from aion.systems.knowledge.types import Entity, EntityType

        await knowledge_manager.initialize()

        # Add some test data
        for i in range(5):
            entity = Entity(name=f"Entity{i}", entity_type=EntityType.CONCEPT)
            await knowledge_manager.add_entity(entity)

        stats = await knowledge_manager.get_stats()
        assert stats.total_entities >= 5

        await knowledge_manager.shutdown()


class TestKnowledgeConfig:
    """Test knowledge graph configuration."""

    def test_config_defaults(self):
        """Test default configuration values."""
        from aion.core.config import KnowledgeGraphConfig

        config = KnowledgeGraphConfig()

        assert config.store_type == "sqlite"
        assert config.embedding_dimension == 384
        assert config.enable_inference is True
        assert config.vector_weight == 0.4
        assert config.graph_weight == 0.3
        assert config.text_weight == 0.3

    def test_config_in_aion_config(self):
        """Test knowledge graph config in main config."""
        from aion.core.config import AIONConfig

        config = AIONConfig()

        assert hasattr(config, "knowledge_graph")
        assert config.knowledge_graph.store_type == "sqlite"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
