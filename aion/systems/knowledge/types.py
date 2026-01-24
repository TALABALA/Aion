"""
AION Knowledge Graph Types

State-of-the-art dataclasses for entity-relationship modeling with:
- Temporal reasoning (time-scoped relationships)
- Uncertainty quantification (probabilistic weights)
- Graph embeddings (TransE-style for relation prediction)
- Provenance tracking
- Schema versioning
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Set, Tuple, Union
import hashlib
import uuid


# =============================================================================
# Entity Types
# =============================================================================

class EntityType(str, Enum):
    """
    Built-in entity types with hierarchical semantics.

    Hierarchy:
    - THING (root)
      - AGENT (can take actions)
        - PERSON
        - ORGANIZATION
        - AI_AGENT
      - ARTIFACT (created things)
        - PROJECT
        - DOCUMENT
        - CODE
        - TOOL
      - ABSTRACT
        - CONCEPT
        - SKILL
        - TOPIC
      - EVENT (temporal occurrences)
        - MEETING
        - MILESTONE
      - SPATIAL
        - LOCATION
      - TEMPORAL
        - TIME_PERIOD
      - TASK (actionable items)
        - GOAL
        - OBJECTIVE
      - MEMORY (AION internal)
    """
    # Root
    THING = "thing"

    # Agents
    PERSON = "person"
    ORGANIZATION = "organization"
    TEAM = "team"
    AI_AGENT = "ai_agent"

    # Artifacts
    PROJECT = "project"
    DOCUMENT = "document"
    CODE = "code"
    TOOL = "tool"
    ARTIFACT = "artifact"

    # Abstract concepts
    CONCEPT = "concept"
    SKILL = "skill"
    TOPIC = "topic"
    CATEGORY = "category"

    # Events
    EVENT = "event"
    MEETING = "meeting"
    MILESTONE = "milestone"

    # Spatial
    LOCATION = "location"

    # Temporal
    TIME_PERIOD = "time_period"

    # Tasks
    TASK = "task"
    GOAL = "goal"
    OBJECTIVE = "objective"

    # AION internal
    MEMORY = "memory"
    QUERY = "query"

    # Custom
    CUSTOM = "custom"

    @classmethod
    def get_hierarchy(cls) -> Dict[str, List[str]]:
        """Get the type hierarchy as parent -> children mapping."""
        return {
            "thing": ["agent", "artifact", "abstract", "event", "spatial", "temporal", "task", "memory"],
            "agent": ["person", "organization", "team", "ai_agent"],
            "artifact": ["project", "document", "code", "tool"],
            "abstract": ["concept", "skill", "topic", "category"],
            "event": ["meeting", "milestone"],
            "spatial": ["location"],
            "temporal": ["time_period"],
            "task": ["goal", "objective"],
        }

    def is_subtype_of(self, parent: EntityType) -> bool:
        """Check if this type is a subtype of another."""
        if self == parent:
            return True
        hierarchy = self.get_hierarchy()
        for p, children in hierarchy.items():
            if self.value in children and (parent.value == p or EntityType(p).is_subtype_of(parent)):
                return True
        return False


# =============================================================================
# Relationship Types
# =============================================================================

class RelationType(str, Enum):
    """
    Built-in relationship types with semantic properties.

    Properties:
    - Symmetric: A-R-B implies B-R-A
    - Transitive: A-R-B and B-R-C implies A-R-C
    - Reflexive: A-R-A is always true
    - Inverse: R has an inverse R' such that A-R-B implies B-R'-A
    """
    # Organizational (hierarchical)
    WORKS_FOR = "works_for"           # inverse: employs
    MANAGES = "manages"               # inverse: managed_by
    REPORTS_TO = "reports_to"         # inverse: supervises
    MEMBER_OF = "member_of"           # inverse: has_member
    LEADS = "leads"                   # inverse: led_by

    # Social (often symmetric)
    KNOWS = "knows"                   # symmetric
    COLLABORATES_WITH = "collaborates_with"  # symmetric
    COMMUNICATES_WITH = "communicates_with"  # symmetric

    # Creation/Ownership
    CREATED = "created"               # inverse: created_by
    OWNS = "owns"                     # inverse: owned_by
    AUTHORED = "authored"             # inverse: authored_by
    CONTRIBUTED_TO = "contributed_to" # inverse: has_contributor
    MAINTAINS = "maintains"           # inverse: maintained_by

    # Conceptual (taxonomic)
    IS_A = "is_a"                     # transitive
    INSTANCE_OF = "instance_of"
    SUBCLASS_OF = "subclass_of"       # transitive
    PART_OF = "part_of"               # transitive
    HAS_PART = "has_part"             # inverse of part_of

    # Semantic similarity
    RELATED_TO = "related_to"         # symmetric
    SIMILAR_TO = "similar_to"         # symmetric
    DERIVED_FROM = "derived_from"     # inverse: has_derivative
    EQUIVALENT_TO = "equivalent_to"   # symmetric, transitive, reflexive

    # Temporal/Causal
    PRECEDED_BY = "preceded_by"       # inverse: followed_by
    FOLLOWED_BY = "followed_by"       # inverse: preceded_by
    CAUSED = "caused"                 # inverse: caused_by
    ENABLES = "enables"               # inverse: enabled_by
    TRIGGERS = "triggers"             # inverse: triggered_by

    # Spatial
    LOCATED_IN = "located_in"         # transitive
    CONTAINS = "contains"             # inverse: located_in
    NEAR = "near"                     # symmetric

    # Task/Goal
    DEPENDS_ON = "depends_on"         # inverse: dependency_of
    BLOCKS = "blocks"                 # inverse: blocked_by
    ASSIGNED_TO = "assigned_to"       # inverse: assigned
    REQUIRED_FOR = "required_for"     # inverse: requires
    ACHIEVES = "achieves"             # inverse: achieved_by

    # Reference
    REFERENCES = "references"         # inverse: referenced_by
    MENTIONS = "mentions"             # inverse: mentioned_in
    CITES = "cites"                   # inverse: cited_by

    # Custom
    CUSTOM = "custom"

    @classmethod
    def get_properties(cls) -> Dict[str, Dict[str, Any]]:
        """Get semantic properties for each relation type."""
        return {
            "knows": {"symmetric": True},
            "collaborates_with": {"symmetric": True},
            "communicates_with": {"symmetric": True},
            "related_to": {"symmetric": True},
            "similar_to": {"symmetric": True},
            "equivalent_to": {"symmetric": True, "transitive": True, "reflexive": True},
            "near": {"symmetric": True},
            "is_a": {"transitive": True},
            "subclass_of": {"transitive": True},
            "part_of": {"transitive": True},
            "located_in": {"transitive": True},
            "works_for": {"inverse": "employs"},
            "manages": {"inverse": "managed_by"},
            "reports_to": {"inverse": "supervises"},
            "member_of": {"inverse": "has_member"},
            "created": {"inverse": "created_by"},
            "owns": {"inverse": "owned_by"},
            "preceded_by": {"inverse": "followed_by"},
            "followed_by": {"inverse": "preceded_by"},
            "caused": {"inverse": "caused_by"},
            "depends_on": {"inverse": "dependency_of"},
            "blocks": {"inverse": "blocked_by"},
            "part_of": {"inverse": "has_part"},
            "contains": {"inverse": "located_in"},
        }

    def is_symmetric(self) -> bool:
        """Check if this relation is symmetric."""
        props = self.get_properties().get(self.value, {})
        return props.get("symmetric", False)

    def is_transitive(self) -> bool:
        """Check if this relation is transitive."""
        props = self.get_properties().get(self.value, {})
        return props.get("transitive", False)

    def get_inverse(self) -> Optional[str]:
        """Get the inverse relation type."""
        props = self.get_properties().get(self.value, {})
        return props.get("inverse")


# =============================================================================
# Provenance
# =============================================================================

@dataclass
class Provenance:
    """
    Track the origin and confidence of knowledge.

    Supports:
    - Source attribution
    - Extraction method tracking
    - Confidence propagation
    - Temporal validity
    """
    source_type: str = "manual"  # manual, extraction, inference, external
    source_id: Optional[str] = None  # ID of source document/entity
    source_uri: Optional[str] = None  # External URI if applicable
    extraction_method: Optional[str] = None  # LLM model, rule, etc.
    extraction_confidence: float = 1.0  # Confidence from extraction
    human_verified: bool = False
    verified_by: Optional[str] = None
    verified_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "source_type": self.source_type,
            "source_id": self.source_id,
            "source_uri": self.source_uri,
            "extraction_method": self.extraction_method,
            "extraction_confidence": self.extraction_confidence,
            "human_verified": self.human_verified,
            "created_at": self.created_at.isoformat(),
        }


# =============================================================================
# Entity Property
# =============================================================================

@dataclass
class EntityProperty:
    """
    A typed property of an entity with provenance.
    """
    key: str
    value: Any
    data_type: str = "string"  # string, number, boolean, date, json, embedding
    indexed: bool = False
    searchable: bool = True
    provenance: Optional[Provenance] = None
    confidence: float = 1.0
    valid_from: Optional[datetime] = None
    valid_until: Optional[datetime] = None

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "value": self.value,
            "data_type": self.data_type,
            "confidence": self.confidence,
        }

    def is_valid_at(self, timestamp: datetime) -> bool:
        """Check if property is valid at a given time."""
        if self.valid_from and timestamp < self.valid_from:
            return False
        if self.valid_until and timestamp > self.valid_until:
            return False
        return True


# =============================================================================
# Graph Embedding
# =============================================================================

@dataclass
class GraphEmbedding:
    """
    Learned embedding for graph elements (entities/relations).

    Supports multiple embedding types:
    - TransE: h + r ≈ t
    - RotatE: h ∘ r ≈ t (rotation in complex space)
    - DistMult: <h, r, t> score
    - Text: Semantic embedding from description
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    embedding_type: str = "text"  # text, transe, rotate, distmult
    vector: List[float] = field(default_factory=list)
    dimension: int = 0
    model_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        if self.vector and not self.dimension:
            self.dimension = len(self.vector)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.embedding_type,
            "dimension": self.dimension,
            "model_id": self.model_id,
        }


# =============================================================================
# Entity
# =============================================================================

@dataclass
class Entity:
    """
    A node in the knowledge graph.

    Entities represent things: people, organizations, concepts, etc.
    Features:
    - Multiple embeddings (text, graph)
    - Temporal properties with validity windows
    - Provenance tracking
    - Version history
    - Computed hash for deduplication
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Core identity
    name: str = ""
    entity_type: EntityType = EntityType.CUSTOM
    custom_type: Optional[str] = None

    # Description
    description: str = ""
    aliases: List[str] = field(default_factory=list)

    # Properties (key -> value with metadata)
    properties: Dict[str, Any] = field(default_factory=dict)
    typed_properties: List[EntityProperty] = field(default_factory=list)

    # Embeddings
    text_embedding: Optional[GraphEmbedding] = None
    graph_embedding: Optional[GraphEmbedding] = None

    # Provenance
    provenance: Optional[Provenance] = None

    # Confidence and importance
    confidence: float = 1.0
    importance: float = 0.5  # PageRank-style importance score

    # Centrality metrics (computed)
    degree_centrality: float = 0.0
    betweenness_centrality: float = 0.0
    pagerank: float = 0.0

    # Version tracking
    version: int = 1
    previous_version_id: Optional[str] = None

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    # Soft delete
    deleted_at: Optional[datetime] = None

    # Content hash for deduplication
    _content_hash: Optional[str] = field(default=None, repr=False)

    def __post_init__(self):
        """Compute content hash after initialization."""
        self._compute_hash()

    def _compute_hash(self) -> None:
        """Compute content hash for deduplication."""
        content = f"{self.name}|{self.entity_type.value}|{self.description}"
        self._content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

    @property
    def content_hash(self) -> str:
        """Get content hash, computing if needed."""
        if not self._content_hash:
            self._compute_hash()
        return self._content_hash

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.entity_type.value,
            "custom_type": self.custom_type,
            "description": self.description,
            "aliases": self.aliases,
            "properties": self.properties,
            "confidence": self.confidence,
            "importance": self.importance,
            "pagerank": self.pagerank,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    def matches_type(self, type_filter: Union[EntityType, str]) -> bool:
        """Check if entity matches a type filter (including subtypes)."""
        if isinstance(type_filter, str):
            try:
                type_filter = EntityType(type_filter)
            except ValueError:
                return self.custom_type == type_filter

        if self.entity_type == type_filter:
            return True

        return self.entity_type.is_subtype_of(type_filter)

    def get_property(self, key: str, default: Any = None) -> Any:
        """Get property value with optional default."""
        return self.properties.get(key, default)

    def set_property(self, key: str, value: Any, **kwargs) -> None:
        """Set a property with optional metadata."""
        self.properties[key] = value
        if kwargs:
            prop = EntityProperty(key=key, value=value, **kwargs)
            # Update or add typed property
            for i, p in enumerate(self.typed_properties):
                if p.key == key:
                    self.typed_properties[i] = prop
                    return
            self.typed_properties.append(prop)

    def merge_from(self, other: Entity) -> None:
        """Merge properties from another entity."""
        # Merge properties
        for key, value in other.properties.items():
            if key not in self.properties:
                self.properties[key] = value

        # Merge aliases
        for alias in other.aliases:
            if alias not in self.aliases:
                self.aliases.append(alias)

        # Update confidence (max)
        self.confidence = max(self.confidence, other.confidence)

        # Update timestamp
        self.updated_at = datetime.now()
        self.version += 1


# =============================================================================
# Relationship
# =============================================================================

@dataclass
class Relationship:
    """
    An edge in the knowledge graph.

    Relationships connect entities with typed, directional edges.
    Features:
    - Temporal scoping (valid_from/valid_until)
    - Confidence and weight
    - Properties with provenance
    - Bidirectional flag for symmetric relations
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Endpoints
    source_id: str = ""
    target_id: str = ""

    # Type
    relation_type: RelationType = RelationType.RELATED_TO
    custom_type: Optional[str] = None

    # Properties
    properties: Dict[str, Any] = field(default_factory=dict)

    # Strength/confidence
    weight: float = 1.0  # Edge weight for graph algorithms
    confidence: float = 1.0  # Confidence in relationship existence

    # Semantic properties (can override type defaults)
    bidirectional: bool = False
    transitive: bool = False

    # Temporal bounds
    valid_from: Optional[datetime] = None
    valid_until: Optional[datetime] = None

    # Provenance
    provenance: Optional[Provenance] = None

    # Version tracking
    version: int = 1

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Initialize bidirectional flag from relation type properties."""
        if not self.bidirectional:
            self.bidirectional = self.relation_type.is_symmetric()
        if not self.transitive:
            self.transitive = self.relation_type.is_transitive()

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "type": self.relation_type.value,
            "custom_type": self.custom_type,
            "weight": self.weight,
            "confidence": self.confidence,
            "bidirectional": self.bidirectional,
            "properties": self.properties,
            "valid_from": self.valid_from.isoformat() if self.valid_from else None,
            "valid_until": self.valid_until.isoformat() if self.valid_until else None,
            "created_at": self.created_at.isoformat(),
        }

    def is_valid_at(self, timestamp: datetime) -> bool:
        """Check if relationship is valid at a given time."""
        if self.valid_from and timestamp < self.valid_from:
            return False
        if self.valid_until and timestamp > self.valid_until:
            return False
        return True

    def is_active(self) -> bool:
        """Check if relationship is currently active."""
        return self.is_valid_at(datetime.now())

    def matches_type(self, type_filter: Union[RelationType, str]) -> bool:
        """Check if relationship matches a type filter."""
        if isinstance(type_filter, RelationType):
            return self.relation_type == type_filter
        return self.relation_type.value == type_filter or self.custom_type == type_filter


# =============================================================================
# Triple
# =============================================================================

@dataclass
class Triple:
    """
    A subject-predicate-object triple.

    The fundamental unit of knowledge representation.
    """
    subject: Entity
    predicate: RelationType
    object: Entity
    relationship: Optional[Relationship] = None
    confidence: float = 1.0

    def to_string(self) -> str:
        """Convert to human-readable string."""
        return f"({self.subject.name})-[{self.predicate.value}]->({self.object.name})"

    def to_dict(self) -> dict:
        return {
            "subject": self.subject.to_dict(),
            "predicate": self.predicate.value,
            "object": self.object.to_dict(),
            "confidence": self.confidence,
        }

    def reverse(self) -> Triple:
        """Create reverse triple if relation has inverse."""
        inverse = self.predicate.get_inverse()
        if inverse:
            try:
                inverse_type = RelationType(inverse)
            except ValueError:
                inverse_type = RelationType.CUSTOM
            return Triple(
                subject=self.object,
                predicate=inverse_type,
                object=self.subject,
                confidence=self.confidence,
            )
        return None


# =============================================================================
# Path
# =============================================================================

@dataclass
class Path:
    """
    A path through the graph.

    Represents a sequence of entities connected by relationships.
    """
    entities: List[Entity] = field(default_factory=list)
    relationships: List[Relationship] = field(default_factory=list)

    # Path metrics
    total_weight: float = 0.0
    min_confidence: float = 1.0

    @property
    def length(self) -> int:
        """Number of hops in the path."""
        return len(self.relationships)

    @property
    def start(self) -> Optional[Entity]:
        """First entity in path."""
        return self.entities[0] if self.entities else None

    @property
    def end(self) -> Optional[Entity]:
        """Last entity in path."""
        return self.entities[-1] if self.entities else None

    def to_string(self) -> str:
        """Convert to human-readable string."""
        if not self.entities:
            return ""

        parts = [self.entities[0].name]
        for i, rel in enumerate(self.relationships):
            parts.append(f"-[{rel.relation_type.value}]->")
            parts.append(self.entities[i + 1].name)

        return "".join(parts)

    def to_dict(self) -> dict:
        return {
            "path": self.to_string(),
            "length": self.length,
            "total_weight": self.total_weight,
            "min_confidence": self.min_confidence,
            "entities": [e.to_dict() for e in self.entities],
            "relationships": [r.to_dict() for r in self.relationships],
        }

    def compute_metrics(self) -> None:
        """Compute path metrics (weight, confidence)."""
        if not self.relationships:
            return

        self.total_weight = sum(r.weight for r in self.relationships)
        self.min_confidence = min(r.confidence for r in self.relationships)

    def extend(self, entity: Entity, relationship: Relationship) -> Path:
        """Create new path extended by one hop."""
        new_path = Path(
            entities=self.entities + [entity],
            relationships=self.relationships + [relationship],
        )
        new_path.compute_metrics()
        return new_path


# =============================================================================
# Subgraph
# =============================================================================

@dataclass
class Subgraph:
    """
    A subset of the knowledge graph.

    Used for query results, neighborhood extraction, and subgraph analysis.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""

    entities: Dict[str, Entity] = field(default_factory=dict)
    relationships: Dict[str, Relationship] = field(default_factory=dict)

    # Adjacency indices for fast lookup
    _outgoing: Dict[str, Set[str]] = field(default_factory=dict, repr=False)
    _incoming: Dict[str, Set[str]] = field(default_factory=dict, repr=False)

    @property
    def num_entities(self) -> int:
        return len(self.entities)

    @property
    def num_relationships(self) -> int:
        return len(self.relationships)

    def add_entity(self, entity: Entity) -> None:
        """Add an entity to the subgraph."""
        self.entities[entity.id] = entity
        if entity.id not in self._outgoing:
            self._outgoing[entity.id] = set()
        if entity.id not in self._incoming:
            self._incoming[entity.id] = set()

    def add_relationship(self, rel: Relationship) -> None:
        """Add a relationship to the subgraph."""
        self.relationships[rel.id] = rel

        # Update adjacency indices
        if rel.source_id not in self._outgoing:
            self._outgoing[rel.source_id] = set()
        self._outgoing[rel.source_id].add(rel.id)

        if rel.target_id not in self._incoming:
            self._incoming[rel.target_id] = set()
        self._incoming[rel.target_id].add(rel.id)

        # Handle bidirectional
        if rel.bidirectional:
            if rel.target_id not in self._outgoing:
                self._outgoing[rel.target_id] = set()
            self._outgoing[rel.target_id].add(rel.id)

            if rel.source_id not in self._incoming:
                self._incoming[rel.source_id] = set()
            self._incoming[rel.source_id].add(rel.id)

    def get_neighbors(
        self,
        entity_id: str,
        direction: str = "both",  # outgoing, incoming, both
    ) -> List[Entity]:
        """Get all neighbors of an entity."""
        neighbors = []
        seen = set()

        if direction in ("outgoing", "both"):
            for rel_id in self._outgoing.get(entity_id, set()):
                rel = self.relationships.get(rel_id)
                if rel:
                    target_id = rel.target_id if rel.source_id == entity_id else rel.source_id
                    if target_id not in seen and target_id in self.entities:
                        neighbors.append(self.entities[target_id])
                        seen.add(target_id)

        if direction in ("incoming", "both"):
            for rel_id in self._incoming.get(entity_id, set()):
                rel = self.relationships.get(rel_id)
                if rel:
                    source_id = rel.source_id if rel.target_id == entity_id else rel.target_id
                    if source_id not in seen and source_id in self.entities:
                        neighbors.append(self.entities[source_id])
                        seen.add(source_id)

        return neighbors

    def get_relationships_for(
        self,
        entity_id: str,
        direction: str = "both",
    ) -> List[Relationship]:
        """Get all relationships for an entity."""
        rels = []
        seen = set()

        if direction in ("outgoing", "both"):
            for rel_id in self._outgoing.get(entity_id, set()):
                if rel_id not in seen:
                    rel = self.relationships.get(rel_id)
                    if rel:
                        rels.append(rel)
                        seen.add(rel_id)

        if direction in ("incoming", "both"):
            for rel_id in self._incoming.get(entity_id, set()):
                if rel_id not in seen:
                    rel = self.relationships.get(rel_id)
                    if rel:
                        rels.append(rel)
                        seen.add(rel_id)

        return rels

    def get_degree(self, entity_id: str) -> int:
        """Get the degree (number of connections) of an entity."""
        return len(self._outgoing.get(entity_id, set())) + len(self._incoming.get(entity_id, set()))

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "num_entities": self.num_entities,
            "num_relationships": self.num_relationships,
            "entities": [e.to_dict() for e in self.entities.values()],
            "relationships": [r.to_dict() for r in self.relationships.values()],
        }

    def to_adjacency_list(self) -> Dict[str, List[Tuple[str, str]]]:
        """Convert to adjacency list format: entity_id -> [(target_id, rel_type), ...]"""
        adj = {}
        for entity_id in self.entities:
            adj[entity_id] = []
            for rel in self.get_relationships_for(entity_id, direction="outgoing"):
                target = rel.target_id if rel.source_id == entity_id else rel.source_id
                adj[entity_id].append((target, rel.relation_type.value))
        return adj


# =============================================================================
# Query Types
# =============================================================================

@dataclass
class GraphQuery:
    """
    A query against the knowledge graph.

    Supports:
    - Pattern matching (MATCH)
    - Path finding (PATH)
    - Aggregation (AGGREGATE)
    - Subgraph extraction (SUBGRAPH)
    - Natural language (NL)
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Query type
    query_type: str = "match"  # match, path, aggregate, subgraph, nl

    # Entity constraints
    entity_types: List[EntityType] = field(default_factory=list)
    entity_filters: Dict[str, Any] = field(default_factory=dict)

    # Relationship constraints
    relation_types: List[RelationType] = field(default_factory=list)
    relation_filters: Dict[str, Any] = field(default_factory=dict)

    # Path finding
    start_entity_id: Optional[str] = None
    end_entity_id: Optional[str] = None
    max_depth: int = 5
    path_algorithm: str = "bfs"  # bfs, dijkstra, astar, bidirectional

    # Aggregation
    aggregation_type: Optional[str] = None  # count, sum, avg, min, max, group_by
    group_by_field: Optional[str] = None

    # Temporal filter
    valid_at: Optional[datetime] = None

    # Pagination
    limit: int = 100
    offset: int = 0

    # Ordering
    order_by: Optional[str] = None
    order_direction: str = "desc"  # asc, desc

    # Natural language
    natural_language: Optional[str] = None

    # Raw query (Cypher-like)
    raw_query: Optional[str] = None

    # Execution hints
    use_cache: bool = True
    explain: bool = False  # Return query plan
    timeout_ms: int = 30000

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.query_type,
            "entity_types": [t.value for t in self.entity_types],
            "relation_types": [t.value for t in self.relation_types],
            "entity_filters": self.entity_filters,
            "relation_filters": self.relation_filters,
            "max_depth": self.max_depth,
            "limit": self.limit,
            "natural_language": self.natural_language,
        }


@dataclass
class QueryResult:
    """Result from a graph query."""
    query_id: str = ""

    # Results
    entities: List[Entity] = field(default_factory=list)
    relationships: List[Relationship] = field(default_factory=list)
    paths: List[Path] = field(default_factory=list)
    triples: List[Triple] = field(default_factory=list)

    # Subgraph result
    subgraph: Optional[Subgraph] = None

    # Aggregates
    aggregates: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    total_count: int = 0
    has_more: bool = False
    execution_time_ms: float = 0.0

    # Query plan (if explain=True)
    query_plan: Optional[Dict[str, Any]] = None

    # Cache info
    from_cache: bool = False

    def to_dict(self) -> dict:
        return {
            "query_id": self.query_id,
            "entities": [e.to_dict() for e in self.entities],
            "relationships": [r.to_dict() for r in self.relationships],
            "paths": [p.to_dict() for p in self.paths],
            "aggregates": self.aggregates,
            "total_count": self.total_count,
            "has_more": self.has_more,
            "execution_time_ms": self.execution_time_ms,
            "from_cache": self.from_cache,
        }


# =============================================================================
# Inference Types
# =============================================================================

@dataclass
class InferenceRule:
    """
    A rule for inferring new relationships.

    Pattern format: (a:Person)-[:MANAGES]->(b:Person)-[:WORKS_ON]->(c:Project)
    Inference: (a)-[:RESPONSIBLE_FOR]->(c)
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""

    # Pattern to match (simplified Cypher-like syntax)
    pattern: str = ""

    # What to infer
    infer_relation: RelationType = RelationType.RELATED_TO
    infer_custom_type: Optional[str] = None
    infer_properties: Dict[str, Any] = field(default_factory=dict)

    # Conditions (Python expressions evaluated with matched entities)
    conditions: List[str] = field(default_factory=list)

    # Confidence propagation
    base_confidence: float = 0.8
    confidence_decay: float = 0.1  # Per hop in pattern

    # Rule metadata
    priority: int = 0  # Higher priority rules evaluated first
    enabled: bool = True
    bidirectional_output: bool = False

    # Statistics
    times_matched: int = 0
    times_inferred: int = 0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "pattern": self.pattern,
            "infer_relation": self.infer_relation.value,
            "base_confidence": self.base_confidence,
            "enabled": self.enabled,
        }


@dataclass
class InferenceResult:
    """Result from inference engine."""
    rule_id: str = ""
    rule_name: str = ""

    # Inferred relationships
    inferred_relationships: List[Relationship] = field(default_factory=list)

    # Matched patterns
    matched_patterns: int = 0

    # Execution info
    execution_time_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "rule_id": self.rule_id,
            "rule_name": self.rule_name,
            "inferred_count": len(self.inferred_relationships),
            "matched_patterns": self.matched_patterns,
            "execution_time_ms": self.execution_time_ms,
        }


# =============================================================================
# Graph Statistics
# =============================================================================

@dataclass
class GraphStatistics:
    """Statistics about the knowledge graph."""
    total_entities: int = 0
    total_relationships: int = 0

    entities_by_type: Dict[str, int] = field(default_factory=dict)
    relationships_by_type: Dict[str, int] = field(default_factory=dict)

    avg_degree: float = 0.0
    max_degree: int = 0
    density: float = 0.0

    # Connected components
    num_components: int = 0
    largest_component_size: int = 0

    # Clustering
    avg_clustering_coefficient: float = 0.0

    # Temporal
    oldest_entity: Optional[datetime] = None
    newest_entity: Optional[datetime] = None

    computed_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "total_entities": self.total_entities,
            "total_relationships": self.total_relationships,
            "entities_by_type": self.entities_by_type,
            "relationships_by_type": self.relationships_by_type,
            "avg_degree": self.avg_degree,
            "max_degree": self.max_degree,
            "density": self.density,
            "num_components": self.num_components,
            "avg_clustering_coefficient": self.avg_clustering_coefficient,
            "computed_at": self.computed_at.isoformat(),
        }
