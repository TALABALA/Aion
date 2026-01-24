"""
Semantic Memory System

Knowledge graph-based memory for storing concepts, relations,
and factual knowledge with reasoning capabilities.
"""

import asyncio
import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional
import heapq

import structlog

logger = structlog.get_logger()


class RelationType(Enum):
    """Standard relation types in the knowledge graph."""

    IS_A = "is_a"  # Taxonomic
    HAS_A = "has_a"  # Composition
    PART_OF = "part_of"  # Mereology
    CAUSES = "causes"  # Causal
    REQUIRES = "requires"  # Dependency
    SIMILAR_TO = "similar_to"  # Similarity
    OPPOSITE_OF = "opposite_of"  # Antonymy
    RELATED_TO = "related_to"  # Generic
    LOCATED_IN = "located_in"  # Spatial
    OCCURS_BEFORE = "occurs_before"  # Temporal
    OCCURS_AFTER = "occurs_after"  # Temporal
    INSTANCE_OF = "instance_of"  # Type-instance
    PROPERTY_OF = "property_of"  # Attribution
    USED_FOR = "used_for"  # Functional
    CREATED_BY = "created_by"  # Provenance


@dataclass
class Concept:
    """A concept node in the knowledge graph."""

    id: str
    name: str
    description: str = ""
    concept_type: str = "entity"  # entity, action, attribute, event, etc.
    properties: dict[str, Any] = field(default_factory=dict)
    embeddings: Optional[list[float]] = None
    confidence: float = 1.0
    source: str = "learned"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    access_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "concept_type": self.concept_type,
            "properties": self.properties,
            "embeddings": self.embeddings,
            "confidence": self.confidence,
            "source": self.source,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "access_count": self.access_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Concept":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            description=data.get("description", ""),
            concept_type=data.get("concept_type", "entity"),
            properties=data.get("properties", {}),
            embeddings=data.get("embeddings"),
            confidence=data.get("confidence", 1.0),
            source=data.get("source", "learned"),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else datetime.now(),
            access_count=data.get("access_count", 0),
        )


@dataclass
class Relation:
    """A relation edge in the knowledge graph."""

    id: str
    source_id: str
    target_id: str
    relation_type: RelationType
    properties: dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    confidence: float = 1.0
    bidirectional: bool = False
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation_type": self.relation_type.value,
            "properties": self.properties,
            "weight": self.weight,
            "confidence": self.confidence,
            "bidirectional": self.bidirectional,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Relation":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            source_id=data["source_id"],
            target_id=data["target_id"],
            relation_type=RelationType(data["relation_type"]),
            properties=data.get("properties", {}),
            weight=data.get("weight", 1.0),
            confidence=data.get("confidence", 1.0),
            bidirectional=data.get("bidirectional", False),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(),
        )


@dataclass
class KnowledgeTriple:
    """A subject-predicate-object triple."""

    subject: str
    predicate: str
    object: str
    confidence: float = 1.0
    source: str = ""
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "confidence": self.confidence,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "KnowledgeTriple":
        """Create from dictionary."""
        return cls(
            subject=data["subject"],
            predicate=data["predicate"],
            object=data["object"],
            confidence=data.get("confidence", 1.0),
            source=data.get("source", ""),
            timestamp=datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else datetime.now(),
        )


@dataclass
class InferenceResult:
    """Result of a knowledge inference."""

    query: str
    answer: Any
    confidence: float
    reasoning_chain: list[str]
    supporting_facts: list[KnowledgeTriple]
    inference_type: str  # deductive, inductive, abductive


class SemanticMemory:
    """
    Semantic memory system using knowledge graphs.

    Features:
    - Concept and relation storage
    - Graph-based reasoning
    - Taxonomic inference (IS_A hierarchies)
    - Path finding between concepts
    - Subgraph extraction
    - Knowledge consolidation
    - Contradiction detection
    """

    def __init__(
        self,
        storage_path: Optional[Path] = None,
        max_concepts: int = 100000,
        max_relations: int = 500000,
    ):
        self.storage_path = storage_path
        self.max_concepts = max_concepts
        self.max_relations = max_relations

        # Graph storage
        self._concepts: dict[str, Concept] = {}
        self._relations: dict[str, Relation] = {}

        # Adjacency lists for efficient traversal
        self._outgoing: dict[str, list[str]] = defaultdict(list)  # concept_id -> relation_ids
        self._incoming: dict[str, list[str]] = defaultdict(list)  # concept_id -> relation_ids

        # Indexes
        self._name_index: dict[str, str] = {}  # name -> concept_id
        self._type_index: dict[str, set[str]] = defaultdict(set)  # concept_type -> concept_ids
        self._relation_type_index: dict[RelationType, set[str]] = defaultdict(set)

        self._initialized = False
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize semantic memory."""
        if self._initialized:
            return

        if self.storage_path and self.storage_path.exists():
            await self._load_from_disk()

        self._initialized = True
        logger.info("semantic_memory_initialized", concepts=len(self._concepts), relations=len(self._relations))

    async def shutdown(self) -> None:
        """Shutdown and persist."""
        if self.storage_path:
            await self._save_to_disk()

        self._initialized = False
        logger.info("semantic_memory_shutdown")

    async def add_concept(
        self,
        name: str,
        description: str = "",
        concept_type: str = "entity",
        properties: Optional[dict[str, Any]] = None,
        concept_id: Optional[str] = None,
        confidence: float = 1.0,
        source: str = "learned",
    ) -> Concept:
        """Add a concept to the knowledge graph."""
        async with self._lock:
            # Check if already exists
            if name.lower() in self._name_index:
                existing_id = self._name_index[name.lower()]
                existing = self._concepts[existing_id]
                # Update existing concept
                if description:
                    existing.description = description
                if properties:
                    existing.properties.update(properties)
                existing.updated_at = datetime.now()
                existing.confidence = max(existing.confidence, confidence)
                return existing

            # Create new concept
            if concept_id is None:
                concept_id = f"c-{len(self._concepts)}-{datetime.now().strftime('%H%M%S%f')}"

            concept = Concept(
                id=concept_id,
                name=name,
                description=description,
                concept_type=concept_type,
                properties=properties or {},
                confidence=confidence,
                source=source,
            )

            # Check capacity
            if len(self._concepts) >= self.max_concepts:
                await self._prune_concepts()

            self._concepts[concept_id] = concept
            self._name_index[name.lower()] = concept_id
            self._type_index[concept_type].add(concept_id)

            logger.debug("concept_added", concept_id=concept_id, name=name)

            return concept

    async def add_relation(
        self,
        source: str | Concept,
        relation_type: RelationType | str,
        target: str | Concept,
        properties: Optional[dict[str, Any]] = None,
        weight: float = 1.0,
        confidence: float = 1.0,
        bidirectional: bool = False,
    ) -> Relation:
        """Add a relation between concepts."""
        async with self._lock:
            # Get concept IDs
            source_id = source.id if isinstance(source, Concept) else self._resolve_concept_id(source)
            target_id = target.id if isinstance(target, Concept) else self._resolve_concept_id(target)

            if not source_id or not target_id:
                raise ValueError("Source or target concept not found")

            if isinstance(relation_type, str):
                relation_type = RelationType(relation_type)

            # Check for existing relation
            for rel_id in self._outgoing.get(source_id, []):
                rel = self._relations.get(rel_id)
                if rel and rel.target_id == target_id and rel.relation_type == relation_type:
                    # Update existing relation
                    rel.weight = max(rel.weight, weight)
                    rel.confidence = max(rel.confidence, confidence)
                    if properties:
                        rel.properties.update(properties)
                    return rel

            # Create new relation
            relation_id = f"r-{len(self._relations)}-{datetime.now().strftime('%H%M%S%f')}"

            relation = Relation(
                id=relation_id,
                source_id=source_id,
                target_id=target_id,
                relation_type=relation_type,
                properties=properties or {},
                weight=weight,
                confidence=confidence,
                bidirectional=bidirectional,
            )

            # Check capacity
            if len(self._relations) >= self.max_relations:
                await self._prune_relations()

            self._relations[relation_id] = relation
            self._outgoing[source_id].append(relation_id)
            self._incoming[target_id].append(relation_id)
            self._relation_type_index[relation_type].add(relation_id)

            logger.debug(
                "relation_added",
                relation_id=relation_id,
                source=source_id,
                type=relation_type.value,
                target=target_id,
            )

            return relation

    async def add_triple(self, triple: KnowledgeTriple) -> tuple[Concept, Relation, Concept]:
        """Add a knowledge triple (subject-predicate-object)."""
        # Ensure concepts exist
        subject = await self.add_concept(triple.subject, source=triple.source, confidence=triple.confidence)
        obj = await self.add_concept(triple.object, source=triple.source, confidence=triple.confidence)

        # Parse predicate to relation type
        relation_type = self._parse_predicate(triple.predicate)

        relation = await self.add_relation(
            source=subject,
            relation_type=relation_type,
            target=obj,
            confidence=triple.confidence,
        )

        return subject, relation, obj

    def _parse_predicate(self, predicate: str) -> RelationType:
        """Parse a natural language predicate to a relation type."""
        predicate_lower = predicate.lower()

        mappings = {
            "is a": RelationType.IS_A,
            "is an": RelationType.IS_A,
            "has": RelationType.HAS_A,
            "has a": RelationType.HAS_A,
            "part of": RelationType.PART_OF,
            "causes": RelationType.CAUSES,
            "requires": RelationType.REQUIRES,
            "needs": RelationType.REQUIRES,
            "similar to": RelationType.SIMILAR_TO,
            "like": RelationType.SIMILAR_TO,
            "opposite of": RelationType.OPPOSITE_OF,
            "related to": RelationType.RELATED_TO,
            "located in": RelationType.LOCATED_IN,
            "in": RelationType.LOCATED_IN,
            "before": RelationType.OCCURS_BEFORE,
            "after": RelationType.OCCURS_AFTER,
            "instance of": RelationType.INSTANCE_OF,
            "type of": RelationType.IS_A,
            "property of": RelationType.PROPERTY_OF,
            "used for": RelationType.USED_FOR,
            "created by": RelationType.CREATED_BY,
            "made by": RelationType.CREATED_BY,
        }

        for phrase, rel_type in mappings.items():
            if phrase in predicate_lower:
                return rel_type

        return RelationType.RELATED_TO

    def _resolve_concept_id(self, name_or_id: str) -> Optional[str]:
        """Resolve a concept name or ID to an ID."""
        if name_or_id in self._concepts:
            return name_or_id
        return self._name_index.get(name_or_id.lower())

    async def get_concept(self, name_or_id: str) -> Optional[Concept]:
        """Get a concept by name or ID."""
        concept_id = self._resolve_concept_id(name_or_id)
        if concept_id:
            concept = self._concepts.get(concept_id)
            if concept:
                concept.access_count += 1
            return concept
        return None

    async def get_relations(
        self,
        concept: str | Concept,
        relation_type: Optional[RelationType] = None,
        direction: str = "both",  # "outgoing", "incoming", "both"
    ) -> list[Relation]:
        """Get relations for a concept."""
        concept_id = concept.id if isinstance(concept, Concept) else self._resolve_concept_id(concept)
        if not concept_id:
            return []

        relations = []

        if direction in ("outgoing", "both"):
            for rel_id in self._outgoing.get(concept_id, []):
                rel = self._relations.get(rel_id)
                if rel and (relation_type is None or rel.relation_type == relation_type):
                    relations.append(rel)

        if direction in ("incoming", "both"):
            for rel_id in self._incoming.get(concept_id, []):
                rel = self._relations.get(rel_id)
                if rel and (relation_type is None or rel.relation_type == relation_type):
                    relations.append(rel)

        return relations

    async def find_path(
        self,
        source: str | Concept,
        target: str | Concept,
        max_depth: int = 5,
        relation_types: Optional[list[RelationType]] = None,
    ) -> Optional[list[tuple[Concept, Relation]]]:
        """Find a path between two concepts using BFS."""
        source_id = source.id if isinstance(source, Concept) else self._resolve_concept_id(source)
        target_id = target.id if isinstance(target, Concept) else self._resolve_concept_id(target)

        if not source_id or not target_id:
            return None

        if source_id == target_id:
            return []

        # BFS
        queue = [(source_id, [])]
        visited = {source_id}

        while queue:
            current_id, path = queue.pop(0)

            if len(path) >= max_depth:
                continue

            for rel_id in self._outgoing.get(current_id, []):
                rel = self._relations.get(rel_id)
                if not rel:
                    continue

                if relation_types and rel.relation_type not in relation_types:
                    continue

                next_id = rel.target_id

                if next_id == target_id:
                    # Found path
                    result_path = []
                    for c_id, r in path:
                        result_path.append((self._concepts[c_id], r))
                    result_path.append((self._concepts[current_id], rel))
                    return result_path

                if next_id not in visited:
                    visited.add(next_id)
                    new_path = path + [(current_id, rel)]
                    queue.append((next_id, new_path))

        return None

    async def get_ancestors(
        self,
        concept: str | Concept,
        relation_type: RelationType = RelationType.IS_A,
        max_depth: int = 10,
    ) -> list[Concept]:
        """Get all ancestors of a concept (taxonomic hierarchy)."""
        concept_id = concept.id if isinstance(concept, Concept) else self._resolve_concept_id(concept)
        if not concept_id:
            return []

        ancestors = []
        visited = {concept_id}
        queue = [concept_id]
        depth = 0

        while queue and depth < max_depth:
            next_queue = []
            for current_id in queue:
                for rel_id in self._outgoing.get(current_id, []):
                    rel = self._relations.get(rel_id)
                    if rel and rel.relation_type == relation_type:
                        if rel.target_id not in visited:
                            visited.add(rel.target_id)
                            ancestor = self._concepts.get(rel.target_id)
                            if ancestor:
                                ancestors.append(ancestor)
                                next_queue.append(rel.target_id)
            queue = next_queue
            depth += 1

        return ancestors

    async def get_descendants(
        self,
        concept: str | Concept,
        relation_type: RelationType = RelationType.IS_A,
        max_depth: int = 10,
    ) -> list[Concept]:
        """Get all descendants of a concept."""
        concept_id = concept.id if isinstance(concept, Concept) else self._resolve_concept_id(concept)
        if not concept_id:
            return []

        descendants = []
        visited = {concept_id}
        queue = [concept_id]
        depth = 0

        while queue and depth < max_depth:
            next_queue = []
            for current_id in queue:
                for rel_id in self._incoming.get(current_id, []):
                    rel = self._relations.get(rel_id)
                    if rel and rel.relation_type == relation_type:
                        if rel.source_id not in visited:
                            visited.add(rel.source_id)
                            descendant = self._concepts.get(rel.source_id)
                            if descendant:
                                descendants.append(descendant)
                                next_queue.append(rel.source_id)
            queue = next_queue
            depth += 1

        return descendants

    async def query(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str | RelationType] = None,
        object: Optional[str] = None,
        limit: int = 100,
    ) -> list[tuple[Concept, Relation, Concept]]:
        """Query the knowledge graph with pattern matching."""
        results = []

        # Convert predicate to RelationType if string
        rel_type = None
        if predicate:
            if isinstance(predicate, str):
                rel_type = self._parse_predicate(predicate)
            else:
                rel_type = predicate

        # Resolve concept IDs
        subject_id = self._resolve_concept_id(subject) if subject else None
        object_id = self._resolve_concept_id(object) if object else None

        # Find matching relations
        if subject_id:
            # Start from subject
            for rel_id in self._outgoing.get(subject_id, []):
                rel = self._relations.get(rel_id)
                if not rel:
                    continue

                if rel_type and rel.relation_type != rel_type:
                    continue

                if object_id and rel.target_id != object_id:
                    continue

                subj_concept = self._concepts.get(rel.source_id)
                obj_concept = self._concepts.get(rel.target_id)

                if subj_concept and obj_concept:
                    results.append((subj_concept, rel, obj_concept))

                if len(results) >= limit:
                    break

        elif object_id:
            # Start from object
            for rel_id in self._incoming.get(object_id, []):
                rel = self._relations.get(rel_id)
                if not rel:
                    continue

                if rel_type and rel.relation_type != rel_type:
                    continue

                subj_concept = self._concepts.get(rel.source_id)
                obj_concept = self._concepts.get(rel.target_id)

                if subj_concept and obj_concept:
                    results.append((subj_concept, rel, obj_concept))

                if len(results) >= limit:
                    break

        elif rel_type:
            # Search by relation type
            for rel_id in self._relation_type_index.get(rel_type, set()):
                rel = self._relations.get(rel_id)
                if not rel:
                    continue

                subj_concept = self._concepts.get(rel.source_id)
                obj_concept = self._concepts.get(rel.target_id)

                if subj_concept and obj_concept:
                    results.append((subj_concept, rel, obj_concept))

                if len(results) >= limit:
                    break

        return results

    async def infer(
        self,
        query: str,
        max_depth: int = 3,
    ) -> InferenceResult:
        """
        Perform inference to answer a query.

        Supports:
        - Taxonomic inference (X is_a Y, Y is_a Z => X is_a Z)
        - Transitivity
        - Property inheritance
        """
        reasoning_chain = []
        supporting_facts = []

        # Parse query (simple pattern matching)
        # e.g., "Is X a Y?" or "What is X related to?"

        query_lower = query.lower()

        if "is" in query_lower and ("a " in query_lower or "an " in query_lower):
            # IS-A query
            parts = query_lower.replace("?", "").replace("is a ", "IS_A ").replace("is an ", "IS_A ").split("IS_A ")
            if len(parts) == 2:
                subject = parts[0].strip()
                target = parts[1].strip()

                subject_concept = await self.get_concept(subject)
                target_concept = await self.get_concept(target)

                if subject_concept and target_concept:
                    # Check direct relation
                    relations = await self.get_relations(subject_concept, RelationType.IS_A, "outgoing")
                    for rel in relations:
                        if rel.target_id == target_concept.id:
                            return InferenceResult(
                                query=query,
                                answer=True,
                                confidence=rel.confidence,
                                reasoning_chain=[f"{subject} is_a {target} (direct)"],
                                supporting_facts=[
                                    KnowledgeTriple(subject=subject, predicate="is_a", object=target)
                                ],
                                inference_type="deductive",
                            )

                    # Check transitive IS-A
                    ancestors = await self.get_ancestors(subject_concept, RelationType.IS_A)
                    for ancestor in ancestors:
                        if ancestor.id == target_concept.id:
                            # Build reasoning chain
                            path = await self.find_path(
                                subject_concept, target_concept,
                                relation_types=[RelationType.IS_A],
                            )
                            if path:
                                for concept, rel in path:
                                    reasoning_chain.append(
                                        f"{concept.name} is_a {self._concepts[rel.target_id].name}"
                                    )
                                    supporting_facts.append(
                                        KnowledgeTriple(
                                            subject=concept.name,
                                            predicate="is_a",
                                            object=self._concepts[rel.target_id].name,
                                        )
                                    )

                            return InferenceResult(
                                query=query,
                                answer=True,
                                confidence=0.9,  # Reduced for transitive
                                reasoning_chain=reasoning_chain,
                                supporting_facts=supporting_facts,
                                inference_type="deductive",
                            )

                    return InferenceResult(
                        query=query,
                        answer=False,
                        confidence=0.8,
                        reasoning_chain=["No IS-A path found"],
                        supporting_facts=[],
                        inference_type="deductive",
                    )

        # Default: unknown
        return InferenceResult(
            query=query,
            answer=None,
            confidence=0.0,
            reasoning_chain=["Could not parse query"],
            supporting_facts=[],
            inference_type="abductive",
        )

    async def detect_contradictions(
        self,
        concept: str | Concept,
    ) -> list[tuple[Relation, Relation, str]]:
        """Detect contradictory relations for a concept."""
        concept_id = concept.id if isinstance(concept, Concept) else self._resolve_concept_id(concept)
        if not concept_id:
            return []

        contradictions = []
        relations = await self.get_relations(concept)

        # Check for contradictory relation pairs
        opposite_pairs = [
            (RelationType.CAUSES, RelationType.OPPOSITE_OF),
            (RelationType.REQUIRES, RelationType.OPPOSITE_OF),
        ]

        for i, rel1 in enumerate(relations):
            for rel2 in relations[i + 1:]:
                # Check if relations point to same target with contradictory types
                if rel1.target_id == rel2.target_id:
                    for type1, type2 in opposite_pairs:
                        if (rel1.relation_type == type1 and rel2.relation_type == type2) or \
                           (rel1.relation_type == type2 and rel2.relation_type == type1):
                            contradictions.append((rel1, rel2, "Contradictory relation types"))

        return contradictions

    async def get_subgraph(
        self,
        center: str | Concept,
        radius: int = 2,
    ) -> tuple[list[Concept], list[Relation]]:
        """Extract a subgraph around a concept."""
        center_id = center.id if isinstance(center, Concept) else self._resolve_concept_id(center)
        if not center_id:
            return [], []

        concepts = {}
        relations = {}

        visited = {center_id}
        queue = [(center_id, 0)]

        while queue:
            current_id, depth = queue.pop(0)

            if current_id in self._concepts:
                concepts[current_id] = self._concepts[current_id]

            if depth >= radius:
                continue

            # Outgoing
            for rel_id in self._outgoing.get(current_id, []):
                rel = self._relations.get(rel_id)
                if rel:
                    relations[rel_id] = rel
                    if rel.target_id not in visited:
                        visited.add(rel.target_id)
                        queue.append((rel.target_id, depth + 1))

            # Incoming
            for rel_id in self._incoming.get(current_id, []):
                rel = self._relations.get(rel_id)
                if rel:
                    relations[rel_id] = rel
                    if rel.source_id not in visited:
                        visited.add(rel.source_id)
                        queue.append((rel.source_id, depth + 1))

        return list(concepts.values()), list(relations.values())

    async def _prune_concepts(self) -> None:
        """Remove least accessed concepts."""
        if not self._concepts:
            return

        # Sort by access count and recency
        sorted_concepts = sorted(
            self._concepts.values(),
            key=lambda c: (c.access_count, c.updated_at),
        )

        # Remove bottom 10%
        to_remove = sorted_concepts[:max(1, len(sorted_concepts) // 10)]

        for concept in to_remove:
            await self._remove_concept(concept.id)

        logger.info("concepts_pruned", count=len(to_remove))

    async def _prune_relations(self) -> None:
        """Remove low-weight relations."""
        if not self._relations:
            return

        sorted_relations = sorted(
            self._relations.values(),
            key=lambda r: (r.weight * r.confidence, r.created_at),
        )

        to_remove = sorted_relations[:max(1, len(sorted_relations) // 10)]

        for rel in to_remove:
            await self._remove_relation(rel.id)

        logger.info("relations_pruned", count=len(to_remove))

    async def _remove_concept(self, concept_id: str) -> None:
        """Remove a concept and its relations."""
        concept = self._concepts.pop(concept_id, None)
        if not concept:
            return

        # Remove from indexes
        self._name_index.pop(concept.name.lower(), None)
        self._type_index[concept.concept_type].discard(concept_id)

        # Remove relations
        for rel_id in list(self._outgoing.get(concept_id, [])):
            await self._remove_relation(rel_id)

        for rel_id in list(self._incoming.get(concept_id, [])):
            await self._remove_relation(rel_id)

        self._outgoing.pop(concept_id, None)
        self._incoming.pop(concept_id, None)

    async def _remove_relation(self, relation_id: str) -> None:
        """Remove a relation."""
        rel = self._relations.pop(relation_id, None)
        if not rel:
            return

        if relation_id in self._outgoing.get(rel.source_id, []):
            self._outgoing[rel.source_id].remove(relation_id)

        if relation_id in self._incoming.get(rel.target_id, []):
            self._incoming[rel.target_id].remove(relation_id)

        self._relation_type_index[rel.relation_type].discard(relation_id)

    async def _save_to_disk(self) -> None:
        """Save to disk."""
        if not self.storage_path:
            return

        self.storage_path.mkdir(parents=True, exist_ok=True)

        concepts_data = {cid: c.to_dict() for cid, c in self._concepts.items()}
        relations_data = {rid: r.to_dict() for rid, r in self._relations.items()}

        with open(self.storage_path / "concepts.json", "w") as f:
            json.dump(concepts_data, f)

        with open(self.storage_path / "relations.json", "w") as f:
            json.dump(relations_data, f)

        logger.info(
            "semantic_memory_saved",
            path=str(self.storage_path),
            concepts=len(self._concepts),
            relations=len(self._relations),
        )

    async def _load_from_disk(self) -> None:
        """Load from disk."""
        if not self.storage_path:
            return

        concepts_path = self.storage_path / "concepts.json"
        relations_path = self.storage_path / "relations.json"

        if concepts_path.exists():
            with open(concepts_path) as f:
                concepts_data = json.load(f)

            for cid, data in concepts_data.items():
                concept = Concept.from_dict(data)
                self._concepts[cid] = concept
                self._name_index[concept.name.lower()] = cid
                self._type_index[concept.concept_type].add(cid)

        if relations_path.exists():
            with open(relations_path) as f:
                relations_data = json.load(f)

            for rid, data in relations_data.items():
                rel = Relation.from_dict(data)
                self._relations[rid] = rel
                self._outgoing[rel.source_id].append(rid)
                self._incoming[rel.target_id].append(rid)
                self._relation_type_index[rel.relation_type].add(rid)

        logger.info(
            "semantic_memory_loaded",
            path=str(self.storage_path),
            concepts=len(self._concepts),
            relations=len(self._relations),
        )

    def get_stats(self) -> dict[str, Any]:
        """Get memory statistics."""
        return {
            "total_concepts": len(self._concepts),
            "total_relations": len(self._relations),
            "max_concepts": self.max_concepts,
            "max_relations": self.max_relations,
            "concept_types": {t: len(ids) for t, ids in self._type_index.items()},
            "relation_types": {t.value: len(ids) for t, ids in self._relation_type_index.items()},
        }
