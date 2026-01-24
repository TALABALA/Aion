"""
Analogical Reasoning System

Implements structure-mapping based analogical reasoning for
knowledge transfer and creative problem solving.

Based on Gentner's Structure-Mapping Theory.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional, Callable, Awaitable

import structlog

logger = structlog.get_logger()


class MappingType(Enum):
    """Types of analogical mappings."""

    LITERAL = "literal"  # Direct similarity
    RELATIONAL = "relational"  # Relational similarity
    STRUCTURAL = "structural"  # Deep structural similarity
    SUPERFICIAL = "superficial"  # Surface similarity only


@dataclass
class Entity:
    """An entity in a domain."""

    id: str
    name: str
    attributes: dict[str, Any] = field(default_factory=dict)
    entity_type: str = "object"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "attributes": self.attributes,
            "entity_type": self.entity_type,
        }


@dataclass
class Relation:
    """A relation between entities."""

    id: str
    name: str
    arguments: list[str]  # Entity IDs
    relation_type: str = "binary"  # binary, ternary, etc.
    higher_order: bool = False  # Whether this relates relations
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "arguments": self.arguments,
            "relation_type": self.relation_type,
            "higher_order": self.higher_order,
            "attributes": self.attributes,
        }


@dataclass
class Domain:
    """A domain/situation for analogy."""

    id: str
    name: str
    description: str
    entities: dict[str, Entity] = field(default_factory=dict)
    relations: dict[str, Relation] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_entity(
        self,
        name: str,
        attributes: Optional[dict[str, Any]] = None,
        entity_type: str = "object",
    ) -> Entity:
        """Add an entity to the domain."""
        entity_id = f"{self.id}-e{len(self.entities)}"
        entity = Entity(
            id=entity_id,
            name=name,
            attributes=attributes or {},
            entity_type=entity_type,
        )
        self.entities[entity_id] = entity
        return entity

    def add_relation(
        self,
        name: str,
        arguments: list[str | Entity],
        higher_order: bool = False,
        attributes: Optional[dict[str, Any]] = None,
    ) -> Relation:
        """Add a relation to the domain."""
        # Convert entities to IDs
        arg_ids = [
            arg.id if isinstance(arg, Entity) else arg
            for arg in arguments
        ]

        relation_id = f"{self.id}-r{len(self.relations)}"
        relation = Relation(
            id=relation_id,
            name=name,
            arguments=arg_ids,
            relation_type="binary" if len(arg_ids) == 2 else f"n-ary-{len(arg_ids)}",
            higher_order=higher_order,
            attributes=attributes or {},
        )
        self.relations[relation_id] = relation
        return relation

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "entities": {eid: e.to_dict() for eid, e in self.entities.items()},
            "relations": {rid: r.to_dict() for rid, r in self.relations.items()},
            "metadata": self.metadata,
        }


@dataclass
class EntityMapping:
    """Mapping between entities in two domains."""

    source_id: str
    target_id: str
    confidence: float = 0.5
    mapping_type: MappingType = MappingType.STRUCTURAL

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "confidence": self.confidence,
            "mapping_type": self.mapping_type.value,
        }


@dataclass
class RelationMapping:
    """Mapping between relations in two domains."""

    source_id: str
    target_id: str
    confidence: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "confidence": self.confidence,
        }


@dataclass
class StructuralMapping:
    """A complete structural mapping between two domains."""

    id: str
    source_domain: str
    target_domain: str
    entity_mappings: list[EntityMapping] = field(default_factory=list)
    relation_mappings: list[RelationMapping] = field(default_factory=list)
    structural_score: float = 0.0  # How systematic the mapping is
    similarity_score: float = 0.0  # Overall similarity
    inferences: list[str] = field(default_factory=list)  # Candidate inferences
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "source_domain": self.source_domain,
            "target_domain": self.target_domain,
            "entity_mappings": [m.to_dict() for m in self.entity_mappings],
            "relation_mappings": [m.to_dict() for m in self.relation_mappings],
            "structural_score": self.structural_score,
            "similarity_score": self.similarity_score,
            "inferences": self.inferences,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class Analogy:
    """A complete analogy between source and target."""

    id: str
    source: Domain
    target: Domain
    mapping: StructuralMapping
    explanation: str = ""
    usefulness_score: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "source": self.source.to_dict(),
            "target": self.target.to_dict(),
            "mapping": self.mapping.to_dict(),
            "explanation": self.explanation,
            "usefulness_score": self.usefulness_score,
            "created_at": self.created_at.isoformat(),
        }


# Type for LLM generation function
GenerateFn = Callable[[str], Awaitable[str]]


class AnalogicalReasoner:
    """
    Analogical reasoning system.

    Features:
    - Structure mapping between domains
    - Candidate inference generation
    - Analogy retrieval from memory
    - Systematic preference (Gentner's SMT)
    - Far and near transfer
    """

    def __init__(
        self,
        generate_fn: Optional[GenerateFn] = None,
        systematicity_weight: float = 0.7,  # How much to prefer systematic mappings
    ):
        self.generate_fn = generate_fn
        self.systematicity_weight = systematicity_weight

        # Storage
        self._domains: dict[str, Domain] = {}
        self._analogies: list[Analogy] = []
        self._mapping_counter = 0
        self._analogy_counter = 0

    async def create_domain(
        self,
        name: str,
        description: str,
        structure: Optional[dict[str, Any]] = None,
    ) -> Domain:
        """Create a domain representation."""
        domain_id = f"domain-{len(self._domains)}"
        domain = Domain(
            id=domain_id,
            name=name,
            description=description,
        )

        # Parse structure if provided
        if structure:
            for entity_data in structure.get("entities", []):
                domain.add_entity(
                    name=entity_data.get("name", "unnamed"),
                    attributes=entity_data.get("attributes", {}),
                    entity_type=entity_data.get("type", "object"),
                )

            for relation_data in structure.get("relations", []):
                domain.add_relation(
                    name=relation_data.get("name", "related"),
                    arguments=relation_data.get("arguments", []),
                    higher_order=relation_data.get("higher_order", False),
                )

        self._domains[domain_id] = domain
        return domain

    async def find_mapping(
        self,
        source: Domain,
        target: Domain,
    ) -> StructuralMapping:
        """
        Find the best structural mapping between two domains.

        Uses structure-mapping algorithm.
        """
        self._mapping_counter += 1
        mapping = StructuralMapping(
            id=f"mapping-{self._mapping_counter}",
            source_domain=source.id,
            target_domain=target.id,
        )

        # Phase 1: Find matching relations
        relation_matches = self._find_relation_matches(source, target)

        # Phase 2: Build entity correspondences from relation matches
        entity_matches = self._build_entity_correspondences(
            source, target, relation_matches
        )

        # Phase 3: Evaluate mappings and select best
        mapping.entity_mappings = entity_matches
        mapping.relation_mappings = relation_matches

        # Calculate structural score (systematicity)
        mapping.structural_score = self._calculate_structural_score(
            source, target, mapping
        )

        # Calculate similarity score
        mapping.similarity_score = self._calculate_similarity_score(
            source, target, mapping
        )

        # Generate candidate inferences
        mapping.inferences = await self._generate_inferences(source, target, mapping)

        return mapping

    async def make_analogy(
        self,
        source: Domain,
        target: Domain,
        generate_explanation: bool = True,
    ) -> Analogy:
        """
        Create a complete analogy between source and target.
        """
        self._analogy_counter += 1

        # Find structural mapping
        mapping = await self.find_mapping(source, target)

        # Generate explanation
        explanation = ""
        if generate_explanation and self.generate_fn:
            explanation = await self._generate_explanation(source, target, mapping)

        analogy = Analogy(
            id=f"analogy-{self._analogy_counter}",
            source=source,
            target=target,
            mapping=mapping,
            explanation=explanation,
            usefulness_score=mapping.structural_score * 0.7 + mapping.similarity_score * 0.3,
        )

        self._analogies.append(analogy)

        return analogy

    async def find_analogies(
        self,
        target: Domain,
        candidate_sources: Optional[list[Domain]] = None,
        top_k: int = 3,
    ) -> list[Analogy]:
        """
        Find the best analogies for a target domain.
        """
        sources = candidate_sources or list(self._domains.values())

        analogies = []
        for source in sources:
            if source.id == target.id:
                continue

            analogy = await self.make_analogy(source, target, generate_explanation=False)
            analogies.append(analogy)

        # Sort by usefulness
        analogies.sort(key=lambda a: a.usefulness_score, reverse=True)

        return analogies[:top_k]

    async def transfer_knowledge(
        self,
        source: Domain,
        target: Domain,
        knowledge: str,
    ) -> list[str]:
        """
        Transfer knowledge from source to target domain.

        Returns list of transferred/adapted knowledge statements.
        """
        # Find mapping
        mapping = await self.find_mapping(source, target)

        # Build substitution map
        substitutions = {}
        for em in mapping.entity_mappings:
            source_entity = source.entities.get(em.source_id)
            target_entity = target.entities.get(em.target_id)
            if source_entity and target_entity:
                substitutions[source_entity.name] = target_entity.name

        for rm in mapping.relation_mappings:
            source_rel = source.relations.get(rm.source_id)
            target_rel = target.relations.get(rm.target_id)
            if source_rel and target_rel:
                substitutions[source_rel.name] = target_rel.name

        # Apply substitutions to knowledge
        transferred = knowledge
        for source_term, target_term in substitutions.items():
            transferred = transferred.replace(source_term, target_term)

        # If we have an LLM, refine the transfer
        if self.generate_fn:
            prompt = f"""Adapt this knowledge from one domain to another.

Source domain: {source.name} - {source.description}
Target domain: {target.name} - {target.description}

Original knowledge: {knowledge}

Rough transfer: {transferred}

Provide a refined, domain-appropriate version:"""

            refined = await self.generate_fn(prompt)
            return [transferred, refined]

        return [transferred]

    def _find_relation_matches(
        self,
        source: Domain,
        target: Domain,
    ) -> list[RelationMapping]:
        """Find matching relations between domains."""
        matches = []

        for source_rel in source.relations.values():
            for target_rel in target.relations.values():
                # Same relation name
                if source_rel.name == target_rel.name:
                    matches.append(RelationMapping(
                        source_id=source_rel.id,
                        target_id=target_rel.id,
                        confidence=0.9,
                    ))
                # Same arity
                elif len(source_rel.arguments) == len(target_rel.arguments):
                    # Check for similar relation types
                    if source_rel.higher_order == target_rel.higher_order:
                        matches.append(RelationMapping(
                            source_id=source_rel.id,
                            target_id=target_rel.id,
                            confidence=0.5,
                        ))

        return matches

    def _build_entity_correspondences(
        self,
        source: Domain,
        target: Domain,
        relation_mappings: list[RelationMapping],
    ) -> list[EntityMapping]:
        """Build entity correspondences from relation mappings."""
        entity_votes: dict[tuple[str, str], float] = {}

        for rm in relation_mappings:
            source_rel = source.relations.get(rm.source_id)
            target_rel = target.relations.get(rm.target_id)

            if not source_rel or not target_rel:
                continue

            # Entities in same argument positions correspond
            for i, (s_arg, t_arg) in enumerate(zip(source_rel.arguments, target_rel.arguments)):
                key = (s_arg, t_arg)
                entity_votes[key] = entity_votes.get(key, 0) + rm.confidence

        # Create mappings from votes
        mappings = []
        used_targets = set()

        # Sort by vote count and greedily assign
        sorted_votes = sorted(entity_votes.items(), key=lambda x: x[1], reverse=True)

        for (source_id, target_id), votes in sorted_votes:
            if target_id in used_targets:
                continue

            source_entity = source.entities.get(source_id)
            target_entity = target.entities.get(target_id)

            if source_entity and target_entity:
                # Check attribute similarity
                attr_match = self._attribute_similarity(
                    source_entity.attributes,
                    target_entity.attributes,
                )

                mappings.append(EntityMapping(
                    source_id=source_id,
                    target_id=target_id,
                    confidence=min(1.0, votes * 0.3 + attr_match * 0.3 + 0.2),
                    mapping_type=MappingType.STRUCTURAL if votes > 1 else MappingType.SUPERFICIAL,
                ))
                used_targets.add(target_id)

        return mappings

    def _attribute_similarity(
        self,
        attrs1: dict[str, Any],
        attrs2: dict[str, Any],
    ) -> float:
        """Calculate attribute similarity between entities."""
        if not attrs1 and not attrs2:
            return 0.5

        all_keys = set(attrs1.keys()) | set(attrs2.keys())
        if not all_keys:
            return 0.5

        matching = 0
        for key in all_keys:
            if key in attrs1 and key in attrs2:
                if attrs1[key] == attrs2[key]:
                    matching += 1
                elif type(attrs1[key]) == type(attrs2[key]):
                    matching += 0.5

        return matching / len(all_keys)

    def _calculate_structural_score(
        self,
        source: Domain,
        target: Domain,
        mapping: StructuralMapping,
    ) -> float:
        """
        Calculate structural/systematicity score.

        Rewards mappings that preserve relational structure.
        """
        if not mapping.relation_mappings:
            return 0.0

        # Count higher-order relations mapped
        higher_order_mapped = sum(
            1 for rm in mapping.relation_mappings
            if source.relations.get(rm.source_id, Relation("", "", [])).higher_order
        )

        # Structural score based on relation preservation
        relation_score = len(mapping.relation_mappings) / max(1, len(source.relations))

        # Bonus for higher-order mappings (systematicity)
        ho_bonus = higher_order_mapped * 0.2

        # Consistency check - entities should map consistently across relations
        consistency = self._check_consistency(source, target, mapping)

        return min(1.0, relation_score * 0.5 + ho_bonus + consistency * 0.3)

    def _calculate_similarity_score(
        self,
        source: Domain,
        target: Domain,
        mapping: StructuralMapping,
    ) -> float:
        """Calculate overall similarity score."""
        if not mapping.entity_mappings:
            return 0.0

        # Average confidence of mappings
        entity_conf = sum(em.confidence for em in mapping.entity_mappings) / len(mapping.entity_mappings)

        rel_conf = 0.0
        if mapping.relation_mappings:
            rel_conf = sum(rm.confidence for rm in mapping.relation_mappings) / len(mapping.relation_mappings)

        return entity_conf * 0.5 + rel_conf * 0.5

    def _check_consistency(
        self,
        source: Domain,
        target: Domain,
        mapping: StructuralMapping,
    ) -> float:
        """Check if entity mappings are consistent across relations."""
        # Build entity mapping dict
        entity_map = {em.source_id: em.target_id for em in mapping.entity_mappings}

        consistent = 0
        total = 0

        for rm in mapping.relation_mappings:
            source_rel = source.relations.get(rm.source_id)
            target_rel = target.relations.get(rm.target_id)

            if not source_rel or not target_rel:
                continue

            for i, s_arg in enumerate(source_rel.arguments):
                if i < len(target_rel.arguments):
                    total += 1
                    expected_target = entity_map.get(s_arg)
                    if expected_target == target_rel.arguments[i]:
                        consistent += 1

        return consistent / max(1, total)

    async def _generate_inferences(
        self,
        source: Domain,
        target: Domain,
        mapping: StructuralMapping,
    ) -> list[str]:
        """Generate candidate inferences from the analogy."""
        inferences = []

        # Find unmapped source relations that could transfer
        mapped_source_rels = {rm.source_id for rm in mapping.relation_mappings}
        entity_map = {em.source_id: em.target_id for em in mapping.entity_mappings}

        for source_rel in source.relations.values():
            if source_rel.id in mapped_source_rels:
                continue

            # Check if all arguments are mapped
            can_transfer = all(
                arg in entity_map for arg in source_rel.arguments
            )

            if can_transfer:
                # Build inference
                mapped_args = [
                    target.entities.get(entity_map[arg], Entity("", "unknown")).name
                    for arg in source_rel.arguments
                ]

                inference = f"Based on analogy: {mapped_args[0]} {source_rel.name} {mapped_args[1] if len(mapped_args) > 1 else ''}"
                inferences.append(inference)

        return inferences

    async def _generate_explanation(
        self,
        source: Domain,
        target: Domain,
        mapping: StructuralMapping,
    ) -> str:
        """Generate a natural language explanation of the analogy."""
        if not self.generate_fn:
            return ""

        # Build mapping description
        entity_pairs = []
        for em in mapping.entity_mappings:
            s_ent = source.entities.get(em.source_id)
            t_ent = target.entities.get(em.target_id)
            if s_ent and t_ent:
                entity_pairs.append(f"{s_ent.name} → {t_ent.name}")

        relation_pairs = []
        for rm in mapping.relation_mappings:
            s_rel = source.relations.get(rm.source_id)
            t_rel = target.relations.get(rm.target_id)
            if s_rel and t_rel:
                relation_pairs.append(f"{s_rel.name} → {t_rel.name}")

        prompt = f"""Explain this analogy clearly:

Source domain: {source.name} - {source.description}
Target domain: {target.name} - {target.description}

Correspondences:
- Entities: {', '.join(entity_pairs[:5])}
- Relations: {', '.join(relation_pairs[:5])}

Candidate inferences: {', '.join(mapping.inferences[:3])}

Provide a clear, insightful explanation of this analogy:"""

        return await self.generate_fn(prompt)

    def get_stats(self) -> dict[str, Any]:
        """Get analogical reasoning statistics."""
        return {
            "domains_stored": len(self._domains),
            "analogies_made": len(self._analogies),
            "mappings_created": self._mapping_counter,
        }
