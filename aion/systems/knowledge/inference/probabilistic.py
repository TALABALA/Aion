"""
AION Probabilistic Reasoning

Uncertainty propagation and probabilistic inference.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import structlog

from aion.systems.knowledge.types import (
    Entity,
    Relationship,
    RelationType,
)
from aion.systems.knowledge.store.base import GraphStore

logger = structlog.get_logger(__name__)


@dataclass
class BeliefState:
    """Belief state for an entity or relationship."""
    confidence: float = 1.0
    prior: float = 0.5
    evidence_count: int = 0
    supporting_evidence: List[str] = field(default_factory=list)
    contradicting_evidence: List[str] = field(default_factory=list)


class ProbabilisticReasoner:
    """
    Probabilistic reasoning for knowledge graphs.

    Features:
    - Confidence propagation through relationships
    - Evidence combination using Dempster-Shafer
    - Bayesian belief updates
    - Uncertainty quantification
    """

    def __init__(self, store: GraphStore):
        self.store = store
        self._beliefs: Dict[str, BeliefState] = {}

    async def propagate_confidence(
        self,
        entity_ids: Optional[List[str]] = None,
        iterations: int = 10,
        damping: float = 0.8,
    ) -> int:
        """
        Propagate confidence scores through the graph.

        Uses iterative belief propagation similar to loopy BP.
        """
        # Get entities
        if entity_ids:
            entities = await self.store.get_entities_by_ids(entity_ids)
        else:
            entities = await self.store.search_entities(limit=100000)

        entity_map = {e.id: e for e in entities}
        updates = 0

        # Initialize beliefs
        for entity in entities:
            if entity.id not in self._beliefs:
                self._beliefs[entity.id] = BeliefState(
                    confidence=entity.confidence,
                    prior=0.5,
                )

        # Iterative propagation
        for _ in range(iterations):
            new_confidences = {}

            for entity in entities:
                # Get incoming relationships
                rels = await self.store.get_relationships(
                    entity.id,
                    direction="incoming",
                )

                if not rels:
                    new_confidences[entity.id] = entity.confidence
                    continue

                # Aggregate evidence from neighbors
                evidence_scores = []

                for rel in rels:
                    source = entity_map.get(rel.source_id)
                    if source:
                        # Confidence flows from source through relationship
                        evidence = source.confidence * rel.confidence
                        evidence_scores.append(evidence)

                if evidence_scores:
                    # Combine evidence using noisy-OR
                    combined = self._noisy_or(evidence_scores)

                    # Apply damping
                    new_conf = damping * combined + (1 - damping) * entity.confidence
                    new_confidences[entity.id] = new_conf
                else:
                    new_confidences[entity.id] = entity.confidence

            # Update entities
            for entity in entities:
                old_conf = entity.confidence
                new_conf = new_confidences.get(entity.id, old_conf)

                if abs(new_conf - old_conf) > 0.01:
                    entity.confidence = new_conf
                    await self.store.update_entity(entity)
                    updates += 1

        return updates

    def _noisy_or(self, probabilities: List[float]) -> float:
        """
        Combine probabilities using noisy-OR.

        P(effect) = 1 - product(1 - p_i)
        """
        if not probabilities:
            return 0.5

        product = 1.0
        for p in probabilities:
            product *= (1 - p)

        return 1 - product

    def _dempster_combine(
        self,
        beliefs: List[Tuple[float, float]],  # List of (belief, plausibility)
    ) -> Tuple[float, float]:
        """
        Combine beliefs using Dempster-Shafer rule.

        Each belief is (bel, pl) where bel <= pl.
        """
        if not beliefs:
            return (0.0, 1.0)

        combined_bel = beliefs[0][0]
        combined_pl = beliefs[0][1]

        for bel, pl in beliefs[1:]:
            # Conflict
            k = combined_bel * (1 - pl) + (1 - combined_pl) * bel

            if k >= 1.0:
                # Total conflict
                return (0.5, 0.5)

            # Combine
            new_bel = (combined_bel * bel) / (1 - k)
            new_pl = 1 - ((1 - combined_pl) * (1 - pl)) / (1 - k)

            combined_bel = new_bel
            combined_pl = new_pl

        return (combined_bel, combined_pl)

    async def compute_relationship_probability(
        self,
        source_id: str,
        target_id: str,
        relation_type: RelationType,
    ) -> float:
        """
        Compute probability of a relationship existing.

        Uses evidence from:
        - Existing relationship (if any)
        - Path evidence (indirect connections)
        - Type compatibility
        - Historical patterns
        """
        evidence_scores = []

        # 1. Check for direct relationship
        existing = await self.store.get_relationship_between(
            source_id, target_id, relation_type
        )
        if existing:
            evidence_scores.append(existing.confidence)

        # 2. Path evidence
        path = await self.store.find_path(source_id, target_id, max_depth=3)
        if path and path.length > 0:
            # Confidence decays with path length
            path_conf = 0.8 ** path.length * path.min_confidence
            evidence_scores.append(path_conf)

        # 3. Type compatibility
        source = await self.store.get_entity(source_id)
        target = await self.store.get_entity(target_id)

        if source and target:
            type_compatibility = self._get_type_compatibility(
                source.entity_type,
                target.entity_type,
                relation_type,
            )
            if type_compatibility > 0:
                evidence_scores.append(type_compatibility * 0.5)

        # Combine evidence
        if evidence_scores:
            return self._noisy_or(evidence_scores)

        return 0.1  # Default low probability for unknown

    def _get_type_compatibility(
        self,
        source_type,
        target_type,
        relation_type: RelationType,
    ) -> float:
        """Get compatibility score for entity types and relation."""
        # Define expected type pairs for common relations
        expected_pairs = {
            RelationType.WORKS_FOR: [("person", "organization")],
            RelationType.MANAGES: [("person", "person")],
            RelationType.CREATED: [("person", "project"), ("person", "document")],
            RelationType.MEMBER_OF: [("person", "team"), ("person", "organization")],
            RelationType.PART_OF: [("project", "project"), ("team", "organization")],
            RelationType.LOCATED_IN: [("person", "location"), ("organization", "location")],
        }

        expected = expected_pairs.get(relation_type, [])

        source_val = source_type.value if hasattr(source_type, 'value') else str(source_type)
        target_val = target_type.value if hasattr(target_type, 'value') else str(target_type)

        for expected_source, expected_target in expected:
            if source_val == expected_source and target_val == expected_target:
                return 1.0

        # Partial match
        for expected_source, expected_target in expected:
            if source_val == expected_source or target_val == expected_target:
                return 0.5

        return 0.2  # Unknown compatibility

    async def suggest_relationships(
        self,
        entity_id: str,
        min_probability: float = 0.5,
        limit: int = 10,
    ) -> List[Tuple[str, RelationType, float]]:
        """
        Suggest potential relationships for an entity.

        Returns list of (target_id, relation_type, probability).
        """
        suggestions = []

        entity = await self.store.get_entity(entity_id)
        if not entity:
            return suggestions

        # Get 2-hop neighborhood
        subgraph = await self.store.get_neighbors(entity_id, depth=2)

        # Consider each entity in neighborhood
        for other_id, other in subgraph.entities.items():
            if other_id == entity_id:
                continue

            # Check if direct relationship exists
            existing_rels = await self.store.get_relationships(entity_id, direction="outgoing")
            existing_targets = {r.target_id: r.relation_type for r in existing_rels}

            if other_id in existing_targets:
                continue

            # Compute probabilities for different relation types
            for rel_type in RelationType:
                if rel_type == RelationType.CUSTOM:
                    continue

                prob = await self.compute_relationship_probability(
                    entity_id, other_id, rel_type
                )

                if prob >= min_probability:
                    suggestions.append((other_id, rel_type, prob))

        # Sort by probability
        suggestions.sort(key=lambda x: x[2], reverse=True)

        return suggestions[:limit]

    async def compute_uncertainty(self, entity_id: str) -> Dict[str, float]:
        """
        Compute uncertainty metrics for an entity.

        Returns:
        - confidence: Base confidence
        - entropy: Uncertainty in relationships
        - agreement: Agreement among sources
        """
        entity = await self.store.get_entity(entity_id)
        if not entity:
            return {"confidence": 0.0, "entropy": 1.0, "agreement": 0.0}

        # Get all relationships
        rels = await self.store.get_relationships(entity_id, direction="both")

        if not rels:
            return {
                "confidence": entity.confidence,
                "entropy": 0.5,
                "agreement": 1.0,
            }

        # Compute entropy from relationship confidences
        confidences = [r.confidence for r in rels]
        avg_conf = sum(confidences) / len(confidences)

        # Entropy-like measure
        import math
        entropy = 0.0
        for c in confidences:
            if 0 < c < 1:
                entropy -= c * math.log2(c) + (1 - c) * math.log2(1 - c)
        entropy /= len(confidences) if confidences else 1

        # Agreement (inverse of variance)
        if len(confidences) > 1:
            variance = sum((c - avg_conf) ** 2 for c in confidences) / len(confidences)
            agreement = 1 - min(variance * 4, 1.0)  # Scale variance to [0,1]
        else:
            agreement = 1.0

        return {
            "confidence": entity.confidence,
            "entropy": entropy,
            "agreement": agreement,
        }
