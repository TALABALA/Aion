"""
AION Entity Linker

Link extracted entities to existing entities in the knowledge graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import structlog

from aion.systems.knowledge.types import Entity, EntityType

logger = structlog.get_logger(__name__)


@dataclass
class LinkCandidate:
    """A candidate for entity linking."""
    entity: Entity
    score: float
    match_type: str  # exact, alias, fuzzy, embedding


@dataclass
class LinkerConfig:
    """Configuration for entity linker."""
    exact_match_threshold: float = 1.0
    fuzzy_match_threshold: float = 0.8
    embedding_threshold: float = 0.85
    max_candidates: int = 5
    type_must_match: bool = False


class EntityLinker:
    """
    Link extracted entities to existing knowledge graph entities.

    Strategies:
    1. Exact name match
    2. Alias matching
    3. Fuzzy string matching
    4. Embedding similarity
    """

    def __init__(self, config: Optional[LinkerConfig] = None):
        self.config = config or LinkerConfig()

    def find_best_match(
        self,
        entity: Entity,
        candidates: List[Entity],
        context: Optional[str] = None,
    ) -> Optional[Entity]:
        """
        Find the best matching entity from candidates.

        Args:
            entity: Entity to match
            candidates: List of candidate entities
            context: Optional context for disambiguation

        Returns:
            Best matching entity or None
        """
        if not candidates:
            return None

        scored = self.score_candidates(entity, candidates)

        if not scored:
            return None

        best = scored[0]

        # Check threshold
        if best.score >= self.config.fuzzy_match_threshold:
            return best.entity

        return None

    def score_candidates(
        self,
        entity: Entity,
        candidates: List[Entity],
    ) -> List[LinkCandidate]:
        """Score all candidates for an entity."""
        results = []

        for candidate in candidates:
            score, match_type = self._compute_similarity(entity, candidate)

            # Type filter
            if self.config.type_must_match:
                if entity.entity_type != candidate.entity_type:
                    continue

            if score > 0:
                results.append(LinkCandidate(
                    entity=candidate,
                    score=score,
                    match_type=match_type,
                ))

        # Sort by score
        results.sort(key=lambda x: x.score, reverse=True)

        return results[:self.config.max_candidates]

    def _compute_similarity(
        self,
        entity: Entity,
        candidate: Entity,
    ) -> Tuple[float, str]:
        """Compute similarity between two entities."""
        name1 = entity.name.lower().strip()
        name2 = candidate.name.lower().strip()

        # 1. Exact match
        if name1 == name2:
            return 1.0, "exact"

        # 2. Alias match
        aliases1 = {a.lower() for a in entity.aliases}
        aliases2 = {a.lower() for a in candidate.aliases}

        if name1 in aliases2 or name2 in aliases1:
            return 0.95, "alias"

        if aliases1 & aliases2:
            return 0.9, "alias"

        # 3. Fuzzy string match
        fuzzy_score = self._fuzzy_similarity(name1, name2)

        if fuzzy_score >= self.config.fuzzy_match_threshold:
            return fuzzy_score, "fuzzy"

        # 4. Token overlap
        tokens1 = set(name1.split())
        tokens2 = set(name2.split())

        if tokens1 and tokens2:
            jaccard = len(tokens1 & tokens2) / len(tokens1 | tokens2)
            if jaccard > 0.5:
                return jaccard * 0.8, "token"

        return 0.0, "none"

    def _fuzzy_similarity(self, s1: str, s2: str) -> float:
        """Compute fuzzy string similarity using Levenshtein ratio."""
        if not s1 or not s2:
            return 0.0

        # Try using rapidfuzz if available
        try:
            from rapidfuzz import fuzz
            return fuzz.ratio(s1, s2) / 100.0
        except ImportError:
            pass

        # Fallback to simple implementation
        return self._levenshtein_ratio(s1, s2)

    def _levenshtein_ratio(self, s1: str, s2: str) -> float:
        """Compute Levenshtein distance ratio."""
        if s1 == s2:
            return 1.0

        len1, len2 = len(s1), len(s2)

        if len1 == 0 or len2 == 0:
            return 0.0

        # Create distance matrix
        matrix = [[0] * (len2 + 1) for _ in range(len1 + 1)]

        for i in range(len1 + 1):
            matrix[i][0] = i
        for j in range(len2 + 1):
            matrix[0][j] = j

        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                cost = 0 if s1[i-1] == s2[j-1] else 1
                matrix[i][j] = min(
                    matrix[i-1][j] + 1,      # Deletion
                    matrix[i][j-1] + 1,      # Insertion
                    matrix[i-1][j-1] + cost  # Substitution
                )

        distance = matrix[len1][len2]
        max_len = max(len1, len2)

        return 1.0 - (distance / max_len)

    def merge_entities(
        self,
        primary: Entity,
        secondary: Entity,
    ) -> Entity:
        """
        Merge two entities, keeping primary as base.

        Combines properties, aliases, and descriptions.
        """
        # Merge aliases
        all_aliases = set(primary.aliases + secondary.aliases)
        all_aliases.add(secondary.name)  # Add secondary name as alias
        all_aliases.discard(primary.name)  # Remove primary name

        primary.aliases = list(all_aliases)

        # Merge properties (secondary fills gaps)
        for key, value in secondary.properties.items():
            if key not in primary.properties:
                primary.properties[key] = value

        # Merge description if primary is empty
        if not primary.description and secondary.description:
            primary.description = secondary.description

        # Use higher confidence
        primary.confidence = max(primary.confidence, secondary.confidence)

        # Use higher importance
        primary.importance = max(primary.importance, secondary.importance)

        return primary

    async def link_to_graph(
        self,
        entity: Entity,
        store: Any,  # GraphStore
        create_if_missing: bool = True,
    ) -> Entity:
        """
        Link entity to knowledge graph.

        Args:
            entity: Entity to link
            store: Graph store
            create_if_missing: Create entity if no match found

        Returns:
            Linked or newly created entity
        """
        # Search for existing entities
        candidates = await store.search_entities(
            query=entity.name,
            entity_type=entity.entity_type if self.config.type_must_match else None,
            limit=10,
        )

        # Find best match
        match = self.find_best_match(entity, candidates)

        if match:
            # Merge with existing
            merged = self.merge_entities(match, entity)
            await store.update_entity(merged)
            return merged

        elif create_if_missing:
            # Create new entity
            await store.create_entity(entity)
            return entity

        return entity


class BatchLinker:
    """Batch entity linking for efficiency."""

    def __init__(self, linker: Optional[EntityLinker] = None):
        self.linker = linker or EntityLinker()
        self._cache: Dict[str, Entity] = {}

    async def link_batch(
        self,
        entities: List[Entity],
        store: Any,
    ) -> List[Entity]:
        """Link a batch of entities to the graph."""
        # Get all potential candidates
        all_names = {e.name for e in entities}
        all_names.update(a for e in entities for a in e.aliases)

        # Search for each unique name
        candidates = []
        for name in all_names:
            results = await store.search_entities(query=name, limit=5)
            candidates.extend(results)

        # Deduplicate candidates
        seen = set()
        unique_candidates = []
        for c in candidates:
            if c.id not in seen:
                seen.add(c.id)
                unique_candidates.append(c)

        # Link each entity
        linked = []
        for entity in entities:
            match = self.linker.find_best_match(entity, unique_candidates)
            if match:
                merged = self.linker.merge_entities(match, entity)
                linked.append(merged)
            else:
                linked.append(entity)

        return linked
