"""
AION Knowledge Graph Inference Engine

State-of-the-art reasoning with:
- Rule-based inference (forward/backward chaining)
- Probabilistic reasoning with uncertainty propagation
- Transitive closure computation
- Graph analytics (centrality, communities)
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import structlog

from aion.systems.knowledge.types import (
    Entity,
    EntityType,
    Relationship,
    RelationType,
    InferenceRule,
    InferenceResult,
    Subgraph,
    Triple,
)
from aion.systems.knowledge.store.base import GraphStore
from aion.systems.knowledge.inference.rules import RuleEngine
from aion.systems.knowledge.inference.probabilistic import ProbabilisticReasoner
from aion.systems.knowledge.inference.paths import PathFinder

logger = structlog.get_logger(__name__)


@dataclass
class InferenceConfig:
    """Configuration for inference engine."""
    max_iterations: int = 100
    min_confidence: float = 0.5
    enable_transitive: bool = True
    enable_symmetric: bool = True
    enable_probabilistic: bool = True
    parallel_workers: int = 4


@dataclass
class InferenceSummary:
    """Summary of inference results."""
    rules_applied: int = 0
    relationships_inferred: int = 0
    transitive_closures: int = 0
    confidence_updates: int = 0
    execution_time_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "rules_applied": self.rules_applied,
            "relationships_inferred": self.relationships_inferred,
            "transitive_closures": self.transitive_closures,
            "confidence_updates": self.confidence_updates,
            "execution_time_ms": self.execution_time_ms,
        }


class InferenceEngine:
    """
    Main inference engine for knowledge graph reasoning.

    Combines:
    - Rule-based inference (RuleEngine)
    - Probabilistic reasoning (ProbabilisticReasoner)
    - Path analysis (PathFinder)
    - Graph analytics
    """

    def __init__(
        self,
        store: GraphStore,
        config: Optional[InferenceConfig] = None,
    ):
        self.store = store
        self.config = config or InferenceConfig()

        self.rule_engine = RuleEngine(store)
        self.probabilistic = ProbabilisticReasoner(store)
        self.path_finder = PathFinder(store)

        self._rules: List[InferenceRule] = []
        self._inferred_cache: Dict[str, Set[str]] = defaultdict(set)  # rel_key -> set of inferred_rel_ids

    def add_rule(self, rule: InferenceRule) -> None:
        """Add an inference rule."""
        self._rules.append(rule)
        self.rule_engine.add_rule(rule)

    def remove_rule(self, rule_id: str) -> bool:
        """Remove an inference rule."""
        for i, rule in enumerate(self._rules):
            if rule.id == rule_id:
                self._rules.pop(i)
                return True
        return False

    async def run_inference(
        self,
        entity_ids: Optional[List[str]] = None,
        materialized: bool = True,
    ) -> InferenceSummary:
        """
        Run full inference on the graph.

        Args:
            entity_ids: Limit inference to these entities (None = all)
            materialized: If True, persist inferred relationships

        Returns:
            Summary of inference operations
        """
        import time
        start_time = time.time()

        summary = InferenceSummary()

        # 1. Apply inference rules
        if self._rules:
            rule_results = await self.rule_engine.apply_all_rules(entity_ids)
            for result in rule_results:
                summary.rules_applied += 1
                summary.relationships_inferred += len(result.inferred_relationships)

                if materialized:
                    for rel in result.inferred_relationships:
                        await self.store.create_relationship(rel)

        # 2. Compute transitive closures
        if self.config.enable_transitive:
            transitive_rels = await self._compute_transitive_closures(entity_ids)
            summary.transitive_closures = len(transitive_rels)

            if materialized:
                for rel in transitive_rels:
                    await self.store.create_relationship(rel)

        # 3. Propagate symmetric relationships
        if self.config.enable_symmetric:
            await self._propagate_symmetric(entity_ids)

        # 4. Update confidence scores
        if self.config.enable_probabilistic:
            updates = await self.probabilistic.propagate_confidence(entity_ids)
            summary.confidence_updates = updates

        summary.execution_time_ms = (time.time() - start_time) * 1000

        return summary

    async def _compute_transitive_closures(
        self,
        entity_ids: Optional[List[str]] = None,
    ) -> List[Relationship]:
        """Compute transitive closures for transitive relations."""
        inferred = []

        # Get transitive relation types
        transitive_types = [
            rt for rt in RelationType
            if rt.is_transitive()
        ]

        if not transitive_types:
            return inferred

        # Get entities to process
        if entity_ids:
            entities = await self.store.get_entities_by_ids(entity_ids)
        else:
            entities = await self.store.search_entities(limit=10000)

        # For each entity and transitive relation, compute closure
        for entity in entities:
            for rel_type in transitive_types:
                # BFS to find all reachable nodes
                visited = set()
                queue = [entity.id]
                reachable = []

                while queue:
                    current = queue.pop(0)
                    if current in visited:
                        continue
                    visited.add(current)

                    rels = await self.store.get_relationships(
                        current,
                        direction="outgoing",
                        relation_type=rel_type,
                    )

                    for rel in rels:
                        target = rel.target_id
                        if target not in visited:
                            queue.append(target)
                            if target != entity.id:  # Skip self-loops
                                reachable.append((target, len(visited)))  # (target, distance)

                # Create transitive relationships for reachable nodes
                for target_id, distance in reachable:
                    if distance > 1:  # Only infer non-direct relationships
                        # Check if relationship already exists
                        existing = await self.store.get_relationship_between(
                            entity.id, target_id, rel_type
                        )
                        if not existing:
                            # Compute confidence decay based on distance
                            confidence = 0.9 ** (distance - 1)

                            if confidence >= self.config.min_confidence:
                                rel = Relationship(
                                    source_id=entity.id,
                                    target_id=target_id,
                                    relation_type=rel_type,
                                    confidence=confidence,
                                    properties={"inferred": True, "distance": distance},
                                )
                                inferred.append(rel)

        return inferred

    async def _propagate_symmetric(
        self,
        entity_ids: Optional[List[str]] = None,
    ) -> int:
        """Ensure symmetric relations have reverse edges."""
        count = 0

        symmetric_types = [
            rt for rt in RelationType
            if rt.is_symmetric()
        ]

        if not symmetric_types:
            return count

        # Get all relationships of symmetric types
        for rel_type in symmetric_types:
            if entity_ids:
                for entity_id in entity_ids:
                    rels = await self.store.get_relationships(
                        entity_id,
                        direction="both",
                        relation_type=rel_type,
                    )
                    for rel in rels:
                        if not rel.bidirectional:
                            # Check for reverse
                            reverse = await self.store.get_relationship_between(
                                rel.target_id, rel.source_id, rel_type
                            )
                            if not reverse:
                                # Create reverse relationship
                                reverse_rel = Relationship(
                                    source_id=rel.target_id,
                                    target_id=rel.source_id,
                                    relation_type=rel_type,
                                    confidence=rel.confidence,
                                    properties={"inferred": True, "symmetric_of": rel.id},
                                )
                                await self.store.create_relationship(reverse_rel)
                                count += 1

        return count

    async def infer_relationships(
        self,
        entity_id: str,
        max_depth: int = 2,
    ) -> List[Relationship]:
        """
        Infer new relationships for a specific entity.

        Uses pattern matching and learned embeddings.
        """
        inferred = []

        # Get entity neighborhood
        subgraph = await self.store.get_neighbors(entity_id, depth=max_depth)

        # Apply rules to this subgraph
        for rule in self._rules:
            if not rule.enabled:
                continue

            matches = self.rule_engine.match_pattern(rule.pattern, subgraph)
            for match in matches:
                if entity_id in [m.id for m in match.get("nodes", [])]:
                    rel = self.rule_engine.apply_inference(rule, match)
                    if rel:
                        inferred.append(rel)

        return inferred

    async def compute_centrality(
        self,
        algorithm: str = "pagerank",
        damping: float = 0.85,
        iterations: int = 100,
    ) -> Dict[str, float]:
        """
        Compute centrality scores for all entities.

        Algorithms:
        - pagerank: PageRank centrality
        - degree: Degree centrality
        - betweenness: Betweenness centrality (approximate)
        """
        entities = await self.store.search_entities(limit=100000)
        entity_ids = [e.id for e in entities]

        if algorithm == "pagerank":
            return await self._compute_pagerank(entity_ids, damping, iterations)
        elif algorithm == "degree":
            return await self._compute_degree_centrality(entity_ids)
        elif algorithm == "betweenness":
            return await self._compute_betweenness_centrality(entity_ids)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    async def _compute_pagerank(
        self,
        entity_ids: List[str],
        damping: float,
        iterations: int,
    ) -> Dict[str, float]:
        """Compute PageRank centrality."""
        n = len(entity_ids)
        if n == 0:
            return {}

        # Initialize scores
        scores = {eid: 1.0 / n for eid in entity_ids}
        new_scores = dict(scores)

        # Build adjacency for efficiency
        outgoing: Dict[str, List[str]] = defaultdict(list)
        for eid in entity_ids:
            rels = await self.store.get_relationships(eid, direction="outgoing")
            for rel in rels:
                if rel.target_id in scores:
                    outgoing[eid].append(rel.target_id)

        # Iterate
        for _ in range(iterations):
            for eid in entity_ids:
                # Sum incoming contributions
                incoming_sum = 0.0
                rels = await self.store.get_relationships(eid, direction="incoming")
                for rel in rels:
                    source = rel.source_id
                    if source in scores:
                        out_degree = len(outgoing.get(source, []))
                        if out_degree > 0:
                            incoming_sum += scores[source] / out_degree

                new_scores[eid] = (1 - damping) / n + damping * incoming_sum

            # Normalize
            total = sum(new_scores.values())
            if total > 0:
                for eid in entity_ids:
                    new_scores[eid] /= total

            scores = dict(new_scores)

        return scores

    async def _compute_degree_centrality(
        self,
        entity_ids: List[str],
    ) -> Dict[str, float]:
        """Compute normalized degree centrality."""
        n = len(entity_ids)
        if n <= 1:
            return {eid: 0.0 for eid in entity_ids}

        scores = {}
        max_degree = n - 1  # Maximum possible degree

        for eid in entity_ids:
            degree = await self.store.get_entity_degree(eid)
            scores[eid] = degree / max_degree

        return scores

    async def _compute_betweenness_centrality(
        self,
        entity_ids: List[str],
        sample_size: int = 100,
    ) -> Dict[str, float]:
        """
        Compute approximate betweenness centrality.

        Uses sampling for large graphs.
        """
        import random

        n = len(entity_ids)
        if n <= 2:
            return {eid: 0.0 for eid in entity_ids}

        scores = {eid: 0.0 for eid in entity_ids}

        # Sample pairs
        sample = min(sample_size, n * (n - 1) // 2)
        pairs = random.sample(
            [(a, b) for i, a in enumerate(entity_ids) for b in entity_ids[i+1:]],
            sample
        )

        for source, target in pairs:
            path = await self.store.find_path(source, target, max_depth=5)
            if path and path.length > 1:
                # Add score to intermediate nodes
                for entity in path.entities[1:-1]:
                    if entity.id in scores:
                        scores[entity.id] += 1

        # Normalize
        normalization = 2.0 / ((n - 1) * (n - 2)) if n > 2 else 1.0
        scale = len(pairs) / max(sample, 1)

        for eid in entity_ids:
            scores[eid] *= normalization * scale

        return scores

    async def update_entity_scores(self) -> int:
        """Update centrality scores for all entities."""
        # Compute PageRank
        pagerank_scores = await self.compute_centrality("pagerank")

        # Compute degree centrality
        degree_scores = await self.compute_centrality("degree")

        # Update entities
        count = 0
        for eid, pr_score in pagerank_scores.items():
            entity = await self.store.get_entity(eid)
            if entity:
                entity.pagerank = pr_score
                entity.degree_centrality = degree_scores.get(eid, 0.0)
                await self.store.update_entity(entity)
                count += 1

        return count

    async def find_communities(
        self,
        algorithm: str = "label_propagation",
        min_community_size: int = 2,
    ) -> Dict[str, List[str]]:
        """
        Find communities in the graph.

        Returns mapping of community_id -> list of entity_ids.
        """
        entities = await self.store.search_entities(limit=100000)
        entity_ids = [e.id for e in entities]

        if algorithm == "label_propagation":
            return await self._label_propagation(entity_ids, min_community_size)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    async def _label_propagation(
        self,
        entity_ids: List[str],
        min_size: int,
    ) -> Dict[str, List[str]]:
        """Label propagation community detection."""
        import random

        # Initialize each node with its own label
        labels = {eid: eid for eid in entity_ids}

        # Iterate until convergence
        changed = True
        max_iterations = 100
        iteration = 0

        while changed and iteration < max_iterations:
            changed = False
            iteration += 1

            # Shuffle order
            random.shuffle(entity_ids)

            for eid in entity_ids:
                # Get neighbor labels
                rels = await self.store.get_relationships(eid, direction="both")
                neighbor_labels = []

                for rel in rels:
                    neighbor = rel.target_id if rel.source_id == eid else rel.source_id
                    if neighbor in labels:
                        neighbor_labels.append(labels[neighbor])

                if neighbor_labels:
                    # Find most common label
                    label_counts: Dict[str, int] = {}
                    for label in neighbor_labels:
                        label_counts[label] = label_counts.get(label, 0) + 1

                    max_count = max(label_counts.values())
                    top_labels = [l for l, c in label_counts.items() if c == max_count]
                    new_label = random.choice(top_labels)

                    if labels[eid] != new_label:
                        labels[eid] = new_label
                        changed = True

        # Group by label
        communities: Dict[str, List[str]] = defaultdict(list)
        for eid, label in labels.items():
            communities[label].append(eid)

        # Filter by minimum size
        return {
            cid: members
            for cid, members in communities.items()
            if len(members) >= min_size
        }
