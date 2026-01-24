"""
AION Rule-Based Inference Engine

Forward and backward chaining inference with pattern matching.
"""

from __future__ import annotations

import re
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
)
from aion.systems.knowledge.store.base import GraphStore

logger = structlog.get_logger(__name__)


@dataclass
class PatternMatch:
    """A match of a rule pattern against the graph."""
    nodes: Dict[str, Entity] = field(default_factory=dict)  # variable -> entity
    relationships: Dict[str, Relationship] = field(default_factory=dict)  # variable -> relationship
    confidence: float = 1.0


class RuleEngine:
    """
    Rule-based inference engine.

    Supports patterns like:
    - (a:Person)-[:MANAGES]->(b:Person)-[:WORKS_ON]->(c:Project)
      => (a)-[:RESPONSIBLE_FOR]->(c)

    - (a)-[:IS_A]->(b)-[:IS_A]->(c)
      => (a)-[:IS_A]->(c)  [transitive closure]
    """

    # Pattern regex
    NODE_PATTERN = re.compile(r'\((\w+)(?::(\w+))?\)')
    REL_PATTERN = re.compile(r'-\[:(\w+)\]->')

    def __init__(self, store: GraphStore):
        self.store = store
        self._rules: List[InferenceRule] = []

    def add_rule(self, rule: InferenceRule) -> None:
        """Add an inference rule."""
        self._rules.append(rule)
        # Sort by priority
        self._rules.sort(key=lambda r: r.priority, reverse=True)

    def remove_rule(self, rule_id: str) -> bool:
        """Remove a rule by ID."""
        for i, rule in enumerate(self._rules):
            if rule.id == rule_id:
                self._rules.pop(i)
                return True
        return False

    def get_rules(self) -> List[InferenceRule]:
        """Get all rules."""
        return list(self._rules)

    async def apply_all_rules(
        self,
        entity_ids: Optional[List[str]] = None,
    ) -> List[InferenceResult]:
        """Apply all rules and return inference results."""
        results = []

        for rule in self._rules:
            if not rule.enabled:
                continue

            result = await self.apply_rule(rule, entity_ids)
            if result.inferred_relationships:
                results.append(result)

        return results

    async def apply_rule(
        self,
        rule: InferenceRule,
        entity_ids: Optional[List[str]] = None,
    ) -> InferenceResult:
        """Apply a single rule."""
        import time
        start_time = time.time()

        result = InferenceResult(
            rule_id=rule.id,
            rule_name=rule.name,
        )

        # Parse pattern
        pattern_nodes, pattern_rels = self._parse_pattern(rule.pattern)

        if len(pattern_nodes) < 2:
            return result

        # Get candidate entities for first node
        first_var, first_type = pattern_nodes[0]

        if entity_ids:
            candidates = await self.store.get_entities_by_ids(entity_ids)
        else:
            candidates = await self.store.search_entities(
                entity_type=EntityType(first_type) if first_type else None,
                limit=10000,
            )

        # Match pattern starting from each candidate
        for candidate in candidates:
            if first_type and candidate.entity_type.value != first_type.lower():
                continue

            matches = await self._match_pattern_from(
                candidate,
                pattern_nodes,
                pattern_rels,
            )

            for match in matches:
                # Check conditions
                if self._check_conditions(rule.conditions, match):
                    # Create inferred relationship
                    inferred = self._create_inference(rule, match)
                    if inferred:
                        result.inferred_relationships.append(inferred)
                        result.matched_patterns += 1

        result.execution_time_ms = (time.time() - start_time) * 1000

        # Update rule statistics
        rule.times_matched += result.matched_patterns
        rule.times_inferred += len(result.inferred_relationships)

        return result

    def _parse_pattern(
        self,
        pattern: str,
    ) -> Tuple[List[Tuple[str, Optional[str]]], List[str]]:
        """
        Parse a pattern string.

        Returns (nodes, relationships) where:
        - nodes: [(variable, type), ...]
        - relationships: [rel_type, ...]
        """
        nodes = []
        for match in self.NODE_PATTERN.finditer(pattern):
            var = match.group(1)
            node_type = match.group(2)
            nodes.append((var, node_type))

        rels = []
        for match in self.REL_PATTERN.finditer(pattern):
            rels.append(match.group(1))

        return nodes, rels

    async def _match_pattern_from(
        self,
        start: Entity,
        pattern_nodes: List[Tuple[str, Optional[str]]],
        pattern_rels: List[str],
    ) -> List[PatternMatch]:
        """Match pattern starting from an entity."""
        if not pattern_nodes:
            return []

        first_var, _ = pattern_nodes[0]
        initial_match = PatternMatch(nodes={first_var: start})

        matches = [initial_match]

        # Extend match for each relationship in pattern
        for i, rel_type in enumerate(pattern_rels):
            if i + 1 >= len(pattern_nodes):
                break

            next_var, next_type = pattern_nodes[i + 1]
            new_matches = []

            for match in matches:
                # Get current end node
                current_var, _ = pattern_nodes[i]
                current_entity = match.nodes.get(current_var)

                if not current_entity:
                    continue

                # Get relationships matching type
                try:
                    rel_type_enum = RelationType(rel_type.lower())
                except ValueError:
                    continue

                rels = await self.store.get_relationships(
                    current_entity.id,
                    direction="outgoing",
                    relation_type=rel_type_enum,
                )

                for rel in rels:
                    # Get target entity
                    target = await self.store.get_entity(rel.target_id)
                    if not target:
                        continue

                    # Check type constraint
                    if next_type and target.entity_type.value != next_type.lower():
                        continue

                    # Create extended match
                    new_match = PatternMatch(
                        nodes={**match.nodes, next_var: target},
                        relationships={**match.relationships, f"r{i}": rel},
                        confidence=match.confidence * rel.confidence,
                    )
                    new_matches.append(new_match)

            matches = new_matches

        return matches

    def _check_conditions(
        self,
        conditions: List[str],
        match: PatternMatch,
    ) -> bool:
        """Check if conditions are satisfied."""
        if not conditions:
            return True

        # Build context for evaluation
        context = {}
        for var, entity in match.nodes.items():
            context[var] = entity
            # Add properties
            for key, value in entity.properties.items():
                context[f"{var}_{key}"] = value

        for condition in conditions:
            try:
                # Simple condition evaluation
                # Format: "a.property == b.property" or "a.property > value"
                result = self._eval_condition(condition, context)
                if not result:
                    return False
            except Exception as e:
                logger.warning(f"Condition evaluation failed: {condition} - {e}")
                return False

        return True

    def _eval_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """Evaluate a simple condition."""
        # Parse condition
        # Support: ==, !=, <, >, <=, >=, contains
        operators = ["==", "!=", "<=", ">=", "<", ">", " contains "]

        for op in operators:
            if op in condition:
                parts = condition.split(op, 1)
                if len(parts) != 2:
                    return False

                left = self._resolve_value(parts[0].strip(), context)
                right = self._resolve_value(parts[1].strip(), context)

                if op == "==":
                    return left == right
                elif op == "!=":
                    return left != right
                elif op == "<":
                    return left < right
                elif op == ">":
                    return left > right
                elif op == "<=":
                    return left <= right
                elif op == ">=":
                    return left >= right
                elif op == " contains ":
                    return str(right) in str(left)

        return True

    def _resolve_value(self, expr: str, context: Dict[str, Any]) -> Any:
        """Resolve a value expression."""
        # Check for variable.property
        if "." in expr:
            parts = expr.split(".", 1)
            var = parts[0]
            prop = parts[1]

            if var in context:
                entity = context[var]
                if isinstance(entity, Entity):
                    if prop == "name":
                        return entity.name
                    elif prop == "type":
                        return entity.entity_type.value
                    else:
                        return entity.properties.get(prop)

        # Check context directly
        if expr in context:
            return context[expr]

        # Try parsing as literal
        if expr.startswith('"') and expr.endswith('"'):
            return expr[1:-1]
        if expr.startswith("'") and expr.endswith("'"):
            return expr[1:-1]
        try:
            return int(expr)
        except ValueError:
            pass
        try:
            return float(expr)
        except ValueError:
            pass

        return expr

    def _create_inference(
        self,
        rule: InferenceRule,
        match: PatternMatch,
    ) -> Optional[Relationship]:
        """Create an inferred relationship from a rule match."""
        # Get first and last nodes as source and target
        node_list = list(match.nodes.values())
        if len(node_list) < 2:
            return None

        source = node_list[0]
        target = node_list[-1]

        # Don't create self-loops (unless rule explicitly allows)
        if source.id == target.id:
            return None

        # Compute confidence
        confidence = rule.base_confidence * match.confidence

        # Apply decay for pattern length
        pattern_hops = len(match.relationships)
        confidence *= (1 - rule.confidence_decay) ** pattern_hops

        if confidence < 0.1:  # Minimum threshold
            return None

        # Create relationship
        rel = Relationship(
            source_id=source.id,
            target_id=target.id,
            relation_type=rule.infer_relation,
            custom_type=rule.infer_custom_type,
            confidence=confidence,
            bidirectional=rule.bidirectional_output,
            properties={
                **rule.infer_properties,
                "inferred": True,
                "rule_id": rule.id,
                "rule_name": rule.name,
                "pattern_confidence": match.confidence,
            },
        )

        return rel

    def match_pattern(
        self,
        pattern: str,
        subgraph: Subgraph,
    ) -> List[PatternMatch]:
        """
        Match a pattern against a subgraph (synchronous, in-memory).

        Used for fast matching against loaded subgraphs.
        """
        pattern_nodes, pattern_rels = self._parse_pattern(pattern)

        if not pattern_nodes:
            return []

        first_var, first_type = pattern_nodes[0]

        # Find starting candidates
        candidates = []
        for entity in subgraph.entities.values():
            if first_type and entity.entity_type.value != first_type.lower():
                continue
            candidates.append(entity)

        matches = []

        for candidate in candidates:
            initial = PatternMatch(nodes={first_var: candidate})
            partial_matches = [initial]

            for i, rel_type in enumerate(pattern_rels):
                if i + 1 >= len(pattern_nodes):
                    break

                next_var, next_type = pattern_nodes[i + 1]
                new_partial = []

                for pm in partial_matches:
                    current_var, _ = pattern_nodes[i]
                    current = pm.nodes.get(current_var)

                    if not current:
                        continue

                    # Get outgoing relationships from subgraph
                    for rel in subgraph.get_relationships_for(current.id, direction="outgoing"):
                        if rel.relation_type.value != rel_type.lower():
                            continue

                        target = subgraph.entities.get(rel.target_id)
                        if not target:
                            continue

                        if next_type and target.entity_type.value != next_type.lower():
                            continue

                        new_pm = PatternMatch(
                            nodes={**pm.nodes, next_var: target},
                            relationships={**pm.relationships, f"r{i}": rel},
                            confidence=pm.confidence * rel.confidence,
                        )
                        new_partial.append(new_pm)

                partial_matches = new_partial

            matches.extend(partial_matches)

        return matches

    def apply_inference(self, rule: InferenceRule, match: PatternMatch) -> Optional[Relationship]:
        """Apply inference from a pattern match (synchronous wrapper)."""
        return self._create_inference(rule, match)


# =============================================================================
# Default Rules
# =============================================================================

def get_default_rules() -> List[InferenceRule]:
    """Get default inference rules."""
    return [
        # Transitive IS_A
        InferenceRule(
            name="Transitive IS_A",
            description="If A is-a B and B is-a C, then A is-a C",
            pattern="(a)-[:IS_A]->(b)-[:IS_A]->(c)",
            infer_relation=RelationType.IS_A,
            base_confidence=0.95,
            confidence_decay=0.05,
            priority=10,
        ),

        # Transitive PART_OF
        InferenceRule(
            name="Transitive PART_OF",
            description="If A is part-of B and B is part-of C, then A is part-of C",
            pattern="(a)-[:PART_OF]->(b)-[:PART_OF]->(c)",
            infer_relation=RelationType.PART_OF,
            base_confidence=0.9,
            confidence_decay=0.1,
            priority=10,
        ),

        # Manager knows subordinates
        InferenceRule(
            name="Manager Knows Reports",
            description="A manager knows people who report to them",
            pattern="(a:Person)-[:MANAGES]->(b:Person)",
            infer_relation=RelationType.KNOWS,
            base_confidence=0.95,
            bidirectional_output=True,
            priority=5,
        ),

        # Same team implies collaboration
        InferenceRule(
            name="Team Collaboration",
            description="People on the same team collaborate",
            pattern="(a:Person)-[:MEMBER_OF]->(t:Team)<-[:MEMBER_OF]-(b:Person)",
            infer_relation=RelationType.COLLABORATES_WITH,
            base_confidence=0.7,
            bidirectional_output=True,
            priority=5,
        ),

        # Project author owns project
        InferenceRule(
            name="Author Owns Creation",
            description="Authors own what they create",
            pattern="(a:Person)-[:AUTHORED]->(p:Project)",
            infer_relation=RelationType.OWNS,
            base_confidence=0.8,
            priority=3,
        ),
    ]
