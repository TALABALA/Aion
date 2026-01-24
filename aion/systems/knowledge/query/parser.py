"""
AION Knowledge Graph Query Parser

Parses Cypher-like queries into GraphQuery objects.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import structlog

from aion.systems.knowledge.types import (
    EntityType,
    RelationType,
    GraphQuery,
)

logger = structlog.get_logger(__name__)


@dataclass
class ParsedNode:
    """A parsed node pattern."""
    variable: str = ""
    labels: List[str] = None
    properties: Dict[str, Any] = None

    def __post_init__(self):
        if self.labels is None:
            self.labels = []
        if self.properties is None:
            self.properties = {}


@dataclass
class ParsedRelationship:
    """A parsed relationship pattern."""
    variable: str = ""
    types: List[str] = None
    properties: Dict[str, Any] = None
    direction: str = "outgoing"  # outgoing, incoming, both

    def __post_init__(self):
        if self.types is None:
            self.types = []
        if self.properties is None:
            self.properties = {}


@dataclass
class ParsedPattern:
    """A parsed graph pattern."""
    nodes: List[ParsedNode] = None
    relationships: List[ParsedRelationship] = None

    def __post_init__(self):
        if self.nodes is None:
            self.nodes = []
        if self.relationships is None:
            self.relationships = []


class QueryParser:
    """
    Parser for Cypher-like graph queries.

    Supports a subset of Cypher syntax:
    - MATCH (n:Person {name: "Alice"})-[:KNOWS]->(m:Person)
    - MATCH (a)-[r:WORKS_FOR*1..3]->(b:Organization)
    - WHERE n.age > 30
    - RETURN n, m
    - LIMIT 10
    """

    # Regex patterns
    NODE_PATTERN = re.compile(
        r'\((\w*)?(?::(\w+))?\s*(?:\{([^}]*)\})?\)'
    )

    REL_PATTERN = re.compile(
        r'-\[(\w*)?(?::(\w+))?\s*(?:\*(\d+)?(?:\.\.(\d+))?)?\s*(?:\{([^}]*)\})?\]-[>]?'
    )

    def parse(self, query_string: str) -> GraphQuery:
        """
        Parse a Cypher-like query string into a GraphQuery.

        Example:
            MATCH (p:Person)-[:WORKS_FOR]->(o:Organization)
            WHERE p.name CONTAINS "Alice"
            RETURN p, o
            LIMIT 10
        """
        query_string = query_string.strip()

        # Initialize query
        graph_query = GraphQuery()

        # Extract clauses
        match_clause = self._extract_clause(query_string, "MATCH")
        where_clause = self._extract_clause(query_string, "WHERE")
        return_clause = self._extract_clause(query_string, "RETURN")
        limit_clause = self._extract_clause(query_string, "LIMIT")
        order_clause = self._extract_clause(query_string, "ORDER BY")

        # Parse MATCH clause
        if match_clause:
            pattern = self._parse_pattern(match_clause)

            # Extract entity types
            for node in pattern.nodes:
                for label in node.labels:
                    try:
                        entity_type = EntityType(label.lower())
                        if entity_type not in graph_query.entity_types:
                            graph_query.entity_types.append(entity_type)
                    except ValueError:
                        pass

            # Extract relation types
            for rel in pattern.relationships:
                for rel_type in rel.types:
                    try:
                        relation_type = RelationType(rel_type.lower())
                        if relation_type not in graph_query.relation_types:
                            graph_query.relation_types.append(relation_type)
                    except ValueError:
                        pass

            # Check for path pattern (variable length relationships)
            if any(rel.types for rel in pattern.relationships):
                if len(pattern.nodes) == 2:
                    # Potential path query
                    graph_query.query_type = "path"

        # Parse WHERE clause
        if where_clause:
            filters = self._parse_where(where_clause)
            graph_query.entity_filters.update(filters)

        # Parse LIMIT clause
        if limit_clause:
            try:
                graph_query.limit = int(limit_clause.strip())
            except ValueError:
                pass

        # Parse ORDER BY clause
        if order_clause:
            parts = order_clause.strip().split()
            if parts:
                graph_query.order_by = parts[0].split(".")[-1]  # Remove variable prefix
                if len(parts) > 1 and parts[1].upper() == "ASC":
                    graph_query.order_direction = "asc"

        return graph_query

    def _extract_clause(self, query: str, clause_name: str) -> Optional[str]:
        """Extract a specific clause from the query."""
        # Find clause start
        pattern = re.compile(rf'{clause_name}\s+', re.IGNORECASE)
        match = pattern.search(query)

        if not match:
            return None

        start = match.end()

        # Find next clause or end
        next_clauses = ["MATCH", "WHERE", "RETURN", "LIMIT", "ORDER", "SKIP"]
        end = len(query)

        for next_clause in next_clauses:
            if next_clause == clause_name:
                continue
            next_pattern = re.compile(rf'\s+{next_clause}\s+', re.IGNORECASE)
            next_match = next_pattern.search(query, start)
            if next_match and next_match.start() < end:
                end = next_match.start()

        return query[start:end].strip()

    def _parse_pattern(self, pattern_string: str) -> ParsedPattern:
        """Parse a graph pattern string."""
        pattern = ParsedPattern()

        # Find all nodes
        for match in self.NODE_PATTERN.finditer(pattern_string):
            variable = match.group(1) or ""
            label = match.group(2)
            props_str = match.group(3)

            node = ParsedNode(
                variable=variable,
                labels=[label] if label else [],
                properties=self._parse_properties(props_str) if props_str else {},
            )
            pattern.nodes.append(node)

        # Find all relationships
        for match in self.REL_PATTERN.finditer(pattern_string):
            variable = match.group(1) or ""
            rel_type = match.group(2)
            min_hops = match.group(3)
            max_hops = match.group(4)
            props_str = match.group(5)

            # Determine direction from arrow
            direction = "outgoing"
            if "<-" in pattern_string[:match.start()]:
                direction = "incoming"

            rel = ParsedRelationship(
                variable=variable,
                types=[rel_type] if rel_type else [],
                properties=self._parse_properties(props_str) if props_str else {},
                direction=direction,
            )
            pattern.relationships.append(rel)

        return pattern

    def _parse_properties(self, props_string: str) -> Dict[str, Any]:
        """Parse property map string."""
        if not props_string:
            return {}

        props = {}
        # Simple key: value parsing
        pairs = props_string.split(",")

        for pair in pairs:
            if ":" not in pair:
                continue

            key, value = pair.split(":", 1)
            key = key.strip().strip('"\'')
            value = value.strip()

            # Parse value type
            if value.startswith('"') or value.startswith("'"):
                value = value.strip('"\'')
            elif value.lower() == "true":
                value = True
            elif value.lower() == "false":
                value = False
            elif value.isdigit():
                value = int(value)
            else:
                try:
                    value = float(value)
                except ValueError:
                    pass

            props[key] = value

        return props

    def _parse_where(self, where_string: str) -> Dict[str, Any]:
        """Parse WHERE clause into filters."""
        filters = {}

        # Simple pattern matching for common conditions
        # property CONTAINS value
        contains_match = re.search(r'(\w+)\.(\w+)\s+CONTAINS\s+["\']([^"\']+)["\']', where_string, re.IGNORECASE)
        if contains_match:
            prop = contains_match.group(2)
            value = contains_match.group(3)
            filters[f"{prop}_contains"] = value

        # property = value
        equals_match = re.search(r'(\w+)\.(\w+)\s*=\s*["\']([^"\']+)["\']', where_string)
        if equals_match:
            prop = equals_match.group(2)
            value = equals_match.group(3)
            filters[prop] = value

        # property > value (for numbers)
        gt_match = re.search(r'(\w+)\.(\w+)\s*>\s*(\d+)', where_string)
        if gt_match:
            prop = gt_match.group(2)
            value = int(gt_match.group(3))
            filters[f"{prop}_gt"] = value

        # property < value
        lt_match = re.search(r'(\w+)\.(\w+)\s*<\s*(\d+)', where_string)
        if lt_match:
            prop = lt_match.group(2)
            value = int(lt_match.group(3))
            filters[f"{prop}_lt"] = value

        return filters


def parse_cypher(query_string: str) -> GraphQuery:
    """Convenience function to parse a Cypher query."""
    parser = QueryParser()
    return parser.parse(query_string)
