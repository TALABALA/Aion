"""
AION Knowledge Graph Query Engine

State-of-the-art query execution with:
- Multi-strategy query planning
- Cost-based optimization
- Natural language query translation
- Caching and incremental updates
"""

from __future__ import annotations

import hashlib
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import structlog

from aion.systems.knowledge.types import (
    Entity,
    EntityType,
    Relationship,
    RelationType,
    Path,
    Subgraph,
    GraphQuery,
    QueryResult,
    Triple,
)
from aion.systems.knowledge.store.base import GraphStore

logger = structlog.get_logger(__name__)


@dataclass
class QueryPlan:
    """Execution plan for a query."""
    steps: List[Dict[str, Any]] = field(default_factory=list)
    estimated_cost: float = 0.0
    strategy: str = "default"


class QueryCache:
    """LRU cache for query results."""

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, Tuple[QueryResult, float]] = {}
        self._access_order: List[str] = []

    def _compute_key(self, query: GraphQuery) -> str:
        """Compute cache key from query."""
        key_parts = [
            query.query_type,
            str(sorted([t.value for t in query.entity_types])),
            str(sorted([t.value for t in query.relation_types])),
            str(sorted(query.entity_filters.items())),
            str(query.start_entity_id),
            str(query.end_entity_id),
            str(query.max_depth),
            str(query.limit),
            str(query.offset),
            str(query.natural_language),
        ]
        return hashlib.sha256("|".join(key_parts).encode()).hexdigest()

    def get(self, query: GraphQuery) -> Optional[QueryResult]:
        """Get cached result if valid."""
        key = self._compute_key(query)
        if key in self._cache:
            result, timestamp = self._cache[key]
            if time.time() - timestamp < self.ttl_seconds:
                # Move to end (most recently used)
                if key in self._access_order:
                    self._access_order.remove(key)
                self._access_order.append(key)
                return result
            else:
                # Expired
                del self._cache[key]
                if key in self._access_order:
                    self._access_order.remove(key)
        return None

    def put(self, query: GraphQuery, result: QueryResult) -> None:
        """Cache a result."""
        key = self._compute_key(query)

        # Evict if at capacity
        while len(self._cache) >= self.max_size and self._access_order:
            oldest = self._access_order.pop(0)
            if oldest in self._cache:
                del self._cache[oldest]

        self._cache[key] = (result, time.time())
        self._access_order.append(key)

    def invalidate(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self._access_order.clear()


class QueryEngine:
    """
    Query engine for the knowledge graph.

    Features:
    - Pattern matching queries (MATCH)
    - Path finding with multiple algorithms
    - Aggregations and analytics
    - Natural language query translation
    - Query optimization and caching
    """

    def __init__(
        self,
        store: GraphStore,
        enable_cache: bool = True,
        cache_size: int = 1000,
        cache_ttl: int = 300,
    ):
        self.store = store
        self._cache = QueryCache(cache_size, cache_ttl) if enable_cache else None
        self._llm = None

    async def _ensure_llm(self) -> None:
        """Lazy-load LLM for NL queries."""
        if self._llm is None:
            try:
                from aion.core.llm import LLMAdapter
                self._llm = LLMAdapter()
                await self._llm.initialize()
            except Exception as e:
                logger.warning(f"Could not initialize LLM for NL queries: {e}")

    async def execute(self, query: GraphQuery) -> QueryResult:
        """Execute a graph query."""
        start_time = time.time()

        # Check cache
        if self._cache and query.use_cache:
            cached = self._cache.get(query)
            if cached:
                cached.from_cache = True
                return cached

        # Build execution plan
        plan = await self._build_plan(query)

        # Execute based on query type
        try:
            if query.natural_language:
                result = await self._execute_natural_language(query)
            elif query.query_type == "match":
                result = await self._execute_match(query)
            elif query.query_type == "path":
                result = await self._execute_path(query)
            elif query.query_type == "subgraph":
                result = await self._execute_subgraph(query)
            elif query.query_type == "aggregate":
                result = await self._execute_aggregate(query)
            else:
                result = QueryResult()

        except Exception as e:
            logger.error(f"Query execution error: {e}")
            raise

        result.query_id = query.id
        result.execution_time_ms = (time.time() - start_time) * 1000

        if query.explain:
            result.query_plan = {
                "steps": plan.steps,
                "estimated_cost": plan.estimated_cost,
                "strategy": plan.strategy,
            }

        # Cache result
        if self._cache and query.use_cache:
            self._cache.put(query, result)

        return result

    async def _build_plan(self, query: GraphQuery) -> QueryPlan:
        """Build an optimized execution plan."""
        plan = QueryPlan()

        if query.query_type == "match":
            # Estimate selectivity
            selectivity = 1.0
            if query.entity_types:
                selectivity *= 0.2
            if query.entity_filters:
                selectivity *= 0.1

            plan.steps = [
                {"op": "scan_entities", "filters": query.entity_filters, "selectivity": selectivity},
                {"op": "filter_type", "types": [t.value for t in query.entity_types]},
            ]
            if query.relation_types:
                plan.steps.append({"op": "expand_relationships", "types": [t.value for t in query.relation_types]})

            plan.estimated_cost = 100 * selectivity

        elif query.query_type == "path":
            # Choose algorithm based on graph characteristics
            if query.max_depth <= 3:
                plan.strategy = "bfs"
            else:
                plan.strategy = "bidirectional_bfs"

            plan.steps = [
                {"op": "path_find", "algorithm": plan.strategy, "max_depth": query.max_depth},
            ]
            plan.estimated_cost = 2 ** query.max_depth

        elif query.query_type == "subgraph":
            plan.steps = [
                {"op": "traverse", "start": query.start_entity_id, "depth": query.max_depth},
            ]
            plan.estimated_cost = 10 * query.max_depth

        return plan

    async def _execute_match(self, query: GraphQuery) -> QueryResult:
        """Execute a pattern matching query."""
        # Get entities matching filters
        entity_type = query.entity_types[0] if query.entity_types else None

        entities = await self.store.search_entities(
            query=query.entity_filters.get("name_contains") or query.entity_filters.get("query"),
            entity_type=entity_type,
            properties=query.entity_filters.get("properties"),
            limit=query.limit,
            offset=query.offset,
        )

        # Filter by additional criteria
        if query.valid_at:
            # Temporal filter (entities don't have validity, but relationships do)
            pass

        # Get relationships if relation types specified
        relationships = []
        if query.relation_types:
            entity_ids = {e.id for e in entities}
            for entity in entities:
                rels = await self.store.get_relationships(entity.id, direction="both")
                for rel in rels:
                    if rel.relation_type in query.relation_types:
                        # Apply temporal filter
                        if query.valid_at and not rel.is_valid_at(query.valid_at):
                            continue
                        if rel.id not in {r.id for r in relationships}:
                            relationships.append(rel)

        # Build triples
        triples = []
        entity_map = {e.id: e for e in entities}
        for rel in relationships:
            source = entity_map.get(rel.source_id)
            target = entity_map.get(rel.target_id)
            if source and target:
                triples.append(Triple(
                    subject=source,
                    predicate=rel.relation_type,
                    object=target,
                    relationship=rel,
                    confidence=rel.confidence,
                ))

        # Order results
        if query.order_by:
            reverse = query.order_direction == "desc"
            if query.order_by == "name":
                entities.sort(key=lambda e: e.name, reverse=reverse)
            elif query.order_by == "importance":
                entities.sort(key=lambda e: e.importance, reverse=reverse)
            elif query.order_by == "pagerank":
                entities.sort(key=lambda e: e.pagerank, reverse=reverse)
            elif query.order_by == "created_at":
                entities.sort(key=lambda e: e.created_at, reverse=reverse)

        total_count = len(entities)
        has_more = len(entities) == query.limit

        return QueryResult(
            entities=entities,
            relationships=relationships,
            triples=triples,
            total_count=total_count,
            has_more=has_more,
        )

    async def _execute_path(self, query: GraphQuery) -> QueryResult:
        """Execute a path finding query."""
        if not query.start_entity_id or not query.end_entity_id:
            return QueryResult()

        path = await self.store.find_path(
            query.start_entity_id,
            query.end_entity_id,
            max_depth=query.max_depth,
            relation_types=query.relation_types or None,
        )

        if path:
            return QueryResult(
                entities=path.entities,
                relationships=path.relationships,
                paths=[path],
                total_count=1,
            )

        return QueryResult(total_count=0)

    async def _execute_subgraph(self, query: GraphQuery) -> QueryResult:
        """Execute a subgraph extraction query."""
        if not query.start_entity_id:
            return QueryResult()

        subgraph = await self.store.get_neighbors(
            query.start_entity_id,
            relation_types=query.relation_types or None,
            depth=query.max_depth,
        )

        return QueryResult(
            entities=list(subgraph.entities.values()),
            relationships=list(subgraph.relationships.values()),
            subgraph=subgraph,
            total_count=len(subgraph.entities),
        )

    async def _execute_aggregate(self, query: GraphQuery) -> QueryResult:
        """Execute an aggregation query."""
        stats = await self.store.get_stats()

        aggregates: Dict[str, Any] = {
            "total_entities": stats.total_entities,
            "total_relationships": stats.total_relationships,
            "entities_by_type": stats.entities_by_type,
            "relationships_by_type": stats.relationships_by_type,
            "avg_degree": stats.avg_degree,
            "max_degree": stats.max_degree,
            "density": stats.density,
        }

        # Handle specific aggregation type
        if query.aggregation_type == "count":
            if query.entity_types:
                aggregates["count"] = sum(
                    stats.entities_by_type.get(t.value, 0) for t in query.entity_types
                )
            else:
                aggregates["count"] = stats.total_entities

        elif query.aggregation_type == "group_by" and query.group_by_field:
            if query.group_by_field == "entity_type":
                aggregates["groups"] = stats.entities_by_type
            elif query.group_by_field == "relation_type":
                aggregates["groups"] = stats.relationships_by_type

        return QueryResult(aggregates=aggregates)

    async def _execute_natural_language(self, query: GraphQuery) -> QueryResult:
        """Execute a natural language query using LLM."""
        await self._ensure_llm()

        if not self._llm:
            logger.warning("LLM not available for NL query")
            return QueryResult()

        # Get graph schema info
        stats = await self.store.get_stats()

        prompt = f"""Convert this natural language query about a knowledge graph to a structured query.

Natural Language Query: {query.natural_language}

Available Entity Types: {', '.join(EntityType.__members__.keys())}
Available Relation Types: {', '.join(RelationType.__members__.keys())}

Current Graph Statistics:
- Total entities: {stats.total_entities}
- Total relationships: {stats.total_relationships}
- Entities by type: {stats.entities_by_type}
- Relationships by type: {stats.relationships_by_type}

Respond with ONLY a valid JSON object (no other text):
{{
    "query_type": "match" | "path" | "subgraph" | "aggregate",
    "entity_types": ["PERSON", "ORGANIZATION", ...],
    "relation_types": ["WORKS_FOR", "KNOWS", ...],
    "entity_filters": {{"name_contains": "..."}},
    "start_entity_name": "...",
    "end_entity_name": "...",
    "max_depth": 3
}}

Only include fields that are relevant to the query. For example:
- "Who works at Acme?" -> match with entity_types=["PERSON"], relation_types=["WORKS_FOR"], entity_filters={{"name_contains": "Acme"}}
- "How are Alice and Bob connected?" -> path with start_entity_name="Alice", end_entity_name="Bob"
- "Show me all projects" -> match with entity_types=["PROJECT"]
"""

        try:
            response = await self._llm.complete(
                messages=[{"role": "user", "content": prompt}],
                system="You are a knowledge graph query translator. Output only valid JSON.",
            )

            # Parse LLM response
            import json
            text = response.content if hasattr(response, 'content') else str(response)

            # Extract JSON from response
            json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
            if not json_match:
                logger.warning(f"No JSON found in LLM response: {text[:200]}")
                return QueryResult()

            params = json.loads(json_match.group())

            # Build structured query
            structured_query = GraphQuery(
                query_type=params.get("query_type", "match"),
                entity_types=[EntityType(t.lower()) for t in params.get("entity_types", []) if t.lower() in EntityType.__members__.values()],
                relation_types=[RelationType(t.lower()) for t in params.get("relation_types", []) if t.lower() in RelationType.__members__.values()],
                entity_filters=params.get("entity_filters", {}),
                max_depth=params.get("max_depth", 3),
                limit=query.limit,
                offset=query.offset,
            )

            # Resolve entity names for path queries
            if params.get("start_entity_name"):
                start = await self.store.get_entity_by_name(params["start_entity_name"])
                if start:
                    structured_query.start_entity_id = start.id

            if params.get("end_entity_name"):
                end = await self.store.get_entity_by_name(params["end_entity_name"])
                if end:
                    structured_query.end_entity_id = end.id

            # Execute the structured query
            return await self.execute(structured_query)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse NL query response: {e}")
            return QueryResult()
        except Exception as e:
            logger.error(f"NL query execution failed: {e}")
            return QueryResult()

    # ==========================================================================
    # Convenience Methods
    # ==========================================================================

    async def find_entities_by_type(
        self,
        entity_type: EntityType,
        limit: int = 100,
    ) -> List[Entity]:
        """Find all entities of a type."""
        result = await self.execute(GraphQuery(
            query_type="match",
            entity_types=[entity_type],
            limit=limit,
        ))
        return result.entities

    async def find_related(
        self,
        entity_id: str,
        relation_type: Optional[RelationType] = None,
        depth: int = 1,
    ) -> List[Entity]:
        """Find entities related to a given entity."""
        result = await self.execute(GraphQuery(
            query_type="subgraph",
            start_entity_id=entity_id,
            relation_types=[relation_type] if relation_type else [],
            max_depth=depth,
        ))
        return [e for e in result.entities if e.id != entity_id]

    async def find_path_between(
        self,
        start_name: str,
        end_name: str,
        max_depth: int = 5,
    ) -> Optional[Path]:
        """Find path between two entities by name."""
        start = await self.store.get_entity_by_name(start_name)
        end = await self.store.get_entity_by_name(end_name)

        if not start or not end:
            return None

        result = await self.execute(GraphQuery(
            query_type="path",
            start_entity_id=start.id,
            end_entity_id=end.id,
            max_depth=max_depth,
        ))

        return result.paths[0] if result.paths else None

    async def search(
        self,
        query: str,
        entity_types: Optional[List[EntityType]] = None,
        limit: int = 20,
    ) -> List[Entity]:
        """Simple text search."""
        result = await self.execute(GraphQuery(
            query_type="match",
            entity_types=entity_types or [],
            entity_filters={"query": query},
            limit=limit,
        ))
        return result.entities

    def invalidate_cache(self) -> None:
        """Clear the query cache."""
        if self._cache:
            self._cache.invalidate()
