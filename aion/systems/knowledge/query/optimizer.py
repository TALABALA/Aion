"""
AION Knowledge Graph Query Optimizer

Cost-based query optimization with statistics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import structlog

from aion.systems.knowledge.types import (
    EntityType,
    RelationType,
    GraphQuery,
    GraphStatistics,
)
from aion.systems.knowledge.store.base import GraphStore

logger = structlog.get_logger(__name__)


@dataclass
class QueryCost:
    """Cost estimate for a query."""
    estimated_rows: float = 0.0
    io_cost: float = 0.0
    cpu_cost: float = 0.0
    total_cost: float = 0.0

    def compute_total(self) -> float:
        self.total_cost = self.io_cost + self.cpu_cost
        return self.total_cost


@dataclass
class OptimizationHint:
    """Hint for query optimization."""
    hint_type: str = ""
    description: str = ""
    benefit: float = 0.0
    recommendation: str = ""


@dataclass
class OptimizedQuery:
    """Query with optimization metadata."""
    query: GraphQuery
    cost: QueryCost
    hints: List[OptimizationHint] = field(default_factory=list)
    rewritten: bool = False
    original_query: Optional[GraphQuery] = None


class QueryOptimizer:
    """
    Cost-based query optimizer.

    Optimization strategies:
    1. Predicate pushdown - apply filters early
    2. Index selection - use appropriate indices
    3. Join ordering - optimal traversal order
    4. Cardinality estimation - selectivity-based planning
    """

    def __init__(self, store: GraphStore):
        self.store = store
        self._stats: Optional[GraphStatistics] = None
        self._stats_timestamp: float = 0

    async def _refresh_stats(self) -> None:
        """Refresh statistics if stale."""
        import time
        now = time.time()
        if not self._stats or now - self._stats_timestamp > 300:  # 5 min cache
            self._stats = await self.store.get_stats()
            self._stats_timestamp = now

    async def optimize(self, query: GraphQuery) -> OptimizedQuery:
        """
        Optimize a query.

        Returns optimized query with cost estimate and hints.
        """
        await self._refresh_stats()

        cost = await self._estimate_cost(query)
        hints = await self._generate_hints(query, cost)

        # Try query rewrites
        rewritten_query, was_rewritten = await self._try_rewrite(query)

        return OptimizedQuery(
            query=rewritten_query,
            cost=cost,
            hints=hints,
            rewritten=was_rewritten,
            original_query=query if was_rewritten else None,
        )

    async def _estimate_cost(self, query: GraphQuery) -> QueryCost:
        """Estimate query execution cost."""
        cost = QueryCost()

        if not self._stats:
            return cost

        # Base row estimate
        if query.query_type == "match":
            cost.estimated_rows = self._stats.total_entities

            # Apply selectivity for type filter
            if query.entity_types:
                type_selectivity = sum(
                    self._stats.entities_by_type.get(t.value, 0)
                    for t in query.entity_types
                ) / max(self._stats.total_entities, 1)
                cost.estimated_rows *= type_selectivity

            # Apply selectivity for text search
            if query.entity_filters.get("query") or query.entity_filters.get("name_contains"):
                cost.estimated_rows *= 0.1  # Assume 10% match

            # Property filters
            if query.entity_filters.get("properties"):
                cost.estimated_rows *= 0.05  # Assume 5% match per property

        elif query.query_type == "path":
            # Path cost grows exponentially with depth
            avg_degree = self._stats.avg_degree if self._stats.avg_degree > 0 else 2
            cost.estimated_rows = min(avg_degree ** query.max_depth, self._stats.total_entities)

        elif query.query_type == "subgraph":
            avg_degree = self._stats.avg_degree if self._stats.avg_degree > 0 else 2
            cost.estimated_rows = min(
                sum(avg_degree ** d for d in range(query.max_depth + 1)),
                self._stats.total_entities
            )

        # IO cost (based on estimated rows)
        cost.io_cost = cost.estimated_rows * 1.0  # 1 cost unit per row

        # CPU cost (filtering, sorting)
        cost.cpu_cost = cost.estimated_rows * 0.1

        if query.order_by:
            cost.cpu_cost += cost.estimated_rows * 0.5  # Sort cost

        cost.compute_total()

        return cost

    async def _generate_hints(
        self,
        query: GraphQuery,
        cost: QueryCost,
    ) -> List[OptimizationHint]:
        """Generate optimization hints."""
        hints = []

        # Check for missing type filter
        if not query.entity_types and query.query_type == "match":
            hints.append(OptimizationHint(
                hint_type="add_type_filter",
                description="Query scans all entity types",
                benefit=0.8,
                recommendation="Add entity_types filter to reduce scan",
            ))

        # Check for unbounded subgraph
        if query.query_type == "subgraph" and query.max_depth > 3:
            hints.append(OptimizationHint(
                hint_type="reduce_depth",
                description=f"Subgraph depth {query.max_depth} may be expensive",
                benefit=0.5,
                recommendation="Consider reducing max_depth to 3 or less",
            ))

        # Check for missing relationship filter on path
        if query.query_type == "path" and not query.relation_types:
            hints.append(OptimizationHint(
                hint_type="add_relation_filter",
                description="Path query explores all relationship types",
                benefit=0.7,
                recommendation="Add relation_types filter to constrain paths",
            ))

        # Check for high cardinality result
        if cost.estimated_rows > 10000 and query.limit > 1000:
            hints.append(OptimizationHint(
                hint_type="reduce_limit",
                description=f"Query may return {cost.estimated_rows:.0f} rows",
                benefit=0.3,
                recommendation="Consider reducing limit for faster response",
            ))

        return hints

    async def _try_rewrite(
        self,
        query: GraphQuery,
    ) -> tuple[GraphQuery, bool]:
        """
        Try to rewrite query for better performance.

        Returns (rewritten_query, was_rewritten).
        """
        rewritten = False

        # Convert simple path to subgraph if depth is 1
        if query.query_type == "path" and query.max_depth == 1:
            if query.start_entity_id and not query.end_entity_id:
                # This is really a neighbor query
                query = GraphQuery(
                    id=query.id,
                    query_type="subgraph",
                    start_entity_id=query.start_entity_id,
                    relation_types=query.relation_types,
                    max_depth=1,
                    limit=query.limit,
                )
                rewritten = True

        # Add default limit if missing
        if query.limit == 0 or query.limit > 10000:
            query.limit = 100
            rewritten = True

        return query, rewritten

    def explain(self, optimized: OptimizedQuery) -> str:
        """Generate human-readable explanation of the optimization."""
        lines = [
            "Query Optimization Report",
            "=" * 40,
            f"Query Type: {optimized.query.query_type}",
            f"Estimated Rows: {optimized.cost.estimated_rows:.0f}",
            f"IO Cost: {optimized.cost.io_cost:.2f}",
            f"CPU Cost: {optimized.cost.cpu_cost:.2f}",
            f"Total Cost: {optimized.cost.total_cost:.2f}",
        ]

        if optimized.rewritten:
            lines.append("\n[Query was rewritten for optimization]")

        if optimized.hints:
            lines.append("\nOptimization Hints:")
            for hint in optimized.hints:
                lines.append(f"  - [{hint.hint_type}] {hint.description}")
                lines.append(f"    Recommendation: {hint.recommendation}")

        return "\n".join(lines)
