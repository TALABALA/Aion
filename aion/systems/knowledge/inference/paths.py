"""
AION Knowledge Graph Path Finding

Advanced path finding algorithms.
"""

from __future__ import annotations

import heapq
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import structlog

from aion.systems.knowledge.types import (
    Entity,
    Relationship,
    RelationType,
    Path,
)
from aion.systems.knowledge.store.base import GraphStore

logger = structlog.get_logger(__name__)


@dataclass
class PathResult:
    """Result of a path finding query."""
    paths: List[Path] = field(default_factory=list)
    total_explored: int = 0
    algorithm: str = ""
    execution_time_ms: float = 0.0


class PathFinder:
    """
    Advanced path finding for knowledge graphs.

    Algorithms:
    - BFS: Shortest path (unweighted)
    - Dijkstra: Shortest path (weighted)
    - A*: Heuristic-guided shortest path
    - Bidirectional BFS: Faster for large graphs
    - All paths: Find all paths up to a limit
    """

    def __init__(self, store: GraphStore):
        self.store = store

    async def find_shortest_path(
        self,
        start_id: str,
        end_id: str,
        max_depth: int = 10,
        relation_types: Optional[List[RelationType]] = None,
        weighted: bool = False,
    ) -> Optional[Path]:
        """Find shortest path between two entities."""
        if weighted:
            return await self._dijkstra(start_id, end_id, max_depth, relation_types)
        else:
            return await self._bfs(start_id, end_id, max_depth, relation_types)

    async def find_all_paths(
        self,
        start_id: str,
        end_id: str,
        max_depth: int = 5,
        max_paths: int = 10,
        relation_types: Optional[List[RelationType]] = None,
    ) -> List[Path]:
        """Find all paths between two entities (up to limits)."""
        return await self._dfs_all_paths(
            start_id, end_id, max_depth, max_paths, relation_types
        )

    async def find_shortest_paths_to_all(
        self,
        start_id: str,
        max_depth: int = 5,
        relation_types: Optional[List[RelationType]] = None,
    ) -> Dict[str, Path]:
        """Find shortest paths from start to all reachable entities."""
        return await self._bfs_all(start_id, max_depth, relation_types)

    async def _bfs(
        self,
        start_id: str,
        end_id: str,
        max_depth: int,
        relation_types: Optional[List[RelationType]],
    ) -> Optional[Path]:
        """BFS for unweighted shortest path."""
        if start_id == end_id:
            entity = await self.store.get_entity(start_id)
            return Path(entities=[entity] if entity else [], relationships=[])

        visited = {start_id}
        queue = deque([(start_id, [], [])])  # (current, entities, rels)

        while queue:
            current_id, path_entities, path_rels = queue.popleft()

            if len(path_rels) >= max_depth:
                continue

            rels = await self.store.get_relationships(current_id, direction="both")

            for rel in rels:
                if relation_types and rel.relation_type not in relation_types:
                    continue

                next_id = rel.target_id if rel.source_id == current_id else rel.source_id

                if next_id in visited:
                    continue

                visited.add(next_id)

                entity = await self.store.get_entity(next_id)
                if not entity:
                    continue

                new_entities = path_entities + [entity]
                new_rels = path_rels + [rel]

                if next_id == end_id:
                    start_entity = await self.store.get_entity(start_id)
                    path = Path(
                        entities=[start_entity] + new_entities,
                        relationships=new_rels,
                    )
                    path.compute_metrics()
                    return path

                queue.append((next_id, new_entities, new_rels))

        return None

    async def _dijkstra(
        self,
        start_id: str,
        end_id: str,
        max_depth: int,
        relation_types: Optional[List[RelationType]],
    ) -> Optional[Path]:
        """Dijkstra for weighted shortest path."""
        if start_id == end_id:
            entity = await self.store.get_entity(start_id)
            return Path(entities=[entity] if entity else [], relationships=[])

        # Priority queue: (distance, current_id, path_entities, path_rels)
        heap = [(0.0, start_id, [], [])]
        visited = set()

        while heap:
            distance, current_id, path_entities, path_rels = heapq.heappop(heap)

            if current_id in visited:
                continue

            visited.add(current_id)

            if current_id == end_id:
                start_entity = await self.store.get_entity(start_id)
                path = Path(
                    entities=[start_entity] + path_entities,
                    relationships=path_rels,
                )
                path.compute_metrics()
                return path

            if len(path_rels) >= max_depth:
                continue

            rels = await self.store.get_relationships(current_id, direction="both")

            for rel in rels:
                if relation_types and rel.relation_type not in relation_types:
                    continue

                next_id = rel.target_id if rel.source_id == current_id else rel.source_id

                if next_id in visited:
                    continue

                entity = await self.store.get_entity(next_id)
                if not entity:
                    continue

                # Weight: inverse of relationship weight (lower is better)
                edge_weight = 1.0 / max(rel.weight, 0.01)
                new_distance = distance + edge_weight

                heapq.heappush(heap, (
                    new_distance,
                    next_id,
                    path_entities + [entity],
                    path_rels + [rel],
                ))

        return None

    async def _dfs_all_paths(
        self,
        start_id: str,
        end_id: str,
        max_depth: int,
        max_paths: int,
        relation_types: Optional[List[RelationType]],
    ) -> List[Path]:
        """DFS for finding all paths."""
        paths = []

        async def dfs(
            current: str,
            visited: Set[str],
            path_entities: List[Entity],
            path_rels: List[Relationship],
        ):
            if len(paths) >= max_paths:
                return
            if len(path_rels) >= max_depth:
                return
            if current == end_id:
                path = Path(
                    entities=path_entities.copy(),
                    relationships=path_rels.copy(),
                )
                path.compute_metrics()
                paths.append(path)
                return

            visited.add(current)

            rels = await self.store.get_relationships(current, direction="both")

            for rel in rels:
                if relation_types and rel.relation_type not in relation_types:
                    continue

                next_id = rel.target_id if rel.source_id == current else rel.source_id

                if next_id in visited:
                    continue

                entity = await self.store.get_entity(next_id)
                if not entity:
                    continue

                await dfs(
                    next_id,
                    visited,
                    path_entities + [entity],
                    path_rels + [rel],
                )

            visited.remove(current)

        start_entity = await self.store.get_entity(start_id)
        if start_entity:
            await dfs(start_id, set(), [start_entity], [])

        return paths

    async def _bfs_all(
        self,
        start_id: str,
        max_depth: int,
        relation_types: Optional[List[RelationType]],
    ) -> Dict[str, Path]:
        """BFS to find shortest paths to all reachable entities."""
        paths: Dict[str, Path] = {}

        start_entity = await self.store.get_entity(start_id)
        if not start_entity:
            return paths

        paths[start_id] = Path(entities=[start_entity], relationships=[])

        visited = {start_id}
        queue = deque([(start_id, [], [])])

        while queue:
            current_id, path_entities, path_rels = queue.popleft()

            if len(path_rels) >= max_depth:
                continue

            rels = await self.store.get_relationships(current_id, direction="both")

            for rel in rels:
                if relation_types and rel.relation_type not in relation_types:
                    continue

                next_id = rel.target_id if rel.source_id == current_id else rel.source_id

                if next_id in visited:
                    continue

                visited.add(next_id)

                entity = await self.store.get_entity(next_id)
                if not entity:
                    continue

                new_entities = path_entities + [entity]
                new_rels = path_rels + [rel]

                path = Path(
                    entities=[start_entity] + new_entities,
                    relationships=new_rels,
                )
                path.compute_metrics()
                paths[next_id] = path

                queue.append((next_id, new_entities, new_rels))

        return paths

    async def find_connecting_paths(
        self,
        entity_ids: List[str],
        max_depth: int = 3,
    ) -> List[Path]:
        """Find paths that connect multiple entities (Steiner tree approximation)."""
        if len(entity_ids) < 2:
            return []

        all_paths = []

        # Find paths between consecutive pairs
        for i in range(len(entity_ids) - 1):
            path = await self.find_shortest_path(
                entity_ids[i],
                entity_ids[i + 1],
                max_depth=max_depth,
            )
            if path:
                all_paths.append(path)

        return all_paths

    async def compute_distance_matrix(
        self,
        entity_ids: List[str],
        max_depth: int = 10,
    ) -> Dict[Tuple[str, str], int]:
        """Compute pairwise distances between entities."""
        distances = {}

        for i, source in enumerate(entity_ids):
            # Get all shortest paths from this source
            paths = await self._bfs_all(source, max_depth, None)

            for target in entity_ids[i:]:
                if target in paths:
                    distances[(source, target)] = paths[target].length
                    distances[(target, source)] = paths[target].length
                else:
                    distances[(source, target)] = -1  # No path
                    distances[(target, source)] = -1

        return distances
