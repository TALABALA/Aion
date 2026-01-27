"""
AION Dependency Graph

Graph data structure for dependency management.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Set

import structlog

logger = structlog.get_logger(__name__)


class DependencyGraph:
    """
    Directed graph for dependency tracking.

    Features:
    - Node and edge management
    - Topological sort
    - Cycle detection
    - Reachability analysis
    - Subgraph extraction
    """

    def __init__(self):
        self._nodes: Set[str] = set()
        self._edges: Dict[str, Set[str]] = defaultdict(set)  # from -> to
        self._reverse_edges: Dict[str, Set[str]] = defaultdict(set)  # to -> from
        self._node_data: Dict[str, Any] = {}

    def add_node(self, node: str, data: Optional[Any] = None) -> None:
        """Add a node to the graph."""
        self._nodes.add(node)
        if data is not None:
            self._node_data[node] = data

    def remove_node(self, node: str) -> None:
        """Remove a node and all its edges."""
        if node not in self._nodes:
            return

        # Remove outgoing edges
        for target in self._edges.get(node, set()).copy():
            self.remove_edge(node, target)

        # Remove incoming edges
        for source in self._reverse_edges.get(node, set()).copy():
            self.remove_edge(source, node)

        self._nodes.discard(node)
        self._node_data.pop(node, None)

    def add_edge(self, from_node: str, to_node: str) -> None:
        """Add an edge (from_node must be loaded before to_node)."""
        self.add_node(from_node)
        self.add_node(to_node)
        self._edges[from_node].add(to_node)
        self._reverse_edges[to_node].add(from_node)

    def remove_edge(self, from_node: str, to_node: str) -> None:
        """Remove an edge."""
        self._edges[from_node].discard(to_node)
        self._reverse_edges[to_node].discard(from_node)

    def has_edge(self, from_node: str, to_node: str) -> bool:
        """Check if edge exists."""
        return to_node in self._edges.get(from_node, set())

    def get_dependencies(self, node: str) -> Set[str]:
        """Get direct dependencies of a node."""
        return self._reverse_edges.get(node, set()).copy()

    def get_dependents(self, node: str) -> Set[str]:
        """Get nodes that depend on this node."""
        return self._edges.get(node, set()).copy()

    def get_all_dependencies(self, node: str) -> Set[str]:
        """Get all dependencies (transitive closure)."""
        visited: Set[str] = set()
        queue = deque([node])

        while queue:
            current = queue.popleft()
            for dep in self._reverse_edges.get(current, set()):
                if dep not in visited:
                    visited.add(dep)
                    queue.append(dep)

        return visited

    def get_all_dependents(self, node: str) -> Set[str]:
        """Get all dependents (transitive closure)."""
        visited: Set[str] = set()
        queue = deque([node])

        while queue:
            current = queue.popleft()
            for dependent in self._edges.get(current, set()):
                if dependent not in visited:
                    visited.add(dependent)
                    queue.append(dependent)

        return visited

    def clear(self) -> None:
        """Clear the graph."""
        self._nodes.clear()
        self._edges.clear()
        self._reverse_edges.clear()
        self._node_data.clear()

    # === Topological Sort ===

    def topological_sort(self) -> List[str]:
        """
        Perform topological sort using Kahn's algorithm.

        Returns:
            Nodes in dependency order (dependencies first)

        Raises:
            ValueError: If graph has cycles
        """
        # Calculate in-degrees
        in_degree: Dict[str, int] = {node: 0 for node in self._nodes}

        for from_node, targets in self._edges.items():
            for target in targets:
                in_degree[target] = in_degree.get(target, 0) + 1

        # Find nodes with no incoming edges
        queue = deque([node for node, degree in in_degree.items() if degree == 0])
        result: List[str] = []

        while queue:
            node = queue.popleft()
            result.append(node)

            for target in self._edges.get(node, set()):
                in_degree[target] -= 1
                if in_degree[target] == 0:
                    queue.append(target)

        if len(result) != len(self._nodes):
            raise ValueError("Graph has cycles, topological sort not possible")

        return result

    def topological_sort_reverse(self) -> List[str]:
        """
        Topological sort in reverse order (dependents first).

        Useful for shutdown order.
        """
        return list(reversed(self.topological_sort()))

    # === Cycle Detection ===

    def has_cycle(self) -> bool:
        """Check if graph has any cycles."""
        return len(self.find_cycles()) > 0

    def find_cycles(self) -> List[List[str]]:
        """
        Find all cycles in the graph.

        Returns:
            List of cycles, each cycle is a list of nodes
        """
        cycles: List[List[str]] = []
        visited: Set[str] = set()
        rec_stack: Set[str] = set()
        path: List[str] = []

        def dfs(node: str) -> None:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in self._edges.get(node, set()):
                if neighbor not in visited:
                    dfs(neighbor)
                elif neighbor in rec_stack:
                    # Found cycle
                    cycle_start = path.index(neighbor)
                    cycles.append(path[cycle_start:] + [neighbor])

            path.pop()
            rec_stack.remove(node)

        for node in self._nodes:
            if node not in visited:
                dfs(node)

        return cycles

    # === Analysis ===

    def get_roots(self) -> Set[str]:
        """Get nodes with no dependencies (entry points)."""
        return {
            node for node in self._nodes
            if not self._reverse_edges.get(node)
        }

    def get_leaves(self) -> Set[str]:
        """Get nodes with no dependents (end points)."""
        return {
            node for node in self._nodes
            if not self._edges.get(node)
        }

    def get_isolated(self) -> Set[str]:
        """Get nodes with no edges."""
        return {
            node for node in self._nodes
            if not self._edges.get(node) and not self._reverse_edges.get(node)
        }

    def is_connected(self, from_node: str, to_node: str) -> bool:
        """Check if there's a path from from_node to to_node."""
        if from_node not in self._nodes or to_node not in self._nodes:
            return False

        visited: Set[str] = set()
        queue = deque([from_node])

        while queue:
            current = queue.popleft()
            if current == to_node:
                return True

            if current in visited:
                continue

            visited.add(current)
            queue.extend(self._edges.get(current, set()))

        return False

    def get_path(self, from_node: str, to_node: str) -> Optional[List[str]]:
        """Find a path between two nodes."""
        if from_node not in self._nodes or to_node not in self._nodes:
            return None

        visited: Set[str] = set()
        queue = deque([(from_node, [from_node])])

        while queue:
            current, path = queue.popleft()

            if current == to_node:
                return path

            if current in visited:
                continue

            visited.add(current)

            for neighbor in self._edges.get(current, set()):
                if neighbor not in visited:
                    queue.append((neighbor, path + [neighbor]))

        return None

    # === Subgraph ===

    def get_subgraph(self, nodes: Set[str]) -> "DependencyGraph":
        """Extract a subgraph containing only specified nodes."""
        subgraph = DependencyGraph()

        for node in nodes:
            if node in self._nodes:
                subgraph.add_node(node, self._node_data.get(node))

        for from_node in nodes:
            for to_node in self._edges.get(from_node, set()):
                if to_node in nodes:
                    subgraph.add_edge(from_node, to_node)

        return subgraph

    def get_dependency_subgraph(self, node: str) -> "DependencyGraph":
        """Get subgraph of all dependencies of a node."""
        deps = self.get_all_dependencies(node)
        deps.add(node)
        return self.get_subgraph(deps)

    # === Serialization ===

    def to_dict(self) -> Dict[str, Any]:
        """Convert graph to dictionary."""
        return {
            "nodes": list(self._nodes),
            "edges": {
                from_node: list(targets)
                for from_node, targets in self._edges.items()
                if targets
            },
            "node_data": self._node_data.copy(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DependencyGraph":
        """Create graph from dictionary."""
        graph = cls()

        for node in data.get("nodes", []):
            node_data = data.get("node_data", {}).get(node)
            graph.add_node(node, node_data)

        for from_node, targets in data.get("edges", {}).items():
            for to_node in targets:
                graph.add_edge(from_node, to_node)

        return graph

    # === Visualization ===

    def to_dot(self) -> str:
        """Generate DOT format for visualization."""
        lines = ["digraph dependencies {"]
        lines.append("  rankdir=LR;")

        for node in self._nodes:
            label = node.replace("-", "_")
            lines.append(f'  "{label}";')

        for from_node, targets in self._edges.items():
            from_label = from_node.replace("-", "_")
            for to_node in targets:
                to_label = to_node.replace("-", "_")
                lines.append(f'  "{from_label}" -> "{to_label}";')

        lines.append("}")
        return "\n".join(lines)

    def to_mermaid(self) -> str:
        """Generate Mermaid format for visualization."""
        lines = ["graph LR"]

        for from_node, targets in self._edges.items():
            from_label = from_node.replace("-", "_")
            for to_node in targets:
                to_label = to_node.replace("-", "_")
                lines.append(f"  {from_label} --> {to_label}")

        # Add isolated nodes
        for node in self.get_isolated():
            label = node.replace("-", "_")
            lines.append(f"  {label}")

        return "\n".join(lines)

    # === Stats ===

    def __len__(self) -> int:
        return len(self._nodes)

    def __contains__(self, node: str) -> bool:
        return node in self._nodes

    def __iter__(self):
        return iter(self._nodes)

    @property
    def node_count(self) -> int:
        """Get number of nodes."""
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        """Get number of edges."""
        return sum(len(targets) for targets in self._edges.values())

    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        return {
            "nodes": self.node_count,
            "edges": self.edge_count,
            "roots": len(self.get_roots()),
            "leaves": len(self.get_leaves()),
            "isolated": len(self.get_isolated()),
            "has_cycles": self.has_cycle(),
        }
