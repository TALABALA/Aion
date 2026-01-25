"""
Service Topology and Dependency Mapping

SOTA features:
- Automatic service discovery from traces
- Dependency graph construction
- Health propagation
- Impact analysis
- Visualization export (Graphviz, D3)
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from aion.observability.types import Span, Trace, HealthStatus

logger = logging.getLogger(__name__)


# =============================================================================
# Types
# =============================================================================

class DependencyType(Enum):
    """Type of service dependency."""
    SYNC_HTTP = "sync_http"
    ASYNC_HTTP = "async_http"
    GRPC = "grpc"
    MESSAGE_QUEUE = "message_queue"
    DATABASE = "database"
    CACHE = "cache"
    EXTERNAL_API = "external_api"
    INTERNAL = "internal"


class ServiceHealth(Enum):
    """Health status of a service."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ServiceMetrics:
    """Aggregated metrics for a service."""
    request_count: int = 0
    error_count: int = 0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    throughput_rps: float = 0.0
    error_rate: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_count": self.request_count,
            "error_count": self.error_count,
            "latency_p50_ms": self.latency_p50_ms,
            "latency_p95_ms": self.latency_p95_ms,
            "latency_p99_ms": self.latency_p99_ms,
            "throughput_rps": self.throughput_rps,
            "error_rate": self.error_rate,
        }


@dataclass
class ServiceNode:
    """A node in the service topology graph."""
    name: str
    service_type: str = "service"
    version: str = ""
    namespace: str = "default"
    cluster: str = "default"

    # Health and metrics
    health: ServiceHealth = ServiceHealth.UNKNOWN
    metrics: ServiceMetrics = field(default_factory=ServiceMetrics)

    # Metadata
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)

    # Discovery info
    first_seen: datetime = field(default_factory=datetime.utcnow)
    last_seen: datetime = field(default_factory=datetime.utcnow)
    instance_count: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.service_type,
            "version": self.version,
            "namespace": self.namespace,
            "cluster": self.cluster,
            "health": self.health.value,
            "metrics": self.metrics.to_dict(),
            "labels": self.labels,
            "instance_count": self.instance_count,
            "first_seen": self.first_seen.isoformat(),
            "last_seen": self.last_seen.isoformat(),
        }


@dataclass
class ServiceEdge:
    """An edge (dependency) in the service topology."""
    source: str  # Source service name
    target: str  # Target service name
    dependency_type: DependencyType = DependencyType.SYNC_HTTP

    # Metrics
    call_count: int = 0
    error_count: int = 0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0

    # Protocol info
    protocol: str = "http"
    port: int = 0
    path: str = ""

    # Temporal
    first_seen: datetime = field(default_factory=datetime.utcnow)
    last_seen: datetime = field(default_factory=datetime.utcnow)

    @property
    def error_rate(self) -> float:
        if self.call_count == 0:
            return 0.0
        return self.error_count / self.call_count

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "type": self.dependency_type.value,
            "call_count": self.call_count,
            "error_count": self.error_count,
            "error_rate": self.error_rate,
            "latency_p50_ms": self.latency_p50_ms,
            "latency_p95_ms": self.latency_p95_ms,
            "protocol": self.protocol,
            "port": self.port,
            "first_seen": self.first_seen.isoformat(),
            "last_seen": self.last_seen.isoformat(),
        }


@dataclass
class ServiceTopology:
    """The complete service topology graph."""
    nodes: Dict[str, ServiceNode] = field(default_factory=dict)
    edges: Dict[Tuple[str, str], ServiceEdge] = field(default_factory=dict)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def add_node(self, node: ServiceNode) -> None:
        """Add or update a node."""
        if node.name in self.nodes:
            existing = self.nodes[node.name]
            existing.last_seen = datetime.utcnow()
            existing.instance_count = max(existing.instance_count, node.instance_count)
            # Merge labels
            existing.labels.update(node.labels)
        else:
            self.nodes[node.name] = node

    def add_edge(self, edge: ServiceEdge) -> None:
        """Add or update an edge."""
        key = (edge.source, edge.target)
        if key in self.edges:
            existing = self.edges[key]
            existing.call_count += edge.call_count
            existing.error_count += edge.error_count
            existing.last_seen = datetime.utcnow()
        else:
            self.edges[key] = edge

    def get_node(self, name: str) -> Optional[ServiceNode]:
        """Get a node by name."""
        return self.nodes.get(name)

    def get_dependencies(self, service: str) -> List[ServiceNode]:
        """Get services that a service depends on."""
        deps = []
        for (source, target), edge in self.edges.items():
            if source == service and target in self.nodes:
                deps.append(self.nodes[target])
        return deps

    def get_dependents(self, service: str) -> List[ServiceNode]:
        """Get services that depend on a service."""
        deps = []
        for (source, target), edge in self.edges.items():
            if target == service and source in self.nodes:
                deps.append(self.nodes[source])
        return deps

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": [e.to_dict() for e in self.edges.values()],
            "updated_at": self.updated_at.isoformat(),
            "stats": {
                "node_count": len(self.nodes),
                "edge_count": len(self.edges),
            },
        }

    def to_graphviz(self) -> str:
        """Export to Graphviz DOT format."""
        lines = ["digraph ServiceTopology {"]
        lines.append("  rankdir=LR;")
        lines.append("  node [shape=box, style=filled];")

        # Color nodes by health
        health_colors = {
            ServiceHealth.HEALTHY: "#90EE90",
            ServiceHealth.DEGRADED: "#FFD700",
            ServiceHealth.UNHEALTHY: "#FF6B6B",
            ServiceHealth.UNKNOWN: "#D3D3D3",
        }

        for name, node in self.nodes.items():
            color = health_colors.get(node.health, "#D3D3D3")
            label = f"{name}\\n{node.service_type}"
            lines.append(f'  "{name}" [label="{label}", fillcolor="{color}"];')

        for (source, target), edge in self.edges.items():
            label = f"{edge.call_count} calls"
            style = "solid" if edge.error_rate < 0.01 else "dashed"
            lines.append(f'  "{source}" -> "{target}" [label="{label}", style={style}];')

        lines.append("}")
        return "\n".join(lines)

    def to_d3_json(self) -> Dict[str, Any]:
        """Export to D3.js force graph format."""
        nodes = []
        for name, node in self.nodes.items():
            nodes.append({
                "id": name,
                "group": node.service_type,
                "health": node.health.value,
                "metrics": node.metrics.to_dict(),
            })

        links = []
        for (source, target), edge in self.edges.items():
            links.append({
                "source": source,
                "target": target,
                "value": edge.call_count,
                "type": edge.dependency_type.value,
            })

        return {"nodes": nodes, "links": links}


# =============================================================================
# Topology Builder
# =============================================================================

class TopologyBuilder:
    """
    Builds service topology from traces.

    Automatically discovers services and dependencies.
    """

    def __init__(
        self,
        max_traces: int = 10000,
        edge_ttl_hours: float = 24.0,
    ):
        self.max_traces = max_traces
        self.edge_ttl_hours = edge_ttl_hours

        self._topology = ServiceTopology()
        self._latencies: Dict[Tuple[str, str], List[float]] = defaultdict(list)
        self._processed_traces: Set[str] = set()

    def process_trace(self, trace: Trace) -> None:
        """Process a trace to extract topology information."""
        if trace.trace_id in self._processed_traces:
            return

        self._processed_traces.add(trace.trace_id)

        # Limit processed traces
        if len(self._processed_traces) > self.max_traces:
            oldest = list(self._processed_traces)[:len(self._processed_traces) // 2]
            self._processed_traces = set(list(self._processed_traces)[len(self._processed_traces) // 2:])

        # Extract services and dependencies
        for span in trace.spans:
            # Add service node
            service_name = span.service_name
            service_type = self._infer_service_type(span)

            node = ServiceNode(
                name=service_name,
                service_type=service_type,
                labels=dict(span.attributes),
            )
            self._topology.add_node(node)

            # Find parent span to create edge
            if span.parent_span_id:
                parent_span = self._find_span(trace, span.parent_span_id)
                if parent_span and parent_span.service_name != service_name:
                    # Cross-service call
                    dep_type = self._infer_dependency_type(span)

                    edge = ServiceEdge(
                        source=parent_span.service_name,
                        target=service_name,
                        dependency_type=dep_type,
                        call_count=1,
                        error_count=1 if span.status.is_error else 0,
                        protocol=span.attributes.get("http.method", "unknown"),
                        path=span.attributes.get("http.url", span.operation_name),
                    )
                    self._topology.add_edge(edge)

                    # Track latency
                    if span.duration_ms:
                        key = (parent_span.service_name, service_name)
                        self._latencies[key].append(span.duration_ms)
                        # Keep only recent latencies
                        if len(self._latencies[key]) > 1000:
                            self._latencies[key] = self._latencies[key][-500:]

        self._topology.updated_at = datetime.utcnow()

    def process_span(self, span: Span, parent_service: Optional[str] = None) -> None:
        """Process a single span."""
        service_name = span.service_name
        service_type = self._infer_service_type(span)

        node = ServiceNode(
            name=service_name,
            service_type=service_type,
        )
        self._topology.add_node(node)

        if parent_service and parent_service != service_name:
            dep_type = self._infer_dependency_type(span)

            edge = ServiceEdge(
                source=parent_service,
                target=service_name,
                dependency_type=dep_type,
                call_count=1,
                error_count=1 if span.status.is_error else 0,
            )
            self._topology.add_edge(edge)

            if span.duration_ms:
                key = (parent_service, service_name)
                self._latencies[key].append(span.duration_ms)

    def _find_span(self, trace: Trace, span_id: str) -> Optional[Span]:
        """Find a span by ID in a trace."""
        for span in trace.spans:
            if span.span_id == span_id:
                return span
        return None

    def _infer_service_type(self, span: Span) -> str:
        """Infer service type from span attributes."""
        attrs = span.attributes

        # Database
        if attrs.get("db.system"):
            return f"database:{attrs.get('db.system')}"

        # Cache
        if "redis" in span.operation_name.lower() or "cache" in span.operation_name.lower():
            return "cache"

        # Message queue
        if attrs.get("messaging.system"):
            return f"queue:{attrs.get('messaging.system')}"

        # External API
        if attrs.get("http.url", "").startswith("https://"):
            host = attrs.get("http.host", "")
            if not host.endswith(".internal") and not host.endswith(".local"):
                return "external_api"

        return "service"

    def _infer_dependency_type(self, span: Span) -> DependencyType:
        """Infer dependency type from span."""
        attrs = span.attributes

        if attrs.get("db.system"):
            return DependencyType.DATABASE

        if attrs.get("messaging.system"):
            return DependencyType.MESSAGE_QUEUE

        if "redis" in span.operation_name.lower() or "cache" in attrs.get("db.system", ""):
            return DependencyType.CACHE

        if attrs.get("rpc.system") == "grpc":
            return DependencyType.GRPC

        if attrs.get("http.url", "").startswith("https://"):
            return DependencyType.EXTERNAL_API

        return DependencyType.SYNC_HTTP

    def calculate_metrics(self) -> None:
        """Calculate aggregated metrics for edges."""
        import numpy as np

        for key, latencies in self._latencies.items():
            if key not in self._topology.edges:
                continue

            edge = self._topology.edges[key]

            if latencies:
                edge.latency_p50_ms = float(np.percentile(latencies, 50))
                edge.latency_p95_ms = float(np.percentile(latencies, 95))

    def get_topology(self) -> ServiceTopology:
        """Get the current topology."""
        self.calculate_metrics()
        return self._topology

    def cleanup_stale(self) -> int:
        """Remove stale nodes and edges."""
        now = datetime.utcnow()
        cutoff = now - timedelta(hours=self.edge_ttl_hours)

        removed = 0

        # Remove stale edges
        stale_edges = [
            key for key, edge in self._topology.edges.items()
            if edge.last_seen < cutoff
        ]
        for key in stale_edges:
            del self._topology.edges[key]
            removed += 1

        # Remove orphan nodes (no edges)
        connected = set()
        for source, target in self._topology.edges.keys():
            connected.add(source)
            connected.add(target)

        stale_nodes = [
            name for name in self._topology.nodes
            if name not in connected and self._topology.nodes[name].last_seen < cutoff
        ]
        for name in stale_nodes:
            del self._topology.nodes[name]
            removed += 1

        return removed


# =============================================================================
# Topology Analyzer
# =============================================================================

class TopologyAnalyzer:
    """
    Analyzes service topology for insights.

    Features:
    - Critical path detection
    - Single points of failure
    - Impact analysis
    - Health propagation
    """

    def __init__(self, topology: ServiceTopology):
        self.topology = topology

    def find_critical_path(self, start: str, end: str) -> List[str]:
        """Find critical path between two services."""
        if start not in self.topology.nodes or end not in self.topology.nodes:
            return []

        # BFS to find path
        visited = set()
        queue = [(start, [start])]

        while queue:
            current, path = queue.pop(0)
            if current == end:
                return path

            if current in visited:
                continue
            visited.add(current)

            for (source, target) in self.topology.edges:
                if source == current and target not in visited:
                    queue.append((target, path + [target]))

        return []

    def find_single_points_of_failure(self) -> List[ServiceNode]:
        """Find services that are single points of failure."""
        spof = []

        for name, node in self.topology.nodes.items():
            # Get dependents
            dependents = self.topology.get_dependents(name)

            if len(dependents) > 2:
                # Many services depend on this
                # Check if it's the only path
                for dependent in dependents:
                    alt_paths = self._count_paths(dependent.name, name)
                    if alt_paths <= 1:
                        if node not in spof:
                            spof.append(node)
                        break

        return spof

    def _count_paths(self, source: str, target: str, max_depth: int = 5) -> int:
        """Count number of paths between two services."""
        count = 0

        def dfs(current: str, visited: Set[str], depth: int) -> int:
            nonlocal count
            if depth > max_depth:
                return 0
            if current == target:
                return 1

            paths = 0
            for (s, t) in self.topology.edges:
                if s == current and t not in visited:
                    visited.add(t)
                    paths += dfs(t, visited, depth + 1)
                    visited.remove(t)

            return paths

        return dfs(source, {source}, 0)

    def analyze_impact(self, service: str) -> Dict[str, Any]:
        """Analyze the impact of a service failure."""
        if service not in self.topology.nodes:
            return {"error": "Service not found"}

        # Find all affected services (transitive dependents)
        affected = set()

        def find_affected(current: str, visited: Set[str]) -> None:
            for (source, target) in self.topology.edges:
                if target == current and source not in visited:
                    affected.add(source)
                    visited.add(source)
                    find_affected(source, visited)

        find_affected(service, {service})

        # Calculate impact metrics
        node = self.topology.nodes[service]

        return {
            "service": service,
            "directly_affected": [
                dep.name for dep in self.topology.get_dependents(service)
            ],
            "transitively_affected": list(affected),
            "total_affected_count": len(affected),
            "service_metrics": node.metrics.to_dict(),
            "criticality_score": len(affected) / max(len(self.topology.nodes), 1),
        }

    def propagate_health(self) -> None:
        """Propagate health status based on dependencies."""
        # Topological sort
        sorted_services = self._topological_sort()

        for name in sorted_services:
            node = self.topology.nodes.get(name)
            if not node:
                continue

            # Check dependencies
            dependencies = self.topology.get_dependencies(name)

            if not dependencies:
                # Leaf service - use own health
                continue

            # Propagate worst dependency health
            dep_healths = [dep.health for dep in dependencies]

            if ServiceHealth.UNHEALTHY in dep_healths:
                # If any dependency is unhealthy, mark as degraded at best
                if node.health == ServiceHealth.HEALTHY:
                    node.health = ServiceHealth.DEGRADED

            elif ServiceHealth.DEGRADED in dep_healths:
                # If dependencies degraded and we're healthy, we're at risk
                if node.health == ServiceHealth.HEALTHY:
                    node.health = ServiceHealth.DEGRADED

    def _topological_sort(self) -> List[str]:
        """Topological sort of services."""
        visited = set()
        result = []

        def visit(name: str) -> None:
            if name in visited:
                return
            visited.add(name)

            for (source, target) in self.topology.edges:
                if source == name:
                    visit(target)

            result.append(name)

        for name in self.topology.nodes:
            visit(name)

        return result[::-1]

    def get_service_ranks(self) -> List[Tuple[str, float]]:
        """Rank services by importance (PageRank-like)."""
        damping = 0.85
        iterations = 20
        n = len(self.topology.nodes)

        if n == 0:
            return []

        # Initialize ranks
        ranks = {name: 1.0 / n for name in self.topology.nodes}

        # Build adjacency
        inbound: Dict[str, List[str]] = defaultdict(list)
        outbound: Dict[str, int] = defaultdict(int)

        for (source, target) in self.topology.edges:
            inbound[target].append(source)
            outbound[source] += 1

        # Iterate
        for _ in range(iterations):
            new_ranks = {}
            for name in self.topology.nodes:
                rank = (1 - damping) / n
                for source in inbound[name]:
                    if outbound[source] > 0:
                        rank += damping * ranks[source] / outbound[source]
                new_ranks[name] = rank
            ranks = new_ranks

        # Sort by rank
        return sorted(ranks.items(), key=lambda x: x[1], reverse=True)

    def detect_cycles(self) -> List[List[str]]:
        """Detect circular dependencies."""
        cycles = []
        visited = set()
        rec_stack = set()

        def dfs(node: str, path: List[str]) -> None:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for (source, target) in self.topology.edges:
                if source == node:
                    if target in rec_stack:
                        # Found cycle
                        cycle_start = path.index(target)
                        cycle = path[cycle_start:] + [target]
                        if cycle not in cycles:
                            cycles.append(cycle)
                    elif target not in visited:
                        dfs(target, path.copy())

            rec_stack.remove(node)

        for name in self.topology.nodes:
            if name not in visited:
                dfs(name, [])

        return cycles

    def get_stats(self) -> Dict[str, Any]:
        """Get topology analysis statistics."""
        return {
            "total_services": len(self.topology.nodes),
            "total_edges": len(self.topology.edges),
            "single_points_of_failure": [
                spof.name for spof in self.find_single_points_of_failure()
            ],
            "cycles": self.detect_cycles(),
            "service_ranks": self.get_service_ranks()[:10],
            "health_summary": {
                health.value: sum(
                    1 for n in self.topology.nodes.values()
                    if n.health == health
                )
                for health in ServiceHealth
            },
        }
