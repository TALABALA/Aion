"""
AION Planning Graph System

A deterministic planning system using directed acyclic graphs (DAGs) for:
- Multi-step workflow planning
- Dependency resolution and parallel execution
- State machine-based execution tracking
- Checkpoint/rollback capabilities
- Reproducible execution paths
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Optional, TypeVar, Generic

import networkx as nx
import structlog

logger = structlog.get_logger(__name__)

T = TypeVar("T")


class NodeStatus(Enum):
    """Execution status of a plan node."""
    PENDING = auto()
    READY = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    SKIPPED = auto()
    CANCELLED = auto()


class NodeType(str, Enum):
    """Types of plan nodes."""
    START = "start"
    END = "end"
    ACTION = "action"
    CONDITION = "condition"
    PARALLEL = "parallel"
    JOIN = "join"
    SUBGRAPH = "subgraph"
    CHECKPOINT = "checkpoint"


@dataclass
class PlanNode:
    """A node in the planning graph."""
    id: str
    name: str
    node_type: NodeType
    action: Optional[str] = None
    params: dict[str, Any] = field(default_factory=dict)
    status: NodeStatus = NodeStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    retries: int = 0
    max_retries: int = 3
    timeout: float = 300.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "node_type": self.node_type.value,
            "action": self.action,
            "params": self.params,
            "status": self.status.name,
            "result": str(self.result)[:500] if self.result else None,
            "error": self.error,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "retries": self.retries,
            "metadata": self.metadata,
        }

    def duration_ms(self) -> float:
        """Get execution duration in milliseconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds() * 1000
        return 0.0


@dataclass
class PlanEdge:
    """An edge in the planning graph."""
    source: str
    target: str
    condition: Optional[str] = None
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source": self.source,
            "target": self.target,
            "condition": self.condition,
            "weight": self.weight,
            "metadata": self.metadata,
        }


@dataclass
class ExecutionCheckpoint:
    """A checkpoint in plan execution for rollback support."""
    id: str
    plan_id: str
    timestamp: datetime
    node_states: dict[str, dict[str, Any]]
    graph_state: dict[str, Any]
    context: dict[str, Any]


@dataclass
class ExecutionPlan:
    """A complete execution plan."""
    id: str
    name: str
    description: str
    nodes: dict[str, PlanNode]
    edges: list[PlanEdge]
    created_at: datetime
    status: NodeStatus = NodeStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    checkpoints: list[ExecutionCheckpoint] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "nodes": {nid: n.to_dict() for nid, n in self.nodes.items()},
            "edges": [e.to_dict() for e in self.edges],
            "created_at": self.created_at.isoformat(),
            "status": self.status.name,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "metadata": self.metadata,
        }

    def get_hash(self) -> str:
        """Get a deterministic hash of the plan structure."""
        plan_str = json.dumps(
            {
                "name": self.name,
                "nodes": sorted(self.nodes.keys()),
                "edges": [(e.source, e.target) for e in sorted(self.edges, key=lambda x: (x.source, x.target))],
            },
            sort_keys=True,
        )
        return hashlib.sha256(plan_str.encode()).hexdigest()[:16]


class PlanningGraph:
    """
    AION Planning Graph Engine

    Manages creation, validation, optimization, and execution of
    directed acyclic graphs representing execution plans.
    """

    def __init__(
        self,
        max_depth: int = 20,
        max_parallel: int = 10,
        checkpoint_interval: int = 1,
        enable_caching: bool = True,
    ):
        self.max_depth = max_depth
        self.max_parallel = max_parallel
        self.checkpoint_interval = checkpoint_interval
        self.enable_caching = enable_caching

        # Active plans
        self._plans: dict[str, ExecutionPlan] = {}
        self._graphs: dict[str, nx.DiGraph] = {}

        # Action registry
        self._actions: dict[str, Callable] = {}

        # Plan cache
        self._cache: dict[str, ExecutionPlan] = {}

        # Statistics
        self._stats = {
            "plans_created": 0,
            "plans_executed": 0,
            "plans_succeeded": 0,
            "plans_failed": 0,
            "total_nodes_executed": 0,
            "cache_hits": 0,
        }

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the planning graph system."""
        if self._initialized:
            return

        logger.info("Initializing Planning Graph System")
        self._initialized = True

    async def shutdown(self) -> None:
        """Shutdown the planning graph system."""
        logger.info("Shutting down Planning Graph System")
        self._plans.clear()
        self._graphs.clear()
        self._initialized = False

    def register_action(
        self,
        name: str,
        handler: Callable,
        description: str = "",
    ) -> None:
        """
        Register an action handler.

        Args:
            name: Action name
            handler: Async callable that executes the action
            description: Human-readable description
        """
        self._actions[name] = handler
        logger.debug("Registered action", action=name)

    def create_plan(
        self,
        name: str,
        description: str = "",
        metadata: Optional[dict[str, Any]] = None,
    ) -> ExecutionPlan:
        """
        Create a new execution plan.

        Args:
            name: Plan name
            description: Plan description
            metadata: Additional metadata

        Returns:
            New ExecutionPlan
        """
        plan_id = str(uuid.uuid4())

        # Create start and end nodes
        start_node = PlanNode(
            id=f"{plan_id}_start",
            name="Start",
            node_type=NodeType.START,
        )
        end_node = PlanNode(
            id=f"{plan_id}_end",
            name="End",
            node_type=NodeType.END,
        )

        plan = ExecutionPlan(
            id=plan_id,
            name=name,
            description=description,
            nodes={
                start_node.id: start_node,
                end_node.id: end_node,
            },
            edges=[],
            created_at=datetime.now(),
            metadata=metadata or {},
        )

        # Create NetworkX graph
        graph = nx.DiGraph()
        graph.add_node(start_node.id, node=start_node)
        graph.add_node(end_node.id, node=end_node)

        self._plans[plan_id] = plan
        self._graphs[plan_id] = graph
        self._stats["plans_created"] += 1

        logger.info("Created plan", plan_id=plan_id, name=name)
        return plan

    def add_node(
        self,
        plan_id: str,
        name: str,
        action: str,
        node_type: NodeType = NodeType.ACTION,
        params: Optional[dict[str, Any]] = None,
        timeout: float = 300.0,
        max_retries: int = 3,
        metadata: Optional[dict[str, Any]] = None,
    ) -> PlanNode:
        """
        Add a node to a plan.

        Args:
            plan_id: Plan ID
            name: Node name
            action: Action to execute
            node_type: Type of node
            params: Action parameters
            timeout: Execution timeout
            max_retries: Maximum retries on failure
            metadata: Additional metadata

        Returns:
            Created PlanNode
        """
        if plan_id not in self._plans:
            raise KeyError(f"Plan not found: {plan_id}")

        plan = self._plans[plan_id]
        graph = self._graphs[plan_id]

        node_id = f"{plan_id}_{name.lower().replace(' ', '_')}_{len(plan.nodes)}"
        node = PlanNode(
            id=node_id,
            name=name,
            node_type=node_type,
            action=action,
            params=params or {},
            timeout=timeout,
            max_retries=max_retries,
            metadata=metadata or {},
        )

        plan.nodes[node_id] = node
        graph.add_node(node_id, node=node)

        logger.debug("Added node", plan_id=plan_id, node_id=node_id, action=action)
        return node

    def add_edge(
        self,
        plan_id: str,
        source: str,
        target: str,
        condition: Optional[str] = None,
        weight: float = 1.0,
    ) -> PlanEdge:
        """
        Add an edge between nodes.

        Args:
            plan_id: Plan ID
            source: Source node ID
            target: Target node ID
            condition: Optional condition expression
            weight: Edge weight for optimization

        Returns:
            Created PlanEdge
        """
        if plan_id not in self._plans:
            raise KeyError(f"Plan not found: {plan_id}")

        plan = self._plans[plan_id]
        graph = self._graphs[plan_id]

        if source not in plan.nodes:
            raise KeyError(f"Source node not found: {source}")
        if target not in plan.nodes:
            raise KeyError(f"Target node not found: {target}")

        edge = PlanEdge(
            source=source,
            target=target,
            condition=condition,
            weight=weight,
        )

        plan.edges.append(edge)
        graph.add_edge(source, target, edge=edge, weight=weight)

        # Validate no cycles
        if not nx.is_directed_acyclic_graph(graph):
            # Remove the edge and raise error
            plan.edges.remove(edge)
            graph.remove_edge(source, target)
            raise ValueError(f"Adding edge {source} -> {target} would create a cycle")

        logger.debug("Added edge", plan_id=plan_id, source=source, target=target)
        return edge

    def validate_plan(self, plan_id: str) -> tuple[bool, list[str]]:
        """
        Validate a plan for execution.

        Args:
            plan_id: Plan ID

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        if plan_id not in self._plans:
            return False, [f"Plan not found: {plan_id}"]

        plan = self._plans[plan_id]
        graph = self._graphs[plan_id]
        errors = []

        # Check for DAG
        if not nx.is_directed_acyclic_graph(graph):
            errors.append("Plan contains cycles")

        # Check depth
        try:
            longest_path = nx.dag_longest_path_length(graph)
            if longest_path > self.max_depth:
                errors.append(f"Plan depth {longest_path} exceeds maximum {self.max_depth}")
        except nx.NetworkXError:
            errors.append("Cannot compute plan depth")

        # Check connectivity
        if not nx.is_weakly_connected(graph):
            errors.append("Plan is not fully connected")

        # Check for start and end nodes
        start_nodes = [n for n, d in plan.nodes.items() if d.node_type == NodeType.START]
        end_nodes = [n for n, d in plan.nodes.items() if d.node_type == NodeType.END]

        if len(start_nodes) != 1:
            errors.append(f"Plan must have exactly one start node (found {len(start_nodes)})")
        if len(end_nodes) != 1:
            errors.append(f"Plan must have exactly one end node (found {len(end_nodes)})")

        # Check action nodes have valid actions
        for node_id, node in plan.nodes.items():
            if node.node_type == NodeType.ACTION:
                if node.action and node.action not in self._actions:
                    errors.append(f"Node {node_id} has unregistered action: {node.action}")

        return len(errors) == 0, errors

    def get_execution_order(self, plan_id: str) -> list[list[str]]:
        """
        Get the execution order with parallel groups.

        Args:
            plan_id: Plan ID

        Returns:
            List of parallel execution groups (each group can run in parallel)
        """
        if plan_id not in self._graphs:
            raise KeyError(f"Plan not found: {plan_id}")

        graph = self._graphs[plan_id]

        # Topological generations give us natural parallelization
        try:
            generations = list(nx.topological_generations(graph))
            return [list(gen) for gen in generations]
        except nx.NetworkXError as e:
            raise ValueError(f"Cannot determine execution order: {e}")

    def optimize_plan(self, plan_id: str) -> dict[str, Any]:
        """
        Optimize a plan for execution.

        Optimizations include:
        - Parallel execution grouping
        - Critical path identification
        - Resource allocation suggestions

        Args:
            plan_id: Plan ID

        Returns:
            Optimization results
        """
        if plan_id not in self._graphs:
            raise KeyError(f"Plan not found: {plan_id}")

        graph = self._graphs[plan_id]
        plan = self._plans[plan_id]

        # Get execution order
        execution_order = self.get_execution_order(plan_id)

        # Find critical path
        try:
            critical_path = nx.dag_longest_path(graph, weight="weight")
        except nx.NetworkXError:
            critical_path = []

        # Calculate parallelization potential
        max_parallel = max(len(group) for group in execution_order) if execution_order else 0

        # Estimate execution time
        total_time = sum(
            plan.nodes[node_id].timeout
            for node_id in critical_path
            if node_id in plan.nodes
        )

        return {
            "execution_order": execution_order,
            "critical_path": critical_path,
            "max_parallelization": max_parallel,
            "estimated_time_seconds": total_time,
            "total_nodes": len(plan.nodes),
            "total_edges": len(plan.edges),
        }

    async def execute_plan(
        self,
        plan_id: str,
        context: Optional[dict[str, Any]] = None,
        on_node_start: Optional[Callable] = None,
        on_node_complete: Optional[Callable] = None,
    ) -> dict[str, Any]:
        """
        Execute a plan.

        Args:
            plan_id: Plan ID
            context: Execution context
            on_node_start: Callback when node starts
            on_node_complete: Callback when node completes

        Returns:
            Execution results
        """
        if plan_id not in self._plans:
            raise KeyError(f"Plan not found: {plan_id}")

        plan = self._plans[plan_id]
        context = context or {}

        # Validate
        is_valid, errors = self.validate_plan(plan_id)
        if not is_valid:
            raise ValueError(f"Invalid plan: {errors}")

        logger.info("Executing plan", plan_id=plan_id, name=plan.name)

        plan.status = NodeStatus.RUNNING
        plan.started_at = datetime.now()
        self._stats["plans_executed"] += 1

        try:
            # Get execution order
            execution_order = self.get_execution_order(plan_id)
            results = {}

            for generation_idx, node_ids in enumerate(execution_order):
                # Execute nodes in parallel
                tasks = []

                for node_id in node_ids:
                    node = plan.nodes[node_id]

                    # Skip non-action nodes
                    if node.node_type in (NodeType.START, NodeType.END):
                        node.status = NodeStatus.COMPLETED
                        continue

                    task = self._execute_node(
                        node, context, on_node_start, on_node_complete
                    )
                    tasks.append((node_id, task))

                # Wait for all parallel tasks
                if tasks:
                    parallel_results = await asyncio.gather(
                        *[t for _, t in tasks],
                        return_exceptions=True,
                    )

                    for (node_id, _), result in zip(tasks, parallel_results):
                        if isinstance(result, Exception):
                            plan.nodes[node_id].status = NodeStatus.FAILED
                            plan.nodes[node_id].error = str(result)
                            raise result
                        results[node_id] = result

                # Create checkpoint
                if generation_idx % self.checkpoint_interval == 0:
                    self._create_checkpoint(plan, context)

            plan.status = NodeStatus.COMPLETED
            plan.completed_at = datetime.now()
            self._stats["plans_succeeded"] += 1

            logger.info(
                "Plan execution completed",
                plan_id=plan_id,
                duration_ms=(plan.completed_at - plan.started_at).total_seconds() * 1000,
            )

            return {
                "status": "completed",
                "results": results,
                "duration_ms": (plan.completed_at - plan.started_at).total_seconds() * 1000,
            }

        except Exception as e:
            plan.status = NodeStatus.FAILED
            plan.completed_at = datetime.now()
            self._stats["plans_failed"] += 1

            logger.error("Plan execution failed", plan_id=plan_id, error=str(e))
            raise

    async def _execute_node(
        self,
        node: PlanNode,
        context: dict[str, Any],
        on_start: Optional[Callable] = None,
        on_complete: Optional[Callable] = None,
    ) -> Any:
        """Execute a single node."""
        node.status = NodeStatus.RUNNING
        node.started_at = datetime.now()
        self._stats["total_nodes_executed"] += 1

        if on_start:
            await on_start(node) if asyncio.iscoroutinefunction(on_start) else on_start(node)

        try:
            # Get action handler
            if node.action and node.action in self._actions:
                handler = self._actions[node.action]

                # Execute with timeout
                result = await asyncio.wait_for(
                    handler(node.params, context),
                    timeout=node.timeout,
                )
            else:
                # No action, just pass through
                result = node.params

            node.status = NodeStatus.COMPLETED
            node.result = result
            node.completed_at = datetime.now()

            if on_complete:
                await on_complete(node) if asyncio.iscoroutinefunction(on_complete) else on_complete(node)

            return result

        except asyncio.TimeoutError:
            node.status = NodeStatus.FAILED
            node.error = f"Timeout after {node.timeout}s"
            node.completed_at = datetime.now()
            raise

        except Exception as e:
            # Retry logic
            if node.retries < node.max_retries:
                node.retries += 1
                logger.warning(
                    "Node failed, retrying",
                    node_id=node.id,
                    retry=node.retries,
                    error=str(e),
                )
                return await self._execute_node(node, context, on_start, on_complete)

            node.status = NodeStatus.FAILED
            node.error = str(e)
            node.completed_at = datetime.now()
            raise

    def _create_checkpoint(
        self,
        plan: ExecutionPlan,
        context: dict[str, Any],
    ) -> ExecutionCheckpoint:
        """Create an execution checkpoint."""
        checkpoint = ExecutionCheckpoint(
            id=str(uuid.uuid4()),
            plan_id=plan.id,
            timestamp=datetime.now(),
            node_states={
                nid: node.to_dict() for nid, node in plan.nodes.items()
            },
            graph_state={
                "edges": [e.to_dict() for e in plan.edges],
            },
            context=context.copy(),
        )
        plan.checkpoints.append(checkpoint)
        return checkpoint

    def rollback_to_checkpoint(
        self,
        plan_id: str,
        checkpoint_id: str,
    ) -> ExecutionPlan:
        """
        Rollback a plan to a checkpoint.

        Args:
            plan_id: Plan ID
            checkpoint_id: Checkpoint ID

        Returns:
            Restored ExecutionPlan
        """
        if plan_id not in self._plans:
            raise KeyError(f"Plan not found: {plan_id}")

        plan = self._plans[plan_id]

        checkpoint = next(
            (c for c in plan.checkpoints if c.id == checkpoint_id),
            None,
        )
        if not checkpoint:
            raise KeyError(f"Checkpoint not found: {checkpoint_id}")

        # Restore node states
        for node_id, state in checkpoint.node_states.items():
            if node_id in plan.nodes:
                node = plan.nodes[node_id]
                node.status = NodeStatus[state["status"]]
                node.result = state.get("result")
                node.error = state.get("error")
                node.retries = state.get("retries", 0)

        logger.info(
            "Rolled back to checkpoint",
            plan_id=plan_id,
            checkpoint_id=checkpoint_id,
        )

        return plan

    def get_plan(self, plan_id: str) -> Optional[ExecutionPlan]:
        """Get a plan by ID."""
        return self._plans.get(plan_id)

    def list_plans(self) -> list[ExecutionPlan]:
        """List all plans."""
        return list(self._plans.values())

    def delete_plan(self, plan_id: str) -> bool:
        """Delete a plan."""
        if plan_id in self._plans:
            del self._plans[plan_id]
            del self._graphs[plan_id]
            return True
        return False

    def get_stats(self) -> dict[str, Any]:
        """Get planning system statistics."""
        return self._stats.copy()

    def export_plan(self, plan_id: str) -> dict[str, Any]:
        """Export a plan to JSON-serializable format."""
        if plan_id not in self._plans:
            raise KeyError(f"Plan not found: {plan_id}")
        return self._plans[plan_id].to_dict()

    def import_plan(self, plan_data: dict[str, Any]) -> ExecutionPlan:
        """Import a plan from JSON data."""
        plan_id = plan_data.get("id", str(uuid.uuid4()))

        # Recreate nodes
        nodes = {}
        for node_id, node_data in plan_data.get("nodes", {}).items():
            nodes[node_id] = PlanNode(
                id=node_id,
                name=node_data["name"],
                node_type=NodeType(node_data["node_type"]),
                action=node_data.get("action"),
                params=node_data.get("params", {}),
                metadata=node_data.get("metadata", {}),
            )

        # Recreate edges
        edges = [
            PlanEdge(
                source=e["source"],
                target=e["target"],
                condition=e.get("condition"),
                weight=e.get("weight", 1.0),
            )
            for e in plan_data.get("edges", [])
        ]

        plan = ExecutionPlan(
            id=plan_id,
            name=plan_data["name"],
            description=plan_data.get("description", ""),
            nodes=nodes,
            edges=edges,
            created_at=datetime.now(),
            metadata=plan_data.get("metadata", {}),
        )

        # Create graph
        graph = nx.DiGraph()
        for node_id, node in nodes.items():
            graph.add_node(node_id, node=node)
        for edge in edges:
            graph.add_edge(edge.source, edge.target, edge=edge, weight=edge.weight)

        self._plans[plan_id] = plan
        self._graphs[plan_id] = graph

        return plan
