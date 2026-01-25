"""
AION Planning Repository

Persistence for the planning graph system including:
- Execution plans with full graph structure
- Plan checkpoints for rollback
- Execution history and analytics
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional
import uuid

import structlog

from aion.persistence.repositories.base import BaseRepository, QueryOptions
from aion.persistence.database import DatabaseManager
from aion.persistence.backends.redis_cache import CacheManager

logger = structlog.get_logger(__name__)

# Import planning types
try:
    from aion.systems.planning.graph import (
        ExecutionPlan,
        PlanNode,
        PlanEdge,
        ExecutionCheckpoint,
        NodeStatus,
        NodeType,
    )
except ImportError:
    # Fallback types
    from enum import Enum, auto
    from dataclasses import dataclass, field

    class NodeStatus(Enum):
        PENDING = auto()
        READY = auto()
        RUNNING = auto()
        COMPLETED = auto()
        FAILED = auto()
        SKIPPED = auto()
        CANCELLED = auto()

    class NodeType(str, Enum):
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
        id: str
        name: str
        node_type: NodeType
        action: Optional[str] = None
        params: dict = field(default_factory=dict)
        status: NodeStatus = NodeStatus.PENDING
        result: Any = None
        error: Optional[str] = None
        started_at: Optional[datetime] = None
        completed_at: Optional[datetime] = None
        retries: int = 0
        max_retries: int = 3
        timeout: float = 300.0
        metadata: dict = field(default_factory=dict)

    @dataclass
    class PlanEdge:
        source: str
        target: str
        condition: Optional[str] = None
        weight: float = 1.0
        metadata: dict = field(default_factory=dict)

    @dataclass
    class ExecutionPlan:
        id: str
        name: str
        description: str
        nodes: dict
        edges: list
        created_at: datetime
        status: NodeStatus = NodeStatus.PENDING
        started_at: Optional[datetime] = None
        completed_at: Optional[datetime] = None
        checkpoints: list = field(default_factory=list)
        metadata: dict = field(default_factory=dict)

    @dataclass
    class ExecutionCheckpoint:
        id: str
        plan_id: str
        timestamp: datetime
        node_states: dict
        graph_state: dict
        context: dict


class PlanningRepository(BaseRepository[ExecutionPlan]):
    """
    Repository for planning graph persistence.

    Features:
    - Full plan graph serialization
    - Node and edge persistence
    - Checkpoint management
    - Execution history
    """

    def __init__(
        self,
        db: DatabaseManager,
        cache: Optional[CacheManager] = None,
    ):
        super().__init__(db, cache)
        self._table_name = "plans"
        self._soft_delete_column = None  # No soft delete for plans

    def _serialize(self, plan: ExecutionPlan) -> dict[str, Any]:
        """Serialize ExecutionPlan to database row."""
        # Serialize nodes
        nodes_dict = {}
        for node_id, node in plan.nodes.items():
            if hasattr(node, 'to_dict'):
                nodes_dict[node_id] = node.to_dict()
            else:
                nodes_dict[node_id] = {
                    "id": node.id,
                    "name": node.name,
                    "node_type": node.node_type.value if hasattr(node.node_type, 'value') else str(node.node_type),
                    "action": node.action,
                    "params": node.params,
                    "status": node.status.name if hasattr(node.status, 'name') else str(node.status),
                    "result": str(node.result)[:500] if node.result else None,
                    "error": node.error,
                    "started_at": node.started_at.isoformat() if node.started_at else None,
                    "completed_at": node.completed_at.isoformat() if node.completed_at else None,
                    "retries": node.retries,
                    "metadata": node.metadata,
                }

        # Serialize edges
        edges_list = []
        for edge in plan.edges:
            if hasattr(edge, 'to_dict'):
                edges_list.append(edge.to_dict())
            else:
                edges_list.append({
                    "source": edge.source,
                    "target": edge.target,
                    "condition": edge.condition,
                    "weight": edge.weight,
                    "metadata": getattr(edge, 'metadata', {}),
                })

        return {
            "id": plan.id,
            "name": plan.name,
            "description": plan.description,
            "status": plan.status.name if hasattr(plan.status, 'name') else str(plan.status),
            "nodes": self._to_json(nodes_dict),
            "edges": self._to_json(edges_list),
            "created_at": self._from_datetime(plan.created_at),
            "started_at": self._from_datetime(plan.started_at),
            "completed_at": self._from_datetime(plan.completed_at),
            "metadata": self._to_json(plan.metadata),
            "created_by": plan.metadata.get("created_by"),
        }

    def _deserialize(self, row: dict[str, Any]) -> ExecutionPlan:
        """Deserialize database row to ExecutionPlan."""
        # Deserialize nodes
        nodes_data = self._from_json(row.get("nodes")) or {}
        nodes = {}
        for node_id, node_data in nodes_data.items():
            nodes[node_id] = PlanNode(
                id=node_data["id"],
                name=node_data["name"],
                node_type=NodeType(node_data["node_type"]),
                action=node_data.get("action"),
                params=node_data.get("params", {}),
                status=NodeStatus[node_data.get("status", "PENDING")],
                result=node_data.get("result"),
                error=node_data.get("error"),
                started_at=self._to_datetime(node_data.get("started_at")),
                completed_at=self._to_datetime(node_data.get("completed_at")),
                retries=node_data.get("retries", 0),
                metadata=node_data.get("metadata", {}),
            )

        # Deserialize edges
        edges_data = self._from_json(row.get("edges")) or []
        edges = [
            PlanEdge(
                source=e["source"],
                target=e["target"],
                condition=e.get("condition"),
                weight=e.get("weight", 1.0),
                metadata=e.get("metadata", {}),
            )
            for e in edges_data
        ]

        metadata = self._from_json(row.get("metadata")) or {}

        return ExecutionPlan(
            id=row["id"],
            name=row["name"],
            description=row.get("description", ""),
            nodes=nodes,
            edges=edges,
            created_at=self._to_datetime(row.get("created_at")) or datetime.now(),
            status=NodeStatus[row.get("status", "PENDING")],
            started_at=self._to_datetime(row.get("started_at")),
            completed_at=self._to_datetime(row.get("completed_at")),
            checkpoints=[],  # Loaded separately
            metadata=metadata,
        )

    # === Checkpoint Operations ===

    async def save_checkpoint(
        self,
        checkpoint: ExecutionCheckpoint,
    ) -> str:
        """Save an execution checkpoint."""
        checkpoint_id = checkpoint.id or str(uuid.uuid4())

        query = """
            INSERT INTO plan_checkpoints (id, plan_id, checkpoint_name, node_states, context_snapshot, results_snapshot, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """

        # Extract results from node states
        results = {
            node_id: state.get("result")
            for node_id, state in checkpoint.node_states.items()
            if state.get("result")
        }

        await self.db.execute(query, (
            checkpoint_id,
            checkpoint.plan_id,
            checkpoint.graph_state.get("name", ""),
            self._to_json(checkpoint.node_states),
            self._to_json(checkpoint.context),
            self._to_json(results),
            checkpoint.timestamp.isoformat(),
        ))

        return checkpoint_id

    async def get_checkpoints(
        self,
        plan_id: str,
    ) -> list[ExecutionCheckpoint]:
        """Get all checkpoints for a plan."""
        query = """
            SELECT * FROM plan_checkpoints
            WHERE plan_id = ?
            ORDER BY created_at ASC
        """

        rows = await self.db.fetch_all(query, (plan_id,))

        return [
            ExecutionCheckpoint(
                id=row["id"],
                plan_id=row["plan_id"],
                timestamp=self._to_datetime(row["created_at"]) or datetime.now(),
                node_states=self._from_json(row.get("node_states")) or {},
                graph_state={"name": row.get("checkpoint_name", "")},
                context=self._from_json(row.get("context_snapshot")) or {},
            )
            for row in rows
        ]

    async def get_latest_checkpoint(
        self,
        plan_id: str,
    ) -> Optional[ExecutionCheckpoint]:
        """Get the most recent checkpoint for a plan."""
        query = """
            SELECT * FROM plan_checkpoints
            WHERE plan_id = ?
            ORDER BY created_at DESC
            LIMIT 1
        """

        row = await self.db.fetch_one(query, (plan_id,))

        if not row:
            return None

        return ExecutionCheckpoint(
            id=row["id"],
            plan_id=row["plan_id"],
            timestamp=self._to_datetime(row["created_at"]) or datetime.now(),
            node_states=self._from_json(row.get("node_states")) or {},
            graph_state={"name": row.get("checkpoint_name", "")},
            context=self._from_json(row.get("context_snapshot")) or {},
        )

    async def delete_checkpoints(
        self,
        plan_id: str,
        keep_last: int = 0,
    ) -> int:
        """Delete checkpoints for a plan, optionally keeping the last N."""
        if keep_last > 0:
            # Get IDs to keep
            keep_query = """
                SELECT id FROM plan_checkpoints
                WHERE plan_id = ?
                ORDER BY created_at DESC
                LIMIT ?
            """
            keep_rows = await self.db.fetch_all(keep_query, (plan_id, keep_last))
            keep_ids = [r["id"] for r in keep_rows]

            if keep_ids:
                placeholders = ", ".join(["?" for _ in keep_ids])
                query = f"""
                    DELETE FROM plan_checkpoints
                    WHERE plan_id = ? AND id NOT IN ({placeholders})
                """
                await self.db.execute(query, (plan_id, *keep_ids))
            return 0
        else:
            query = "DELETE FROM plan_checkpoints WHERE plan_id = ?"
            await self.db.execute(query, (plan_id,))
            return 0

    # === Query Helpers ===

    async def find_by_status(
        self,
        status: NodeStatus,
        options: Optional[QueryOptions] = None,
    ) -> list[ExecutionPlan]:
        """Find plans by status."""
        status_name = status.name if hasattr(status, 'name') else str(status)
        return await self.find_where("status = ?", (status_name,), options=options)

    async def find_running(
        self,
        options: Optional[QueryOptions] = None,
    ) -> list[ExecutionPlan]:
        """Find currently running plans."""
        return await self.find_by_status(NodeStatus.RUNNING, options)

    async def find_failed(
        self,
        options: Optional[QueryOptions] = None,
    ) -> list[ExecutionPlan]:
        """Find failed plans."""
        return await self.find_by_status(NodeStatus.FAILED, options)

    async def find_by_creator(
        self,
        created_by: str,
        options: Optional[QueryOptions] = None,
    ) -> list[ExecutionPlan]:
        """Find plans by creator."""
        return await self.find_where("created_by = ?", (created_by,), options=options)

    async def update_status(
        self,
        plan_id: str,
        status: NodeStatus,
        error: Optional[str] = None,
    ) -> bool:
        """Update plan status."""
        fields = {
            "status": status.name if hasattr(status, 'name') else str(status),
        }

        if status == NodeStatus.RUNNING and not await self._has_started(plan_id):
            fields["started_at"] = datetime.now().isoformat()

        if status in (NodeStatus.COMPLETED, NodeStatus.FAILED, NodeStatus.CANCELLED):
            fields["completed_at"] = datetime.now().isoformat()

        if error:
            fields["error"] = error

        return await self.update_fields(plan_id, fields)

    async def _has_started(self, plan_id: str) -> bool:
        """Check if plan has already started."""
        query = "SELECT started_at FROM plans WHERE id = ?"
        row = await self.db.fetch_one(query, (plan_id,))
        return row is not None and row.get("started_at") is not None

    async def update_node_status(
        self,
        plan_id: str,
        node_id: str,
        status: NodeStatus,
        result: Any = None,
        error: Optional[str] = None,
    ) -> bool:
        """Update a specific node's status within a plan."""
        # Get current plan
        plan = await self.get(plan_id)
        if not plan or node_id not in plan.nodes:
            return False

        # Update node
        node = plan.nodes[node_id]
        node.status = status

        if result is not None:
            node.result = result
        if error:
            node.error = error

        if status == NodeStatus.RUNNING:
            node.started_at = datetime.now()
        elif status in (NodeStatus.COMPLETED, NodeStatus.FAILED):
            node.completed_at = datetime.now()

        # Save updated plan
        return await self.update(plan_id, plan)

    async def get_execution_stats(
        self,
        plan_id: str,
    ) -> dict[str, Any]:
        """Get execution statistics for a plan."""
        plan = await self.get(plan_id)
        if not plan:
            return {}

        total_nodes = len(plan.nodes)
        completed = sum(1 for n in plan.nodes.values() if n.status == NodeStatus.COMPLETED)
        failed = sum(1 for n in plan.nodes.values() if n.status == NodeStatus.FAILED)
        running = sum(1 for n in plan.nodes.values() if n.status == NodeStatus.RUNNING)
        pending = sum(1 for n in plan.nodes.values() if n.status == NodeStatus.PENDING)

        total_duration = 0
        node_durations = []
        for node in plan.nodes.values():
            if node.started_at and node.completed_at:
                duration = (node.completed_at - node.started_at).total_seconds() * 1000
                node_durations.append(duration)
                total_duration += duration

        return {
            "plan_id": plan_id,
            "status": plan.status.name if hasattr(plan.status, 'name') else str(plan.status),
            "nodes": {
                "total": total_nodes,
                "completed": completed,
                "failed": failed,
                "running": running,
                "pending": pending,
            },
            "progress": completed / total_nodes if total_nodes > 0 else 0,
            "duration_ms": total_duration,
            "avg_node_duration_ms": sum(node_durations) / len(node_durations) if node_durations else 0,
            "started_at": self._from_datetime(plan.started_at),
            "completed_at": self._from_datetime(plan.completed_at),
        }

    async def get_global_stats(self) -> dict[str, Any]:
        """Get global planning statistics."""
        query = """
            SELECT
                COUNT(*) as total,
                COUNT(CASE WHEN status = 'COMPLETED' THEN 1 END) as completed,
                COUNT(CASE WHEN status = 'FAILED' THEN 1 END) as failed,
                COUNT(CASE WHEN status = 'RUNNING' THEN 1 END) as running,
                COUNT(CASE WHEN status = 'PENDING' THEN 1 END) as pending
            FROM plans
        """

        stats = await self.db.fetch_one(query)

        checkpoint_query = "SELECT COUNT(*) as count FROM plan_checkpoints"
        checkpoints = await self.db.fetch_one(checkpoint_query)

        return {
            "plans": {
                "total": stats["total"] if stats else 0,
                "completed": stats["completed"] if stats else 0,
                "failed": stats["failed"] if stats else 0,
                "running": stats["running"] if stats else 0,
                "pending": stats["pending"] if stats else 0,
                "success_rate": (
                    stats["completed"] / (stats["completed"] + stats["failed"])
                    if stats and (stats["completed"] + stats["failed"]) > 0 else 0
                ),
            },
            "checkpoints": {
                "total": checkpoints["count"] if checkpoints else 0,
            },
        }
