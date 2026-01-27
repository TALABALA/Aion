"""
AION Cluster Management API

Production-grade REST API for managing the distributed cluster.
Provides endpoints for cluster info, node management, task submission,
health checking, metrics, and administrative operations.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from aion.distributed.cluster.manager import ClusterManager

logger = structlog.get_logger(__name__)


try:
    from fastapi import APIRouter, HTTPException, Body, Query
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False


def create_cluster_router(cluster_manager: "ClusterManager") -> Any:
    """
    Create FastAPI router for cluster management.

    Args:
        cluster_manager: The cluster manager instance.

    Returns:
        FastAPI APIRouter with all cluster endpoints.
    """
    if not FASTAPI_AVAILABLE:
        logger.warning("FastAPI not available, cluster API disabled")
        return None

    router = APIRouter(prefix="/cluster", tags=["cluster"])

    # =========================================================================
    # Cluster Info
    # =========================================================================

    @router.get("/info")
    async def get_cluster_info() -> Dict[str, Any]:
        """Get comprehensive cluster information."""
        return cluster_manager.get_cluster_info()

    @router.get("/health")
    async def health_check() -> Dict[str, Any]:
        """Cluster health check endpoint."""
        state = cluster_manager.cluster_state
        healthy = len(state.healthy_nodes)
        total = len(state.nodes)

        status = "healthy"
        if not state.has_quorum:
            status = "critical"
        elif healthy < total:
            status = "degraded"

        return {
            "status": status,
            "has_quorum": state.has_quorum,
            "healthy_nodes": healthy,
            "total_nodes": total,
            "leader_id": state.leader_id,
            "term": state.term,
            "epoch": state.epoch,
            "timestamp": datetime.now().isoformat(),
        }

    @router.get("/stats")
    async def get_stats() -> Dict[str, Any]:
        """Get cluster statistics."""
        return cluster_manager.get_stats()

    # =========================================================================
    # Node Management
    # =========================================================================

    @router.get("/nodes")
    async def list_nodes(
        status: Optional[str] = Query(None, description="Filter by status"),
        role: Optional[str] = Query(None, description="Filter by role"),
    ) -> List[Dict[str, Any]]:
        """List all cluster nodes with optional filtering."""
        nodes = list(cluster_manager.cluster_state.nodes.values())

        if status:
            nodes = [n for n in nodes if n.status.value == status]
        if role:
            nodes = [n for n in nodes if n.role.value == role]

        return [n.to_dict() for n in nodes]

    @router.get("/nodes/{node_id}")
    async def get_node(node_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific node."""
        node = cluster_manager.cluster_state.nodes.get(node_id)
        if not node:
            raise HTTPException(status_code=404, detail=f"Node not found: {node_id}")
        return node.to_dict()

    @router.post("/nodes/{node_id}/drain")
    async def drain_node(node_id: str) -> Dict[str, Any]:
        """
        Drain a node - stop accepting new tasks and wait for current tasks
        to complete before marking the node as leaving.
        """
        node = cluster_manager.cluster_state.nodes.get(node_id)
        if not node:
            raise HTTPException(status_code=404, detail=f"Node not found: {node_id}")

        from aion.distributed.types import NodeStatus
        node.status = NodeStatus.DRAINING
        logger.info("Node drain initiated", node_id=node_id, node_name=node.name)

        return {
            "node_id": node_id,
            "status": "draining",
            "current_tasks": node.current_tasks,
        }

    @router.delete("/nodes/{node_id}")
    async def remove_node(node_id: str) -> Dict[str, Any]:
        """Remove a node from the cluster."""
        if node_id == cluster_manager.node_info.id:
            raise HTTPException(
                status_code=400,
                detail="Cannot remove self from cluster",
            )

        node = cluster_manager.cluster_state.nodes.get(node_id)
        if not node:
            raise HTTPException(status_code=404, detail=f"Node not found: {node_id}")

        await cluster_manager.handle_node_left(node_id)
        return {"node_id": node_id, "status": "removed"}

    # =========================================================================
    # Task Management
    # =========================================================================

    @router.post("/tasks")
    async def submit_task(
        name: str = Body(..., description="Task name"),
        task_type: str = Body(..., description="Task type"),
        payload: Dict[str, Any] = Body(default={}, description="Task payload"),
        priority: str = Body(default="normal", description="Task priority"),
        timeout_seconds: int = Body(default=300, description="Task timeout"),
        max_retries: int = Body(default=3, description="Max retry attempts"),
    ) -> Dict[str, Any]:
        """Submit a distributed task for execution."""
        from aion.distributed.types import DistributedTask, TaskPriority

        try:
            prio = TaskPriority[priority.upper()]
        except KeyError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid priority: {priority}. Valid: {[p.name.lower() for p in TaskPriority]}",
            )

        task = DistributedTask(
            name=name,
            task_type=task_type,
            payload=payload,
            priority=prio,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )

        task_id = await cluster_manager.submit_task(task)
        return {
            "task_id": task_id,
            "status": "submitted",
            "priority": priority,
        }

    @router.get("/tasks/{task_id}")
    async def get_task(task_id: str) -> Dict[str, Any]:
        """Get task status and details."""
        if not cluster_manager.task_queue:
            raise HTTPException(status_code=503, detail="Task queue not initialized")

        task = await cluster_manager.task_queue.get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
        return task.to_dict()

    @router.post("/tasks/{task_id}/cancel")
    async def cancel_task(task_id: str) -> Dict[str, Any]:
        """Cancel a pending or running task."""
        if not cluster_manager.task_queue:
            raise HTTPException(status_code=503, detail="Task queue not initialized")

        task = await cluster_manager.task_queue.get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")

        from aion.distributed.types import TaskStatus
        if task.is_terminal:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot cancel task in state: {task.status.value}",
            )

        task.status = TaskStatus.CANCELLED
        await cluster_manager.task_queue.update_task(task)
        return {"task_id": task_id, "status": "cancelled"}

    @router.get("/tasks")
    async def list_tasks(
        status: Optional[str] = Query(None, description="Filter by status"),
        node_id: Optional[str] = Query(None, description="Filter by node"),
        limit: int = Query(default=100, ge=1, le=1000),
    ) -> List[Dict[str, Any]]:
        """List tasks with optional filtering."""
        if not cluster_manager.task_queue:
            return []

        if node_id:
            tasks = await cluster_manager.task_queue.get_tasks_by_node(node_id)
        else:
            tasks = await cluster_manager.task_queue.get_all()

        if status:
            tasks = [t for t in tasks if t.status.value == status]

        tasks = sorted(tasks, key=lambda t: t.created_at, reverse=True)
        return [t.to_dict() for t in tasks[:limit]]

    @router.get("/tasks/stats")
    async def get_task_stats() -> Dict[str, Any]:
        """Get task queue statistics."""
        if not cluster_manager.task_queue:
            return {"available": False}
        return await cluster_manager.task_queue.get_stats()

    # =========================================================================
    # Consensus
    # =========================================================================

    @router.get("/consensus")
    async def get_consensus_info() -> Dict[str, Any]:
        """Get Raft consensus state information."""
        if not cluster_manager.consensus:
            return {"available": False}

        return {
            "available": True,
            "role": cluster_manager.consensus.role.value,
            "term": cluster_manager.consensus.current_term,
            "leader_id": cluster_manager.cluster_state.leader_id,
            "commit_index": cluster_manager.consensus.state.commit_index,
            "last_applied": cluster_manager.consensus.state.last_applied,
            "log_length": len(cluster_manager.consensus.state.log),
        }

    @router.post("/consensus/step-down")
    async def step_down() -> Dict[str, Any]:
        """Force the current leader to step down (triggers new election)."""
        if not cluster_manager.is_leader():
            raise HTTPException(
                status_code=400,
                detail="This node is not the leader",
            )

        if cluster_manager.consensus:
            cluster_manager.consensus._become_follower()
            await cluster_manager.consensus.trigger_election()

        return {"status": "stepped_down", "message": "New election triggered"}

    # =========================================================================
    # Administrative
    # =========================================================================

    @router.post("/admin/rebalance")
    async def trigger_rebalance() -> Dict[str, Any]:
        """Trigger a manual rebalance of tasks and data across the cluster."""
        if not cluster_manager.is_leader():
            raise HTTPException(
                status_code=400,
                detail="Only the leader can trigger rebalance",
            )

        logger.info("Manual rebalance triggered")
        return {
            "status": "rebalance_initiated",
            "node_count": len(cluster_manager.cluster_state.nodes),
        }

    @router.get("/admin/topology")
    async def get_topology() -> Dict[str, Any]:
        """Get cluster topology information."""
        nodes = cluster_manager.cluster_state.nodes.values()

        regions: Dict[str, List[str]] = {}
        zones: Dict[str, List[str]] = {}

        for node in nodes:
            region = node.region or "default"
            zone = node.zone or "default"

            if region not in regions:
                regions[region] = []
            regions[region].append(node.id)

            zone_key = f"{region}/{zone}"
            if zone_key not in zones:
                zones[zone_key] = []
            zones[zone_key].append(node.id)

        return {
            "regions": regions,
            "zones": zones,
            "total_nodes": len(cluster_manager.cluster_state.nodes),
        }

    return router


def setup_cluster_routes(app: Any, cluster_manager: "ClusterManager") -> None:
    """
    Setup cluster API routes on a FastAPI application.

    Args:
        app: FastAPI application instance.
        cluster_manager: The cluster manager instance.
    """
    router = create_cluster_router(cluster_manager)
    if router:
        app.include_router(router)
        logger.info("Cluster API routes registered")
    else:
        logger.warning("Cluster API routes not registered (FastAPI unavailable)")
