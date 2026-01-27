"""
AION Cluster Manager

Production-grade cluster management implementing:
- Full node lifecycle management (join, leave, failure detection)
- Leader-coordinated task distribution with capability-aware scheduling
- Background heartbeat, health check, and task processing loops
- Event-driven architecture with pluggable handlers
- Integration with Raft consensus, RPC communication, and distributed task queue
"""

from __future__ import annotations

import asyncio
import time
import uuid
from datetime import datetime
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Coroutine,
    Dict,
    List,
    Optional,
    Set,
)

import structlog

from aion.distributed.types import (
    ClusterMetrics,
    ClusterState,
    DistributedTask,
    HeartbeatMessage,
    NodeCapability,
    NodeInfo,
    NodeRole,
    NodeStatus,
    TaskStatus,
    TaskType,
)

if TYPE_CHECKING:
    from aion.core.kernel import AIONKernel

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Type aliases for event callbacks
# ---------------------------------------------------------------------------
EventHandler = Callable[..., Coroutine[Any, Any, None]]


class ClusterManager:
    """
    Central coordinator for the AION distributed cluster.

    Responsibilities:
    - Node lifecycle: join, leave, failure detection & removal
    - Leader election coordination via Raft consensus
    - Capability-aware task distribution to cluster nodes
    - Background loops: heartbeat, health checks, task processing
    - Cluster-wide event propagation
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, kernel: AIONKernel, config: Dict[str, Any]) -> None:
        self._kernel = kernel
        self._config = config

        # Cluster identity
        self._node_id: str = config.get("node_id", str(uuid.uuid4()))
        self._cluster_name: str = config.get("cluster_name", "aion-cluster")

        # Build local node info
        self._local_node = NodeInfo(
            id=self._node_id,
            name=config.get("node_name", f"node-{self._node_id[:8]}"),
            host=config.get("host", "0.0.0.0"),
            port=config.get("port", 5000),
            grpc_port=config.get("grpc_port", 5001),
            role=NodeRole.FOLLOWER,
            status=NodeStatus.STARTING,
            capabilities=set(config.get("capabilities", [
                NodeCapability.COMPUTE.value,
                NodeCapability.MEMORY.value,
                NodeCapability.TOOLS.value,
            ])),
            max_concurrent_tasks=config.get("max_tasks", 10),
        )

        # Cluster state
        self._state = ClusterState(
            name=self._cluster_name,
            min_nodes=config.get("min_nodes", 1),
            replication_factor=config.get("replication_factor", 3),
        )
        self._state.nodes[self._node_id] = self._local_node

        # Sub-components (lazily initialised in start())
        self._consensus: Optional[Any] = None
        self._rpc_client: Optional[Any] = None
        self._rpc_server: Optional[Any] = None
        self._task_queue: Optional[Any] = None
        self._discovery: Optional[Any] = None
        self._health_checker: Optional[Any] = None

        # Background tasks
        self._background_tasks: List[asyncio.Task[None]] = []
        self._running = False
        self._started_at: Optional[datetime] = None

        # Intervals (seconds)
        self._heartbeat_interval: float = config.get("heartbeat_interval", 1.0)
        self._health_check_interval: float = config.get("health_check_interval", 5.0)
        self._task_process_interval: float = config.get("task_process_interval", 0.5)

        # Event handlers
        self._event_handlers: Dict[str, List[EventHandler]] = {
            "node_joined": [],
            "node_left": [],
            "leader_changed": [],
            "task_completed": [],
        }

        # Metrics counters
        self._tasks_submitted: int = 0
        self._tasks_completed: int = 0
        self._tasks_failed: int = 0
        self._nodes_joined: int = 0
        self._nodes_left: int = 0

        # Task handler registry  (task_type -> handler)
        self._task_handlers: Dict[str, Callable[..., Coroutine[Any, Any, Any]]] = {
            TaskType.TOOL_EXECUTION.value: self._handle_tool_task,
            TaskType.MEMORY_OPERATION.value: self._handle_memory_task,
            TaskType.AGENT_OPERATION.value: self._handle_agent_task,
            TaskType.PLANNING_OPERATION.value: self._handle_planning_task,
        }

        logger.info(
            "cluster_manager.init",
            node_id=self._node_id,
            cluster=self._cluster_name,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def node_id(self) -> str:
        return self._node_id

    @property
    def local_node(self) -> NodeInfo:
        return self._local_node

    @property
    def state(self) -> ClusterState:
        return self._state

    @property
    def running(self) -> bool:
        return self._running

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the cluster manager and all sub-components."""
        if self._running:
            logger.warning("cluster_manager.already_running")
            return

        logger.info("cluster_manager.starting", node_id=self._node_id)

        # --- Initialise sub-components ---
        try:
            from aion.distributed.consensus.raft import RaftConsensus
            self._consensus = RaftConsensus(self)
        except ImportError:
            logger.warning("cluster_manager.raft_unavailable")

        try:
            from aion.distributed.communication.rpc import RPCClient
            self._rpc_client = RPCClient(self._config)
        except ImportError:
            logger.warning("cluster_manager.rpc_client_unavailable")

        try:
            from aion.distributed.communication.server import RPCServer
            self._rpc_server = RPCServer(self._config, self)
        except ImportError:
            logger.warning("cluster_manager.rpc_server_unavailable")

        try:
            from aion.distributed.tasks.queue import DistributedTaskQueue
            self._task_queue = DistributedTaskQueue(self._config)
        except ImportError:
            logger.warning("cluster_manager.task_queue_unavailable")

        try:
            from aion.distributed.cluster.discovery import NodeDiscovery
            self._discovery = NodeDiscovery(self._config)
        except ImportError:
            logger.warning("cluster_manager.discovery_unavailable")

        try:
            from aion.distributed.cluster.health import HealthChecker
            self._health_checker = HealthChecker(self)
        except ImportError:
            logger.warning("cluster_manager.health_checker_unavailable")

        # Start sub-components that have their own lifecycle
        if self._rpc_server is not None:
            await self._rpc_server.start()
        if self._discovery is not None:
            await self._discovery.start()
        if self._consensus is not None:
            await self._consensus.start()

        # Mark local node healthy
        self._local_node.status = NodeStatus.HEALTHY
        self._local_node.joined_at = datetime.now()
        self._running = True
        self._started_at = datetime.now()

        # Discover and join existing cluster nodes
        await self._discover_and_join()

        # Launch background loops
        self._background_tasks.append(
            asyncio.create_task(self._heartbeat_loop())
        )
        self._background_tasks.append(
            asyncio.create_task(self._health_check_loop())
        )
        self._background_tasks.append(
            asyncio.create_task(self._task_processing_loop())
        )

        logger.info(
            "cluster_manager.started",
            node_id=self._node_id,
            nodes=len(self._state.nodes),
        )

    async def stop(self) -> None:
        """Gracefully stop the cluster manager."""
        if not self._running:
            return

        logger.info("cluster_manager.stopping", node_id=self._node_id)
        self._running = False

        # Signal draining
        self._local_node.status = NodeStatus.DRAINING

        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        self._background_tasks.clear()

        # Notify peers of departure
        await self._notify_departure()

        # Stop sub-components
        if self._consensus is not None:
            await self._consensus.stop()
        if self._rpc_server is not None:
            await self._rpc_server.stop()
        if self._discovery is not None:
            await self._discovery.stop()

        self._local_node.status = NodeStatus.OFFLINE
        logger.info("cluster_manager.stopped", node_id=self._node_id)

    # ------------------------------------------------------------------
    # Event registration
    # ------------------------------------------------------------------

    def on(self, event: str, handler: EventHandler) -> None:
        """Register an event handler."""
        if event in self._event_handlers:
            self._event_handlers[event].append(handler)

    async def _emit(self, event: str, **kwargs: Any) -> None:
        """Emit an event to all registered handlers."""
        for handler in self._event_handlers.get(event, []):
            try:
                await handler(**kwargs)
            except Exception:
                logger.exception("cluster_manager.event_handler_error", event=event)

    # ------------------------------------------------------------------
    # Node lifecycle
    # ------------------------------------------------------------------

    async def handle_node_joined(self, node: NodeInfo) -> None:
        """Handle a new node joining the cluster."""
        if node.id == self._node_id:
            return

        node.status = NodeStatus.HEALTHY
        node.joined_at = datetime.now()
        self._state.nodes[node.id] = node
        self._state.increment_epoch()
        self._nodes_joined += 1

        # Record first heartbeat so the health checker has data
        if self._health_checker is not None:
            self._health_checker.record_heartbeat(node.id)

        logger.info(
            "cluster_manager.node_joined",
            node_id=node.id,
            node_name=node.name,
            total_nodes=len(self._state.nodes),
        )

        await self._emit("node_joined", node=node)

    async def handle_node_left(self, node_id: str) -> None:
        """Handle a node leaving the cluster (graceful or failure)."""
        node = self._state.nodes.pop(node_id, None)
        if node is None:
            return

        self._state.increment_epoch()
        self._nodes_left += 1

        # If the departed node was the leader, trigger re-election
        if self._state.leader_id == node_id:
            self._state.leader_id = None
            if self._consensus is not None:
                await self._consensus.trigger_election()

        # Re-assign tasks that were on the departed node
        await self._reassign_tasks(node_id)

        logger.info(
            "cluster_manager.node_left",
            node_id=node_id,
            node_name=node.name,
            total_nodes=len(self._state.nodes),
        )

        await self._emit("node_left", node_id=node_id, node=node)

    async def handle_leader_change(self, new_leader_id: str, term: int) -> None:
        """Handle a leader change in the cluster."""
        old_leader = self._state.leader_id
        self._state.leader_id = new_leader_id
        self._state.term = term

        # Update roles
        for nid, node in self._state.nodes.items():
            if nid == new_leader_id:
                node.role = NodeRole.LEADER
            elif node.role == NodeRole.LEADER:
                node.role = NodeRole.FOLLOWER

        logger.info(
            "cluster_manager.leader_changed",
            old_leader=old_leader,
            new_leader=new_leader_id,
            term=term,
        )

        await self._emit(
            "leader_changed",
            old_leader_id=old_leader,
            new_leader_id=new_leader_id,
            term=term,
        )

    # ------------------------------------------------------------------
    # Task submission & distribution
    # ------------------------------------------------------------------

    async def submit_task(self, task: DistributedTask) -> DistributedTask:
        """
        Submit a task for distributed execution.

        The task is assigned to the best-fit node based on load score and
        required capabilities, then dispatched via RPC.
        """
        task.source_node = self._node_id
        self._tasks_submitted += 1

        # Select target node
        target = self._select_node(task)
        if target is None:
            task.status = TaskStatus.FAILED
            task.error = "No suitable node available"
            self._tasks_failed += 1
            logger.warning(
                "cluster_manager.no_node_for_task",
                task_id=task.id,
                required=list(task.required_capabilities),
            )
            return task

        task.assigned_node = target.id
        task.status = TaskStatus.ASSIGNED

        # Enqueue or dispatch
        if target.id == self._node_id:
            # Execute locally
            asyncio.create_task(self._execute_task(task))
        else:
            # Send to remote node
            if self._rpc_client is not None:
                try:
                    await self._rpc_client.send_task(target.address, task)
                    task.status = TaskStatus.QUEUED
                except Exception as exc:
                    logger.error(
                        "cluster_manager.task_dispatch_failed",
                        task_id=task.id,
                        target=target.id,
                        error=str(exc),
                    )
                    task.status = TaskStatus.FAILED
                    task.error = str(exc)
                    self._tasks_failed += 1
            else:
                # Fallback: execute locally if no RPC client
                asyncio.create_task(self._execute_task(task))

        if self._task_queue is not None:
            await self._task_queue.track(task)

        logger.info(
            "cluster_manager.task_submitted",
            task_id=task.id,
            task_type=task.task_type,
            assigned_to=task.assigned_node,
        )
        return task

    def _select_node(self, task: DistributedTask) -> Optional[NodeInfo]:
        """
        Select the best node for a task.

        Strategy: pick the available node with the lowest ``load_score``
        whose capabilities are a superset of the task's requirements.
        Preferred nodes receive a tie-breaking bonus.
        """
        candidates: List[NodeInfo] = []
        for node in self._state.nodes.values():
            if not node.is_available():
                continue
            if node.id in task.excluded_nodes:
                continue
            # Check capability match
            if task.required_capabilities and not task.required_capabilities.issubset(
                node.capabilities
            ):
                continue
            candidates.append(node)

        if not candidates:
            return None

        # Sort by load_score ascending; prefer preferred_nodes on tie
        preferred_set: Set[str] = set(task.preferred_nodes)

        def sort_key(n: NodeInfo) -> tuple:
            prefer_bonus = 0.0 if n.id in preferred_set else 0.001
            return (n.load_score + prefer_bonus, n.current_tasks)

        candidates.sort(key=sort_key)
        return candidates[0]

    # ------------------------------------------------------------------
    # Local task execution
    # ------------------------------------------------------------------

    async def _execute_task(self, task: DistributedTask) -> None:
        """Execute a task on the local node."""
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now()
        self._local_node.current_tasks += 1

        handler = self._task_handlers.get(task.task_type)
        try:
            if handler is not None:
                task.result = await handler(task)
            else:
                task.result = {"status": "unknown_task_type", "task_type": task.task_type}

            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            self._tasks_completed += 1
            logger.info("cluster_manager.task_completed", task_id=task.id)
        except Exception as exc:
            task.status = TaskStatus.FAILED
            task.error = str(exc)
            task.completed_at = datetime.now()
            self._tasks_failed += 1
            task.record_attempt(self._node_id, error=str(exc))
            logger.error(
                "cluster_manager.task_failed",
                task_id=task.id,
                error=str(exc),
            )
            # Retry if eligible
            if task.can_retry():
                task.retry_count += 1
                task.status = TaskStatus.RETRYING
                task.assigned_node = None
                await self.submit_task(task)
        finally:
            self._local_node.current_tasks = max(
                0, self._local_node.current_tasks - 1
            )

        await self._emit("task_completed", task=task)

    # ------------------------------------------------------------------
    # Task type handlers
    # ------------------------------------------------------------------

    async def _handle_tool_task(self, task: DistributedTask) -> Any:
        """Execute a tool invocation task."""
        tool_name = task.payload.get("tool_name", "")
        tool_args = task.payload.get("arguments", {})
        logger.debug(
            "cluster_manager.handle_tool_task",
            tool=tool_name,
            task_id=task.id,
        )
        # Delegate to kernel's tool subsystem
        if hasattr(self._kernel, "tool_manager"):
            result = await self._kernel.tool_manager.execute(tool_name, **tool_args)
            return result
        return {"tool": tool_name, "status": "tool_manager_unavailable"}

    async def _handle_memory_task(self, task: DistributedTask) -> Any:
        """Execute a memory operation task."""
        operation = task.payload.get("operation", "read")
        key = task.payload.get("key", "")
        value = task.payload.get("value")
        logger.debug(
            "cluster_manager.handle_memory_task",
            operation=operation,
            key=key,
            task_id=task.id,
        )
        if hasattr(self._kernel, "memory"):
            if operation == "read":
                return await self._kernel.memory.retrieve(key)
            elif operation == "write":
                await self._kernel.memory.store(key, value)
                return {"key": key, "stored": True}
            elif operation == "delete":
                await self._kernel.memory.delete(key)
                return {"key": key, "deleted": True}
        return {"operation": operation, "status": "memory_unavailable"}

    async def _handle_agent_task(self, task: DistributedTask) -> Any:
        """Execute an agent operation task."""
        agent_name = task.payload.get("agent_name", "")
        action = task.payload.get("action", "")
        params = task.payload.get("params", {})
        logger.debug(
            "cluster_manager.handle_agent_task",
            agent=agent_name,
            action=action,
            task_id=task.id,
        )
        if hasattr(self._kernel, "agent_manager"):
            result = await self._kernel.agent_manager.dispatch(
                agent_name, action, **params
            )
            return result
        return {"agent": agent_name, "status": "agent_manager_unavailable"}

    async def _handle_planning_task(self, task: DistributedTask) -> Any:
        """Execute a planning operation task."""
        goal = task.payload.get("goal", "")
        context = task.payload.get("context", {})
        logger.debug(
            "cluster_manager.handle_planning_task",
            goal=goal,
            task_id=task.id,
        )
        if hasattr(self._kernel, "planner"):
            plan = await self._kernel.planner.create_plan(goal, context=context)
            return plan
        return {"goal": goal, "status": "planner_unavailable"}

    # ------------------------------------------------------------------
    # Background loops
    # ------------------------------------------------------------------

    async def _heartbeat_loop(self) -> None:
        """Periodically send heartbeats to all peer nodes."""
        while self._running:
            try:
                heartbeat = HeartbeatMessage(
                    node_id=self._node_id,
                    term=self._state.term,
                    timestamp=datetime.now(),
                    load_score=self._local_node.load_score,
                    status=self._local_node.status,
                    current_tasks=self._local_node.current_tasks,
                    cpu_usage=self._local_node.cpu_usage,
                    memory_usage=self._local_node.memory_usage,
                    epoch=self._state.epoch,
                )

                if self._rpc_client is not None:
                    for nid, node in list(self._state.nodes.items()):
                        if nid == self._node_id:
                            continue
                        try:
                            await self._rpc_client.send_heartbeat(
                                node.address, heartbeat
                            )
                        except Exception:
                            logger.debug(
                                "cluster_manager.heartbeat_send_failed",
                                target=nid,
                            )
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("cluster_manager.heartbeat_loop_error")

            await asyncio.sleep(self._heartbeat_interval)

    async def _health_check_loop(self) -> None:
        """Periodically check the health of all known nodes."""
        while self._running:
            try:
                if self._health_checker is not None:
                    for nid in list(self._state.nodes.keys()):
                        if nid == self._node_id:
                            continue
                        alive = self._health_checker.is_alive(nid)
                        node = self._state.nodes.get(nid)
                        if node is None:
                            continue

                        if not alive and node.status == NodeStatus.HEALTHY:
                            node.status = NodeStatus.SUSPECTED
                            logger.warning(
                                "cluster_manager.node_suspected",
                                node_id=nid,
                            )

                        if not alive and node.status == NodeStatus.SUSPECTED:
                            phi = self._health_checker.get_phi(nid)
                            if phi > self._health_checker.threshold * 1.5:
                                await self.handle_node_left(nid)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("cluster_manager.health_check_loop_error")

            await asyncio.sleep(self._health_check_interval)

    async def _task_processing_loop(self) -> None:
        """Process pending tasks from the distributed queue."""
        while self._running:
            try:
                if self._task_queue is not None and self.is_leader:
                    pending = await self._task_queue.dequeue_batch(batch_size=5)
                    for task in pending:
                        if not task.is_terminal:
                            await self.submit_task(task)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("cluster_manager.task_processing_loop_error")

            await asyncio.sleep(self._task_process_interval)

    # ------------------------------------------------------------------
    # Heartbeat reception
    # ------------------------------------------------------------------

    async def handle_heartbeat(self, heartbeat: HeartbeatMessage) -> None:
        """Process an incoming heartbeat from a peer."""
        node = self._state.nodes.get(heartbeat.node_id)
        if node is None:
            # Unknown node -- could trigger discovery
            logger.debug(
                "cluster_manager.heartbeat_unknown_node",
                sender=heartbeat.node_id,
            )
            return

        # Update node metrics
        node.last_heartbeat = heartbeat.timestamp
        node.cpu_usage = heartbeat.cpu_usage
        node.memory_usage = heartbeat.memory_usage
        node.current_tasks = heartbeat.current_tasks
        if heartbeat.status != NodeStatus.OFFLINE:
            node.status = heartbeat.status

        # Feed the health checker
        if self._health_checker is not None:
            self._health_checker.record_heartbeat(heartbeat.node_id)

    # ------------------------------------------------------------------
    # Discovery helpers
    # ------------------------------------------------------------------

    async def _discover_and_join(self) -> None:
        """Discover cluster peers and join them."""
        if self._discovery is None:
            return

        try:
            peers = await self._discovery.discover()
            for peer in peers:
                if peer.id != self._node_id and peer.id not in self._state.nodes:
                    await self.handle_node_joined(peer)
        except Exception:
            logger.exception("cluster_manager.discovery_failed")

    async def _notify_departure(self) -> None:
        """Notify peers that this node is leaving."""
        if self._rpc_client is None:
            return
        for nid, node in list(self._state.nodes.items()):
            if nid == self._node_id:
                continue
            try:
                await self._rpc_client.send_leave(node.address, self._node_id)
            except Exception:
                logger.debug(
                    "cluster_manager.departure_notify_failed",
                    target=nid,
                )

    async def _reassign_tasks(self, departed_node_id: str) -> None:
        """Re-submit tasks that were assigned to a departed node."""
        if self._task_queue is None:
            return
        orphaned = await self._task_queue.get_tasks_for_node(departed_node_id)
        for task in orphaned:
            task.assigned_node = None
            task.status = TaskStatus.PENDING
            task.excluded_nodes.add(departed_node_id)
            await self.submit_task(task)
            logger.info(
                "cluster_manager.task_reassigned",
                task_id=task.id,
                old_node=departed_node_id,
            )

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    @property
    def is_leader(self) -> bool:
        """Return True if the local node is the current leader."""
        return self._state.leader_id == self._node_id

    def get_cluster_info(self) -> Dict[str, Any]:
        """Return a summary of the current cluster state."""
        return {
            "cluster_name": self._cluster_name,
            "node_id": self._node_id,
            "is_leader": self.is_leader,
            "leader_id": self._state.leader_id,
            "term": self._state.term,
            "epoch": self._state.epoch,
            "total_nodes": len(self._state.nodes),
            "healthy_nodes": len(self._state.healthy_nodes),
            "has_quorum": self._state.has_quorum,
            "nodes": {
                nid: node.to_dict() for nid, node in self._state.nodes.items()
            },
        }

    def get_stats(self) -> Dict[str, Any]:
        """Return operational statistics."""
        uptime = 0.0
        if self._started_at is not None:
            uptime = (datetime.now() - self._started_at).total_seconds()

        return {
            "node_id": self._node_id,
            "uptime_seconds": round(uptime, 2),
            "tasks_submitted": self._tasks_submitted,
            "tasks_completed": self._tasks_completed,
            "tasks_failed": self._tasks_failed,
            "nodes_joined": self._nodes_joined,
            "nodes_left": self._nodes_left,
            "current_tasks": self._local_node.current_tasks,
            "load_score": round(self._local_node.load_score, 4),
        }

    def get_metrics(self) -> ClusterMetrics:
        """Compute aggregated cluster metrics."""
        nodes = list(self._state.nodes.values())
        healthy = [n for n in nodes if n.status == NodeStatus.HEALTHY]
        total_cpu = sum(n.cpu_usage for n in nodes) if nodes else 0.0
        total_mem = sum(n.memory_usage for n in nodes) if nodes else 0.0
        total_running = sum(n.current_tasks for n in nodes)

        return ClusterMetrics(
            total_nodes=len(nodes),
            healthy_nodes=len(healthy),
            has_quorum=self._state.has_quorum,
            avg_cpu_usage=total_cpu / len(nodes) if nodes else 0.0,
            avg_memory_usage=total_mem / len(nodes) if nodes else 0.0,
            total_tasks_running=total_running,
            current_term=self._state.term,
            leader_id=self._state.leader_id,
        )
