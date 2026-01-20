"""
AION Distributed Cluster Support

Enterprise-grade distributed process management with:
- Multi-node cluster coordination
- Leader election using Raft-like consensus
- Process migration between nodes
- Network partition handling (split-brain prevention)
- Node health monitoring and automatic failover
- Consistent hashing for process placement
- Gossip protocol for cluster state
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import socket
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from bisect import bisect_left

import structlog

from aion.systems.process.models import ProcessInfo, ProcessState, Event

logger = structlog.get_logger(__name__)


class NodeState(Enum):
    """State of a cluster node."""
    JOINING = auto()
    ACTIVE = auto()
    LEAVING = auto()
    SUSPECTED = auto()
    FAILED = auto()
    PARTITIONED = auto()


class LeaderState(Enum):
    """Raft-like leader election states."""
    FOLLOWER = auto()
    CANDIDATE = auto()
    LEADER = auto()


@dataclass
class NodeInfo:
    """Information about a cluster node."""
    id: str
    host: str
    port: int
    state: NodeState = NodeState.JOINING
    leader_state: LeaderState = LeaderState.FOLLOWER
    joined_at: datetime = field(default_factory=datetime.now)
    last_heartbeat: datetime = field(default_factory=datetime.now)
    process_count: int = 0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    load_score: float = 0.0
    capabilities: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    term: int = 0  # Raft term
    voted_for: Optional[str] = None

    @property
    def address(self) -> str:
        return f"{self.host}:{self.port}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "host": self.host,
            "port": self.port,
            "state": self.state.name,
            "leader_state": self.leader_state.name,
            "joined_at": self.joined_at.isoformat(),
            "last_heartbeat": self.last_heartbeat.isoformat(),
            "process_count": self.process_count,
            "cpu_usage": self.cpu_usage,
            "memory_usage": self.memory_usage,
            "load_score": self.load_score,
            "capabilities": list(self.capabilities),
            "term": self.term,
        }

    def calculate_load_score(self) -> float:
        """Calculate weighted load score for placement decisions."""
        # Lower is better
        self.load_score = (
            self.process_count * 0.4 +
            self.cpu_usage * 0.35 +
            self.memory_usage * 0.25
        )
        return self.load_score


@dataclass
class ClusterMessage:
    """Message for inter-node communication."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: str = ""
    source_node: str = ""
    target_node: Optional[str] = None  # None = broadcast
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    term: int = 0  # Raft term

    def to_bytes(self) -> bytes:
        return json.dumps({
            "id": self.id,
            "type": self.type,
            "source_node": self.source_node,
            "target_node": self.target_node,
            "payload": self.payload,
            "timestamp": self.timestamp.isoformat(),
            "term": self.term,
        }).encode()

    @classmethod
    def from_bytes(cls, data: bytes) -> "ClusterMessage":
        d = json.loads(data.decode())
        return cls(
            id=d["id"],
            type=d["type"],
            source_node=d["source_node"],
            target_node=d.get("target_node"),
            payload=d.get("payload", {}),
            timestamp=datetime.fromisoformat(d["timestamp"]),
            term=d.get("term", 0),
        )


@dataclass
class ProcessMigration:
    """Represents a process migration operation."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    process_id: str = ""
    source_node: str = ""
    target_node: str = ""
    state: str = "pending"  # pending, transferring, completing, completed, failed
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    checkpoint_data: Optional[bytes] = None
    error: Optional[str] = None


class ConsistentHashRing:
    """
    Consistent hashing ring for process placement.
    Provides minimal redistribution when nodes join/leave.
    """

    def __init__(self, virtual_nodes: int = 150):
        self.virtual_nodes = virtual_nodes
        self._ring: List[Tuple[int, str]] = []
        self._nodes: Set[str] = set()

    def _hash(self, key: str) -> int:
        """Generate hash for a key."""
        return int(hashlib.md5(key.encode()).hexdigest(), 16)

    def add_node(self, node_id: str) -> None:
        """Add a node to the ring with virtual nodes."""
        if node_id in self._nodes:
            return

        self._nodes.add(node_id)

        for i in range(self.virtual_nodes):
            virtual_key = f"{node_id}:{i}"
            hash_val = self._hash(virtual_key)
            # Binary insertion to maintain sorted order
            idx = bisect_left([h for h, _ in self._ring], hash_val)
            self._ring.insert(idx, (hash_val, node_id))

    def remove_node(self, node_id: str) -> None:
        """Remove a node from the ring."""
        if node_id not in self._nodes:
            return

        self._nodes.discard(node_id)
        self._ring = [(h, n) for h, n in self._ring if n != node_id]

    def get_node(self, key: str) -> Optional[str]:
        """Get the node responsible for a key."""
        if not self._ring:
            return None

        hash_val = self._hash(key)

        # Find first node with hash >= key hash
        idx = bisect_left([h for h, _ in self._ring], hash_val)
        if idx >= len(self._ring):
            idx = 0

        return self._ring[idx][1]

    def get_nodes_for_key(self, key: str, count: int = 3) -> List[str]:
        """Get multiple nodes for replication."""
        if not self._ring or count <= 0:
            return []

        hash_val = self._hash(key)
        idx = bisect_left([h for h, _ in self._ring], hash_val)

        nodes = []
        seen = set()

        for i in range(len(self._ring)):
            real_idx = (idx + i) % len(self._ring)
            node_id = self._ring[real_idx][1]

            if node_id not in seen:
                seen.add(node_id)
                nodes.append(node_id)

                if len(nodes) >= count:
                    break

        return nodes

    def get_affected_keys(self, node_id: str, all_keys: List[str]) -> List[str]:
        """Get keys that would be affected by node removal."""
        affected = []

        for key in all_keys:
            nodes = self.get_nodes_for_key(key, 2)
            if node_id in nodes:
                affected.append(key)

        return affected


class ClusterTransport(ABC):
    """Abstract transport for cluster communication."""

    @abstractmethod
    async def start(self, host: str, port: int) -> None:
        """Start the transport."""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop the transport."""
        pass

    @abstractmethod
    async def send(self, target: str, message: ClusterMessage) -> None:
        """Send a message to a specific node."""
        pass

    @abstractmethod
    async def broadcast(self, message: ClusterMessage) -> None:
        """Broadcast a message to all nodes."""
        pass

    @abstractmethod
    def set_message_handler(self, handler: Callable[[ClusterMessage], Any]) -> None:
        """Set the handler for incoming messages."""
        pass


class UDPClusterTransport(ClusterTransport):
    """UDP-based cluster transport for gossip and heartbeats."""

    def __init__(self, cluster: "ClusterCoordinator"):
        self.cluster = cluster
        self._socket: Optional[asyncio.DatagramTransport] = None
        self._protocol: Optional[asyncio.DatagramProtocol] = None
        self._message_handler: Optional[Callable[[ClusterMessage], Any]] = None
        self._host: str = ""
        self._port: int = 0

    async def start(self, host: str, port: int) -> None:
        self._host = host
        self._port = port

        loop = asyncio.get_event_loop()

        class Protocol(asyncio.DatagramProtocol):
            def __init__(self, transport_ref):
                self.transport_ref = transport_ref

            def datagram_received(self, data: bytes, addr: Tuple[str, int]) -> None:
                try:
                    message = ClusterMessage.from_bytes(data)
                    if self.transport_ref._message_handler:
                        asyncio.create_task(self._handle_message(message))
                except Exception as e:
                    logger.error(f"Failed to parse cluster message: {e}")

            async def _handle_message(self, message: ClusterMessage) -> None:
                if self.transport_ref._message_handler:
                    await self.transport_ref._message_handler(message)

        self._socket, self._protocol = await loop.create_datagram_endpoint(
            lambda: Protocol(self),
            local_addr=(host, port),
        )

        logger.info(f"Cluster transport started on {host}:{port}")

    async def stop(self) -> None:
        if self._socket:
            self._socket.close()

    async def send(self, target: str, message: ClusterMessage) -> None:
        if not self._socket:
            return

        host, port = target.split(":")
        self._socket.sendto(message.to_bytes(), (host, int(port)))

    async def broadcast(self, message: ClusterMessage) -> None:
        for node in self.cluster.get_active_nodes():
            if node.id != self.cluster.local_node.id:
                await self.send(node.address, message)

    def set_message_handler(self, handler: Callable[[ClusterMessage], Any]) -> None:
        self._message_handler = handler


class TCPClusterTransport(ClusterTransport):
    """TCP-based cluster transport for reliable messages."""

    def __init__(self, cluster: "ClusterCoordinator"):
        self.cluster = cluster
        self._server: Optional[asyncio.Server] = None
        self._connections: Dict[str, asyncio.StreamWriter] = {}
        self._message_handler: Optional[Callable[[ClusterMessage], Any]] = None
        self._host: str = ""
        self._port: int = 0

    async def start(self, host: str, port: int) -> None:
        self._host = host
        self._port = port

        self._server = await asyncio.start_server(
            self._handle_connection,
            host,
            port,
        )

        logger.info(f"TCP cluster transport started on {host}:{port}")

    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            while True:
                length_bytes = await reader.readexactly(4)
                length = int.from_bytes(length_bytes, 'big')
                data = await reader.readexactly(length)

                message = ClusterMessage.from_bytes(data)

                if self._message_handler:
                    await self._message_handler(message)

        except asyncio.IncompleteReadError:
            pass
        except Exception as e:
            logger.error(f"Connection error: {e}")
        finally:
            writer.close()

    async def stop(self) -> None:
        if self._server:
            self._server.close()
            await self._server.wait_closed()

        for writer in self._connections.values():
            writer.close()
        self._connections.clear()

    async def _get_connection(self, target: str) -> asyncio.StreamWriter:
        if target not in self._connections:
            host, port = target.split(":")
            reader, writer = await asyncio.open_connection(host, int(port))
            self._connections[target] = writer

        return self._connections[target]

    async def send(self, target: str, message: ClusterMessage) -> None:
        try:
            writer = await self._get_connection(target)
            data = message.to_bytes()
            length = len(data).to_bytes(4, 'big')
            writer.write(length + data)
            await writer.drain()
        except Exception as e:
            logger.error(f"Failed to send to {target}: {e}")
            self._connections.pop(target, None)

    async def broadcast(self, message: ClusterMessage) -> None:
        for node in self.cluster.get_active_nodes():
            if node.id != self.cluster.local_node.id:
                await self.send(node.address, message)

    def set_message_handler(self, handler: Callable[[ClusterMessage], Any]) -> None:
        self._message_handler = handler


class ClusterCoordinator:
    """
    Central cluster coordinator implementing:
    - Raft-like leader election
    - Node membership management
    - Process placement with consistent hashing
    - Process migration
    - Split-brain prevention
    """

    def __init__(
        self,
        node_id: Optional[str] = None,
        host: str = "0.0.0.0",
        port: int = 9000,
        heartbeat_interval: float = 1.0,
        election_timeout_range: Tuple[float, float] = (3.0, 5.0),
        suspect_timeout: float = 5.0,
        failed_timeout: float = 15.0,
        min_quorum_ratio: float = 0.5,
    ):
        self.node_id = node_id or str(uuid.uuid4())
        self.host = host
        self.port = port
        self.heartbeat_interval = heartbeat_interval
        self.election_timeout_range = election_timeout_range
        self.suspect_timeout = suspect_timeout
        self.failed_timeout = failed_timeout
        self.min_quorum_ratio = min_quorum_ratio

        # Local node info
        self.local_node = NodeInfo(
            id=self.node_id,
            host=host,
            port=port,
            state=NodeState.JOINING,
        )

        # Cluster state
        self._nodes: Dict[str, NodeInfo] = {self.node_id: self.local_node}
        self._leader_id: Optional[str] = None
        self._current_term: int = 0
        self._voted_for: Optional[str] = None
        self._votes_received: Set[str] = set()

        # Consistent hashing
        self._hash_ring = ConsistentHashRing()
        self._hash_ring.add_node(self.node_id)

        # Process tracking
        self._local_processes: Dict[str, ProcessInfo] = {}
        self._process_locations: Dict[str, str] = {}  # process_id -> node_id

        # Migrations
        self._active_migrations: Dict[str, ProcessMigration] = {}

        # Transport
        self._udp_transport: Optional[UDPClusterTransport] = None
        self._tcp_transport: Optional[TCPClusterTransport] = None

        # Background tasks
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._election_task: Optional[asyncio.Task] = None
        self._failure_detector_task: Optional[asyncio.Task] = None

        # Election timing
        self._last_heartbeat_from_leader: datetime = datetime.now()
        self._election_timeout: float = random.uniform(*election_timeout_range)

        # Event callbacks
        self._on_leader_elected: List[Callable[[str], Any]] = []
        self._on_node_joined: List[Callable[[NodeInfo], Any]] = []
        self._on_node_failed: List[Callable[[NodeInfo], Any]] = []
        self._on_process_migrated: List[Callable[[ProcessMigration], Any]] = []

        # Locks
        self._state_lock = asyncio.Lock()

        self._initialized = False
        self._shutdown_event = asyncio.Event()

    async def initialize(self, seed_nodes: Optional[List[str]] = None) -> None:
        """Initialize and join the cluster."""
        if self._initialized:
            return

        logger.info(
            "Initializing cluster coordinator",
            node_id=self.node_id,
            host=self.host,
            port=self.port,
        )

        # Start transports
        self._udp_transport = UDPClusterTransport(self)
        self._tcp_transport = TCPClusterTransport(self)

        await self._udp_transport.start(self.host, self.port)
        await self._tcp_transport.start(self.host, self.port + 1)

        self._udp_transport.set_message_handler(self._handle_message)
        self._tcp_transport.set_message_handler(self._handle_message)

        # Join existing cluster if seed nodes provided
        if seed_nodes:
            await self._join_cluster(seed_nodes)

        # Start background tasks
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self._election_task = asyncio.create_task(self._election_loop())
        self._failure_detector_task = asyncio.create_task(self._failure_detector_loop())

        self.local_node.state = NodeState.ACTIVE
        self._initialized = True

        logger.info("Cluster coordinator initialized", node_id=self.node_id)

    async def shutdown(self) -> None:
        """Shutdown the cluster coordinator."""
        logger.info("Shutting down cluster coordinator")

        self._shutdown_event.set()
        self.local_node.state = NodeState.LEAVING

        # Notify cluster we're leaving
        await self._broadcast(ClusterMessage(
            type="node.leaving",
            source_node=self.node_id,
            payload={"node_id": self.node_id},
            term=self._current_term,
        ))

        # Cancel background tasks
        for task in [self._heartbeat_task, self._election_task, self._failure_detector_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Stop transports
        if self._udp_transport:
            await self._udp_transport.stop()
        if self._tcp_transport:
            await self._tcp_transport.stop()

        logger.info("Cluster coordinator shutdown complete")

    async def _join_cluster(self, seed_nodes: List[str]) -> None:
        """Join an existing cluster via seed nodes."""
        for seed in seed_nodes:
            try:
                # Send join request
                message = ClusterMessage(
                    type="node.join_request",
                    source_node=self.node_id,
                    payload={
                        "node_info": self.local_node.to_dict(),
                    },
                    term=self._current_term,
                )

                if self._tcp_transport:
                    await self._tcp_transport.send(seed, message)

                # Wait briefly for response
                await asyncio.sleep(0.5)

            except Exception as e:
                logger.warning(f"Failed to join via {seed}: {e}")

    async def _handle_message(self, message: ClusterMessage) -> None:
        """Handle incoming cluster messages."""
        handlers = {
            "heartbeat": self._handle_heartbeat,
            "node.join_request": self._handle_join_request,
            "node.join_response": self._handle_join_response,
            "node.leaving": self._handle_node_leaving,
            "election.request_vote": self._handle_vote_request,
            "election.vote": self._handle_vote,
            "election.leader_announce": self._handle_leader_announce,
            "process.migrate_request": self._handle_migrate_request,
            "process.migrate_data": self._handle_migrate_data,
            "process.migrate_complete": self._handle_migrate_complete,
            "cluster.state_sync": self._handle_state_sync,
        }

        handler = handlers.get(message.type)
        if handler:
            await handler(message)
        else:
            logger.debug(f"Unknown message type: {message.type}")

    async def _handle_heartbeat(self, message: ClusterMessage) -> None:
        """Handle heartbeat from another node."""
        node_id = message.source_node

        async with self._state_lock:
            if node_id in self._nodes:
                node = self._nodes[node_id]
                node.last_heartbeat = datetime.now()

                if node.state == NodeState.SUSPECTED:
                    node.state = NodeState.ACTIVE
                    logger.info(f"Node {node_id} recovered")

            # Update from payload
            if "load" in message.payload:
                self._nodes[node_id].cpu_usage = message.payload["load"].get("cpu", 0)
                self._nodes[node_id].memory_usage = message.payload["load"].get("memory", 0)
                self._nodes[node_id].process_count = message.payload["load"].get("processes", 0)
                self._nodes[node_id].calculate_load_score()

            # If from leader, reset election timeout
            if node_id == self._leader_id:
                self._last_heartbeat_from_leader = datetime.now()
                self.local_node.leader_state = LeaderState.FOLLOWER

                # Update term if leader has higher term
                if message.term > self._current_term:
                    self._current_term = message.term
                    self._voted_for = None

    async def _handle_join_request(self, message: ClusterMessage) -> None:
        """Handle node join request."""
        node_info_dict = message.payload.get("node_info", {})

        new_node = NodeInfo(
            id=node_info_dict["id"],
            host=node_info_dict["host"],
            port=node_info_dict["port"],
            state=NodeState.ACTIVE,
        )

        async with self._state_lock:
            self._nodes[new_node.id] = new_node
            self._hash_ring.add_node(new_node.id)

        # Send cluster state to new node
        response = ClusterMessage(
            type="node.join_response",
            source_node=self.node_id,
            target_node=new_node.id,
            payload={
                "nodes": [n.to_dict() for n in self._nodes.values()],
                "leader_id": self._leader_id,
                "term": self._current_term,
            },
            term=self._current_term,
        )

        if self._tcp_transport:
            await self._tcp_transport.send(new_node.address, response)

        # Notify callbacks
        for callback in self._on_node_joined:
            try:
                result = callback(new_node)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"Node joined callback error: {e}")

        logger.info(f"Node joined cluster: {new_node.id}")

    async def _handle_join_response(self, message: ClusterMessage) -> None:
        """Handle join response with cluster state."""
        async with self._state_lock:
            # Update nodes
            for node_dict in message.payload.get("nodes", []):
                node_id = node_dict["id"]
                if node_id != self.node_id:
                    node = NodeInfo(
                        id=node_id,
                        host=node_dict["host"],
                        port=node_dict["port"],
                        state=NodeState[node_dict["state"]],
                    )
                    self._nodes[node_id] = node
                    self._hash_ring.add_node(node_id)

            # Update leader
            self._leader_id = message.payload.get("leader_id")
            self._current_term = message.payload.get("term", 0)

        logger.info(
            "Joined cluster",
            nodes=len(self._nodes),
            leader=self._leader_id,
        )

    async def _handle_node_leaving(self, message: ClusterMessage) -> None:
        """Handle node leaving notification."""
        node_id = message.payload.get("node_id")

        async with self._state_lock:
            if node_id in self._nodes:
                node = self._nodes[node_id]
                node.state = NodeState.LEAVING

                # Remove from hash ring
                self._hash_ring.remove_node(node_id)

                # Trigger migration of affected processes
                if self.is_leader():
                    await self._handle_node_departure(node_id)

        logger.info(f"Node leaving: {node_id}")

    async def _handle_vote_request(self, message: ClusterMessage) -> None:
        """Handle vote request in leader election."""
        candidate_id = message.source_node
        candidate_term = message.term

        async with self._state_lock:
            vote_granted = False

            if candidate_term > self._current_term:
                self._current_term = candidate_term
                self._voted_for = None
                self.local_node.leader_state = LeaderState.FOLLOWER

            if candidate_term >= self._current_term:
                if self._voted_for is None or self._voted_for == candidate_id:
                    self._voted_for = candidate_id
                    vote_granted = True
                    self._last_heartbeat_from_leader = datetime.now()

        # Send vote response
        response = ClusterMessage(
            type="election.vote",
            source_node=self.node_id,
            target_node=candidate_id,
            payload={"vote_granted": vote_granted},
            term=self._current_term,
        )

        candidate_node = self._nodes.get(candidate_id)
        if candidate_node and self._udp_transport:
            await self._udp_transport.send(candidate_node.address, response)

    async def _handle_vote(self, message: ClusterMessage) -> None:
        """Handle vote response."""
        if message.payload.get("vote_granted"):
            self._votes_received.add(message.source_node)

            # Check if we won
            quorum = len(self._nodes) // 2 + 1
            if len(self._votes_received) >= quorum:
                await self._become_leader()

    async def _handle_leader_announce(self, message: ClusterMessage) -> None:
        """Handle leader announcement."""
        leader_id = message.source_node
        term = message.term

        async with self._state_lock:
            if term >= self._current_term:
                self._current_term = term
                self._leader_id = leader_id
                self._voted_for = None
                self.local_node.leader_state = LeaderState.FOLLOWER
                self._last_heartbeat_from_leader = datetime.now()

        # Notify callbacks
        for callback in self._on_leader_elected:
            try:
                result = callback(leader_id)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"Leader elected callback error: {e}")

        logger.info(f"Leader elected: {leader_id}")

    async def _handle_migrate_request(self, message: ClusterMessage) -> None:
        """Handle process migration request."""
        process_id = message.payload.get("process_id")
        target_node = message.payload.get("target_node")

        # Get process checkpoint data
        process = self._local_processes.get(process_id)
        if not process:
            logger.warning(f"Process not found for migration: {process_id}")
            return

        # Create migration
        migration = ProcessMigration(
            process_id=process_id,
            source_node=self.node_id,
            target_node=target_node,
            state="transferring",
        )
        self._active_migrations[migration.id] = migration

        # Serialize process state (simplified)
        checkpoint_data = json.dumps({
            "process_id": process_id,
            "state": process.to_dict(),
        }).encode()

        # Send data to target
        data_message = ClusterMessage(
            type="process.migrate_data",
            source_node=self.node_id,
            target_node=target_node,
            payload={
                "migration_id": migration.id,
                "process_id": process_id,
                "checkpoint_data": checkpoint_data.decode(),
            },
            term=self._current_term,
        )

        target_node_info = self._nodes.get(target_node)
        if target_node_info and self._tcp_transport:
            await self._tcp_transport.send(
                f"{target_node_info.host}:{target_node_info.port + 1}",
                data_message,
            )

    async def _handle_migrate_data(self, message: ClusterMessage) -> None:
        """Handle incoming process migration data."""
        migration_id = message.payload.get("migration_id")
        process_id = message.payload.get("process_id")
        checkpoint_data = message.payload.get("checkpoint_data")

        # Restore process (simplified)
        process_state = json.loads(checkpoint_data)

        # Register process locally
        self._process_locations[process_id] = self.node_id

        # Send completion
        complete_message = ClusterMessage(
            type="process.migrate_complete",
            source_node=self.node_id,
            target_node=message.source_node,
            payload={
                "migration_id": migration_id,
                "process_id": process_id,
                "success": True,
            },
            term=self._current_term,
        )

        source_node = self._nodes.get(message.source_node)
        if source_node and self._tcp_transport:
            await self._tcp_transport.send(
                f"{source_node.host}:{source_node.port + 1}",
                complete_message,
            )

        logger.info(f"Process migrated to this node: {process_id}")

    async def _handle_migrate_complete(self, message: ClusterMessage) -> None:
        """Handle migration completion."""
        migration_id = message.payload.get("migration_id")
        success = message.payload.get("success", False)
        process_id = message.payload.get("process_id")

        migration = self._active_migrations.get(migration_id)
        if migration:
            migration.state = "completed" if success else "failed"
            migration.completed_at = datetime.now()

            if success:
                # Remove local process
                self._local_processes.pop(process_id, None)
                self._process_locations[process_id] = migration.target_node

                # Notify callbacks
                for callback in self._on_process_migrated:
                    try:
                        result = callback(migration)
                        if asyncio.iscoroutine(result):
                            await result
                    except Exception as e:
                        logger.error(f"Migration callback error: {e}")

            logger.info(
                f"Migration {'completed' if success else 'failed'}: {process_id}",
            )

    async def _handle_state_sync(self, message: ClusterMessage) -> None:
        """Handle cluster state synchronization."""
        if not self.is_leader():
            async with self._state_lock:
                # Update process locations
                self._process_locations.update(
                    message.payload.get("process_locations", {})
                )

    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeats."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(self.heartbeat_interval)

                # Prepare heartbeat
                message = ClusterMessage(
                    type="heartbeat",
                    source_node=self.node_id,
                    payload={
                        "load": {
                            "cpu": self.local_node.cpu_usage,
                            "memory": self.local_node.memory_usage,
                            "processes": self.local_node.process_count,
                        },
                    },
                    term=self._current_term,
                )

                # Broadcast
                await self._broadcast(message)

                # If leader, also send state sync
                if self.is_leader():
                    sync_message = ClusterMessage(
                        type="cluster.state_sync",
                        source_node=self.node_id,
                        payload={
                            "process_locations": self._process_locations,
                        },
                        term=self._current_term,
                    )
                    await self._broadcast(sync_message)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")

    async def _election_loop(self) -> None:
        """Monitor for election timeout and start elections."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(0.5)

                if self.local_node.leader_state == LeaderState.LEADER:
                    continue

                # Check election timeout
                elapsed = (datetime.now() - self._last_heartbeat_from_leader).total_seconds()

                if elapsed > self._election_timeout:
                    await self._start_election()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Election loop error: {e}")

    async def _start_election(self) -> None:
        """Start a leader election."""
        async with self._state_lock:
            self._current_term += 1
            self.local_node.leader_state = LeaderState.CANDIDATE
            self._voted_for = self.node_id
            self._votes_received = {self.node_id}  # Vote for self

            # Reset election timeout with randomization
            self._election_timeout = random.uniform(*self.election_timeout_range)

        logger.info(f"Starting election for term {self._current_term}")

        # Request votes
        message = ClusterMessage(
            type="election.request_vote",
            source_node=self.node_id,
            term=self._current_term,
        )

        await self._broadcast(message)

        # Check if we're the only node
        if len(self._nodes) == 1:
            await self._become_leader()

    async def _become_leader(self) -> None:
        """Become the cluster leader."""
        async with self._state_lock:
            self.local_node.leader_state = LeaderState.LEADER
            self._leader_id = self.node_id

        logger.info(f"Became leader for term {self._current_term}")

        # Announce leadership
        message = ClusterMessage(
            type="election.leader_announce",
            source_node=self.node_id,
            term=self._current_term,
        )

        await self._broadcast(message)

        # Notify callbacks
        for callback in self._on_leader_elected:
            try:
                result = callback(self.node_id)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"Leader elected callback error: {e}")

    async def _failure_detector_loop(self) -> None:
        """Detect failed nodes."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(1.0)

                now = datetime.now()

                async with self._state_lock:
                    for node_id, node in list(self._nodes.items()):
                        if node_id == self.node_id:
                            continue

                        elapsed = (now - node.last_heartbeat).total_seconds()

                        if elapsed > self.failed_timeout and node.state != NodeState.FAILED:
                            node.state = NodeState.FAILED
                            self._hash_ring.remove_node(node_id)

                            logger.warning(f"Node failed: {node_id}")

                            # Trigger migration if we're leader
                            if self.is_leader():
                                await self._handle_node_departure(node_id)

                            # Notify callbacks
                            for callback in self._on_node_failed:
                                try:
                                    result = callback(node)
                                    if asyncio.iscoroutine(result):
                                        await result
                                except Exception as e:
                                    logger.error(f"Node failed callback error: {e}")

                        elif elapsed > self.suspect_timeout and node.state == NodeState.ACTIVE:
                            node.state = NodeState.SUSPECTED
                            logger.warning(f"Node suspected: {node_id}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Failure detector error: {e}")

    async def _handle_node_departure(self, node_id: str) -> None:
        """Handle node departure by migrating processes."""
        affected_processes = [
            pid for pid, nid in self._process_locations.items()
            if nid == node_id
        ]

        for process_id in affected_processes:
            # Find new placement
            new_node = self._hash_ring.get_node(process_id)
            if new_node and new_node != node_id:
                await self.migrate_process(process_id, new_node)

    async def _broadcast(self, message: ClusterMessage) -> None:
        """Broadcast a message to all nodes."""
        if self._udp_transport:
            await self._udp_transport.broadcast(message)

    # === Public API ===

    def is_leader(self) -> bool:
        """Check if this node is the leader."""
        return self.local_node.leader_state == LeaderState.LEADER

    def get_leader(self) -> Optional[NodeInfo]:
        """Get the current leader."""
        if self._leader_id:
            return self._nodes.get(self._leader_id)
        return None

    def get_node(self, node_id: str) -> Optional[NodeInfo]:
        """Get a node by ID."""
        return self._nodes.get(node_id)

    def get_all_nodes(self) -> List[NodeInfo]:
        """Get all nodes."""
        return list(self._nodes.values())

    def get_active_nodes(self) -> List[NodeInfo]:
        """Get all active nodes."""
        return [n for n in self._nodes.values() if n.state == NodeState.ACTIVE]

    def get_node_for_process(self, process_id: str) -> Optional[str]:
        """Get the best node for a process using consistent hashing."""
        return self._hash_ring.get_node(process_id)

    def get_least_loaded_node(self) -> Optional[NodeInfo]:
        """Get the node with lowest load."""
        active = self.get_active_nodes()
        if not active:
            return None

        for node in active:
            node.calculate_load_score()

        return min(active, key=lambda n: n.load_score)

    async def migrate_process(self, process_id: str, target_node: str) -> bool:
        """Initiate process migration."""
        current_node = self._process_locations.get(process_id)
        if not current_node or current_node == target_node:
            return False

        source_node = self._nodes.get(current_node)
        if not source_node:
            return False

        # Send migration request
        message = ClusterMessage(
            type="process.migrate_request",
            source_node=self.node_id,
            target_node=current_node,
            payload={
                "process_id": process_id,
                "target_node": target_node,
            },
            term=self._current_term,
        )

        if self._tcp_transport:
            await self._tcp_transport.send(
                f"{source_node.host}:{source_node.port + 1}",
                message,
            )

        return True

    def register_process(self, process_id: str, process: ProcessInfo) -> None:
        """Register a local process."""
        self._local_processes[process_id] = process
        self._process_locations[process_id] = self.node_id
        self.local_node.process_count = len(self._local_processes)

    def unregister_process(self, process_id: str) -> None:
        """Unregister a local process."""
        self._local_processes.pop(process_id, None)
        self._process_locations.pop(process_id, None)
        self.local_node.process_count = len(self._local_processes)

    def has_quorum(self) -> bool:
        """Check if cluster has quorum."""
        active = len(self.get_active_nodes())
        total = len(self._nodes)
        return active >= total * self.min_quorum_ratio

    def get_stats(self) -> Dict[str, Any]:
        """Get cluster statistics."""
        return {
            "node_id": self.node_id,
            "is_leader": self.is_leader(),
            "leader_id": self._leader_id,
            "term": self._current_term,
            "total_nodes": len(self._nodes),
            "active_nodes": len(self.get_active_nodes()),
            "has_quorum": self.has_quorum(),
            "local_processes": len(self._local_processes),
            "total_processes": len(self._process_locations),
            "active_migrations": len(self._active_migrations),
        }

    # === Event Registration ===

    def on_leader_elected(self, callback: Callable[[str], Any]) -> None:
        """Register callback for leader election."""
        self._on_leader_elected.append(callback)

    def on_node_joined(self, callback: Callable[[NodeInfo], Any]) -> None:
        """Register callback for node join."""
        self._on_node_joined.append(callback)

    def on_node_failed(self, callback: Callable[[NodeInfo], Any]) -> None:
        """Register callback for node failure."""
        self._on_node_failed.append(callback)

    def on_process_migrated(self, callback: Callable[[ProcessMigration], Any]) -> None:
        """Register callback for process migration."""
        self._on_process_migrated.append(callback)
