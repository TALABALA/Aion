"""
AION Raft Transport Layer

Production-grade network transport for Raft consensus:
- TCP-based reliable transport
- Connection pooling with keep-alive
- Automatic reconnection with exponential backoff
- Message framing with length prefix
- TLS support for encryption
- Compression for large messages
- Pipelining support for high throughput
"""

from __future__ import annotations

import asyncio
import json
import ssl
import struct
import time
import zlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type
import uuid

import structlog

logger = structlog.get_logger(__name__)


# Message frame format:
# | Magic (4 bytes) | Version (1 byte) | Type (1 byte) | Flags (2 bytes) | Length (4 bytes) | Payload |
FRAME_MAGIC = b"RAFT"
FRAME_HEADER_SIZE = 12
MAX_MESSAGE_SIZE = 64 * 1024 * 1024  # 64MB max


class MessageType(Enum):
    """Raft RPC message types."""
    APPEND_ENTRIES_REQUEST = 1
    APPEND_ENTRIES_RESPONSE = 2
    REQUEST_VOTE_REQUEST = 3
    REQUEST_VOTE_RESPONSE = 4
    INSTALL_SNAPSHOT_REQUEST = 5
    INSTALL_SNAPSHOT_RESPONSE = 6
    # Additional types
    HEARTBEAT = 7
    PRE_VOTE_REQUEST = 8
    PRE_VOTE_RESPONSE = 9
    TRANSFER_LEADER_REQUEST = 10
    TRANSFER_LEADER_RESPONSE = 11
    # Membership
    ADD_SERVER_REQUEST = 12
    ADD_SERVER_RESPONSE = 13
    REMOVE_SERVER_REQUEST = 14
    REMOVE_SERVER_RESPONSE = 15


class MessageFlags(Enum):
    """Message flags."""
    NONE = 0
    COMPRESSED = 1
    ENCRYPTED = 2
    URGENT = 4
    PIPELINE = 8  # Part of a pipeline batch


@dataclass
class RaftMessage:
    """A Raft RPC message."""
    msg_type: MessageType
    payload: Dict[str, Any]
    msg_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    flags: int = 0
    timestamp: float = field(default_factory=time.time)

    def to_bytes(self, compress: bool = False) -> bytes:
        """Serialize message to bytes."""
        payload_bytes = json.dumps(self.payload).encode()

        flags = self.flags
        if compress and len(payload_bytes) > 1024:
            payload_bytes = zlib.compress(payload_bytes, level=6)
            flags |= MessageFlags.COMPRESSED.value

        # Build frame
        frame = (
            FRAME_MAGIC +
            struct.pack("<BBHI", 1, self.msg_type.value, flags, len(payload_bytes)) +
            payload_bytes
        )
        return frame

    @classmethod
    def from_bytes(cls, data: bytes) -> Optional["RaftMessage"]:
        """Deserialize message from bytes."""
        if len(data) < FRAME_HEADER_SIZE:
            return None

        if data[:4] != FRAME_MAGIC:
            raise ValueError("Invalid frame magic")

        version, msg_type, flags, length = struct.unpack("<BBHI", data[4:FRAME_HEADER_SIZE])

        if version != 1:
            raise ValueError(f"Unsupported version: {version}")

        if len(data) < FRAME_HEADER_SIZE + length:
            return None

        payload_bytes = data[FRAME_HEADER_SIZE:FRAME_HEADER_SIZE + length]

        # Decompress if needed
        if flags & MessageFlags.COMPRESSED.value:
            payload_bytes = zlib.decompress(payload_bytes)

        payload = json.loads(payload_bytes.decode())

        return cls(
            msg_type=MessageType(msg_type),
            payload=payload,
            flags=flags,
        )


class RaftTransport(ABC):
    """Abstract base class for Raft transport."""

    @abstractmethod
    async def start(self) -> None:
        """Start the transport."""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop the transport."""
        pass

    @abstractmethod
    async def send(self, target: str, message: RaftMessage) -> Optional[RaftMessage]:
        """Send a message and wait for response."""
        pass

    @abstractmethod
    async def send_async(self, target: str, message: RaftMessage) -> None:
        """Send a message without waiting for response."""
        pass

    @abstractmethod
    def set_handler(self, handler: Callable[[str, RaftMessage], RaftMessage]) -> None:
        """Set the message handler."""
        pass


@dataclass
class ConnectionConfig:
    """Configuration for TCP connections."""
    connect_timeout: float = 5.0
    read_timeout: float = 10.0
    write_timeout: float = 5.0
    keepalive_interval: float = 30.0
    max_retries: int = 3
    retry_backoff_base: float = 0.5
    retry_backoff_max: float = 30.0
    max_connections_per_peer: int = 3
    enable_compression: bool = True
    compression_threshold: int = 1024


class TCPConnection:
    """A single TCP connection to a peer."""

    def __init__(
        self,
        peer_id: str,
        address: str,
        port: int,
        config: ConnectionConfig,
        ssl_context: Optional[ssl.SSLContext] = None,
    ):
        self.peer_id = peer_id
        self.address = address
        self.port = port
        self.config = config
        self.ssl_context = ssl_context

        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._connected = False
        self._last_used = 0.0
        self._lock = asyncio.Lock()

        # Pipeline support
        self._pending_requests: Dict[str, asyncio.Future] = {}
        self._receive_task: Optional[asyncio.Task] = None

    async def connect(self) -> bool:
        """Establish connection."""
        async with self._lock:
            if self._connected:
                return True

            try:
                self._reader, self._writer = await asyncio.wait_for(
                    asyncio.open_connection(
                        self.address,
                        self.port,
                        ssl=self.ssl_context,
                    ),
                    timeout=self.config.connect_timeout,
                )
                self._connected = True
                self._last_used = time.time()

                # Start receive task for pipelining
                self._receive_task = asyncio.create_task(self._receive_loop())

                logger.debug(f"Connected to {self.peer_id} at {self.address}:{self.port}")
                return True

            except Exception as e:
                logger.warning(f"Failed to connect to {self.peer_id}: {e}")
                return False

    async def disconnect(self) -> None:
        """Close connection."""
        async with self._lock:
            if self._receive_task:
                self._receive_task.cancel()
                try:
                    await self._receive_task
                except asyncio.CancelledError:
                    pass

            if self._writer:
                self._writer.close()
                try:
                    await self._writer.wait_closed()
                except Exception:
                    pass

            self._reader = None
            self._writer = None
            self._connected = False

            # Cancel pending requests
            for future in self._pending_requests.values():
                if not future.done():
                    future.cancel()
            self._pending_requests.clear()

    async def send(self, message: RaftMessage, wait_response: bool = True) -> Optional[RaftMessage]:
        """Send a message, optionally waiting for response."""
        if not self._connected:
            if not await self.connect():
                return None

        async with self._lock:
            try:
                # Serialize and send
                data = message.to_bytes(compress=self.config.enable_compression)

                self._writer.write(data)
                await asyncio.wait_for(
                    self._writer.drain(),
                    timeout=self.config.write_timeout,
                )
                self._last_used = time.time()

                if not wait_response:
                    return None

                # For pipelining, register future and wait
                if message.flags & MessageFlags.PIPELINE.value:
                    future = asyncio.get_event_loop().create_future()
                    self._pending_requests[message.msg_id] = future
                    return await asyncio.wait_for(future, timeout=self.config.read_timeout)

                # Wait for immediate response
                response = await self._read_message()
                return response

            except Exception as e:
                logger.warning(f"Send to {self.peer_id} failed: {e}")
                await self.disconnect()
                return None

    async def _read_message(self) -> Optional[RaftMessage]:
        """Read a single message from the stream."""
        try:
            # Read header
            header = await asyncio.wait_for(
                self._reader.readexactly(FRAME_HEADER_SIZE),
                timeout=self.config.read_timeout,
            )

            if header[:4] != FRAME_MAGIC:
                raise ValueError("Invalid frame magic")

            _, _, _, length = struct.unpack("<BBHI", header[4:])

            if length > MAX_MESSAGE_SIZE:
                raise ValueError(f"Message too large: {length}")

            # Read payload
            payload = await asyncio.wait_for(
                self._reader.readexactly(length),
                timeout=self.config.read_timeout,
            )

            return RaftMessage.from_bytes(header + payload)

        except asyncio.IncompleteReadError:
            return None
        except Exception as e:
            logger.debug(f"Read error: {e}")
            return None

    async def _receive_loop(self) -> None:
        """Background task for receiving pipelined responses."""
        while self._connected:
            try:
                message = await self._read_message()
                if message is None:
                    break

                # Match to pending request
                # For responses, msg_id should match request
                request_id = message.payload.get("request_id", message.msg_id)
                future = self._pending_requests.pop(request_id, None)

                if future and not future.done():
                    future.set_result(message)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"Receive loop error: {e}")
                break

    def is_healthy(self) -> bool:
        """Check if connection is healthy."""
        return self._connected


class ConnectionPool:
    """Pool of connections to a single peer."""

    def __init__(
        self,
        peer_id: str,
        address: str,
        port: int,
        config: ConnectionConfig,
        ssl_context: Optional[ssl.SSLContext] = None,
    ):
        self.peer_id = peer_id
        self.address = address
        self.port = port
        self.config = config
        self.ssl_context = ssl_context

        self._connections: List[TCPConnection] = []
        self._lock = asyncio.Lock()
        self._round_robin = 0

    async def get_connection(self) -> Optional[TCPConnection]:
        """Get a connection from the pool."""
        async with self._lock:
            # Find healthy connection
            for conn in self._connections:
                if conn.is_healthy():
                    return conn

            # Create new connection if under limit
            if len(self._connections) < self.config.max_connections_per_peer:
                conn = TCPConnection(
                    peer_id=self.peer_id,
                    address=self.address,
                    port=self.port,
                    config=self.config,
                    ssl_context=self.ssl_context,
                )
                if await conn.connect():
                    self._connections.append(conn)
                    return conn

            # Try to reconnect existing connection
            for conn in self._connections:
                if await conn.connect():
                    return conn

            return None

    async def close_all(self) -> None:
        """Close all connections."""
        for conn in self._connections:
            await conn.disconnect()
        self._connections.clear()


class TCPRaftTransport(RaftTransport):
    """
    TCP-based Raft transport with:
    - Connection pooling
    - Automatic reconnection
    - Message pipelining
    - TLS support
    """

    def __init__(
        self,
        node_id: str,
        listen_address: str,
        listen_port: int,
        peers: Dict[str, Tuple[str, int]],  # peer_id -> (address, port)
        config: Optional[ConnectionConfig] = None,
        ssl_context: Optional[ssl.SSLContext] = None,
    ):
        self.node_id = node_id
        self.listen_address = listen_address
        self.listen_port = listen_port
        self.peers = peers
        self.config = config or ConnectionConfig()
        self.ssl_context = ssl_context

        self._pools: Dict[str, ConnectionPool] = {}
        self._handler: Optional[Callable[[str, RaftMessage], RaftMessage]] = None
        self._server: Optional[asyncio.Server] = None
        self._shutdown = False

        # Stats
        self._stats = {
            "messages_sent": 0,
            "messages_received": 0,
            "bytes_sent": 0,
            "bytes_received": 0,
            "connection_failures": 0,
            "retries": 0,
        }

    async def start(self) -> None:
        """Start the transport."""
        # Create connection pools
        for peer_id, (address, port) in self.peers.items():
            self._pools[peer_id] = ConnectionPool(
                peer_id=peer_id,
                address=address,
                port=port,
                config=self.config,
                ssl_context=self.ssl_context,
            )

        # Start server
        self._server = await asyncio.start_server(
            self._handle_connection,
            self.listen_address,
            self.listen_port,
            ssl=self.ssl_context,
        )

        logger.info(f"Raft transport listening on {self.listen_address}:{self.listen_port}")

    async def stop(self) -> None:
        """Stop the transport."""
        self._shutdown = True

        # Close server
        if self._server:
            self._server.close()
            await self._server.wait_closed()

        # Close all connection pools
        for pool in self._pools.values():
            await pool.close_all()

        logger.info("Raft transport stopped")

    def set_handler(self, handler: Callable[[str, RaftMessage], RaftMessage]) -> None:
        """Set the message handler."""
        self._handler = handler

    async def send(
        self,
        target: str,
        message: RaftMessage,
        timeout: Optional[float] = None,
    ) -> Optional[RaftMessage]:
        """Send a message and wait for response with retries."""
        pool = self._pools.get(target)
        if not pool:
            logger.warning(f"No connection pool for peer {target}")
            return None

        timeout = timeout or self.config.read_timeout
        retries = 0
        backoff = self.config.retry_backoff_base

        while retries <= self.config.max_retries:
            conn = await pool.get_connection()
            if conn is None:
                self._stats["connection_failures"] += 1
                retries += 1
                if retries <= self.config.max_retries:
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, self.config.retry_backoff_max)
                    self._stats["retries"] += 1
                continue

            try:
                response = await asyncio.wait_for(
                    conn.send(message, wait_response=True),
                    timeout=timeout,
                )

                if response:
                    self._stats["messages_sent"] += 1
                    return response

            except asyncio.TimeoutError:
                logger.warning(f"Timeout sending to {target}")

            except Exception as e:
                logger.warning(f"Error sending to {target}: {e}")

            retries += 1
            if retries <= self.config.max_retries:
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, self.config.retry_backoff_max)
                self._stats["retries"] += 1

        return None

    async def send_async(self, target: str, message: RaftMessage) -> None:
        """Send a message without waiting for response."""
        pool = self._pools.get(target)
        if not pool:
            return

        conn = await pool.get_connection()
        if conn:
            await conn.send(message, wait_response=False)
            self._stats["messages_sent"] += 1

    async def send_pipeline(
        self,
        target: str,
        messages: List[RaftMessage],
    ) -> List[Optional[RaftMessage]]:
        """Send multiple messages in a pipeline."""
        pool = self._pools.get(target)
        if not pool:
            return [None] * len(messages)

        conn = await pool.get_connection()
        if not conn:
            return [None] * len(messages)

        # Mark messages for pipelining
        for msg in messages:
            msg.flags |= MessageFlags.PIPELINE.value

        # Send all messages
        futures = []
        for msg in messages:
            await conn.send(msg, wait_response=False)
            future = asyncio.get_event_loop().create_future()
            conn._pending_requests[msg.msg_id] = future
            futures.append(future)
            self._stats["messages_sent"] += 1

        # Wait for all responses
        responses = []
        for future in futures:
            try:
                response = await asyncio.wait_for(future, timeout=self.config.read_timeout)
                responses.append(response)
            except Exception:
                responses.append(None)

        return responses

    async def broadcast(
        self,
        message: RaftMessage,
        exclude: Optional[Set[str]] = None,
    ) -> Dict[str, Optional[RaftMessage]]:
        """Broadcast message to all peers."""
        exclude = exclude or set()

        tasks = {}
        for peer_id in self.peers:
            if peer_id not in exclude:
                tasks[peer_id] = asyncio.create_task(self.send(peer_id, message))

        results = {}
        for peer_id, task in tasks.items():
            try:
                results[peer_id] = await task
            except Exception:
                results[peer_id] = None

        return results

    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handle incoming connection."""
        peer_addr = writer.get_extra_info("peername")
        logger.debug(f"Incoming connection from {peer_addr}")

        try:
            while not self._shutdown:
                # Read message
                try:
                    header = await asyncio.wait_for(
                        reader.readexactly(FRAME_HEADER_SIZE),
                        timeout=60.0,
                    )
                except asyncio.TimeoutError:
                    continue
                except asyncio.IncompleteReadError:
                    break

                if header[:4] != FRAME_MAGIC:
                    logger.warning(f"Invalid frame from {peer_addr}")
                    break

                _, _, _, length = struct.unpack("<BBHI", header[4:])

                if length > MAX_MESSAGE_SIZE:
                    logger.warning(f"Message too large from {peer_addr}")
                    break

                payload = await reader.readexactly(length)
                message = RaftMessage.from_bytes(header + payload)

                if message is None:
                    continue

                self._stats["messages_received"] += 1
                self._stats["bytes_received"] += len(header) + length

                # Handle message
                if self._handler:
                    sender_id = message.payload.get("sender_id", "unknown")
                    try:
                        response = self._handler(sender_id, message)

                        if response:
                            # Add request_id for pipelining
                            response.payload["request_id"] = message.msg_id
                            response_data = response.to_bytes(
                                compress=self.config.enable_compression
                            )
                            writer.write(response_data)
                            await writer.drain()
                            self._stats["bytes_sent"] += len(response_data)

                    except Exception as e:
                        logger.error(f"Handler error: {e}")

        except Exception as e:
            logger.debug(f"Connection error from {peer_addr}: {e}")

        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    def add_peer(self, peer_id: str, address: str, port: int) -> None:
        """Add a new peer dynamically."""
        if peer_id not in self._pools:
            self._pools[peer_id] = ConnectionPool(
                peer_id=peer_id,
                address=address,
                port=port,
                config=self.config,
                ssl_context=self.ssl_context,
            )
            self.peers[peer_id] = (address, port)

    async def remove_peer(self, peer_id: str) -> None:
        """Remove a peer."""
        if peer_id in self._pools:
            await self._pools[peer_id].close_all()
            del self._pools[peer_id]
            del self.peers[peer_id]

    def get_stats(self) -> Dict[str, Any]:
        """Get transport statistics."""
        return {
            **self._stats,
            "active_peers": len(self._pools),
            "connected_peers": sum(
                1 for pool in self._pools.values()
                if any(c.is_healthy() for c in pool._connections)
            ),
        }


# === Utility Functions for Raft Node Integration ===

def create_append_entries_request(
    term: int,
    leader_id: str,
    prev_log_index: int,
    prev_log_term: int,
    entries: List[Dict[str, Any]],
    leader_commit: int,
) -> RaftMessage:
    """Create an AppendEntries request message."""
    return RaftMessage(
        msg_type=MessageType.APPEND_ENTRIES_REQUEST,
        payload={
            "sender_id": leader_id,
            "term": term,
            "leader_id": leader_id,
            "prev_log_index": prev_log_index,
            "prev_log_term": prev_log_term,
            "entries": entries,
            "leader_commit": leader_commit,
        },
    )


def create_append_entries_response(
    term: int,
    success: bool,
    match_index: int = 0,
    conflict_index: int = 0,
    conflict_term: int = 0,
    responder_id: str = "",
) -> RaftMessage:
    """Create an AppendEntries response message."""
    return RaftMessage(
        msg_type=MessageType.APPEND_ENTRIES_RESPONSE,
        payload={
            "sender_id": responder_id,
            "term": term,
            "success": success,
            "match_index": match_index,
            "conflict_index": conflict_index,
            "conflict_term": conflict_term,
        },
    )


def create_request_vote_request(
    term: int,
    candidate_id: str,
    last_log_index: int,
    last_log_term: int,
    is_pre_vote: bool = False,
) -> RaftMessage:
    """Create a RequestVote request message."""
    return RaftMessage(
        msg_type=MessageType.PRE_VOTE_REQUEST if is_pre_vote else MessageType.REQUEST_VOTE_REQUEST,
        payload={
            "sender_id": candidate_id,
            "term": term,
            "candidate_id": candidate_id,
            "last_log_index": last_log_index,
            "last_log_term": last_log_term,
            "is_pre_vote": is_pre_vote,
        },
    )


def create_request_vote_response(
    term: int,
    vote_granted: bool,
    responder_id: str = "",
) -> RaftMessage:
    """Create a RequestVote response message."""
    return RaftMessage(
        msg_type=MessageType.REQUEST_VOTE_RESPONSE,
        payload={
            "sender_id": responder_id,
            "term": term,
            "vote_granted": vote_granted,
        },
    )


def create_install_snapshot_request(
    term: int,
    leader_id: str,
    last_included_index: int,
    last_included_term: int,
    offset: int,
    data: bytes,
    done: bool,
) -> RaftMessage:
    """Create an InstallSnapshot request message."""
    import base64
    return RaftMessage(
        msg_type=MessageType.INSTALL_SNAPSHOT_REQUEST,
        payload={
            "sender_id": leader_id,
            "term": term,
            "leader_id": leader_id,
            "last_included_index": last_included_index,
            "last_included_term": last_included_term,
            "offset": offset,
            "data": base64.b64encode(data).decode(),
            "done": done,
        },
    )


def create_install_snapshot_response(
    term: int,
    responder_id: str = "",
) -> RaftMessage:
    """Create an InstallSnapshot response message."""
    return RaftMessage(
        msg_type=MessageType.INSTALL_SNAPSHOT_RESPONSE,
        payload={
            "sender_id": responder_id,
            "term": term,
        },
    )
