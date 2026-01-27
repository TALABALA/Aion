"""
AION Distributed Communication - RPC Client

Async RPC client for node-to-node communication in the AION cluster.
Uses ``aiohttp`` for HTTP-based RPC (simulated gRPC-style), with:

* Connection pooling with configurable limits
* Per-endpoint circuit breaker (threshold / timeout / half-open reset)
* Exponential back-off retry logic
* Request timeout handling
* Automatic serialization/deserialization via ``MessageSerializer``
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import aiohttp
import structlog

from aion.distributed.communication.serialization import MessageSerializer
from aion.distributed.types import (
    AppendEntriesRequest,
    AppendEntriesResponse,
    AppendEntriesResponse as _AER,
    DistributedTask,
    HeartbeatMessage,
    InstallSnapshotRequest,
    InstallSnapshotResponse,
    NodeInfo,
    VoteRequest,
    VoteResponse,
)

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreaker:
    """Per-endpoint circuit breaker state."""

    failure_threshold: int = 5
    reset_timeout: float = 30.0
    half_open_max_calls: int = 1

    # Mutable state
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: float = 0.0
    half_open_calls: int = 0

    def record_success(self) -> None:
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            # Fully close after a successful half-open probe
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.half_open_calls = 0
        elif self.state == CircuitState.CLOSED:
            self.success_count += 1
            self.failure_count = max(0, self.failure_count - 1)

    def record_failure(self) -> None:
        self.failure_count += 1
        self.last_failure_time = time.monotonic()
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            self.half_open_calls = 0
        elif self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN

    def allow_request(self) -> bool:
        if self.state == CircuitState.CLOSED:
            return True
        if self.state == CircuitState.OPEN:
            elapsed = time.monotonic() - self.last_failure_time
            if elapsed >= self.reset_timeout:
                self.state = CircuitState.HALF_OPEN
                self.half_open_calls = 0
                return True
            return False
        # HALF_OPEN
        if self.half_open_calls < self.half_open_max_calls:
            self.half_open_calls += 1
            return True
        return False


# ---------------------------------------------------------------------------
# RPCClient
# ---------------------------------------------------------------------------


class RPCClient:
    """
    Async RPC client for AION node-to-node communication.

    All public methods are coroutines that resolve the target node by
    ``address`` (``host:port``) and issue HTTP POST requests to well-known
    endpoints.
    """

    def __init__(
        self,
        *,
        max_connections: int = 100,
        max_connections_per_host: int = 10,
        request_timeout: float = 10.0,
        connect_timeout: float = 5.0,
        max_retries: int = 3,
        retry_base_delay: float = 0.5,
        retry_max_delay: float = 10.0,
        circuit_breaker_threshold: int = 5,
        circuit_breaker_timeout: float = 30.0,
    ) -> None:
        self._max_connections = max_connections
        self._max_per_host = max_connections_per_host
        self._request_timeout = request_timeout
        self._connect_timeout = connect_timeout
        self._max_retries = max_retries
        self._retry_base_delay = retry_base_delay
        self._retry_max_delay = retry_max_delay
        self._cb_threshold = circuit_breaker_threshold
        self._cb_timeout = circuit_breaker_timeout

        self._session: Optional[aiohttp.ClientSession] = None
        self._serializer = MessageSerializer(verify_checksum=False)
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._logger = structlog.get_logger("aion.distributed.rpc.client")
        self._request_count: int = 0
        self._error_count: int = 0

    # -- Lifecycle -----------------------------------------------------------

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(
                limit=self._max_connections,
                limit_per_host=self._max_per_host,
                enable_cleanup_closed=True,
            )
            timeout = aiohttp.ClientTimeout(
                total=self._request_timeout,
                connect=self._connect_timeout,
            )
            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
            )
        return self._session

    async def close(self) -> None:
        """Close the underlying HTTP session and connection pool."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
        self._logger.info("rpc_client_closed")

    # -- Circuit breaker helper ----------------------------------------------

    def _get_breaker(self, address: str) -> CircuitBreaker:
        if address not in self._circuit_breakers:
            self._circuit_breakers[address] = CircuitBreaker(
                failure_threshold=self._cb_threshold,
                reset_timeout=self._cb_timeout,
            )
        return self._circuit_breakers[address]

    # -- Core request helper -------------------------------------------------

    async def _make_request(
        self,
        address: str,
        endpoint: str,
        data: Any,
        *,
        timeout_override: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """Issue an HTTP POST to ``http://{address}/rpc/{endpoint}``.

        Implements retry with exponential back-off and circuit-breaker
        gating.
        """
        breaker = self._get_breaker(address)
        url = f"http://{address}/rpc/{endpoint}"

        for attempt in range(1, self._max_retries + 1):
            if not breaker.allow_request():
                self._logger.warning(
                    "circuit_breaker_open",
                    address=address,
                    endpoint=endpoint,
                    state=breaker.state.value,
                )
                return None

            try:
                session = await self._ensure_session()
                self._request_count += 1

                # Serialise payload
                payload = self._serializer.serialize_task(data) if hasattr(data, "__dataclass_fields__") else data
                if isinstance(payload, bytes):
                    payload = payload.decode("utf-8")

                timeout = aiohttp.ClientTimeout(
                    total=timeout_override or self._request_timeout,
                )

                async with session.post(
                    url,
                    json=payload,
                    timeout=timeout,
                ) as resp:
                    if resp.status == 200:
                        breaker.record_success()
                        body = await resp.json()
                        return body
                    else:
                        self._logger.warning(
                            "rpc_request_failed",
                            address=address,
                            endpoint=endpoint,
                            status=resp.status,
                            attempt=attempt,
                        )
                        breaker.record_failure()

            except asyncio.TimeoutError:
                self._error_count += 1
                breaker.record_failure()
                self._logger.warning(
                    "rpc_request_timeout",
                    address=address,
                    endpoint=endpoint,
                    attempt=attempt,
                )
            except aiohttp.ClientError as exc:
                self._error_count += 1
                breaker.record_failure()
                self._logger.warning(
                    "rpc_request_error",
                    address=address,
                    endpoint=endpoint,
                    attempt=attempt,
                    error=str(exc),
                )
            except Exception as exc:
                self._error_count += 1
                breaker.record_failure()
                self._logger.error(
                    "rpc_request_unexpected_error",
                    address=address,
                    endpoint=endpoint,
                    attempt=attempt,
                    error=str(exc),
                    exc_info=True,
                )

            # Exponential back-off
            if attempt < self._max_retries:
                delay = min(
                    self._retry_base_delay * (2 ** (attempt - 1)),
                    self._retry_max_delay,
                )
                await asyncio.sleep(delay)

        self._logger.error(
            "rpc_request_exhausted_retries",
            address=address,
            endpoint=endpoint,
            max_retries=self._max_retries,
        )
        return None

    # -- Serialization helpers -----------------------------------------------

    def _serialize_dataclass(self, obj: Any) -> Dict[str, Any]:
        """Convert a dataclass to a JSON-compatible dict via MessageSerializer."""
        from dataclasses import asdict, is_dataclass as _is_dc

        if _is_dc(obj) and not isinstance(obj, type):
            import json as _json
            from aion.distributed.communication.serialization import _DistributedEncoder

            raw = asdict(obj)
            return _json.loads(_json.dumps(raw, cls=_DistributedEncoder))
        if isinstance(obj, dict):
            return obj
        return {"data": obj}

    # -- Heartbeat -----------------------------------------------------------

    async def send_heartbeat(
        self,
        address: str,
        message: HeartbeatMessage,
    ) -> Optional[Dict[str, Any]]:
        """Send a heartbeat to the node at *address*."""
        data = self._serialize_dataclass(message)
        return await self._make_request(address, "heartbeat", data)

    # -- Raft consensus RPCs -------------------------------------------------

    async def request_vote(
        self,
        address: str,
        request: VoteRequest,
    ) -> Optional[VoteResponse]:
        """Send a RequestVote RPC."""
        data = self._serialize_dataclass(request)
        result = await self._make_request(address, "request_vote", data)
        if result is None:
            return None
        try:
            return VoteResponse(
                term=result["term"],
                vote_granted=result["vote_granted"],
                voter_id=result.get("voter_id", ""),
            )
        except (KeyError, TypeError) as exc:
            self._logger.error("vote_response_parse_error", error=str(exc))
            return None

    async def append_entries(
        self,
        address: str,
        request: AppendEntriesRequest,
    ) -> Optional[AppendEntriesResponse]:
        """Send an AppendEntries RPC."""
        data = self._serialize_dataclass(request)
        result = await self._make_request(address, "append_entries", data)
        if result is None:
            return None
        try:
            return AppendEntriesResponse(
                term=result["term"],
                success=result["success"],
                match_index=result.get("match_index", -1),
                node_id=result.get("node_id", ""),
                conflict_term=result.get("conflict_term"),
                conflict_index=result.get("conflict_index"),
            )
        except (KeyError, TypeError) as exc:
            self._logger.error("append_entries_response_parse_error", error=str(exc))
            return None

    async def install_snapshot(
        self,
        address: str,
        request: InstallSnapshotRequest,
    ) -> Optional[InstallSnapshotResponse]:
        """Send an InstallSnapshot RPC."""
        data = self._serialize_dataclass(request)
        result = await self._make_request(address, "install_snapshot", data)
        if result is None:
            return None
        try:
            return InstallSnapshotResponse(
                term=result["term"],
                node_id=result.get("node_id", ""),
            )
        except (KeyError, TypeError) as exc:
            self._logger.error("install_snapshot_response_parse_error", error=str(exc))
            return None

    # -- Task operations -----------------------------------------------------

    async def submit_task(
        self,
        address: str,
        task: DistributedTask,
    ) -> Optional[Dict[str, Any]]:
        """Submit a new task to the node at *address*."""
        data = self._serializer.serialize_task(task)
        return await self._make_request(address, "submit_task", data)

    async def assign_task(
        self,
        address: str,
        task: DistributedTask,
    ) -> Optional[Dict[str, Any]]:
        """Assign a task directly to the node at *address*."""
        data = self._serializer.serialize_task(task)
        return await self._make_request(address, "assign_task", data)

    # -- State synchronization -----------------------------------------------

    async def sync_state(
        self,
        address: str,
        state_snapshot: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Push a state snapshot to the node at *address*."""
        return await self._make_request(address, "sync_state", state_snapshot)

    async def update_node_status(
        self,
        address: str,
        node_info: NodeInfo,
    ) -> Optional[Dict[str, Any]]:
        """Inform *address* about an updated node status."""
        data = self._serializer.serialize_node(node_info)
        return await self._make_request(address, "update_node_status", data)

    # -- Distributed memory --------------------------------------------------

    async def store_memory(
        self,
        address: str,
        key: str,
        value: Any,
    ) -> Optional[Dict[str, Any]]:
        """Store a key/value pair on the node at *address*."""
        return await self._make_request(
            address, "store_memory", {"key": key, "value": value},
        )

    async def get_memory(
        self,
        address: str,
        key: str,
    ) -> Optional[Any]:
        """Retrieve a value by key from the node at *address*."""
        result = await self._make_request(
            address, "get_memory", {"key": key},
        )
        if result is None:
            return None
        return result.get("value")

    # -- Vector search -------------------------------------------------------

    async def vector_search(
        self,
        address: str,
        query_vector: List[float],
        k: int = 10,
    ) -> List[Any]:
        """Perform a vector similarity search on the remote node."""
        result = await self._make_request(
            address, "vector_search", {"query_vector": query_vector, "k": k},
        )
        if result is None:
            return []
        return result.get("results", [])

    async def add_vector(
        self,
        address: str,
        id: str,
        vector: List[float],
    ) -> Optional[Dict[str, Any]]:
        """Add a vector to the remote node's index."""
        return await self._make_request(
            address, "add_vector", {"id": id, "vector": vector},
        )

    # -- Distributed training ------------------------------------------------

    async def get_gradients(
        self,
        address: str,
    ) -> Optional[Dict[str, Any]]:
        """Retrieve accumulated gradients from the remote node."""
        return await self._make_request(address, "get_gradients", {})

    async def update_parameters(
        self,
        address: str,
        params: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Push updated model parameters to the remote node."""
        return await self._make_request(address, "update_parameters", params)

    async def submit_gradients(
        self,
        address: str,
        gradients: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Submit gradient updates to the remote node."""
        return await self._make_request(address, "submit_gradients", gradients)

    async def share_experiences(
        self,
        address: str,
        experiences: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Share collected experiences with a peer node."""
        return await self._make_request(
            address, "share_experiences", {"experiences": experiences},
        )

    # -- Diagnostics ---------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Return RPC client statistics."""
        breaker_states = {
            addr: {
                "state": cb.state.value,
                "failure_count": cb.failure_count,
                "success_count": cb.success_count,
            }
            for addr, cb in self._circuit_breakers.items()
        }
        return {
            "total_requests": self._request_count,
            "total_errors": self._error_count,
            "circuit_breakers": breaker_states,
            "session_open": self._session is not None and not self._session.closed,
        }
