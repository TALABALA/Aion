"""
AION Distributed Communication - RPC Server

Async HTTP-based RPC server built on ``aiohttp``.  Exposes route handlers
for every RPC endpoint consumed by ``RPCClient``, plus a health endpoint.
Request payloads are validated and logged with structured logging.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Coroutine, Dict, Optional

import structlog
from aiohttp import web

from aion.distributed.communication.serialization import MessageSerializer
from aion.distributed.types import (
    AppendEntriesRequest,
    AppendEntriesResponse,
    DistributedTask,
    HeartbeatMessage,
    InstallSnapshotRequest,
    InstallSnapshotResponse,
    NodeInfo,
    NodeStatus,
    RaftLogEntry,
    VoteRequest,
    VoteResponse,
)

if TYPE_CHECKING:
    from aion.distributed.cluster.manager import ClusterManager

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# RPCServer
# ---------------------------------------------------------------------------


class RPCServer:
    """
    Async RPC server for AION node-to-node communication.

    The server binds to the address described in *node_info* and delegates
    incoming RPC calls to *cluster_manager* (or its sub-components) for
    processing.
    """

    def __init__(
        self,
        node_info: NodeInfo,
        cluster_manager: "ClusterManager",
    ) -> None:
        self._node_info = node_info
        self._cluster_manager = cluster_manager
        self._serializer = MessageSerializer(verify_checksum=False)
        self._logger = structlog.get_logger("aion.distributed.rpc.server")

        self._app: Optional[web.Application] = None
        self._runner: Optional[web.AppRunner] = None
        self._site: Optional[web.TCPSite] = None

        # Metrics
        self._request_count: int = 0
        self._error_count: int = 0
        self._started_at: Optional[datetime] = None

    # -- Lifecycle -----------------------------------------------------------

    async def start(self) -> None:
        """Build the web application, register routes, and start listening."""
        self._app = web.Application(
            middlewares=[self._logging_middleware],
        )
        self._register_routes(self._app)

        self._runner = web.AppRunner(self._app)
        await self._runner.setup()

        self._site = web.TCPSite(
            self._runner,
            host=self._node_info.host,
            port=self._node_info.port,
        )
        await self._site.start()
        self._started_at = datetime.now()

        self._logger.info(
            "rpc_server_started",
            host=self._node_info.host,
            port=self._node_info.port,
            node_id=self._node_info.id,
        )

    async def stop(self) -> None:
        """Gracefully shut down the server."""
        if self._site:
            await self._site.stop()
        if self._runner:
            await self._runner.cleanup()
        self._app = None
        self._runner = None
        self._site = None
        self._logger.info("rpc_server_stopped", node_id=self._node_info.id)

    # -- Route registration --------------------------------------------------

    def _register_routes(self, app: web.Application) -> None:
        """Register all RPC endpoint routes."""
        app.router.add_get("/health", self._handle_health)
        app.router.add_get("/rpc/stats", self._handle_stats)

        # Raft consensus
        app.router.add_post("/rpc/heartbeat", self._handle_heartbeat)
        app.router.add_post("/rpc/request_vote", self._handle_request_vote)
        app.router.add_post("/rpc/append_entries", self._handle_append_entries)
        app.router.add_post("/rpc/install_snapshot", self._handle_install_snapshot)

        # Task management
        app.router.add_post("/rpc/submit_task", self._handle_submit_task)
        app.router.add_post("/rpc/assign_task", self._handle_assign_task)

        # State synchronization
        app.router.add_post("/rpc/sync_state", self._handle_sync_state)
        app.router.add_post("/rpc/update_node_status", self._handle_update_node_status)

        # Distributed memory
        app.router.add_post("/rpc/store_memory", self._handle_store_memory)
        app.router.add_post("/rpc/get_memory", self._handle_get_memory)

        # Vector search
        app.router.add_post("/rpc/vector_search", self._handle_vector_search)
        app.router.add_post("/rpc/add_vector", self._handle_add_vector)

        # Distributed training
        app.router.add_post("/rpc/get_gradients", self._handle_get_gradients)
        app.router.add_post("/rpc/update_parameters", self._handle_update_parameters)
        app.router.add_post("/rpc/submit_gradients", self._handle_submit_gradients)
        app.router.add_post("/rpc/share_experiences", self._handle_share_experiences)

    # -- Middleware -----------------------------------------------------------

    @web.middleware
    async def _logging_middleware(
        self,
        request: web.Request,
        handler: Callable[[web.Request], Coroutine[Any, Any, web.StreamResponse]],
    ) -> web.StreamResponse:
        """Log every request and capture timing."""
        self._request_count += 1
        start = time.monotonic()
        request_id = f"req-{self._request_count}"

        self._logger.debug(
            "rpc_request_received",
            method=request.method,
            path=request.path,
            request_id=request_id,
            remote=request.remote,
        )

        try:
            response = await handler(request)
            elapsed_ms = (time.monotonic() - start) * 1000
            self._logger.debug(
                "rpc_request_completed",
                method=request.method,
                path=request.path,
                status=response.status,
                elapsed_ms=round(elapsed_ms, 2),
                request_id=request_id,
            )
            return response
        except web.HTTPException:
            raise
        except Exception as exc:
            self._error_count += 1
            elapsed_ms = (time.monotonic() - start) * 1000
            self._logger.error(
                "rpc_request_error",
                method=request.method,
                path=request.path,
                error=str(exc),
                elapsed_ms=round(elapsed_ms, 2),
                request_id=request_id,
                exc_info=True,
            )
            return web.json_response(
                {"error": "internal_server_error", "detail": str(exc)},
                status=500,
            )

    # -- Validation helper ---------------------------------------------------

    @staticmethod
    async def _read_json(request: web.Request) -> Dict[str, Any]:
        """Read and validate JSON body."""
        try:
            body = await request.json()
        except Exception as exc:
            raise web.HTTPBadRequest(
                text=f'{{"error": "invalid_json", "detail": "{exc}"}}',
                content_type="application/json",
            )
        if not isinstance(body, dict):
            raise web.HTTPBadRequest(
                text='{"error": "expected_json_object"}',
                content_type="application/json",
            )
        return body

    @staticmethod
    def _require_fields(data: Dict[str, Any], *fields: str) -> None:
        """Raise 400 if any required field is missing."""
        missing = [f for f in fields if f not in data]
        if missing:
            raise web.HTTPBadRequest(
                text=f'{{"error": "missing_fields", "fields": {missing}}}',
                content_type="application/json",
            )

    # -- Health & stats endpoints --------------------------------------------

    async def _handle_health(self, request: web.Request) -> web.Response:
        """Health check endpoint."""
        return web.json_response({
            "status": "healthy",
            "node_id": self._node_info.id,
            "node_status": self._node_info.status.value,
            "uptime_seconds": (
                (datetime.now() - self._started_at).total_seconds()
                if self._started_at else 0
            ),
        })

    async def _handle_stats(self, request: web.Request) -> web.Response:
        """Server statistics endpoint."""
        return web.json_response({
            "total_requests": self._request_count,
            "total_errors": self._error_count,
            "node_id": self._node_info.id,
            "started_at": self._started_at.isoformat() if self._started_at else None,
        })

    # -- Raft consensus handlers ---------------------------------------------

    async def _handle_heartbeat(self, request: web.Request) -> web.Response:
        """Handle incoming heartbeat."""
        data = await self._read_json(request)
        self._require_fields(data, "node_id", "term")

        self._logger.debug(
            "heartbeat_received",
            from_node=data["node_id"],
            term=data["term"],
        )

        # Delegate to cluster manager
        if hasattr(self._cluster_manager, "handle_heartbeat"):
            result = await self._cluster_manager.handle_heartbeat(data)
            return web.json_response(result or {"status": "ok"})

        return web.json_response({"status": "ok"})

    async def _handle_request_vote(self, request: web.Request) -> web.Response:
        """Handle Raft RequestVote RPC."""
        data = await self._read_json(request)
        self._require_fields(data, "term", "candidate_id", "last_log_index", "last_log_term")

        self._logger.info(
            "vote_request_received",
            candidate_id=data["candidate_id"],
            term=data["term"],
        )

        if hasattr(self._cluster_manager, "handle_vote_request"):
            vote_req = VoteRequest(
                term=data["term"],
                candidate_id=data["candidate_id"],
                last_log_index=data["last_log_index"],
                last_log_term=data["last_log_term"],
                is_pre_vote=data.get("is_pre_vote", False),
            )
            response = await self._cluster_manager.handle_vote_request(vote_req)
            if isinstance(response, VoteResponse):
                return web.json_response({
                    "term": response.term,
                    "vote_granted": response.vote_granted,
                    "voter_id": response.voter_id,
                })
            return web.json_response(response or {"term": 0, "vote_granted": False, "voter_id": ""})

        return web.json_response({
            "term": data["term"],
            "vote_granted": False,
            "voter_id": self._node_info.id,
        })

    async def _handle_append_entries(self, request: web.Request) -> web.Response:
        """Handle Raft AppendEntries RPC."""
        data = await self._read_json(request)
        self._require_fields(data, "term", "leader_id", "prev_log_index", "prev_log_term")

        self._logger.debug(
            "append_entries_received",
            leader_id=data["leader_id"],
            term=data["term"],
            entry_count=len(data.get("entries", [])),
        )

        if hasattr(self._cluster_manager, "handle_append_entries"):
            # Reconstruct RaftLogEntry objects
            entries = []
            for entry_data in data.get("entries", []):
                timestamp = entry_data.get("timestamp")
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp)
                elif isinstance(timestamp, dict) and "__datetime__" in timestamp:
                    timestamp = datetime.fromisoformat(timestamp["isoformat"])
                else:
                    timestamp = datetime.now()
                entries.append(RaftLogEntry(
                    index=entry_data.get("index", 0),
                    term=entry_data.get("term", 0),
                    command=entry_data.get("command", ""),
                    data=entry_data.get("data", {}),
                    timestamp=timestamp,
                    is_config_change=entry_data.get("is_config_change", False),
                    config_data=entry_data.get("config_data"),
                    is_noop=entry_data.get("is_noop", False),
                ))

            ae_request = AppendEntriesRequest(
                term=data["term"],
                leader_id=data["leader_id"],
                prev_log_index=data["prev_log_index"],
                prev_log_term=data["prev_log_term"],
                entries=entries,
                leader_commit=data.get("leader_commit", -1),
            )
            response = await self._cluster_manager.handle_append_entries(ae_request)
            if isinstance(response, AppendEntriesResponse):
                return web.json_response({
                    "term": response.term,
                    "success": response.success,
                    "match_index": response.match_index,
                    "node_id": response.node_id,
                    "conflict_term": response.conflict_term,
                    "conflict_index": response.conflict_index,
                })
            return web.json_response(response or {
                "term": data["term"],
                "success": False,
                "match_index": -1,
                "node_id": self._node_info.id,
            })

        return web.json_response({
            "term": data["term"],
            "success": False,
            "match_index": -1,
            "node_id": self._node_info.id,
        })

    async def _handle_install_snapshot(self, request: web.Request) -> web.Response:
        """Handle Raft InstallSnapshot RPC."""
        data = await self._read_json(request)
        self._require_fields(data, "term", "leader_id", "last_included_index", "last_included_term")

        self._logger.info(
            "install_snapshot_received",
            leader_id=data["leader_id"],
            term=data["term"],
            last_included_index=data["last_included_index"],
        )

        if hasattr(self._cluster_manager, "handle_install_snapshot"):
            snapshot_data = data.get("data", b"")
            if isinstance(snapshot_data, str):
                import base64
                snapshot_data = base64.b64decode(snapshot_data)

            snap_request = InstallSnapshotRequest(
                term=data["term"],
                leader_id=data["leader_id"],
                last_included_index=data["last_included_index"],
                last_included_term=data["last_included_term"],
                offset=data.get("offset", 0),
                data=snapshot_data,
                done=data.get("done", False),
            )
            response = await self._cluster_manager.handle_install_snapshot(snap_request)
            if isinstance(response, InstallSnapshotResponse):
                return web.json_response({
                    "term": response.term,
                    "node_id": response.node_id,
                })
            return web.json_response(response or {
                "term": data["term"],
                "node_id": self._node_info.id,
            })

        return web.json_response({
            "term": data["term"],
            "node_id": self._node_info.id,
        })

    # -- Task management handlers --------------------------------------------

    async def _handle_submit_task(self, request: web.Request) -> web.Response:
        """Handle task submission."""
        data = await self._read_json(request)

        self._logger.info(
            "task_submitted",
            task_id=data.get("id", "unknown"),
            task_type=data.get("task_type", "unknown"),
        )

        if hasattr(self._cluster_manager, "handle_submit_task"):
            task = self._serializer.deserialize_task(data)
            result = await self._cluster_manager.handle_submit_task(task)
            return web.json_response(result or {"status": "accepted", "task_id": data.get("id")})

        return web.json_response({"status": "accepted", "task_id": data.get("id")})

    async def _handle_assign_task(self, request: web.Request) -> web.Response:
        """Handle direct task assignment."""
        data = await self._read_json(request)

        self._logger.info(
            "task_assigned",
            task_id=data.get("id", "unknown"),
            task_type=data.get("task_type", "unknown"),
        )

        if hasattr(self._cluster_manager, "handle_assign_task"):
            task = self._serializer.deserialize_task(data)
            result = await self._cluster_manager.handle_assign_task(task)
            return web.json_response(result or {"status": "accepted", "task_id": data.get("id")})

        return web.json_response({"status": "accepted", "task_id": data.get("id")})

    # -- State synchronization handlers --------------------------------------

    async def _handle_sync_state(self, request: web.Request) -> web.Response:
        """Handle state synchronization."""
        data = await self._read_json(request)

        self._logger.debug("sync_state_received")

        if hasattr(self._cluster_manager, "handle_sync_state"):
            result = await self._cluster_manager.handle_sync_state(data)
            return web.json_response(result or {"status": "ok"})

        return web.json_response({"status": "ok"})

    async def _handle_update_node_status(self, request: web.Request) -> web.Response:
        """Handle node status updates."""
        data = await self._read_json(request)

        self._logger.debug(
            "node_status_update_received",
            node_id=data.get("id", "unknown"),
            status=data.get("status", "unknown"),
        )

        if hasattr(self._cluster_manager, "handle_update_node_status"):
            node_info = self._serializer.deserialize_node(data)
            result = await self._cluster_manager.handle_update_node_status(node_info)
            return web.json_response(result or {"status": "ok"})

        return web.json_response({"status": "ok"})

    # -- Distributed memory handlers -----------------------------------------

    async def _handle_store_memory(self, request: web.Request) -> web.Response:
        """Handle memory store operation."""
        data = await self._read_json(request)
        self._require_fields(data, "key", "value")

        self._logger.debug("store_memory_received", key=data["key"])

        if hasattr(self._cluster_manager, "handle_store_memory"):
            result = await self._cluster_manager.handle_store_memory(
                data["key"], data["value"],
            )
            return web.json_response(result or {"status": "ok"})

        return web.json_response({"status": "ok"})

    async def _handle_get_memory(self, request: web.Request) -> web.Response:
        """Handle memory retrieval."""
        data = await self._read_json(request)
        self._require_fields(data, "key")

        self._logger.debug("get_memory_received", key=data["key"])

        if hasattr(self._cluster_manager, "handle_get_memory"):
            value = await self._cluster_manager.handle_get_memory(data["key"])
            return web.json_response({"key": data["key"], "value": value})

        return web.json_response({"key": data["key"], "value": None})

    # -- Vector search handlers ----------------------------------------------

    async def _handle_vector_search(self, request: web.Request) -> web.Response:
        """Handle vector similarity search."""
        data = await self._read_json(request)
        self._require_fields(data, "query_vector")

        k = data.get("k", 10)
        self._logger.debug("vector_search_received", k=k)

        if hasattr(self._cluster_manager, "handle_vector_search"):
            results = await self._cluster_manager.handle_vector_search(
                data["query_vector"], k,
            )
            return web.json_response({"results": results or []})

        return web.json_response({"results": []})

    async def _handle_add_vector(self, request: web.Request) -> web.Response:
        """Handle adding a vector to the index."""
        data = await self._read_json(request)
        self._require_fields(data, "id", "vector")

        self._logger.debug("add_vector_received", vector_id=data["id"])

        if hasattr(self._cluster_manager, "handle_add_vector"):
            result = await self._cluster_manager.handle_add_vector(
                data["id"], data["vector"],
            )
            return web.json_response(result or {"status": "ok"})

        return web.json_response({"status": "ok"})

    # -- Distributed training handlers ---------------------------------------

    async def _handle_get_gradients(self, request: web.Request) -> web.Response:
        """Handle gradient retrieval."""
        _ = await self._read_json(request)

        self._logger.debug("get_gradients_received")

        if hasattr(self._cluster_manager, "handle_get_gradients"):
            gradients = await self._cluster_manager.handle_get_gradients()
            return web.json_response(gradients or {})

        return web.json_response({})

    async def _handle_update_parameters(self, request: web.Request) -> web.Response:
        """Handle model parameter updates."""
        data = await self._read_json(request)

        self._logger.debug("update_parameters_received")

        if hasattr(self._cluster_manager, "handle_update_parameters"):
            result = await self._cluster_manager.handle_update_parameters(data)
            return web.json_response(result or {"status": "ok"})

        return web.json_response({"status": "ok"})

    async def _handle_submit_gradients(self, request: web.Request) -> web.Response:
        """Handle gradient submission."""
        data = await self._read_json(request)

        self._logger.debug("submit_gradients_received")

        if hasattr(self._cluster_manager, "handle_submit_gradients"):
            result = await self._cluster_manager.handle_submit_gradients(data)
            return web.json_response(result or {"status": "ok"})

        return web.json_response({"status": "ok"})

    async def _handle_share_experiences(self, request: web.Request) -> web.Response:
        """Handle experience sharing."""
        data = await self._read_json(request)

        experiences = data.get("experiences", [])
        self._logger.debug(
            "share_experiences_received",
            count=len(experiences),
        )

        if hasattr(self._cluster_manager, "handle_share_experiences"):
            result = await self._cluster_manager.handle_share_experiences(experiences)
            return web.json_response(result or {"status": "ok", "received": len(experiences)})

        return web.json_response({"status": "ok", "received": len(experiences)})

    # -- Diagnostics ---------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Return server statistics."""
        return {
            "total_requests": self._request_count,
            "total_errors": self._error_count,
            "node_id": self._node_info.id,
            "started_at": self._started_at.isoformat() if self._started_at else None,
            "listening": self._site is not None,
        }
