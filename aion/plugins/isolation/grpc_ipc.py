"""
gRPC-Based Inter-Process Communication

High-performance IPC using gRPC for plugin isolation.
Significantly faster than pickle/JSON serialization with
strong typing and streaming support.

Requires:
    pip install grpcio grpcio-tools protobuf

Protocol Buffer definitions would be in plugin_ipc.proto:
```protobuf
syntax = "proto3";

package aion.plugins;

service PluginService {
    rpc Initialize(InitializeRequest) returns (InitializeResponse);
    rpc Shutdown(ShutdownRequest) returns (ShutdownResponse);
    rpc Execute(ExecuteRequest) returns (ExecuteResponse);
    rpc ExecuteStream(ExecuteRequest) returns (stream ExecuteResponse);
    rpc Healthcheck(HealthcheckRequest) returns (HealthcheckResponse);
    rpc GetMetrics(MetricsRequest) returns (MetricsResponse);
}

message InitializeRequest {
    string plugin_id = 1;
    bytes config = 2;  // JSON-encoded config
    map<string, string> metadata = 3;
}

message InitializeResponse {
    bool success = 1;
    string error = 2;
    bytes manifest = 3;  // JSON-encoded manifest
}

message ExecuteRequest {
    string method = 1;
    bytes args = 2;  // msgpack/JSON encoded args
    bytes kwargs = 3;
    string trace_context = 4;
    int64 timeout_ms = 5;
}

message ExecuteResponse {
    bool success = 1;
    bytes result = 2;
    string error = 3;
    string error_type = 4;
    bytes traceback = 5;
    int64 execution_time_ms = 6;
}

message HealthcheckRequest {}

message HealthcheckResponse {
    bool healthy = 1;
    string status = 2;
    map<string, string> details = 3;
}

message MetricsRequest {}

message MetricsResponse {
    int64 memory_bytes = 1;
    double cpu_percent = 2;
    int64 call_count = 3;
    int64 error_count = 4;
    double avg_latency_ms = 5;
}

message ShutdownRequest {
    bool force = 1;
    int64 timeout_ms = 2;
}

message ShutdownResponse {
    bool success = 1;
}
```
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
import time
import traceback
from abc import ABC, abstractmethod
from concurrent import futures
from dataclasses import dataclass, field
from enum import Enum
from multiprocessing import Process
from typing import Any, AsyncIterator, Callable, Dict, Iterator, Optional, Type

import structlog

logger = structlog.get_logger(__name__)

# Try to import gRPC
try:
    import grpc
    from grpc import aio as grpc_aio
    GRPC_AVAILABLE = True
except ImportError:
    GRPC_AVAILABLE = False
    logger.warning("grpcio not installed. gRPC IPC not available.")

# Try to import msgpack for faster serialization
try:
    import msgpack
    MSGPACK_AVAILABLE = True
except ImportError:
    MSGPACK_AVAILABLE = False


class SerializationFormat(str, Enum):
    """Serialization format for IPC messages."""
    JSON = "json"
    MSGPACK = "msgpack"


@dataclass
class GRPCConfig:
    """Configuration for gRPC IPC."""

    # Server settings
    host: str = "127.0.0.1"
    port: int = 0  # 0 = auto-assign
    max_workers: int = 10
    max_message_size: int = 100 * 1024 * 1024  # 100MB

    # Serialization
    serialization: SerializationFormat = SerializationFormat.MSGPACK

    # Timeouts
    connect_timeout: float = 10.0
    call_timeout: float = 60.0
    shutdown_timeout: float = 30.0

    # Keep-alive
    keepalive_time_ms: int = 30000  # 30 seconds
    keepalive_timeout_ms: int = 10000  # 10 seconds

    # SSL/TLS (optional)
    use_tls: bool = False
    cert_file: Optional[str] = None
    key_file: Optional[str] = None
    ca_file: Optional[str] = None


def serialize(data: Any, format: SerializationFormat) -> bytes:
    """Serialize data to bytes."""
    if format == SerializationFormat.MSGPACK and MSGPACK_AVAILABLE:
        return msgpack.packb(data, use_bin_type=True)
    return json.dumps(data).encode("utf-8")


def deserialize(data: bytes, format: SerializationFormat) -> Any:
    """Deserialize bytes to data."""
    if format == SerializationFormat.MSGPACK and MSGPACK_AVAILABLE:
        return msgpack.unpackb(data, raw=False)
    return json.loads(data.decode("utf-8"))


# ============================================================
# Protocol Messages (without protobuf, using dataclasses)
# ============================================================

@dataclass
class InitializeRequest:
    """Request to initialize a plugin."""
    plugin_id: str
    config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class InitializeResponse:
    """Response from plugin initialization."""
    success: bool
    error: str = ""
    manifest: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecuteRequest:
    """Request to execute a plugin method."""
    method: str
    args: tuple = ()
    kwargs: Dict[str, Any] = field(default_factory=dict)
    trace_context: str = ""
    timeout_ms: int = 60000


@dataclass
class ExecuteResponse:
    """Response from plugin method execution."""
    success: bool
    result: Any = None
    error: str = ""
    error_type: str = ""
    traceback: str = ""
    execution_time_ms: int = 0


@dataclass
class HealthcheckResponse:
    """Response from healthcheck."""
    healthy: bool
    status: str = ""
    details: Dict[str, str] = field(default_factory=dict)


@dataclass
class MetricsResponse:
    """Response containing plugin metrics."""
    memory_bytes: int = 0
    cpu_percent: float = 0.0
    call_count: int = 0
    error_count: int = 0
    avg_latency_ms: float = 0.0


# ============================================================
# Server Implementation (runs in plugin subprocess)
# ============================================================

class PluginServicer:
    """
    gRPC servicer that handles plugin execution.

    Runs in the plugin subprocess and exposes the plugin
    instance via gRPC.
    """

    def __init__(
        self,
        plugin_class: Type,
        config: GRPCConfig,
    ):
        self.plugin_class = plugin_class
        self.config = config
        self.plugin_instance = None
        self._call_count = 0
        self._error_count = 0
        self._total_latency = 0.0

    async def Initialize(
        self,
        request_data: bytes,
        context,
    ) -> bytes:
        """Handle initialization request."""
        try:
            request = deserialize(request_data, self.config.serialization)
            req = InitializeRequest(**request)

            # Create plugin instance
            self.plugin_instance = self.plugin_class()

            # Call plugin's initialize method
            if hasattr(self.plugin_instance, "initialize"):
                await self.plugin_instance.initialize(None, req.config)

            # Get manifest
            manifest = {}
            if hasattr(self.plugin_instance, "get_manifest"):
                manifest_obj = self.plugin_instance.get_manifest()
                if hasattr(manifest_obj, "__dict__"):
                    manifest = manifest_obj.__dict__

            response = InitializeResponse(
                success=True,
                manifest=manifest,
            )

        except Exception as e:
            logger.error(f"Plugin initialization failed: {e}")
            response = InitializeResponse(
                success=False,
                error=str(e),
            )

        return serialize(response.__dict__, self.config.serialization)

    async def Execute(
        self,
        request_data: bytes,
        context,
    ) -> bytes:
        """Handle method execution request."""
        start_time = time.time()
        self._call_count += 1

        try:
            request = deserialize(request_data, self.config.serialization)
            req = ExecuteRequest(**request)

            if not self.plugin_instance:
                raise RuntimeError("Plugin not initialized")

            # Get method
            method = getattr(self.plugin_instance, req.method, None)
            if method is None:
                raise AttributeError(f"Plugin has no method: {req.method}")

            # Execute with timeout
            if asyncio.iscoroutinefunction(method):
                result = await asyncio.wait_for(
                    method(*req.args, **req.kwargs),
                    timeout=req.timeout_ms / 1000,
                )
            else:
                result = method(*req.args, **req.kwargs)

            execution_time = int((time.time() - start_time) * 1000)
            self._total_latency += execution_time

            response = ExecuteResponse(
                success=True,
                result=result,
                execution_time_ms=execution_time,
            )

        except asyncio.TimeoutError:
            self._error_count += 1
            response = ExecuteResponse(
                success=False,
                error="Execution timed out",
                error_type="TimeoutError",
            )

        except Exception as e:
            self._error_count += 1
            response = ExecuteResponse(
                success=False,
                error=str(e),
                error_type=type(e).__name__,
                traceback=traceback.format_exc(),
            )

        return serialize(response.__dict__, self.config.serialization)

    async def Healthcheck(
        self,
        request_data: bytes,
        context,
    ) -> bytes:
        """Handle healthcheck request."""
        healthy = self.plugin_instance is not None

        details = {}
        if self.plugin_instance and hasattr(self.plugin_instance, "health_check"):
            try:
                health = await self.plugin_instance.health_check()
                healthy = health.get("healthy", True)
                details = {k: str(v) for k, v in health.items()}
            except Exception as e:
                healthy = False
                details["error"] = str(e)

        response = HealthcheckResponse(
            healthy=healthy,
            status="running" if healthy else "unhealthy",
            details=details,
        )

        return serialize(response.__dict__, self.config.serialization)

    async def GetMetrics(
        self,
        request_data: bytes,
        context,
    ) -> bytes:
        """Handle metrics request."""
        import psutil
        process = psutil.Process()

        response = MetricsResponse(
            memory_bytes=process.memory_info().rss,
            cpu_percent=process.cpu_percent(),
            call_count=self._call_count,
            error_count=self._error_count,
            avg_latency_ms=(
                self._total_latency / self._call_count
                if self._call_count > 0 else 0.0
            ),
        )

        return serialize(response.__dict__, self.config.serialization)

    async def Shutdown(
        self,
        request_data: bytes,
        context,
    ) -> bytes:
        """Handle shutdown request."""
        try:
            if self.plugin_instance and hasattr(self.plugin_instance, "shutdown"):
                await self.plugin_instance.shutdown()
            return serialize({"success": True}, self.config.serialization)
        except Exception as e:
            return serialize({"success": False, "error": str(e)}, self.config.serialization)


class GRPCPluginServer:
    """
    gRPC server for running plugins in isolation.

    Runs in the plugin subprocess.
    """

    def __init__(
        self,
        plugin_class: Type,
        config: Optional[GRPCConfig] = None,
    ):
        self.plugin_class = plugin_class
        self.config = config or GRPCConfig()
        self.servicer = PluginServicer(plugin_class, self.config)
        self._server = None
        self._port = 0

    async def start(self) -> int:
        """Start the gRPC server."""
        if not GRPC_AVAILABLE:
            raise RuntimeError("grpcio not installed")

        # Create server
        self._server = grpc_aio.server(
            futures.ThreadPoolExecutor(max_workers=self.config.max_workers),
            options=[
                ("grpc.max_receive_message_length", self.config.max_message_size),
                ("grpc.max_send_message_length", self.config.max_message_size),
                ("grpc.keepalive_time_ms", self.config.keepalive_time_ms),
                ("grpc.keepalive_timeout_ms", self.config.keepalive_timeout_ms),
            ],
        )

        # Add generic handler for our protocol
        # In a real implementation, this would use generated protobuf stubs
        # For now, we use a generic unary handler
        self._server.add_generic_rpc_handlers([
            GenericRpcHandler(self.servicer, self.config),
        ])

        # Bind to port
        address = f"{self.config.host}:{self.config.port}"

        if self.config.use_tls and self.config.cert_file and self.config.key_file:
            # Secure channel
            with open(self.config.cert_file, "rb") as f:
                cert = f.read()
            with open(self.config.key_file, "rb") as f:
                key = f.read()
            credentials = grpc.ssl_server_credentials([(key, cert)])
            self._port = self._server.add_secure_port(address, credentials)
        else:
            self._port = self._server.add_insecure_port(address)

        await self._server.start()

        logger.info(
            "gRPC plugin server started",
            port=self._port,
            tls=self.config.use_tls,
        )

        return self._port

    async def wait_for_termination(self) -> None:
        """Wait for server termination."""
        if self._server:
            await self._server.wait_for_termination()

    async def stop(self, grace: float = 5.0) -> None:
        """Stop the gRPC server."""
        if self._server:
            await self._server.stop(grace)
            logger.info("gRPC plugin server stopped")


class GenericRpcHandler(grpc.GenericRpcHandler):
    """Generic RPC handler for our plugin protocol."""

    def __init__(self, servicer: PluginServicer, config: GRPCConfig):
        self.servicer = servicer
        self.config = config
        self._methods = {
            "/aion.plugins.PluginService/Initialize": servicer.Initialize,
            "/aion.plugins.PluginService/Execute": servicer.Execute,
            "/aion.plugins.PluginService/Healthcheck": servicer.Healthcheck,
            "/aion.plugins.PluginService/GetMetrics": servicer.GetMetrics,
            "/aion.plugins.PluginService/Shutdown": servicer.Shutdown,
        }

    def service(self, handler_call_details):
        method = self._methods.get(handler_call_details.method)
        if method:
            return grpc.unary_unary_rpc_method_handler(
                self._wrap_method(method),
            )
        return None

    def _wrap_method(self, method):
        async def handler(request, context):
            return await method(request, context)
        return handler


# ============================================================
# Client Implementation (runs in main process)
# ============================================================

class GRPCPluginClient:
    """
    gRPC client for communicating with isolated plugins.

    Runs in the main process and communicates with plugin
    subprocesses via gRPC.
    """

    def __init__(
        self,
        host: str,
        port: int,
        config: Optional[GRPCConfig] = None,
    ):
        self.host = host
        self.port = port
        self.config = config or GRPCConfig()
        self._channel = None
        self._connected = False

    async def connect(self) -> None:
        """Connect to the plugin server."""
        if not GRPC_AVAILABLE:
            raise RuntimeError("grpcio not installed")

        address = f"{self.host}:{self.port}"

        options = [
            ("grpc.max_receive_message_length", self.config.max_message_size),
            ("grpc.max_send_message_length", self.config.max_message_size),
            ("grpc.keepalive_time_ms", self.config.keepalive_time_ms),
            ("grpc.keepalive_timeout_ms", self.config.keepalive_timeout_ms),
        ]

        if self.config.use_tls and self.config.ca_file:
            with open(self.config.ca_file, "rb") as f:
                ca_cert = f.read()
            credentials = grpc.ssl_channel_credentials(ca_cert)
            self._channel = grpc_aio.secure_channel(address, credentials, options)
        else:
            self._channel = grpc_aio.insecure_channel(address, options)

        # Wait for channel to be ready
        try:
            await asyncio.wait_for(
                self._channel.channel_ready(),
                timeout=self.config.connect_timeout,
            )
            self._connected = True
            logger.debug(f"Connected to plugin at {address}")
        except asyncio.TimeoutError:
            raise ConnectionError(f"Timeout connecting to plugin at {address}")

    async def disconnect(self) -> None:
        """Disconnect from the plugin server."""
        if self._channel:
            await self._channel.close()
            self._connected = False

    async def _call(self, method: str, request_data: Any) -> bytes:
        """Make a unary RPC call."""
        if not self._connected:
            raise RuntimeError("Not connected")

        request_bytes = serialize(request_data, self.config.serialization)

        response_bytes = await self._channel.unary_unary(
            f"/aion.plugins.PluginService/{method}",
            request_serializer=lambda x: x,
            response_deserializer=lambda x: x,
        )(request_bytes, timeout=self.config.call_timeout)

        return response_bytes

    async def initialize(
        self,
        plugin_id: str,
        config: Dict[str, Any],
        metadata: Optional[Dict[str, str]] = None,
    ) -> InitializeResponse:
        """Initialize the plugin."""
        request = {
            "plugin_id": plugin_id,
            "config": config,
            "metadata": metadata or {},
        }
        response_bytes = await self._call("Initialize", request)
        response_data = deserialize(response_bytes, self.config.serialization)
        return InitializeResponse(**response_data)

    async def execute(
        self,
        method: str,
        args: tuple = (),
        kwargs: Optional[Dict[str, Any]] = None,
        trace_context: str = "",
        timeout_ms: int = 60000,
    ) -> ExecuteResponse:
        """Execute a plugin method."""
        request = {
            "method": method,
            "args": args,
            "kwargs": kwargs or {},
            "trace_context": trace_context,
            "timeout_ms": timeout_ms,
        }
        response_bytes = await self._call("Execute", request)
        response_data = deserialize(response_bytes, self.config.serialization)
        return ExecuteResponse(**response_data)

    async def healthcheck(self) -> HealthcheckResponse:
        """Check plugin health."""
        response_bytes = await self._call("Healthcheck", {})
        response_data = deserialize(response_bytes, self.config.serialization)
        return HealthcheckResponse(**response_data)

    async def get_metrics(self) -> MetricsResponse:
        """Get plugin metrics."""
        response_bytes = await self._call("GetMetrics", {})
        response_data = deserialize(response_bytes, self.config.serialization)
        return MetricsResponse(**response_data)

    async def shutdown(self, force: bool = False, timeout_ms: int = 30000) -> bool:
        """Shutdown the plugin."""
        request = {"force": force, "timeout_ms": timeout_ms}
        response_bytes = await self._call("Shutdown", request)
        response_data = deserialize(response_bytes, self.config.serialization)
        return response_data.get("success", False)


# ============================================================
# Process Isolation with gRPC
# ============================================================

class GRPCIsolatedPlugin:
    """
    Plugin isolation using gRPC IPC.

    Spawns the plugin in a subprocess and communicates via gRPC.
    """

    def __init__(
        self,
        plugin_id: str,
        plugin_module: str,
        plugin_class: str,
        config: Optional[GRPCConfig] = None,
    ):
        self.plugin_id = plugin_id
        self.plugin_module = plugin_module
        self.plugin_class = plugin_class
        self.config = config or GRPCConfig()

        self._process: Optional[Process] = None
        self._client: Optional[GRPCPluginClient] = None
        self._port: Optional[int] = None

    async def start(self) -> None:
        """Start the isolated plugin."""
        # Find a free port
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            self._port = s.getsockname()[1]

        # Start subprocess
        self._process = Process(
            target=_run_plugin_server,
            args=(
                self.plugin_module,
                self.plugin_class,
                self.config.host,
                self._port,
            ),
        )
        self._process.start()

        # Wait for server to be ready
        await asyncio.sleep(0.5)

        # Connect client
        self._client = GRPCPluginClient(
            self.config.host,
            self._port,
            self.config,
        )
        await self._client.connect()

        logger.info(
            "Started isolated plugin",
            plugin_id=self.plugin_id,
            pid=self._process.pid,
            port=self._port,
        )

    async def initialize(self, config: Dict[str, Any]) -> InitializeResponse:
        """Initialize the plugin."""
        if not self._client:
            raise RuntimeError("Plugin not started")
        return await self._client.initialize(self.plugin_id, config)

    async def execute(
        self,
        method: str,
        *args,
        **kwargs,
    ) -> Any:
        """Execute a plugin method."""
        if not self._client:
            raise RuntimeError("Plugin not started")

        response = await self._client.execute(
            method,
            args=args,
            kwargs=kwargs,
            timeout_ms=int(self.config.call_timeout * 1000),
        )

        if not response.success:
            raise RuntimeError(f"{response.error_type}: {response.error}")

        return response.result

    async def healthcheck(self) -> HealthcheckResponse:
        """Check plugin health."""
        if not self._client:
            raise RuntimeError("Plugin not started")
        return await self._client.healthcheck()

    async def get_metrics(self) -> MetricsResponse:
        """Get plugin metrics."""
        if not self._client:
            raise RuntimeError("Plugin not started")
        return await self._client.get_metrics()

    async def stop(self, force: bool = False) -> None:
        """Stop the isolated plugin."""
        if self._client:
            try:
                await self._client.shutdown(force=force)
            except Exception as e:
                logger.warning(f"Error during shutdown: {e}")
            await self._client.disconnect()

        if self._process:
            if force:
                self._process.kill()
            else:
                self._process.terminate()

            self._process.join(timeout=5.0)
            if self._process.is_alive():
                self._process.kill()
                self._process.join()

        logger.info(
            "Stopped isolated plugin",
            plugin_id=self.plugin_id,
        )


def _run_plugin_server(
    module_name: str,
    class_name: str,
    host: str,
    port: int,
) -> None:
    """
    Entry point for plugin subprocess.

    Loads the plugin class and starts the gRPC server.
    """
    import importlib

    # Load plugin class
    module = importlib.import_module(module_name)
    plugin_class = getattr(module, class_name)

    # Create and run server
    config = GRPCConfig(host=host, port=port)
    server = GRPCPluginServer(plugin_class, config)

    async def run():
        await server.start()
        await server.wait_for_termination()

    asyncio.run(run())


# ============================================================
# Performance Comparison
# ============================================================

class IPCBenchmark:
    """Benchmark different IPC methods."""

    @staticmethod
    async def benchmark_grpc(
        client: GRPCPluginClient,
        iterations: int = 1000,
    ) -> Dict[str, float]:
        """Benchmark gRPC IPC."""
        import time

        latencies = []
        errors = 0

        for _ in range(iterations):
            start = time.perf_counter()
            try:
                await client.healthcheck()
                latencies.append((time.perf_counter() - start) * 1000)
            except Exception:
                errors += 1

        return {
            "iterations": iterations,
            "errors": errors,
            "min_latency_ms": min(latencies) if latencies else 0,
            "max_latency_ms": max(latencies) if latencies else 0,
            "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
            "p99_latency_ms": sorted(latencies)[int(len(latencies) * 0.99)] if latencies else 0,
        }
