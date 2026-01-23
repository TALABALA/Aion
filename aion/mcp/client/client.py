"""
AION MCP Client

Connects to MCP servers and invokes their capabilities.
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime
from typing import Any, AsyncGenerator, Callable, Optional

import structlog

from aion.mcp.types import (
    MCP_PROTOCOL_VERSION,
    JsonRpcRequest,
    JsonRpcResponse,
    JsonRpcNotification,
    ServerCapabilities,
    ClientCapabilities,
    RootsCapability,
    SamplingCapability,
    Implementation,
    InitializeResult,
    Tool,
    ToolResult,
    Resource,
    ResourceContent,
    Prompt,
    PromptMessage,
    ServerConfig,
    ConnectedServer,
    TransportType,
    TextContent,
)
from aion.mcp.protocol import (
    MCPProtocol,
    MCPProtocolError,
    MCPMethods,
    MCPMessageBuilder,
)
from aion.mcp.transports.base import Transport
from aion.mcp.transports.stdio import StdioTransport
from aion.mcp.transports.sse import SSETransport
from aion.mcp.transports.websocket import WebSocketTransport

logger = structlog.get_logger(__name__)


class MCPError(Exception):
    """MCP protocol error."""

    def __init__(self, code: int, message: str, data: Any = None):
        self.code = code
        self.message = message
        self.data = data
        super().__init__(f"MCP Error {code}: {message}")


class MCPClient:
    """
    MCP Client for connecting to MCP servers.

    Handles:
    - Connection lifecycle
    - Capability negotiation
    - Tool/resource/prompt discovery
    - Request/response handling
    """

    def __init__(
        self,
        config: ServerConfig,
        client_info: Optional[Implementation] = None,
    ):
        """
        Initialize MCP client.

        Args:
            config: Server configuration
            client_info: Client implementation info
        """
        self.config = config
        self.client_info = client_info or Implementation(
            name="aion",
            version="1.0.0",
        )

        # Protocol handler
        self._protocol = MCPProtocol()
        self._message_builder = MCPMessageBuilder(self._protocol)

        # Transport
        self._transport: Optional[Transport] = None

        # Server state
        self._server_info: Optional[Implementation] = None
        self._capabilities: Optional[ServerCapabilities] = None
        self._tools: list[Tool] = []
        self._resources: list[Resource] = []
        self._prompts: list[Prompt] = []
        self._instructions: Optional[str] = None

        # Request tracking
        self._pending_requests: dict[str, asyncio.Future] = {}

        # Notification handlers
        self._notification_handlers: dict[str, list[Callable]] = {}

        # State
        self._connected = False
        self._initialized = False

        # Background task
        self._receive_task: Optional[asyncio.Task] = None

        # Statistics
        self._stats = ConnectedServer(config=config)

    async def connect(self) -> None:
        """Connect to the MCP server."""
        if self._connected:
            return

        logger.info(
            "Connecting to MCP server",
            name=self.config.name,
            transport=self.config.transport.value,
        )

        # Create transport
        self._transport = self._create_transport()
        await self._transport.connect()

        # Start receive loop
        self._receive_task = asyncio.create_task(self._receive_loop())

        self._connected = True
        self._stats.last_connected = datetime.now()

        # Initialize protocol
        await self._initialize()

        logger.info(
            "Connected to MCP server",
            name=self.config.name,
            server=self._server_info.name if self._server_info else "unknown",
        )

    async def disconnect(self) -> None:
        """Disconnect from the MCP server."""
        if not self._connected:
            return

        logger.info("Disconnecting from MCP server", name=self.config.name)

        # Cancel receive loop
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass

        # Cancel pending requests
        for request_id, future in self._pending_requests.items():
            if not future.done():
                future.cancel()
        self._pending_requests.clear()

        # Close transport
        if self._transport:
            await self._transport.close()

        self._connected = False
        self._initialized = False
        self._stats.connected = False

        logger.info("Disconnected from MCP server", name=self.config.name)

    def _create_transport(self) -> Transport:
        """Create the appropriate transport based on config."""
        if self.config.transport == TransportType.STDIO:
            if not self.config.command:
                raise ValueError("Stdio transport requires 'command' in config")
            return StdioTransport(
                command=self.config.command,
                args=self.config.args,
                env=self.config.env,
                cwd=self.config.cwd,
            )
        elif self.config.transport == TransportType.SSE:
            if not self.config.url:
                raise ValueError("SSE transport requires 'url' in config")
            return SSETransport(
                url=self.config.url,
                headers=self.config.headers,
                timeout=self.config.timeout,
            )
        elif self.config.transport == TransportType.WEBSOCKET:
            url = self.config.ws_url or self.config.url
            if not url:
                raise ValueError("WebSocket transport requires 'ws_url' or 'url' in config")
            return WebSocketTransport(
                url=url,
                headers=self.config.headers,
            )
        else:
            raise ValueError(f"Unknown transport type: {self.config.transport}")

    async def _initialize(self) -> None:
        """Initialize the MCP protocol."""
        # Send initialize request
        result = await self._request(MCPMethods.INITIALIZE, {
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "capabilities": ClientCapabilities(
                roots=RootsCapability(listChanged=True),
                sampling=SamplingCapability(),
            ).to_dict(),
            "clientInfo": self.client_info.to_dict(),
        })

        # Parse response
        init_result = InitializeResult.from_dict(result)
        self._server_info = init_result.serverInfo
        self._capabilities = init_result.capabilities
        self._instructions = init_result.instructions

        # Send initialized notification
        await self._notify(MCPMethods.INITIALIZED, {})

        # Discover capabilities
        await self._discover_capabilities()

        self._initialized = True
        self._stats.connected = True
        self._stats.capabilities = self._capabilities

    async def _discover_capabilities(self) -> None:
        """Discover server's tools, resources, and prompts."""
        # Discover tools
        if self._capabilities and self._capabilities.tools:
            try:
                result = await self._request(MCPMethods.TOOLS_LIST, {})
                self._tools = [
                    Tool.from_dict(t)
                    for t in result.get("tools", [])
                ]
                self._stats.tools = self._tools
                logger.info(
                    "Discovered tools",
                    server=self.config.name,
                    count=len(self._tools),
                )
            except Exception as e:
                logger.warning("Failed to list tools", error=str(e))

        # Discover resources
        if self._capabilities and self._capabilities.resources:
            try:
                result = await self._request(MCPMethods.RESOURCES_LIST, {})
                self._resources = [
                    Resource.from_dict(r)
                    for r in result.get("resources", [])
                ]
                self._stats.resources = self._resources
                logger.info(
                    "Discovered resources",
                    server=self.config.name,
                    count=len(self._resources),
                )
            except Exception as e:
                logger.warning("Failed to list resources", error=str(e))

        # Discover prompts
        if self._capabilities and self._capabilities.prompts:
            try:
                result = await self._request(MCPMethods.PROMPTS_LIST, {})
                self._prompts = [
                    Prompt.from_dict(p)
                    for p in result.get("prompts", [])
                ]
                self._stats.prompts = self._prompts
                logger.info(
                    "Discovered prompts",
                    server=self.config.name,
                    count=len(self._prompts),
                )
            except Exception as e:
                logger.warning("Failed to list prompts", error=str(e))

    # === Tool Operations ===

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> ToolResult:
        """
        Call a tool on the server.

        Args:
            name: Tool name
            arguments: Tool arguments

        Returns:
            ToolResult with content
        """
        if not self._initialized:
            raise RuntimeError("Client not initialized")

        start_time = time.monotonic()

        try:
            result = await self._request(MCPMethods.TOOLS_CALL, {
                "name": name,
                "arguments": arguments,
            })

            latency_ms = (time.monotonic() - start_time) * 1000
            self._stats.requests_successful += 1
            self._stats.total_latency_ms += latency_ms

            return ToolResult.from_dict(result)

        except Exception as e:
            self._stats.errors += 1
            raise

    def list_tools(self) -> list[Tool]:
        """Get list of available tools."""
        return self._tools.copy()

    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a specific tool by name."""
        for tool in self._tools:
            if tool.name == name:
                return tool
        return None

    # === Resource Operations ===

    async def read_resource(self, uri: str) -> ResourceContent:
        """
        Read a resource from the server.

        Args:
            uri: Resource URI

        Returns:
            ResourceContent
        """
        if not self._initialized:
            raise RuntimeError("Client not initialized")

        result = await self._request(MCPMethods.RESOURCES_READ, {"uri": uri})

        contents = result.get("contents", [])
        if not contents:
            raise ValueError(f"Resource not found: {uri}")

        return ResourceContent.from_dict(contents[0])

    async def subscribe_resource(self, uri: str) -> None:
        """Subscribe to resource updates."""
        if not self._initialized:
            raise RuntimeError("Client not initialized")

        if self._capabilities and self._capabilities.resources:
            if self._capabilities.resources.subscribe:
                await self._request(MCPMethods.RESOURCES_SUBSCRIBE, {"uri": uri})

    async def unsubscribe_resource(self, uri: str) -> None:
        """Unsubscribe from resource updates."""
        if not self._initialized:
            raise RuntimeError("Client not initialized")

        await self._request(MCPMethods.RESOURCES_UNSUBSCRIBE, {"uri": uri})

    def list_resources(self) -> list[Resource]:
        """Get list of available resources."""
        return self._resources.copy()

    def get_resource(self, uri: str) -> Optional[Resource]:
        """Get a specific resource by URI."""
        for resource in self._resources:
            if resource.uri == uri:
                return resource
        return None

    # === Prompt Operations ===

    async def get_prompt(
        self,
        name: str,
        arguments: Optional[dict[str, str]] = None,
    ) -> list[PromptMessage]:
        """
        Get a prompt from the server.

        Args:
            name: Prompt name
            arguments: Prompt arguments

        Returns:
            List of prompt messages
        """
        if not self._initialized:
            raise RuntimeError("Client not initialized")

        params = {"name": name}
        if arguments:
            params["arguments"] = arguments

        result = await self._request(MCPMethods.PROMPTS_GET, params)

        return [
            PromptMessage.from_dict(m)
            for m in result.get("messages", [])
        ]

    def list_prompts(self) -> list[Prompt]:
        """Get list of available prompts."""
        return self._prompts.copy()

    def get_prompt_info(self, name: str) -> Optional[Prompt]:
        """Get a specific prompt by name."""
        for prompt in self._prompts:
            if prompt.name == name:
                return prompt
        return None

    # === Ping ===

    async def ping(self) -> bool:
        """
        Ping the server to check connectivity.

        Returns:
            True if server responds
        """
        if not self._initialized:
            return False

        try:
            await self._request(MCPMethods.PING, {})
            return True
        except Exception:
            return False

    # === Notification Handling ===

    def on_notification(self, method: str, handler: Callable) -> None:
        """
        Register a notification handler.

        Args:
            method: Notification method to handle
            handler: Callback function
        """
        if method not in self._notification_handlers:
            self._notification_handlers[method] = []
        self._notification_handlers[method].append(handler)

    def off_notification(self, method: str, handler: Callable) -> None:
        """
        Unregister a notification handler.

        Args:
            method: Notification method
            handler: Callback function to remove
        """
        if method in self._notification_handlers:
            handlers = self._notification_handlers[method]
            if handler in handlers:
                handlers.remove(handler)

    # === Low-level Protocol ===

    async def _request(
        self,
        method: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Send a request and wait for response."""
        request = self._protocol.create_request(method, params)

        # Create future for response
        future: asyncio.Future = asyncio.Future()
        self._pending_requests[str(request.id)] = future

        self._stats.requests_sent += 1

        try:
            # Send request
            await self._transport.send(self._protocol.serialize(request))

            # Wait for response
            response = await asyncio.wait_for(future, timeout=self.config.timeout)

            if response.error:
                raise MCPError(
                    response.error.get("code", -1),
                    response.error.get("message", "Unknown error"),
                    response.error.get("data"),
                )

            return response.result or {}

        except asyncio.TimeoutError:
            self._stats.errors += 1
            raise MCPError(-1, f"Request timed out: {method}")
        finally:
            self._pending_requests.pop(str(request.id), None)

    async def _notify(self, method: str, params: dict[str, Any]) -> None:
        """Send a notification (no response expected)."""
        notification = self._protocol.create_notification(method, params)
        await self._transport.send(self._protocol.serialize(notification))

    async def _receive_loop(self) -> None:
        """Background loop to receive messages."""
        try:
            async for message in self._transport.receive():
                await self._handle_message(message)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("Receive loop error", error=str(e))
            self._connected = False

    async def _handle_message(self, message: str) -> None:
        """Handle an incoming message."""
        try:
            parsed = self._protocol.parse(message)
        except Exception as e:
            logger.warning("Invalid message received", error=str(e))
            return

        # Check if it's a response
        if self._protocol.is_response(parsed):
            response = parsed
            request_id = str(response.id)

            future = self._pending_requests.get(request_id)
            if future and not future.done():
                future.set_result(response)

        # Check if it's a notification
        elif self._protocol.is_notification(parsed):
            notification = parsed
            method = notification.method
            params = notification.params or {}

            handlers = self._notification_handlers.get(method, [])
            for handler in handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(params)
                    else:
                        handler(params)
                except Exception as e:
                    logger.error(
                        "Notification handler error",
                        method=method,
                        error=str(e),
                    )

            # Handle list changed notifications
            if method == MCPMethods.NOTIFICATION_TOOLS_LIST_CHANGED:
                await self._refresh_tools()
            elif method == MCPMethods.NOTIFICATION_RESOURCES_LIST_CHANGED:
                await self._refresh_resources()
            elif method == MCPMethods.NOTIFICATION_PROMPTS_LIST_CHANGED:
                await self._refresh_prompts()

        # Check if it's a request (server to client)
        elif self._protocol.is_request(parsed):
            request = parsed
            # Handle server-to-client requests
            await self._handle_server_request(request)

    async def _handle_server_request(self, request: JsonRpcRequest) -> None:
        """Handle a request from the server."""
        method = request.method
        params = request.params or {}

        try:
            if method == MCPMethods.ROOTS_LIST:
                # Return empty roots list for now
                result = {"roots": []}
            elif method == MCPMethods.SAMPLING_CREATE_MESSAGE:
                # Sampling not implemented yet
                raise MCPError(-32601, "Sampling not implemented")
            else:
                raise MCPError(-32601, f"Unknown method: {method}")

            response = self._protocol.create_response(request.id, result=result)

        except MCPError as e:
            response = self._protocol.create_response(
                request.id,
                error=e.to_error() if hasattr(e, 'to_error') else {
                    "code": e.code,
                    "message": e.message,
                    "data": e.data,
                },
            )
        except Exception as e:
            response = self._protocol.create_response(
                request.id,
                error={"code": -32603, "message": str(e)},
            )

        await self._transport.send(self._protocol.serialize(response))

    async def _refresh_tools(self) -> None:
        """Refresh the tools list."""
        try:
            result = await self._request(MCPMethods.TOOLS_LIST, {})
            self._tools = [Tool.from_dict(t) for t in result.get("tools", [])]
            self._stats.tools = self._tools
        except Exception as e:
            logger.warning("Failed to refresh tools", error=str(e))

    async def _refresh_resources(self) -> None:
        """Refresh the resources list."""
        try:
            result = await self._request(MCPMethods.RESOURCES_LIST, {})
            self._resources = [Resource.from_dict(r) for r in result.get("resources", [])]
            self._stats.resources = self._resources
        except Exception as e:
            logger.warning("Failed to refresh resources", error=str(e))

    async def _refresh_prompts(self) -> None:
        """Refresh the prompts list."""
        try:
            result = await self._request(MCPMethods.PROMPTS_LIST, {})
            self._prompts = [Prompt.from_dict(p) for p in result.get("prompts", [])]
            self._stats.prompts = self._prompts
        except Exception as e:
            logger.warning("Failed to refresh prompts", error=str(e))

    # === Properties ===

    @property
    def connected(self) -> bool:
        """Check if connected to server."""
        return self._connected

    @property
    def initialized(self) -> bool:
        """Check if protocol is initialized."""
        return self._initialized

    @property
    def server_info(self) -> Optional[Implementation]:
        """Get server implementation info."""
        return self._server_info

    @property
    def capabilities(self) -> Optional[ServerCapabilities]:
        """Get server capabilities."""
        return self._capabilities

    @property
    def instructions(self) -> Optional[str]:
        """Get server instructions."""
        return self._instructions

    def get_state(self) -> ConnectedServer:
        """Get current server state."""
        return ConnectedServer(
            config=self.config,
            capabilities=self._capabilities,
            tools=self._tools,
            resources=self._resources,
            prompts=self._prompts,
            connected=self._connected,
            last_connected=self._stats.last_connected,
            last_error=self._stats.last_error,
            requests_sent=self._stats.requests_sent,
            requests_successful=self._stats.requests_successful,
            errors=self._stats.errors,
            total_latency_ms=self._stats.total_latency_ms,
        )
