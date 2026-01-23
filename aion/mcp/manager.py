"""
AION MCP Manager

Central manager for all MCP connections:
- Server lifecycle management
- Connection pooling
- Health monitoring
- Unified tool access
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime
from typing import Any, Callable, Optional, TYPE_CHECKING

import structlog

from aion.mcp.types import (
    ServerConfig,
    ConnectedServer,
    Tool,
    Resource,
    Prompt,
    ToolResult,
    ResourceContent,
    PromptMessage,
    TransportType,
)
from aion.mcp.client.client import MCPClient, MCPError
from aion.mcp.registry import ServerRegistry
from aion.mcp.credentials import CredentialManager

if TYPE_CHECKING:
    from aion.mcp.bridge import MCPToolBridge

logger = structlog.get_logger(__name__)


class MCPManager:
    """
    Central manager for MCP connections.

    Provides:
    - Multi-server management
    - Unified tool/resource/prompt access
    - Health monitoring
    - Auto-reconnection
    - Tool bridging to AION's tool system
    """

    def __init__(
        self,
        registry: Optional[ServerRegistry] = None,
        credentials: Optional[CredentialManager] = None,
        health_check_interval: float = 30.0,
        auto_reconnect: bool = True,
        max_reconnect_attempts: int = 3,
    ):
        """
        Initialize MCP manager.

        Args:
            registry: Server registry (uses default if None)
            credentials: Credential manager (uses default if None)
            health_check_interval: Interval between health checks
            auto_reconnect: Whether to auto-reconnect failed connections
            max_reconnect_attempts: Maximum reconnection attempts
        """
        self.registry = registry or ServerRegistry()
        self.credentials = credentials or CredentialManager()
        self.health_check_interval = health_check_interval
        self.auto_reconnect = auto_reconnect
        self.max_reconnect_attempts = max_reconnect_attempts

        # Connected clients
        self._clients: dict[str, MCPClient] = {}

        # Tool bridge (created lazily)
        self._bridge: Optional["MCPToolBridge"] = None

        # Background tasks
        self._health_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

        # Statistics
        self._stats = {
            "connections_established": 0,
            "connections_failed": 0,
            "reconnections": 0,
            "tool_calls": 0,
            "tool_errors": 0,
            "resource_reads": 0,
            "prompt_gets": 0,
        }

        self._initialized = False
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize the MCP manager."""
        if self._initialized:
            return

        logger.info("Initializing MCP Manager")

        # Load server configurations
        await self.registry.load()

        # Initialize credentials
        await self.credentials.initialize()

        # Connect to enabled servers
        for config in self.registry.get_enabled_servers():
            try:
                await self.connect_server(config.name)
            except Exception as e:
                logger.warning(
                    "Failed to connect to server during initialization",
                    server=config.name,
                    error=str(e),
                )

        # Start health monitor
        self._health_task = asyncio.create_task(self._health_monitor_loop())

        self._initialized = True
        logger.info(
            "MCP Manager initialized",
            servers_connected=len(self.get_connected_servers()),
            total_tools=sum(
                len(c.list_tools()) for c in self._clients.values() if c.connected
            ),
        )

    async def shutdown(self) -> None:
        """Shutdown the MCP manager."""
        logger.info("Shutting down MCP Manager")

        self._shutdown_event.set()

        # Stop health monitor
        if self._health_task:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass

        # Disconnect all servers
        for name in list(self._clients.keys()):
            await self.disconnect_server(name)

        self._initialized = False
        logger.info("MCP Manager shutdown complete")

    # === Server Management ===

    async def connect_server(self, name: str) -> bool:
        """
        Connect to an MCP server.

        Args:
            name: Server name (from registry)

        Returns:
            True if connected successfully
        """
        async with self._lock:
            # Get configuration
            config = self.registry.get_server(name)
            if not config:
                raise ValueError(f"Unknown server: {name}")

            # Check if already connected
            if name in self._clients and self._clients[name].connected:
                return True

            # Resolve credentials if needed
            if config.credential_id:
                creds = await self.credentials.get(config.credential_id)
                if creds:
                    config.env.update(creds)

            # Create and connect client
            client = MCPClient(config)

            try:
                await client.connect()
                self._clients[name] = client
                self._stats["connections_established"] += 1

                logger.info(
                    "Connected to MCP server",
                    server=name,
                    tools=len(client.list_tools()),
                    resources=len(client.list_resources()),
                )

                return True

            except Exception as e:
                self._stats["connections_failed"] += 1
                logger.error(
                    "Failed to connect to MCP server",
                    server=name,
                    error=str(e),
                )
                raise

    async def disconnect_server(self, name: str) -> bool:
        """
        Disconnect from an MCP server.

        Args:
            name: Server name

        Returns:
            True if disconnected
        """
        async with self._lock:
            client = self._clients.pop(name, None)
            if client:
                await client.disconnect()
                logger.info("Disconnected from MCP server", server=name)
                return True
            return False

    async def reconnect_server(self, name: str) -> bool:
        """
        Reconnect to an MCP server.

        Args:
            name: Server name

        Returns:
            True if reconnected successfully
        """
        await self.disconnect_server(name)
        self._stats["reconnections"] += 1
        return await self.connect_server(name)

    def is_connected(self, name: str) -> bool:
        """Check if server is connected."""
        client = self._clients.get(name)
        return client is not None and client.connected

    # === Tool Operations ===

    async def call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> ToolResult:
        """
        Call a tool on a specific server.

        Args:
            server_name: Name of the server
            tool_name: Name of the tool
            arguments: Tool arguments

        Returns:
            ToolResult
        """
        client = self._clients.get(server_name)
        if not client or not client.connected:
            raise RuntimeError(f"Server not connected: {server_name}")

        self._stats["tool_calls"] += 1

        try:
            return await client.call_tool(tool_name, arguments)
        except Exception as e:
            self._stats["tool_errors"] += 1
            raise

    async def call_tool_by_name(
        self,
        full_tool_name: str,
        arguments: dict[str, Any],
    ) -> ToolResult:
        """
        Call a tool by its full name (server:tool).

        Args:
            full_tool_name: Full tool name (e.g., "postgres:query")
            arguments: Tool arguments

        Returns:
            ToolResult
        """
        if ":" in full_tool_name:
            server_name, tool_name = full_tool_name.split(":", 1)
        else:
            # Find server that has this tool
            server_name = self._find_server_with_tool(full_tool_name)
            tool_name = full_tool_name

        if not server_name:
            raise ValueError(f"Tool not found: {full_tool_name}")

        return await self.call_tool(server_name, tool_name, arguments)

    def _find_server_with_tool(self, tool_name: str) -> Optional[str]:
        """Find the server that has a specific tool."""
        for name, client in self._clients.items():
            if client.connected:
                for tool in client.list_tools():
                    if tool.name == tool_name:
                        return name
        return None

    def list_all_tools(self) -> dict[str, list[Tool]]:
        """Get all tools from all connected servers."""
        result = {}
        for name, client in self._clients.items():
            if client.connected:
                result[name] = client.list_tools()
        return result

    def get_tools_flat(self) -> list[tuple[str, Tool]]:
        """Get all tools as flat list of (server_name, tool) tuples."""
        result = []
        for name, client in self._clients.items():
            if client.connected:
                for tool in client.list_tools():
                    result.append((name, tool))
        return result

    def get_tool(self, server_name: str, tool_name: str) -> Optional[Tool]:
        """Get a specific tool."""
        client = self._clients.get(server_name)
        if client and client.connected:
            return client.get_tool(tool_name)
        return None

    # === Resource Operations ===

    async def read_resource(
        self,
        server_name: str,
        uri: str,
    ) -> ResourceContent:
        """
        Read a resource from a server.

        Args:
            server_name: Name of the server
            uri: Resource URI

        Returns:
            ResourceContent
        """
        client = self._clients.get(server_name)
        if not client or not client.connected:
            raise RuntimeError(f"Server not connected: {server_name}")

        self._stats["resource_reads"] += 1
        return await client.read_resource(uri)

    def list_all_resources(self) -> dict[str, list[Resource]]:
        """Get all resources from all connected servers."""
        result = {}
        for name, client in self._clients.items():
            if client.connected:
                result[name] = client.list_resources()
        return result

    # === Prompt Operations ===

    async def get_prompt(
        self,
        server_name: str,
        prompt_name: str,
        arguments: Optional[dict[str, str]] = None,
    ) -> list[PromptMessage]:
        """
        Get a prompt from a server.

        Args:
            server_name: Name of the server
            prompt_name: Name of the prompt
            arguments: Prompt arguments

        Returns:
            List of prompt messages
        """
        client = self._clients.get(server_name)
        if not client or not client.connected:
            raise RuntimeError(f"Server not connected: {server_name}")

        self._stats["prompt_gets"] += 1
        return await client.get_prompt(prompt_name, arguments)

    def list_all_prompts(self) -> dict[str, list[Prompt]]:
        """Get all prompts from all connected servers."""
        result = {}
        for name, client in self._clients.items():
            if client.connected:
                result[name] = client.list_prompts()
        return result

    # === Bridge to AION Tools ===

    def get_tool_bridge(self) -> "MCPToolBridge":
        """Get the tool bridge for integration with AION's tool system."""
        if not self._bridge:
            from aion.mcp.bridge import MCPToolBridge
            self._bridge = MCPToolBridge(self)
        return self._bridge

    # === Health Monitoring ===

    async def _health_monitor_loop(self) -> None:
        """Background health monitoring loop."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(self.health_check_interval)

                for name, client in list(self._clients.items()):
                    if not client.connected and self.auto_reconnect:
                        config = self.registry.get_server(name)
                        if config and config.auto_reconnect:
                            logger.info(f"Attempting reconnection to {name}")
                            try:
                                await self.reconnect_server(name)
                            except Exception as e:
                                logger.warning(
                                    "Reconnection failed",
                                    server=name,
                                    error=str(e),
                                )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Health monitor error", error=str(e))

    async def ping_server(self, name: str) -> bool:
        """
        Ping a server to check connectivity.

        Args:
            name: Server name

        Returns:
            True if server responds
        """
        client = self._clients.get(name)
        if not client:
            return False
        return await client.ping()

    # === Status ===

    def get_server_states(self) -> dict[str, ConnectedServer]:
        """Get state of all servers."""
        return {
            name: client.get_state()
            for name, client in self._clients.items()
        }

    def get_server_state(self, name: str) -> Optional[ConnectedServer]:
        """Get state of a specific server."""
        client = self._clients.get(name)
        if client:
            return client.get_state()
        return None

    def get_connected_servers(self) -> list[str]:
        """Get names of connected servers."""
        return [
            name for name, client in self._clients.items()
            if client.connected
        ]

    def get_stats(self) -> dict[str, Any]:
        """Get manager statistics."""
        return {
            **self._stats,
            "servers_configured": len(self.registry.get_all_servers()),
            "servers_connected": len(self.get_connected_servers()),
            "total_tools": sum(
                len(client.list_tools())
                for client in self._clients.values()
                if client.connected
            ),
            "total_resources": sum(
                len(client.list_resources())
                for client in self._clients.values()
                if client.connected
            ),
            "total_prompts": sum(
                len(client.list_prompts())
                for client in self._clients.values()
                if client.connected
            ),
        }

    # === Context Manager ===

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.shutdown()
