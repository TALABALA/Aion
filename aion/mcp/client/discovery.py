"""
AION MCP Capability Discovery

Utilities for discovering and analyzing MCP server capabilities.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import structlog

from aion.mcp.types import (
    ServerConfig,
    ServerCapabilities,
    Tool,
    Resource,
    Prompt,
    TransportType,
)
from aion.mcp.client.client import MCPClient

logger = structlog.get_logger(__name__)


@dataclass
class ServerDiscoveryResult:
    """Result of discovering a server's capabilities."""
    config: ServerConfig
    server_name: str
    server_version: str
    capabilities: ServerCapabilities
    tools: list[Tool]
    resources: list[Resource]
    prompts: list[Prompt]
    instructions: Optional[str]
    discovery_time: datetime
    latency_ms: float
    error: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "config_name": self.config.name,
            "server_name": self.server_name,
            "server_version": self.server_version,
            "capabilities": self.capabilities.to_dict() if self.capabilities else None,
            "tools": [t.to_dict() for t in self.tools],
            "resources": [r.to_dict() for r in self.resources],
            "prompts": [p.to_dict() for p in self.prompts],
            "instructions": self.instructions,
            "discovery_time": self.discovery_time.isoformat(),
            "latency_ms": self.latency_ms,
            "error": self.error,
        }


@dataclass
class ToolDiscoveryResult:
    """Result of discovering tools across servers."""
    tool: Tool
    server_name: str
    server_config: ServerConfig

    @property
    def full_name(self) -> str:
        """Get the fully qualified tool name."""
        return f"{self.server_name}:{self.tool.name}"


class MCPDiscovery:
    """
    MCP capability discovery service.

    Helps discover and catalog capabilities across multiple MCP servers.
    """

    def __init__(self, timeout: float = 30.0):
        """
        Initialize discovery service.

        Args:
            timeout: Timeout for discovery operations
        """
        self.timeout = timeout

    async def discover_server(
        self,
        config: ServerConfig,
    ) -> ServerDiscoveryResult:
        """
        Discover capabilities of a single server.

        Args:
            config: Server configuration

        Returns:
            ServerDiscoveryResult
        """
        import time
        start_time = time.monotonic()

        client = MCPClient(config)

        try:
            await asyncio.wait_for(
                client.connect(),
                timeout=self.timeout,
            )

            latency_ms = (time.monotonic() - start_time) * 1000

            result = ServerDiscoveryResult(
                config=config,
                server_name=client.server_info.name if client.server_info else "unknown",
                server_version=client.server_info.version if client.server_info else "0.0.0",
                capabilities=client.capabilities,
                tools=client.list_tools(),
                resources=client.list_resources(),
                prompts=client.list_prompts(),
                instructions=client.instructions,
                discovery_time=datetime.now(),
                latency_ms=latency_ms,
            )

            logger.info(
                "Server discovery complete",
                server=config.name,
                tools=len(result.tools),
                resources=len(result.resources),
                prompts=len(result.prompts),
            )

            return result

        except asyncio.TimeoutError:
            return ServerDiscoveryResult(
                config=config,
                server_name="unknown",
                server_version="0.0.0",
                capabilities=ServerCapabilities(),
                tools=[],
                resources=[],
                prompts=[],
                instructions=None,
                discovery_time=datetime.now(),
                latency_ms=(time.monotonic() - start_time) * 1000,
                error="Connection timeout",
            )

        except Exception as e:
            return ServerDiscoveryResult(
                config=config,
                server_name="unknown",
                server_version="0.0.0",
                capabilities=ServerCapabilities(),
                tools=[],
                resources=[],
                prompts=[],
                instructions=None,
                discovery_time=datetime.now(),
                latency_ms=(time.monotonic() - start_time) * 1000,
                error=str(e),
            )

        finally:
            await client.disconnect()

    async def discover_servers(
        self,
        configs: list[ServerConfig],
        parallel: bool = True,
    ) -> list[ServerDiscoveryResult]:
        """
        Discover capabilities of multiple servers.

        Args:
            configs: List of server configurations
            parallel: Whether to discover in parallel

        Returns:
            List of discovery results
        """
        if parallel:
            tasks = [
                self.discover_server(config)
                for config in configs
            ]
            return await asyncio.gather(*tasks)
        else:
            results = []
            for config in configs:
                result = await self.discover_server(config)
                results.append(result)
            return results

    def aggregate_tools(
        self,
        results: list[ServerDiscoveryResult],
    ) -> list[ToolDiscoveryResult]:
        """
        Aggregate tools from multiple discovery results.

        Args:
            results: List of discovery results

        Returns:
            List of tool discovery results
        """
        tools = []
        for result in results:
            if not result.error:
                for tool in result.tools:
                    tools.append(ToolDiscoveryResult(
                        tool=tool,
                        server_name=result.config.name,
                        server_config=result.config,
                    ))
        return tools

    def find_tool(
        self,
        results: list[ServerDiscoveryResult],
        tool_name: str,
    ) -> Optional[ToolDiscoveryResult]:
        """
        Find a specific tool across servers.

        Args:
            results: Discovery results
            tool_name: Tool name (can be "server:tool" or just "tool")

        Returns:
            ToolDiscoveryResult if found
        """
        # Parse server:tool format
        if ":" in tool_name:
            server_name, name = tool_name.split(":", 1)
            for result in results:
                if result.config.name == server_name:
                    for tool in result.tools:
                        if tool.name == name:
                            return ToolDiscoveryResult(
                                tool=tool,
                                server_name=server_name,
                                server_config=result.config,
                            )
        else:
            # Search all servers
            for result in results:
                for tool in result.tools:
                    if tool.name == tool_name:
                        return ToolDiscoveryResult(
                            tool=tool,
                            server_name=result.config.name,
                            server_config=result.config,
                        )

        return None

    def search_tools(
        self,
        results: list[ServerDiscoveryResult],
        query: str,
    ) -> list[ToolDiscoveryResult]:
        """
        Search for tools by name or description.

        Args:
            results: Discovery results
            query: Search query

        Returns:
            Matching tools
        """
        query_lower = query.lower()
        matches = []

        for result in results:
            if not result.error:
                for tool in result.tools:
                    if (
                        query_lower in tool.name.lower() or
                        query_lower in tool.description.lower()
                    ):
                        matches.append(ToolDiscoveryResult(
                            tool=tool,
                            server_name=result.config.name,
                            server_config=result.config,
                        ))

        return matches


class ServerProbe:
    """
    Probe MCP servers to determine their status and health.
    """

    def __init__(self, timeout: float = 10.0):
        """
        Initialize server probe.

        Args:
            timeout: Timeout for probe operations
        """
        self.timeout = timeout

    async def probe(self, config: ServerConfig) -> dict[str, Any]:
        """
        Probe a server for status information.

        Args:
            config: Server configuration

        Returns:
            Probe results
        """
        import time
        start_time = time.monotonic()

        client = MCPClient(config)

        try:
            await asyncio.wait_for(
                client.connect(),
                timeout=self.timeout,
            )

            # Try to ping
            ping_ok = await client.ping()

            return {
                "status": "ok",
                "server_name": client.server_info.name if client.server_info else None,
                "server_version": client.server_info.version if client.server_info else None,
                "ping": ping_ok,
                "latency_ms": (time.monotonic() - start_time) * 1000,
                "tools_count": len(client.list_tools()),
                "resources_count": len(client.list_resources()),
                "prompts_count": len(client.list_prompts()),
            }

        except asyncio.TimeoutError:
            return {
                "status": "timeout",
                "latency_ms": (time.monotonic() - start_time) * 1000,
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "latency_ms": (time.monotonic() - start_time) * 1000,
            }

        finally:
            await client.disconnect()

    async def probe_all(
        self,
        configs: list[ServerConfig],
    ) -> dict[str, dict[str, Any]]:
        """
        Probe multiple servers.

        Args:
            configs: List of server configurations

        Returns:
            Dictionary of probe results keyed by server name
        """
        tasks = {
            config.name: self.probe(config)
            for config in configs
        }

        results = {}
        for name, task in tasks.items():
            results[name] = await task

        return results
