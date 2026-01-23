"""
AION MCP Client

Client implementation for connecting to MCP servers.
"""

from aion.mcp.client.client import MCPClient, MCPError
from aion.mcp.client.session import MCPSessionManager, MCPClientPool, SessionState
from aion.mcp.client.discovery import (
    MCPDiscovery,
    ServerDiscoveryResult,
    ToolDiscoveryResult,
    ServerProbe,
)

__all__ = [
    # Client
    "MCPClient",
    "MCPError",
    # Session Management
    "MCPSessionManager",
    "MCPClientPool",
    "SessionState",
    # Discovery
    "MCPDiscovery",
    "ServerDiscoveryResult",
    "ToolDiscoveryResult",
    "ServerProbe",
]
