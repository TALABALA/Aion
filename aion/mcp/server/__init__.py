"""
AION MCP Server

Server implementation for exposing AION as an MCP server.
"""

from aion.mcp.server.server import MCPServer
from aion.mcp.server.handlers import MCPRequestHandler
from aion.mcp.server.aion_tools import (
    register_aion_tools,
    register_aion_resources,
    register_aion_prompts,
    setup_aion_mcp_server,
)

__all__ = [
    "MCPServer",
    "MCPRequestHandler",
    "register_aion_tools",
    "register_aion_resources",
    "register_aion_prompts",
    "setup_aion_mcp_server",
]
