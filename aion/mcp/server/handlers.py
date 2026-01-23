"""
AION MCP Server Request Handlers

Handles incoming MCP requests for the server.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, TYPE_CHECKING

import structlog

from aion.mcp.types import (
    MCP_PROTOCOL_VERSION,
    TextContent,
)
from aion.mcp.protocol import (
    MCPMethods,
    MethodNotFoundError,
    InvalidParamsError,
)

if TYPE_CHECKING:
    from aion.mcp.server.server import MCPServer

logger = structlog.get_logger(__name__)


class MCPRequestHandler:
    """
    Handles MCP requests for the server.

    Routes requests to appropriate handlers and manages
    protocol-level interactions.
    """

    def __init__(self, server: "MCPServer"):
        """
        Initialize request handler.

        Args:
            server: MCP server instance
        """
        self.server = server

        # Method handlers
        self._handlers: Dict[str, Any] = {
            MCPMethods.INITIALIZE: self._handle_initialize,
            MCPMethods.PING: self._handle_ping,
            MCPMethods.TOOLS_LIST: self._handle_tools_list,
            MCPMethods.TOOLS_CALL: self._handle_tools_call,
            MCPMethods.RESOURCES_LIST: self._handle_resources_list,
            MCPMethods.RESOURCES_READ: self._handle_resources_read,
            MCPMethods.RESOURCES_TEMPLATES_LIST: self._handle_resources_templates_list,
            MCPMethods.PROMPTS_LIST: self._handle_prompts_list,
            MCPMethods.PROMPTS_GET: self._handle_prompts_get,
            MCPMethods.LOGGING_SET_LEVEL: self._handle_logging_set_level,
        }

    async def handle(
        self,
        method: str,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Handle a request.

        Args:
            method: Request method
            params: Request parameters

        Returns:
            Response result
        """
        handler = self._handlers.get(method)
        if not handler:
            raise MethodNotFoundError(method)

        return await handler(params)

    async def _handle_initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle initialize request."""
        protocol_version = params.get("protocolVersion", MCP_PROTOCOL_VERSION)
        client_info = params.get("clientInfo", {})
        client_capabilities = params.get("capabilities", {})

        logger.info(
            "Client initializing",
            client_name=client_info.get("name"),
            client_version=client_info.get("version"),
            protocol_version=protocol_version,
        )

        # Store client info
        self.server._client_capabilities = client_capabilities

        return {
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "capabilities": self.server.capabilities.to_dict(),
            "serverInfo": self.server.server_info.to_dict(),
            "instructions": "AION AI Operating System - Use tools to interact with AION's cognitive capabilities.",
        }

    async def _handle_ping(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle ping request."""
        return {}

    async def _handle_tools_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tools/list request."""
        cursor = params.get("cursor")

        tools = self.server.list_tools()

        # Simple pagination (if needed)
        return {
            "tools": [t.to_dict() for t in tools],
        }

    async def _handle_tools_call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tools/call request."""
        name = params.get("name")
        arguments = params.get("arguments", {})

        if not name:
            raise InvalidParamsError("Missing tool name")

        result = await self.server.call_tool(name, arguments)

        return result.to_dict()

    async def _handle_resources_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle resources/list request."""
        cursor = params.get("cursor")

        resources = self.server.list_resources()

        return {
            "resources": [r.to_dict() for r in resources],
        }

    async def _handle_resources_read(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle resources/read request."""
        uri = params.get("uri")

        if not uri:
            raise InvalidParamsError("Missing resource URI")

        content = await self.server.read_resource(uri)

        return {
            "contents": [content.to_dict()],
        }

    async def _handle_resources_templates_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle resources/templates/list request."""
        # No templates for now
        return {
            "resourceTemplates": [],
        }

    async def _handle_prompts_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle prompts/list request."""
        cursor = params.get("cursor")

        prompts = self.server.list_prompts()

        return {
            "prompts": [p.to_dict() for p in prompts],
        }

    async def _handle_prompts_get(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle prompts/get request."""
        name = params.get("name")
        arguments = params.get("arguments")

        if not name:
            raise InvalidParamsError("Missing prompt name")

        messages = await self.server.get_prompt(name, arguments)

        return {
            "messages": [m.to_dict() for m in messages],
        }

    async def _handle_logging_set_level(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle logging/setLevel request."""
        from aion.mcp.types import LogLevel

        level_str = params.get("level", "info")

        try:
            self.server._log_level = LogLevel(level_str)
            logger.info("Log level set", level=level_str)
        except ValueError:
            raise InvalidParamsError(f"Invalid log level: {level_str}")

        return {}
