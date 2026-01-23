"""
AION MCP Tool Bridge

Bridges MCP tools to AION's native tool system.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, TYPE_CHECKING

import structlog

from aion.mcp.types import (
    Tool as MCPTool,
    ToolResult,
    TextContent,
    ToolInputSchema,
)

if TYPE_CHECKING:
    from aion.mcp.manager import MCPManager
    from aion.systems.tools.registry import Tool as AIONTool, ToolCategory
    from aion.systems.tools.orchestrator import ToolOrchestrator

logger = structlog.get_logger(__name__)


class MCPToolBridge:
    """
    Bridges MCP tools to AION's tool system.

    Allows MCP tools to be:
    - Discovered by AION's tool orchestrator
    - Called through AION's unified tool interface
    - Used in AION's planning system
    """

    def __init__(self, mcp_manager: "MCPManager"):
        """
        Initialize tool bridge.

        Args:
            mcp_manager: MCP manager instance
        """
        self.mcp_manager = mcp_manager

        # Tracking registered tools
        self._registered_tools: Dict[str, tuple[str, str]] = {}  # aion_name -> (server, mcp_tool)

    def register_with_orchestrator(
        self,
        orchestrator: "ToolOrchestrator",
        server_filter: Optional[list[str]] = None,
        prefix: str = "mcp",
    ) -> int:
        """
        Register all MCP tools with AION's tool orchestrator.

        Args:
            orchestrator: AION tool orchestrator
            server_filter: Only register tools from these servers (or all if None)
            prefix: Prefix for tool names

        Returns:
            Number of tools registered
        """
        from aion.systems.tools.registry import Tool as AIONTool, ToolCategory, ToolParameter, ParameterType

        count = 0

        for server_name, tools in self.mcp_manager.list_all_tools().items():
            if server_filter and server_name not in server_filter:
                continue

            for mcp_tool in tools:
                # Create unique tool name
                aion_name = f"{prefix}_{server_name}_{mcp_tool.name}"

                # Create async handler
                handler = self._create_handler(server_name, mcp_tool.name)

                # Convert parameters
                parameters = self._convert_parameters(mcp_tool.inputSchema)

                # Create AION tool
                aion_tool = AIONTool(
                    name=aion_name,
                    description=f"[MCP:{server_name}] {mcp_tool.description}",
                    handler=handler,
                    parameters=parameters,
                    category=ToolCategory.CUSTOM,
                    tags=["mcp", server_name],
                )

                # Register with orchestrator
                orchestrator.registry.register(aion_tool)

                # Track registration
                self._registered_tools[aion_name] = (server_name, mcp_tool.name)
                count += 1

        logger.info(f"Registered {count} MCP tools with orchestrator")
        return count

    def _create_handler(self, server_name: str, tool_name: str):
        """Create an async handler for an MCP tool."""
        async def handler(params: Dict[str, Any]) -> Dict[str, Any]:
            result = await self.mcp_manager.call_tool(
                server_name,
                tool_name,
                params,
            )

            # Extract text content
            text_parts = []
            for content in result.content:
                if content.get("type") == "text":
                    text_parts.append(content.get("text", ""))

            return {
                "success": not result.isError,
                "content": "\n".join(text_parts),
                "raw": result.content,
            }

        return handler

    def _convert_parameters(self, input_schema: Any) -> list:
        """Convert MCP input schema to AION parameter format."""
        from aion.systems.tools.registry import ToolParameter, ParameterType

        parameters = []

        if isinstance(input_schema, ToolInputSchema):
            properties = input_schema.properties
            required = input_schema.required
        elif isinstance(input_schema, dict):
            properties = input_schema.get("properties", {})
            required = input_schema.get("required", [])
        else:
            return parameters

        for name, prop in properties.items():
            param_type = self._map_json_type(prop.get("type", "string"))

            parameters.append(ToolParameter(
                name=name,
                type=param_type,
                description=prop.get("description", ""),
                required=name in required,
                default=prop.get("default"),
                enum=prop.get("enum"),
            ))

        return parameters

    def _map_json_type(self, json_type: str):
        """Map JSON Schema type to AION ParameterType."""
        from aion.systems.tools.registry import ParameterType

        type_map = {
            "string": ParameterType.STRING,
            "integer": ParameterType.INTEGER,
            "number": ParameterType.FLOAT,
            "boolean": ParameterType.BOOLEAN,
            "array": ParameterType.ARRAY,
            "object": ParameterType.OBJECT,
        }
        return type_map.get(json_type, ParameterType.STRING)

    def unregister_from_orchestrator(
        self,
        orchestrator: "ToolOrchestrator",
        server_filter: Optional[list[str]] = None,
    ) -> int:
        """
        Unregister MCP tools from AION's tool orchestrator.

        Args:
            orchestrator: AION tool orchestrator
            server_filter: Only unregister tools from these servers

        Returns:
            Number of tools unregistered
        """
        count = 0

        for aion_name, (server_name, _) in list(self._registered_tools.items()):
            if server_filter and server_name not in server_filter:
                continue

            # Try to unregister from orchestrator
            if hasattr(orchestrator.registry, '_tools'):
                if aion_name in orchestrator.registry._tools:
                    del orchestrator.registry._tools[aion_name]
                    count += 1

            # Remove from tracking
            del self._registered_tools[aion_name]

        logger.info(f"Unregistered {count} MCP tools from orchestrator")
        return count

    def get_tool_mappings(self) -> Dict[str, tuple[str, str]]:
        """
        Get mapping of AION tool names to MCP server:tool pairs.

        Returns:
            Dict of {aion_tool_name: (server_name, mcp_tool_name)}
        """
        return self._registered_tools.copy()

    def get_mcp_tool_name(self, aion_name: str) -> Optional[tuple[str, str]]:
        """
        Get the MCP server and tool name for an AION tool.

        Args:
            aion_name: AION tool name

        Returns:
            Tuple of (server_name, tool_name) or None
        """
        return self._registered_tools.get(aion_name)

    async def call_via_bridge(
        self,
        aion_name: str,
        arguments: Dict[str, Any],
    ) -> ToolResult:
        """
        Call an MCP tool via the bridge.

        Args:
            aion_name: AION tool name
            arguments: Tool arguments

        Returns:
            ToolResult
        """
        mapping = self._registered_tools.get(aion_name)
        if not mapping:
            raise ValueError(f"Tool not registered: {aion_name}")

        server_name, tool_name = mapping
        return await self.mcp_manager.call_tool(server_name, tool_name, arguments)

    def list_bridged_tools(self) -> list[Dict[str, Any]]:
        """
        List all bridged tools with their mappings.

        Returns:
            List of tool info dictionaries
        """
        result = []

        for aion_name, (server_name, tool_name) in self._registered_tools.items():
            mcp_tool = self.mcp_manager.get_tool(server_name, tool_name)

            result.append({
                "aion_name": aion_name,
                "server_name": server_name,
                "mcp_tool_name": tool_name,
                "description": mcp_tool.description if mcp_tool else "",
            })

        return result


class MCPResourceBridge:
    """
    Bridges MCP resources to AION's resource system.
    """

    def __init__(self, mcp_manager: "MCPManager"):
        """
        Initialize resource bridge.

        Args:
            mcp_manager: MCP manager instance
        """
        self.mcp_manager = mcp_manager

    async def read_resource(
        self,
        uri: str,
        server_name: Optional[str] = None,
    ) -> Optional[str]:
        """
        Read a resource through the bridge.

        Args:
            uri: Resource URI
            server_name: Server name (auto-detect if None)

        Returns:
            Resource content as string
        """
        if server_name:
            content = await self.mcp_manager.read_resource(server_name, uri)
            return content.text or (content.blob if content.blob else None)

        # Try to find the resource in any connected server
        for server, resources in self.mcp_manager.list_all_resources().items():
            for resource in resources:
                if resource.uri == uri:
                    content = await self.mcp_manager.read_resource(server, uri)
                    return content.text or (content.blob if content.blob else None)

        return None

    def list_all_resources(self) -> list[Dict[str, Any]]:
        """
        List all available resources across servers.

        Returns:
            List of resource info dictionaries
        """
        result = []

        for server_name, resources in self.mcp_manager.list_all_resources().items():
            for resource in resources:
                result.append({
                    "server_name": server_name,
                    "uri": resource.uri,
                    "name": resource.name,
                    "description": resource.description,
                    "mime_type": resource.mimeType,
                })

        return result


class MCPPromptBridge:
    """
    Bridges MCP prompts to AION's prompt system.
    """

    def __init__(self, mcp_manager: "MCPManager"):
        """
        Initialize prompt bridge.

        Args:
            mcp_manager: MCP manager instance
        """
        self.mcp_manager = mcp_manager

    async def get_prompt(
        self,
        name: str,
        arguments: Optional[Dict[str, str]] = None,
        server_name: Optional[str] = None,
    ) -> Optional[list[Dict[str, Any]]]:
        """
        Get a prompt through the bridge.

        Args:
            name: Prompt name
            arguments: Prompt arguments
            server_name: Server name (auto-detect if None)

        Returns:
            List of prompt messages
        """
        if server_name:
            messages = await self.mcp_manager.get_prompt(server_name, name, arguments)
            return [m.to_dict() for m in messages]

        # Try to find the prompt in any connected server
        for server, prompts in self.mcp_manager.list_all_prompts().items():
            for prompt in prompts:
                if prompt.name == name:
                    messages = await self.mcp_manager.get_prompt(server, name, arguments)
                    return [m.to_dict() for m in messages]

        return None

    def list_all_prompts(self) -> list[Dict[str, Any]]:
        """
        List all available prompts across servers.

        Returns:
            List of prompt info dictionaries
        """
        result = []

        for server_name, prompts in self.mcp_manager.list_all_prompts().items():
            for prompt in prompts:
                result.append({
                    "server_name": server_name,
                    "name": prompt.name,
                    "description": prompt.description,
                    "arguments": [
                        {
                            "name": arg.name,
                            "description": arg.description,
                            "required": arg.required,
                        }
                        for arg in prompt.arguments
                    ],
                })

        return result
