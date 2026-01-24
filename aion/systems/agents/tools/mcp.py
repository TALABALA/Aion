"""
MCP (Model Context Protocol) Integration

Implements MCP client and server for tool integration,
enabling agents to use external tools via standardized protocol.
"""

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional, Union
import structlog

logger = structlog.get_logger()


class ToolCapability(str, Enum):
    """Tool capability types."""

    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    NETWORK = "network"
    FILESYSTEM = "filesystem"
    DATABASE = "database"
    API = "api"
    COMPUTE = "compute"


@dataclass
class ToolParameter:
    """A parameter for a tool."""

    name: str
    type: str  # JSON Schema type
    description: str
    required: bool = True
    default: Any = None
    enum: Optional[list[Any]] = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    pattern: Optional[str] = None

    def to_json_schema(self) -> dict[str, Any]:
        """Convert to JSON Schema format."""
        schema: dict[str, Any] = {
            "type": self.type,
            "description": self.description,
        }

        if self.enum:
            schema["enum"] = self.enum
        if self.minimum is not None:
            schema["minimum"] = self.minimum
        if self.maximum is not None:
            schema["maximum"] = self.maximum
        if self.pattern:
            schema["pattern"] = self.pattern
        if self.default is not None:
            schema["default"] = self.default

        return schema


@dataclass
class MCPTool:
    """An MCP-compatible tool definition."""

    name: str
    description: str
    parameters: list[ToolParameter] = field(default_factory=list)
    capabilities: list[ToolCapability] = field(default_factory=list)
    handler: Optional[Callable] = None
    version: str = "1.0.0"
    timeout: float = 30.0
    rate_limit: Optional[int] = None  # calls per minute
    requires_confirmation: bool = False

    def to_mcp_schema(self) -> dict[str, Any]:
        """Convert to MCP tool schema format."""
        required = [p.name for p in self.parameters if p.required]
        properties = {p.name: p.to_json_schema() for p in self.parameters}

        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }


@dataclass
class MCPToolResult:
    """Result from tool execution."""

    tool_name: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_mcp_format(self) -> dict[str, Any]:
        """Convert to MCP result format."""
        if self.success:
            content = [{"type": "text", "text": json.dumps(self.result)}]
        else:
            content = [{"type": "text", "text": f"Error: {self.error}"}]

        return {
            "content": content,
            "isError": not self.success,
        }


class MCPServer:
    """
    MCP Server implementation.

    Exposes tools via the Model Context Protocol,
    allowing external clients to discover and use them.
    """

    def __init__(
        self,
        name: str = "aion-mcp-server",
        version: str = "1.0.0",
    ):
        self.name = name
        self.version = version
        self._tools: dict[str, MCPTool] = {}
        self._call_counts: dict[str, list[datetime]] = {}
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize MCP server."""
        self._initialized = True
        logger.info("mcp_server_initialized", name=self.name)

    async def shutdown(self) -> None:
        """Shutdown MCP server."""
        self._initialized = False
        logger.info("mcp_server_shutdown")

    def register_tool(self, tool: MCPTool) -> None:
        """Register a tool with the server."""
        self._tools[tool.name] = tool
        self._call_counts[tool.name] = []
        logger.info("tool_registered", tool=tool.name)

    def unregister_tool(self, name: str) -> bool:
        """Unregister a tool."""
        if name in self._tools:
            del self._tools[name]
            del self._call_counts[name]
            return True
        return False

    def list_tools(self) -> list[dict[str, Any]]:
        """List available tools in MCP format."""
        return [tool.to_mcp_schema() for tool in self._tools.values()]

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> MCPToolResult:
        """Execute a tool call."""
        start_time = datetime.now()

        if name not in self._tools:
            return MCPToolResult(
                tool_name=name,
                success=False,
                error=f"Tool not found: {name}",
            )

        tool = self._tools[name]

        # Check rate limit
        if tool.rate_limit:
            if not self._check_rate_limit(name, tool.rate_limit):
                return MCPToolResult(
                    tool_name=name,
                    success=False,
                    error="Rate limit exceeded",
                )

        # Validate arguments
        validation_error = self._validate_arguments(tool, arguments)
        if validation_error:
            return MCPToolResult(
                tool_name=name,
                success=False,
                error=validation_error,
            )

        # Execute tool
        try:
            if tool.handler:
                if asyncio.iscoroutinefunction(tool.handler):
                    result = await asyncio.wait_for(
                        tool.handler(**arguments),
                        timeout=tool.timeout,
                    )
                else:
                    result = tool.handler(**arguments)
            else:
                result = {"message": "Tool has no handler"}

            execution_time = (datetime.now() - start_time).total_seconds()

            # Record call
            self._call_counts[name].append(datetime.now())

            return MCPToolResult(
                tool_name=name,
                success=True,
                result=result,
                execution_time=execution_time,
            )

        except asyncio.TimeoutError:
            return MCPToolResult(
                tool_name=name,
                success=False,
                error=f"Tool execution timed out after {tool.timeout}s",
                execution_time=tool.timeout,
            )
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return MCPToolResult(
                tool_name=name,
                success=False,
                error=str(e),
                execution_time=execution_time,
            )

    def _check_rate_limit(self, tool_name: str, limit: int) -> bool:
        """Check if tool is within rate limit."""
        now = datetime.now()
        minute_ago = datetime.now().replace(
            second=now.second - 60 if now.second >= 60 else 0
        )

        # Clean old entries
        self._call_counts[tool_name] = [
            t for t in self._call_counts[tool_name]
            if t > minute_ago
        ]

        return len(self._call_counts[tool_name]) < limit

    def _validate_arguments(
        self,
        tool: MCPTool,
        arguments: dict[str, Any],
    ) -> Optional[str]:
        """Validate tool arguments."""
        for param in tool.parameters:
            if param.required and param.name not in arguments:
                return f"Missing required parameter: {param.name}"

            if param.name in arguments:
                value = arguments[param.name]

                # Type checking
                if param.type == "string" and not isinstance(value, str):
                    return f"Parameter {param.name} must be a string"
                elif param.type == "number" and not isinstance(value, (int, float)):
                    return f"Parameter {param.name} must be a number"
                elif param.type == "boolean" and not isinstance(value, bool):
                    return f"Parameter {param.name} must be a boolean"
                elif param.type == "array" and not isinstance(value, list):
                    return f"Parameter {param.name} must be an array"
                elif param.type == "object" and not isinstance(value, dict):
                    return f"Parameter {param.name} must be an object"

                # Enum checking
                if param.enum and value not in param.enum:
                    return f"Parameter {param.name} must be one of: {param.enum}"

                # Range checking
                if isinstance(value, (int, float)):
                    if param.minimum is not None and value < param.minimum:
                        return f"Parameter {param.name} must be >= {param.minimum}"
                    if param.maximum is not None and value > param.maximum:
                        return f"Parameter {param.name} must be <= {param.maximum}"

        return None

    def get_server_info(self) -> dict[str, Any]:
        """Get server information."""
        return {
            "name": self.name,
            "version": self.version,
            "capabilities": {
                "tools": {"listChanged": True},
            },
            "tool_count": len(self._tools),
        }


class MCPClient:
    """
    MCP Client implementation.

    Connects to MCP servers to discover and use tools.
    """

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self._servers: dict[str, MCPServer] = {}
        self._tool_cache: dict[str, tuple[str, MCPTool]] = {}  # tool -> (server, tool)
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize MCP client."""
        self._initialized = True
        logger.info("mcp_client_initialized", agent_id=self.agent_id)

    async def shutdown(self) -> None:
        """Shutdown MCP client."""
        self._initialized = False
        logger.info("mcp_client_shutdown")

    async def connect_server(
        self,
        server_id: str,
        server: MCPServer,
    ) -> bool:
        """Connect to an MCP server."""
        try:
            self._servers[server_id] = server

            # Cache tools
            for tool_schema in server.list_tools():
                tool_name = tool_schema["name"]
                if tool_name in server._tools:
                    self._tool_cache[tool_name] = (server_id, server._tools[tool_name])

            logger.info(
                "mcp_server_connected",
                server_id=server_id,
                tools=len(server._tools),
            )
            return True

        except Exception as e:
            logger.error("mcp_server_connection_failed", error=str(e))
            return False

    async def disconnect_server(self, server_id: str) -> bool:
        """Disconnect from an MCP server."""
        if server_id in self._servers:
            # Remove cached tools
            self._tool_cache = {
                name: (sid, tool)
                for name, (sid, tool) in self._tool_cache.items()
                if sid != server_id
            }
            del self._servers[server_id]
            return True
        return False

    def list_available_tools(self) -> list[dict[str, Any]]:
        """List all available tools from connected servers."""
        tools = []
        for server_id, server in self._servers.items():
            for tool_schema in server.list_tools():
                tool_schema["server"] = server_id
                tools.append(tool_schema)
        return tools

    def get_tool(self, name: str) -> Optional[MCPTool]:
        """Get a tool by name."""
        if name in self._tool_cache:
            return self._tool_cache[name][1]
        return None

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> MCPToolResult:
        """Call a tool by name."""
        if name not in self._tool_cache:
            return MCPToolResult(
                tool_name=name,
                success=False,
                error=f"Tool not found: {name}",
            )

        server_id, _ = self._tool_cache[name]
        server = self._servers[server_id]

        return await server.call_tool(name, arguments)

    async def call_tool_with_retry(
        self,
        name: str,
        arguments: dict[str, Any],
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> MCPToolResult:
        """Call a tool with retry logic."""
        last_result = None

        for attempt in range(max_retries):
            result = await self.call_tool(name, arguments)

            if result.success:
                return result

            last_result = result

            # Don't retry on certain errors
            if result.error and any(
                msg in result.error.lower()
                for msg in ["not found", "missing", "invalid", "rate limit"]
            ):
                break

            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay * (attempt + 1))

        return last_result or MCPToolResult(
            tool_name=name,
            success=False,
            error="All retries failed",
        )

    def get_tools_by_capability(
        self,
        capability: ToolCapability,
    ) -> list[MCPTool]:
        """Get tools that have a specific capability."""
        return [
            tool
            for _, tool in self._tool_cache.values()
            if capability in tool.capabilities
        ]

    def get_stats(self) -> dict[str, Any]:
        """Get client statistics."""
        return {
            "agent_id": self.agent_id,
            "connected_servers": len(self._servers),
            "available_tools": len(self._tool_cache),
            "servers": [
                {
                    "id": sid,
                    "name": server.name,
                    "tools": len(server._tools),
                }
                for sid, server in self._servers.items()
            ],
        }


# Built-in tools factory
def create_builtin_tools() -> list[MCPTool]:
    """Create built-in utility tools."""

    async def echo_handler(message: str) -> dict[str, str]:
        """Echo back a message."""
        return {"echoed": message}

    async def sleep_handler(seconds: float) -> dict[str, str]:
        """Sleep for specified seconds."""
        await asyncio.sleep(seconds)
        return {"slept": seconds}

    async def timestamp_handler() -> dict[str, str]:
        """Get current timestamp."""
        return {"timestamp": datetime.now().isoformat()}

    async def json_parse_handler(text: str) -> dict[str, Any]:
        """Parse JSON text."""
        return {"parsed": json.loads(text)}

    return [
        MCPTool(
            name="echo",
            description="Echo back a message",
            parameters=[
                ToolParameter(
                    name="message",
                    type="string",
                    description="Message to echo",
                ),
            ],
            capabilities=[ToolCapability.READ],
            handler=echo_handler,
        ),
        MCPTool(
            name="sleep",
            description="Sleep for specified duration",
            parameters=[
                ToolParameter(
                    name="seconds",
                    type="number",
                    description="Seconds to sleep",
                    minimum=0,
                    maximum=60,
                ),
            ],
            capabilities=[ToolCapability.COMPUTE],
            handler=sleep_handler,
        ),
        MCPTool(
            name="timestamp",
            description="Get current timestamp",
            parameters=[],
            capabilities=[ToolCapability.READ],
            handler=timestamp_handler,
        ),
        MCPTool(
            name="json_parse",
            description="Parse JSON text",
            parameters=[
                ToolParameter(
                    name="text",
                    type="string",
                    description="JSON text to parse",
                ),
            ],
            capabilities=[ToolCapability.COMPUTE],
            handler=json_parse_handler,
        ),
    ]
