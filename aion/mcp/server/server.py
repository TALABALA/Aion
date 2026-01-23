"""
AION MCP Server

Exposes AION's capabilities via MCP protocol.
Allows other systems to use AION as an MCP server.
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any, Callable, Dict, Optional, TYPE_CHECKING

import structlog

from aion.mcp.types import (
    MCP_PROTOCOL_VERSION,
    JsonRpcRequest,
    JsonRpcResponse,
    JsonRpcNotification,
    JsonRpcError,
    JsonRpcErrorCode,
    ServerCapabilities,
    ToolsCapability,
    ResourcesCapability,
    PromptsCapability,
    LoggingCapability,
    Implementation,
    Tool,
    ToolInputSchema,
    ToolResult,
    Resource,
    ResourceContent,
    Prompt,
    PromptArgument,
    PromptMessage,
    TextContent,
    LogLevel,
)
from aion.mcp.protocol import (
    MCPProtocol,
    MCPMethods,
    MCPProtocolError,
    MethodNotFoundError,
    InvalidParamsError,
    InternalError,
    ToolNotFoundError,
    ResourceNotFoundError,
    PromptNotFoundError,
)
from aion.mcp.server.handlers import MCPRequestHandler

if TYPE_CHECKING:
    from aion.core.kernel import AIONKernel

logger = structlog.get_logger(__name__)


class MCPServer:
    """
    MCP Server exposing AION's capabilities.

    Allows external systems to:
    - Discover and use AION's tools
    - Read AION's resources (memories, plans, etc.)
    - Use prompt templates
    """

    def __init__(
        self,
        kernel: Optional["AIONKernel"] = None,
        server_name: str = "aion",
        server_version: str = "1.0.0",
    ):
        """
        Initialize MCP server.

        Args:
            kernel: AION kernel instance
            server_name: Server name for MCP protocol
            server_version: Server version for MCP protocol
        """
        self.kernel = kernel
        self.server_info = Implementation(
            name=server_name,
            version=server_version,
        )

        # Protocol handler
        self._protocol = MCPProtocol()

        # Request handler
        self._handler = MCPRequestHandler(self)

        # State
        self._initialized = False
        self._client_info: Optional[Implementation] = None
        self._client_capabilities: Optional[dict] = None

        # Tools registry
        self._tools: Dict[str, Tool] = {}
        self._tool_handlers: Dict[str, Callable] = {}

        # Resources registry
        self._resources: Dict[str, Resource] = {}
        self._resource_handlers: Dict[str, Callable] = {}

        # Prompts registry
        self._prompts: Dict[str, Prompt] = {}
        self._prompt_handlers: Dict[str, Callable] = {}

        # Log level
        self._log_level = LogLevel.INFO

        # Input/output streams (for stdio transport)
        self._input_stream: Optional[asyncio.StreamReader] = None
        self._output_stream: Optional[asyncio.StreamWriter] = None

    @property
    def capabilities(self) -> ServerCapabilities:
        """Get server capabilities."""
        return ServerCapabilities(
            tools=ToolsCapability(listChanged=True) if self._tools else None,
            resources=ResourcesCapability(
                subscribe=False,
                listChanged=True,
            ) if self._resources else None,
            prompts=PromptsCapability(listChanged=True) if self._prompts else None,
            logging=LoggingCapability(),
        )

    # === Tool Registration ===

    def register_tool(
        self,
        name: str,
        description: str,
        handler: Callable,
        parameters: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Register a tool.

        Args:
            name: Tool name
            description: Tool description
            handler: Async callable to handle tool calls
            parameters: JSON Schema for parameters
        """
        input_schema = ToolInputSchema(
            type="object",
            properties=parameters.get("properties", {}) if parameters else {},
            required=parameters.get("required", []) if parameters else [],
        )

        self._tools[name] = Tool(
            name=name,
            description=description,
            inputSchema=input_schema,
        )
        self._tool_handlers[name] = handler

        logger.debug("Registered tool", name=name)

    def unregister_tool(self, name: str) -> None:
        """Unregister a tool."""
        self._tools.pop(name, None)
        self._tool_handlers.pop(name, None)

    def list_tools(self) -> list[Tool]:
        """List all registered tools."""
        return list(self._tools.values())

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> ToolResult:
        """
        Call a registered tool.

        Args:
            name: Tool name
            arguments: Tool arguments

        Returns:
            ToolResult
        """
        handler = self._tool_handlers.get(name)
        if not handler:
            raise ToolNotFoundError(name)

        try:
            result = await handler(arguments)

            # Convert result to content
            if isinstance(result, str):
                content = [TextContent(text=result).to_dict()]
            elif isinstance(result, dict):
                content = [TextContent(text=json.dumps(result, indent=2)).to_dict()]
            elif isinstance(result, list):
                content = result
            else:
                content = [TextContent(text=str(result)).to_dict()]

            return ToolResult(content=content, isError=False)

        except Exception as e:
            logger.error("Tool execution error", tool=name, error=str(e))
            return ToolResult(
                content=[TextContent(text=f"Error: {str(e)}").to_dict()],
                isError=True,
            )

    # === Resource Registration ===

    def register_resource(
        self,
        uri: str,
        name: str,
        handler: Callable,
        description: Optional[str] = None,
        mime_type: Optional[str] = None,
    ) -> None:
        """
        Register a resource.

        Args:
            uri: Resource URI
            name: Resource name
            handler: Async callable to read resource content
            description: Resource description
            mime_type: MIME type
        """
        self._resources[uri] = Resource(
            uri=uri,
            name=name,
            description=description,
            mimeType=mime_type,
        )
        self._resource_handlers[uri] = handler

        logger.debug("Registered resource", uri=uri)

    def unregister_resource(self, uri: str) -> None:
        """Unregister a resource."""
        self._resources.pop(uri, None)
        self._resource_handlers.pop(uri, None)

    def list_resources(self) -> list[Resource]:
        """List all registered resources."""
        return list(self._resources.values())

    async def read_resource(self, uri: str) -> ResourceContent:
        """
        Read a resource.

        Args:
            uri: Resource URI

        Returns:
            ResourceContent
        """
        handler = self._resource_handlers.get(uri)
        if not handler:
            raise ResourceNotFoundError(uri)

        try:
            content = await handler()

            if isinstance(content, str):
                return ResourceContent(uri=uri, text=content)
            elif isinstance(content, bytes):
                import base64
                return ResourceContent(
                    uri=uri,
                    blob=base64.b64encode(content).decode("ascii"),
                )
            elif isinstance(content, dict):
                return ResourceContent(uri=uri, text=json.dumps(content, indent=2))
            else:
                return ResourceContent(uri=uri, text=str(content))

        except Exception as e:
            logger.error("Resource read error", uri=uri, error=str(e))
            raise

    # === Prompt Registration ===

    def register_prompt(
        self,
        name: str,
        handler: Callable,
        description: Optional[str] = None,
        arguments: Optional[list[dict]] = None,
    ) -> None:
        """
        Register a prompt.

        Args:
            name: Prompt name
            handler: Async callable to generate prompt messages
            description: Prompt description
            arguments: Prompt arguments definition
        """
        prompt_args = []
        if arguments:
            for arg in arguments:
                prompt_args.append(PromptArgument(
                    name=arg["name"],
                    description=arg.get("description"),
                    required=arg.get("required", False),
                ))

        self._prompts[name] = Prompt(
            name=name,
            description=description,
            arguments=prompt_args,
        )
        self._prompt_handlers[name] = handler

        logger.debug("Registered prompt", name=name)

    def unregister_prompt(self, name: str) -> None:
        """Unregister a prompt."""
        self._prompts.pop(name, None)
        self._prompt_handlers.pop(name, None)

    def list_prompts(self) -> list[Prompt]:
        """List all registered prompts."""
        return list(self._prompts.values())

    async def get_prompt(
        self,
        name: str,
        arguments: Optional[dict[str, str]] = None,
    ) -> list[PromptMessage]:
        """
        Get a prompt.

        Args:
            name: Prompt name
            arguments: Prompt arguments

        Returns:
            List of prompt messages
        """
        handler = self._prompt_handlers.get(name)
        if not handler:
            raise PromptNotFoundError(name)

        try:
            messages = await handler(arguments or {})

            # Convert to PromptMessage format
            result = []
            for msg in messages:
                if isinstance(msg, dict):
                    result.append(PromptMessage(
                        role=msg.get("role", "user"),
                        content=msg.get("content", {}),
                    ))
                elif isinstance(msg, PromptMessage):
                    result.append(msg)
            return result

        except Exception as e:
            logger.error("Prompt get error", name=name, error=str(e))
            raise

    # === Message Handling ===

    async def handle_message(self, message: str) -> Optional[str]:
        """
        Handle an incoming message.

        Args:
            message: JSON-RPC message

        Returns:
            Response message (or None for notifications)
        """
        try:
            parsed = self._protocol.parse(message)
        except MCPProtocolError as e:
            error_response = JsonRpcResponse(
                id=None,
                error=e.to_error().to_dict(),
            )
            return self._protocol.serialize(error_response)

        if self._protocol.is_request(parsed):
            response = await self._handle_request(parsed)
            return self._protocol.serialize(response)

        elif self._protocol.is_notification(parsed):
            await self._handle_notification(parsed)
            return None

        return None

    async def _handle_request(self, request: JsonRpcRequest) -> JsonRpcResponse:
        """Handle an incoming request."""
        try:
            result = await self._handler.handle(request.method, request.params or {})
            return self._protocol.create_response(request.id, result=result)

        except MCPProtocolError as e:
            return self._protocol.create_response(
                request.id,
                error=e.to_error(),
            )

        except Exception as e:
            logger.error("Request handler error", method=request.method, error=str(e))
            return self._protocol.create_response(
                request.id,
                error=JsonRpcError(
                    code=JsonRpcErrorCode.INTERNAL_ERROR,
                    message=str(e),
                ),
            )

    async def _handle_notification(self, notification: JsonRpcNotification) -> None:
        """Handle an incoming notification."""
        if notification.method == MCPMethods.INITIALIZED:
            self._initialized = True
            logger.info("Client initialized")
        else:
            logger.debug(
                "Received notification",
                method=notification.method,
            )

    # === Stdio Server ===

    async def serve_stdio(self) -> None:
        """
        Serve MCP over stdin/stdout.

        This is the standard way to run an MCP server as a subprocess.
        """
        logger.info("Starting MCP server on stdio")

        # Setup streams
        loop = asyncio.get_event_loop()

        reader = asyncio.StreamReader()
        reader_protocol = asyncio.StreamReaderProtocol(reader)
        await loop.connect_read_pipe(lambda: reader_protocol, sys.stdin)

        writer_transport, writer_protocol = await loop.connect_write_pipe(
            lambda: asyncio.streams.FlowControlMixin(),
            sys.stdout,
        )
        writer = asyncio.StreamWriter(writer_transport, writer_protocol, None, loop)

        self._input_stream = reader
        self._output_stream = writer

        try:
            while True:
                line = await reader.readline()
                if not line:
                    break

                message = line.decode("utf-8").strip()
                if not message:
                    continue

                response = await self.handle_message(message)
                if response:
                    writer.write((response + "\n").encode("utf-8"))
                    await writer.drain()

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("Stdio server error", error=str(e))

        finally:
            logger.info("MCP server stopped")

    async def send_notification(self, method: str, params: dict[str, Any]) -> None:
        """
        Send a notification to the client.

        Args:
            method: Notification method
            params: Notification parameters
        """
        if not self._output_stream:
            return

        notification = self._protocol.create_notification(method, params)
        message = self._protocol.serialize(notification)

        self._output_stream.write((message + "\n").encode("utf-8"))
        await self._output_stream.drain()

    async def notify_tools_changed(self) -> None:
        """Notify client that tools list has changed."""
        await self.send_notification(
            MCPMethods.NOTIFICATION_TOOLS_LIST_CHANGED,
            {},
        )

    async def notify_resources_changed(self) -> None:
        """Notify client that resources list has changed."""
        await self.send_notification(
            MCPMethods.NOTIFICATION_RESOURCES_LIST_CHANGED,
            {},
        )

    async def notify_prompts_changed(self) -> None:
        """Notify client that prompts list has changed."""
        await self.send_notification(
            MCPMethods.NOTIFICATION_PROMPTS_LIST_CHANGED,
            {},
        )

    async def send_log(
        self,
        level: LogLevel,
        message: str,
        logger_name: Optional[str] = None,
        data: Optional[Any] = None,
    ) -> None:
        """
        Send a log message to the client.

        Args:
            level: Log level
            message: Log message
            logger_name: Logger name
            data: Additional data
        """
        params = {
            "level": level.value,
            "message": message,
        }
        if logger_name:
            params["logger"] = logger_name
        if data is not None:
            params["data"] = data

        await self.send_notification(MCPMethods.NOTIFICATION_LOG, params)
