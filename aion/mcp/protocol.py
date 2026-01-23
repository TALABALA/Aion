"""
AION MCP Protocol Handler

Handles MCP message serialization, deserialization, and routing.
Implements the JSON-RPC 2.0 based MCP protocol.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Optional, Union

import structlog

from aion.mcp.types import (
    JsonRpcRequest,
    JsonRpcResponse,
    JsonRpcNotification,
    JsonRpcError,
    JsonRpcErrorCode,
    MCP_PROTOCOL_VERSION,
)

logger = structlog.get_logger(__name__)


class MCPProtocolError(Exception):
    """Base exception for MCP protocol errors."""

    def __init__(
        self,
        code: int,
        message: str,
        data: Optional[Any] = None,
    ):
        self.code = code
        self.message = message
        self.data = data
        super().__init__(f"MCP Error {code}: {message}")

    def to_error(self) -> JsonRpcError:
        """Convert to JSON-RPC error object."""
        return JsonRpcError(
            code=self.code,
            message=self.message,
            data=self.data,
        )


class ParseError(MCPProtocolError):
    """Error parsing JSON-RPC message."""

    def __init__(self, message: str = "Parse error", data: Optional[Any] = None):
        super().__init__(JsonRpcErrorCode.PARSE_ERROR, message, data)


class InvalidRequestError(MCPProtocolError):
    """Invalid JSON-RPC request."""

    def __init__(self, message: str = "Invalid request", data: Optional[Any] = None):
        super().__init__(JsonRpcErrorCode.INVALID_REQUEST, message, data)


class MethodNotFoundError(MCPProtocolError):
    """Method not found."""

    def __init__(self, method: str):
        super().__init__(
            JsonRpcErrorCode.METHOD_NOT_FOUND,
            f"Method not found: {method}",
        )


class InvalidParamsError(MCPProtocolError):
    """Invalid method parameters."""

    def __init__(self, message: str = "Invalid params", data: Optional[Any] = None):
        super().__init__(JsonRpcErrorCode.INVALID_PARAMS, message, data)


class InternalError(MCPProtocolError):
    """Internal server error."""

    def __init__(self, message: str = "Internal error", data: Optional[Any] = None):
        super().__init__(JsonRpcErrorCode.INTERNAL_ERROR, message, data)


class ToolNotFoundError(MCPProtocolError):
    """Tool not found."""

    def __init__(self, tool_name: str):
        super().__init__(
            JsonRpcErrorCode.TOOL_NOT_FOUND,
            f"Tool not found: {tool_name}",
        )


class ResourceNotFoundError(MCPProtocolError):
    """Resource not found."""

    def __init__(self, uri: str):
        super().__init__(
            JsonRpcErrorCode.RESOURCE_NOT_FOUND,
            f"Resource not found: {uri}",
        )


class PromptNotFoundError(MCPProtocolError):
    """Prompt not found."""

    def __init__(self, prompt_name: str):
        super().__init__(
            JsonRpcErrorCode.PROMPT_NOT_FOUND,
            f"Prompt not found: {prompt_name}",
        )


class ConnectionClosedError(MCPProtocolError):
    """Connection closed."""

    def __init__(self, message: str = "Connection closed"):
        super().__init__(JsonRpcErrorCode.CONNECTION_CLOSED, message)


class MCPProtocol:
    """
    MCP Protocol handler for message serialization and routing.

    Handles:
    - JSON-RPC message parsing and serialization
    - Request/response matching
    - Notification handling
    - Error handling
    """

    def __init__(self):
        self._request_id_counter = 0

    def next_request_id(self) -> str:
        """Generate the next request ID."""
        self._request_id_counter += 1
        return str(self._request_id_counter)

    def create_request(
        self,
        method: str,
        params: Optional[dict[str, Any]] = None,
        request_id: Optional[Union[str, int]] = None,
    ) -> JsonRpcRequest:
        """
        Create a JSON-RPC request.

        Args:
            method: The method to call
            params: Optional parameters
            request_id: Optional request ID (auto-generated if not provided)

        Returns:
            JsonRpcRequest object
        """
        if request_id is None:
            request_id = self.next_request_id()

        return JsonRpcRequest(
            method=method,
            id=request_id,
            params=params,
        )

    def create_response(
        self,
        request_id: Union[str, int],
        result: Optional[Any] = None,
        error: Optional[JsonRpcError] = None,
    ) -> JsonRpcResponse:
        """
        Create a JSON-RPC response.

        Args:
            request_id: The request ID this responds to
            result: The result (if successful)
            error: The error (if failed)

        Returns:
            JsonRpcResponse object
        """
        return JsonRpcResponse(
            id=request_id,
            result=result,
            error=error.to_dict() if error else None,
        )

    def create_notification(
        self,
        method: str,
        params: Optional[dict[str, Any]] = None,
    ) -> JsonRpcNotification:
        """
        Create a JSON-RPC notification.

        Args:
            method: The notification method
            params: Optional parameters

        Returns:
            JsonRpcNotification object
        """
        return JsonRpcNotification(
            method=method,
            params=params,
        )

    def create_error_response(
        self,
        request_id: Union[str, int],
        error: MCPProtocolError,
    ) -> JsonRpcResponse:
        """
        Create an error response.

        Args:
            request_id: The request ID this responds to
            error: The error

        Returns:
            JsonRpcResponse object
        """
        return self.create_response(
            request_id=request_id,
            error=error.to_error(),
        )

    def serialize(
        self,
        message: Union[JsonRpcRequest, JsonRpcResponse, JsonRpcNotification],
    ) -> str:
        """
        Serialize a message to JSON string.

        Args:
            message: The message to serialize

        Returns:
            JSON string
        """
        return json.dumps(message.to_dict(), separators=(",", ":"))

    def serialize_pretty(
        self,
        message: Union[JsonRpcRequest, JsonRpcResponse, JsonRpcNotification],
    ) -> str:
        """
        Serialize a message to pretty-printed JSON string.

        Args:
            message: The message to serialize

        Returns:
            Pretty-printed JSON string
        """
        return json.dumps(message.to_dict(), indent=2)

    def parse(
        self,
        data: str,
    ) -> Union[JsonRpcRequest, JsonRpcResponse, JsonRpcNotification]:
        """
        Parse a JSON-RPC message from string.

        Args:
            data: JSON string to parse

        Returns:
            Parsed message object

        Raises:
            ParseError: If JSON is invalid
            InvalidRequestError: If message format is invalid
        """
        try:
            obj = json.loads(data)
        except json.JSONDecodeError as e:
            raise ParseError(f"Invalid JSON: {e}")

        return self.parse_dict(obj)

    def parse_dict(
        self,
        obj: dict,
    ) -> Union[JsonRpcRequest, JsonRpcResponse, JsonRpcNotification]:
        """
        Parse a JSON-RPC message from dictionary.

        Args:
            obj: Dictionary to parse

        Returns:
            Parsed message object

        Raises:
            InvalidRequestError: If message format is invalid
        """
        if not isinstance(obj, dict):
            raise InvalidRequestError("Message must be an object")

        # Check if it's a response (has result or error, and id)
        if "id" in obj and ("result" in obj or "error" in obj):
            return JsonRpcResponse.from_dict(obj)

        # Check if it's a request (has method and id)
        if "method" in obj and "id" in obj:
            return JsonRpcRequest.from_dict(obj)

        # Check if it's a notification (has method, no id)
        if "method" in obj and "id" not in obj:
            return JsonRpcNotification.from_dict(obj)

        raise InvalidRequestError("Cannot determine message type")

    def is_request(self, message: Any) -> bool:
        """Check if message is a request."""
        return isinstance(message, JsonRpcRequest)

    def is_response(self, message: Any) -> bool:
        """Check if message is a response."""
        return isinstance(message, JsonRpcResponse)

    def is_notification(self, message: Any) -> bool:
        """Check if message is a notification."""
        return isinstance(message, JsonRpcNotification)


# ============================================
# MCP Method Constants
# ============================================

class MCPMethods:
    """MCP protocol method names."""

    # Lifecycle
    INITIALIZE = "initialize"
    INITIALIZED = "notifications/initialized"
    PING = "ping"

    # Tools
    TOOLS_LIST = "tools/list"
    TOOLS_CALL = "tools/call"

    # Resources
    RESOURCES_LIST = "resources/list"
    RESOURCES_READ = "resources/read"
    RESOURCES_SUBSCRIBE = "resources/subscribe"
    RESOURCES_UNSUBSCRIBE = "resources/unsubscribe"
    RESOURCES_TEMPLATES_LIST = "resources/templates/list"

    # Prompts
    PROMPTS_LIST = "prompts/list"
    PROMPTS_GET = "prompts/get"

    # Logging
    LOGGING_SET_LEVEL = "logging/setLevel"

    # Notifications from server
    NOTIFICATION_MESSAGE = "notifications/message"
    NOTIFICATION_RESOURCES_LIST_CHANGED = "notifications/resources/list_changed"
    NOTIFICATION_RESOURCES_UPDATED = "notifications/resources/updated"
    NOTIFICATION_TOOLS_LIST_CHANGED = "notifications/tools/list_changed"
    NOTIFICATION_PROMPTS_LIST_CHANGED = "notifications/prompts/list_changed"
    NOTIFICATION_PROGRESS = "notifications/progress"
    NOTIFICATION_LOG = "notifications/log"

    # Sampling (client-side)
    SAMPLING_CREATE_MESSAGE = "sampling/createMessage"

    # Roots (client-side)
    ROOTS_LIST = "roots/list"
    NOTIFICATION_ROOTS_LIST_CHANGED = "notifications/roots/list_changed"


class MCPMessageBuilder:
    """Helper class to build common MCP messages."""

    def __init__(self, protocol: Optional[MCPProtocol] = None):
        self.protocol = protocol or MCPProtocol()

    def initialize_request(
        self,
        client_name: str,
        client_version: str,
        capabilities: Optional[dict[str, Any]] = None,
    ) -> JsonRpcRequest:
        """Build an initialize request."""
        return self.protocol.create_request(
            method=MCPMethods.INITIALIZE,
            params={
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": capabilities or {
                    "roots": {"listChanged": True},
                    "sampling": {},
                },
                "clientInfo": {
                    "name": client_name,
                    "version": client_version,
                },
            },
        )

    def initialized_notification(self) -> JsonRpcNotification:
        """Build an initialized notification."""
        return self.protocol.create_notification(
            method=MCPMethods.INITIALIZED,
            params={},
        )

    def ping_request(self) -> JsonRpcRequest:
        """Build a ping request."""
        return self.protocol.create_request(
            method=MCPMethods.PING,
            params={},
        )

    def tools_list_request(self, cursor: Optional[str] = None) -> JsonRpcRequest:
        """Build a tools/list request."""
        params = {}
        if cursor:
            params["cursor"] = cursor
        return self.protocol.create_request(
            method=MCPMethods.TOOLS_LIST,
            params=params if params else None,
        )

    def tools_call_request(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> JsonRpcRequest:
        """Build a tools/call request."""
        return self.protocol.create_request(
            method=MCPMethods.TOOLS_CALL,
            params={
                "name": name,
                "arguments": arguments,
            },
        )

    def resources_list_request(self, cursor: Optional[str] = None) -> JsonRpcRequest:
        """Build a resources/list request."""
        params = {}
        if cursor:
            params["cursor"] = cursor
        return self.protocol.create_request(
            method=MCPMethods.RESOURCES_LIST,
            params=params if params else None,
        )

    def resources_read_request(self, uri: str) -> JsonRpcRequest:
        """Build a resources/read request."""
        return self.protocol.create_request(
            method=MCPMethods.RESOURCES_READ,
            params={"uri": uri},
        )

    def resources_subscribe_request(self, uri: str) -> JsonRpcRequest:
        """Build a resources/subscribe request."""
        return self.protocol.create_request(
            method=MCPMethods.RESOURCES_SUBSCRIBE,
            params={"uri": uri},
        )

    def prompts_list_request(self, cursor: Optional[str] = None) -> JsonRpcRequest:
        """Build a prompts/list request."""
        params = {}
        if cursor:
            params["cursor"] = cursor
        return self.protocol.create_request(
            method=MCPMethods.PROMPTS_LIST,
            params=params if params else None,
        )

    def prompts_get_request(
        self,
        name: str,
        arguments: Optional[dict[str, str]] = None,
    ) -> JsonRpcRequest:
        """Build a prompts/get request."""
        params = {"name": name}
        if arguments:
            params["arguments"] = arguments
        return self.protocol.create_request(
            method=MCPMethods.PROMPTS_GET,
            params=params,
        )

    def logging_set_level_request(self, level: str) -> JsonRpcRequest:
        """Build a logging/setLevel request."""
        return self.protocol.create_request(
            method=MCPMethods.LOGGING_SET_LEVEL,
            params={"level": level},
        )
