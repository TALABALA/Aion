"""
AION MCP Protocol Types

Dataclasses representing MCP protocol messages and structures.
Based on MCP specification: https://modelcontextprotocol.io/

This module defines all the type structures needed for the Model Context Protocol,
including JSON-RPC message types, capability types, and configuration types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Literal, Optional, Union
import uuid


# ============================================
# Protocol Version
# ============================================

MCP_PROTOCOL_VERSION = "2024-11-05"


# ============================================
# JSON-RPC Base Types
# ============================================

@dataclass
class JsonRpcRequest:
    """JSON-RPC 2.0 request."""
    method: str
    id: Union[str, int]
    params: Optional[dict[str, Any]] = None
    jsonrpc: str = "2.0"

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        d = {"jsonrpc": self.jsonrpc, "method": self.method, "id": self.id}
        if self.params is not None:
            d["params"] = self.params
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "JsonRpcRequest":
        """Create from dictionary."""
        return cls(
            method=data["method"],
            id=data["id"],
            params=data.get("params"),
            jsonrpc=data.get("jsonrpc", "2.0"),
        )


@dataclass
class JsonRpcResponse:
    """JSON-RPC 2.0 response."""
    id: Union[str, int]
    result: Optional[Any] = None
    error: Optional[dict[str, Any]] = None
    jsonrpc: str = "2.0"

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        d = {"jsonrpc": self.jsonrpc, "id": self.id}
        if self.error is not None:
            d["error"] = self.error
        else:
            d["result"] = self.result
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "JsonRpcResponse":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            result=data.get("result"),
            error=data.get("error"),
            jsonrpc=data.get("jsonrpc", "2.0"),
        )

    @property
    def is_error(self) -> bool:
        """Check if this is an error response."""
        return self.error is not None


@dataclass
class JsonRpcNotification:
    """JSON-RPC 2.0 notification (no id, no response expected)."""
    method: str
    params: Optional[dict[str, Any]] = None
    jsonrpc: str = "2.0"

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        d = {"jsonrpc": self.jsonrpc, "method": self.method}
        if self.params is not None:
            d["params"] = self.params
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "JsonRpcNotification":
        """Create from dictionary."""
        return cls(
            method=data["method"],
            params=data.get("params"),
            jsonrpc=data.get("jsonrpc", "2.0"),
        )


class JsonRpcErrorCode(int, Enum):
    """Standard JSON-RPC error codes."""
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603

    # MCP-specific error codes
    SERVER_ERROR = -32000
    TOOL_NOT_FOUND = -32001
    RESOURCE_NOT_FOUND = -32002
    PROMPT_NOT_FOUND = -32003
    CONNECTION_CLOSED = -32004


@dataclass
class JsonRpcError:
    """JSON-RPC error object."""
    code: int
    message: str
    data: Optional[Any] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        d = {"code": self.code, "message": self.message}
        if self.data is not None:
            d["data"] = self.data
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "JsonRpcError":
        """Create from dictionary."""
        return cls(
            code=data["code"],
            message=data["message"],
            data=data.get("data"),
        )


# ============================================
# MCP Capability Types
# ============================================

@dataclass
class ToolsCapability:
    """Tools capability configuration."""
    listChanged: bool = False

    def to_dict(self) -> dict:
        return {"listChanged": self.listChanged}


@dataclass
class ResourcesCapability:
    """Resources capability configuration."""
    subscribe: bool = False
    listChanged: bool = False

    def to_dict(self) -> dict:
        return {"subscribe": self.subscribe, "listChanged": self.listChanged}


@dataclass
class PromptsCapability:
    """Prompts capability configuration."""
    listChanged: bool = False

    def to_dict(self) -> dict:
        return {"listChanged": self.listChanged}


@dataclass
class LoggingCapability:
    """Logging capability configuration."""

    def to_dict(self) -> dict:
        return {}


@dataclass
class ServerCapabilities:
    """Capabilities advertised by an MCP server."""
    tools: Optional[ToolsCapability] = None
    resources: Optional[ResourcesCapability] = None
    prompts: Optional[PromptsCapability] = None
    logging: Optional[LoggingCapability] = None
    experimental: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        d = {}
        if self.tools:
            d["tools"] = self.tools.to_dict()
        if self.resources:
            d["resources"] = self.resources.to_dict()
        if self.prompts:
            d["prompts"] = self.prompts.to_dict()
        if self.logging:
            d["logging"] = self.logging.to_dict()
        if self.experimental:
            d["experimental"] = self.experimental
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "ServerCapabilities":
        """Create from dictionary."""
        return cls(
            tools=ToolsCapability(**data["tools"]) if data.get("tools") else None,
            resources=ResourcesCapability(**data["resources"]) if data.get("resources") else None,
            prompts=PromptsCapability(**data["prompts"]) if data.get("prompts") else None,
            logging=LoggingCapability() if data.get("logging") is not None else None,
            experimental=data.get("experimental"),
        )


@dataclass
class RootsCapability:
    """Roots capability configuration."""
    listChanged: bool = False

    def to_dict(self) -> dict:
        return {"listChanged": self.listChanged}


@dataclass
class SamplingCapability:
    """Sampling capability configuration."""

    def to_dict(self) -> dict:
        return {}


@dataclass
class ClientCapabilities:
    """Capabilities advertised by an MCP client."""
    roots: Optional[RootsCapability] = None
    sampling: Optional[SamplingCapability] = None
    experimental: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        d = {}
        if self.roots:
            d["roots"] = self.roots.to_dict()
        if self.sampling:
            d["sampling"] = self.sampling.to_dict()
        if self.experimental:
            d["experimental"] = self.experimental
        return d


@dataclass
class Implementation:
    """Implementation info for client or server."""
    name: str
    version: str

    def to_dict(self) -> dict:
        return {"name": self.name, "version": self.version}

    @classmethod
    def from_dict(cls, data: dict) -> "Implementation":
        return cls(name=data["name"], version=data["version"])


# ============================================
# Tool Types
# ============================================

@dataclass
class ToolInputSchema:
    """JSON Schema for tool input."""
    type: str = "object"
    properties: dict[str, Any] = field(default_factory=dict)
    required: list[str] = field(default_factory=list)
    additionalProperties: Optional[bool] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        d = {
            "type": self.type,
            "properties": self.properties,
        }
        if self.required:
            d["required"] = self.required
        if self.additionalProperties is not None:
            d["additionalProperties"] = self.additionalProperties
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "ToolInputSchema":
        """Create from dictionary."""
        if isinstance(data, cls):
            return data
        return cls(
            type=data.get("type", "object"),
            properties=data.get("properties", {}),
            required=data.get("required", []),
            additionalProperties=data.get("additionalProperties"),
        )


@dataclass
class Tool:
    """An MCP tool definition."""
    name: str
    description: str
    inputSchema: ToolInputSchema

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.inputSchema.to_dict() if hasattr(self.inputSchema, 'to_dict') else self.inputSchema,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Tool":
        """Create from dictionary."""
        input_schema = data.get("inputSchema", {})
        if isinstance(input_schema, dict):
            input_schema = ToolInputSchema.from_dict(input_schema)
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            inputSchema=input_schema,
        )


@dataclass
class ToolCall:
    """A request to call a tool."""
    name: str
    arguments: dict[str, Any]

    def to_dict(self) -> dict:
        return {"name": self.name, "arguments": self.arguments}

    @classmethod
    def from_dict(cls, data: dict) -> "ToolCall":
        return cls(name=data["name"], arguments=data.get("arguments", {}))


@dataclass
class ToolResult:
    """Result from a tool call."""
    content: list[dict[str, Any]]  # Array of content blocks
    isError: bool = False

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "content": self.content,
            "isError": self.isError,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ToolResult":
        """Create from dictionary."""
        return cls(
            content=data.get("content", []),
            isError=data.get("isError", False),
        )

    def get_text(self) -> str:
        """Extract text content from result."""
        texts = []
        for item in self.content:
            if item.get("type") == "text":
                texts.append(item.get("text", ""))
        return "\n".join(texts)


# ============================================
# Resource Types
# ============================================

@dataclass
class Resource:
    """An MCP resource."""
    uri: str
    name: str
    description: Optional[str] = None
    mimeType: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        d = {"uri": self.uri, "name": self.name}
        if self.description:
            d["description"] = self.description
        if self.mimeType:
            d["mimeType"] = self.mimeType
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "Resource":
        """Create from dictionary."""
        return cls(
            uri=data["uri"],
            name=data["name"],
            description=data.get("description"),
            mimeType=data.get("mimeType"),
        )


@dataclass
class ResourceTemplate:
    """A URI template for resources."""
    uriTemplate: str
    name: str
    description: Optional[str] = None
    mimeType: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        d = {"uriTemplate": self.uriTemplate, "name": self.name}
        if self.description:
            d["description"] = self.description
        if self.mimeType:
            d["mimeType"] = self.mimeType
        return d


@dataclass
class ResourceContent:
    """Content of a resource."""
    uri: str
    mimeType: Optional[str] = None
    text: Optional[str] = None
    blob: Optional[str] = None  # Base64 encoded

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        d = {"uri": self.uri}
        if self.mimeType:
            d["mimeType"] = self.mimeType
        if self.text is not None:
            d["text"] = self.text
        if self.blob is not None:
            d["blob"] = self.blob
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "ResourceContent":
        """Create from dictionary."""
        return cls(
            uri=data["uri"],
            mimeType=data.get("mimeType"),
            text=data.get("text"),
            blob=data.get("blob"),
        )


# ============================================
# Prompt Types
# ============================================

@dataclass
class PromptArgument:
    """An argument for a prompt template."""
    name: str
    description: Optional[str] = None
    required: bool = False

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        d = {"name": self.name}
        if self.description:
            d["description"] = self.description
        d["required"] = self.required
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "PromptArgument":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            description=data.get("description"),
            required=data.get("required", False),
        )


@dataclass
class Prompt:
    """An MCP prompt template."""
    name: str
    description: Optional[str] = None
    arguments: list[PromptArgument] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        d = {"name": self.name}
        if self.description:
            d["description"] = self.description
        if self.arguments:
            d["arguments"] = [a.to_dict() for a in self.arguments]
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "Prompt":
        """Create from dictionary."""
        arguments = []
        for arg_data in data.get("arguments", []):
            arguments.append(PromptArgument.from_dict(arg_data))
        return cls(
            name=data["name"],
            description=data.get("description"),
            arguments=arguments,
        )


@dataclass
class PromptMessage:
    """A message in a prompt."""
    role: Literal["user", "assistant"]
    content: dict[str, Any]  # TextContent or ImageContent or EmbeddedResource

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}

    @classmethod
    def from_dict(cls, data: dict) -> "PromptMessage":
        return cls(role=data["role"], content=data["content"])


# ============================================
# Content Types
# ============================================

@dataclass
class TextContent:
    """Text content block."""
    text: str
    type: Literal["text"] = "text"

    def to_dict(self) -> dict:
        return {"type": self.type, "text": self.text}

    @classmethod
    def from_dict(cls, data: dict) -> "TextContent":
        return cls(text=data.get("text", ""))


@dataclass
class ImageContent:
    """Image content block."""
    data: str  # Base64 encoded
    mimeType: str = "image/png"
    type: Literal["image"] = "image"

    def to_dict(self) -> dict:
        return {"type": self.type, "data": self.data, "mimeType": self.mimeType}

    @classmethod
    def from_dict(cls, data: dict) -> "ImageContent":
        return cls(data=data.get("data", ""), mimeType=data.get("mimeType", "image/png"))


@dataclass
class EmbeddedResource:
    """Embedded resource content."""
    resource: ResourceContent
    type: Literal["resource"] = "resource"

    def to_dict(self) -> dict:
        return {"type": self.type, "resource": self.resource.to_dict()}

    @classmethod
    def from_dict(cls, data: dict) -> "EmbeddedResource":
        return cls(resource=ResourceContent.from_dict(data.get("resource", {"uri": ""})))


# ============================================
# Logging Types
# ============================================

class LogLevel(str, Enum):
    """MCP log levels."""
    DEBUG = "debug"
    INFO = "info"
    NOTICE = "notice"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    ALERT = "alert"
    EMERGENCY = "emergency"


@dataclass
class LogEntry:
    """A log entry."""
    level: LogLevel
    message: str
    logger: Optional[str] = None
    data: Optional[Any] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        d = {"level": self.level.value, "message": self.message}
        if self.logger:
            d["logger"] = self.logger
        if self.data is not None:
            d["data"] = self.data
        return d


# ============================================
# Server Configuration
# ============================================

class TransportType(str, Enum):
    """MCP transport types."""
    STDIO = "stdio"
    SSE = "sse"
    WEBSOCKET = "websocket"


@dataclass
class ServerConfig:
    """Configuration for an MCP server connection."""
    name: str
    transport: TransportType

    # For stdio transport
    command: Optional[str] = None
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    cwd: Optional[str] = None

    # For HTTP/SSE transport
    url: Optional[str] = None
    headers: dict[str, str] = field(default_factory=dict)

    # For WebSocket transport
    ws_url: Optional[str] = None

    # Common settings
    timeout: float = 30.0
    auto_reconnect: bool = True
    max_retries: int = 3
    retry_delay: float = 1.0

    # Credentials (reference to secure storage)
    credential_id: Optional[str] = None

    # Metadata
    description: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    enabled: bool = True

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "transport": self.transport.value,
            "command": self.command,
            "args": self.args,
            "env": self.env,
            "cwd": self.cwd,
            "url": self.url,
            "headers": self.headers,
            "ws_url": self.ws_url,
            "timeout": self.timeout,
            "auto_reconnect": self.auto_reconnect,
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay,
            "credential_id": self.credential_id,
            "description": self.description,
            "tags": self.tags,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ServerConfig":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            transport=TransportType(data.get("transport", "stdio")),
            command=data.get("command"),
            args=data.get("args", []),
            env=data.get("env", {}),
            cwd=data.get("cwd"),
            url=data.get("url"),
            headers=data.get("headers", {}),
            ws_url=data.get("ws_url"),
            timeout=data.get("timeout", 30.0),
            auto_reconnect=data.get("auto_reconnect", True),
            max_retries=data.get("max_retries", 3),
            retry_delay=data.get("retry_delay", 1.0),
            credential_id=data.get("credential_id"),
            description=data.get("description"),
            tags=data.get("tags", []),
            enabled=data.get("enabled", True),
        )


@dataclass
class ConnectedServer:
    """State of a connected MCP server."""
    config: ServerConfig
    capabilities: Optional[ServerCapabilities] = None
    tools: list[Tool] = field(default_factory=list)
    resources: list[Resource] = field(default_factory=list)
    prompts: list[Prompt] = field(default_factory=list)

    # Connection state
    connected: bool = False
    last_connected: Optional[datetime] = None
    last_error: Optional[str] = None

    # Statistics
    requests_sent: int = 0
    requests_successful: int = 0
    errors: int = 0
    total_latency_ms: float = 0.0

    @property
    def avg_latency_ms(self) -> float:
        """Calculate average latency."""
        if self.requests_successful == 0:
            return 0.0
        return self.total_latency_ms / self.requests_successful

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "config": self.config.to_dict(),
            "capabilities": self.capabilities.to_dict() if self.capabilities else None,
            "tools": [t.to_dict() for t in self.tools],
            "resources": [r.to_dict() for r in self.resources],
            "prompts": [p.to_dict() for p in self.prompts],
            "connected": self.connected,
            "last_connected": self.last_connected.isoformat() if self.last_connected else None,
            "last_error": self.last_error,
            "requests_sent": self.requests_sent,
            "requests_successful": self.requests_successful,
            "errors": self.errors,
            "avg_latency_ms": self.avg_latency_ms,
        }


# ============================================
# Initialize / Initialized Messages
# ============================================

@dataclass
class InitializeParams:
    """Parameters for initialize request."""
    protocolVersion: str
    capabilities: ClientCapabilities
    clientInfo: Implementation

    def to_dict(self) -> dict:
        return {
            "protocolVersion": self.protocolVersion,
            "capabilities": self.capabilities.to_dict(),
            "clientInfo": self.clientInfo.to_dict(),
        }


@dataclass
class InitializeResult:
    """Result of initialize request."""
    protocolVersion: str
    capabilities: ServerCapabilities
    serverInfo: Implementation
    instructions: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict) -> "InitializeResult":
        return cls(
            protocolVersion=data.get("protocolVersion", MCP_PROTOCOL_VERSION),
            capabilities=ServerCapabilities.from_dict(data.get("capabilities", {})),
            serverInfo=Implementation.from_dict(data.get("serverInfo", {"name": "unknown", "version": "0.0.0"})),
            instructions=data.get("instructions"),
        )
