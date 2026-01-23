"""
AION MCP Integration Layer

Model Context Protocol (MCP) integration for AION, enabling:
- Connection to external MCP servers for tools, resources, and prompts
- Exposing AION's capabilities as an MCP server
- Unified tool access across multiple MCP servers
- Secure credential management

Usage:
    from aion.mcp import MCPManager, ServerConfig, TransportType

    # Create and initialize manager
    mcp = MCPManager()
    await mcp.initialize()

    # Connect to a server
    await mcp.connect_server("filesystem")

    # Call a tool
    result = await mcp.call_tool("filesystem", "read_file", {"path": "/etc/hosts"})

    # List all available tools
    tools = mcp.list_all_tools()
"""

from aion.mcp.types import (
    # Protocol version
    MCP_PROTOCOL_VERSION,
    # JSON-RPC types
    JsonRpcRequest,
    JsonRpcResponse,
    JsonRpcNotification,
    JsonRpcError,
    JsonRpcErrorCode,
    # Capability types
    ServerCapabilities,
    ClientCapabilities,
    ToolsCapability,
    ResourcesCapability,
    PromptsCapability,
    LoggingCapability,
    RootsCapability,
    SamplingCapability,
    Implementation,
    # Tool types
    Tool,
    ToolInputSchema,
    ToolCall,
    ToolResult,
    # Resource types
    Resource,
    ResourceTemplate,
    ResourceContent,
    # Prompt types
    Prompt,
    PromptArgument,
    PromptMessage,
    # Content types
    TextContent,
    ImageContent,
    EmbeddedResource,
    # Logging
    LogLevel,
    LogEntry,
    # Configuration
    TransportType,
    ServerConfig,
    ConnectedServer,
    # Initialize types
    InitializeParams,
    InitializeResult,
)

from aion.mcp.protocol import (
    MCPProtocol,
    MCPProtocolError,
    ParseError,
    InvalidRequestError,
    MethodNotFoundError,
    InvalidParamsError,
    InternalError,
    ToolNotFoundError,
    ResourceNotFoundError,
    PromptNotFoundError,
    ConnectionClosedError,
    MCPMethods,
    MCPMessageBuilder,
)

from aion.mcp.transports import (
    Transport,
    TransportError,
    ConnectionError,
    SendError,
    ReceiveError,
    TimeoutError,
    StdioTransport,
    SSETransport,
    WebSocketTransport,
)

from aion.mcp.client import (
    MCPClient,
    MCPError,
    MCPSessionManager,
    MCPClientPool,
    MCPDiscovery,
    ServerDiscoveryResult,
    ToolDiscoveryResult,
    ServerProbe,
)

from aion.mcp.server import (
    MCPServer,
    MCPRequestHandler,
    setup_aion_mcp_server,
)

from aion.mcp.manager import MCPManager
from aion.mcp.registry import ServerRegistry
from aion.mcp.credentials import (
    CredentialManager,
    EnvironmentCredentialProvider,
    VaultCredentialProvider,
)
from aion.mcp.bridge import (
    MCPToolBridge,
    MCPResourceBridge,
    MCPPromptBridge,
)

__all__ = [
    # Version
    "MCP_PROTOCOL_VERSION",

    # Types - JSON-RPC
    "JsonRpcRequest",
    "JsonRpcResponse",
    "JsonRpcNotification",
    "JsonRpcError",
    "JsonRpcErrorCode",

    # Types - Capabilities
    "ServerCapabilities",
    "ClientCapabilities",
    "ToolsCapability",
    "ResourcesCapability",
    "PromptsCapability",
    "LoggingCapability",
    "RootsCapability",
    "SamplingCapability",
    "Implementation",

    # Types - Tools
    "Tool",
    "ToolInputSchema",
    "ToolCall",
    "ToolResult",

    # Types - Resources
    "Resource",
    "ResourceTemplate",
    "ResourceContent",

    # Types - Prompts
    "Prompt",
    "PromptArgument",
    "PromptMessage",

    # Types - Content
    "TextContent",
    "ImageContent",
    "EmbeddedResource",

    # Types - Logging
    "LogLevel",
    "LogEntry",

    # Types - Configuration
    "TransportType",
    "ServerConfig",
    "ConnectedServer",

    # Types - Initialize
    "InitializeParams",
    "InitializeResult",

    # Protocol
    "MCPProtocol",
    "MCPProtocolError",
    "ParseError",
    "InvalidRequestError",
    "MethodNotFoundError",
    "InvalidParamsError",
    "InternalError",
    "ToolNotFoundError",
    "ResourceNotFoundError",
    "PromptNotFoundError",
    "ConnectionClosedError",
    "MCPMethods",
    "MCPMessageBuilder",

    # Transports
    "Transport",
    "TransportError",
    "ConnectionError",
    "SendError",
    "ReceiveError",
    "TimeoutError",
    "StdioTransport",
    "SSETransport",
    "WebSocketTransport",

    # Client
    "MCPClient",
    "MCPError",
    "MCPSessionManager",
    "MCPClientPool",
    "MCPDiscovery",
    "ServerDiscoveryResult",
    "ToolDiscoveryResult",
    "ServerProbe",

    # Server
    "MCPServer",
    "MCPRequestHandler",
    "setup_aion_mcp_server",

    # Manager
    "MCPManager",

    # Registry
    "ServerRegistry",

    # Credentials
    "CredentialManager",
    "EnvironmentCredentialProvider",
    "VaultCredentialProvider",

    # Bridge
    "MCPToolBridge",
    "MCPResourceBridge",
    "MCPPromptBridge",
]
