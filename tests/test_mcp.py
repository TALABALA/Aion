"""
AION MCP Integration Tests

Comprehensive tests for the MCP integration layer.
"""

from __future__ import annotations

import asyncio
import json
import pytest
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

# Import MCP types
from aion.mcp.types import (
    MCP_PROTOCOL_VERSION,
    JsonRpcRequest,
    JsonRpcResponse,
    JsonRpcNotification,
    JsonRpcError,
    JsonRpcErrorCode,
    ServerCapabilities,
    ClientCapabilities,
    ToolsCapability,
    ResourcesCapability,
    PromptsCapability,
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
    TransportType,
    ServerConfig,
    ConnectedServer,
    InitializeResult,
)

from aion.mcp.protocol import (
    MCPProtocol,
    MCPProtocolError,
    ParseError,
    InvalidRequestError,
    MethodNotFoundError,
    MCPMethods,
    MCPMessageBuilder,
)

from aion.mcp.transports.base import Transport
from aion.mcp.transports.stdio import StdioTransport
from aion.mcp.registry import ServerRegistry, DEFAULT_SERVERS
from aion.mcp.credentials import CredentialManager


# ==================== Fixtures ====================

@pytest.fixture
def protocol():
    """Create MCPProtocol instance."""
    return MCPProtocol()


@pytest.fixture
def message_builder(protocol):
    """Create MCPMessageBuilder instance."""
    return MCPMessageBuilder(protocol)


@pytest.fixture
def sample_server_config():
    """Create sample server configuration."""
    return ServerConfig(
        name="test-server",
        transport=TransportType.STDIO,
        command="echo",
        args=["test"],
        description="Test MCP server",
        tags=["test"],
    )


@pytest.fixture
def sample_tool():
    """Create sample tool."""
    return Tool(
        name="test_tool",
        description="A test tool",
        inputSchema=ToolInputSchema(
            type="object",
            properties={
                "param1": {"type": "string", "description": "First parameter"},
                "param2": {"type": "integer", "description": "Second parameter"},
            },
            required=["param1"],
        ),
    )


@pytest.fixture
def sample_resource():
    """Create sample resource."""
    return Resource(
        uri="file:///test/resource.txt",
        name="Test Resource",
        description="A test resource",
        mimeType="text/plain",
    )


@pytest.fixture
def sample_prompt():
    """Create sample prompt."""
    return Prompt(
        name="test_prompt",
        description="A test prompt",
        arguments=[
            PromptArgument(name="topic", description="Topic to discuss", required=True),
            PromptArgument(name="style", description="Writing style", required=False),
        ],
    )


# ==================== Type Tests ====================

class TestJsonRpcTypes:
    """Test JSON-RPC type classes."""

    def test_request_creation(self):
        """Test creating a JSON-RPC request."""
        request = JsonRpcRequest(
            method="test/method",
            id="1",
            params={"key": "value"},
        )

        assert request.method == "test/method"
        assert request.id == "1"
        assert request.params == {"key": "value"}
        assert request.jsonrpc == "2.0"

    def test_request_to_dict(self):
        """Test request serialization."""
        request = JsonRpcRequest(method="test", id=1, params={"a": 1})
        d = request.to_dict()

        assert d["jsonrpc"] == "2.0"
        assert d["method"] == "test"
        assert d["id"] == 1
        assert d["params"] == {"a": 1}

    def test_response_creation(self):
        """Test creating a JSON-RPC response."""
        response = JsonRpcResponse(id="1", result={"data": "test"})

        assert response.id == "1"
        assert response.result == {"data": "test"}
        assert response.error is None
        assert not response.is_error

    def test_error_response(self):
        """Test creating an error response."""
        response = JsonRpcResponse(
            id="1",
            error={"code": -32600, "message": "Invalid request"},
        )

        assert response.is_error
        assert response.error["code"] == -32600

    def test_notification_creation(self):
        """Test creating a notification."""
        notification = JsonRpcNotification(
            method="test/notification",
            params={"event": "occurred"},
        )

        assert notification.method == "test/notification"
        assert notification.params == {"event": "occurred"}

    def test_request_from_dict(self):
        """Test creating request from dictionary."""
        data = {
            "jsonrpc": "2.0",
            "method": "test",
            "id": "123",
            "params": {"x": 1},
        }
        request = JsonRpcRequest.from_dict(data)

        assert request.method == "test"
        assert request.id == "123"
        assert request.params == {"x": 1}


class TestCapabilityTypes:
    """Test capability type classes."""

    def test_server_capabilities(self):
        """Test ServerCapabilities creation and serialization."""
        caps = ServerCapabilities(
            tools=ToolsCapability(listChanged=True),
            resources=ResourcesCapability(subscribe=True, listChanged=True),
            prompts=PromptsCapability(listChanged=False),
        )

        d = caps.to_dict()
        assert d["tools"]["listChanged"] is True
        assert d["resources"]["subscribe"] is True
        assert d["prompts"]["listChanged"] is False

    def test_implementation(self):
        """Test Implementation class."""
        impl = Implementation(name="test-client", version="1.0.0")

        d = impl.to_dict()
        assert d["name"] == "test-client"
        assert d["version"] == "1.0.0"


class TestToolTypes:
    """Test tool type classes."""

    def test_tool_input_schema(self):
        """Test ToolInputSchema creation."""
        schema = ToolInputSchema(
            type="object",
            properties={
                "name": {"type": "string"},
                "count": {"type": "integer"},
            },
            required=["name"],
        )

        d = schema.to_dict()
        assert d["type"] == "object"
        assert "name" in d["properties"]
        assert "name" in d["required"]

    def test_tool_creation(self, sample_tool):
        """Test Tool creation and serialization."""
        d = sample_tool.to_dict()

        assert d["name"] == "test_tool"
        assert d["description"] == "A test tool"
        assert "inputSchema" in d

    def test_tool_result(self):
        """Test ToolResult creation."""
        result = ToolResult(
            content=[{"type": "text", "text": "Success!"}],
            isError=False,
        )

        assert result.get_text() == "Success!"
        assert not result.isError

    def test_tool_result_error(self):
        """Test ToolResult with error."""
        result = ToolResult(
            content=[{"type": "text", "text": "Failed!"}],
            isError=True,
        )

        assert result.isError
        assert result.get_text() == "Failed!"


class TestResourceTypes:
    """Test resource type classes."""

    def test_resource_creation(self, sample_resource):
        """Test Resource creation."""
        d = sample_resource.to_dict()

        assert d["uri"] == "file:///test/resource.txt"
        assert d["name"] == "Test Resource"
        assert d["mimeType"] == "text/plain"

    def test_resource_content(self):
        """Test ResourceContent creation."""
        content = ResourceContent(
            uri="file:///test.txt",
            mimeType="text/plain",
            text="Hello, World!",
        )

        d = content.to_dict()
        assert d["uri"] == "file:///test.txt"
        assert d["text"] == "Hello, World!"


class TestPromptTypes:
    """Test prompt type classes."""

    def test_prompt_creation(self, sample_prompt):
        """Test Prompt creation."""
        d = sample_prompt.to_dict()

        assert d["name"] == "test_prompt"
        assert len(d["arguments"]) == 2
        assert d["arguments"][0]["name"] == "topic"
        assert d["arguments"][0]["required"] is True

    def test_prompt_message(self):
        """Test PromptMessage creation."""
        message = PromptMessage(
            role="user",
            content={"type": "text", "text": "Hello!"},
        )

        d = message.to_dict()
        assert d["role"] == "user"
        assert d["content"]["text"] == "Hello!"


class TestServerConfig:
    """Test ServerConfig class."""

    def test_stdio_config(self, sample_server_config):
        """Test stdio server configuration."""
        assert sample_server_config.name == "test-server"
        assert sample_server_config.transport == TransportType.STDIO
        assert sample_server_config.command == "echo"

    def test_sse_config(self):
        """Test SSE server configuration."""
        config = ServerConfig(
            name="sse-server",
            transport=TransportType.SSE,
            url="https://api.example.com/mcp",
            headers={"Authorization": "Bearer token"},
        )

        assert config.transport == TransportType.SSE
        assert config.url == "https://api.example.com/mcp"

    def test_config_serialization(self, sample_server_config):
        """Test config serialization and deserialization."""
        d = sample_server_config.to_dict()
        restored = ServerConfig.from_dict(d)

        assert restored.name == sample_server_config.name
        assert restored.transport == sample_server_config.transport
        assert restored.command == sample_server_config.command


# ==================== Protocol Tests ====================

class TestMCPProtocol:
    """Test MCPProtocol class."""

    def test_create_request(self, protocol):
        """Test creating a request."""
        request = protocol.create_request(
            method="test/method",
            params={"key": "value"},
        )

        assert request.method == "test/method"
        assert request.params == {"key": "value"}
        assert request.id is not None

    def test_create_response(self, protocol):
        """Test creating a response."""
        response = protocol.create_response(
            request_id="1",
            result={"success": True},
        )

        assert response.id == "1"
        assert response.result == {"success": True}

    def test_create_notification(self, protocol):
        """Test creating a notification."""
        notification = protocol.create_notification(
            method="test/notify",
            params={"event": "test"},
        )

        assert notification.method == "test/notify"
        assert notification.params == {"event": "test"}

    def test_serialize(self, protocol):
        """Test message serialization."""
        request = protocol.create_request("test", {})
        serialized = protocol.serialize(request)

        assert isinstance(serialized, str)
        data = json.loads(serialized)
        assert data["method"] == "test"

    def test_parse_request(self, protocol):
        """Test parsing a request."""
        data = json.dumps({
            "jsonrpc": "2.0",
            "method": "test",
            "id": "1",
            "params": {},
        })

        parsed = protocol.parse(data)
        assert protocol.is_request(parsed)
        assert parsed.method == "test"

    def test_parse_response(self, protocol):
        """Test parsing a response."""
        data = json.dumps({
            "jsonrpc": "2.0",
            "id": "1",
            "result": {"data": "test"},
        })

        parsed = protocol.parse(data)
        assert protocol.is_response(parsed)

    def test_parse_notification(self, protocol):
        """Test parsing a notification."""
        data = json.dumps({
            "jsonrpc": "2.0",
            "method": "notify",
            "params": {},
        })

        parsed = protocol.parse(data)
        assert protocol.is_notification(parsed)

    def test_parse_invalid_json(self, protocol):
        """Test parsing invalid JSON."""
        with pytest.raises(ParseError):
            protocol.parse("not valid json")

    def test_request_id_generation(self, protocol):
        """Test request ID auto-generation."""
        req1 = protocol.create_request("test1", {})
        req2 = protocol.create_request("test2", {})

        assert req1.id != req2.id


class TestMCPMessageBuilder:
    """Test MCPMessageBuilder class."""

    def test_initialize_request(self, message_builder):
        """Test building initialize request."""
        request = message_builder.initialize_request(
            client_name="test-client",
            client_version="1.0.0",
        )

        assert request.method == MCPMethods.INITIALIZE
        assert request.params["clientInfo"]["name"] == "test-client"
        assert request.params["protocolVersion"] == MCP_PROTOCOL_VERSION

    def test_tools_list_request(self, message_builder):
        """Test building tools/list request."""
        request = message_builder.tools_list_request()
        assert request.method == MCPMethods.TOOLS_LIST

    def test_tools_call_request(self, message_builder):
        """Test building tools/call request."""
        request = message_builder.tools_call_request(
            name="test_tool",
            arguments={"param": "value"},
        )

        assert request.method == MCPMethods.TOOLS_CALL
        assert request.params["name"] == "test_tool"
        assert request.params["arguments"] == {"param": "value"}


# ==================== Registry Tests ====================

class TestServerRegistry:
    """Test ServerRegistry class."""

    @pytest.fixture
    def registry(self, tmp_path):
        """Create registry with temp config path."""
        return ServerRegistry(config_path=tmp_path / "mcp_servers.json")

    @pytest.mark.asyncio
    async def test_load_defaults(self, registry):
        """Test loading default servers."""
        await registry.load()

        servers = registry.get_all_servers()
        assert len(servers) > 0

    @pytest.mark.asyncio
    async def test_register_server(self, registry, sample_server_config):
        """Test registering a server."""
        await registry.load()

        registry.register(sample_server_config)
        server = registry.get_server("test-server")

        assert server is not None
        assert server.name == "test-server"

    @pytest.mark.asyncio
    async def test_unregister_server(self, registry, sample_server_config):
        """Test unregistering a server."""
        await registry.load()

        registry.register(sample_server_config)
        result = registry.unregister("test-server")

        assert result is True
        assert registry.get_server("test-server") is None

    @pytest.mark.asyncio
    async def test_enable_disable_server(self, registry, sample_server_config):
        """Test enabling and disabling servers."""
        await registry.load()

        sample_server_config.enabled = False
        registry.register(sample_server_config)

        assert not registry.is_enabled("test-server")

        registry.enable_server("test-server")
        assert registry.is_enabled("test-server")

        registry.disable_server("test-server")
        assert not registry.is_enabled("test-server")

    @pytest.mark.asyncio
    async def test_get_enabled_servers(self, registry):
        """Test getting only enabled servers."""
        await registry.load()

        enabled = registry.get_enabled_servers()
        for server in enabled:
            assert server.enabled

    @pytest.mark.asyncio
    async def test_find_by_tag(self, registry, sample_server_config):
        """Test finding servers by tag."""
        await registry.load()

        sample_server_config.tags = ["custom", "testing"]
        registry.register(sample_server_config)

        results = registry.find_by_tag("testing")
        assert len(results) >= 1
        assert any(s.name == "test-server" for s in results)

    @pytest.mark.asyncio
    async def test_save_load_config(self, registry, sample_server_config, tmp_path):
        """Test saving and loading configuration."""
        await registry.load()

        registry.register(sample_server_config)
        await registry.save()

        # Create new registry and load
        new_registry = ServerRegistry(config_path=tmp_path / "mcp_servers.json")
        await new_registry.load()

        server = new_registry.get_server("test-server")
        assert server is not None


# ==================== Credential Tests ====================

class TestCredentialManager:
    """Test CredentialManager class."""

    @pytest.fixture
    def credentials(self, tmp_path):
        """Create credential manager with temp path."""
        return CredentialManager(credentials_path=tmp_path / "creds.json")

    @pytest.mark.asyncio
    async def test_initialize(self, credentials):
        """Test credential manager initialization."""
        await credentials.initialize()
        assert credentials._initialized

    @pytest.mark.asyncio
    async def test_set_get_credentials(self, credentials):
        """Test setting and getting credentials."""
        await credentials.initialize()

        await credentials.set(
            "test_cred",
            {"API_KEY": "secret123"},
            persist=False,
        )

        cred = await credentials.get("test_cred")
        assert cred is not None
        assert cred["API_KEY"] == "secret123"

    @pytest.mark.asyncio
    async def test_delete_credentials(self, credentials):
        """Test deleting credentials."""
        await credentials.initialize()

        await credentials.set("test_cred", {"key": "value"}, persist=False)
        result = await credentials.delete("test_cred", persist=False)

        assert result is True
        assert await credentials.get("test_cred") is None

    @pytest.mark.asyncio
    async def test_has_credential(self, credentials):
        """Test checking credential existence."""
        await credentials.initialize()

        assert not credentials.has_credential("nonexistent")

        await credentials.set("test", {"key": "value"}, persist=False)
        assert credentials.has_credential("test")

    @pytest.mark.asyncio
    async def test_list_credentials(self, credentials):
        """Test listing credentials."""
        await credentials.initialize()

        await credentials.set("cred1", {"a": "1"}, persist=False)
        await credentials.set("cred2", {"b": "2"}, persist=False)

        creds = credentials.list_credentials()
        assert "cred1" in creds
        assert "cred2" in creds


# ==================== Connected Server Tests ====================

class TestConnectedServer:
    """Test ConnectedServer class."""

    def test_connected_server_creation(self, sample_server_config, sample_tool):
        """Test creating ConnectedServer."""
        server = ConnectedServer(
            config=sample_server_config,
            capabilities=ServerCapabilities(tools=ToolsCapability()),
            tools=[sample_tool],
            connected=True,
            last_connected=datetime.now(),
        )

        assert server.connected
        assert len(server.tools) == 1

    def test_avg_latency_calculation(self, sample_server_config):
        """Test average latency calculation."""
        server = ConnectedServer(
            config=sample_server_config,
            requests_successful=10,
            total_latency_ms=1000.0,
        )

        assert server.avg_latency_ms == 100.0

    def test_avg_latency_zero_requests(self, sample_server_config):
        """Test average latency with zero requests."""
        server = ConnectedServer(
            config=sample_server_config,
            requests_successful=0,
        )

        assert server.avg_latency_ms == 0.0

    def test_to_dict(self, sample_server_config, sample_tool):
        """Test serialization."""
        server = ConnectedServer(
            config=sample_server_config,
            tools=[sample_tool],
            connected=True,
        )

        d = server.to_dict()
        assert d["connected"] is True
        assert len(d["tools"]) == 1


# ==================== Transport Tests ====================

class TestStdioTransport:
    """Test StdioTransport class."""

    def test_creation(self):
        """Test transport creation."""
        transport = StdioTransport(
            command="echo",
            args=["test"],
        )

        assert transport.command == "echo"
        assert transport.args == ["test"]
        assert not transport.connected

    @pytest.mark.asyncio
    async def test_connect_disconnect(self):
        """Test connecting and disconnecting."""
        transport = StdioTransport(
            command="cat",
            args=[],
        )

        await transport.connect()
        assert transport.connected
        assert transport.process_pid is not None

        await transport.close()
        assert not transport.connected


# ==================== Integration Tests ====================

class TestMCPIntegration:
    """Integration tests for MCP system."""

    @pytest.mark.asyncio
    async def test_full_tool_workflow(self):
        """Test complete tool discovery and call workflow."""
        # This is a mock test - would need real MCP server for full integration
        protocol = MCPProtocol()
        builder = MCPMessageBuilder(protocol)

        # Build initialize request
        init_request = builder.initialize_request("test", "1.0.0")
        assert init_request.method == MCPMethods.INITIALIZE

        # Build tools list request
        tools_request = builder.tools_list_request()
        assert tools_request.method == MCPMethods.TOOLS_LIST

        # Build tool call request
        call_request = builder.tools_call_request(
            name="test_tool",
            arguments={"param": "value"},
        )
        assert call_request.method == MCPMethods.TOOLS_CALL

    @pytest.mark.asyncio
    async def test_full_resource_workflow(self):
        """Test complete resource discovery and read workflow."""
        protocol = MCPProtocol()
        builder = MCPMessageBuilder(protocol)

        # Build resources list request
        list_request = builder.resources_list_request()
        assert list_request.method == MCPMethods.RESOURCES_LIST

        # Build resource read request
        read_request = builder.resources_read_request("file:///test.txt")
        assert read_request.method == MCPMethods.RESOURCES_READ
        assert read_request.params["uri"] == "file:///test.txt"

    @pytest.mark.asyncio
    async def test_full_prompt_workflow(self):
        """Test complete prompt discovery and get workflow."""
        protocol = MCPProtocol()
        builder = MCPMessageBuilder(protocol)

        # Build prompts list request
        list_request = builder.prompts_list_request()
        assert list_request.method == MCPMethods.PROMPTS_LIST

        # Build prompt get request
        get_request = builder.prompts_get_request(
            name="test_prompt",
            arguments={"topic": "AI"},
        )
        assert get_request.method == MCPMethods.PROMPTS_GET
        assert get_request.params["name"] == "test_prompt"


# ==================== Error Handling Tests ====================

class TestErrorHandling:
    """Test error handling in MCP system."""

    def test_protocol_error_creation(self):
        """Test creating protocol errors."""
        error = MCPProtocolError(
            code=JsonRpcErrorCode.INVALID_REQUEST,
            message="Invalid request",
            data={"details": "Missing method"},
        )

        assert error.code == JsonRpcErrorCode.INVALID_REQUEST
        assert "Invalid request" in str(error)

    def test_parse_error(self):
        """Test ParseError."""
        error = ParseError("Could not parse JSON")
        assert error.code == JsonRpcErrorCode.PARSE_ERROR

    def test_method_not_found_error(self):
        """Test MethodNotFoundError."""
        error = MethodNotFoundError("unknown/method")
        assert error.code == JsonRpcErrorCode.METHOD_NOT_FOUND
        assert "unknown/method" in error.message


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
