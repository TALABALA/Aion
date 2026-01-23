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


# ==================== SOTA Feature Tests ====================

# Import SOTA modules
from aion.mcp.resilience import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerError,
    CircuitState,
    ExponentialBackoff,
    BackoffConfig,
    retry_with_backoff,
    RetryExhaustedError,
    TokenBucketRateLimiter,
    RateLimitExceededError,
    LRUCache,
    RequestDeduplicator,
    Bulkhead,
    BulkheadFullError,
)

from aion.mcp.validation import (
    SchemaValidator,
    ToolArgumentValidator,
    TypeCoercer,
    ValidationResult,
    SchemaValidationError,
    validate_tool_arguments,
)

from aion.mcp.streaming import (
    ProgressState,
    ProgressUpdate,
    ProgressNotifier,
    ProgressContext,
    StreamChunk,
    StreamingToolResult,
    ToolResultStreamer,
    ProgressStore,
    get_progress_store,
)

from aion.mcp.metrics import (
    MCPTracer,
    MCPMetrics,
    MCPHealthChecker,
    HealthStatus,
    HealthCheckResult,
)


class TestCircuitBreaker:
    """Test Circuit Breaker pattern."""

    @pytest.fixture
    def circuit(self):
        """Create circuit breaker with low thresholds for testing."""
        return CircuitBreaker(
            name="test_cb",
            config=CircuitBreakerConfig(
                failure_threshold=3,
                success_threshold=2,
                timeout=1.0,
            ),
        )

    def test_initial_state_closed(self, circuit):
        """Test circuit starts in closed state."""
        assert circuit.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_opens_after_failures(self, circuit):
        """Test circuit opens after failure threshold."""
        for _ in range(3):
            try:
                async with circuit:
                    raise ValueError("Simulated failure")
            except ValueError:
                pass

        assert circuit.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_rejects_when_open(self, circuit):
        """Test circuit rejects requests when open."""
        # Force open
        for _ in range(3):
            try:
                async with circuit:
                    raise ValueError("Simulated failure")
            except ValueError:
                pass

        with pytest.raises(CircuitBreakerError):
            async with circuit:
                pass

    @pytest.mark.asyncio
    async def test_success_resets_failures(self, circuit):
        """Test successful calls reset failure count."""
        # Two failures
        for _ in range(2):
            try:
                async with circuit:
                    raise ValueError("Failure")
            except ValueError:
                pass

        # One success should reset
        async with circuit:
            pass

        assert circuit._failure_count == 0

    def test_stats(self, circuit):
        """Test circuit breaker statistics."""
        stats = circuit.get_stats()

        assert stats["name"] == "test_cb"
        assert stats["state"] == "closed"
        assert "failure_count" in stats


class TestExponentialBackoff:
    """Test Exponential Backoff."""

    def test_default_config(self):
        """Test default backoff configuration."""
        backoff = ExponentialBackoff()

        assert backoff.config.base_delay == 1.0
        assert backoff.config.max_delay == 60.0
        assert backoff.config.max_retries == 5

    def test_delay_increases(self):
        """Test delay increases exponentially."""
        backoff = ExponentialBackoff(BackoffConfig(
            base_delay=1.0,
            multiplier=2.0,
            jitter=0.0,
            jitter_mode="equal",  # Use equal to reduce randomness
        ))

        delay0 = backoff.calculate_delay(0)
        delay1 = backoff.calculate_delay(1)
        delay2 = backoff.calculate_delay(2)

        # Due to jitter, we check approximate values
        assert delay1 > delay0 * 0.9
        assert delay2 > delay1 * 0.9

    def test_max_delay_cap(self):
        """Test delay is capped at max_delay."""
        backoff = ExponentialBackoff(BackoffConfig(
            base_delay=1.0,
            max_delay=5.0,
            multiplier=10.0,
            jitter=0.0,
        ))

        delay = backoff.calculate_delay(10)
        assert delay <= 5.0

    @pytest.mark.asyncio
    async def test_retry_with_backoff_success(self):
        """Test retry eventually succeeds."""
        call_count = 0

        async def eventually_succeeds():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Not yet")
            return "success"

        result = await retry_with_backoff(
            eventually_succeeds,
            config=BackoffConfig(base_delay=0.01, max_retries=5),
        )

        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_retry_exhausted(self):
        """Test retry raises after exhaustion."""
        async def always_fails():
            raise ValueError("Always fails")

        with pytest.raises(RetryExhaustedError):
            await retry_with_backoff(
                always_fails,
                config=BackoffConfig(base_delay=0.01, max_retries=2),
            )


class TestRateLimiter:
    """Test Rate Limiter."""

    @pytest.fixture
    def limiter(self):
        """Create rate limiter for testing."""
        return TokenBucketRateLimiter(
            name="test_rl",
            rate=10.0,  # 10 tokens per second
            capacity=5,  # Burst of 5
        )

    @pytest.mark.asyncio
    async def test_allows_burst(self, limiter):
        """Test allows burst up to capacity."""
        for _ in range(5):
            result = await limiter.acquire(block=False)
            assert result is True

    @pytest.mark.asyncio
    async def test_rejects_over_capacity(self, limiter):
        """Test rejects when over capacity."""
        # Exhaust capacity
        for _ in range(5):
            await limiter.acquire(block=False)

        # Should reject
        with pytest.raises(RateLimitExceededError):
            await limiter.acquire(block=False)

    def test_stats(self, limiter):
        """Test rate limiter statistics."""
        stats = limiter.get_stats()

        assert stats["name"] == "test_rl"
        assert stats["rate"] == 10.0
        assert stats["capacity"] == 5


class TestLRUCache:
    """Test LRU Cache."""

    @pytest.fixture
    def cache(self):
        """Create LRU cache for testing."""
        return LRUCache[str](
            name="test_cache",
            max_size=3,
            default_ttl=10.0,
        )

    @pytest.mark.asyncio
    async def test_set_get(self, cache):
        """Test basic set and get."""
        await cache.set("key1", "value1")
        result = await cache.get("key1")

        assert result == "value1"

    @pytest.mark.asyncio
    async def test_miss(self, cache):
        """Test cache miss returns None."""
        result = await cache.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_eviction_on_capacity(self, cache):
        """Test LRU eviction when at capacity."""
        await cache.set("a", "1")
        await cache.set("b", "2")
        await cache.set("c", "3")
        await cache.set("d", "4")  # Should evict "a"

        assert await cache.get("a") is None
        assert await cache.get("d") == "4"

    @pytest.mark.asyncio
    async def test_delete(self, cache):
        """Test deleting cache entry."""
        await cache.set("key", "value")
        await cache.delete("key")

        assert await cache.get("key") is None

    def test_stats(self, cache):
        """Test cache statistics."""
        stats = cache.get_stats()

        assert stats["name"] == "test_cache"
        assert stats["max_size"] == 3


class TestBulkhead:
    """Test Bulkhead pattern."""

    @pytest.fixture
    def bulkhead(self):
        """Create bulkhead for testing."""
        return Bulkhead(
            name="test_bh",
            max_concurrent=2,
        )

    @pytest.mark.asyncio
    async def test_allows_under_limit(self, bulkhead):
        """Test allows requests under limit."""
        async with bulkhead:
            async with bulkhead:
                assert bulkhead._current == 2

    @pytest.mark.asyncio
    async def test_rejects_at_capacity(self, bulkhead):
        """Test rejects at capacity."""
        async with bulkhead:
            async with bulkhead:
                with pytest.raises(BulkheadFullError):
                    await bulkhead.acquire()

    def test_stats(self, bulkhead):
        """Test bulkhead statistics."""
        stats = bulkhead.get_stats()

        assert stats["name"] == "test_bh"
        assert stats["max_concurrent"] == 2


class TestSchemaValidator:
    """Test Schema Validator."""

    @pytest.fixture
    def validator(self):
        """Create schema validator for testing."""
        return SchemaValidator(coerce_types=True, apply_defaults=True)

    def test_valid_data(self, validator):
        """Test validation of valid data."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name"],
        }

        result = validator.validate({"name": "Alice", "age": 30}, schema)

        assert result.valid
        assert len(result.errors) == 0

    def test_missing_required(self, validator):
        """Test validation catches missing required field."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
            },
            "required": ["name"],
        }

        result = validator.validate({}, schema)

        assert not result.valid
        assert any("name" in e.path for e in result.errors)

    def test_type_coercion(self, validator):
        """Test type coercion."""
        schema = {
            "type": "object",
            "properties": {
                "count": {"type": "integer"},
            },
        }

        result = validator.validate({"count": "42"}, schema)

        assert result.valid
        assert result.coerced_data["count"] == 42

    def test_default_values(self, validator):
        """Test default value application."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "default": "Unknown"},
            },
        }

        result = validator.validate({}, schema)

        assert result.coerced_data["name"] == "Unknown"


class TestTypeCoercer:
    """Test Type Coercer."""

    def test_string_coercion(self):
        """Test coercing to string."""
        assert TypeCoercer.coerce_string(123) == "123"
        assert TypeCoercer.coerce_string(True) == "True"

    def test_integer_coercion(self):
        """Test coercing to integer."""
        assert TypeCoercer.coerce_integer("42") == 42
        assert TypeCoercer.coerce_integer(3.0) == 3
        assert TypeCoercer.coerce_integer("invalid") is None

    def test_boolean_coercion(self):
        """Test coercing to boolean."""
        assert TypeCoercer.coerce_boolean("true") is True
        assert TypeCoercer.coerce_boolean("false") is False
        assert TypeCoercer.coerce_boolean("yes") is True
        assert TypeCoercer.coerce_boolean(1) is True


class TestToolArgumentValidator:
    """Test Tool Argument Validator."""

    @pytest.fixture
    def validator(self):
        """Create tool argument validator."""
        return ToolArgumentValidator()

    def test_validate_tool_arguments(self, validator):
        """Test validating tool arguments."""
        schema = {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "default": 10},
            },
            "required": ["query"],
        }

        result = validator.validate_tool_arguments(
            "test_tool",
            {"query": "search term"},
            schema,
        )

        assert result.valid
        assert result.coerced_data["limit"] == 10

    def test_prepare_arguments_raises(self, validator):
        """Test prepare_arguments raises on invalid."""
        schema = {
            "type": "object",
            "properties": {"required_field": {"type": "string"}},
            "required": ["required_field"],
        }

        with pytest.raises(SchemaValidationError):
            validator.prepare_arguments("test_tool", {}, schema)


class TestProgressNotifier:
    """Test Progress Notifier."""

    @pytest.fixture
    def progress(self):
        """Create progress notifier for testing."""
        return ProgressNotifier(
            total_steps=10,
            description="Test operation",
        )

    @pytest.mark.asyncio
    async def test_initial_state(self, progress):
        """Test initial progress state."""
        assert progress._state == ProgressState.PENDING
        assert progress.progress == 0.0

    @pytest.mark.asyncio
    async def test_start(self, progress):
        """Test starting progress."""
        await progress.start()
        assert progress._state == ProgressState.RUNNING

    @pytest.mark.asyncio
    async def test_update_progress(self, progress):
        """Test updating progress."""
        await progress.start()
        await progress.update(increment=5)

        assert progress._current_step == 5
        assert progress.progress == 0.5

    @pytest.mark.asyncio
    async def test_complete(self, progress):
        """Test completing progress."""
        await progress.start()
        await progress.complete()

        assert progress._state == ProgressState.COMPLETED
        assert progress.progress == 1.0

    @pytest.mark.asyncio
    async def test_fail(self, progress):
        """Test failing progress."""
        await progress.start()
        await progress.fail("Something went wrong")

        assert progress._state == ProgressState.FAILED

    @pytest.mark.asyncio
    async def test_subscriber_notification(self, progress):
        """Test subscriber receives notifications."""
        updates = []

        def on_update(update: ProgressUpdate):
            updates.append(update)

        progress.subscribe(on_update)
        await progress.start()
        await progress.update(increment=1)

        assert len(updates) >= 2


class TestProgressContext:
    """Test Progress Context Manager."""

    @pytest.mark.asyncio
    async def test_context_complete(self):
        """Test context manager on successful completion."""
        updates = []

        async with ProgressContext(
            "Test task",
            total=5,
            on_progress=updates.append,
        ) as progress:
            for i in range(5):
                await progress.update(increment=1)

        # Should have start + 5 updates + complete
        assert len(updates) >= 5
        assert updates[-1].state == ProgressState.COMPLETED

    @pytest.mark.asyncio
    async def test_context_failure(self):
        """Test context manager on failure."""
        updates = []

        with pytest.raises(ValueError):
            async with ProgressContext(
                "Failing task",
                on_progress=updates.append,
            ) as progress:
                raise ValueError("Error")

        assert updates[-1].state == ProgressState.FAILED


class TestStreamingToolResult:
    """Test Streaming Tool Result."""

    @pytest.mark.asyncio
    async def test_send_and_iterate(self):
        """Test sending and iterating over chunks."""
        result = StreamingToolResult()
        await result.start()

        # Send in background
        async def send_chunks():
            for i in range(3):
                await result.send(f"chunk_{i}")
            await result.complete()

        asyncio.create_task(send_chunks())

        chunks = []
        async for chunk in result:
            chunks.append(chunk.data)

        assert len(chunks) == 3
        assert chunks == ["chunk_0", "chunk_1", "chunk_2"]

    @pytest.mark.asyncio
    async def test_stats(self):
        """Test streaming statistics."""
        result = StreamingToolResult()
        await result.start()

        await result.send("data")
        await result.complete()

        stats = result.get_stats()
        assert stats["chunks_sent"] == 1
        assert stats["completed"] is True


class TestProgressStore:
    """Test Progress Store."""

    @pytest.fixture
    def store(self):
        """Create progress store for testing."""
        return ProgressStore(max_entries=10)

    @pytest.mark.asyncio
    async def test_create_and_get(self, store):
        """Test creating and getting progress."""
        notifier = await store.create(
            total_steps=100,
            description="Test",
        )

        retrieved = await store.get(notifier.operation_id)
        assert retrieved is notifier

    @pytest.mark.asyncio
    async def test_get_status(self, store):
        """Test getting operation status."""
        notifier = await store.create(description="Test")
        await notifier.start()

        status = await store.get_status(notifier.operation_id)
        assert status is not None
        assert status.state == ProgressState.RUNNING

    @pytest.mark.asyncio
    async def test_list_active(self, store):
        """Test listing active operations."""
        n1 = await store.create(description="Active 1")
        n2 = await store.create(description="Active 2")

        await n1.start()
        await n2.start()

        active = await store.list_active()
        assert len(active) == 2


class TestRequestDeduplicator:
    """Test Request Deduplicator."""

    @pytest.fixture
    def dedup(self):
        """Create deduplicator for testing."""
        return RequestDeduplicator(name="test_dedup", ttl=1.0)

    @pytest.mark.asyncio
    async def test_deduplication(self, dedup):
        """Test identical concurrent requests are deduplicated."""
        call_count = 0

        async def expensive_operation(x):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.1)
            return x * 2

        # Run two identical requests concurrently
        result1, result2 = await asyncio.gather(
            dedup.execute(expensive_operation, 5),
            dedup.execute(expensive_operation, 5),
        )

        assert result1 == result2 == 10
        assert call_count == 1  # Only executed once

    @pytest.mark.asyncio
    async def test_different_args_not_deduplicated(self, dedup):
        """Test different arguments are not deduplicated."""
        call_count = 0

        async def operation(x):
            nonlocal call_count
            call_count += 1
            return x

        # Run with different args
        result1, result2 = await asyncio.gather(
            dedup.execute(operation, 1),
            dedup.execute(operation, 2),
        )

        assert result1 == 1
        assert result2 == 2
        assert call_count == 2


class TestMCPMetrics:
    """Test MCP Metrics."""

    @pytest.fixture
    def metrics(self):
        """Create metrics for testing."""
        return MCPMetrics(namespace="test_mcp")

    def test_record_connection(self, metrics):
        """Test recording connection."""
        # Should not raise even if Prometheus not available
        metrics.record_connection(
            server="test",
            transport="stdio",
            success=True,
            duration=0.5,
        )

    def test_record_tool_call(self, metrics):
        """Test recording tool call."""
        metrics.record_tool_call(
            server="test",
            tool="example_tool",
            success=True,
            duration=0.1,
        )


class TestHealthCheck:
    """Test Health Check."""

    def test_health_status_values(self):
        """Test health status enum values."""
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.DEGRADED.value == "degraded"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"

    def test_health_check_result(self):
        """Test health check result creation."""
        result = HealthCheckResult(
            name="test_check",
            status=HealthStatus.HEALTHY,
            message="All good",
        )

        d = result.to_dict()
        assert d["name"] == "test_check"
        assert d["status"] == "healthy"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
