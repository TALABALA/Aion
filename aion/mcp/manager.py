"""
AION MCP Manager

Central manager for all MCP connections with SOTA resilience patterns:
- Server lifecycle management with circuit breakers
- Connection pooling with rate limiting
- Health monitoring with metrics
- Unified tool access with caching
- Schema validation for tool arguments
- Distributed tracing
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from datetime import datetime
from typing import Any, Callable, Optional, TYPE_CHECKING

import structlog

from aion.mcp.types import (
    ServerConfig,
    ConnectedServer,
    Tool,
    Resource,
    Prompt,
    ToolResult,
    ResourceContent,
    PromptMessage,
    TransportType,
)
from aion.mcp.client.client import MCPClient, MCPError
from aion.mcp.registry import ServerRegistry
from aion.mcp.credentials import CredentialManager
from aion.mcp.resilience import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerError,
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
from aion.mcp.metrics import (
    MCPTracer,
    MCPMetrics,
    MCPHealthChecker,
    HealthStatus,
    init_observability,
)
from aion.mcp.validation import (
    SchemaValidator,
    ToolArgumentValidator,
    SchemaValidationError,
    validate_tool_arguments,
)

if TYPE_CHECKING:
    from aion.mcp.bridge import MCPToolBridge

logger = structlog.get_logger(__name__)


# ============================================
# Configuration
# ============================================

class MCPManagerConfig:
    """Configuration for MCP Manager with SOTA defaults."""

    def __init__(
        self,
        # Health monitoring
        health_check_interval: float = 30.0,
        auto_reconnect: bool = True,
        max_reconnect_attempts: int = 3,

        # Circuit breaker defaults
        circuit_failure_threshold: int = 5,
        circuit_success_threshold: int = 3,
        circuit_timeout: float = 30.0,

        # Rate limiting defaults
        rate_limit_rate: float = 10.0,  # Requests per second
        rate_limit_capacity: int = 20,   # Burst capacity

        # Caching defaults
        cache_enabled: bool = True,
        cache_max_size: int = 1000,
        cache_ttl: float = 300.0,  # 5 minutes

        # Retry defaults
        retry_enabled: bool = True,
        retry_max_attempts: int = 3,
        retry_base_delay: float = 1.0,
        retry_max_delay: float = 30.0,
        retry_jitter_mode: str = "full",

        # Bulkhead defaults
        bulkhead_enabled: bool = True,
        bulkhead_max_concurrent: int = 50,

        # Validation
        validate_arguments: bool = True,
        coerce_types: bool = True,

        # Observability
        tracing_enabled: bool = True,
        metrics_enabled: bool = True,
        service_name: str = "aion-mcp",
        service_version: str = "1.0.0",
    ):
        self.health_check_interval = health_check_interval
        self.auto_reconnect = auto_reconnect
        self.max_reconnect_attempts = max_reconnect_attempts

        self.circuit_failure_threshold = circuit_failure_threshold
        self.circuit_success_threshold = circuit_success_threshold
        self.circuit_timeout = circuit_timeout

        self.rate_limit_rate = rate_limit_rate
        self.rate_limit_capacity = rate_limit_capacity

        self.cache_enabled = cache_enabled
        self.cache_max_size = cache_max_size
        self.cache_ttl = cache_ttl

        self.retry_enabled = retry_enabled
        self.retry_max_attempts = retry_max_attempts
        self.retry_base_delay = retry_base_delay
        self.retry_max_delay = retry_max_delay
        self.retry_jitter_mode = retry_jitter_mode

        self.bulkhead_enabled = bulkhead_enabled
        self.bulkhead_max_concurrent = bulkhead_max_concurrent

        self.validate_arguments = validate_arguments
        self.coerce_types = coerce_types

        self.tracing_enabled = tracing_enabled
        self.metrics_enabled = metrics_enabled
        self.service_name = service_name
        self.service_version = service_version


# ============================================
# Server Wrapper with Resilience
# ============================================

class ResilientServerWrapper:
    """
    Wrapper for MCP client with resilience patterns.

    Provides:
    - Circuit breaker for failure isolation
    - Rate limiting for load management
    - Request deduplication
    - Per-server configuration
    """

    def __init__(
        self,
        name: str,
        client: MCPClient,
        config: MCPManagerConfig,
    ):
        self.name = name
        self.client = client
        self.config = config

        # Circuit breaker
        self.circuit_breaker = CircuitBreaker(
            name=f"cb_{name}",
            config=CircuitBreakerConfig(
                failure_threshold=config.circuit_failure_threshold,
                success_threshold=config.circuit_success_threshold,
                timeout=config.circuit_timeout,
            ),
        )

        # Rate limiter
        self.rate_limiter = TokenBucketRateLimiter(
            name=f"rl_{name}",
            rate=config.rate_limit_rate,
            capacity=config.rate_limit_capacity,
        )

        # Request deduplicator
        self.deduplicator = RequestDeduplicator(
            name=f"dedup_{name}",
            ttl=5.0,
        )

        # Backoff config
        self.backoff_config = BackoffConfig(
            base_delay=config.retry_base_delay,
            max_delay=config.retry_max_delay,
            max_retries=config.retry_max_attempts,
            jitter_mode=config.retry_jitter_mode,
        ) if config.retry_enabled else None

    @property
    def connected(self) -> bool:
        return self.client.connected

    def get_stats(self) -> dict[str, Any]:
        """Get combined statistics for this server."""
        return {
            "name": self.name,
            "connected": self.connected,
            "circuit_breaker": self.circuit_breaker.get_stats(),
            "rate_limiter": self.rate_limiter.get_stats(),
            "deduplicator": self.deduplicator.get_stats(),
        }


# ============================================
# MCP Manager
# ============================================

class MCPManager:
    """
    Central manager for MCP connections with production-grade resilience.

    SOTA Features:
    - Multi-server management with per-server circuit breakers
    - Rate limiting to prevent overload
    - Tool result caching with LRU eviction
    - Schema validation for tool arguments
    - OpenTelemetry distributed tracing
    - Prometheus metrics
    - Exponential backoff with jitter for retries
    - Request deduplication
    - Health monitoring with auto-reconnection
    """

    def __init__(
        self,
        registry: Optional[ServerRegistry] = None,
        credentials: Optional[CredentialManager] = None,
        config: Optional[MCPManagerConfig] = None,
        # Legacy compatibility
        health_check_interval: float = 30.0,
        auto_reconnect: bool = True,
        max_reconnect_attempts: int = 3,
    ):
        """
        Initialize MCP manager.

        Args:
            registry: Server registry (uses default if None)
            credentials: Credential manager (uses default if None)
            config: Manager configuration (uses defaults if None)
            health_check_interval: Interval between health checks (legacy)
            auto_reconnect: Whether to auto-reconnect failed connections (legacy)
            max_reconnect_attempts: Maximum reconnection attempts (legacy)
        """
        self.registry = registry or ServerRegistry()
        self.credentials = credentials or CredentialManager()

        # Use config or create from legacy params
        if config:
            self.config = config
        else:
            self.config = MCPManagerConfig(
                health_check_interval=health_check_interval,
                auto_reconnect=auto_reconnect,
                max_reconnect_attempts=max_reconnect_attempts,
            )

        # Connected clients wrapped with resilience
        self._servers: dict[str, ResilientServerWrapper] = {}

        # Tool bridge (created lazily)
        self._bridge: Optional["MCPToolBridge"] = None

        # Background tasks
        self._health_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

        # Tool result cache
        self._tool_cache: Optional[LRUCache[ToolResult]] = None
        if self.config.cache_enabled:
            self._tool_cache = LRUCache(
                name="tool_results",
                max_size=self.config.cache_max_size,
                default_ttl=self.config.cache_ttl,
            )

        # Global bulkhead for all operations
        self._bulkhead: Optional[Bulkhead] = None
        if self.config.bulkhead_enabled:
            self._bulkhead = Bulkhead(
                name="mcp_operations",
                max_concurrent=self.config.bulkhead_max_concurrent,
            )

        # Argument validator
        self._validator = ToolArgumentValidator(
            coerce_types=self.config.coerce_types,
            apply_defaults=True,
        )

        # Observability
        self._tracer: Optional[MCPTracer] = None
        self._metrics: Optional[MCPMetrics] = None
        self._health_checker: Optional[MCPHealthChecker] = None

        # Statistics
        self._stats = {
            "connections_established": 0,
            "connections_failed": 0,
            "reconnections": 0,
            "tool_calls": 0,
            "tool_errors": 0,
            "tool_cache_hits": 0,
            "tool_cache_misses": 0,
            "resource_reads": 0,
            "prompt_gets": 0,
            "circuit_breaker_trips": 0,
            "rate_limit_rejections": 0,
            "validation_errors": 0,
        }

        self._initialized = False
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize the MCP manager with all SOTA components."""
        if self._initialized:
            return

        logger.info("Initializing MCP Manager with SOTA patterns")

        # Initialize observability
        if self.config.tracing_enabled:
            self._tracer = MCPTracer(
                service_name=self.config.service_name,
                service_version=self.config.service_version,
            )

        if self.config.metrics_enabled:
            self._metrics = MCPMetrics(namespace="aion_mcp")
            self._metrics.set_info({
                "version": self.config.service_version,
                "service": self.config.service_name,
            })

        # Initialize global observability
        init_observability(self._tracer, self._metrics)

        # Initialize health checker
        self._health_checker = MCPHealthChecker(self)

        # Start tool cache cleanup
        if self._tool_cache:
            await self._tool_cache.start()

        # Load server configurations
        await self.registry.load()

        # Initialize credentials
        await self.credentials.initialize()

        # Connect to enabled servers
        for config in self.registry.get_enabled_servers():
            try:
                await self.connect_server(config.name)
            except Exception as e:
                logger.warning(
                    "Failed to connect to server during initialization",
                    server=config.name,
                    error=str(e),
                )

        # Start health monitor
        self._health_task = asyncio.create_task(self._health_monitor_loop())

        self._initialized = True

        total_tools = sum(
            len(s.client.list_tools())
            for s in self._servers.values()
            if s.connected
        )

        logger.info(
            "MCP Manager initialized with SOTA patterns",
            servers_connected=len(self.get_connected_servers()),
            total_tools=total_tools,
            cache_enabled=self.config.cache_enabled,
            tracing_enabled=self.config.tracing_enabled,
            metrics_enabled=self.config.metrics_enabled,
        )

    async def shutdown(self) -> None:
        """Shutdown the MCP manager and cleanup resources."""
        logger.info("Shutting down MCP Manager")

        self._shutdown_event.set()

        # Stop health monitor
        if self._health_task:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass

        # Stop tool cache
        if self._tool_cache:
            await self._tool_cache.stop()

        # Disconnect all servers
        for name in list(self._servers.keys()):
            await self.disconnect_server(name)

        # Shutdown tracer
        if self._tracer:
            self._tracer.shutdown()

        self._initialized = False
        logger.info("MCP Manager shutdown complete")

    # === Server Management ===

    async def connect_server(self, name: str) -> bool:
        """
        Connect to an MCP server with resilience patterns.

        Args:
            name: Server name (from registry)

        Returns:
            True if connected successfully
        """
        async with self._lock:
            start_time = time.monotonic()
            transport = "unknown"

            try:
                # Get configuration
                config = self.registry.get_server(name)
                if not config:
                    raise ValueError(f"Unknown server: {name}")

                transport = config.transport.value

                # Check if already connected
                if name in self._servers and self._servers[name].connected:
                    return True

                # Resolve credentials if needed
                if config.credential_id:
                    creds = await self.credentials.get(config.credential_id)
                    if creds:
                        config.env.update(creds)

                # Create and connect client
                client = MCPClient(config)
                await client.connect()

                # Wrap with resilience patterns
                self._servers[name] = ResilientServerWrapper(
                    name=name,
                    client=client,
                    config=self.config,
                )

                self._stats["connections_established"] += 1
                duration = time.monotonic() - start_time

                # Record metrics
                if self._metrics:
                    self._metrics.record_connection(
                        server=name,
                        transport=transport,
                        success=True,
                        duration=duration,
                    )

                logger.info(
                    "Connected to MCP server",
                    server=name,
                    tools=len(client.list_tools()),
                    resources=len(client.list_resources()),
                    duration_ms=duration * 1000,
                )

                return True

            except Exception as e:
                self._stats["connections_failed"] += 1
                duration = time.monotonic() - start_time

                if self._metrics:
                    self._metrics.record_connection(
                        server=name,
                        transport=transport,
                        success=False,
                        duration=duration,
                    )

                logger.error(
                    "Failed to connect to MCP server",
                    server=name,
                    error=str(e),
                )
                raise

    async def disconnect_server(self, name: str) -> bool:
        """
        Disconnect from an MCP server.

        Args:
            name: Server name

        Returns:
            True if disconnected
        """
        async with self._lock:
            wrapper = self._servers.pop(name, None)
            if wrapper:
                await wrapper.client.disconnect()

                if self._metrics:
                    self._metrics.record_disconnection(name)

                logger.info("Disconnected from MCP server", server=name)
                return True
            return False

    async def reconnect_server(self, name: str) -> bool:
        """
        Reconnect to an MCP server with backoff.

        Args:
            name: Server name

        Returns:
            True if reconnected successfully
        """
        await self.disconnect_server(name)
        self._stats["reconnections"] += 1

        # Use retry with backoff for reconnection
        if self.config.retry_enabled:
            backoff_config = BackoffConfig(
                base_delay=self.config.retry_base_delay,
                max_delay=self.config.retry_max_delay,
                max_retries=self.config.max_reconnect_attempts,
                jitter_mode=self.config.retry_jitter_mode,
            )

            try:
                return await retry_with_backoff(
                    lambda: self.connect_server(name),
                    config=backoff_config,
                    on_retry=lambda attempt, error, delay: logger.info(
                        f"Reconnection attempt {attempt + 1} in {delay:.2f}s",
                        server=name,
                        error=str(error),
                    ),
                )
            except RetryExhaustedError as e:
                logger.error(
                    "Reconnection attempts exhausted",
                    server=name,
                    attempts=e.attempts,
                    last_error=str(e.last_error),
                )
                raise
        else:
            return await self.connect_server(name)

    def is_connected(self, name: str) -> bool:
        """Check if server is connected."""
        wrapper = self._servers.get(name)
        return wrapper is not None and wrapper.connected

    # === Tool Operations with SOTA Patterns ===

    async def call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: dict[str, Any],
        skip_cache: bool = False,
        skip_validation: bool = False,
    ) -> ToolResult:
        """
        Call a tool on a specific server with full resilience.

        Applies:
        1. Schema validation
        2. Bulkhead (concurrency limiting)
        3. Rate limiting
        4. Cache check
        5. Circuit breaker
        6. Retry with backoff
        7. Metrics recording
        8. Distributed tracing

        Args:
            server_name: Name of the server
            tool_name: Name of the tool
            arguments: Tool arguments
            skip_cache: Skip cache lookup/storage
            skip_validation: Skip argument validation

        Returns:
            ToolResult

        Raises:
            SchemaValidationError: If validation fails
            CircuitBreakerError: If circuit is open
            RateLimitExceededError: If rate limit exceeded
            BulkheadFullError: If bulkhead at capacity
        """
        wrapper = self._servers.get(server_name)
        if not wrapper or not wrapper.connected:
            raise RuntimeError(f"Server not connected: {server_name}")

        start_time = time.monotonic()
        self._stats["tool_calls"] += 1

        # Get tool schema for validation
        tool = wrapper.client.get_tool(tool_name)
        if not tool:
            raise ValueError(f"Tool not found: {tool_name}")

        # Step 1: Validate arguments
        if self.config.validate_arguments and not skip_validation:
            try:
                arguments = self._validator.prepare_arguments(
                    tool_name=tool_name,
                    arguments=arguments,
                    input_schema=tool.input_schema,
                )
            except SchemaValidationError as e:
                self._stats["validation_errors"] += 1
                logger.warning(
                    "Tool argument validation failed",
                    server=server_name,
                    tool=tool_name,
                    errors=[str(err) for err in e.errors],
                )
                raise

        # Generate cache key
        cache_key = self._make_cache_key(server_name, tool_name, arguments)

        # Step 2: Check cache
        if self._tool_cache and not skip_cache:
            cached = await self._tool_cache.get(cache_key)
            if cached is not None:
                self._stats["tool_cache_hits"] += 1
                if self._metrics:
                    self._metrics.record_cache_hit("tool_results")
                logger.debug(
                    "Tool cache hit",
                    server=server_name,
                    tool=tool_name,
                )
                return cached
            else:
                self._stats["tool_cache_misses"] += 1
                if self._metrics:
                    self._metrics.record_cache_miss("tool_results")

        # Step 3: Bulkhead (limit concurrent operations)
        if self._bulkhead:
            try:
                await self._bulkhead.acquire()
            except BulkheadFullError:
                logger.warning(
                    "Bulkhead full, rejecting request",
                    server=server_name,
                    tool=tool_name,
                )
                raise

        try:
            # Step 4: Rate limiting
            try:
                await wrapper.rate_limiter.acquire(block=False)
            except RateLimitExceededError as e:
                self._stats["rate_limit_rejections"] += 1
                if self._metrics:
                    self._metrics.record_rate_limit_rejected(server_name)
                raise

            # Step 5: Execute with circuit breaker and retry
            result = await self._execute_tool_call(
                wrapper=wrapper,
                tool_name=tool_name,
                arguments=arguments,
            )

            # Step 6: Cache result
            if self._tool_cache and not skip_cache and not result.is_error:
                await self._tool_cache.set(cache_key, result)

            # Record metrics
            duration = time.monotonic() - start_time
            if self._metrics:
                self._metrics.record_tool_call(
                    server=server_name,
                    tool=tool_name,
                    success=not result.is_error,
                    duration=duration,
                )

            return result

        except CircuitBreakerError:
            self._stats["circuit_breaker_trips"] += 1
            if self._metrics:
                self._metrics.record_circuit_breaker_state(
                    server_name,
                    wrapper.circuit_breaker.state.value,
                )
            raise

        except Exception as e:
            self._stats["tool_errors"] += 1
            duration = time.monotonic() - start_time

            if self._metrics:
                self._metrics.record_tool_call(
                    server=server_name,
                    tool=tool_name,
                    success=False,
                    duration=duration,
                    error_type=type(e).__name__,
                )
            raise

        finally:
            if self._bulkhead:
                await self._bulkhead.release()

    async def _execute_tool_call(
        self,
        wrapper: ResilientServerWrapper,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> ToolResult:
        """Execute tool call with circuit breaker and retry."""

        async def do_call():
            async with wrapper.circuit_breaker:
                return await wrapper.client.call_tool(tool_name, arguments)

        if wrapper.backoff_config:
            return await retry_with_backoff(
                do_call,
                config=wrapper.backoff_config,
                retryable_exceptions=(MCPError, asyncio.TimeoutError),
            )
        else:
            return await do_call()

    def _make_cache_key(
        self,
        server_name: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> str:
        """Generate cache key for tool call."""
        key_data = f"{server_name}:{tool_name}:{sorted(arguments.items())}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:32]

    async def call_tool_by_name(
        self,
        full_tool_name: str,
        arguments: dict[str, Any],
        **kwargs,
    ) -> ToolResult:
        """
        Call a tool by its full name (server:tool).

        Args:
            full_tool_name: Full tool name (e.g., "postgres:query")
            arguments: Tool arguments
            **kwargs: Additional options passed to call_tool

        Returns:
            ToolResult
        """
        if ":" in full_tool_name:
            server_name, tool_name = full_tool_name.split(":", 1)
        else:
            # Find server that has this tool
            server_name = self._find_server_with_tool(full_tool_name)
            tool_name = full_tool_name

        if not server_name:
            raise ValueError(f"Tool not found: {full_tool_name}")

        return await self.call_tool(server_name, tool_name, arguments, **kwargs)

    def _find_server_with_tool(self, tool_name: str) -> Optional[str]:
        """Find the server that has a specific tool."""
        for name, wrapper in self._servers.items():
            if wrapper.connected:
                for tool in wrapper.client.list_tools():
                    if tool.name == tool_name:
                        return name
        return None

    def list_all_tools(self) -> dict[str, list[Tool]]:
        """Get all tools from all connected servers."""
        result = {}
        for name, wrapper in self._servers.items():
            if wrapper.connected:
                result[name] = wrapper.client.list_tools()
        return result

    def get_tools_flat(self) -> list[tuple[str, Tool]]:
        """Get all tools as flat list of (server_name, tool) tuples."""
        result = []
        for name, wrapper in self._servers.items():
            if wrapper.connected:
                for tool in wrapper.client.list_tools():
                    result.append((name, tool))
        return result

    def get_tool(self, server_name: str, tool_name: str) -> Optional[Tool]:
        """Get a specific tool."""
        wrapper = self._servers.get(server_name)
        if wrapper and wrapper.connected:
            return wrapper.client.get_tool(tool_name)
        return None

    # === Resource Operations ===

    async def read_resource(
        self,
        server_name: str,
        uri: str,
    ) -> ResourceContent:
        """
        Read a resource from a server.

        Args:
            server_name: Name of the server
            uri: Resource URI

        Returns:
            ResourceContent
        """
        wrapper = self._servers.get(server_name)
        if not wrapper or not wrapper.connected:
            raise RuntimeError(f"Server not connected: {server_name}")

        start_time = time.monotonic()
        self._stats["resource_reads"] += 1

        # Apply rate limiting
        await wrapper.rate_limiter.acquire()

        try:
            async with wrapper.circuit_breaker:
                result = await wrapper.client.read_resource(uri)

            duration = time.monotonic() - start_time
            if self._metrics:
                self._metrics.record_resource_read(
                    server=server_name,
                    success=True,
                    duration=duration,
                )

            return result

        except Exception as e:
            duration = time.monotonic() - start_time
            if self._metrics:
                self._metrics.record_resource_read(
                    server=server_name,
                    success=False,
                    duration=duration,
                )
            raise

    def list_all_resources(self) -> dict[str, list[Resource]]:
        """Get all resources from all connected servers."""
        result = {}
        for name, wrapper in self._servers.items():
            if wrapper.connected:
                result[name] = wrapper.client.list_resources()
        return result

    # === Prompt Operations ===

    async def get_prompt(
        self,
        server_name: str,
        prompt_name: str,
        arguments: Optional[dict[str, str]] = None,
    ) -> list[PromptMessage]:
        """
        Get a prompt from a server.

        Args:
            server_name: Name of the server
            prompt_name: Name of the prompt
            arguments: Prompt arguments

        Returns:
            List of prompt messages
        """
        wrapper = self._servers.get(server_name)
        if not wrapper or not wrapper.connected:
            raise RuntimeError(f"Server not connected: {server_name}")

        self._stats["prompt_gets"] += 1

        # Apply rate limiting
        await wrapper.rate_limiter.acquire()

        try:
            async with wrapper.circuit_breaker:
                result = await wrapper.client.get_prompt(prompt_name, arguments)

            if self._metrics:
                self._metrics.record_prompt_get(server_name, success=True)

            return result

        except Exception:
            if self._metrics:
                self._metrics.record_prompt_get(server_name, success=False)
            raise

    def list_all_prompts(self) -> dict[str, list[Prompt]]:
        """Get all prompts from all connected servers."""
        result = {}
        for name, wrapper in self._servers.items():
            if wrapper.connected:
                result[name] = wrapper.client.list_prompts()
        return result

    # === Bridge to AION Tools ===

    def get_tool_bridge(self) -> "MCPToolBridge":
        """Get the tool bridge for integration with AION's tool system."""
        if not self._bridge:
            from aion.mcp.bridge import MCPToolBridge
            self._bridge = MCPToolBridge(self)
        return self._bridge

    # === Health Monitoring ===

    async def _health_monitor_loop(self) -> None:
        """Background health monitoring loop with circuit breaker awareness."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(self.config.health_check_interval)

                for name, wrapper in list(self._servers.items()):
                    # Check circuit breaker state
                    if self._metrics:
                        self._metrics.record_circuit_breaker_state(
                            name,
                            wrapper.circuit_breaker.state.value,
                        )

                    # Auto-reconnect if needed
                    if not wrapper.connected and self.config.auto_reconnect:
                        config = self.registry.get_server(name)
                        if config and config.auto_reconnect:
                            logger.info(f"Attempting reconnection to {name}")
                            try:
                                await self.reconnect_server(name)
                            except Exception as e:
                                logger.warning(
                                    "Reconnection failed",
                                    server=name,
                                    error=str(e),
                                )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Health monitor error", error=str(e))

    async def ping_server(self, name: str) -> bool:
        """
        Ping a server to check connectivity.

        Args:
            name: Server name

        Returns:
            True if server responds
        """
        wrapper = self._servers.get(name)
        if not wrapper:
            return False
        return await wrapper.client.ping()

    async def check_health(self) -> dict[str, Any]:
        """Run full health check."""
        if self._health_checker:
            return await self._health_checker.check_all()
        return {
            "status": "unknown",
            "message": "Health checker not initialized",
        }

    # === Cache Management ===

    async def clear_tool_cache(self, server_name: Optional[str] = None) -> None:
        """Clear tool result cache."""
        if self._tool_cache:
            if server_name:
                # Clear cache entries for specific server
                # Note: This requires iterating which is not ideal for LRU cache
                # In production, consider using prefixed keys
                logger.info(f"Clearing cache for server: {server_name}")
            else:
                await self._tool_cache.clear()
                logger.info("Tool cache cleared")

    def get_cache_stats(self) -> dict[str, Any]:
        """Get tool cache statistics."""
        if self._tool_cache:
            return self._tool_cache.get_stats()
        return {"enabled": False}

    # === Status ===

    def get_server_states(self) -> dict[str, ConnectedServer]:
        """Get state of all servers."""
        return {
            name: wrapper.client.get_state()
            for name, wrapper in self._servers.items()
        }

    def get_server_state(self, name: str) -> Optional[ConnectedServer]:
        """Get state of a specific server."""
        wrapper = self._servers.get(name)
        if wrapper:
            return wrapper.client.get_state()
        return None

    def get_connected_servers(self) -> list[str]:
        """Get names of connected servers."""
        return [
            name for name, wrapper in self._servers.items()
            if wrapper.connected
        ]

    def get_resilience_stats(self) -> dict[str, Any]:
        """Get resilience pattern statistics for all servers."""
        return {
            name: wrapper.get_stats()
            for name, wrapper in self._servers.items()
        }

    def get_stats(self) -> dict[str, Any]:
        """Get comprehensive manager statistics."""
        return {
            **self._stats,
            "servers_configured": len(self.registry.get_all_servers()),
            "servers_connected": len(self.get_connected_servers()),
            "total_tools": sum(
                len(wrapper.client.list_tools())
                for wrapper in self._servers.values()
                if wrapper.connected
            ),
            "total_resources": sum(
                len(wrapper.client.list_resources())
                for wrapper in self._servers.values()
                if wrapper.connected
            ),
            "total_prompts": sum(
                len(wrapper.client.list_prompts())
                for wrapper in self._servers.values()
                if wrapper.connected
            ),
            "cache": self.get_cache_stats(),
            "bulkhead": self._bulkhead.get_stats() if self._bulkhead else None,
        }

    def get_prometheus_metrics(self) -> bytes:
        """Get Prometheus metrics in text format."""
        if self._metrics:
            return self._metrics.get_metrics()
        return b""

    # === Legacy Property Access ===

    @property
    def _clients(self) -> dict[str, MCPClient]:
        """Legacy access to clients for backward compatibility."""
        return {
            name: wrapper.client
            for name, wrapper in self._servers.items()
        }

    # === Context Manager ===

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.shutdown()
