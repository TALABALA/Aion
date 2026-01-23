"""
AION MCP Client Session Management

Manages multiple MCP client sessions with connection pooling and lifecycle management.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

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
)
from aion.mcp.client.client import MCPClient, MCPError

logger = structlog.get_logger(__name__)


class SessionState:
    """State of a client session."""

    def __init__(self, client: MCPClient):
        self.client = client
        self.created_at = datetime.now()
        self.last_used = datetime.now()
        self.use_count = 0

    def mark_used(self) -> None:
        """Mark session as used."""
        self.last_used = datetime.now()
        self.use_count += 1

    @property
    def idle_time(self) -> timedelta:
        """Get time since last use."""
        return datetime.now() - self.last_used


class MCPSessionManager:
    """
    Manages multiple MCP client sessions.

    Provides:
    - Connection pooling
    - Automatic reconnection
    - Session lifecycle management
    - Idle session cleanup
    """

    def __init__(
        self,
        max_sessions: int = 100,
        idle_timeout: float = 300.0,  # 5 minutes
        cleanup_interval: float = 60.0,  # 1 minute
        auto_reconnect: bool = True,
        max_reconnect_attempts: int = 3,
        reconnect_delay: float = 1.0,
    ):
        """
        Initialize session manager.

        Args:
            max_sessions: Maximum number of concurrent sessions
            idle_timeout: Time before idle sessions are closed
            cleanup_interval: Interval for cleanup checks
            auto_reconnect: Whether to auto-reconnect failed sessions
            max_reconnect_attempts: Maximum reconnection attempts
            reconnect_delay: Delay between reconnection attempts
        """
        self.max_sessions = max_sessions
        self.idle_timeout = idle_timeout
        self.cleanup_interval = cleanup_interval
        self.auto_reconnect = auto_reconnect
        self.max_reconnect_attempts = max_reconnect_attempts
        self.reconnect_delay = reconnect_delay

        self._sessions: Dict[str, SessionState] = {}
        self._configs: Dict[str, ServerConfig] = {}
        self._cleanup_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        self._shutdown_event = asyncio.Event()

    async def start(self) -> None:
        """Start the session manager."""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("Session manager started")

    async def stop(self) -> None:
        """Stop the session manager and close all sessions."""
        self._shutdown_event.set()

        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        # Close all sessions
        async with self._lock:
            for name in list(self._sessions.keys()):
                await self._close_session(name)

        logger.info("Session manager stopped")

    async def get_session(self, name: str) -> MCPClient:
        """
        Get a client session, creating one if necessary.

        Args:
            name: Server name

        Returns:
            MCPClient for the server
        """
        async with self._lock:
            # Check if session exists and is connected
            if name in self._sessions:
                session = self._sessions[name]
                if session.client.connected:
                    session.mark_used()
                    return session.client
                else:
                    # Session disconnected, try to reconnect
                    if self.auto_reconnect:
                        client = await self._reconnect(name)
                        if client:
                            return client

            # Create new session
            if name not in self._configs:
                raise ValueError(f"Unknown server: {name}")

            return await self._create_session(name)

    async def add_server(self, config: ServerConfig) -> None:
        """
        Add a server configuration.

        Args:
            config: Server configuration
        """
        async with self._lock:
            self._configs[config.name] = config

    async def remove_server(self, name: str) -> None:
        """
        Remove a server and close its session.

        Args:
            name: Server name
        """
        async with self._lock:
            if name in self._sessions:
                await self._close_session(name)
            self._configs.pop(name, None)

    async def connect(self, name: str) -> MCPClient:
        """
        Explicitly connect to a server.

        Args:
            name: Server name

        Returns:
            Connected MCPClient
        """
        return await self.get_session(name)

    async def disconnect(self, name: str) -> None:
        """
        Disconnect from a server.

        Args:
            name: Server name
        """
        async with self._lock:
            await self._close_session(name)

    async def reconnect(self, name: str) -> MCPClient:
        """
        Reconnect to a server.

        Args:
            name: Server name

        Returns:
            Reconnected MCPClient
        """
        async with self._lock:
            await self._close_session(name)
            return await self._create_session(name)

    def is_connected(self, name: str) -> bool:
        """Check if server is connected."""
        session = self._sessions.get(name)
        return session is not None and session.client.connected

    def get_stats(self, name: str) -> Optional[ConnectedServer]:
        """Get session statistics for a server."""
        session = self._sessions.get(name)
        if session:
            return session.client.get_state()
        return None

    def list_sessions(self) -> list[str]:
        """List all active session names."""
        return [
            name for name, session in self._sessions.items()
            if session.client.connected
        ]

    async def _create_session(self, name: str) -> MCPClient:
        """Create a new session."""
        if len(self._sessions) >= self.max_sessions:
            # Try to clean up idle sessions
            await self._cleanup_idle_sessions()

            if len(self._sessions) >= self.max_sessions:
                raise RuntimeError("Maximum sessions reached")

        config = self._configs.get(name)
        if not config:
            raise ValueError(f"Unknown server: {name}")

        client = MCPClient(config)
        await client.connect()

        self._sessions[name] = SessionState(client)
        logger.info("Created session", server=name)

        return client

    async def _close_session(self, name: str) -> None:
        """Close a session."""
        session = self._sessions.pop(name, None)
        if session:
            await session.client.disconnect()
            logger.info("Closed session", server=name)

    async def _reconnect(self, name: str) -> Optional[MCPClient]:
        """Try to reconnect a session."""
        config = self._configs.get(name)
        if not config:
            return None

        for attempt in range(self.max_reconnect_attempts):
            try:
                logger.info(
                    "Attempting reconnection",
                    server=name,
                    attempt=attempt + 1,
                )

                client = MCPClient(config)
                await client.connect()

                self._sessions[name] = SessionState(client)
                logger.info("Reconnection successful", server=name)

                return client

            except Exception as e:
                logger.warning(
                    "Reconnection failed",
                    server=name,
                    attempt=attempt + 1,
                    error=str(e),
                )

                if attempt < self.max_reconnect_attempts - 1:
                    await asyncio.sleep(self.reconnect_delay * (attempt + 1))

        return None

    async def _cleanup_idle_sessions(self) -> None:
        """Clean up idle sessions."""
        idle_threshold = timedelta(seconds=self.idle_timeout)

        for name, session in list(self._sessions.items()):
            if session.idle_time > idle_threshold:
                logger.info(
                    "Closing idle session",
                    server=name,
                    idle_seconds=session.idle_time.total_seconds(),
                )
                await self._close_session(name)

    async def _cleanup_loop(self) -> None:
        """Background cleanup loop."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(self.cleanup_interval)
                async with self._lock:
                    await self._cleanup_idle_sessions()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Cleanup loop error", error=str(e))


class MCPClientPool:
    """
    Connection pool for MCP clients.

    Provides a simpler interface for managing multiple connections
    with automatic pooling.
    """

    def __init__(
        self,
        configs: list[ServerConfig],
        max_connections_per_server: int = 3,
    ):
        """
        Initialize client pool.

        Args:
            configs: List of server configurations
            max_connections_per_server: Maximum connections per server
        """
        self._configs = {c.name: c for c in configs}
        self._max_per_server = max_connections_per_server
        self._pools: Dict[str, list[MCPClient]] = {}
        self._available: Dict[str, asyncio.Queue] = {}
        self._lock = asyncio.Lock()

    async def acquire(self, name: str) -> MCPClient:
        """
        Acquire a client from the pool.

        Args:
            name: Server name

        Returns:
            MCPClient instance
        """
        async with self._lock:
            if name not in self._available:
                self._available[name] = asyncio.Queue()
                self._pools[name] = []

        # Try to get from available pool
        try:
            client = self._available[name].get_nowait()
            if client.connected:
                return client
        except asyncio.QueueEmpty:
            pass

        # Create new connection if under limit
        async with self._lock:
            if len(self._pools[name]) < self._max_per_server:
                config = self._configs.get(name)
                if not config:
                    raise ValueError(f"Unknown server: {name}")

                client = MCPClient(config)
                await client.connect()
                self._pools[name].append(client)
                return client

        # Wait for available connection
        client = await self._available[name].get()
        if not client.connected:
            await client.connect()
        return client

    async def release(self, name: str, client: MCPClient) -> None:
        """
        Release a client back to the pool.

        Args:
            name: Server name
            client: Client to release
        """
        if name in self._available:
            await self._available[name].put(client)

    async def close_all(self) -> None:
        """Close all connections in the pool."""
        async with self._lock:
            for name, clients in self._pools.items():
                for client in clients:
                    await client.disconnect()
            self._pools.clear()
            self._available.clear()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close_all()
