"""
AION MCP WebSocket Transport

Communicates with MCP servers via WebSocket.
Provides bidirectional real-time communication.
"""

from __future__ import annotations

import asyncio
from typing import AsyncGenerator, Optional

import structlog

try:
    import websockets
    from websockets.client import WebSocketClientProtocol
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False

from aion.mcp.transports.base import (
    Transport,
    ConnectionError,
    SendError,
    ReceiveError,
)

logger = structlog.get_logger(__name__)


class WebSocketTransport(Transport):
    """
    WebSocket transport for MCP.

    Provides bidirectional communication with MCP servers
    over WebSocket connections.
    """

    def __init__(
        self,
        url: str,
        headers: Optional[dict[str, str]] = None,
        ping_interval: float = 20.0,
        ping_timeout: float = 10.0,
        close_timeout: float = 5.0,
        max_message_size: int = 10 * 1024 * 1024,  # 10MB
    ):
        """
        Initialize WebSocket transport.

        Args:
            url: WebSocket URL (ws:// or wss://)
            headers: HTTP headers for the upgrade request
            ping_interval: Interval between ping frames
            ping_timeout: Timeout for ping response
            close_timeout: Timeout for close handshake
            max_message_size: Maximum message size in bytes
        """
        if not WEBSOCKETS_AVAILABLE:
            raise ImportError("websockets is required for WebSocket transport. Install with: pip install websockets")

        # Ensure URL uses WebSocket protocol
        if url.startswith("http://"):
            url = "ws://" + url[7:]
        elif url.startswith("https://"):
            url = "wss://" + url[8:]
        elif not url.startswith(("ws://", "wss://")):
            url = "wss://" + url

        self.url = url
        self.headers = headers or {}
        self.ping_interval = ping_interval
        self.ping_timeout = ping_timeout
        self.close_timeout = close_timeout
        self.max_message_size = max_message_size

        self._websocket: Optional[WebSocketClientProtocol] = None
        self._connected = False
        self._receive_task: Optional[asyncio.Task] = None
        self._message_queue: asyncio.Queue[str] = asyncio.Queue()

    async def connect(self) -> None:
        """Establish WebSocket connection."""
        if self._connected:
            return

        logger.info("Connecting to MCP server via WebSocket", url=self.url)

        try:
            self._websocket = await websockets.connect(
                self.url,
                additional_headers=self.headers,
                ping_interval=self.ping_interval,
                ping_timeout=self.ping_timeout,
                close_timeout=self.close_timeout,
                max_size=self.max_message_size,
            )

            self._connected = True

            # Start receive loop
            self._receive_task = asyncio.create_task(self._receive_loop())

            logger.debug("Connected to MCP server via WebSocket", url=self.url)

        except Exception as e:
            raise ConnectionError(f"Failed to connect: {e}", cause=e)

    async def close(self) -> None:
        """Close the WebSocket connection."""
        if not self._connected:
            return

        logger.debug("Closing WebSocket connection", url=self.url)

        # Cancel receive task
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass

        # Close WebSocket
        if self._websocket:
            try:
                await self._websocket.close()
            except Exception as e:
                logger.warning("Error closing WebSocket", error=str(e))
            self._websocket = None

        self._connected = False

        logger.debug("WebSocket connection closed", url=self.url)

    async def send(self, message: str) -> None:
        """Send a message through the WebSocket."""
        if not self._connected or not self._websocket:
            raise SendError("Transport not connected")

        try:
            await self._websocket.send(message)

            logger.debug(
                "Sent message via WebSocket",
                message_length=len(message),
            )

        except websockets.exceptions.ConnectionClosed as e:
            self._connected = False
            raise SendError(f"Connection closed: {e}", cause=e)
        except Exception as e:
            raise SendError(f"Failed to send message: {e}", cause=e)

    async def receive(self) -> AsyncGenerator[str, None]:
        """Receive messages from the WebSocket."""
        if not self._connected:
            raise ReceiveError("Transport not connected")

        while self._connected:
            try:
                # Get message from queue with timeout
                message = await asyncio.wait_for(
                    self._message_queue.get(),
                    timeout=1.0,
                )
                yield message

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error receiving message", error=str(e))
                raise ReceiveError(f"Failed to receive message: {e}", cause=e)

    async def _receive_loop(self) -> None:
        """Background loop to receive WebSocket messages."""
        if not self._websocket:
            return

        try:
            async for message in self._websocket:
                if isinstance(message, bytes):
                    message = message.decode("utf-8")

                logger.debug(
                    "Received message via WebSocket",
                    message_length=len(message),
                )

                await self._message_queue.put(message)

        except websockets.exceptions.ConnectionClosed:
            logger.debug("WebSocket connection closed by server")
            self._connected = False
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("WebSocket receive error", error=str(e))
            self._connected = False

    @property
    def connected(self) -> bool:
        """Check if transport is connected."""
        if not self._connected or not self._websocket:
            return False

        # Check if WebSocket is still open
        if self._websocket.closed:
            self._connected = False
            return False

        return True

    @property
    def local_address(self) -> Optional[tuple]:
        """Get the local address."""
        if self._websocket:
            return self._websocket.local_address
        return None

    @property
    def remote_address(self) -> Optional[tuple]:
        """Get the remote address."""
        if self._websocket:
            return self._websocket.remote_address
        return None
