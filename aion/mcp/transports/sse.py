"""
AION MCP SSE Transport

Communicates with MCP servers via HTTP/SSE (Server-Sent Events).
Used for remote MCP servers that expose an HTTP endpoint.
"""

from __future__ import annotations

import asyncio
import json
from typing import AsyncGenerator, Optional
from urllib.parse import urljoin

import structlog

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

from aion.mcp.transports.base import (
    Transport,
    ConnectionError,
    SendError,
    ReceiveError,
)

logger = structlog.get_logger(__name__)


class SSETransport(Transport):
    """
    HTTP/SSE transport for MCP.

    Uses Server-Sent Events for receiving messages and HTTP POST
    for sending messages to remote MCP servers.
    """

    def __init__(
        self,
        url: str,
        headers: Optional[dict[str, str]] = None,
        timeout: float = 30.0,
        sse_endpoint: str = "/sse",
        message_endpoint: str = "/message",
    ):
        """
        Initialize SSE transport.

        Args:
            url: Base URL of the MCP server
            headers: HTTP headers to include in requests
            timeout: Request timeout in seconds
            sse_endpoint: Endpoint for SSE connection
            message_endpoint: Endpoint for sending messages
        """
        if not HTTPX_AVAILABLE:
            raise ImportError("httpx is required for SSE transport. Install with: pip install httpx")

        self.url = url.rstrip("/")
        self.headers = headers or {}
        self.timeout = timeout
        self.sse_endpoint = sse_endpoint
        self.message_endpoint = message_endpoint

        self._client: Optional[httpx.AsyncClient] = None
        self._sse_response: Optional[httpx.Response] = None
        self._connected = False
        self._session_id: Optional[str] = None
        self._message_queue: asyncio.Queue[str] = asyncio.Queue()
        self._sse_task: Optional[asyncio.Task] = None

    async def connect(self) -> None:
        """Establish connection to the SSE endpoint."""
        if self._connected:
            return

        logger.info("Connecting to MCP server via SSE", url=self.url)

        try:
            # Create HTTP client
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout, connect=10.0),
                headers=self.headers,
            )

            # Start SSE connection
            self._sse_task = asyncio.create_task(self._sse_loop())

            # Wait a bit for connection to establish
            await asyncio.sleep(0.1)

            self._connected = True

            logger.debug("Connected to MCP server via SSE", url=self.url)

        except Exception as e:
            if self._client:
                await self._client.aclose()
                self._client = None
            raise ConnectionError(f"Failed to connect: {e}", cause=e)

    async def close(self) -> None:
        """Close the SSE connection."""
        if not self._connected:
            return

        logger.debug("Closing SSE connection", url=self.url)

        # Cancel SSE task
        if self._sse_task:
            self._sse_task.cancel()
            try:
                await self._sse_task
            except asyncio.CancelledError:
                pass

        # Close HTTP client
        if self._client:
            await self._client.aclose()
            self._client = None

        self._connected = False

        logger.debug("SSE connection closed", url=self.url)

    async def send(self, message: str) -> None:
        """Send a message to the server via HTTP POST."""
        if not self._connected or not self._client:
            raise SendError("Transport not connected")

        message_url = urljoin(self.url + "/", self.message_endpoint.lstrip("/"))

        try:
            response = await self._client.post(
                message_url,
                content=message,
                headers={"Content-Type": "application/json"},
            )

            response.raise_for_status()

            logger.debug(
                "Sent message to MCP server",
                url=message_url,
                status=response.status_code,
            )

        except httpx.HTTPStatusError as e:
            raise SendError(f"HTTP error {e.response.status_code}: {e.response.text}", cause=e)
        except httpx.RequestError as e:
            raise SendError(f"Request failed: {e}", cause=e)
        except Exception as e:
            raise SendError(f"Failed to send message: {e}", cause=e)

    async def receive(self) -> AsyncGenerator[str, None]:
        """Receive messages from the SSE stream."""
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

    async def _sse_loop(self) -> None:
        """Background task to read SSE stream."""
        if not self._client:
            return

        sse_url = urljoin(self.url + "/", self.sse_endpoint.lstrip("/"))

        try:
            async with self._client.stream(
                "GET",
                sse_url,
                headers={"Accept": "text/event-stream"},
            ) as response:
                response.raise_for_status()

                logger.debug("SSE stream connected", url=sse_url)

                buffer = ""
                event_type = "message"
                event_data = []

                async for chunk in response.aiter_text():
                    buffer += chunk

                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        line = line.strip()

                        if not line:
                            # Empty line = end of event
                            if event_data:
                                data = "\n".join(event_data)
                                await self._handle_sse_event(event_type, data)
                                event_data = []
                                event_type = "message"
                            continue

                        if line.startswith("event:"):
                            event_type = line[6:].strip()
                        elif line.startswith("data:"):
                            event_data.append(line[5:].strip())
                        elif line.startswith("id:"):
                            self._session_id = line[3:].strip()
                        elif line.startswith("retry:"):
                            pass  # Ignore retry directive

        except asyncio.CancelledError:
            pass
        except httpx.HTTPStatusError as e:
            logger.error("SSE HTTP error", status=e.response.status_code)
            self._connected = False
        except Exception as e:
            logger.error("SSE stream error", error=str(e))
            self._connected = False

    async def _handle_sse_event(self, event_type: str, data: str) -> None:
        """Handle an SSE event."""
        if event_type == "message":
            # Queue the message for receive()
            await self._message_queue.put(data)

            logger.debug(
                "Received SSE message",
                event_type=event_type,
                data_length=len(data),
            )
        elif event_type == "endpoint":
            # Server is telling us the message endpoint
            logger.debug("Received endpoint event", endpoint=data)
        elif event_type == "error":
            logger.warning("SSE server error", error=data)
        else:
            logger.debug(
                "Received unknown SSE event",
                event_type=event_type,
                data=data[:100],
            )

    @property
    def connected(self) -> bool:
        """Check if transport is connected."""
        return self._connected

    @property
    def session_id(self) -> Optional[str]:
        """Get the SSE session ID."""
        return self._session_id
