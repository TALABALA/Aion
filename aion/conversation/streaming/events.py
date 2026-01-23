"""
AION Server-Sent Events

Utilities for server-sent events (SSE) in streaming responses.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, AsyncIterator, Optional
import structlog

from aion.conversation.types import StreamEvent

logger = structlog.get_logger(__name__)


def format_sse(event: StreamEvent, event_name: Optional[str] = None) -> str:
    """
    Format an event for Server-Sent Events protocol.

    Args:
        event: The stream event
        event_name: Optional event name for SSE

    Returns:
        SSE-formatted string
    """
    lines = []

    if event_name:
        lines.append(f"event: {event_name}")

    data = json.dumps(event.to_dict())
    lines.append(f"data: {data}")

    lines.append("")

    return "\n".join(lines) + "\n"


def format_sse_data(data: Any) -> str:
    """
    Format raw data for SSE.

    Args:
        data: Data to format

    Returns:
        SSE-formatted string
    """
    if isinstance(data, str):
        json_data = json.dumps({"text": data})
    elif isinstance(data, dict):
        json_data = json.dumps(data)
    else:
        json_data = json.dumps({"data": str(data)})

    return f"data: {json_data}\n\n"


def format_sse_comment(comment: str) -> str:
    """Format a comment for SSE (used for keep-alive)."""
    return f": {comment}\n\n"


async def sse_generator(
    stream: AsyncIterator[StreamEvent],
    include_event_names: bool = True,
    keepalive_interval: float = 15.0,
) -> AsyncIterator[str]:
    """
    Generate SSE-formatted strings from a stream.

    Args:
        stream: Stream of events
        include_event_names: Whether to include event type as SSE event name
        keepalive_interval: Interval for keepalive comments

    Yields:
        SSE-formatted strings
    """
    last_event_time = datetime.now()

    async def keepalive():
        """Send keepalive comments periodically."""
        nonlocal last_event_time
        while True:
            await asyncio.sleep(keepalive_interval)
            if (datetime.now() - last_event_time).total_seconds() >= keepalive_interval:
                yield format_sse_comment("keepalive")

    try:
        async for event in stream:
            last_event_time = datetime.now()

            event_name = event.type if include_event_names else None
            yield format_sse(event, event_name)

            if event.type == "done":
                break

    except asyncio.CancelledError:
        logger.debug("SSE stream cancelled")
        raise

    except Exception as e:
        logger.error(f"SSE stream error: {e}")
        error_event = StreamEvent.error(str(e))
        yield format_sse(error_event, "error")


@dataclass
class SSEConnection:
    """Represents an SSE connection."""
    id: str
    created_at: datetime
    last_event_at: datetime
    event_count: int = 0
    is_active: bool = True

    def record_event(self) -> None:
        """Record that an event was sent."""
        self.last_event_at = datetime.now()
        self.event_count += 1


class SSEManager:
    """
    Manages multiple SSE connections.

    Useful for broadcasting events to multiple clients.
    """

    def __init__(self, max_connections: int = 1000):
        self.max_connections = max_connections
        self._connections: dict[str, SSEConnection] = {}
        self._queues: dict[str, asyncio.Queue] = {}

    def register(self, connection_id: str) -> asyncio.Queue:
        """
        Register a new SSE connection.

        Returns:
            Queue for receiving events
        """
        if len(self._connections) >= self.max_connections:
            raise RuntimeError("Maximum SSE connections reached")

        now = datetime.now()
        self._connections[connection_id] = SSEConnection(
            id=connection_id,
            created_at=now,
            last_event_at=now,
        )
        self._queues[connection_id] = asyncio.Queue()

        logger.debug(f"SSE connection registered: {connection_id}")
        return self._queues[connection_id]

    def unregister(self, connection_id: str) -> None:
        """Unregister an SSE connection."""
        if connection_id in self._connections:
            self._connections[connection_id].is_active = False
            del self._connections[connection_id]

        self._queues.pop(connection_id, None)
        logger.debug(f"SSE connection unregistered: {connection_id}")

    async def send(self, connection_id: str, event: StreamEvent) -> bool:
        """
        Send an event to a specific connection.

        Returns:
            True if sent successfully
        """
        if connection_id not in self._queues:
            return False

        try:
            self._queues[connection_id].put_nowait(event)
            self._connections[connection_id].record_event()
            return True
        except asyncio.QueueFull:
            logger.warning(f"SSE queue full for connection: {connection_id}")
            return False

    async def broadcast(self, event: StreamEvent) -> int:
        """
        Broadcast an event to all connections.

        Returns:
            Number of connections that received the event
        """
        sent_count = 0

        for connection_id in list(self._connections.keys()):
            if await self.send(connection_id, event):
                sent_count += 1

        return sent_count

    async def stream_for_connection(
        self,
        connection_id: str,
        timeout: float = 30.0,
    ) -> AsyncIterator[StreamEvent]:
        """
        Stream events for a specific connection.

        Yields events from the connection's queue.
        """
        if connection_id not in self._queues:
            raise ValueError(f"Unknown connection: {connection_id}")

        queue = self._queues[connection_id]

        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=timeout)
                    yield event

                    if event.type == "done":
                        break

                except asyncio.TimeoutError:
                    yield StreamEvent(type="keepalive", data=None)

        finally:
            self.unregister(connection_id)

    def get_stats(self) -> dict[str, Any]:
        """Get SSE manager statistics."""
        active_connections = sum(
            1 for c in self._connections.values() if c.is_active
        )

        return {
            "total_connections": len(self._connections),
            "active_connections": active_connections,
            "max_connections": self.max_connections,
        }


class EventAggregator:
    """
    Aggregates multiple events into batches.

    Useful for reducing network overhead.
    """

    def __init__(
        self,
        batch_size: int = 10,
        batch_timeout: float = 0.1,
    ):
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self._batch: list[StreamEvent] = []
        self._lock = asyncio.Lock()

    async def add(self, event: StreamEvent) -> Optional[list[StreamEvent]]:
        """
        Add an event to the batch.

        Returns the batch if it's ready to be sent.
        """
        async with self._lock:
            self._batch.append(event)

            if event.type in ("done", "error"):
                batch = self._batch.copy()
                self._batch.clear()
                return batch

            if len(self._batch) >= self.batch_size:
                batch = self._batch.copy()
                self._batch.clear()
                return batch

            return None

    async def flush(self) -> list[StreamEvent]:
        """Flush any remaining events."""
        async with self._lock:
            batch = self._batch.copy()
            self._batch.clear()
            return batch


def create_event_stream(
    events: list[StreamEvent],
    delay: float = 0.0,
) -> AsyncIterator[StreamEvent]:
    """
    Create an async iterator from a list of events.

    Useful for testing.

    Args:
        events: List of events to stream
        delay: Delay between events in seconds
    """

    async def generator():
        for event in events:
            if delay > 0:
                await asyncio.sleep(delay)
            yield event

    return generator()
