"""
AION Conversation Streaming

Streaming response handling and server-sent events.
"""

from aion.conversation.streaming.handler import (
    StreamingHandler,
    StreamingState,
    StreamBuffer,
    aggregate_text_events,
)
from aion.conversation.streaming.events import (
    format_sse,
    format_sse_data,
    format_sse_comment,
    sse_generator,
    SSEConnection,
    SSEManager,
    EventAggregator,
    create_event_stream,
)

__all__ = [
    "StreamingHandler",
    "StreamingState",
    "StreamBuffer",
    "aggregate_text_events",
    "format_sse",
    "format_sse_data",
    "format_sse_comment",
    "sse_generator",
    "SSEConnection",
    "SSEManager",
    "EventAggregator",
    "create_event_stream",
]
