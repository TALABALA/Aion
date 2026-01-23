"""
AION Streaming Response Handler

Handles streaming responses from LLM providers.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, AsyncIterator, Callable, Optional
import structlog

from aion.conversation.types import (
    StreamEvent,
    StreamEventType,
    Message,
    MessageRole,
    TextContent,
    ToolUseContent,
    ThinkingContent,
)

logger = structlog.get_logger(__name__)


@dataclass
class StreamingState:
    """Tracks the state of a streaming response."""
    started_at: datetime = field(default_factory=datetime.now)

    text_buffer: str = ""
    thinking_buffer: str = ""

    current_tool_id: Optional[str] = None
    current_tool_name: Optional[str] = None
    current_tool_input: str = ""

    completed_tool_ids: list[str] = field(default_factory=list)

    is_complete: bool = False
    has_error: bool = False
    error_message: Optional[str] = None

    input_tokens: int = 0
    output_tokens: int = 0

    def to_message(self) -> Message:
        """Convert accumulated state to a Message."""
        content = []

        if self.thinking_buffer:
            content.append(ThinkingContent(thinking=self.thinking_buffer))

        if self.text_buffer:
            content.append(TextContent(text=self.text_buffer))

        return Message(
            role=MessageRole.ASSISTANT,
            content=content,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
        )

    def reset(self) -> None:
        """Reset the streaming state."""
        self.text_buffer = ""
        self.thinking_buffer = ""
        self.current_tool_id = None
        self.current_tool_name = None
        self.current_tool_input = ""
        self.completed_tool_ids.clear()
        self.is_complete = False
        self.has_error = False
        self.error_message = None
        self.started_at = datetime.now()


class StreamingHandler:
    """
    Handles streaming responses from LLM.

    Features:
    - Event buffering and aggregation
    - Tool use tracking
    - Error handling
    - Progress callbacks
    """

    def __init__(
        self,
        on_text: Optional[Callable[[str], None]] = None,
        on_tool_start: Optional[Callable[[str, str], None]] = None,
        on_tool_end: Optional[Callable[[str, dict], None]] = None,
        on_thinking: Optional[Callable[[str], None]] = None,
        on_complete: Optional[Callable[[StreamingState], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
        text_buffer_size: int = 0,
    ):
        self.on_text = on_text
        self.on_tool_start = on_tool_start
        self.on_tool_end = on_tool_end
        self.on_thinking = on_thinking
        self.on_complete = on_complete
        self.on_error = on_error

        self.text_buffer_size = text_buffer_size
        self._pending_text = ""

        self.state = StreamingState()

    async def handle_stream(
        self,
        stream: AsyncIterator[StreamEvent],
    ) -> AsyncIterator[StreamEvent]:
        """
        Handle a stream of events.

        Processes events, updates state, and yields processed events.
        """
        self.state.reset()

        try:
            async for event in stream:
                processed = await self._process_event(event)
                if processed:
                    yield processed

        except Exception as e:
            logger.error(f"Stream handling error: {e}")
            self.state.has_error = True
            self.state.error_message = str(e)

            if self.on_error:
                self.on_error(str(e))

            yield StreamEvent.error(str(e))

        finally:
            if self._pending_text:
                yield StreamEvent.text(self._pending_text)
                self._pending_text = ""

    async def _process_event(self, event: StreamEvent) -> Optional[StreamEvent]:
        """Process a single streaming event."""
        event_type = event.type

        if event_type == StreamEventType.TEXT.value or event_type == "text":
            return await self._handle_text(event)

        elif event_type == StreamEventType.THINKING.value or event_type == "thinking":
            return await self._handle_thinking(event)

        elif event_type == StreamEventType.THINKING_START.value or event_type == "thinking_start":
            return event

        elif event_type == StreamEventType.TOOL_USE_START.value or event_type == "tool_use_start":
            return await self._handle_tool_start(event)

        elif event_type == StreamEventType.TOOL_USE_INPUT.value or event_type == "tool_use_input":
            return await self._handle_tool_input(event)

        elif event_type == StreamEventType.TOOL_USE_END.value or event_type == "tool_use_end":
            return await self._handle_tool_end(event)

        elif event_type == StreamEventType.DONE.value or event_type == "done":
            return await self._handle_done(event)

        elif event_type == StreamEventType.ERROR.value or event_type == "error":
            return await self._handle_error(event)

        return event

    async def _handle_text(self, event: StreamEvent) -> Optional[StreamEvent]:
        """Handle text event."""
        text = event.data or ""
        self.state.text_buffer += text

        if self.on_text:
            self.on_text(text)

        if self.text_buffer_size > 0:
            self._pending_text += text
            if len(self._pending_text) >= self.text_buffer_size:
                buffered = self._pending_text
                self._pending_text = ""
                return StreamEvent.text(buffered)
            return None

        return event

    async def _handle_thinking(self, event: StreamEvent) -> Optional[StreamEvent]:
        """Handle thinking event."""
        thinking = event.data or ""
        self.state.thinking_buffer += thinking

        if self.on_thinking:
            self.on_thinking(thinking)

        return event

    async def _handle_tool_start(self, event: StreamEvent) -> Optional[StreamEvent]:
        """Handle tool use start event."""
        data = event.data or {}
        self.state.current_tool_id = data.get("id", "")
        self.state.current_tool_name = data.get("name", "")
        self.state.current_tool_input = ""

        if self.on_tool_start:
            self.on_tool_start(
                self.state.current_tool_id,
                self.state.current_tool_name,
            )

        return event

    async def _handle_tool_input(self, event: StreamEvent) -> Optional[StreamEvent]:
        """Handle tool input event."""
        if isinstance(event.data, dict):
            import json
            self.state.current_tool_input = json.dumps(event.data)
        elif isinstance(event.data, str):
            self.state.current_tool_input += event.data

        return None

    async def _handle_tool_end(self, event: StreamEvent) -> Optional[StreamEvent]:
        """Handle tool use end event."""
        import json

        try:
            tool_input = (
                json.loads(self.state.current_tool_input)
                if self.state.current_tool_input
                else {}
            )
        except json.JSONDecodeError:
            tool_input = {}

        if self.on_tool_end and self.state.current_tool_id:
            self.on_tool_end(self.state.current_tool_id, tool_input)

        if self.state.current_tool_id:
            self.state.completed_tool_ids.append(self.state.current_tool_id)

        self.state.current_tool_id = None
        self.state.current_tool_name = None
        self.state.current_tool_input = ""

        return event

    async def _handle_done(self, event: StreamEvent) -> Optional[StreamEvent]:
        """Handle stream completion."""
        self.state.is_complete = True

        if isinstance(event.data, dict):
            self.state.input_tokens = event.data.get("input_tokens", 0)
            self.state.output_tokens = event.data.get("output_tokens", 0)

        if self.on_complete:
            self.on_complete(self.state)

        return event

    async def _handle_error(self, event: StreamEvent) -> Optional[StreamEvent]:
        """Handle error event."""
        self.state.has_error = True
        self.state.error_message = str(event.data)

        if self.on_error:
            self.on_error(str(event.data))

        return event


class StreamBuffer:
    """
    Buffers streaming events for batch processing.
    """

    def __init__(self, max_size: int = 100, flush_interval: float = 0.1):
        self.max_size = max_size
        self.flush_interval = flush_interval
        self._buffer: list[StreamEvent] = []
        self._lock = asyncio.Lock()
        self._last_flush = datetime.now()

    async def add(self, event: StreamEvent) -> Optional[list[StreamEvent]]:
        """
        Add an event to the buffer.

        Returns buffered events if flush threshold reached.
        """
        async with self._lock:
            self._buffer.append(event)

            should_flush = (
                len(self._buffer) >= self.max_size
                or (datetime.now() - self._last_flush).total_seconds() >= self.flush_interval
            )

            if should_flush:
                events = self._buffer.copy()
                self._buffer.clear()
                self._last_flush = datetime.now()
                return events

            return None

    async def flush(self) -> list[StreamEvent]:
        """Force flush the buffer."""
        async with self._lock:
            events = self._buffer.copy()
            self._buffer.clear()
            self._last_flush = datetime.now()
            return events


async def aggregate_text_events(
    stream: AsyncIterator[StreamEvent],
    min_chars: int = 10,
) -> AsyncIterator[StreamEvent]:
    """
    Aggregate text events into larger chunks.

    Useful for reducing the number of events sent over network.
    """
    text_buffer = ""

    async for event in stream:
        if event.type == "text":
            text_buffer += event.data or ""

            if len(text_buffer) >= min_chars:
                yield StreamEvent.text(text_buffer)
                text_buffer = ""
        else:
            if text_buffer:
                yield StreamEvent.text(text_buffer)
                text_buffer = ""

            yield event

    if text_buffer:
        yield StreamEvent.text(text_buffer)
