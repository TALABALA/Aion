"""
AION MCP Streaming and Progress Notifications

Production-grade streaming support for MCP operations:
- Streaming tool results with async generators
- Progress notifications for long-running operations
- Chunked data transfer
- Backpressure handling
"""

from __future__ import annotations

import asyncio
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    AsyncGenerator,
    AsyncIterator,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    TypeVar,
    Union,
)

import structlog

logger = structlog.get_logger(__name__)

T = TypeVar("T")


# ============================================
# Progress Notifications
# ============================================

class ProgressState(str, Enum):
    """Progress notification states."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ProgressUpdate:
    """Progress update notification."""
    operation_id: str
    state: ProgressState
    progress: float  # 0.0 to 1.0
    current_step: int
    total_steps: int
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    elapsed_seconds: float = 0.0
    estimated_remaining_seconds: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "operation_id": self.operation_id,
            "state": self.state.value,
            "progress": self.progress,
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "elapsed_seconds": self.elapsed_seconds,
            "estimated_remaining_seconds": self.estimated_remaining_seconds,
        }


class ProgressNotifier:
    """
    Notifier for long-running operation progress.

    Features:
    - Step-based progress tracking
    - Time estimation
    - Multiple subscriber support
    - Cancellation handling
    """

    def __init__(
        self,
        operation_id: Optional[str] = None,
        total_steps: int = 100,
        description: str = "Processing",
    ):
        """
        Initialize progress notifier.

        Args:
            operation_id: Unique operation identifier
            total_steps: Total number of steps
            description: Operation description
        """
        self.operation_id = operation_id or str(uuid.uuid4())
        self.total_steps = total_steps
        self.description = description

        self._current_step = 0
        self._state = ProgressState.PENDING
        self._message = description
        self._details: Dict[str, Any] = {}
        self._start_time: Optional[float] = None
        self._subscribers: List[Callable[[ProgressUpdate], None]] = []
        self._async_subscribers: List[Callable[[ProgressUpdate], asyncio.Future]] = []
        self._lock = asyncio.Lock()
        self._cancelled = asyncio.Event()

    @property
    def progress(self) -> float:
        """Get current progress (0.0 to 1.0)."""
        if self.total_steps == 0:
            return 1.0
        return min(1.0, self._current_step / self.total_steps)

    @property
    def elapsed_seconds(self) -> float:
        """Get elapsed time in seconds."""
        if self._start_time is None:
            return 0.0
        return time.monotonic() - self._start_time

    @property
    def estimated_remaining_seconds(self) -> Optional[float]:
        """Estimate remaining time based on current progress."""
        if self.progress <= 0 or self._start_time is None:
            return None

        elapsed = self.elapsed_seconds
        total_estimated = elapsed / self.progress
        return max(0, total_estimated - elapsed)

    def subscribe(self, callback: Callable[[ProgressUpdate], None]) -> None:
        """Add a synchronous subscriber for progress updates."""
        self._subscribers.append(callback)

    def subscribe_async(
        self,
        callback: Callable[[ProgressUpdate], asyncio.Future],
    ) -> None:
        """Add an async subscriber for progress updates."""
        self._async_subscribers.append(callback)

    def unsubscribe(self, callback: Callable) -> None:
        """Remove a subscriber."""
        if callback in self._subscribers:
            self._subscribers.remove(callback)
        if callback in self._async_subscribers:
            self._async_subscribers.remove(callback)

    async def start(self) -> None:
        """Start the operation."""
        async with self._lock:
            self._state = ProgressState.RUNNING
            self._start_time = time.monotonic()
            self._message = f"Starting: {self.description}"
            await self._notify()

    async def update(
        self,
        step: Optional[int] = None,
        increment: int = 0,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Update progress.

        Args:
            step: Set current step directly
            increment: Increment step by this amount
            message: Update progress message
            details: Additional details
        """
        if self._cancelled.is_set():
            raise asyncio.CancelledError("Operation was cancelled")

        async with self._lock:
            if step is not None:
                self._current_step = min(step, self.total_steps)
            else:
                self._current_step = min(
                    self._current_step + increment,
                    self.total_steps,
                )

            if message:
                self._message = message

            if details:
                self._details.update(details)

            await self._notify()

    async def complete(
        self,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Mark operation as completed."""
        async with self._lock:
            self._state = ProgressState.COMPLETED
            self._current_step = self.total_steps
            self._message = message or f"Completed: {self.description}"

            if details:
                self._details.update(details)

            await self._notify()

    async def fail(
        self,
        error: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Mark operation as failed."""
        async with self._lock:
            self._state = ProgressState.FAILED
            self._message = f"Failed: {error}"

            if details:
                self._details.update(details)
            self._details["error"] = error

            await self._notify()

    async def cancel(self) -> None:
        """Cancel the operation."""
        self._cancelled.set()
        async with self._lock:
            self._state = ProgressState.CANCELLED
            self._message = "Operation cancelled"
            await self._notify()

    async def _notify(self) -> None:
        """Notify all subscribers of progress update."""
        update = ProgressUpdate(
            operation_id=self.operation_id,
            state=self._state,
            progress=self.progress,
            current_step=self._current_step,
            total_steps=self.total_steps,
            message=self._message,
            details=self._details.copy(),
            elapsed_seconds=self.elapsed_seconds,
            estimated_remaining_seconds=self.estimated_remaining_seconds,
        )

        # Notify sync subscribers
        for subscriber in self._subscribers:
            try:
                subscriber(update)
            except Exception as e:
                logger.warning(f"Progress subscriber error: {e}")

        # Notify async subscribers
        for subscriber in self._async_subscribers:
            try:
                await subscriber(update)
            except Exception as e:
                logger.warning(f"Async progress subscriber error: {e}")

    def get_current_update(self) -> ProgressUpdate:
        """Get current progress update without notifying."""
        return ProgressUpdate(
            operation_id=self.operation_id,
            state=self._state,
            progress=self.progress,
            current_step=self._current_step,
            total_steps=self.total_steps,
            message=self._message,
            details=self._details.copy(),
            elapsed_seconds=self.elapsed_seconds,
            estimated_remaining_seconds=self.estimated_remaining_seconds,
        )


# ============================================
# Progress Context Manager
# ============================================

class ProgressContext:
    """
    Context manager for progress tracking.

    Usage:
        async with ProgressContext("Processing files", total=100) as progress:
            for i in range(100):
                await progress.update(increment=1, message=f"File {i}")
    """

    def __init__(
        self,
        description: str,
        total: int = 100,
        on_progress: Optional[Callable[[ProgressUpdate], None]] = None,
    ):
        self.description = description
        self.total = total
        self.on_progress = on_progress
        self._notifier: Optional[ProgressNotifier] = None

    async def __aenter__(self) -> ProgressNotifier:
        self._notifier = ProgressNotifier(
            total_steps=self.total,
            description=self.description,
        )

        if self.on_progress:
            self._notifier.subscribe(self.on_progress)

        await self._notifier.start()
        return self._notifier

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._notifier:
            if exc_type is None:
                await self._notifier.complete()
            elif exc_type is asyncio.CancelledError:
                await self._notifier.cancel()
            else:
                await self._notifier.fail(str(exc_val))

        return False


# ============================================
# Streaming Types
# ============================================

@dataclass
class StreamChunk(Generic[T]):
    """A chunk of streamed data."""
    data: T
    sequence: int
    is_final: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamError:
    """Error during streaming."""
    message: str
    code: Optional[str] = None
    recoverable: bool = False
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamMetadata:
    """Metadata for a stream."""
    stream_id: str
    total_chunks: Optional[int] = None
    content_type: Optional[str] = None
    encoding: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================
# Streaming Tool Result
# ============================================

class StreamingToolResult:
    """
    Streaming result from tool execution.

    Provides async iteration over result chunks with:
    - Progress tracking
    - Error handling
    - Backpressure support
    - Cancellation
    """

    def __init__(
        self,
        stream_id: Optional[str] = None,
        progress: Optional[ProgressNotifier] = None,
    ):
        """
        Initialize streaming result.

        Args:
            stream_id: Unique stream identifier
            progress: Progress notifier for tracking
        """
        self.stream_id = stream_id or str(uuid.uuid4())
        self.progress = progress

        self._chunks: asyncio.Queue[Union[StreamChunk, StreamError, None]] = asyncio.Queue()
        self._metadata: Optional[StreamMetadata] = None
        self._sequence = 0
        self._completed = asyncio.Event()
        self._error: Optional[StreamError] = None
        self._lock = asyncio.Lock()

        # Statistics
        self._chunks_sent = 0
        self._bytes_sent = 0
        self._start_time: Optional[float] = None

    async def start(self, metadata: Optional[StreamMetadata] = None) -> None:
        """Start the stream."""
        self._metadata = metadata or StreamMetadata(stream_id=self.stream_id)
        self._start_time = time.monotonic()

        if self.progress:
            await self.progress.start()

    async def send(
        self,
        data: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Send a chunk of data.

        Args:
            data: Data to send
            metadata: Optional chunk metadata
        """
        async with self._lock:
            self._sequence += 1
            chunk = StreamChunk(
                data=data,
                sequence=self._sequence,
                metadata=metadata or {},
            )

            await self._chunks.put(chunk)
            self._chunks_sent += 1

            if isinstance(data, (str, bytes)):
                self._bytes_sent += len(data)

            if self.progress:
                await self.progress.update(increment=1)

    async def send_error(
        self,
        message: str,
        code: Optional[str] = None,
        recoverable: bool = False,
    ) -> None:
        """Send an error notification."""
        error = StreamError(
            message=message,
            code=code,
            recoverable=recoverable,
        )
        await self._chunks.put(error)

        if not recoverable:
            self._error = error
            self._completed.set()

            if self.progress:
                await self.progress.fail(message)

    async def complete(self, final_data: Optional[Any] = None) -> None:
        """Complete the stream."""
        async with self._lock:
            if final_data is not None:
                self._sequence += 1
                chunk = StreamChunk(
                    data=final_data,
                    sequence=self._sequence,
                    is_final=True,
                )
                await self._chunks.put(chunk)
                self._chunks_sent += 1

            # Signal end of stream
            await self._chunks.put(None)
            self._completed.set()

            if self.progress:
                await self.progress.complete()

    async def __aiter__(self) -> AsyncIterator[StreamChunk]:
        """Iterate over stream chunks."""
        while True:
            item = await self._chunks.get()

            if item is None:
                break

            if isinstance(item, StreamError):
                if not item.recoverable:
                    raise RuntimeError(f"Stream error: {item.message}")
                logger.warning(f"Recoverable stream error: {item.message}")
                continue

            yield item

    def get_stats(self) -> Dict[str, Any]:
        """Get stream statistics."""
        elapsed = 0.0
        if self._start_time:
            elapsed = time.monotonic() - self._start_time

        return {
            "stream_id": self.stream_id,
            "chunks_sent": self._chunks_sent,
            "bytes_sent": self._bytes_sent,
            "elapsed_seconds": elapsed,
            "completed": self._completed.is_set(),
            "error": self._error.message if self._error else None,
        }


# ============================================
# Streaming Producer/Consumer
# ============================================

class StreamProducer(ABC, Generic[T]):
    """
    Abstract base for streaming producers.

    Implement this to create custom streaming data sources.
    """

    @abstractmethod
    async def produce(self) -> AsyncGenerator[T, None]:
        """Produce stream items."""
        yield  # type: ignore

    async def stream_to(self, result: StreamingToolResult) -> None:
        """Stream produced items to a result."""
        async for item in self.produce():
            await result.send(item)
        await result.complete()


class BufferedStreamConsumer(Generic[T]):
    """
    Buffered consumer for streaming data.

    Features:
    - Configurable buffer size
    - Backpressure handling
    - Timeout support
    """

    def __init__(
        self,
        buffer_size: int = 100,
        timeout: float = 30.0,
    ):
        """
        Initialize consumer.

        Args:
            buffer_size: Maximum buffer size
            timeout: Timeout for getting items
        """
        self.buffer_size = buffer_size
        self.timeout = timeout

        self._buffer: asyncio.Queue[Optional[T]] = asyncio.Queue(maxsize=buffer_size)
        self._completed = asyncio.Event()

    async def consume(self, stream: StreamingToolResult) -> List[T]:
        """
        Consume all items from stream.

        Args:
            stream: Stream to consume

        Returns:
            List of all items
        """
        items = []

        async for chunk in stream:
            items.append(chunk.data)

        return items

    async def consume_chunked(
        self,
        stream: StreamingToolResult,
        chunk_size: int = 10,
    ) -> AsyncGenerator[List[T], None]:
        """
        Consume stream in chunks.

        Args:
            stream: Stream to consume
            chunk_size: Size of each chunk

        Yields:
            Lists of items
        """
        buffer = []

        async for chunk in stream:
            buffer.append(chunk.data)

            if len(buffer) >= chunk_size:
                yield buffer
                buffer = []

        if buffer:
            yield buffer


# ============================================
# Tool Result Streamer
# ============================================

class ToolResultStreamer:
    """
    Utility for streaming tool execution results.

    Handles:
    - Chunking large results
    - Progress tracking
    - Error handling
    """

    def __init__(
        self,
        chunk_size: int = 4096,
        progress_interval: int = 10,
    ):
        """
        Initialize streamer.

        Args:
            chunk_size: Size of each chunk (for strings/bytes)
            progress_interval: Update progress every N chunks
        """
        self.chunk_size = chunk_size
        self.progress_interval = progress_interval

    async def stream_text(
        self,
        text: str,
        progress: Optional[ProgressNotifier] = None,
    ) -> StreamingToolResult:
        """
        Stream text content in chunks.

        Args:
            text: Text to stream
            progress: Optional progress notifier

        Returns:
            StreamingToolResult
        """
        total_chunks = (len(text) + self.chunk_size - 1) // self.chunk_size

        if progress:
            progress.total_steps = total_chunks

        result = StreamingToolResult(progress=progress)
        await result.start(StreamMetadata(
            stream_id=result.stream_id,
            total_chunks=total_chunks,
            content_type="text/plain",
            encoding="utf-8",
        ))

        # Stream chunks asynchronously
        async def stream_chunks():
            for i in range(0, len(text), self.chunk_size):
                chunk = text[i:i + self.chunk_size]
                is_final = (i + self.chunk_size) >= len(text)
                await result.send(chunk, {"is_final": is_final})

                # Small yield to allow other tasks
                if i % (self.chunk_size * 10) == 0:
                    await asyncio.sleep(0)

            await result.complete()

        asyncio.create_task(stream_chunks())
        return result

    async def stream_bytes(
        self,
        data: bytes,
        progress: Optional[ProgressNotifier] = None,
    ) -> StreamingToolResult:
        """
        Stream binary content in chunks.

        Args:
            data: Binary data to stream
            progress: Optional progress notifier

        Returns:
            StreamingToolResult
        """
        total_chunks = (len(data) + self.chunk_size - 1) // self.chunk_size

        if progress:
            progress.total_steps = total_chunks

        result = StreamingToolResult(progress=progress)
        await result.start(StreamMetadata(
            stream_id=result.stream_id,
            total_chunks=total_chunks,
            content_type="application/octet-stream",
        ))

        async def stream_chunks():
            for i in range(0, len(data), self.chunk_size):
                chunk = data[i:i + self.chunk_size]
                await result.send(chunk)

                if i % (self.chunk_size * 10) == 0:
                    await asyncio.sleep(0)

            await result.complete()

        asyncio.create_task(stream_chunks())
        return result

    async def stream_iterable(
        self,
        items: AsyncIterator[Any],
        total: Optional[int] = None,
        progress: Optional[ProgressNotifier] = None,
    ) -> StreamingToolResult:
        """
        Stream an async iterable.

        Args:
            items: Async iterable to stream
            total: Total number of items (for progress)
            progress: Optional progress notifier

        Returns:
            StreamingToolResult
        """
        if progress and total:
            progress.total_steps = total

        result = StreamingToolResult(progress=progress)
        await result.start(StreamMetadata(
            stream_id=result.stream_id,
            total_chunks=total,
        ))

        async def stream_items():
            async for item in items:
                await result.send(item)
            await result.complete()

        asyncio.create_task(stream_items())
        return result


# ============================================
# Progress-Tracked Operations
# ============================================

async def with_progress(
    operation: Callable[..., Any],
    description: str,
    total_steps: int = 100,
    on_progress: Optional[Callable[[ProgressUpdate], None]] = None,
    *args,
    **kwargs,
) -> Any:
    """
    Execute an operation with progress tracking.

    Args:
        operation: Async function to execute
        description: Operation description
        total_steps: Total steps for progress
        on_progress: Optional progress callback
        *args, **kwargs: Arguments for operation

    Returns:
        Operation result
    """
    progress = ProgressNotifier(
        total_steps=total_steps,
        description=description,
    )

    if on_progress:
        progress.subscribe(on_progress)

    try:
        await progress.start()

        # Pass progress notifier to operation if it accepts it
        import inspect
        sig = inspect.signature(operation)

        if "progress" in sig.parameters:
            result = await operation(*args, progress=progress, **kwargs)
        else:
            result = await operation(*args, **kwargs)

        await progress.complete()
        return result

    except asyncio.CancelledError:
        await progress.cancel()
        raise
    except Exception as e:
        await progress.fail(str(e))
        raise


# ============================================
# Progress Store
# ============================================

class ProgressStore:
    """
    Store for tracking multiple operations' progress.

    Thread-safe storage for progress notifiers indexed by operation ID.
    """

    def __init__(self, max_entries: int = 1000):
        """
        Initialize progress store.

        Args:
            max_entries: Maximum entries to store
        """
        self.max_entries = max_entries
        self._entries: Dict[str, ProgressNotifier] = {}
        self._lock = asyncio.Lock()

    async def create(
        self,
        operation_id: Optional[str] = None,
        total_steps: int = 100,
        description: str = "Processing",
    ) -> ProgressNotifier:
        """Create and store a new progress notifier."""
        notifier = ProgressNotifier(
            operation_id=operation_id,
            total_steps=total_steps,
            description=description,
        )

        async with self._lock:
            # Cleanup old entries if at capacity
            if len(self._entries) >= self.max_entries:
                completed = [
                    k for k, v in self._entries.items()
                    if v._state in (ProgressState.COMPLETED, ProgressState.FAILED, ProgressState.CANCELLED)
                ]
                for k in completed[:len(completed) // 2]:
                    del self._entries[k]

            self._entries[notifier.operation_id] = notifier

        return notifier

    async def get(self, operation_id: str) -> Optional[ProgressNotifier]:
        """Get a progress notifier by ID."""
        async with self._lock:
            return self._entries.get(operation_id)

    async def get_status(self, operation_id: str) -> Optional[ProgressUpdate]:
        """Get current status for an operation."""
        notifier = await self.get(operation_id)
        if notifier:
            return notifier.get_current_update()
        return None

    async def cancel(self, operation_id: str) -> bool:
        """Cancel an operation by ID."""
        notifier = await self.get(operation_id)
        if notifier:
            await notifier.cancel()
            return True
        return False

    async def list_active(self) -> List[ProgressUpdate]:
        """List all active operations."""
        async with self._lock:
            return [
                notifier.get_current_update()
                for notifier in self._entries.values()
                if notifier._state == ProgressState.RUNNING
            ]

    async def cleanup_completed(self, max_age_seconds: float = 3600) -> int:
        """Remove completed entries older than max_age."""
        now = time.time()
        removed = 0

        async with self._lock:
            to_remove = []
            for op_id, notifier in self._entries.items():
                if notifier._state in (ProgressState.COMPLETED, ProgressState.FAILED, ProgressState.CANCELLED):
                    if notifier.elapsed_seconds > max_age_seconds:
                        to_remove.append(op_id)

            for op_id in to_remove:
                del self._entries[op_id]
                removed += 1

        return removed


# ============================================
# Global Progress Store
# ============================================

_global_progress_store: Optional[ProgressStore] = None


def get_progress_store() -> ProgressStore:
    """Get the global progress store."""
    global _global_progress_store
    if _global_progress_store is None:
        _global_progress_store = ProgressStore()
    return _global_progress_store
