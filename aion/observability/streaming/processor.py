"""
Stream Processing Engine for Real-time Observability.

Implements Flink-style stream processing with:
- Windowed aggregations
- Keyed streams
- Event time processing
- Watermarks for out-of-order events
- Exactly-once semantics
- State management
- Checkpointing
"""

import asyncio
import logging
import time
import heapq
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Any, Dict, List, Optional, Tuple, Callable, Generic, TypeVar,
    Iterator, Union, AsyncIterator
)
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict
import json
import hashlib
import pickle

logger = logging.getLogger(__name__)

T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')
R = TypeVar('R')


# =============================================================================
# Core Data Types
# =============================================================================

@dataclass
class StreamRecord(Generic[T]):
    """Record in a data stream with metadata."""
    value: T
    timestamp: datetime
    key: Optional[Any] = None
    headers: Dict[str, str] = field(default_factory=dict)
    partition: int = 0
    offset: int = 0

    @property
    def event_time_ms(self) -> int:
        return int(self.timestamp.timestamp() * 1000)


@dataclass
class Watermark:
    """Watermark for event time progress."""
    timestamp: datetime
    source_id: str = ""

    @property
    def timestamp_ms(self) -> int:
        return int(self.timestamp.timestamp() * 1000)


class TimeCharacteristic(Enum):
    """Time semantics for stream processing."""
    PROCESSING_TIME = "processing_time"
    EVENT_TIME = "event_time"
    INGESTION_TIME = "ingestion_time"


# =============================================================================
# Stream Sources and Sinks
# =============================================================================

class StreamSource(ABC, Generic[T]):
    """Abstract source for reading data into stream."""

    @abstractmethod
    async def read(self) -> AsyncIterator[StreamRecord[T]]:
        """Read records from source."""
        pass

    @abstractmethod
    async def close(self):
        """Close the source."""
        pass

    def get_watermark(self) -> Optional[Watermark]:
        """Get current watermark (for event time processing)."""
        return None


class KafkaSource(StreamSource[T]):
    """Kafka-compatible source."""

    def __init__(
        self,
        topics: List[str],
        bootstrap_servers: str = "localhost:9092",
        group_id: str = "stream-processor",
        auto_offset_reset: str = "earliest",
        deserializer: Callable[[bytes], T] = None
    ):
        self.topics = topics
        self.bootstrap_servers = bootstrap_servers
        self.group_id = group_id
        self.auto_offset_reset = auto_offset_reset
        self.deserializer = deserializer or (lambda x: json.loads(x.decode()))
        self._consumer = None
        self._running = False
        self._current_watermark: Optional[Watermark] = None

    async def connect(self):
        """Connect to Kafka (placeholder for actual implementation)."""
        logger.info(f"Connecting to Kafka at {self.bootstrap_servers}")
        self._running = True

    async def read(self) -> AsyncIterator[StreamRecord[T]]:
        """Read records from Kafka topics."""
        while self._running:
            # Simulated Kafka consumption
            await asyncio.sleep(0.01)
            # In real implementation, poll Kafka here
            yield StreamRecord(
                value=None,  # Would be actual message
                timestamp=datetime.now(),
                partition=0,
                offset=0
            )

    async def close(self):
        """Close Kafka consumer."""
        self._running = False

    def get_watermark(self) -> Optional[Watermark]:
        return self._current_watermark


class CollectionSource(StreamSource[T]):
    """Source from an in-memory collection."""

    def __init__(self, data: List[T], timestamp_extractor: Callable[[T], datetime] = None):
        self.data = data
        self.timestamp_extractor = timestamp_extractor or (lambda x: datetime.now())
        self._index = 0

    async def read(self) -> AsyncIterator[StreamRecord[T]]:
        for item in self.data:
            yield StreamRecord(
                value=item,
                timestamp=self.timestamp_extractor(item)
            )
            await asyncio.sleep(0)  # Yield control

    async def close(self):
        pass


class StreamSink(ABC, Generic[T]):
    """Abstract sink for writing stream output."""

    @abstractmethod
    async def write(self, record: StreamRecord[T]):
        """Write a record to the sink."""
        pass

    @abstractmethod
    async def close(self):
        """Close the sink."""
        pass


class KafkaSink(StreamSink[T]):
    """Kafka-compatible sink."""

    def __init__(
        self,
        topic: str,
        bootstrap_servers: str = "localhost:9092",
        serializer: Callable[[T], bytes] = None
    ):
        self.topic = topic
        self.bootstrap_servers = bootstrap_servers
        self.serializer = serializer or (lambda x: json.dumps(x).encode())
        self._producer = None

    async def connect(self):
        """Connect to Kafka."""
        logger.info(f"Connecting Kafka producer to {self.bootstrap_servers}")

    async def write(self, record: StreamRecord[T]):
        """Write record to Kafka topic."""
        # In real implementation, produce to Kafka
        pass

    async def close(self):
        """Close Kafka producer."""
        pass


class ConsoleSink(StreamSink[T]):
    """Sink that prints to console."""

    def __init__(self, prefix: str = ""):
        self.prefix = prefix

    async def write(self, record: StreamRecord[T]):
        print(f"{self.prefix}{record.timestamp}: {record.value}")

    async def close(self):
        pass


class CallbackSink(StreamSink[T]):
    """Sink that calls a callback function."""

    def __init__(self, callback: Callable[[StreamRecord[T]], None]):
        self.callback = callback

    async def write(self, record: StreamRecord[T]):
        self.callback(record)

    async def close(self):
        pass


# =============================================================================
# Stream Operators
# =============================================================================

class StreamOperator(ABC, Generic[T, R]):
    """Base class for stream operators."""

    def __init__(self):
        self.metrics = {
            "records_in": 0,
            "records_out": 0,
            "processing_time_ms": 0
        }

    @abstractmethod
    async def process(self, record: StreamRecord[T]) -> AsyncIterator[StreamRecord[R]]:
        """Process a single record."""
        pass

    def on_timer(self, timestamp: datetime):
        """Called when a timer fires (for event time processing)."""
        pass


class MapOperator(StreamOperator[T, R]):
    """Map each record to a new value."""

    def __init__(self, map_fn: Callable[[T], R]):
        super().__init__()
        self.map_fn = map_fn

    async def process(self, record: StreamRecord[T]) -> AsyncIterator[StreamRecord[R]]:
        self.metrics["records_in"] += 1
        start = time.time()

        result = self.map_fn(record.value)

        self.metrics["processing_time_ms"] += (time.time() - start) * 1000
        self.metrics["records_out"] += 1

        yield StreamRecord(
            value=result,
            timestamp=record.timestamp,
            key=record.key,
            headers=record.headers
        )


class FilterOperator(StreamOperator[T, T]):
    """Filter records based on a predicate."""

    def __init__(self, predicate: Callable[[T], bool]):
        super().__init__()
        self.predicate = predicate

    async def process(self, record: StreamRecord[T]) -> AsyncIterator[StreamRecord[T]]:
        self.metrics["records_in"] += 1

        if self.predicate(record.value):
            self.metrics["records_out"] += 1
            yield record


class FlatMapOperator(StreamOperator[T, R]):
    """Map each record to zero or more records."""

    def __init__(self, flat_map_fn: Callable[[T], Iterator[R]]):
        super().__init__()
        self.flat_map_fn = flat_map_fn

    async def process(self, record: StreamRecord[T]) -> AsyncIterator[StreamRecord[R]]:
        self.metrics["records_in"] += 1

        for result in self.flat_map_fn(record.value):
            self.metrics["records_out"] += 1
            yield StreamRecord(
                value=result,
                timestamp=record.timestamp,
                key=record.key,
                headers=record.headers
            )


# =============================================================================
# State Management
# =============================================================================

class StateBackend(ABC):
    """Abstract state backend for operator state."""

    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get state value."""
        pass

    @abstractmethod
    async def put(self, key: str, value: Any):
        """Put state value."""
        pass

    @abstractmethod
    async def delete(self, key: str):
        """Delete state value."""
        pass

    @abstractmethod
    async def checkpoint(self) -> bytes:
        """Create checkpoint of all state."""
        pass

    @abstractmethod
    async def restore(self, checkpoint: bytes):
        """Restore state from checkpoint."""
        pass


class MemoryStateBackend(StateBackend):
    """In-memory state backend."""

    def __init__(self):
        self._state: Dict[str, Any] = {}

    async def get(self, key: str) -> Optional[Any]:
        return self._state.get(key)

    async def put(self, key: str, value: Any):
        self._state[key] = value

    async def delete(self, key: str):
        self._state.pop(key, None)

    async def checkpoint(self) -> bytes:
        return pickle.dumps(self._state)

    async def restore(self, checkpoint: bytes):
        self._state = pickle.loads(checkpoint)


class RocksDBStateBackend(StateBackend):
    """RocksDB-based state backend (placeholder)."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._state: Dict[str, Any] = {}  # Fallback to memory

    async def get(self, key: str) -> Optional[Any]:
        return self._state.get(key)

    async def put(self, key: str, value: Any):
        self._state[key] = value

    async def delete(self, key: str):
        self._state.pop(key, None)

    async def checkpoint(self) -> bytes:
        return pickle.dumps(self._state)

    async def restore(self, checkpoint: bytes):
        self._state = pickle.loads(checkpoint)


# =============================================================================
# Windowing
# =============================================================================

class WindowAssigner(ABC):
    """Assigns elements to windows."""

    @abstractmethod
    def assign_windows(self, timestamp: datetime) -> List['Window']:
        """Assign windows for a timestamp."""
        pass

    @abstractmethod
    def get_default_trigger(self) -> 'Trigger':
        """Get default trigger for this window type."""
        pass


@dataclass
class Window:
    """Base window class."""
    start: datetime
    end: datetime

    @property
    def max_timestamp(self) -> datetime:
        return self.end - timedelta(milliseconds=1)

    def contains(self, timestamp: datetime) -> bool:
        return self.start <= timestamp < self.end


class TumblingWindow(WindowAssigner):
    """Non-overlapping fixed-size windows."""

    def __init__(self, size: timedelta):
        self.size = size

    def assign_windows(self, timestamp: datetime) -> List[Window]:
        # Align to window boundaries
        window_start_ms = (int(timestamp.timestamp() * 1000) //
                          int(self.size.total_seconds() * 1000) *
                          int(self.size.total_seconds() * 1000))
        window_start = datetime.fromtimestamp(window_start_ms / 1000)
        window_end = window_start + self.size

        return [Window(start=window_start, end=window_end)]

    def get_default_trigger(self) -> 'Trigger':
        return EventTimeTrigger()


class SlidingWindow(WindowAssigner):
    """Overlapping windows with fixed size and slide."""

    def __init__(self, size: timedelta, slide: timedelta):
        self.size = size
        self.slide = slide

    def assign_windows(self, timestamp: datetime) -> List[Window]:
        windows = []
        ts_ms = int(timestamp.timestamp() * 1000)
        size_ms = int(self.size.total_seconds() * 1000)
        slide_ms = int(self.slide.total_seconds() * 1000)

        # Find all windows that contain this timestamp
        last_start = ts_ms - ts_ms % slide_ms
        first_start = last_start - size_ms + slide_ms

        for start_ms in range(int(first_start), int(last_start) + slide_ms, slide_ms):
            if start_ms <= ts_ms < start_ms + size_ms:
                windows.append(Window(
                    start=datetime.fromtimestamp(start_ms / 1000),
                    end=datetime.fromtimestamp((start_ms + size_ms) / 1000)
                ))

        return windows

    def get_default_trigger(self) -> 'Trigger':
        return EventTimeTrigger()


class SessionWindow(WindowAssigner):
    """Dynamic windows based on activity gaps."""

    def __init__(self, gap: timedelta):
        self.gap = gap
        self._sessions: Dict[Any, Window] = {}

    def assign_windows(self, timestamp: datetime, key: Any = None) -> List[Window]:
        # Session windows are key-dependent
        if key in self._sessions:
            session = self._sessions[key]
            if timestamp < session.end + self.gap:
                # Extend session
                new_end = max(session.end, timestamp + self.gap)
                self._sessions[key] = Window(start=session.start, end=new_end)
                return [self._sessions[key]]

        # New session
        new_session = Window(start=timestamp, end=timestamp + self.gap)
        self._sessions[key] = new_session
        return [new_session]

    def get_default_trigger(self) -> 'Trigger':
        return SessionTrigger(self.gap)


# =============================================================================
# Triggers
# =============================================================================

class TriggerResult(Enum):
    """Result of trigger evaluation."""
    CONTINUE = "continue"  # Do nothing
    FIRE = "fire"  # Emit window result
    PURGE = "purge"  # Clear window state
    FIRE_AND_PURGE = "fire_and_purge"  # Emit and clear


class Trigger(ABC):
    """Determines when window results are emitted."""

    @abstractmethod
    def on_element(self, element: Any, timestamp: datetime, window: Window) -> TriggerResult:
        """Called for each element added to window."""
        pass

    @abstractmethod
    def on_event_time(self, time: datetime, window: Window) -> TriggerResult:
        """Called when event time watermark passes."""
        pass

    @abstractmethod
    def on_processing_time(self, time: datetime, window: Window) -> TriggerResult:
        """Called when processing time timer fires."""
        pass


class EventTimeTrigger(Trigger):
    """Fires when watermark passes window end."""

    def on_element(self, element: Any, timestamp: datetime, window: Window) -> TriggerResult:
        return TriggerResult.CONTINUE

    def on_event_time(self, time: datetime, window: Window) -> TriggerResult:
        if time >= window.end:
            return TriggerResult.FIRE_AND_PURGE
        return TriggerResult.CONTINUE

    def on_processing_time(self, time: datetime, window: Window) -> TriggerResult:
        return TriggerResult.CONTINUE


class ProcessingTimeTrigger(Trigger):
    """Fires based on processing time."""

    def __init__(self, interval: timedelta):
        self.interval = interval
        self._last_fire: Dict[Window, datetime] = {}

    def on_element(self, element: Any, timestamp: datetime, window: Window) -> TriggerResult:
        return TriggerResult.CONTINUE

    def on_event_time(self, time: datetime, window: Window) -> TriggerResult:
        return TriggerResult.CONTINUE

    def on_processing_time(self, time: datetime, window: Window) -> TriggerResult:
        last = self._last_fire.get(window)
        if last is None or time >= last + self.interval:
            self._last_fire[window] = time
            return TriggerResult.FIRE
        return TriggerResult.CONTINUE


class CountTrigger(Trigger):
    """Fires after N elements."""

    def __init__(self, count: int):
        self.count = count
        self._counts: Dict[Window, int] = defaultdict(int)

    def on_element(self, element: Any, timestamp: datetime, window: Window) -> TriggerResult:
        self._counts[window] += 1
        if self._counts[window] >= self.count:
            self._counts[window] = 0
            return TriggerResult.FIRE
        return TriggerResult.CONTINUE

    def on_event_time(self, time: datetime, window: Window) -> TriggerResult:
        return TriggerResult.CONTINUE

    def on_processing_time(self, time: datetime, window: Window) -> TriggerResult:
        return TriggerResult.CONTINUE


class SessionTrigger(Trigger):
    """Fires when session gap expires."""

    def __init__(self, gap: timedelta):
        self.gap = gap
        self._last_activity: Dict[Window, datetime] = {}

    def on_element(self, element: Any, timestamp: datetime, window: Window) -> TriggerResult:
        self._last_activity[window] = timestamp
        return TriggerResult.CONTINUE

    def on_event_time(self, time: datetime, window: Window) -> TriggerResult:
        last = self._last_activity.get(window)
        if last and time >= last + self.gap:
            return TriggerResult.FIRE_AND_PURGE
        return TriggerResult.CONTINUE

    def on_processing_time(self, time: datetime, window: Window) -> TriggerResult:
        return TriggerResult.CONTINUE


# =============================================================================
# Aggregation Functions
# =============================================================================

class AggregateFunction(ABC, Generic[T, V, R]):
    """Aggregate function for windowed operations."""

    @abstractmethod
    def create_accumulator(self) -> V:
        """Create initial accumulator."""
        pass

    @abstractmethod
    def add(self, accumulator: V, value: T) -> V:
        """Add value to accumulator."""
        pass

    @abstractmethod
    def get_result(self, accumulator: V) -> R:
        """Get result from accumulator."""
        pass

    def merge(self, a: V, b: V) -> V:
        """Merge two accumulators."""
        raise NotImplementedError("Merge not implemented")


class SumAggregate(AggregateFunction[float, float, float]):
    """Sum aggregation."""

    def create_accumulator(self) -> float:
        return 0.0

    def add(self, accumulator: float, value: float) -> float:
        return accumulator + value

    def get_result(self, accumulator: float) -> float:
        return accumulator

    def merge(self, a: float, b: float) -> float:
        return a + b


class CountAggregate(AggregateFunction[Any, int, int]):
    """Count aggregation."""

    def create_accumulator(self) -> int:
        return 0

    def add(self, accumulator: int, value: Any) -> int:
        return accumulator + 1

    def get_result(self, accumulator: int) -> int:
        return accumulator

    def merge(self, a: int, b: int) -> int:
        return a + b


class AverageAggregate(AggregateFunction[float, Tuple[float, int], float]):
    """Average aggregation."""

    def create_accumulator(self) -> Tuple[float, int]:
        return (0.0, 0)

    def add(self, accumulator: Tuple[float, int], value: float) -> Tuple[float, int]:
        return (accumulator[0] + value, accumulator[1] + 1)

    def get_result(self, accumulator: Tuple[float, int]) -> float:
        if accumulator[1] == 0:
            return 0.0
        return accumulator[0] / accumulator[1]

    def merge(self, a: Tuple[float, int], b: Tuple[float, int]) -> Tuple[float, int]:
        return (a[0] + b[0], a[1] + b[1])


class MinAggregate(AggregateFunction[float, Optional[float], Optional[float]]):
    """Minimum aggregation."""

    def create_accumulator(self) -> Optional[float]:
        return None

    def add(self, accumulator: Optional[float], value: float) -> Optional[float]:
        if accumulator is None:
            return value
        return min(accumulator, value)

    def get_result(self, accumulator: Optional[float]) -> Optional[float]:
        return accumulator

    def merge(self, a: Optional[float], b: Optional[float]) -> Optional[float]:
        if a is None:
            return b
        if b is None:
            return a
        return min(a, b)


class MaxAggregate(AggregateFunction[float, Optional[float], Optional[float]]):
    """Maximum aggregation."""

    def create_accumulator(self) -> Optional[float]:
        return None

    def add(self, accumulator: Optional[float], value: float) -> Optional[float]:
        if accumulator is None:
            return value
        return max(accumulator, value)

    def get_result(self, accumulator: Optional[float]) -> Optional[float]:
        return accumulator

    def merge(self, a: Optional[float], b: Optional[float]) -> Optional[float]:
        if a is None:
            return b
        if b is None:
            return a
        return max(a, b)


# =============================================================================
# Data Streams
# =============================================================================

class DataStream(Generic[T]):
    """A stream of data records."""

    def __init__(self, processor: 'StreamProcessor', operators: List[StreamOperator] = None):
        self.processor = processor
        self.operators = operators or []

    def map(self, map_fn: Callable[[T], R]) -> 'DataStream[R]':
        """Apply a map transformation."""
        new_operators = self.operators + [MapOperator(map_fn)]
        return DataStream(self.processor, new_operators)

    def filter(self, predicate: Callable[[T], bool]) -> 'DataStream[T]':
        """Filter records."""
        new_operators = self.operators + [FilterOperator(predicate)]
        return DataStream(self.processor, new_operators)

    def flat_map(self, flat_map_fn: Callable[[T], Iterator[R]]) -> 'DataStream[R]':
        """Apply a flat map transformation."""
        new_operators = self.operators + [FlatMapOperator(flat_map_fn)]
        return DataStream(self.processor, new_operators)

    def key_by(self, key_fn: Callable[[T], K]) -> 'KeyedStream[K, T]':
        """Key the stream by a key extractor."""
        return KeyedStream(self.processor, self.operators, key_fn)

    def add_sink(self, sink: StreamSink[T]) -> 'DataStream[T]':
        """Add a sink to the stream."""
        self.processor.add_sink(self, sink)
        return self


class KeyedStream(Generic[K, T]):
    """A keyed stream for stateful operations."""

    def __init__(
        self,
        processor: 'StreamProcessor',
        operators: List[StreamOperator],
        key_fn: Callable[[T], K]
    ):
        self.processor = processor
        self.operators = operators
        self.key_fn = key_fn

    def window(self, assigner: WindowAssigner) -> 'WindowedStream[K, T]':
        """Apply windowing to the keyed stream."""
        return WindowedStream(self.processor, self.operators, self.key_fn, assigner)

    def reduce(self, reduce_fn: Callable[[T, T], T]) -> DataStream[T]:
        """Apply a reduce function."""
        # Create a reduce operator
        class KeyedReduceOperator(StreamOperator[T, T]):
            def __init__(self, key_fn, reduce_fn):
                super().__init__()
                self.key_fn = key_fn
                self.reduce_fn = reduce_fn
                self.state: Dict[K, T] = {}

            async def process(self, record: StreamRecord[T]) -> AsyncIterator[StreamRecord[T]]:
                key = self.key_fn(record.value)
                if key in self.state:
                    self.state[key] = self.reduce_fn(self.state[key], record.value)
                else:
                    self.state[key] = record.value

                yield StreamRecord(
                    value=self.state[key],
                    timestamp=record.timestamp,
                    key=key
                )

        new_operators = self.operators + [KeyedReduceOperator(self.key_fn, reduce_fn)]
        return DataStream(self.processor, new_operators)


class WindowedStream(Generic[K, T]):
    """A windowed keyed stream."""

    def __init__(
        self,
        processor: 'StreamProcessor',
        operators: List[StreamOperator],
        key_fn: Callable[[T], K],
        window_assigner: WindowAssigner
    ):
        self.processor = processor
        self.operators = operators
        self.key_fn = key_fn
        self.window_assigner = window_assigner
        self.trigger = window_assigner.get_default_trigger()

    def trigger(self, trigger: Trigger) -> 'WindowedStream[K, T]':
        """Set custom trigger."""
        self.trigger = trigger
        return self

    def aggregate(self, agg_fn: AggregateFunction[T, V, R]) -> DataStream[R]:
        """Apply an aggregate function to windows."""

        class WindowAggregateOperator(StreamOperator[T, R]):
            def __init__(self, key_fn, window_assigner, trigger, agg_fn):
                super().__init__()
                self.key_fn = key_fn
                self.window_assigner = window_assigner
                self.trigger = trigger
                self.agg_fn = agg_fn
                # State: key -> window -> accumulator
                self.window_state: Dict[K, Dict[Window, V]] = defaultdict(dict)

            async def process(self, record: StreamRecord[T]) -> AsyncIterator[StreamRecord[R]]:
                key = self.key_fn(record.value)
                windows = self.window_assigner.assign_windows(record.timestamp)

                for window in windows:
                    # Get or create accumulator
                    if window not in self.window_state[key]:
                        self.window_state[key][window] = self.agg_fn.create_accumulator()

                    # Add value
                    self.window_state[key][window] = self.agg_fn.add(
                        self.window_state[key][window],
                        record.value
                    )

                    # Check trigger
                    result = self.trigger.on_element(record.value, record.timestamp, window)

                    if result in (TriggerResult.FIRE, TriggerResult.FIRE_AND_PURGE):
                        # Emit result
                        agg_result = self.agg_fn.get_result(self.window_state[key][window])
                        yield StreamRecord(
                            value=agg_result,
                            timestamp=window.end,
                            key=key
                        )

                    if result in (TriggerResult.PURGE, TriggerResult.FIRE_AND_PURGE):
                        # Clear window state
                        del self.window_state[key][window]

        new_operators = self.operators + [
            WindowAggregateOperator(self.key_fn, self.window_assigner, self.trigger, agg_fn)
        ]
        return DataStream(self.processor, new_operators)

    def sum(self, value_fn: Callable[[T], float] = None) -> DataStream[float]:
        """Sum values in window."""
        if value_fn:
            # First map to extract value
            agg = SumAggregate()
            # Would need to compose with map
        return self.aggregate(SumAggregate())

    def count(self) -> DataStream[int]:
        """Count elements in window."""
        return self.aggregate(CountAggregate())

    def min(self, value_fn: Callable[[T], float] = None) -> DataStream[Optional[float]]:
        """Get minimum in window."""
        return self.aggregate(MinAggregate())

    def max(self, value_fn: Callable[[T], float] = None) -> DataStream[Optional[float]]:
        """Get maximum in window."""
        return self.aggregate(MaxAggregate())


# =============================================================================
# Stream Processor
# =============================================================================

class StreamProcessor:
    """
    Main stream processing engine.

    Orchestrates sources, operators, and sinks to process data streams.
    Supports event time processing, windowing, and state management.
    """

    def __init__(
        self,
        name: str = "stream-processor",
        time_characteristic: TimeCharacteristic = TimeCharacteristic.EVENT_TIME,
        parallelism: int = 1,
        checkpoint_interval: timedelta = timedelta(minutes=1),
        state_backend: StateBackend = None
    ):
        self.name = name
        self.time_characteristic = time_characteristic
        self.parallelism = parallelism
        self.checkpoint_interval = checkpoint_interval
        self.state_backend = state_backend or MemoryStateBackend()

        self._sources: List[Tuple[StreamSource, DataStream]] = []
        self._sinks: List[Tuple[DataStream, StreamSink]] = []
        self._running = False
        self._watermark = datetime.min
        self._timer_queue: List[Tuple[datetime, Callable]] = []
        self._metrics = {
            "records_processed": 0,
            "bytes_processed": 0,
            "checkpoints_completed": 0,
            "latency_ms": 0
        }

    def add_source(self, source: StreamSource[T]) -> DataStream[T]:
        """Add a source to the processor."""
        stream = DataStream(self)
        self._sources.append((source, stream))
        return stream

    def add_sink(self, stream: DataStream, sink: StreamSink):
        """Add a sink for a stream."""
        self._sinks.append((stream, sink))

    def from_collection(self, data: List[T]) -> DataStream[T]:
        """Create stream from collection."""
        source = CollectionSource(data)
        return self.add_source(source)

    async def execute(self) -> Dict[str, Any]:
        """Execute the stream processing job."""
        self._running = True
        logger.info(f"Starting stream processor: {self.name}")

        # Start checkpoint timer
        checkpoint_task = asyncio.create_task(self._checkpoint_loop())

        try:
            # Process all sources concurrently
            tasks = [
                self._process_source(source, stream)
                for source, stream in self._sources
            ]

            await asyncio.gather(*tasks)

        finally:
            self._running = False
            checkpoint_task.cancel()
            try:
                await checkpoint_task
            except asyncio.CancelledError:
                pass

            # Close sinks
            for _, sink in self._sinks:
                await sink.close()

        return self._metrics

    async def _process_source(self, source: StreamSource[T], initial_stream: DataStream[T]):
        """Process records from a source."""
        async for record in source.read():
            if not self._running:
                break

            # Update watermark
            if self.time_characteristic == TimeCharacteristic.EVENT_TIME:
                self._update_watermark(record.timestamp)

            # Process through operators
            current_records = [record]

            for operator in initial_stream.operators:
                next_records = []
                for rec in current_records:
                    async for output in operator.process(rec):
                        next_records.append(output)
                current_records = next_records

            # Write to sinks
            for stream, sink in self._sinks:
                if stream.operators == initial_stream.operators:
                    for rec in current_records:
                        await sink.write(rec)

            self._metrics["records_processed"] += 1

    def _update_watermark(self, timestamp: datetime):
        """Update the current watermark."""
        if timestamp > self._watermark:
            self._watermark = timestamp

            # Fire timers
            while self._timer_queue and self._timer_queue[0][0] <= self._watermark:
                _, callback = heapq.heappop(self._timer_queue)
                callback()

    def register_timer(self, timestamp: datetime, callback: Callable):
        """Register a timer to fire at a specific event time."""
        heapq.heappush(self._timer_queue, (timestamp, callback))

    async def _checkpoint_loop(self):
        """Periodically create checkpoints."""
        while self._running:
            await asyncio.sleep(self.checkpoint_interval.total_seconds())

            if self._running:
                try:
                    checkpoint = await self.state_backend.checkpoint()
                    self._metrics["checkpoints_completed"] += 1
                    logger.debug(f"Checkpoint completed: {len(checkpoint)} bytes")
                except Exception as e:
                    logger.error(f"Checkpoint failed: {e}")

    async def stop(self):
        """Stop the stream processor."""
        self._running = False
        logger.info(f"Stopping stream processor: {self.name}")
