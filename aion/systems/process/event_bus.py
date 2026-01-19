"""
AION Event Bus

State-of-the-art pub/sub messaging system for inter-process communication:
- Channel-based subscriptions with wildcard patterns
- Priority-based event delivery
- Request/response correlation with futures
- Event persistence and replay capabilities
- Dead letter queue for failed deliveries
- Comprehensive metrics and monitoring
- Backpressure handling
"""

from __future__ import annotations

import asyncio
import fnmatch
import uuid
import weakref
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Optional, Set, Dict, List
from heapq import heappush, heappop
import structlog

from aion.systems.process.models import Event

logger = structlog.get_logger(__name__)

EventHandler = Callable[[Event], Any]


@dataclass
class Subscription:
    """A subscription to an event channel."""
    id: str
    pattern: str  # Channel pattern (supports wildcards: process.*, agent.#)
    handler: EventHandler
    subscriber_id: Optional[str] = None  # Process ID of subscriber
    created_at: datetime = field(default_factory=datetime.now)
    event_count: int = 0
    last_event_at: Optional[datetime] = None
    filter_func: Optional[Callable[[Event], bool]] = None  # Additional filtering
    max_concurrent: int = 10  # Max concurrent handler calls
    _active_handlers: int = 0
    _semaphore: Optional[asyncio.Semaphore] = None

    def __post_init__(self):
        self._semaphore = asyncio.Semaphore(self.max_concurrent)


@dataclass
class DeadLetterEntry:
    """Entry in the dead letter queue."""
    event: Event
    subscription_id: str
    error: str
    timestamp: datetime = field(default_factory=datetime.now)
    retry_count: int = 0


@dataclass
class EventBusMetrics:
    """Metrics for the event bus."""
    events_emitted: int = 0
    events_delivered: int = 0
    events_failed: int = 0
    events_expired: int = 0
    events_replayed: int = 0
    subscriptions_created: int = 0
    subscriptions_removed: int = 0
    requests_sent: int = 0
    responses_received: int = 0
    response_timeouts: int = 0
    dead_letters: int = 0
    history_size: int = 0
    pending_requests: int = 0
    active_subscriptions: int = 0
    avg_delivery_latency_ms: float = 0.0
    _delivery_latencies: deque = field(default_factory=lambda: deque(maxlen=1000))

    def record_delivery(self, latency_ms: float) -> None:
        """Record a delivery latency."""
        self._delivery_latencies.append(latency_ms)
        if self._delivery_latencies:
            self.avg_delivery_latency_ms = sum(self._delivery_latencies) / len(self._delivery_latencies)

    def to_dict(self) -> dict[str, Any]:
        return {
            "events_emitted": self.events_emitted,
            "events_delivered": self.events_delivered,
            "events_failed": self.events_failed,
            "events_expired": self.events_expired,
            "events_replayed": self.events_replayed,
            "subscriptions_created": self.subscriptions_created,
            "subscriptions_removed": self.subscriptions_removed,
            "requests_sent": self.requests_sent,
            "responses_received": self.responses_received,
            "response_timeouts": self.response_timeouts,
            "dead_letters": self.dead_letters,
            "history_size": self.history_size,
            "pending_requests": self.pending_requests,
            "active_subscriptions": self.active_subscriptions,
            "avg_delivery_latency_ms": self.avg_delivery_latency_ms,
        }


class EventBus:
    """
    Central event bus for AION process communication.

    Features:
    - Pub/sub with glob-style pattern matching
    - Event history for replay (time-based and event-based)
    - Request/response with correlation and timeouts
    - Priority-based delivery
    - Dead letter queue for failed events
    - Backpressure with configurable queue limits
    - Async event handling with concurrency control
    """

    def __init__(
        self,
        max_history: int = 10000,
        max_dead_letters: int = 1000,
        enable_persistence: bool = False,
        default_ttl_seconds: Optional[int] = None,
        max_pending_requests: int = 1000,
    ):
        self.max_history = max_history
        self.max_dead_letters = max_dead_letters
        self.enable_persistence = enable_persistence
        self.default_ttl_seconds = default_ttl_seconds
        self.max_pending_requests = max_pending_requests

        # Subscriptions: pattern -> [subscriptions]
        self._subscriptions: Dict[str, List[Subscription]] = defaultdict(list)
        self._subscription_by_id: Dict[str, Subscription] = {}

        # Event history (ring buffer)
        self._event_history: deque[Event] = deque(maxlen=max_history)

        # Dead letter queue
        self._dead_letters: deque[DeadLetterEntry] = deque(maxlen=max_dead_letters)

        # Pending requests for request/response pattern
        self._pending_requests: Dict[str, asyncio.Future[Event]] = {}

        # Priority queue for events (heap: (priority, timestamp, event))
        self._priority_queue: List[tuple[int, datetime, Event]] = []
        self._queue_processor_task: Optional[asyncio.Task] = None

        # Metrics
        self._metrics = EventBusMetrics()

        # Lifecycle
        self._initialized = False
        self._shutdown_event = asyncio.Event()
        self._lock = asyncio.Lock()

        # Channel index for faster pattern matching
        self._channel_index: Dict[str, Set[str]] = defaultdict(set)

        # Weak references for subscriber cleanup
        self._subscriber_refs: weakref.WeakValueDictionary = weakref.WeakValueDictionary()

    async def initialize(self) -> None:
        """Initialize the event bus."""
        if self._initialized:
            return

        logger.info("Initializing Event Bus")

        # Start background tasks
        self._queue_processor_task = asyncio.create_task(self._process_queue())

        self._initialized = True
        logger.info("Event Bus initialized", max_history=self.max_history)

    async def shutdown(self) -> None:
        """Shutdown the event bus gracefully."""
        logger.info("Shutting down Event Bus")

        self._shutdown_event.set()

        # Stop queue processor
        if self._queue_processor_task:
            self._queue_processor_task.cancel()
            try:
                await self._queue_processor_task
            except asyncio.CancelledError:
                pass

        # Cancel pending requests
        for correlation_id, future in list(self._pending_requests.items()):
            if not future.done():
                future.cancel()
        self._pending_requests.clear()

        logger.info(
            "Event Bus shutdown complete",
            events_emitted=self._metrics.events_emitted,
            events_delivered=self._metrics.events_delivered,
        )

    async def emit(
        self,
        event: Event,
        wait_for_handlers: bool = False,
        use_priority_queue: bool = False,
    ) -> None:
        """
        Emit an event to all matching subscribers.

        Args:
            event: Event to emit
            wait_for_handlers: If True, wait for all handlers to complete
            use_priority_queue: If True, add to priority queue instead of immediate delivery
        """
        # Apply default TTL if not set
        if event.ttl_seconds is None and self.default_ttl_seconds:
            event.ttl_seconds = self.default_ttl_seconds

        # Check if expired
        if event.is_expired():
            self._metrics.events_expired += 1
            return

        self._metrics.events_emitted += 1

        # Store in history
        self._event_history.append(event)
        self._metrics.history_size = len(self._event_history)

        # Update channel index
        self._channel_index[event.type].add(event.id)

        if use_priority_queue:
            # Add to priority queue for ordered processing
            heappush(self._priority_queue, (-event.priority, event.timestamp, event))
        else:
            # Immediate delivery
            await self._deliver_event(event, wait_for_handlers)

        # Check for response to pending request
        if event.correlation_id and event.correlation_id in self._pending_requests:
            future = self._pending_requests.pop(event.correlation_id)
            if not future.done():
                future.set_result(event)
                self._metrics.responses_received += 1
                self._metrics.pending_requests = len(self._pending_requests)

    async def _deliver_event(self, event: Event, wait_for_handlers: bool = False) -> None:
        """Deliver an event to all matching subscribers."""
        start_time = datetime.now()
        handlers_to_call: List[tuple[Subscription, asyncio.Task]] = []

        async with self._lock:
            # Find matching subscriptions
            for pattern, subscriptions in self._subscriptions.items():
                if self._pattern_matches(pattern, event.type):
                    for sub in subscriptions:
                        # Apply subscription filter if present
                        if sub.filter_func and not sub.filter_func(event):
                            continue
                        handlers_to_call.append(sub)

        # Call handlers
        tasks = []
        for sub in handlers_to_call:
            task = asyncio.create_task(self._call_handler_safe(sub, event))
            tasks.append((sub, task))

        if wait_for_handlers and tasks:
            await asyncio.gather(*[t for _, t in tasks], return_exceptions=True)

        # Record latency
        latency_ms = (datetime.now() - start_time).total_seconds() * 1000
        self._metrics.record_delivery(latency_ms)

    async def _call_handler_safe(self, sub: Subscription, event: Event) -> None:
        """Safely call an event handler with error handling and metrics."""
        try:
            async with sub._semaphore:
                sub._active_handlers += 1
                try:
                    result = sub.handler(event)
                    if asyncio.iscoroutine(result):
                        await result

                    sub.event_count += 1
                    sub.last_event_at = datetime.now()
                    self._metrics.events_delivered += 1

                except Exception as e:
                    logger.error(
                        "Event handler failed",
                        event_type=event.type,
                        event_id=event.id,
                        subscription_id=sub.id,
                        error=str(e),
                    )
                    self._metrics.events_failed += 1

                    # Add to dead letter queue
                    self._dead_letters.append(DeadLetterEntry(
                        event=event,
                        subscription_id=sub.id,
                        error=str(e),
                    ))
                    self._metrics.dead_letters = len(self._dead_letters)

                finally:
                    sub._active_handlers -= 1

        except Exception as e:
            logger.error(f"Handler call error: {e}")

    async def _process_queue(self) -> None:
        """Background task to process priority queue."""
        while not self._shutdown_event.is_set():
            try:
                if self._priority_queue:
                    # Get highest priority event
                    _, _, event = heappop(self._priority_queue)

                    # Check if expired
                    if event.is_expired():
                        self._metrics.events_expired += 1
                        continue

                    await self._deliver_event(event)
                else:
                    await asyncio.sleep(0.01)  # Small sleep when queue empty

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Queue processor error: {e}")
                await asyncio.sleep(0.1)

    async def subscribe(
        self,
        pattern: str,
        handler: EventHandler,
        subscriber_id: Optional[str] = None,
        filter_func: Optional[Callable[[Event], bool]] = None,
        max_concurrent: int = 10,
    ) -> str:
        """
        Subscribe to events matching a pattern.

        Args:
            pattern: Channel pattern (supports wildcards: *, #, ?)
                - * matches exactly one segment
                - # matches zero or more segments
                - ? matches any single character
            handler: Callback function (async or sync)
            subscriber_id: Optional ID of subscribing process
            filter_func: Optional filter function for additional filtering
            max_concurrent: Max concurrent handler invocations

        Returns:
            Subscription ID
        """
        sub_id = str(uuid.uuid4())

        subscription = Subscription(
            id=sub_id,
            pattern=pattern,
            handler=handler,
            subscriber_id=subscriber_id,
            filter_func=filter_func,
            max_concurrent=max_concurrent,
        )

        async with self._lock:
            self._subscriptions[pattern].append(subscription)
            self._subscription_by_id[sub_id] = subscription

        self._metrics.subscriptions_created += 1
        self._metrics.active_subscriptions = len(self._subscription_by_id)

        logger.debug(
            "Created subscription",
            subscription_id=sub_id,
            pattern=pattern,
            subscriber_id=subscriber_id,
        )

        return sub_id

    async def unsubscribe(
        self,
        pattern: str,
        handler: Optional[EventHandler] = None,
        subscriber_id: Optional[str] = None,
    ) -> int:
        """
        Unsubscribe from a pattern.

        Args:
            pattern: Channel pattern
            handler: Specific handler to remove (or all if None)
            subscriber_id: Remove subscriptions for specific subscriber

        Returns:
            Number of subscriptions removed
        """
        removed = 0

        async with self._lock:
            if pattern not in self._subscriptions:
                return 0

            original_count = len(self._subscriptions[pattern])
            new_subscriptions = []

            for sub in self._subscriptions[pattern]:
                should_remove = False

                if handler and sub.handler == handler:
                    should_remove = True
                elif subscriber_id and sub.subscriber_id == subscriber_id:
                    should_remove = True
                elif handler is None and subscriber_id is None:
                    should_remove = True

                if should_remove:
                    self._subscription_by_id.pop(sub.id, None)
                    removed += 1
                else:
                    new_subscriptions.append(sub)

            self._subscriptions[pattern] = new_subscriptions

            if not self._subscriptions[pattern]:
                del self._subscriptions[pattern]

        self._metrics.subscriptions_removed += removed
        self._metrics.active_subscriptions = len(self._subscription_by_id)

        return removed

    async def unsubscribe_by_id(self, subscription_id: str) -> bool:
        """Unsubscribe by subscription ID."""
        async with self._lock:
            sub = self._subscription_by_id.pop(subscription_id, None)
            if not sub:
                return False

            self._subscriptions[sub.pattern] = [
                s for s in self._subscriptions[sub.pattern]
                if s.id != subscription_id
            ]

            if not self._subscriptions[sub.pattern]:
                del self._subscriptions[sub.pattern]

        self._metrics.subscriptions_removed += 1
        self._metrics.active_subscriptions = len(self._subscription_by_id)

        return True

    async def unsubscribe_all(self, subscriber_id: str) -> int:
        """Remove all subscriptions for a subscriber."""
        removed = 0

        async with self._lock:
            for pattern in list(self._subscriptions.keys()):
                original = self._subscriptions[pattern]
                filtered = [s for s in original if s.subscriber_id != subscriber_id]
                removed += len(original) - len(filtered)

                for sub in original:
                    if sub.subscriber_id == subscriber_id:
                        self._subscription_by_id.pop(sub.id, None)

                if filtered:
                    self._subscriptions[pattern] = filtered
                else:
                    del self._subscriptions[pattern]

        self._metrics.subscriptions_removed += removed
        self._metrics.active_subscriptions = len(self._subscription_by_id)

        return removed

    async def request(
        self,
        event: Event,
        timeout: float = 30.0,
    ) -> Optional[Event]:
        """
        Send a request and wait for a correlated response.

        Args:
            event: Request event
            timeout: Timeout in seconds

        Returns:
            Response event or None on timeout
        """
        if len(self._pending_requests) >= self.max_pending_requests:
            raise RuntimeError(f"Max pending requests ({self.max_pending_requests}) exceeded")

        # Ensure correlation ID
        if not event.correlation_id:
            event.correlation_id = str(uuid.uuid4())

        # Create future for response
        future: asyncio.Future[Event] = asyncio.get_event_loop().create_future()
        self._pending_requests[event.correlation_id] = future
        self._metrics.requests_sent += 1
        self._metrics.pending_requests = len(self._pending_requests)

        try:
            # Emit request
            await self.emit(event)

            # Wait for response
            return await asyncio.wait_for(future, timeout=timeout)

        except asyncio.TimeoutError:
            self._pending_requests.pop(event.correlation_id, None)
            self._metrics.response_timeouts += 1
            self._metrics.pending_requests = len(self._pending_requests)
            return None

        except asyncio.CancelledError:
            self._pending_requests.pop(event.correlation_id, None)
            self._metrics.pending_requests = len(self._pending_requests)
            raise

    def _pattern_matches(self, pattern: str, channel: str) -> bool:
        """
        Check if a pattern matches a channel.

        Supports:
        - Exact match: "process.started" matches "process.started"
        - Single wildcard (*): "process.*" matches "process.started", "process.stopped"
        - Multi-segment wildcard (#): "process.#" matches "process.started", "process.foo.bar"
        - Character wildcard (?): "process.st?rted" matches "process.started"
        """
        # Convert # to ** for fnmatch compatibility
        fnmatch_pattern = pattern.replace("#", "**")

        # Handle multi-segment wildcard
        if "**" in fnmatch_pattern:
            pattern_parts = fnmatch_pattern.split(".")
            channel_parts = channel.split(".")

            return self._match_segments(pattern_parts, channel_parts)

        return fnmatch.fnmatch(channel, fnmatch_pattern)

    def _match_segments(self, pattern_parts: list[str], channel_parts: list[str]) -> bool:
        """Match pattern segments with multi-segment wildcard support."""
        if not pattern_parts:
            return not channel_parts

        if pattern_parts[0] == "**":
            # ** can match zero or more segments
            if len(pattern_parts) == 1:
                return True  # ** at end matches everything

            # Try matching rest of pattern at each position
            for i in range(len(channel_parts) + 1):
                if self._match_segments(pattern_parts[1:], channel_parts[i:]):
                    return True
            return False

        if not channel_parts:
            return False

        if fnmatch.fnmatch(channel_parts[0], pattern_parts[0]):
            return self._match_segments(pattern_parts[1:], channel_parts[1:])

        return False

    async def replay(
        self,
        pattern: str,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        handler: Optional[EventHandler] = None,
        limit: Optional[int] = None,
    ) -> list[Event]:
        """
        Replay historical events.

        Args:
            pattern: Channel pattern to filter
            since: Only events after this time
            until: Only events before this time
            handler: If provided, call handler for each event
            limit: Max events to return

        Returns:
            List of matching events
        """
        events = []
        count = 0

        for event in self._event_history:
            # Time filters
            if since and event.timestamp < since:
                continue
            if until and event.timestamp > until:
                continue

            # Pattern filter
            if not self._pattern_matches(pattern, event.type):
                continue

            events.append(event)
            count += 1

            if handler:
                await self._call_handler_safe(
                    Subscription(
                        id="replay",
                        pattern=pattern,
                        handler=handler,
                    ),
                    event,
                )
                self._metrics.events_replayed += 1

            if limit and count >= limit:
                break

        return events

    def get_history(
        self,
        pattern: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Event]:
        """Get event history with pagination."""
        events = list(self._event_history)

        if pattern:
            events = [e for e in events if self._pattern_matches(pattern, e.type)]

        # Apply pagination
        events = events[-(offset + limit):]
        if offset > 0:
            events = events[:-offset]

        return events[-limit:]

    def get_dead_letters(self, limit: int = 100) -> list[DeadLetterEntry]:
        """Get dead letter queue entries."""
        return list(self._dead_letters)[-limit:]

    async def retry_dead_letter(self, index: int) -> bool:
        """Retry a dead letter entry."""
        if index >= len(self._dead_letters):
            return False

        entry = list(self._dead_letters)[index]
        entry.retry_count += 1

        await self.emit(entry.event)
        return True

    def get_subscription(self, subscription_id: str) -> Optional[Subscription]:
        """Get subscription by ID."""
        return self._subscription_by_id.get(subscription_id)

    def get_subscriptions_for_pattern(self, pattern: str) -> list[Subscription]:
        """Get all subscriptions for a pattern."""
        return self._subscriptions.get(pattern, []).copy()

    def get_all_patterns(self) -> list[str]:
        """Get all subscribed patterns."""
        return list(self._subscriptions.keys())

    def get_stats(self) -> dict[str, Any]:
        """Get event bus statistics."""
        return {
            **self._metrics.to_dict(),
            "patterns_subscribed": len(self._subscriptions),
            "queue_size": len(self._priority_queue),
        }

    def get_metrics(self) -> EventBusMetrics:
        """Get metrics object."""
        return self._metrics

    # Context manager support
    async def __aenter__(self) -> "EventBus":
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.shutdown()


class TypedEventBus(EventBus):
    """
    Typed event bus with strong typing for event payloads.
    Provides compile-time type checking for event data.
    """

    async def emit_typed(
        self,
        event_type: str,
        source: str,
        payload: Any,
        **kwargs,
    ) -> Event:
        """Emit a typed event with automatic serialization."""
        # Convert dataclass to dict if needed
        if hasattr(payload, "to_dict"):
            payload_dict = payload.to_dict()
        elif hasattr(payload, "__dataclass_fields__"):
            from dataclasses import asdict
            payload_dict = asdict(payload)
        else:
            payload_dict = payload if isinstance(payload, dict) else {"data": payload}

        event = Event(
            id=str(uuid.uuid4()),
            type=event_type,
            source=source,
            payload=payload_dict,
            **kwargs,
        )

        await self.emit(event)
        return event

    async def subscribe_typed(
        self,
        pattern: str,
        handler: Callable[[Event, Any], Any],
        payload_type: type,
        **kwargs,
    ) -> str:
        """Subscribe with automatic payload deserialization."""
        async def typed_handler(event: Event) -> None:
            # Deserialize payload
            if hasattr(payload_type, "from_dict"):
                payload = payload_type.from_dict(event.payload)
            else:
                payload = payload_type(**event.payload)

            result = handler(event, payload)
            if asyncio.iscoroutine(result):
                await result

        return await self.subscribe(pattern, typed_handler, **kwargs)
