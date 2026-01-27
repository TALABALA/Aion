"""
AION Distributed Communication - Pub/Sub Manager

In-process publish/subscribe event bus for distributing internal cluster
events.  Supports exact topics, wildcard subscriptions (``"node.*"``),
bounded message history, and async delivery with per-subscriber error
isolation.
"""

from __future__ import annotations

import asyncio
import fnmatch
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Awaitable, Callable, Deque, Dict, List, Optional, Set, Union

import structlog

logger = structlog.get_logger(__name__)

# Type alias for subscriber callbacks (sync or async)
Callback = Union[Callable[..., None], Callable[..., Awaitable[None]]]

# Well-known topics
TOPIC_NODE_JOINED = "node.joined"
TOPIC_NODE_LEFT = "node.left"
TOPIC_LEADER_CHANGED = "leader.changed"
TOPIC_TASK_COMPLETED = "task.completed"
TOPIC_STATE_CHANGED = "state.changed"
TOPIC_HEALTH_ALERT = "health.alert"

ALL_TOPICS: List[str] = [
    TOPIC_NODE_JOINED,
    TOPIC_NODE_LEFT,
    TOPIC_LEADER_CHANGED,
    TOPIC_TASK_COMPLETED,
    TOPIC_STATE_CHANGED,
    TOPIC_HEALTH_ALERT,
]


# ---------------------------------------------------------------------------
# PubSub message wrapper
# ---------------------------------------------------------------------------


@dataclass
class PubSubMessage:
    """Envelope for a published message."""

    topic: str
    data: Any
    timestamp: datetime = field(default_factory=datetime.now)
    publisher: str = ""
    message_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic": self.topic,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "publisher": self.publisher,
            "message_id": self.message_id,
        }


# ---------------------------------------------------------------------------
# Subscription record
# ---------------------------------------------------------------------------


@dataclass
class _Subscription:
    """Internal representation of a subscription."""

    topic_pattern: str
    callback: Callback
    is_wildcard: bool = False
    created_at: float = field(default_factory=time.monotonic)
    delivery_count: int = 0
    error_count: int = 0


# ---------------------------------------------------------------------------
# PubSubManager
# ---------------------------------------------------------------------------


class PubSubManager:
    """
    In-process pub/sub event bus.

    * Exact and wildcard (``fnmatch``) topic subscriptions
    * Async and sync callbacks (sync callbacks are wrapped automatically)
    * Per-subscriber error isolation -- a failing callback never blocks others
    * Bounded message history per topic with configurable retention
    * ``publish`` for fire-and-forget (schedules onto the running loop)
    * ``publish_async`` for awaitable delivery (returns after all callbacks finish)
    """

    def __init__(
        self,
        *,
        history_size: int = 1000,
        max_history_per_topic: int = 200,
    ) -> None:
        self._subscriptions: Dict[str, List[_Subscription]] = defaultdict(list)
        self._wildcard_subscriptions: List[_Subscription] = []
        self._history: Dict[str, Deque[PubSubMessage]] = defaultdict(
            lambda: deque(maxlen=max_history_per_topic),
        )
        self._global_history: Deque[PubSubMessage] = deque(maxlen=history_size)
        self._max_history_per_topic = max_history_per_topic
        self._message_counter: int = 0
        self._logger = structlog.get_logger("aion.distributed.pubsub")
        self._stats: Dict[str, int] = defaultdict(int)

    # -- Subscribe / unsubscribe ---------------------------------------------

    def subscribe(self, topic: str, callback: Callback) -> None:
        """Register *callback* for messages on *topic*.

        *topic* may contain shell-style wildcards (e.g. ``"node.*"``).
        """
        is_wildcard = any(c in topic for c in ("*", "?", "["))
        sub = _Subscription(
            topic_pattern=topic,
            callback=callback,
            is_wildcard=is_wildcard,
        )
        if is_wildcard:
            # Avoid duplicate wildcard subscriptions
            for existing in self._wildcard_subscriptions:
                if existing.topic_pattern == topic and existing.callback is callback:
                    return
            self._wildcard_subscriptions.append(sub)
        else:
            # Avoid duplicate exact subscriptions
            for existing in self._subscriptions[topic]:
                if existing.callback is callback:
                    return
            self._subscriptions[topic].append(sub)

        self._logger.debug(
            "pubsub_subscribed",
            topic=topic,
            wildcard=is_wildcard,
        )

    def unsubscribe(self, topic: str, callback: Callback) -> None:
        """Remove *callback* from *topic*."""
        is_wildcard = any(c in topic for c in ("*", "?", "["))
        if is_wildcard:
            self._wildcard_subscriptions = [
                s for s in self._wildcard_subscriptions
                if not (s.topic_pattern == topic and s.callback is callback)
            ]
        else:
            self._subscriptions[topic] = [
                s for s in self._subscriptions[topic]
                if s.callback is not callback
            ]
        self._logger.debug("pubsub_unsubscribed", topic=topic)

    # -- Publish -------------------------------------------------------------

    def publish(self, topic: str, data: Any, *, publisher: str = "") -> None:
        """Fire-and-forget publish (schedules async delivery on the running loop)."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop -- fall back to synchronous delivery
            self._deliver_sync(topic, data, publisher)
            return

        loop.create_task(self.publish_async(topic, data, publisher=publisher))

    async def publish_async(
        self,
        topic: str,
        data: Any,
        *,
        publisher: str = "",
    ) -> None:
        """Publish and await delivery to **all** matching subscribers."""
        self._message_counter += 1
        msg = PubSubMessage(
            topic=topic,
            data=data,
            publisher=publisher,
            message_id=f"msg-{self._message_counter}",
        )

        # Record history
        self._history[topic].append(msg)
        self._global_history.append(msg)
        self._stats[topic] += 1

        # Collect matching subscriptions
        matching = list(self._subscriptions.get(topic, []))
        for sub in self._wildcard_subscriptions:
            if fnmatch.fnmatch(topic, sub.topic_pattern):
                matching.append(sub)

        if not matching:
            return

        # Deliver with error isolation
        tasks = [
            self._safe_deliver(sub, msg)
            for sub in matching
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

    # -- History / stats -----------------------------------------------------

    def get_history(
        self,
        topic: Optional[str] = None,
        limit: int = 50,
    ) -> List[PubSubMessage]:
        """Return recent messages, optionally filtered by *topic*."""
        if topic is not None:
            source: Deque[PubSubMessage] = self._history.get(topic, deque())
        else:
            source = self._global_history
        items = list(source)
        return items[-limit:]

    def get_stats(self) -> Dict[str, Any]:
        """Return publishing statistics."""
        return {
            "total_messages": self._message_counter,
            "topics": dict(self._stats),
            "subscription_counts": {
                topic: len(subs) for topic, subs in self._subscriptions.items()
            },
            "wildcard_subscriptions": len(self._wildcard_subscriptions),
        }

    def get_subscriber_count(self, topic: str) -> int:
        """Return the number of subscribers matching *topic*."""
        count = len(self._subscriptions.get(topic, []))
        for sub in self._wildcard_subscriptions:
            if fnmatch.fnmatch(topic, sub.topic_pattern):
                count += 1
        return count

    # -- Cleanup -------------------------------------------------------------

    def clear_history(self, topic: Optional[str] = None) -> None:
        """Clear message history."""
        if topic is not None:
            self._history.pop(topic, None)
        else:
            self._history.clear()
            self._global_history.clear()

    def clear_all(self) -> None:
        """Remove all subscriptions and history."""
        self._subscriptions.clear()
        self._wildcard_subscriptions.clear()
        self._history.clear()
        self._global_history.clear()
        self._message_counter = 0
        self._stats.clear()

    # -- Internal delivery ---------------------------------------------------

    async def _safe_deliver(self, sub: _Subscription, msg: PubSubMessage) -> None:
        """Deliver *msg* to *sub* with error isolation."""
        try:
            result = sub.callback(msg.topic, msg.data)
            if asyncio.iscoroutine(result) or asyncio.isfuture(result):
                await result
            sub.delivery_count += 1
        except Exception:
            sub.error_count += 1
            self._logger.warning(
                "pubsub_delivery_error",
                topic=msg.topic,
                subscriber=sub.topic_pattern,
                error_count=sub.error_count,
                exc_info=True,
            )

    def _deliver_sync(self, topic: str, data: Any, publisher: str) -> None:
        """Synchronous fallback when no event loop is running."""
        self._message_counter += 1
        msg = PubSubMessage(
            topic=topic,
            data=data,
            publisher=publisher,
            message_id=f"msg-{self._message_counter}",
        )
        self._history[topic].append(msg)
        self._global_history.append(msg)
        self._stats[topic] += 1

        matching = list(self._subscriptions.get(topic, []))
        for sub in self._wildcard_subscriptions:
            if fnmatch.fnmatch(topic, sub.topic_pattern):
                matching.append(sub)

        for sub in matching:
            try:
                result = sub.callback(topic, data)
                if asyncio.iscoroutine(result):
                    # Cannot await in sync context -- close the coroutine to avoid warning
                    result.close()
                    self._logger.warning(
                        "pubsub_async_callback_in_sync_context",
                        topic=topic,
                    )
                sub.delivery_count += 1
            except Exception:
                sub.error_count += 1
                self._logger.warning(
                    "pubsub_delivery_error",
                    topic=topic,
                    subscriber=sub.topic_pattern,
                    exc_info=True,
                )
