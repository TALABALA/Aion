"""
AION Agent Message Bus

Inter-agent communication system supporting point-to-point messaging,
broadcasts, pub/sub topics, and request/response patterns.
"""

from __future__ import annotations

import asyncio
import uuid
from collections import defaultdict
from datetime import datetime
from typing import Any, Callable, Optional

import structlog

from aion.systems.agents.types import Message, MessageType

logger = structlog.get_logger(__name__)


MessageHandler = Callable[[Message], None]


class MessageBus:
    """
    Message bus for inter-agent communication.

    Features:
    - Point-to-point messaging
    - Broadcast messaging
    - Topic-based pub/sub
    - Request/response with correlation
    - Message history and replay
    - Priority queuing
    """

    def __init__(
        self,
        max_history: int = 1000,
        max_queue_size: int = 100,
    ):
        self.max_history = max_history
        self.max_queue_size = max_queue_size

        # Message queues per agent (priority queue simulation)
        self._queues: dict[str, asyncio.PriorityQueue] = defaultdict(
            lambda: asyncio.PriorityQueue(maxsize=max_queue_size)
        )

        # Topic subscriptions
        self._subscriptions: dict[str, set[str]] = defaultdict(set)  # topic -> agent_ids

        # Message handlers
        self._handlers: dict[str, list[MessageHandler]] = defaultdict(list)

        # Message history
        self._history: list[Message] = []

        # Pending request/response futures
        self._pending_responses: dict[str, asyncio.Future] = {}

        # Team membership tracking
        self._team_members: dict[str, set[str]] = defaultdict(set)  # team_id -> agent_ids

        # Statistics
        self._stats = {
            "messages_sent": 0,
            "messages_delivered": 0,
            "broadcasts": 0,
            "topic_publishes": 0,
            "requests": 0,
            "responses": 0,
            "timeouts": 0,
        }

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the message bus."""
        if self._initialized:
            return

        logger.info("Initializing Message Bus")
        self._initialized = True

    async def shutdown(self) -> None:
        """Shutdown the message bus."""
        logger.info("Shutting down Message Bus")

        # Cancel pending responses
        for future in self._pending_responses.values():
            if not future.done():
                future.cancel()

        self._pending_responses.clear()
        self._initialized = False

    # === Point-to-Point Messaging ===

    async def send(
        self,
        sender_id: str,
        recipient_id: str,
        message_type: MessageType,
        content: Any,
        subject: str = "",
        priority: int = 5,
        requires_response: bool = False,
        correlation_id: Optional[str] = None,
        team_id: Optional[str] = None,
    ) -> Message:
        """
        Send a message to a specific agent.

        Args:
            sender_id: Sender agent ID
            recipient_id: Recipient agent ID
            message_type: Type of message
            content: Message content
            subject: Message subject
            priority: Priority (1=highest, 10=lowest)
            requires_response: Whether response is expected
            correlation_id: For linking request/response
            team_id: Associated team ID

        Returns:
            The sent message
        """
        message = Message(
            sender_id=sender_id,
            recipient_id=recipient_id,
            message_type=message_type,
            content=content,
            subject=subject,
            priority=priority,
            requires_response=requires_response,
            correlation_id=correlation_id,
            team_id=team_id,
        )

        # Add to recipient's queue (priority, timestamp, message)
        queue = self._queues[recipient_id]
        try:
            queue.put_nowait((priority, message.created_at.timestamp(), message))
            message.delivered = True
            message.delivered_at = datetime.now()
            self._stats["messages_delivered"] += 1
        except asyncio.QueueFull:
            logger.warning(
                "Message queue full",
                recipient=recipient_id[:8],
                sender=sender_id[:8],
            )

        # Record history
        self._add_to_history(message)
        self._stats["messages_sent"] += 1

        # Invoke handlers
        await self._invoke_handlers(recipient_id, message)

        logger.debug(
            "Message sent",
            sender=sender_id[:8],
            recipient=recipient_id[:8],
            type=message_type.value,
        )

        return message

    async def broadcast(
        self,
        sender_id: str,
        content: Any,
        subject: str = "",
        team_id: Optional[str] = None,
        exclude: Optional[list[str]] = None,
        priority: int = 5,
    ) -> Message:
        """
        Broadcast a message to multiple agents.

        Args:
            sender_id: Sender agent ID
            content: Message content
            subject: Message subject
            team_id: Limit to team members (None for all)
            exclude: Agent IDs to exclude
            priority: Message priority

        Returns:
            The broadcast message
        """
        message = Message(
            sender_id=sender_id,
            recipient_id="",  # Empty for broadcast
            message_type=MessageType.BROADCAST,
            content=content,
            subject=subject,
            team_id=team_id,
            priority=priority,
        )

        exclude = set(exclude or [])
        exclude.add(sender_id)  # Don't send to self

        # Determine recipients
        if team_id:
            recipients = self._team_members.get(team_id, set()) - exclude
        else:
            recipients = set(self._queues.keys()) - exclude

        # Deliver to all recipients
        delivered_count = 0
        for recipient_id in recipients:
            queue = self._queues[recipient_id]
            try:
                queue.put_nowait((priority, message.created_at.timestamp(), message))
                delivered_count += 1
            except asyncio.QueueFull:
                pass

        self._stats["messages_delivered"] += delivered_count
        self._stats["broadcasts"] += 1

        self._add_to_history(message)

        logger.debug(
            "Broadcast sent",
            sender=sender_id[:8],
            recipients=delivered_count,
            team=team_id[:8] if team_id else None,
        )

        return message

    # === Request/Response ===

    async def request(
        self,
        sender_id: str,
        recipient_id: str,
        content: Any,
        subject: str = "",
        timeout: float = 30.0,
    ) -> Optional[Message]:
        """
        Send a request and wait for response.

        Args:
            sender_id: Sender agent ID
            recipient_id: Recipient agent ID
            content: Request content
            subject: Request subject
            timeout: Response timeout in seconds

        Returns:
            Response message or None on timeout
        """
        # Create correlation ID and future
        correlation_id = str(uuid.uuid4())
        future: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending_responses[correlation_id] = future

        # Send request
        await self.send(
            sender_id=sender_id,
            recipient_id=recipient_id,
            message_type=MessageType.QUERY,
            content=content,
            subject=subject,
            requires_response=True,
            correlation_id=correlation_id,
        )

        self._stats["requests"] += 1

        try:
            # Wait for response
            response = await asyncio.wait_for(future, timeout=timeout)
            return response
        except asyncio.TimeoutError:
            logger.warning(
                "Request timeout",
                sender=sender_id[:8],
                recipient=recipient_id[:8],
                correlation_id=correlation_id[:8],
            )
            self._stats["timeouts"] += 1
            return None
        except asyncio.CancelledError:
            return None
        finally:
            self._pending_responses.pop(correlation_id, None)

    async def respond(
        self,
        original_message: Message,
        content: Any,
        sender_id: str,
    ) -> Message:
        """
        Send a response to a request message.

        Args:
            original_message: The message being responded to
            content: Response content
            sender_id: Responder agent ID

        Returns:
            Response message
        """
        response = await self.send(
            sender_id=sender_id,
            recipient_id=original_message.sender_id,
            message_type=MessageType.RESPONSE,
            content=content,
            subject=f"Re: {original_message.subject}",
            correlation_id=original_message.correlation_id or original_message.id,
            team_id=original_message.team_id,
        )

        # Resolve pending future if exists
        correlation_id = original_message.correlation_id
        if correlation_id:
            future = self._pending_responses.get(correlation_id)
            if future and not future.done():
                future.set_result(response)
                self._stats["responses"] += 1

        return response

    # === Receiving Messages ===

    async def receive(
        self,
        agent_id: str,
        timeout: Optional[float] = None,
    ) -> Optional[Message]:
        """
        Receive next message for an agent.

        Args:
            agent_id: Agent ID
            timeout: Optional timeout in seconds

        Returns:
            Next message or None on timeout
        """
        queue = self._queues[agent_id]

        try:
            if timeout:
                _, _, message = await asyncio.wait_for(queue.get(), timeout=timeout)
            else:
                _, _, message = await queue.get()

            message.read = True
            message.read_at = datetime.now()
            return message
        except asyncio.TimeoutError:
            return None
        except asyncio.CancelledError:
            return None

    def receive_nowait(self, agent_id: str) -> Optional[Message]:
        """
        Receive next message without waiting.

        Args:
            agent_id: Agent ID

        Returns:
            Next message or None if queue empty
        """
        queue = self._queues[agent_id]

        try:
            _, _, message = queue.get_nowait()
            message.read = True
            message.read_at = datetime.now()
            return message
        except asyncio.QueueEmpty:
            return None

    def peek(self, agent_id: str, count: int = 10) -> list[Message]:
        """
        Peek at messages without removing them.

        Args:
            agent_id: Agent ID
            count: Max messages to peek

        Returns:
            List of messages
        """
        queue = self._queues[agent_id]

        # Access internal queue (implementation detail)
        messages = []
        try:
            items = list(queue._queue)[:count]
            messages = [item[2] for item in items]
        except Exception:
            pass

        return messages

    def pending_count(self, agent_id: str) -> int:
        """Get count of pending messages for an agent."""
        return self._queues[agent_id].qsize()

    # === Topic Subscriptions ===

    def subscribe(self, agent_id: str, topic: str) -> None:
        """Subscribe an agent to a topic."""
        self._subscriptions[topic].add(agent_id)
        logger.debug("Subscribed to topic", agent=agent_id[:8], topic=topic)

    def unsubscribe(self, agent_id: str, topic: str) -> None:
        """Unsubscribe an agent from a topic."""
        self._subscriptions[topic].discard(agent_id)
        logger.debug("Unsubscribed from topic", agent=agent_id[:8], topic=topic)

    def unsubscribe_all(self, agent_id: str) -> None:
        """Unsubscribe an agent from all topics."""
        for topic in self._subscriptions:
            self._subscriptions[topic].discard(agent_id)

    async def publish(
        self,
        sender_id: str,
        topic: str,
        content: Any,
        priority: int = 5,
    ) -> int:
        """
        Publish to a topic.

        Args:
            sender_id: Publisher agent ID
            topic: Topic to publish to
            content: Message content
            priority: Message priority

        Returns:
            Number of subscribers reached
        """
        subscribers = self._subscriptions.get(topic, set())

        delivered = 0
        for subscriber_id in subscribers:
            if subscriber_id != sender_id:
                await self.send(
                    sender_id=sender_id,
                    recipient_id=subscriber_id,
                    message_type=MessageType.BROADCAST,
                    content=content,
                    subject=f"[{topic}]",
                    priority=priority,
                )
                delivered += 1

        self._stats["topic_publishes"] += 1

        logger.debug("Published to topic", topic=topic, subscribers=delivered)

        return delivered

    def get_topic_subscribers(self, topic: str) -> list[str]:
        """Get list of subscribers for a topic."""
        return list(self._subscriptions.get(topic, set()))

    # === Team Management ===

    def register_team_member(self, team_id: str, agent_id: str) -> None:
        """Register an agent as a team member."""
        self._team_members[team_id].add(agent_id)

    def unregister_team_member(self, team_id: str, agent_id: str) -> None:
        """Unregister an agent from a team."""
        self._team_members[team_id].discard(agent_id)

    def disband_team(self, team_id: str) -> None:
        """Remove all team membership records."""
        self._team_members.pop(team_id, None)

    def get_team_members(self, team_id: str) -> list[str]:
        """Get list of team members."""
        return list(self._team_members.get(team_id, set()))

    # === Message Handlers ===

    def register_handler(
        self,
        agent_id: str,
        handler: MessageHandler,
    ) -> None:
        """Register a message handler for an agent."""
        self._handlers[agent_id].append(handler)

    def unregister_handlers(self, agent_id: str) -> None:
        """Unregister all handlers for an agent."""
        self._handlers.pop(agent_id, None)

    async def _invoke_handlers(self, agent_id: str, message: Message) -> None:
        """Invoke registered handlers for a message."""
        handlers = self._handlers.get(agent_id, [])
        for handler in handlers:
            try:
                result = handler(message)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.warning(
                    "Handler error",
                    agent=agent_id[:8],
                    error=str(e),
                )

    # === History ===

    def _add_to_history(self, message: Message) -> None:
        """Add message to history."""
        self._history.append(message)

        # Trim history if needed
        if len(self._history) > self.max_history:
            self._history = self._history[-self.max_history:]

    def get_history(
        self,
        agent_id: Optional[str] = None,
        team_id: Optional[str] = None,
        message_type: Optional[MessageType] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> list[Message]:
        """
        Get message history with filters.

        Args:
            agent_id: Filter by agent (sender or recipient)
            team_id: Filter by team
            message_type: Filter by message type
            since: Filter by timestamp
            limit: Maximum messages to return

        Returns:
            List of matching messages
        """
        messages = self._history

        if agent_id:
            messages = [
                m for m in messages
                if m.sender_id == agent_id or m.recipient_id == agent_id
            ]

        if team_id:
            messages = [m for m in messages if m.team_id == team_id]

        if message_type:
            messages = [m for m in messages if m.message_type == message_type]

        if since:
            messages = [m for m in messages if m.created_at >= since]

        return messages[-limit:]

    def clear_history(self) -> None:
        """Clear message history."""
        self._history.clear()

    # === Agent Cleanup ===

    def cleanup_agent(self, agent_id: str) -> None:
        """Clean up all resources for an agent."""
        # Remove from queues
        self._queues.pop(agent_id, None)

        # Remove from subscriptions
        self.unsubscribe_all(agent_id)

        # Remove handlers
        self.unregister_handlers(agent_id)

        # Remove from teams
        for team_id in list(self._team_members.keys()):
            self._team_members[team_id].discard(agent_id)

    # === Statistics ===

    def get_stats(self) -> dict[str, Any]:
        """Get message bus statistics."""
        return {
            **self._stats,
            "active_queues": len(self._queues),
            "history_size": len(self._history),
            "pending_responses": len(self._pending_responses),
            "teams": len(self._team_members),
            "topic_subscriptions": {
                topic: len(subs)
                for topic, subs in self._subscriptions.items()
            },
        }
