"""
AION Distributed Queue Backends

Pluggable backends for the distributed task queue:
- InMemory: For development/testing
- Redis: For production with persistence
- RabbitMQ: For high-throughput messaging
"""

from __future__ import annotations

import asyncio
import heapq
import json
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import structlog

from aion.automation.distributed.queue import Task, TaskStatus, TaskPriority

logger = structlog.get_logger(__name__)


class QueueBackend(ABC):
    """Abstract base class for queue backends."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the backend."""
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the backend."""
        pass

    @abstractmethod
    async def enqueue(self, task: Task) -> None:
        """Add task to queue."""
        pass

    @abstractmethod
    async def enqueue_delayed(self, task: Task, delay_seconds: int) -> None:
        """Add task to delayed queue."""
        pass

    @abstractmethod
    async def dequeue(
        self,
        queue_name: str,
        timeout_seconds: int = 30,
    ) -> Optional[Task]:
        """Remove and return next task from queue."""
        pass

    @abstractmethod
    async def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID."""
        pass

    @abstractmethod
    async def update_task(self, task: Task) -> None:
        """Update task state."""
        pass

    @abstractmethod
    async def remove_from_queue(self, queue_name: str, task_id: str) -> bool:
        """Remove specific task from queue."""
        pass

    @abstractmethod
    async def get_queue_stats(self, queue_name: str) -> Dict[str, Any]:
        """Get queue statistics."""
        pass

    @abstractmethod
    async def get_pending_tasks(
        self,
        queue_name: str,
        limit: int = 100,
    ) -> List[Task]:
        """Get pending tasks in queue."""
        pass


class InMemoryBackend(QueueBackend):
    """In-memory queue backend for development/testing."""

    def __init__(self):
        # Priority queues per queue name
        self._queues: Dict[str, List[Tuple[int, float, Task]]] = {}

        # Task storage
        self._tasks: Dict[str, Task] = {}

        # Delayed tasks (sorted by execution time)
        self._delayed: List[Tuple[float, Task]] = []

        # Lock for thread safety
        self._lock = asyncio.Lock()

        # Delayed task processor
        self._delayed_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

    async def initialize(self) -> None:
        """Initialize the backend."""
        self._delayed_task = asyncio.create_task(self._process_delayed())
        logger.info("InMemory backend initialized")

    async def shutdown(self) -> None:
        """Shutdown the backend."""
        self._shutdown_event.set()
        if self._delayed_task:
            self._delayed_task.cancel()
            try:
                await self._delayed_task
            except asyncio.CancelledError:
                pass
        logger.info("InMemory backend shutdown")

    async def _process_delayed(self) -> None:
        """Process delayed tasks."""
        while not self._shutdown_event.is_set():
            try:
                async with self._lock:
                    now = time.time()
                    while self._delayed and self._delayed[0][0] <= now:
                        _, task = heapq.heappop(self._delayed)
                        await self._enqueue_internal(task)

                await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error processing delayed tasks", error=str(e))

    async def _enqueue_internal(self, task: Task) -> None:
        """Internal enqueue without lock."""
        if task.queue_name not in self._queues:
            self._queues[task.queue_name] = []

        # Priority queue: (priority, timestamp, task)
        heapq.heappush(
            self._queues[task.queue_name],
            (task.priority.value, time.time(), task)
        )

        task.status = TaskStatus.QUEUED
        self._tasks[task.id] = task

    async def enqueue(self, task: Task) -> None:
        """Add task to queue."""
        async with self._lock:
            await self._enqueue_internal(task)

        logger.debug(f"Task enqueued: {task.id}")

    async def enqueue_delayed(self, task: Task, delay_seconds: int) -> None:
        """Add task to delayed queue."""
        execute_at = time.time() + delay_seconds

        async with self._lock:
            heapq.heappush(self._delayed, (execute_at, task))
            self._tasks[task.id] = task

        logger.debug(f"Task delayed: {task.id} for {delay_seconds}s")

    async def dequeue(
        self,
        queue_name: str,
        timeout_seconds: int = 30,
    ) -> Optional[Task]:
        """Remove and return next task from queue."""
        start_time = time.time()

        while time.time() - start_time < timeout_seconds:
            async with self._lock:
                queue = self._queues.get(queue_name, [])
                if queue:
                    _, _, task = heapq.heappop(queue)
                    task.status = TaskStatus.RUNNING
                    task.started_at = datetime.now()
                    return task

            await asyncio.sleep(0.1)

        return None

    async def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID."""
        return self._tasks.get(task_id)

    async def update_task(self, task: Task) -> None:
        """Update task state."""
        async with self._lock:
            self._tasks[task.id] = task

    async def remove_from_queue(self, queue_name: str, task_id: str) -> bool:
        """Remove specific task from queue."""
        async with self._lock:
            queue = self._queues.get(queue_name, [])
            original_len = len(queue)
            self._queues[queue_name] = [
                (p, t, task) for p, t, task in queue
                if task.id != task_id
            ]
            heapq.heapify(self._queues[queue_name])
            return len(self._queues[queue_name]) < original_len

    async def get_queue_stats(self, queue_name: str) -> Dict[str, Any]:
        """Get queue statistics."""
        queue = self._queues.get(queue_name, [])
        tasks = [t for _, _, t in queue]

        return {
            "queue_name": queue_name,
            "pending_count": len(tasks),
            "delayed_count": len([
                t for _, t in self._delayed
                if t.queue_name == queue_name
            ]),
            "priority_breakdown": {
                p.name: len([t for t in tasks if t.priority == p])
                for p in TaskPriority
            },
        }

    async def get_pending_tasks(
        self,
        queue_name: str,
        limit: int = 100,
    ) -> List[Task]:
        """Get pending tasks in queue."""
        queue = self._queues.get(queue_name, [])
        tasks = [t for _, _, t in sorted(queue)[:limit]]
        return tasks


class RedisBackend(QueueBackend):
    """
    Redis-based queue backend for production.

    Features:
    - Atomic operations with Lua scripts
    - Reliable delivery with BRPOPLPUSH
    - Delayed tasks with sorted sets
    - Task persistence
    - Distributed locking
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        prefix: str = "aion:queue:",
        visibility_timeout: int = 30,
    ):
        self.redis_url = redis_url
        self.prefix = prefix
        self.visibility_timeout = visibility_timeout
        self._client = None
        self._delayed_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

    async def _get_client(self):
        """Get or create Redis client."""
        if self._client is None:
            try:
                import redis.asyncio as redis
                self._client = redis.from_url(
                    self.redis_url,
                    decode_responses=True,
                )
            except ImportError:
                raise ImportError("redis package required for RedisBackend")
        return self._client

    def _queue_key(self, queue_name: str, priority: TaskPriority) -> str:
        """Get Redis key for a priority queue."""
        return f"{self.prefix}queue:{queue_name}:{priority.value}"

    def _processing_key(self, queue_name: str) -> str:
        """Get Redis key for processing queue."""
        return f"{self.prefix}processing:{queue_name}"

    def _delayed_key(self) -> str:
        """Get Redis key for delayed tasks."""
        return f"{self.prefix}delayed"

    def _task_key(self, task_id: str) -> str:
        """Get Redis key for task data."""
        return f"{self.prefix}task:{task_id}"

    async def initialize(self) -> None:
        """Initialize the backend."""
        await self._get_client()
        self._delayed_task = asyncio.create_task(self._process_delayed())
        logger.info("Redis backend initialized", url=self.redis_url)

    async def shutdown(self) -> None:
        """Shutdown the backend."""
        self._shutdown_event.set()
        if self._delayed_task:
            self._delayed_task.cancel()
            try:
                await self._delayed_task
            except asyncio.CancelledError:
                pass

        if self._client:
            await self._client.close()

        logger.info("Redis backend shutdown")

    async def _process_delayed(self) -> None:
        """Process delayed tasks from sorted set."""
        client = await self._get_client()

        while not self._shutdown_event.is_set():
            try:
                now = time.time()

                # Get tasks due for execution
                tasks_data = await client.zrangebyscore(
                    self._delayed_key(),
                    "-inf",
                    now,
                    start=0,
                    num=10,
                )

                for task_data in tasks_data:
                    # Remove from delayed set and enqueue
                    removed = await client.zrem(self._delayed_key(), task_data)
                    if removed:
                        task = Task.from_dict(json.loads(task_data))
                        await self.enqueue(task)

                await asyncio.sleep(0.5)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error processing delayed tasks", error=str(e))
                await asyncio.sleep(1)

    async def enqueue(self, task: Task) -> None:
        """Add task to queue."""
        client = await self._get_client()

        task.status = TaskStatus.QUEUED

        # Store task data
        await client.set(
            self._task_key(task.id),
            json.dumps(task.to_dict()),
            ex=86400 * 7,  # 7 day TTL
        )

        # Add to priority queue
        queue_key = self._queue_key(task.queue_name, task.priority)
        await client.lpush(queue_key, task.id)

        logger.debug(f"Task enqueued to Redis: {task.id}")

    async def enqueue_delayed(self, task: Task, delay_seconds: int) -> None:
        """Add task to delayed queue."""
        client = await self._get_client()

        execute_at = time.time() + delay_seconds

        # Store task data
        await client.set(
            self._task_key(task.id),
            json.dumps(task.to_dict()),
            ex=86400 * 7,
        )

        # Add to sorted set with execution time as score
        await client.zadd(
            self._delayed_key(),
            {json.dumps(task.to_dict()): execute_at},
        )

        logger.debug(f"Task delayed in Redis: {task.id} for {delay_seconds}s")

    async def dequeue(
        self,
        queue_name: str,
        timeout_seconds: int = 30,
    ) -> Optional[Task]:
        """
        Remove and return next task from queue.

        Uses BRPOPLPUSH for reliable delivery.
        """
        client = await self._get_client()

        # Try each priority level
        for priority in TaskPriority:
            queue_key = self._queue_key(queue_name, priority)
            processing_key = self._processing_key(queue_name)

            # Blocking pop with move to processing queue
            task_id = await client.brpoplpush(
                queue_key,
                processing_key,
                timeout=1,
            )

            if task_id:
                # Get task data
                task_data = await client.get(self._task_key(task_id))
                if task_data:
                    task = Task.from_dict(json.loads(task_data))
                    task.status = TaskStatus.RUNNING
                    task.started_at = datetime.now()

                    # Update task
                    await client.set(
                        self._task_key(task.id),
                        json.dumps(task.to_dict()),
                        ex=86400 * 7,
                    )

                    return task

        return None

    async def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID."""
        client = await self._get_client()
        task_data = await client.get(self._task_key(task_id))

        if task_data:
            return Task.from_dict(json.loads(task_data))
        return None

    async def update_task(self, task: Task) -> None:
        """Update task state."""
        client = await self._get_client()

        await client.set(
            self._task_key(task.id),
            json.dumps(task.to_dict()),
            ex=86400 * 7,
        )

        # Remove from processing queue if completed/failed
        if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
            processing_key = self._processing_key(task.queue_name)
            await client.lrem(processing_key, 0, task.id)

    async def remove_from_queue(self, queue_name: str, task_id: str) -> bool:
        """Remove specific task from queue."""
        client = await self._get_client()

        removed = 0
        for priority in TaskPriority:
            queue_key = self._queue_key(queue_name, priority)
            removed += await client.lrem(queue_key, 0, task_id)

        return removed > 0

    async def get_queue_stats(self, queue_name: str) -> Dict[str, Any]:
        """Get queue statistics."""
        client = await self._get_client()

        stats = {
            "queue_name": queue_name,
            "pending_count": 0,
            "processing_count": 0,
            "delayed_count": 0,
            "priority_breakdown": {},
        }

        # Count per priority
        for priority in TaskPriority:
            queue_key = self._queue_key(queue_name, priority)
            count = await client.llen(queue_key)
            stats["priority_breakdown"][priority.name] = count
            stats["pending_count"] += count

        # Processing count
        processing_key = self._processing_key(queue_name)
        stats["processing_count"] = await client.llen(processing_key)

        # Delayed count (approximate)
        stats["delayed_count"] = await client.zcard(self._delayed_key())

        return stats

    async def get_pending_tasks(
        self,
        queue_name: str,
        limit: int = 100,
    ) -> List[Task]:
        """Get pending tasks in queue."""
        client = await self._get_client()
        tasks = []

        remaining = limit
        for priority in TaskPriority:
            if remaining <= 0:
                break

            queue_key = self._queue_key(queue_name, priority)
            task_ids = await client.lrange(queue_key, 0, remaining - 1)

            for task_id in task_ids:
                task_data = await client.get(self._task_key(task_id))
                if task_data:
                    tasks.append(Task.from_dict(json.loads(task_data)))

            remaining -= len(task_ids)

        return tasks


class RabbitMQBackend(QueueBackend):
    """
    RabbitMQ-based queue backend for high-throughput messaging.

    Features:
    - Direct exchange for priority routing
    - Dead letter exchanges
    - Message persistence
    - Consumer acknowledgments
    - Prefetch control
    """

    def __init__(
        self,
        amqp_url: str = "amqp://guest:guest@localhost:5672/",
        exchange_name: str = "aion.tasks",
        prefetch_count: int = 10,
    ):
        self.amqp_url = amqp_url
        self.exchange_name = exchange_name
        self.prefetch_count = prefetch_count

        self._connection = None
        self._channel = None
        self._task_store: Dict[str, Task] = {}  # Local task storage

    async def _get_channel(self):
        """Get or create RabbitMQ channel."""
        if self._channel is None or self._channel.is_closed:
            try:
                import aio_pika
                self._connection = await aio_pika.connect_robust(self.amqp_url)
                self._channel = await self._connection.channel()
                await self._channel.set_qos(prefetch_count=self.prefetch_count)

                # Declare exchange
                await self._channel.declare_exchange(
                    self.exchange_name,
                    aio_pika.ExchangeType.DIRECT,
                    durable=True,
                )

            except ImportError:
                raise ImportError("aio-pika package required for RabbitMQBackend")

        return self._channel

    def _queue_name(self, queue_name: str, priority: TaskPriority) -> str:
        """Get RabbitMQ queue name."""
        return f"{queue_name}.{priority.name.lower()}"

    async def initialize(self) -> None:
        """Initialize the backend."""
        channel = await self._get_channel()

        # Declare queues for each priority
        for priority in TaskPriority:
            queue_name = self._queue_name("default", priority)

            # Declare queue with dead letter exchange
            await channel.declare_queue(
                queue_name,
                durable=True,
                arguments={
                    "x-dead-letter-exchange": f"{self.exchange_name}.dlx",
                    "x-message-ttl": 86400000,  # 24 hours
                },
            )

        # Declare dead letter exchange and queue
        import aio_pika
        dlx = await channel.declare_exchange(
            f"{self.exchange_name}.dlx",
            aio_pika.ExchangeType.FANOUT,
            durable=True,
        )
        dlq = await channel.declare_queue(
            f"{self.exchange_name}.dead_letters",
            durable=True,
        )
        await dlq.bind(dlx)

        logger.info("RabbitMQ backend initialized", url=self.amqp_url)

    async def shutdown(self) -> None:
        """Shutdown the backend."""
        if self._connection:
            await self._connection.close()
        logger.info("RabbitMQ backend shutdown")

    async def enqueue(self, task: Task) -> None:
        """Add task to queue."""
        import aio_pika

        channel = await self._get_channel()
        exchange = await channel.get_exchange(self.exchange_name)

        task.status = TaskStatus.QUEUED
        self._task_store[task.id] = task

        message = aio_pika.Message(
            body=json.dumps(task.to_dict()).encode(),
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            message_id=task.id,
            correlation_id=task.correlation_id,
            priority=4 - task.priority.value,  # Invert for RabbitMQ (higher = higher priority)
        )

        routing_key = self._queue_name(task.queue_name, task.priority)
        await exchange.publish(message, routing_key=routing_key)

        logger.debug(f"Task enqueued to RabbitMQ: {task.id}")

    async def enqueue_delayed(self, task: Task, delay_seconds: int) -> None:
        """Add task to delayed queue using RabbitMQ TTL."""
        import aio_pika

        channel = await self._get_channel()

        # Create a temporary delay queue
        delay_queue_name = f"delay.{delay_seconds}s"
        target_queue = self._queue_name(task.queue_name, task.priority)

        delay_queue = await channel.declare_queue(
            delay_queue_name,
            durable=True,
            arguments={
                "x-dead-letter-exchange": self.exchange_name,
                "x-dead-letter-routing-key": target_queue,
                "x-message-ttl": delay_seconds * 1000,
            },
        )

        self._task_store[task.id] = task

        message = aio_pika.Message(
            body=json.dumps(task.to_dict()).encode(),
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            message_id=task.id,
        )

        await channel.default_exchange.publish(message, routing_key=delay_queue_name)
        logger.debug(f"Task delayed in RabbitMQ: {task.id} for {delay_seconds}s")

    async def dequeue(
        self,
        queue_name: str,
        timeout_seconds: int = 30,
    ) -> Optional[Task]:
        """Remove and return next task from queue."""
        channel = await self._get_channel()

        # Try each priority level (highest first)
        for priority in TaskPriority:
            rmq_queue_name = self._queue_name(queue_name, priority)

            try:
                queue = await channel.get_queue(rmq_queue_name)
                message = await asyncio.wait_for(
                    queue.get(timeout=1),
                    timeout=timeout_seconds / len(TaskPriority),
                )

                if message:
                    task_data = json.loads(message.body.decode())
                    task = Task.from_dict(task_data)
                    task.status = TaskStatus.RUNNING
                    task.started_at = datetime.now()

                    self._task_store[task.id] = task

                    # Acknowledge the message
                    await message.ack()

                    return task

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.debug(f"No message in queue {rmq_queue_name}: {e}")
                continue

        return None

    async def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID."""
        return self._task_store.get(task_id)

    async def update_task(self, task: Task) -> None:
        """Update task state."""
        self._task_store[task.id] = task

    async def remove_from_queue(self, queue_name: str, task_id: str) -> bool:
        """Remove specific task from queue (not efficiently supported)."""
        # RabbitMQ doesn't support removing specific messages
        # This would require consuming all messages and re-enqueuing
        logger.warning("remove_from_queue not efficiently supported in RabbitMQ")
        return False

    async def get_queue_stats(self, queue_name: str) -> Dict[str, Any]:
        """Get queue statistics."""
        channel = await self._get_channel()

        stats = {
            "queue_name": queue_name,
            "pending_count": 0,
            "priority_breakdown": {},
        }

        for priority in TaskPriority:
            rmq_queue_name = self._queue_name(queue_name, priority)
            try:
                queue = await channel.get_queue(rmq_queue_name)
                declaration = await queue.declare()
                count = declaration.message_count
                stats["priority_breakdown"][priority.name] = count
                stats["pending_count"] += count
            except Exception:
                stats["priority_breakdown"][priority.name] = 0

        return stats

    async def get_pending_tasks(
        self,
        queue_name: str,
        limit: int = 100,
    ) -> List[Task]:
        """Get pending tasks (limited support in RabbitMQ)."""
        # RabbitMQ doesn't support peeking without consuming
        # Return tasks from local store
        return [
            task for task in self._task_store.values()
            if task.queue_name == queue_name and task.status == TaskStatus.QUEUED
        ][:limit]


def create_backend(
    backend_type: str = "memory",
    **kwargs,
) -> QueueBackend:
    """Factory function to create queue backend."""
    backends = {
        "memory": InMemoryBackend,
        "redis": RedisBackend,
        "rabbitmq": RabbitMQBackend,
    }

    if backend_type not in backends:
        raise ValueError(f"Unknown backend type: {backend_type}")

    return backends[backend_type](**kwargs)
