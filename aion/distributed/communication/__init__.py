"""
AION Distributed Communication Layer

Provides the communication infrastructure for the AION distributed
computing system:

* :class:`RPCClient` -- async HTTP-based RPC client with connection
  pooling, circuit breakers, and retry logic.
* :class:`RPCServer` -- async HTTP-based RPC server with route
  handlers for all cluster operations.
* :class:`PubSubManager` -- in-process pub/sub event bus with
  wildcard topics, async delivery, and message history.
* :class:`MessageSerializer` -- JSON/msgpack serialization with
  checksum verification and transparent handling of complex types.
"""

from __future__ import annotations

from aion.distributed.communication.pubsub import (
    ALL_TOPICS,
    TOPIC_HEALTH_ALERT,
    TOPIC_LEADER_CHANGED,
    TOPIC_NODE_JOINED,
    TOPIC_NODE_LEFT,
    TOPIC_STATE_CHANGED,
    TOPIC_TASK_COMPLETED,
    PubSubManager,
    PubSubMessage,
)
from aion.distributed.communication.rpc import RPCClient
from aion.distributed.communication.serialization import MessageSerializer
from aion.distributed.communication.server import RPCServer

__all__ = [
    # Core classes
    "RPCClient",
    "RPCServer",
    "PubSubManager",
    "MessageSerializer",
    # PubSub helpers
    "PubSubMessage",
    "ALL_TOPICS",
    "TOPIC_NODE_JOINED",
    "TOPIC_NODE_LEFT",
    "TOPIC_LEADER_CHANGED",
    "TOPIC_TASK_COMPLETED",
    "TOPIC_STATE_CHANGED",
    "TOPIC_HEALTH_ALERT",
]
