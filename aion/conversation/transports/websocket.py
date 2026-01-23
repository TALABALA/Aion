"""
AION Conversation WebSocket

Real-time WebSocket interface for conversations.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Any, Optional
import uuid

from fastapi import WebSocket, WebSocketDisconnect, APIRouter
import structlog

from aion.conversation.manager import ConversationManager
from aion.conversation.types import ConversationRequest, StreamEvent

logger = structlog.get_logger(__name__)


class ConnectionManager:
    """
    Manages WebSocket connections.

    Handles:
    - Connection tracking
    - Broadcasting
    - Connection cleanup
    """

    def __init__(self, max_connections: int = 1000):
        self.max_connections = max_connections
        self._active_connections: dict[str, WebSocket] = {}
        self._connection_info: dict[str, dict[str, Any]] = {}
        self._conversation_connections: dict[str, set[str]] = {}

    async def connect(
        self,
        websocket: WebSocket,
        connection_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> str:
        """Accept a new WebSocket connection."""
        if len(self._active_connections) >= self.max_connections:
            await websocket.close(code=1013, reason="Max connections reached")
            raise RuntimeError("Max connections reached")

        await websocket.accept()

        connection_id = connection_id or str(uuid.uuid4())
        self._active_connections[connection_id] = websocket
        self._connection_info[connection_id] = {
            "user_id": user_id,
            "connected_at": datetime.now().isoformat(),
            "message_count": 0,
        }

        logger.info(
            "WebSocket connected",
            connection_id=connection_id[:8],
            user_id=user_id,
        )

        return connection_id

    def disconnect(self, connection_id: str) -> None:
        """Disconnect a WebSocket connection."""
        self._active_connections.pop(connection_id, None)
        self._connection_info.pop(connection_id, None)

        for conv_id, connections in self._conversation_connections.items():
            connections.discard(connection_id)

        logger.info("WebSocket disconnected", connection_id=connection_id[:8])

    async def send_json(self, connection_id: str, data: dict) -> bool:
        """Send JSON data to a specific connection."""
        websocket = self._active_connections.get(connection_id)
        if not websocket:
            return False

        try:
            await websocket.send_json(data)
            if connection_id in self._connection_info:
                self._connection_info[connection_id]["message_count"] += 1
            return True
        except Exception as e:
            logger.warning(f"Failed to send to {connection_id[:8]}: {e}")
            return False

    async def broadcast(self, data: dict) -> int:
        """Broadcast to all connections."""
        sent_count = 0
        for connection_id in list(self._active_connections.keys()):
            if await self.send_json(connection_id, data):
                sent_count += 1
        return sent_count

    async def broadcast_to_conversation(
        self,
        conversation_id: str,
        data: dict,
    ) -> int:
        """Broadcast to all connections subscribed to a conversation."""
        connections = self._conversation_connections.get(conversation_id, set())
        sent_count = 0

        for connection_id in connections:
            if await self.send_json(connection_id, data):
                sent_count += 1

        return sent_count

    def subscribe_to_conversation(
        self,
        connection_id: str,
        conversation_id: str,
    ) -> None:
        """Subscribe a connection to a conversation."""
        if conversation_id not in self._conversation_connections:
            self._conversation_connections[conversation_id] = set()
        self._conversation_connections[conversation_id].add(connection_id)

    def unsubscribe_from_conversation(
        self,
        connection_id: str,
        conversation_id: str,
    ) -> None:
        """Unsubscribe a connection from a conversation."""
        if conversation_id in self._conversation_connections:
            self._conversation_connections[conversation_id].discard(connection_id)

    def get_stats(self) -> dict[str, Any]:
        """Get connection statistics."""
        return {
            "active_connections": len(self._active_connections),
            "max_connections": self.max_connections,
            "conversation_subscriptions": {
                conv_id: len(connections)
                for conv_id, connections in self._conversation_connections.items()
            },
        }


class ConversationWebSocket:
    """
    WebSocket handler for real-time conversations.

    Protocol:
    Client sends:
    - {"type": "message", "text": "...", "conversation_id": "..."}
    - {"type": "subscribe", "conversation_id": "..."}
    - {"type": "unsubscribe", "conversation_id": "..."}
    - {"type": "ping"}
    - {"type": "get_history", "conversation_id": "..."}

    Server sends:
    - {"type": "text", "data": "..."}
    - {"type": "tool_use_start", "data": {"name": "..."}}
    - {"type": "tool_result", "data": {...}}
    - {"type": "done", "data": {...}}
    - {"type": "error", "data": "..."}
    - {"type": "pong"}
    - {"type": "history", "data": {...}}
    - {"type": "subscribed", "conversation_id": "..."}
    """

    def __init__(
        self,
        manager: ConversationManager,
        connection_manager: Optional[ConnectionManager] = None,
    ):
        self.manager = manager
        self.connections = connection_manager or ConnectionManager()

    async def handle(
        self,
        websocket: WebSocket,
        conversation_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ):
        """Handle a WebSocket connection."""
        connection_id = await self.connections.connect(
            websocket,
            user_id=user_id,
        )

        try:
            if conversation_id:
                conversation = await self.manager.get_conversation(conversation_id)
                if conversation:
                    self.connections.subscribe_to_conversation(
                        connection_id,
                        conversation_id,
                    )
                    await websocket.send_json({
                        "type": "conversation_loaded",
                        "data": conversation.to_dict(),
                    })
                else:
                    await websocket.send_json({
                        "type": "error",
                        "data": "Conversation not found",
                    })

            while True:
                try:
                    data = await websocket.receive_json()
                    await self._handle_message(
                        websocket,
                        connection_id,
                        data,
                        conversation_id,
                        user_id,
                    )
                except WebSocketDisconnect:
                    break
                except json.JSONDecodeError:
                    await websocket.send_json({
                        "type": "error",
                        "data": "Invalid JSON",
                    })

        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            try:
                await websocket.send_json({
                    "type": "error",
                    "data": str(e),
                })
            except Exception:
                pass

        finally:
            self.connections.disconnect(connection_id)

    async def _handle_message(
        self,
        websocket: WebSocket,
        connection_id: str,
        data: dict,
        default_conversation_id: Optional[str],
        user_id: Optional[str],
    ):
        """Handle an incoming WebSocket message."""
        msg_type = data.get("type")

        if msg_type == "message":
            await self._handle_chat_message(
                websocket,
                connection_id,
                data,
                default_conversation_id,
                user_id,
            )

        elif msg_type == "ping":
            await websocket.send_json({"type": "pong"})

        elif msg_type == "subscribe":
            conv_id = data.get("conversation_id")
            if conv_id:
                self.connections.subscribe_to_conversation(connection_id, conv_id)
                await websocket.send_json({
                    "type": "subscribed",
                    "conversation_id": conv_id,
                })

        elif msg_type == "unsubscribe":
            conv_id = data.get("conversation_id")
            if conv_id:
                self.connections.unsubscribe_from_conversation(connection_id, conv_id)
                await websocket.send_json({
                    "type": "unsubscribed",
                    "conversation_id": conv_id,
                })

        elif msg_type == "get_history":
            await self._handle_get_history(
                websocket,
                data,
                default_conversation_id,
            )

        elif msg_type == "create_conversation":
            await self._handle_create_conversation(
                websocket,
                connection_id,
                data,
                user_id,
            )

        else:
            await websocket.send_json({
                "type": "error",
                "data": f"Unknown message type: {msg_type}",
            })

    async def _handle_chat_message(
        self,
        websocket: WebSocket,
        connection_id: str,
        data: dict,
        default_conversation_id: Optional[str],
        user_id: Optional[str],
    ):
        """Handle a chat message."""
        request = ConversationRequest(
            message=data.get("text", ""),
            conversation_id=data.get("conversation_id", default_conversation_id),
            images=data.get("images", []),
            user_id=user_id,
        )

        conversation_id = request.conversation_id

        async for event in self.manager.chat_stream(request):
            await websocket.send_json(event.to_dict())

            if event.type == "conversation_created":
                conversation_id = event.data
                self.connections.subscribe_to_conversation(
                    connection_id,
                    conversation_id,
                )

            if conversation_id:
                await self.connections.broadcast_to_conversation(
                    conversation_id,
                    {
                        "type": "conversation_event",
                        "conversation_id": conversation_id,
                        "event": event.to_dict(),
                    },
                )

    async def _handle_get_history(
        self,
        websocket: WebSocket,
        data: dict,
        default_conversation_id: Optional[str],
    ):
        """Handle get history request."""
        conv_id = data.get("conversation_id", default_conversation_id)
        if not conv_id:
            await websocket.send_json({
                "type": "error",
                "data": "No conversation_id provided",
            })
            return

        conversation = await self.manager.get_conversation(conv_id)
        if not conversation:
            await websocket.send_json({
                "type": "error",
                "data": "Conversation not found",
            })
            return

        limit = data.get("limit", 50)
        messages = conversation.messages[-limit:]

        await websocket.send_json({
            "type": "history",
            "data": {
                "conversation_id": conv_id,
                "messages": [m.to_dict() for m in messages],
                "total": len(conversation.messages),
            },
        })

    async def _handle_create_conversation(
        self,
        websocket: WebSocket,
        connection_id: str,
        data: dict,
        user_id: Optional[str],
    ):
        """Handle create conversation request."""
        conversation = await self.manager.create_conversation(
            user_id=user_id,
            title=data.get("title"),
            metadata=data.get("metadata", {}),
        )

        self.connections.subscribe_to_conversation(
            connection_id,
            conversation.id,
        )

        await websocket.send_json({
            "type": "conversation_created",
            "data": conversation.to_dict(),
        })


def create_websocket_router(
    manager: ConversationManager,
    ws_handler: Optional[ConversationWebSocket] = None,
) -> APIRouter:
    """Create the WebSocket router."""

    router = APIRouter(tags=["websocket"])
    handler = ws_handler or ConversationWebSocket(manager)

    @router.websocket("/ws")
    async def websocket_endpoint(
        websocket: WebSocket,
        conversation_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ):
        """WebSocket endpoint for conversations."""
        await handler.handle(
            websocket,
            conversation_id=conversation_id,
            user_id=user_id,
        )

    @router.websocket("/ws/{conversation_id}")
    async def websocket_conversation_endpoint(
        websocket: WebSocket,
        conversation_id: str,
        user_id: Optional[str] = None,
    ):
        """WebSocket endpoint for a specific conversation."""
        await handler.handle(
            websocket,
            conversation_id=conversation_id,
            user_id=user_id,
        )

    @router.get("/ws/stats")
    async def websocket_stats():
        """Get WebSocket connection statistics."""
        return handler.connections.get_stats()

    return router
