"""
AION Conversation Transports

Transport layer implementations for the conversation interface.
"""

from aion.conversation.transports.rest import (
    create_conversation_router,
    create_health_router,
    ChatRequest,
    CreateConversationRequest,
    UpdateConversationRequest,
)
from aion.conversation.transports.websocket import (
    ConversationWebSocket,
    ConnectionManager,
    create_websocket_router,
)
from aion.conversation.transports.cli import (
    ConversationCLI,
    run_cli,
)

__all__ = [
    # REST
    "create_conversation_router",
    "create_health_router",
    "ChatRequest",
    "CreateConversationRequest",
    "UpdateConversationRequest",
    # WebSocket
    "ConversationWebSocket",
    "ConnectionManager",
    "create_websocket_router",
    # CLI
    "ConversationCLI",
    "run_cli",
]
