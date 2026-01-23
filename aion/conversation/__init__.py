"""
AION Conversation Interface

The conversational agent interface for AION, providing natural language
interaction with all AION systems including memory, tools, planning, and vision.

Main components:
- ConversationManager: Central orchestrator for conversations
- ConversationConfig: Configuration for conversations
- Message, Conversation: Core data types
- StreamEvent: Streaming response events

Example usage:

    from aion.conversation import ConversationManager, ConversationRequest

    # Initialize
    manager = ConversationManager()
    await manager.initialize()

    # Create conversation and chat
    conversation = await manager.create_conversation()

    response = await manager.chat(ConversationRequest(
        message="Hello, what can you help me with?",
        conversation_id=conversation.id,
    ))
    print(response.message.get_text())

    # Or stream the response
    async for event in manager.chat_stream(ConversationRequest(
        message="Search for AI news",
        conversation_id=conversation.id,
    )):
        if event.type == "text":
            print(event.data, end="", flush=True)
"""

from aion.conversation.types import (
    Message,
    MessageRole,
    Conversation,
    ConversationConfig,
    ConversationRequest,
    ConversationResponse,
    ConversationStats,
    StreamEvent,
    StreamEventType,
    ContentType,
    TextContent,
    ImageContent,
    AudioContent,
    FileContent,
    ToolUseContent,
    ToolResultContent,
    ThinkingContent,
    PlanContent,
)

from aion.conversation.manager import ConversationManager

from aion.conversation.session import SessionManager, SessionContext

from aion.conversation.context import ContextBuilder, ContextWindow

from aion.conversation.llm.base import LLMProvider, MockLLMProvider
from aion.conversation.llm.claude import ClaudeProvider

from aion.conversation.memory.integrator import MemoryIntegrator

from aion.conversation.tools.executor import ToolExecutor
from aion.conversation.tools.formatter import ToolResultFormatter

from aion.conversation.routing.intent import IntentClassifier, Intent, IntentType
from aion.conversation.routing.router import RequestRouter, RoutingDecision

from aion.conversation.streaming.handler import StreamingHandler, StreamingState
from aion.conversation.streaming.events import format_sse, sse_generator

from aion.conversation.transports.rest import (
    create_conversation_router,
    create_health_router,
)
from aion.conversation.transports.websocket import (
    ConversationWebSocket,
    ConnectionManager,
    create_websocket_router,
)
from aion.conversation.transports.cli import ConversationCLI

__all__ = [
    # Core types
    "Message",
    "MessageRole",
    "Conversation",
    "ConversationConfig",
    "ConversationRequest",
    "ConversationResponse",
    "ConversationStats",
    "StreamEvent",
    "StreamEventType",
    "ContentType",
    "TextContent",
    "ImageContent",
    "AudioContent",
    "FileContent",
    "ToolUseContent",
    "ToolResultContent",
    "ThinkingContent",
    "PlanContent",
    # Manager
    "ConversationManager",
    # Session
    "SessionManager",
    "SessionContext",
    # Context
    "ContextBuilder",
    "ContextWindow",
    # LLM
    "LLMProvider",
    "MockLLMProvider",
    "ClaudeProvider",
    # Memory
    "MemoryIntegrator",
    # Tools
    "ToolExecutor",
    "ToolResultFormatter",
    # Routing
    "IntentClassifier",
    "Intent",
    "IntentType",
    "RequestRouter",
    "RoutingDecision",
    # Streaming
    "StreamingHandler",
    "StreamingState",
    "format_sse",
    "sse_generator",
    # Transports
    "create_conversation_router",
    "create_health_router",
    "ConversationWebSocket",
    "ConnectionManager",
    "create_websocket_router",
    "ConversationCLI",
]

__version__ = "0.1.0"
