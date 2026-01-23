"""
AION Conversation Interface

The conversational agent interface for AION, providing natural language
interaction with all AION systems including memory, tools, planning, and vision.

## SOTA Features

This module includes state-of-the-art implementations for:

- **Intent Classification**: LLM-based with semantic vector matching
- **Memory System**: RAG with hierarchical memory and graph-based knowledge
- **Context Management**: Semantic chunking and progressive summarization
- **Token Counting**: Claude-specific tokenization
- **Prompt Engineering**: Chain-of-Thought, ReAct patterns
- **Resilience**: Circuit breakers, retry with backoff, fallbacks
- **Observability**: OpenTelemetry tracing, Prometheus metrics
- **Safety**: Content moderation, PII detection, prompt injection defense
- **Session Storage**: Redis/PostgreSQL persistence
- **Semantic Caching**: Embedding-based similarity lookup

## Main Components

- ConversationManager: Central orchestrator for conversations
- ConversationConfig: Configuration for conversations
- Message, Conversation: Core data types
- StreamEvent: Streaming response events

## Example Usage

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

# SOTA Modules
from aion.conversation.prompts.engineering import (
    PromptTemplate,
    PromptBuilder,
    ChainOfThoughtPrompt,
    ReActPrompt,
    ConstitutionalPrompt,
    DynamicPromptOptimizer,
)

from aion.conversation.infrastructure.resilience import (
    CircuitBreaker,
    CircuitState,
    RetryPolicy,
    Fallback,
    ResilientExecutor,
)

from aion.conversation.infrastructure.observability import (
    MetricsCollector,
    Tracer,
    ConversationMetrics,
)

from aion.conversation.infrastructure.safety import (
    ContentModerator,
    PIIDetector,
    SafetyGuard,
    SafetyLevel,
)

from aion.conversation.infrastructure.storage import (
    SessionStore,
    MemorySessionStore,
    RedisSessionStore,
    PostgresSessionStore,
    SessionStoreFactory,
)

from aion.conversation.infrastructure.caching import (
    SemanticCache,
    CacheEntry,
    LLMResponseCache,
)

from aion.conversation.context import (
    ClaudeTokenCounter,
    SemanticChunker,
    ProgressiveSummarizer,
    ConversationSummarizer,
)

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
    # SOTA - Prompt Engineering
    "PromptTemplate",
    "PromptBuilder",
    "ChainOfThoughtPrompt",
    "ReActPrompt",
    "ConstitutionalPrompt",
    "DynamicPromptOptimizer",
    # SOTA - Resilience
    "CircuitBreaker",
    "CircuitState",
    "RetryPolicy",
    "Fallback",
    "ResilientExecutor",
    # SOTA - Observability
    "MetricsCollector",
    "Tracer",
    "ConversationMetrics",
    # SOTA - Safety
    "ContentModerator",
    "PIIDetector",
    "SafetyGuard",
    "SafetyLevel",
    # SOTA - Storage
    "SessionStore",
    "MemorySessionStore",
    "RedisSessionStore",
    "PostgresSessionStore",
    "SessionStoreFactory",
    # SOTA - Caching
    "SemanticCache",
    "CacheEntry",
    "LLMResponseCache",
    # SOTA - Context
    "ClaudeTokenCounter",
    "SemanticChunker",
    "ProgressiveSummarizer",
    "ConversationSummarizer",
]

__version__ = "0.2.0"  # SOTA upgrade
