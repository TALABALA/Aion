"""
Tests for the AION Conversation Interface.

Tests cover:
- Core types and dataclasses
- Session management
- Context building
- Intent classification and routing
- Conversation manager
- Streaming functionality
- Transport layers
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from aion.conversation.types import (
    Message,
    MessageRole,
    Conversation,
    ConversationConfig,
    ConversationRequest,
    ConversationResponse,
    StreamEvent,
    TextContent,
    ToolUseContent,
    ToolResultContent,
    ThinkingContent,
    ContentType,
)
from aion.conversation.session import SessionManager, SessionContext
from aion.conversation.context import ContextBuilder, ContextWindow
from aion.conversation.routing.intent import IntentClassifier, IntentType, TaskComplexity
from aion.conversation.routing.router import RequestRouter, RoutingDecision
from aion.conversation.llm.base import MockLLMProvider
from aion.conversation.manager import ConversationManager
from aion.conversation.memory.integrator import MemoryIntegrator
from aion.conversation.tools.executor import ToolExecutor
from aion.conversation.tools.formatter import ToolResultFormatter
from aion.conversation.streaming.handler import StreamingHandler, StreamingState
from aion.conversation.streaming.events import format_sse, SSEManager


# ==================== Types Tests ====================

class TestMessage:
    """Tests for Message dataclass."""

    def test_create_user_message(self):
        """Test creating a user message."""
        msg = Message.user("Hello, AION!")

        assert msg.role == MessageRole.USER
        assert len(msg.content) == 1
        assert isinstance(msg.content[0], TextContent)
        assert msg.content[0].text == "Hello, AION!"

    def test_create_assistant_message(self):
        """Test creating an assistant message."""
        msg = Message.assistant("Hello! How can I help?")

        assert msg.role == MessageRole.ASSISTANT
        assert msg.get_text() == "Hello! How can I help?"

    def test_create_system_message(self):
        """Test creating a system message."""
        msg = Message.system("You are a helpful assistant.")

        assert msg.role == MessageRole.SYSTEM
        assert msg.get_text() == "You are a helpful assistant."

    def test_message_with_tool_use(self):
        """Test message with tool use content."""
        tool_use = ToolUseContent(
            id="tool_1",
            name="web_search",
            input={"query": "AI news"},
        )
        msg = Message(
            role=MessageRole.ASSISTANT,
            content=[tool_use],
        )

        assert msg.has_tool_use()
        assert len(msg.get_tool_uses()) == 1
        assert msg.get_tool_uses()[0].name == "web_search"

    def test_message_to_dict(self):
        """Test message serialization."""
        msg = Message.user("Test message")
        d = msg.to_dict()

        assert d["role"] == "user"
        assert len(d["content"]) == 1
        assert "id" in d
        assert "created_at" in d

    def test_message_from_dict(self):
        """Test message deserialization."""
        data = {
            "id": "test-id",
            "role": "user",
            "content": [{"type": "text", "text": "Hello"}],
            "created_at": datetime.now().isoformat(),
        }
        msg = Message.from_dict(data)

        assert msg.id == "test-id"
        assert msg.role == MessageRole.USER
        assert msg.get_text() == "Hello"


class TestConversation:
    """Tests for Conversation dataclass."""

    def test_create_conversation(self):
        """Test creating a conversation."""
        conv = Conversation()

        assert conv.id is not None
        assert conv.messages == []
        assert conv.message_count == 0

    def test_add_message(self):
        """Test adding messages to conversation."""
        conv = Conversation()
        msg = Message.user("Hello")

        conv.add_message(msg)

        assert len(conv.messages) == 1
        assert conv.message_count == 1
        assert conv.updated_at >= conv.created_at

    def test_get_context_messages(self):
        """Test getting context window messages."""
        config = ConversationConfig(max_context_messages=5)
        conv = Conversation(config=config)

        for i in range(10):
            conv.add_message(Message.user(f"Message {i}"))

        context = conv.get_context_messages()

        assert len(context) == 5
        assert context[0].get_text() == "Message 5"

    def test_conversation_to_dict(self):
        """Test conversation serialization."""
        conv = Conversation(title="Test Conversation")
        conv.add_message(Message.user("Hello"))

        d = conv.to_dict()

        assert d["title"] == "Test Conversation"
        assert d["message_count"] == 1

    def test_conversation_from_dict(self):
        """Test conversation deserialization."""
        data = {
            "id": "conv-1",
            "title": "Test",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Hi"}]},
            ],
            "config": {"model": "claude-sonnet-4-20250514"},
        }
        conv = Conversation.from_dict(data)

        assert conv.id == "conv-1"
        assert conv.title == "Test"
        assert len(conv.messages) == 1


class TestStreamEvent:
    """Tests for StreamEvent."""

    def test_create_text_event(self):
        """Test creating a text event."""
        event = StreamEvent.text("Hello")

        assert event.type == "text"
        assert event.data == "Hello"

    def test_create_tool_use_event(self):
        """Test creating a tool use start event."""
        event = StreamEvent.tool_use_start("tool_1", "web_search")

        assert event.type == "tool_use_start"
        assert event.data["id"] == "tool_1"
        assert event.data["name"] == "web_search"

    def test_create_done_event(self):
        """Test creating a done event."""
        event = StreamEvent.done({"tokens": 100})

        assert event.type == "done"
        assert event.data["tokens"] == 100

    def test_event_to_dict(self):
        """Test event serialization."""
        event = StreamEvent.text("Test")
        d = event.to_dict()

        assert d["type"] == "text"
        assert d["data"] == "Test"
        assert "timestamp" in d


# ==================== Session Tests ====================

class TestSessionManager:
    """Tests for SessionManager."""

    @pytest.mark.asyncio
    async def test_initialize_and_shutdown(self):
        """Test session manager lifecycle."""
        manager = SessionManager(max_sessions=10)

        await manager.initialize()
        assert manager._initialized

        await manager.shutdown()
        assert not manager._initialized

    @pytest.mark.asyncio
    async def test_save_and_get_conversation(self):
        """Test saving and retrieving conversations."""
        manager = SessionManager()
        await manager.initialize()

        conv = Conversation(title="Test")
        await manager.save_conversation(conv)

        retrieved = await manager.get_conversation(conv.id)

        assert retrieved is not None
        assert retrieved.title == "Test"

        await manager.shutdown()

    @pytest.mark.asyncio
    async def test_delete_conversation(self):
        """Test deleting a conversation."""
        manager = SessionManager()
        await manager.initialize()

        conv = Conversation()
        await manager.save_conversation(conv)

        success = await manager.delete_conversation(conv.id)

        assert success
        assert await manager.get_conversation(conv.id) is None

        await manager.shutdown()

    @pytest.mark.asyncio
    async def test_list_conversations(self):
        """Test listing conversations."""
        manager = SessionManager()
        await manager.initialize()

        for i in range(5):
            conv = Conversation(title=f"Conv {i}", user_id="user1")
            await manager.save_conversation(conv)

        convs = await manager.list_conversations(user_id="user1")

        assert len(convs) == 5

        await manager.shutdown()

    @pytest.mark.asyncio
    async def test_max_sessions_eviction(self):
        """Test eviction when max sessions reached."""
        manager = SessionManager(max_sessions=3)
        await manager.initialize()

        for i in range(5):
            conv = Conversation(title=f"Conv {i}")
            await manager.save_conversation(conv)
            await asyncio.sleep(0.01)

        assert manager.active_count() <= 3

        await manager.shutdown()


# ==================== Context Tests ====================

class TestContextBuilder:
    """Tests for ContextBuilder."""

    def test_count_tokens(self):
        """Test token counting."""
        builder = ContextBuilder()
        tokens = builder.count_tokens("Hello, world!")

        assert tokens > 0
        assert isinstance(tokens, int)

    def test_count_message_tokens(self):
        """Test counting tokens in a message."""
        builder = ContextBuilder()
        message = {
            "role": "user",
            "content": [{"type": "text", "text": "Hello, how are you?"}],
        }

        tokens = builder.count_message_tokens(message)

        assert tokens > 0

    @pytest.mark.asyncio
    async def test_build_context(self):
        """Test building context window."""
        builder = ContextBuilder()
        conv = Conversation()
        conv.add_message(Message.user("Hello"))
        conv.add_message(Message.assistant("Hi there!"))
        current = Message.user("How are you?")

        context = await builder.build_context(conv, current)

        assert isinstance(context, ContextWindow)
        assert len(context.messages) > 0
        assert context.system != ""

    @pytest.mark.asyncio
    async def test_context_with_memories(self):
        """Test context building with memories."""
        builder = ContextBuilder()
        conv = Conversation()
        conv.add_message(Message.user("Tell me about AI"))

        memories = [
            MagicMock(content="AI is artificial intelligence"),
            MagicMock(content="Machine learning is a subset of AI"),
        ]

        context = await builder.build_context(
            conv,
            conv.messages[0],
            memories=memories,
        )

        assert context.memories_included == 2
        assert "relevant_memories" in context.system


# ==================== Intent Classification Tests ====================

class TestIntentClassifier:
    """Tests for IntentClassifier."""

    def test_classify_greeting(self):
        """Test greeting classification."""
        classifier = IntentClassifier()

        intent = classifier.classify("Hello there!")

        assert intent.type == IntentType.GREETING
        assert intent.confidence > 0.5

    def test_classify_question(self):
        """Test question classification."""
        classifier = IntentClassifier()

        intent = classifier.classify("What is the weather today?")

        assert intent.type == IntentType.QUESTION
        assert intent.confidence > 0.5

    def test_classify_command(self):
        """Test command classification."""
        classifier = IntentClassifier()

        intent = classifier.classify("Search for AI news")

        assert intent.type == IntentType.COMMAND
        assert intent.requires_tools

    def test_classify_complex_task(self):
        """Test complex task classification."""
        classifier = IntentClassifier()

        intent = classifier.classify(
            "I need you to search the web for the latest AI news, "
            "then summarize the top 5 articles, and finally create "
            "a report with charts showing trends over time."
        )

        assert intent.complexity in (TaskComplexity.MODERATE, TaskComplexity.COMPLEX)
        assert intent.requires_tools or intent.requires_planning

    def test_tool_suggestion(self):
        """Test tool suggestion."""
        classifier = IntentClassifier()

        intent = classifier.classify("Search the internet for Python tutorials")

        assert "web_search" in intent.suggested_tools

    def test_memory_requirement(self):
        """Test memory requirement detection."""
        classifier = IntentClassifier()

        intent = classifier.classify("What did we discuss earlier about the project?")

        assert intent.requires_memory


# ==================== Router Tests ====================

class TestRequestRouter:
    """Tests for RequestRouter."""

    def test_route_simple_message(self):
        """Test routing a simple message."""
        router = RequestRouter()
        conv = Conversation()
        conv.add_message(Message.user("Hello"))

        decision = router.route("How are you?", conv)

        assert isinstance(decision, RoutingDecision)
        assert decision.use_llm

    def test_route_with_tools(self):
        """Test routing with tool requirement."""
        router = RequestRouter()
        conv = Conversation()
        conv.config.tools_enabled = True

        decision = router.route("Search the web for AI news", conv)

        assert decision.use_tools

    def test_route_with_memory(self):
        """Test routing with memory requirement."""
        router = RequestRouter()
        conv = Conversation()
        conv.config.memory_enabled = True

        decision = router.route("What did I ask you earlier?", conv)

        assert decision.use_memory


# ==================== LLM Provider Tests ====================

class TestMockLLMProvider:
    """Tests for MockLLMProvider."""

    @pytest.mark.asyncio
    async def test_complete(self):
        """Test completion."""
        provider = MockLLMProvider(responses=["Test response"])
        await provider.initialize()

        response = await provider.complete(
            messages=[{"role": "user", "content": "Hello"}]
        )

        assert response.get_text() == "Test response"

    @pytest.mark.asyncio
    async def test_stream(self):
        """Test streaming."""
        provider = MockLLMProvider(responses=["Hello world"])
        await provider.initialize()

        events = []
        async for event in provider.stream(
            messages=[{"role": "user", "content": "Hi"}]
        ):
            events.append(event)

        assert len(events) > 0
        assert events[-1].type == "done"


# ==================== Conversation Manager Tests ====================

class TestConversationManager:
    """Tests for ConversationManager."""

    @pytest.mark.asyncio
    async def test_initialize_and_shutdown(self):
        """Test manager lifecycle."""
        llm = MockLLMProvider()
        manager = ConversationManager(llm_provider=llm)

        await manager.initialize()
        assert manager.is_initialized

        await manager.shutdown()
        assert not manager.is_initialized

    @pytest.mark.asyncio
    async def test_create_conversation(self):
        """Test creating a conversation."""
        llm = MockLLMProvider()
        manager = ConversationManager(llm_provider=llm)
        await manager.initialize()

        conv = await manager.create_conversation(title="Test")

        assert conv.title == "Test"
        assert conv.id is not None

        await manager.shutdown()

    @pytest.mark.asyncio
    async def test_chat(self):
        """Test chat functionality."""
        llm = MockLLMProvider(responses=["Hello! I'm AION."])
        manager = ConversationManager(llm_provider=llm)
        await manager.initialize()

        conv = await manager.create_conversation()
        request = ConversationRequest(
            message="Hello!",
            conversation_id=conv.id,
        )

        response = await manager.chat(request)

        assert response.message.get_text() == "Hello! I'm AION."
        assert response.conversation_id == conv.id

        await manager.shutdown()

    @pytest.mark.asyncio
    async def test_chat_stream(self):
        """Test streaming chat."""
        llm = MockLLMProvider(responses=["Streaming response"])
        manager = ConversationManager(llm_provider=llm)
        await manager.initialize()

        conv = await manager.create_conversation()
        request = ConversationRequest(
            message="Stream this",
            conversation_id=conv.id,
        )

        events = []
        async for event in manager.chat_stream(request):
            events.append(event)

        assert len(events) > 0
        assert any(e.type == "done" for e in events)

        await manager.shutdown()

    @pytest.mark.asyncio
    async def test_get_stats(self):
        """Test getting statistics."""
        llm = MockLLMProvider()
        manager = ConversationManager(llm_provider=llm)
        await manager.initialize()

        await manager.create_conversation()

        stats = manager.get_stats()

        assert stats["conversations_created"] == 1
        assert "active_sessions" in stats

        await manager.shutdown()


# ==================== Tool Executor Tests ====================

class TestToolExecutor:
    """Tests for ToolExecutor."""

    @pytest.mark.asyncio
    async def test_execute_without_orchestrator(self):
        """Test execution without orchestrator."""
        executor = ToolExecutor()
        await executor.initialize()

        result = await executor.execute("some_tool", {})

        assert result["is_error"]
        assert "not available" in result["result"]

    @pytest.mark.asyncio
    async def test_get_tool_definitions(self):
        """Test getting tool definitions."""
        mock_orchestrator = MagicMock()
        mock_orchestrator.registry.list_tools.return_value = []

        executor = ToolExecutor(orchestrator=mock_orchestrator)
        await executor.initialize()

        tools = await executor.get_tool_definitions()

        assert isinstance(tools, list)


class TestToolResultFormatter:
    """Tests for ToolResultFormatter."""

    def test_format_for_llm(self):
        """Test formatting for LLM."""
        formatter = ToolResultFormatter()

        result = formatter.format_for_llm(
            {"data": "test"},
            "test_tool",
        )

        assert "test" in result

    def test_format_error(self):
        """Test error formatting."""
        formatter = ToolResultFormatter()

        result = formatter.format_for_llm(
            "Something went wrong",
            "test_tool",
            is_error=True,
        )

        assert "Error" in result

    def test_format_for_display(self):
        """Test display formatting."""
        formatter = ToolResultFormatter(max_display_length=50)

        result = formatter.format_for_display(
            "A" * 100,
            "test_tool",
        )

        assert len(result) <= 100


# ==================== Memory Integrator Tests ====================

class TestMemoryIntegrator:
    """Tests for MemoryIntegrator."""

    @pytest.mark.asyncio
    async def test_retrieve_without_memory(self):
        """Test retrieval without memory system."""
        integrator = MemoryIntegrator()
        await integrator.initialize()

        results = await integrator.retrieve_relevant("test query")

        assert results == []

    @pytest.mark.asyncio
    async def test_store_without_memory(self):
        """Test storage without memory system."""
        integrator = MemoryIntegrator()
        await integrator.initialize()

        result = await integrator.store_interaction(
            Message.user("Hello"),
            Message.assistant("Hi"),
        )

        assert result is None


# ==================== Streaming Tests ====================

class TestStreamingHandler:
    """Tests for StreamingHandler."""

    @pytest.mark.asyncio
    async def test_handle_text_events(self):
        """Test handling text events."""
        handler = StreamingHandler()

        async def mock_stream():
            yield StreamEvent.text("Hello ")
            yield StreamEvent.text("World")
            yield StreamEvent.done()

        events = []
        async for event in handler.handle_stream(mock_stream()):
            events.append(event)

        assert handler.state.text_buffer == "Hello World"
        assert handler.state.is_complete

    @pytest.mark.asyncio
    async def test_handle_tool_events(self):
        """Test handling tool use events."""
        handler = StreamingHandler()

        async def mock_stream():
            yield StreamEvent(type="tool_use_start", data={"id": "t1", "name": "search"})
            yield StreamEvent(type="tool_use_input", data={"query": "test"})
            yield StreamEvent(type="tool_use_end", data="t1")
            yield StreamEvent.done()

        events = []
        async for event in handler.handle_stream(mock_stream()):
            events.append(event)

        assert "t1" in handler.state.completed_tool_ids


class TestSSEFormatting:
    """Tests for SSE formatting."""

    def test_format_sse_event(self):
        """Test SSE event formatting."""
        event = StreamEvent.text("Hello")

        sse = format_sse(event)

        assert "data:" in sse
        assert "Hello" in sse
        assert sse.endswith("\n")


class TestSSEManager:
    """Tests for SSEManager."""

    def test_register_and_unregister(self):
        """Test connection registration."""
        manager = SSEManager()

        queue = manager.register("conn-1")

        assert "conn-1" in manager._connections
        assert queue is not None

        manager.unregister("conn-1")

        assert "conn-1" not in manager._connections

    @pytest.mark.asyncio
    async def test_send_event(self):
        """Test sending events."""
        manager = SSEManager()
        queue = manager.register("conn-1")

        success = await manager.send("conn-1", StreamEvent.text("Test"))

        assert success
        assert not queue.empty()

    @pytest.mark.asyncio
    async def test_broadcast(self):
        """Test broadcasting."""
        manager = SSEManager()
        manager.register("conn-1")
        manager.register("conn-2")

        count = await manager.broadcast(StreamEvent.text("Broadcast"))

        assert count == 2


# ==================== Integration Tests ====================

class TestConversationIntegration:
    """Integration tests for the conversation system."""

    @pytest.mark.asyncio
    async def test_full_conversation_flow(self):
        """Test a complete conversation flow."""
        llm = MockLLMProvider(responses=[
            "Hello! I'm AION, your AI assistant.",
            "I can help you with many things!",
        ])
        manager = ConversationManager(llm_provider=llm)
        await manager.initialize()

        # Create conversation
        conv = await manager.create_conversation(
            title="Integration Test",
            user_id="test_user",
        )

        # First message
        response1 = await manager.chat(ConversationRequest(
            message="Hello!",
            conversation_id=conv.id,
        ))

        assert "Hello" in response1.message.get_text()

        # Second message
        response2 = await manager.chat(ConversationRequest(
            message="What can you help with?",
            conversation_id=conv.id,
        ))

        assert "help" in response2.message.get_text()

        # Verify conversation state
        updated_conv = await manager.get_conversation(conv.id)
        assert updated_conv.message_count == 4  # 2 user + 2 assistant

        await manager.shutdown()

    @pytest.mark.asyncio
    async def test_conversation_with_config_overrides(self):
        """Test conversation with config overrides."""
        llm = MockLLMProvider(responses=["Custom response"])
        manager = ConversationManager(llm_provider=llm)
        await manager.initialize()

        conv = await manager.create_conversation()

        response = await manager.chat(ConversationRequest(
            message="Test",
            conversation_id=conv.id,
            config_overrides={
                "temperature": 0.5,
                "max_tokens": 100,
            },
        ))

        assert response.message is not None

        await manager.shutdown()


# Run tests with: pytest tests/test_conversation.py -v
