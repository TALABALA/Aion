"""
AION Conversation Manager

Central orchestrator for conversations:
- Session management
- Message routing
- Tool execution
- Memory integration
- Response streaming
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, AsyncIterator, Callable, Optional

import structlog

from aion.conversation.types import (
    Conversation,
    ConversationConfig,
    ConversationRequest,
    ConversationResponse,
    ConversationStats,
    Message,
    MessageRole,
    StreamEvent,
    TextContent,
    ToolUseContent,
    ToolResultContent,
    ThinkingContent,
    ImageContent,
)
from aion.conversation.session import SessionManager
from aion.conversation.context import ContextBuilder
from aion.conversation.llm.base import LLMProvider
from aion.conversation.llm.claude import ClaudeProvider
from aion.conversation.routing.router import RequestRouter, RoutingDecision
from aion.conversation.routing.intent import IntentClassifier
from aion.conversation.memory.integrator import MemoryIntegrator
from aion.conversation.tools.executor import ToolExecutor
from aion.conversation.tools.formatter import ToolResultFormatter
from aion.conversation.streaming.handler import StreamingHandler, StreamingState

logger = structlog.get_logger(__name__)


class ConversationManager:
    """
    Central manager for AION conversations.

    Orchestrates:
    - Session and conversation lifecycle
    - LLM interactions with streaming
    - Tool execution loops
    - Memory retrieval and storage
    - Context window management
    """

    def __init__(
        self,
        llm_provider: Optional[LLMProvider] = None,
        session_manager: Optional[SessionManager] = None,
        memory_integrator: Optional[MemoryIntegrator] = None,
        tool_executor: Optional[ToolExecutor] = None,
        default_config: Optional[ConversationConfig] = None,
    ):
        self.llm = llm_provider
        self.sessions = session_manager or SessionManager()
        self.memory = memory_integrator
        self.tools = tool_executor
        self.default_config = default_config or ConversationConfig()

        self.router = RequestRouter()
        self.intent_classifier = IntentClassifier()
        self.context_builder = ContextBuilder()
        self.tool_formatter = ToolResultFormatter()

        self._on_message_callbacks: list[Callable] = []
        self._on_tool_use_callbacks: list[Callable] = []
        self._on_stream_callbacks: list[Callable] = []

        self._stats = ConversationStats()
        self._latencies: list[float] = []
        self._max_latency_samples = 1000

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the conversation manager."""
        if self._initialized:
            return

        logger.info("Initializing Conversation Manager")

        if self.llm is None:
            self.llm = ClaudeProvider()

        await self.llm.initialize()
        await self.sessions.initialize()

        if self.memory:
            await self.memory.initialize()

        if self.tools:
            await self.tools.initialize()

        self._initialized = True
        logger.info("Conversation Manager initialized")

    async def shutdown(self) -> None:
        """Shutdown the conversation manager."""
        logger.info("Shutting down Conversation Manager")

        await self.sessions.shutdown()

        if self.llm:
            await self.llm.shutdown()

        if self.memory:
            await self.memory.shutdown()

        if self.tools:
            await self.tools.shutdown()

        self._initialized = False
        logger.info("Conversation Manager shutdown complete")

    async def create_conversation(
        self,
        config: Optional[ConversationConfig] = None,
        user_id: Optional[str] = None,
        title: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> Conversation:
        """Create a new conversation."""
        conversation = Conversation(
            config=config or self.default_config,
            user_id=user_id,
            title=title,
            metadata=metadata or {},
        )

        await self.sessions.save_conversation(conversation)
        self._stats.conversations_created += 1

        logger.info(
            "Created conversation",
            conversation_id=conversation.id[:8],
            user_id=user_id,
        )

        return conversation

    async def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Get an existing conversation."""
        return await self.sessions.get_conversation(conversation_id)

    async def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation."""
        return await self.sessions.delete_conversation(conversation_id)

    async def list_conversations(
        self,
        user_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Conversation]:
        """List conversations."""
        return await self.sessions.list_conversations(
            user_id=user_id,
            limit=limit,
            offset=offset,
        )

    async def chat(
        self,
        request: ConversationRequest,
    ) -> ConversationResponse:
        """
        Process a chat request and return a response.

        This is the main entry point for non-streaming interactions.
        """
        start_time = time.time()

        if request.conversation_id:
            conversation = await self.get_conversation(request.conversation_id)
            if not conversation:
                raise ValueError(f"Conversation not found: {request.conversation_id}")
        else:
            conversation = await self.create_conversation(
                user_id=request.user_id,
            )

        config = conversation.config
        for key, value in request.config_overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)

        user_message = self._create_user_message(request)
        conversation.add_message(user_message)

        routing_decision = self.router.route(
            request.message,
            conversation,
            config,
        )

        response_message, execution_info = await self._process_message(
            conversation,
            user_message,
            routing_decision,
        )

        conversation.add_message(response_message)

        await self.sessions.save_conversation(conversation)

        if config.memory_auto_store and self.memory:
            await self.memory.store_interaction(
                user_message,
                response_message,
                conversation_id=conversation.id,
            )
            self._stats.memory_stores += 1

        latency_ms = (time.time() - start_time) * 1000
        self._record_latency(latency_ms)

        self._stats.messages_processed += 1

        for callback in self._on_message_callbacks:
            try:
                await callback(user_message, response_message, conversation)
            except Exception as e:
                logger.warning(f"Message callback error: {e}")

        return ConversationResponse(
            message=response_message,
            conversation_id=conversation.id,
            tools_used=execution_info.get("tools_used", []),
            memories_retrieved=execution_info.get("memories_retrieved", 0),
            plan_executed=execution_info.get("plan_executed", False),
            input_tokens=response_message.input_tokens,
            output_tokens=response_message.output_tokens,
            latency_ms=latency_ms,
        )

    async def chat_stream(
        self,
        request: ConversationRequest,
    ) -> AsyncIterator[StreamEvent]:
        """
        Process a chat request with streaming response.

        Yields StreamEvents for real-time UI updates.
        """
        start_time = time.time()

        if request.conversation_id:
            conversation = await self.get_conversation(request.conversation_id)
            if not conversation:
                yield StreamEvent.error("Conversation not found")
                return
        else:
            conversation = await self.create_conversation(
                user_id=request.user_id,
            )
            yield StreamEvent(type="conversation_created", data=conversation.id)

        config = conversation.config
        for key, value in request.config_overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)

        user_message = self._create_user_message(request)
        conversation.add_message(user_message)

        routing_decision = self.router.route(
            request.message,
            conversation,
            config,
        )

        context = await self._build_context(
            conversation,
            user_message,
            routing_decision,
        )

        memories_retrieved = context.get("memories_retrieved", 0)
        if memories_retrieved > 0:
            self._stats.memory_retrievals += memories_retrieved

        response_content: list = []
        tools_used: list[str] = []
        current_tool_use: Optional[ToolUseContent] = None
        input_tokens = 0
        output_tokens = 0

        tool_iteration = 0
        max_iterations = config.max_tool_iterations

        while tool_iteration < max_iterations:
            tool_iteration += 1
            has_tool_use = False

            try:
                async for event in self.llm.stream(
                    messages=context["messages"],
                    system=context["system"],
                    tools=context["tools"] if routing_decision.use_tools else None,
                    config=config,
                ):
                    for callback in self._on_stream_callbacks:
                        try:
                            await callback(event)
                        except Exception as e:
                            logger.warning(f"Stream callback error: {e}")

                    if event.type == "text":
                        response_content.append(TextContent(text=event.data))
                        yield event

                    elif event.type == "thinking":
                        response_content.append(ThinkingContent(thinking=event.data))
                        yield event

                    elif event.type == "thinking_start":
                        yield event

                    elif event.type == "tool_use_start":
                        current_tool_use = ToolUseContent(
                            id=event.data.get("id", ""),
                            name=event.data.get("name", ""),
                            input={},
                        )
                        yield StreamEvent(
                            type="tool_use_start",
                            data=event.data.get("name", ""),
                        )

                    elif event.type == "tool_use_input":
                        if current_tool_use:
                            current_tool_use.input = event.data

                    elif event.type == "tool_use_end":
                        if current_tool_use:
                            has_tool_use = True
                            response_content.append(current_tool_use)
                            tools_used.append(current_tool_use.name)

                            yield StreamEvent(
                                type="tool_executing",
                                data=current_tool_use.name,
                            )

                            tool_result = await self._execute_tool(current_tool_use)

                            yield StreamEvent(
                                type="tool_result",
                                data={
                                    "tool": current_tool_use.name,
                                    "result": tool_result.content[:500],
                                    "is_error": tool_result.is_error,
                                },
                            )

                            for callback in self._on_tool_use_callbacks:
                                try:
                                    await callback(current_tool_use, tool_result)
                                except Exception as e:
                                    logger.warning(f"Tool callback error: {e}")

                            context["messages"].append({
                                "role": "assistant",
                                "content": [current_tool_use.to_dict()],
                            })
                            context["messages"].append({
                                "role": "user",
                                "content": [tool_result.to_dict()],
                            })

                            current_tool_use = None
                            self._stats.tool_calls += 1

                    elif event.type == "done":
                        if isinstance(event.data, dict):
                            input_tokens = event.data.get("input_tokens", 0)
                            output_tokens = event.data.get("output_tokens", 0)

                    elif event.type == "error":
                        yield event
                        self._stats.errors += 1
                        return

            except Exception as e:
                logger.error(f"Streaming error: {e}")
                yield StreamEvent.error(str(e))
                self._stats.errors += 1
                return

            if not has_tool_use:
                break

        text_parts = [
            c.text for c in response_content
            if isinstance(c, TextContent)
        ]

        response_message = Message(
            role=MessageRole.ASSISTANT,
            content=response_content if response_content else [TextContent(text="".join(text_parts))],
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            metadata={
                "tools_used": tools_used,
                "tool_iterations": tool_iteration,
            },
        )

        conversation.add_message(response_message)
        await self.sessions.save_conversation(conversation)

        if config.memory_auto_store and self.memory:
            await self.memory.store_interaction(
                user_message,
                response_message,
                conversation_id=conversation.id,
            )
            self._stats.memory_stores += 1

        self._stats.messages_processed += 1
        self._stats.total_input_tokens += input_tokens
        self._stats.total_output_tokens += output_tokens

        latency_ms = (time.time() - start_time) * 1000
        self._record_latency(latency_ms)

        yield StreamEvent.done({
            "conversation_id": conversation.id,
            "tools_used": tools_used,
            "memories_retrieved": memories_retrieved,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_ms": latency_ms,
        })

    def _create_user_message(self, request: ConversationRequest) -> Message:
        """Create a user message from request."""
        content: list = [TextContent(text=request.message)]

        for image in request.images:
            content.append(ImageContent(source=image))

        return Message(
            role=MessageRole.USER,
            content=content,
            metadata=request.metadata,
        )

    async def _process_message(
        self,
        conversation: Conversation,
        user_message: Message,
        routing_decision: RoutingDecision,
    ) -> tuple[Message, dict]:
        """Process a message and return response with execution info."""
        context = await self._build_context(
            conversation,
            user_message,
            routing_decision,
        )

        execution_info = {
            "tools_used": [],
            "memories_retrieved": context.get("memories_retrieved", 0),
            "plan_executed": False,
        }

        if execution_info["memories_retrieved"] > 0:
            self._stats.memory_retrievals += execution_info["memories_retrieved"]

        messages_for_llm = context["messages"]
        tool_iteration = 0
        max_iterations = conversation.config.max_tool_iterations

        response = None

        while tool_iteration < max_iterations:
            tool_iteration += 1

            response = await self.llm.complete(
                messages=messages_for_llm,
                system=context["system"],
                tools=context["tools"] if routing_decision.use_tools else None,
                config=conversation.config,
            )

            tool_uses = response.get_tool_uses()

            if not tool_uses:
                self._stats.total_input_tokens += response.input_tokens
                self._stats.total_output_tokens += response.output_tokens
                return response, execution_info

            for tool_use in tool_uses:
                execution_info["tools_used"].append(tool_use.name)

                tool_result = await self._execute_tool(tool_use)

                for callback in self._on_tool_use_callbacks:
                    try:
                        await callback(tool_use, tool_result)
                    except Exception as e:
                        logger.warning(f"Tool callback error: {e}")

                messages_for_llm.append({
                    "role": "assistant",
                    "content": [tool_use.to_dict()],
                })
                messages_for_llm.append({
                    "role": "user",
                    "content": [tool_result.to_dict()],
                })

                self._stats.tool_calls += 1

        self._stats.total_input_tokens += response.input_tokens
        self._stats.total_output_tokens += response.output_tokens

        return response, execution_info

    async def _build_context(
        self,
        conversation: Conversation,
        current_message: Message,
        routing_decision: RoutingDecision,
    ) -> dict[str, Any]:
        """Build the context for LLM call."""
        memories = []
        if routing_decision.use_memory and self.memory:
            memories = await self.memory.retrieve_relevant(
                current_message.get_text(),
                limit=conversation.config.memory_retrieval_count,
            )

        tools = []
        if routing_decision.use_tools and self.tools:
            tools = await self.tools.get_tool_definitions(
                allowed=conversation.config.allowed_tools,
            )

        context_window = await self.context_builder.build_context(
            conversation,
            current_message,
            memories=memories,
            tools=tools,
            config=conversation.config,
        )

        return {
            "messages": context_window.messages,
            "system": context_window.system,
            "tools": context_window.tools,
            "memories_retrieved": len(memories),
            "total_tokens": context_window.total_tokens,
        }

    async def _execute_tool(self, tool_use: ToolUseContent) -> ToolResultContent:
        """Execute a tool and return result."""
        if not self.tools:
            return ToolResultContent(
                tool_use_id=tool_use.id,
                content="Tool execution not available",
                is_error=True,
            )

        try:
            result = await self.tools.execute(
                tool_name=tool_use.name,
                arguments=tool_use.input,
                timeout=self.default_config.tool_timeout_seconds,
            )

            formatted_result = self.tool_formatter.format_for_llm(
                result.get("result", ""),
                tool_use.name,
                result.get("is_error", False),
            )

            return ToolResultContent(
                tool_use_id=tool_use.id,
                content=formatted_result,
                is_error=result.get("is_error", False),
            )

        except Exception as e:
            logger.error(f"Tool execution error: {tool_use.name}", error=str(e))
            return ToolResultContent(
                tool_use_id=tool_use.id,
                content=f"Error: {str(e)}",
                is_error=True,
            )

    def _record_latency(self, latency_ms: float) -> None:
        """Record a latency measurement."""
        self._latencies.append(latency_ms)
        if len(self._latencies) > self._max_latency_samples:
            self._latencies.pop(0)

        self._stats.avg_latency_ms = (
            sum(self._latencies) / len(self._latencies)
            if self._latencies
            else 0.0
        )

    def on_message(self, callback: Callable) -> None:
        """Register callback for message events."""
        self._on_message_callbacks.append(callback)

    def on_tool_use(self, callback: Callable) -> None:
        """Register callback for tool use events."""
        self._on_tool_use_callbacks.append(callback)

    def on_stream(self, callback: Callable) -> None:
        """Register callback for stream events."""
        self._on_stream_callbacks.append(callback)

    def get_stats(self) -> dict[str, Any]:
        """Get conversation manager statistics."""
        stats = self._stats.to_dict()
        stats["active_sessions"] = self.sessions.active_count()

        if self.tools:
            stats["tool_stats"] = self.tools.get_stats()

        return stats

    @property
    def is_initialized(self) -> bool:
        """Check if the manager is initialized."""
        return self._initialized
