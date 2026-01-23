"""
AION Context Builder

Builds and manages conversation context for LLM calls.
Handles:
- Message history windowing
- System prompt construction
- Memory injection
- Token counting and trimming
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
import tiktoken

import structlog

from aion.conversation.types import (
    Conversation,
    Message,
    MessageRole,
    ConversationConfig,
    TextContent,
    ToolUseContent,
    ToolResultContent,
    ThinkingContent,
)

logger = structlog.get_logger(__name__)


@dataclass
class ContextWindow:
    """Represents a context window for LLM input."""
    messages: list[dict[str, Any]] = field(default_factory=list)
    system: str = ""
    tools: list[dict[str, Any]] = field(default_factory=list)
    total_tokens: int = 0
    memories_included: int = 0
    messages_included: int = 0
    truncated: bool = False


class ContextBuilder:
    """
    Builds context windows for LLM calls.

    Features:
    - Smart message windowing
    - System prompt construction
    - Memory injection
    - Token counting and budget management
    - Tool definition formatting
    """

    def __init__(
        self,
        default_system_prompt: Optional[str] = None,
        token_counter: Optional[Any] = None,
    ):
        self.default_system_prompt = default_system_prompt or self._get_default_system_prompt()
        self._token_counter = token_counter
        self._encoding: Optional[Any] = None

    def _get_encoder(self) -> Any:
        """Get or create the token encoder."""
        if self._encoding is None:
            try:
                self._encoding = tiktoken.get_encoding("cl100k_base")
            except Exception:
                self._encoding = None
        return self._encoding

    def count_tokens(self, text: str) -> int:
        """Count tokens in a text string."""
        encoder = self._get_encoder()
        if encoder:
            return len(encoder.encode(text))
        return len(text) // 4

    def count_message_tokens(self, message: dict[str, Any]) -> int:
        """Count tokens in a message."""
        total = 0

        content = message.get("content", [])
        if isinstance(content, str):
            total += self.count_tokens(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        total += self.count_tokens(block.get("text", ""))
                    elif block.get("type") == "tool_use":
                        total += self.count_tokens(str(block.get("input", {})))
                        total += self.count_tokens(block.get("name", ""))
                    elif block.get("type") == "tool_result":
                        total += self.count_tokens(block.get("content", ""))
                    elif block.get("type") == "thinking":
                        total += self.count_tokens(block.get("thinking", ""))
                elif isinstance(block, str):
                    total += self.count_tokens(block)

        total += 4

        return total

    async def build_context(
        self,
        conversation: Conversation,
        current_message: Optional[Message] = None,
        memories: Optional[list[Any]] = None,
        tools: Optional[list[dict[str, Any]]] = None,
        config: Optional[ConversationConfig] = None,
    ) -> ContextWindow:
        """
        Build a context window for an LLM call.

        Args:
            conversation: The conversation
            current_message: The current user message (if not yet in conversation)
            memories: Retrieved memories to include
            tools: Tool definitions to include
            config: Configuration overrides

        Returns:
            ContextWindow with messages, system prompt, and tools
        """
        config = config or conversation.config

        context = ContextWindow()

        system_parts = []
        if config.system_prompt:
            system_parts.append(config.system_prompt)
        elif config.include_system_context:
            system_parts.append(self.default_system_prompt)

        if memories:
            memory_context = self._format_memories(memories)
            system_parts.append(f"\n\n<relevant_memories>\n{memory_context}\n</relevant_memories>")
            context.memories_included = len(memories)

        context.system = "\n".join(system_parts)

        system_tokens = self.count_tokens(context.system)

        tool_tokens = 0
        if tools:
            context.tools = tools
            tool_tokens = sum(
                self.count_tokens(str(tool)) for tool in tools
            )

        available_tokens = config.max_context_tokens - system_tokens - tool_tokens - config.max_tokens

        messages = list(conversation.messages)
        if current_message and current_message not in messages:
            messages.append(current_message)

        context.messages = self._fit_messages_to_budget(
            messages,
            available_tokens,
            config.max_context_messages,
        )

        context.messages_included = len(context.messages)
        context.total_tokens = (
            system_tokens
            + tool_tokens
            + sum(self.count_message_tokens(m) for m in context.messages)
        )
        context.truncated = len(context.messages) < len(messages)

        if context.truncated:
            logger.debug(
                "Context truncated",
                original_messages=len(messages),
                included_messages=len(context.messages),
                total_tokens=context.total_tokens,
            )

        return context

    def _fit_messages_to_budget(
        self,
        messages: list[Message],
        token_budget: int,
        max_messages: int,
    ) -> list[dict[str, Any]]:
        """
        Fit messages into the token budget.

        Uses a sliding window approach, keeping the most recent messages
        while respecting the token budget.
        """
        messages = messages[-max_messages:]

        converted_messages = []
        for msg in messages:
            converted_messages.append(self._convert_message(msg))

        total_tokens = sum(
            self.count_message_tokens(m) for m in converted_messages
        )

        while total_tokens > token_budget and len(converted_messages) > 1:
            removed = converted_messages.pop(0)
            total_tokens -= self.count_message_tokens(removed)

            if (
                converted_messages
                and converted_messages[0].get("role") == "user"
                and any(
                    isinstance(b, dict) and b.get("type") == "tool_result"
                    for b in converted_messages[0].get("content", [])
                )
            ):
                removed = converted_messages.pop(0)
                total_tokens -= self.count_message_tokens(removed)

        return converted_messages

    def _convert_message(self, message: Message) -> dict[str, Any]:
        """Convert a Message to dict format for the LLM."""
        content = []

        for block in message.content:
            content.append(block.to_dict())

        return {
            "role": message.role.value if message.role != MessageRole.TOOL else "user",
            "content": content,
        }

    def _format_memories(self, memories: list[Any]) -> str:
        """Format memories for inclusion in the system prompt."""
        parts = []
        for i, memory in enumerate(memories, 1):
            if hasattr(memory, "content"):
                content = memory.content
            elif isinstance(memory, dict):
                content = memory.get("content", str(memory))
            else:
                content = str(memory)

            importance = ""
            if hasattr(memory, "importance"):
                importance = f" (importance: {memory.importance:.2f})"
            elif isinstance(memory, dict) and "importance" in memory:
                importance = f" (importance: {memory['importance']:.2f})"

            parts.append(f"[Memory {i}{importance}] {content}")

        return "\n".join(parts)

    def _get_default_system_prompt(self) -> str:
        """Get the default AION system prompt."""
        return """You are AION (Artificial Intelligence Operating Nexus), an advanced AI assistant with access to memory, tools, and planning capabilities.

You have access to:
- Long-term memory: You can remember and recall information from past conversations
- Tools: You can execute tools to perform actions and retrieve information
- Planning: You can create and execute multi-step plans for complex tasks
- Vision: You can analyze images and visual content
- Audio: You can process and understand audio content

Guidelines:
- Be helpful, accurate, and thoughtful in your responses
- Use tools when they would help answer the user's question
- Reference relevant memories when appropriate
- For complex tasks, consider creating a plan before executing
- Be transparent about what you're doing and why
- Acknowledge limitations and uncertainties
- Prioritize user safety and well-being

You are designed to be a powerful, helpful AI assistant that can handle a wide range of tasks while maintaining high standards of accuracy and helpfulness."""

    def create_summary_prompt(self, messages: list[Message]) -> str:
        """Create a prompt for summarizing a conversation."""
        text_parts = []
        for msg in messages:
            role = "User" if msg.role == MessageRole.USER else "Assistant"
            text_parts.append(f"{role}: {msg.get_text()}")

        conversation_text = "\n".join(text_parts)

        return f"""Please provide a concise summary of the following conversation:

{conversation_text}

Summary:"""

    async def summarize_for_context(
        self,
        messages: list[Message],
        llm_provider: Any,
    ) -> str:
        """
        Summarize older messages to save context space.

        This is useful for very long conversations where we want to
        preserve context without using all tokens.
        """
        from aion.conversation.types import ConversationConfig

        prompt = self.create_summary_prompt(messages)

        response = await llm_provider.complete(
            messages=[{"role": "user", "content": prompt}],
            config=ConversationConfig(
                max_tokens=500,
                temperature=0.3,
            ),
        )

        return response.get_text()


class ConversationSummarizer:
    """
    Handles conversation summarization for context management.
    """

    def __init__(self, llm_provider: Any):
        self.llm = llm_provider
        self.context_builder = ContextBuilder()

    async def should_summarize(
        self,
        conversation: Conversation,
        threshold_messages: int = 30,
        threshold_tokens: int = 50000,
    ) -> bool:
        """Check if conversation should be summarized."""
        if len(conversation.messages) < threshold_messages:
            return False

        total_tokens = sum(
            self.context_builder.count_message_tokens(
                self.context_builder._convert_message(m)
            )
            for m in conversation.messages
        )

        return total_tokens > threshold_tokens

    async def summarize_conversation(
        self,
        conversation: Conversation,
        keep_recent: int = 10,
    ) -> tuple[str, list[Message]]:
        """
        Summarize older messages and return summary + recent messages.

        Args:
            conversation: The conversation to summarize
            keep_recent: Number of recent messages to keep unsummarized

        Returns:
            Tuple of (summary, recent_messages)
        """
        if len(conversation.messages) <= keep_recent:
            return "", conversation.messages

        old_messages = conversation.messages[:-keep_recent]
        recent_messages = conversation.messages[-keep_recent:]

        summary = await self.context_builder.summarize_for_context(
            old_messages,
            self.llm,
        )

        return summary, recent_messages
