"""
AION LLM Provider Base

Abstract base class for LLM providers in the conversation system.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Optional

from aion.conversation.types import (
    Message,
    ConversationConfig,
    StreamEvent,
)


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    Implementations must support both synchronous and streaming completions.
    """

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the LLM provider."""
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the LLM provider."""
        pass

    @abstractmethod
    async def complete(
        self,
        messages: list[dict[str, Any]],
        system: Optional[str] = None,
        tools: Optional[list[dict[str, Any]]] = None,
        config: Optional[ConversationConfig] = None,
    ) -> Message:
        """
        Generate a completion.

        Args:
            messages: List of messages in the conversation
            system: Optional system prompt
            tools: Optional list of tool definitions
            config: Optional conversation configuration

        Returns:
            The assistant's response message
        """
        pass

    @abstractmethod
    async def stream(
        self,
        messages: list[dict[str, Any]],
        system: Optional[str] = None,
        tools: Optional[list[dict[str, Any]]] = None,
        config: Optional[ConversationConfig] = None,
    ) -> AsyncIterator[StreamEvent]:
        """
        Stream a completion.

        Args:
            messages: List of messages in the conversation
            system: Optional system prompt
            tools: Optional list of tool definitions
            config: Optional conversation configuration

        Yields:
            StreamEvents for real-time updates
        """
        pass

    def get_model_name(self) -> str:
        """Get the name of the current model."""
        return "unknown"

    def supports_tools(self) -> bool:
        """Check if the provider supports tool use."""
        return True

    def supports_vision(self) -> bool:
        """Check if the provider supports vision/images."""
        return True

    def supports_streaming(self) -> bool:
        """Check if the provider supports streaming."""
        return True

    def supports_extended_thinking(self) -> bool:
        """Check if the provider supports extended thinking."""
        return False

    def get_context_window(self) -> int:
        """Get the context window size in tokens."""
        return 100000

    def get_max_output_tokens(self) -> int:
        """Get the maximum output tokens."""
        return 4096


class MockLLMProvider(LLMProvider):
    """
    Mock LLM provider for testing.
    """

    def __init__(self, responses: Optional[list[str]] = None):
        self._responses = responses or ["This is a mock response."]
        self._response_index = 0
        self._initialized = False

    async def initialize(self) -> None:
        self._initialized = True

    async def shutdown(self) -> None:
        self._initialized = False

    async def complete(
        self,
        messages: list[dict[str, Any]],
        system: Optional[str] = None,
        tools: Optional[list[dict[str, Any]]] = None,
        config: Optional[ConversationConfig] = None,
    ) -> Message:
        from aion.conversation.types import MessageRole, TextContent

        response_text = self._responses[self._response_index % len(self._responses)]
        self._response_index += 1

        return Message(
            role=MessageRole.ASSISTANT,
            content=[TextContent(text=response_text)],
            input_tokens=100,
            output_tokens=50,
        )

    async def stream(
        self,
        messages: list[dict[str, Any]],
        system: Optional[str] = None,
        tools: Optional[list[dict[str, Any]]] = None,
        config: Optional[ConversationConfig] = None,
    ) -> AsyncIterator[StreamEvent]:
        response_text = self._responses[self._response_index % len(self._responses)]
        self._response_index += 1

        words = response_text.split()
        for word in words:
            yield StreamEvent.text(word + " ")

        yield StreamEvent.done({"input_tokens": 100, "output_tokens": 50})

    def get_model_name(self) -> str:
        return "mock-model"
