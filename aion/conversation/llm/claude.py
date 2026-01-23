"""
AION Claude LLM Provider

Integration with Anthropic's Claude API for the conversation system.
"""

from __future__ import annotations

import json
import os
from typing import Any, AsyncIterator, Optional

import structlog

from aion.conversation.types import (
    Message,
    MessageRole,
    ConversationConfig,
    StreamEvent,
    TextContent,
    ToolUseContent,
    ThinkingContent,
)
from aion.conversation.llm.base import LLMProvider

logger = structlog.get_logger(__name__)


class ClaudeProvider(LLMProvider):
    """
    Claude API provider for AION.

    Supports:
    - Standard completions
    - Streaming responses
    - Tool use
    - Extended thinking
    - Vision (images)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        default_model: str = "claude-sonnet-4-20250514",
    ):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.base_url = base_url
        self.default_model = default_model

        self._client = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the Claude client."""
        if self._initialized:
            return

        try:
            import anthropic

            client_kwargs = {}
            if self.api_key:
                client_kwargs["api_key"] = self.api_key
            if self.base_url:
                client_kwargs["base_url"] = self.base_url

            self._client = anthropic.AsyncAnthropic(**client_kwargs)
            self._initialized = True
            logger.info("Claude provider initialized", model=self.default_model)

        except ImportError:
            logger.error("anthropic package not installed")
            raise RuntimeError("anthropic package required: pip install anthropic")

    async def shutdown(self) -> None:
        """Shutdown the provider."""
        self._client = None
        self._initialized = False
        logger.info("Claude provider shutdown")

    async def complete(
        self,
        messages: list[dict[str, Any]],
        system: Optional[str] = None,
        tools: Optional[list[dict[str, Any]]] = None,
        config: Optional[ConversationConfig] = None,
    ) -> Message:
        """Generate a completion."""
        if not self._client:
            raise RuntimeError("Provider not initialized")

        config = config or ConversationConfig()

        request_kwargs = self._build_request(messages, system, tools, config)

        try:
            response = await self._client.messages.create(**request_kwargs)
            return self._convert_response(response)

        except Exception as e:
            logger.error("Claude API error", error=str(e))
            raise

    async def stream(
        self,
        messages: list[dict[str, Any]],
        system: Optional[str] = None,
        tools: Optional[list[dict[str, Any]]] = None,
        config: Optional[ConversationConfig] = None,
    ) -> AsyncIterator[StreamEvent]:
        """Stream a completion."""
        if not self._client:
            raise RuntimeError("Provider not initialized")

        config = config or ConversationConfig()

        request_kwargs = self._build_request(messages, system, tools, config)

        current_tool_id: Optional[str] = None
        current_tool_name: Optional[str] = None
        current_tool_input: str = ""
        input_tokens = 0
        output_tokens = 0

        try:
            async with self._client.messages.stream(**request_kwargs) as stream:
                async for event in stream:
                    if event.type == "message_start":
                        if hasattr(event, "message") and hasattr(event.message, "usage"):
                            input_tokens = event.message.usage.input_tokens

                    elif event.type == "content_block_start":
                        block = event.content_block

                        if block.type == "text":
                            pass

                        elif block.type == "tool_use":
                            current_tool_id = block.id
                            current_tool_name = block.name
                            current_tool_input = ""

                            yield StreamEvent(
                                type="tool_use_start",
                                data={"id": current_tool_id, "name": current_tool_name},
                            )

                        elif block.type == "thinking":
                            yield StreamEvent(type="thinking_start", data=None)

                    elif event.type == "content_block_delta":
                        delta = event.delta

                        if delta.type == "text_delta":
                            yield StreamEvent(type="text", data=delta.text)

                        elif delta.type == "input_json_delta":
                            current_tool_input += delta.partial_json

                        elif delta.type == "thinking_delta":
                            yield StreamEvent(type="thinking", data=delta.thinking)

                    elif event.type == "content_block_stop":
                        if current_tool_id:
                            try:
                                tool_input = (
                                    json.loads(current_tool_input)
                                    if current_tool_input
                                    else {}
                                )
                            except json.JSONDecodeError:
                                tool_input = {}

                            yield StreamEvent(type="tool_use_input", data=tool_input)
                            yield StreamEvent(type="tool_use_end", data=current_tool_id)

                            current_tool_id = None
                            current_tool_name = None
                            current_tool_input = ""

                    elif event.type == "message_delta":
                        if hasattr(event, "usage"):
                            output_tokens = event.usage.output_tokens

                    elif event.type == "message_stop":
                        yield StreamEvent(
                            type="done",
                            data={
                                "input_tokens": input_tokens,
                                "output_tokens": output_tokens,
                            },
                        )

        except Exception as e:
            logger.error("Claude streaming error", error=str(e))
            yield StreamEvent(type="error", data=str(e))

    def _build_request(
        self,
        messages: list[dict[str, Any]],
        system: Optional[str],
        tools: Optional[list[dict[str, Any]]],
        config: ConversationConfig,
    ) -> dict[str, Any]:
        """Build the API request kwargs."""
        request_kwargs: dict[str, Any] = {
            "model": config.model or self.default_model,
            "max_tokens": config.max_tokens,
            "messages": self._convert_messages(messages),
        }

        if system:
            request_kwargs["system"] = system

        if tools:
            request_kwargs["tools"] = tools

        if config.temperature is not None:
            request_kwargs["temperature"] = config.temperature

        if config.top_p is not None:
            request_kwargs["top_p"] = config.top_p

        if config.extended_thinking_enabled:
            request_kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": config.thinking_budget_tokens,
            }
            if "temperature" in request_kwargs:
                del request_kwargs["temperature"]

        return request_kwargs

    def _convert_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert messages to Claude format."""
        converted = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", [])

            if role == "system":
                continue

            if role == "tool":
                role = "user"

            converted_content = []
            if isinstance(content, str):
                converted_content = [{"type": "text", "text": content}]
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, str):
                        converted_content.append({"type": "text", "text": block})
                    elif isinstance(block, dict):
                        converted_content.append(block)
            else:
                converted_content = [{"type": "text", "text": str(content)}]

            converted.append({
                "role": role,
                "content": converted_content,
            })

        return converted

    def _convert_response(self, response: Any) -> Message:
        """Convert Claude response to Message."""
        content: list = []

        for block in response.content:
            if block.type == "text":
                content.append(TextContent(text=block.text))

            elif block.type == "tool_use":
                content.append(
                    ToolUseContent(
                        id=block.id,
                        name=block.name,
                        input=block.input,
                    )
                )

            elif block.type == "thinking":
                content.append(ThinkingContent(thinking=block.thinking))

        return Message(
            role=MessageRole.ASSISTANT,
            content=content,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

    def get_model_name(self) -> str:
        return self.default_model

    def supports_tools(self) -> bool:
        return True

    def supports_vision(self) -> bool:
        return True

    def supports_streaming(self) -> bool:
        return True

    def supports_extended_thinking(self) -> bool:
        return True

    def get_context_window(self) -> int:
        model = self.default_model.lower()
        if "opus" in model:
            return 200000
        elif "sonnet" in model:
            return 200000
        elif "haiku" in model:
            return 200000
        return 200000

    def get_max_output_tokens(self) -> int:
        model = self.default_model.lower()
        if "opus" in model:
            return 32000
        elif "sonnet" in model:
            return 16000
        elif "haiku" in model:
            return 8000
        return 4096
