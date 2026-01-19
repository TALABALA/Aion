"""
AION Multi-Provider LLM Adapter

Unified interface for multiple LLM providers:
- OpenAI (GPT-4, GPT-4 Turbo)
- Anthropic (Claude 3, Claude 3.5)
- Local models (via OpenAI-compatible API)
- Mock provider (for testing)
"""

from __future__ import annotations

import asyncio
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncGenerator, Optional, Union

import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

logger = structlog.get_logger(__name__)


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"
    MOCK = "mock"


@dataclass
class Message:
    """A message in a conversation."""
    role: str  # "system", "user", "assistant"
    content: str
    name: Optional[str] = None
    tool_calls: Optional[list[dict]] = None
    tool_call_id: Optional[str] = None


@dataclass
class LLMResponse:
    """Response from an LLM call."""
    content: str
    model: str
    provider: LLMProvider
    finish_reason: str
    usage: dict[str, int] = field(default_factory=dict)
    latency_ms: float = 0.0
    tool_calls: Optional[list[dict]] = None
    raw_response: Optional[dict] = None


@dataclass
class LLMConfig:
    """Configuration for an LLM provider."""
    provider: LLMProvider
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    max_tokens: int = 4096
    temperature: float = 0.7
    timeout: float = 60.0
    max_retries: int = 3


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the provider."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close the provider."""
        pass

    @abstractmethod
    async def complete(
        self,
        messages: list[Message],
        **kwargs,
    ) -> LLMResponse:
        """Generate a completion."""
        pass

    @abstractmethod
    async def stream(
        self,
        messages: list[Message],
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """Stream a completion."""
        pass


class OpenAIProvider(BaseLLMProvider):
    """OpenAI API provider."""

    async def initialize(self) -> None:
        """Initialize the OpenAI client."""
        api_key = self.config.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not provided")

        base_url = self.config.base_url or "https://api.openai.com/v1"

        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=self.config.timeout,
        )

    async def close(self) -> None:
        """Close the client."""
        if self._client:
            await self._client.aclose()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def complete(
        self,
        messages: list[Message],
        tools: Optional[list[dict]] = None,
        tool_choice: Optional[str] = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate a completion using OpenAI."""
        if not self._client:
            raise RuntimeError("Provider not initialized")

        start_time = time.monotonic()

        # Format messages
        formatted_messages = []
        for msg in messages:
            formatted = {"role": msg.role, "content": msg.content}
            if msg.name:
                formatted["name"] = msg.name
            if msg.tool_call_id:
                formatted["tool_call_id"] = msg.tool_call_id
            formatted_messages.append(formatted)

        # Build request
        request_data = {
            "model": self.config.model,
            "messages": formatted_messages,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
        }

        if tools:
            request_data["tools"] = tools
            if tool_choice:
                request_data["tool_choice"] = tool_choice

        # Make request
        response = await self._client.post("/chat/completions", json=request_data)
        response.raise_for_status()
        data = response.json()

        latency_ms = (time.monotonic() - start_time) * 1000

        # Parse response
        choice = data["choices"][0]
        message = choice["message"]

        return LLMResponse(
            content=message.get("content", ""),
            model=data["model"],
            provider=LLMProvider.OPENAI,
            finish_reason=choice["finish_reason"],
            usage=data.get("usage", {}),
            latency_ms=latency_ms,
            tool_calls=message.get("tool_calls"),
            raw_response=data,
        )

    async def stream(
        self,
        messages: list[Message],
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """Stream a completion from OpenAI."""
        if not self._client:
            raise RuntimeError("Provider not initialized")

        formatted_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]

        request_data = {
            "model": self.config.model,
            "messages": formatted_messages,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "stream": True,
        }

        async with self._client.stream(
            "POST", "/chat/completions", json=request_data
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    import json
                    chunk = json.loads(data)
                    delta = chunk["choices"][0].get("delta", {})
                    if content := delta.get("content"):
                        yield content


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude API provider."""

    async def initialize(self) -> None:
        """Initialize the Anthropic client."""
        api_key = self.config.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Anthropic API key not provided")

        base_url = self.config.base_url or "https://api.anthropic.com"

        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            timeout=self.config.timeout,
        )

    async def close(self) -> None:
        """Close the client."""
        if self._client:
            await self._client.aclose()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def complete(
        self,
        messages: list[Message],
        tools: Optional[list[dict]] = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate a completion using Anthropic Claude."""
        if not self._client:
            raise RuntimeError("Provider not initialized")

        start_time = time.monotonic()

        # Separate system message
        system_message = None
        formatted_messages = []

        for msg in messages:
            if msg.role == "system":
                system_message = msg.content
            else:
                formatted_messages.append({
                    "role": msg.role,
                    "content": msg.content,
                })

        # Build request
        request_data = {
            "model": self.config.model,
            "messages": formatted_messages,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
        }

        if system_message:
            request_data["system"] = system_message

        if tools:
            # Convert OpenAI tool format to Anthropic format
            request_data["tools"] = [
                {
                    "name": t["function"]["name"],
                    "description": t["function"].get("description", ""),
                    "input_schema": t["function"].get("parameters", {}),
                }
                for t in tools
            ]

        # Make request
        response = await self._client.post("/v1/messages", json=request_data)
        response.raise_for_status()
        data = response.json()

        latency_ms = (time.monotonic() - start_time) * 1000

        # Parse response
        content = ""
        tool_calls = []

        for block in data.get("content", []):
            if block["type"] == "text":
                content = block["text"]
            elif block["type"] == "tool_use":
                tool_calls.append({
                    "id": block["id"],
                    "type": "function",
                    "function": {
                        "name": block["name"],
                        "arguments": block["input"],
                    },
                })

        return LLMResponse(
            content=content,
            model=data["model"],
            provider=LLMProvider.ANTHROPIC,
            finish_reason=data["stop_reason"] or "stop",
            usage={
                "prompt_tokens": data.get("usage", {}).get("input_tokens", 0),
                "completion_tokens": data.get("usage", {}).get("output_tokens", 0),
            },
            latency_ms=latency_ms,
            tool_calls=tool_calls if tool_calls else None,
            raw_response=data,
        )

    async def stream(
        self,
        messages: list[Message],
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """Stream a completion from Anthropic."""
        if not self._client:
            raise RuntimeError("Provider not initialized")

        # Separate system message
        system_message = None
        formatted_messages = []

        for msg in messages:
            if msg.role == "system":
                system_message = msg.content
            else:
                formatted_messages.append({
                    "role": msg.role,
                    "content": msg.content,
                })

        request_data = {
            "model": self.config.model,
            "messages": formatted_messages,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "stream": True,
        }

        if system_message:
            request_data["system"] = system_message

        async with self._client.stream(
            "POST", "/v1/messages", json=request_data
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    import json
                    data = json.loads(line[6:])
                    if data["type"] == "content_block_delta":
                        delta = data.get("delta", {})
                        if text := delta.get("text"):
                            yield text


class MockProvider(BaseLLMProvider):
    """Mock provider for testing."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._responses: list[str] = []
        self._response_index = 0

    def set_responses(self, responses: list[str]) -> None:
        """Set mock responses."""
        self._responses = responses
        self._response_index = 0

    async def initialize(self) -> None:
        """Initialize the mock provider."""
        pass

    async def close(self) -> None:
        """Close the mock provider."""
        pass

    async def complete(
        self,
        messages: list[Message],
        **kwargs,
    ) -> LLMResponse:
        """Generate a mock completion."""
        if self._responses:
            content = self._responses[self._response_index % len(self._responses)]
            self._response_index += 1
        else:
            # Generate a basic response based on the last message
            last_msg = messages[-1].content if messages else ""
            content = f"Mock response to: {last_msg[:100]}"

        return LLMResponse(
            content=content,
            model=self.config.model,
            provider=LLMProvider.MOCK,
            finish_reason="stop",
            usage={"prompt_tokens": 10, "completion_tokens": 20},
            latency_ms=10.0,
        )

    async def stream(
        self,
        messages: list[Message],
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """Stream a mock completion."""
        response = await self.complete(messages, **kwargs)
        for word in response.content.split():
            yield word + " "
            await asyncio.sleep(0.01)


class LLMAdapter:
    """
    Unified LLM Adapter for AION.

    Provides a consistent interface across multiple LLM providers
    with automatic fallback, retry logic, and performance monitoring.
    """

    def __init__(self, config: LLMConfig):
        self.config = config
        self._provider: Optional[BaseLLMProvider] = None
        self._fallback_providers: list[BaseLLMProvider] = []
        self._call_count = 0
        self._total_latency_ms = 0.0
        self._error_count = 0

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def initialize(self) -> None:
        """Initialize the primary provider."""
        self._provider = self._create_provider(self.config)
        await self._provider.initialize()
        logger.info(
            "LLM adapter initialized",
            provider=self.config.provider.value,
            model=self.config.model,
        )

    async def close(self) -> None:
        """Close all providers."""
        if self._provider:
            await self._provider.close()
        for fallback in self._fallback_providers:
            await fallback.close()

    def _create_provider(self, config: LLMConfig) -> BaseLLMProvider:
        """Create a provider instance based on config."""
        providers = {
            LLMProvider.OPENAI: OpenAIProvider,
            LLMProvider.ANTHROPIC: AnthropicProvider,
            LLMProvider.LOCAL: OpenAIProvider,  # Local uses OpenAI-compatible API
            LLMProvider.MOCK: MockProvider,
        }

        provider_class = providers.get(config.provider)
        if not provider_class:
            raise ValueError(f"Unknown provider: {config.provider}")

        return provider_class(config)

    def add_fallback(self, config: LLMConfig) -> None:
        """Add a fallback provider."""
        provider = self._create_provider(config)
        self._fallback_providers.append(provider)

    async def complete(
        self,
        messages: list[Message],
        **kwargs,
    ) -> LLMResponse:
        """
        Generate a completion with automatic fallback.

        Args:
            messages: List of messages in the conversation
            **kwargs: Additional arguments for the provider

        Returns:
            LLMResponse with the completion

        Raises:
            RuntimeError: If all providers fail
        """
        if not self._provider:
            raise RuntimeError("LLM adapter not initialized")

        providers = [self._provider] + self._fallback_providers
        last_error = None

        for provider in providers:
            try:
                response = await provider.complete(messages, **kwargs)
                self._call_count += 1
                self._total_latency_ms += response.latency_ms
                return response
            except Exception as e:
                last_error = e
                self._error_count += 1
                logger.warning(
                    "Provider failed, trying fallback",
                    provider=provider.config.provider.value,
                    error=str(e),
                )

        raise RuntimeError(f"All LLM providers failed: {last_error}")

    async def stream(
        self,
        messages: list[Message],
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """Stream a completion."""
        if not self._provider:
            raise RuntimeError("LLM adapter not initialized")

        async for chunk in self._provider.stream(messages, **kwargs):
            yield chunk

    async def complete_with_tools(
        self,
        messages: list[Message],
        tools: list[dict],
        tool_choice: Optional[str] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate a completion with tool calling.

        Args:
            messages: List of messages
            tools: List of tool definitions
            tool_choice: Tool choice strategy
            **kwargs: Additional arguments

        Returns:
            LLMResponse with potential tool calls
        """
        return await self.complete(
            messages,
            tools=tools,
            tool_choice=tool_choice,
            **kwargs,
        )

    def get_stats(self) -> dict[str, Any]:
        """Get adapter statistics."""
        return {
            "call_count": self._call_count,
            "error_count": self._error_count,
            "total_latency_ms": self._total_latency_ms,
            "avg_latency_ms": (
                self._total_latency_ms / self._call_count
                if self._call_count > 0
                else 0
            ),
            "error_rate": (
                self._error_count / self._call_count
                if self._call_count > 0
                else 0
            ),
        }

    def reset_stats(self) -> None:
        """Reset statistics."""
        self._call_count = 0
        self._total_latency_ms = 0.0
        self._error_count = 0


# Convenience function for creating an adapter
async def create_llm_adapter(
    provider: Union[str, LLMProvider] = LLMProvider.OPENAI,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs,
) -> LLMAdapter:
    """
    Create and initialize an LLM adapter.

    Args:
        provider: LLM provider to use
        model: Model name
        api_key: API key
        **kwargs: Additional configuration

    Returns:
        Initialized LLMAdapter
    """
    if isinstance(provider, str):
        provider = LLMProvider(provider)

    # Default models
    default_models = {
        LLMProvider.OPENAI: "gpt-4-turbo-preview",
        LLMProvider.ANTHROPIC: "claude-3-opus-20240229",
        LLMProvider.LOCAL: "local-model",
        LLMProvider.MOCK: "mock-model",
    }

    config = LLMConfig(
        provider=provider,
        model=model or default_models[provider],
        api_key=api_key,
        **kwargs,
    )

    adapter = LLMAdapter(config)
    await adapter.initialize()
    return adapter
