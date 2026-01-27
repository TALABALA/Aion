"""AION Tool Mock System - Comprehensive tool mocking for sandboxed execution.

Provides:
- ToolMock: Configurable mock for a single tool with recording, latency
  simulation, conditional responses, and failure injection.
- ToolMockRegistry: Central registry for managing tool mocks with
  call history, assertions, and sequence-based responses.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, List, Optional, Sequence, Union

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class ToolCall:
    """Record of a tool invocation."""

    tool_name: str
    params: Dict[str, Any]
    result: Any = None
    error: Optional[str] = None
    duration_ms: float = 0.0
    timestamp: float = 0.0
    call_index: int = 0


class ToolMock:
    """Configurable mock for a single tool.

    Features:
    - Fixed response or callable.
    - Sequence-based responses (different response per call).
    - Conditional responses based on parameters.
    - Latency simulation.
    - Error injection (deterministic or probabilistic).
    - Call recording for assertions.
    """

    def __init__(
        self,
        name: str,
        default_response: Any = None,
        latency_ms: float = 0.0,
        error: Optional[str] = None,
        error_probability: float = 0.0,
    ) -> None:
        self.name = name
        self._default_response = default_response
        self._latency_ms = latency_ms
        self._error = error
        self._error_probability = error_probability

        # Response sequences
        self._sequence: List[Any] = []
        self._sequence_index = 0

        # Conditional responses
        self._conditionals: List[tuple] = []  # (predicate, response)

        # Callable handler
        self._handler: Optional[Callable] = None

        # Recording
        self._calls: List[ToolCall] = []

    # -- Configuration --

    def returns(self, value: Any) -> "ToolMock":
        """Set the default return value."""
        self._default_response = value
        return self

    def returns_sequence(self, values: Sequence[Any]) -> "ToolMock":
        """Return values in sequence (cycles when exhausted)."""
        self._sequence = list(values)
        self._sequence_index = 0
        return self

    def returns_when(
        self,
        predicate: Callable[[Dict[str, Any]], bool],
        value: Any,
    ) -> "ToolMock":
        """Return value when predicate matches params."""
        self._conditionals.append((predicate, value))
        return self

    def with_handler(self, handler: Callable) -> "ToolMock":
        """Set a custom handler function."""
        self._handler = handler
        return self

    def with_latency(self, ms: float) -> "ToolMock":
        """Simulate latency."""
        self._latency_ms = ms
        return self

    def raises(self, error: str) -> "ToolMock":
        """Always raise an error."""
        self._error = error
        self._error_probability = 1.0
        return self

    def raises_sometimes(self, error: str, probability: float) -> "ToolMock":
        """Raise error with given probability."""
        self._error = error
        self._error_probability = probability
        return self

    # -- Invocation --

    async def __call__(self, params: Dict[str, Any]) -> Any:
        """Invoke the mock."""
        start = time.monotonic()
        call = ToolCall(
            tool_name=self.name,
            params=params,
            timestamp=start,
            call_index=len(self._calls),
        )

        try:
            # Simulate latency
            if self._latency_ms > 0:
                await asyncio.sleep(self._latency_ms / 1000.0)

            # Check for error injection
            if self._error and self._error_probability >= 1.0:
                raise RuntimeError(self._error)
            if self._error and self._error_probability > 0:
                import random
                if random.random() < self._error_probability:
                    raise RuntimeError(self._error)

            # Determine response
            result = await self._resolve_response(params)
            call.result = result
            return result

        except Exception as exc:
            call.error = str(exc)
            raise
        finally:
            call.duration_ms = (time.monotonic() - start) * 1000
            self._calls.append(call)

    async def _resolve_response(self, params: Dict[str, Any]) -> Any:
        """Resolve the response to return."""
        # Custom handler takes precedence
        if self._handler is not None:
            result = self._handler(params)
            if asyncio.iscoroutine(result):
                result = await result
            return result

        # Conditional responses
        for predicate, value in self._conditionals:
            try:
                if predicate(params):
                    return value
            except Exception:
                continue

        # Sequence responses
        if self._sequence:
            result = self._sequence[self._sequence_index % len(self._sequence)]
            self._sequence_index += 1
            return result

        # Default
        if self._default_response is not None:
            return self._default_response

        return {"status": "mocked", "tool": self.name, "params": params}

    # -- Introspection --

    @property
    def calls(self) -> List[ToolCall]:
        return list(self._calls)

    @property
    def call_count(self) -> int:
        return len(self._calls)

    def was_called(self) -> bool:
        return len(self._calls) > 0

    def was_called_with(self, **kwargs: Any) -> bool:
        """Check if tool was called with specific params."""
        for call in self._calls:
            if all(call.params.get(k) == v for k, v in kwargs.items()):
                return True
        return False

    def last_call(self) -> Optional[ToolCall]:
        return self._calls[-1] if self._calls else None

    def reset(self) -> None:
        """Reset call history and sequence index."""
        self._calls.clear()
        self._sequence_index = 0


class ToolMockRegistry:
    """Central registry for managing tool mocks.

    Features:
    - Register/retrieve mocks by name.
    - Global call history.
    - Bulk assertions.
    - Mock snapshots for reset.
    """

    def __init__(self) -> None:
        self._mocks: Dict[str, ToolMock] = {}
        self._global_history: List[ToolCall] = []

    def register(self, mock: ToolMock) -> ToolMock:
        """Register a tool mock."""
        self._mocks[mock.name] = mock
        return mock

    def mock(self, name: str, **kwargs: Any) -> ToolMock:
        """Create and register a tool mock."""
        m = ToolMock(name=name, **kwargs)
        self._mocks[name] = m
        return m

    def get(self, name: str) -> Optional[ToolMock]:
        return self._mocks.get(name)

    def has(self, name: str) -> bool:
        return name in self._mocks

    async def invoke(self, name: str, params: Dict[str, Any]) -> Any:
        """Invoke a mocked tool."""
        mock = self._mocks.get(name)
        if mock is None:
            # Default pass-through mock
            result = {"status": "mocked", "tool": name, "params": params}
            self._global_history.append(ToolCall(
                tool_name=name,
                params=params,
                result=result,
            ))
            return result

        result = await mock(params)
        self._global_history.append(mock.last_call())
        return result

    # -- Assertions --

    def assert_called(self, tool_name: str) -> None:
        """Assert that a tool was called at least once."""
        mock = self._mocks.get(tool_name)
        if mock is None or not mock.was_called():
            raise AssertionError(f"Tool '{tool_name}' was never called")

    def assert_not_called(self, tool_name: str) -> None:
        """Assert that a tool was never called."""
        mock = self._mocks.get(tool_name)
        if mock is not None and mock.was_called():
            raise AssertionError(
                f"Tool '{tool_name}' was called {mock.call_count} time(s)"
            )

    def assert_call_count(self, tool_name: str, count: int) -> None:
        mock = self._mocks.get(tool_name)
        actual = mock.call_count if mock else 0
        if actual != count:
            raise AssertionError(
                f"Tool '{tool_name}' called {actual} times, expected {count}"
            )

    def assert_call_order(self, *tool_names: str) -> None:
        """Assert tools were called in the specified order."""
        called_tools = [c.tool_name for c in self._global_history]
        idx = 0
        for name in tool_names:
            try:
                pos = called_tools.index(name, idx)
                idx = pos + 1
            except ValueError:
                raise AssertionError(
                    f"Tool '{name}' not found in call order after position {idx}. "
                    f"Actual order: {called_tools}"
                )

    # -- Bulk Operations --

    @property
    def all_calls(self) -> List[ToolCall]:
        return list(self._global_history)

    @property
    def total_calls(self) -> int:
        return len(self._global_history)

    def reset_all(self) -> None:
        """Reset all mocks."""
        for mock in self._mocks.values():
            mock.reset()
        self._global_history.clear()

    def clear(self) -> None:
        """Remove all mocks."""
        self._mocks.clear()
        self._global_history.clear()
