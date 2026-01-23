"""
AION Conversation Tool Executor

Executes tools within conversation context.
Provides integration with AION's tool orchestrator.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional
import structlog

from aion.conversation.types import ToolUseContent, ToolResultContent

logger = structlog.get_logger(__name__)


class ToolExecutor:
    """
    Executes tools for the conversation system.

    Features:
    - Integration with AION's tool orchestrator
    - Timeout handling
    - Error recovery
    - Tool result formatting
    - Execution statistics
    """

    def __init__(
        self,
        orchestrator: Optional[Any] = None,
        default_timeout: float = 30.0,
        max_retries: int = 2,
    ):
        self._orchestrator = orchestrator
        self.default_timeout = default_timeout
        self.max_retries = max_retries

        self._execution_count = 0
        self._error_count = 0
        self._total_execution_time = 0.0

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the tool executor."""
        if self._initialized:
            return

        if self._orchestrator and hasattr(self._orchestrator, "initialize"):
            await self._orchestrator.initialize()

        self._initialized = True
        logger.info("Tool executor initialized")

    async def shutdown(self) -> None:
        """Shutdown the tool executor."""
        self._initialized = False

    async def get_tool_definitions(
        self,
        allowed: Optional[list[str]] = None,
        categories: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        """
        Get tool definitions for LLM.

        Returns tools in Claude's expected format.

        Args:
            allowed: List of allowed tool names (None = all)
            categories: Filter by tool categories

        Returns:
            List of tool definitions
        """
        if not self._orchestrator:
            return []

        try:
            if hasattr(self._orchestrator, "registry"):
                tools = self._orchestrator.registry.list_tools()
            elif hasattr(self._orchestrator, "list_tools"):
                tools = self._orchestrator.list_tools()
            elif hasattr(self._orchestrator, "get_tools"):
                tools = self._orchestrator.get_tools()
            else:
                return []

            if allowed:
                tools = [t for t in tools if getattr(t, "name", "") in allowed]

            if categories:
                tools = [
                    t for t in tools
                    if getattr(t, "category", None) in categories
                ]

            return [self._format_tool_definition(tool) for tool in tools]

        except Exception as e:
            logger.error(f"Error getting tool definitions: {e}")
            return []

    async def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        timeout: Optional[float] = None,
    ) -> dict[str, Any]:
        """
        Execute a tool.

        Args:
            tool_name: Name of the tool
            arguments: Tool arguments
            timeout: Execution timeout in seconds

        Returns:
            Tool result dict with 'result' and 'is_error' keys
        """
        if not self._orchestrator:
            return {
                "result": "Tool execution not available",
                "is_error": True,
            }

        timeout = timeout or self.default_timeout
        start_time = asyncio.get_event_loop().time()

        self._execution_count += 1

        for attempt in range(self.max_retries + 1):
            try:
                result = await asyncio.wait_for(
                    self._execute_tool(tool_name, arguments),
                    timeout=timeout,
                )

                execution_time = asyncio.get_event_loop().time() - start_time
                self._total_execution_time += execution_time

                logger.debug(
                    "Tool executed",
                    tool=tool_name,
                    execution_time=execution_time,
                    attempt=attempt + 1,
                )

                return {
                    "result": result,
                    "is_error": False,
                    "execution_time": execution_time,
                }

            except asyncio.TimeoutError:
                logger.warning(
                    f"Tool execution timeout: {tool_name}",
                    timeout=timeout,
                    attempt=attempt + 1,
                )
                if attempt == self.max_retries:
                    self._error_count += 1
                    return {
                        "result": f"Tool execution timed out after {timeout}s",
                        "is_error": True,
                    }

            except Exception as e:
                logger.error(
                    f"Tool execution error: {tool_name}",
                    error=str(e),
                    attempt=attempt + 1,
                )
                if attempt == self.max_retries:
                    self._error_count += 1
                    return {
                        "result": f"Error executing {tool_name}: {str(e)}",
                        "is_error": True,
                    }

        return {
            "result": f"Tool execution failed after {self.max_retries + 1} attempts",
            "is_error": True,
        }

    async def execute_tool_use(
        self,
        tool_use: ToolUseContent,
        timeout: Optional[float] = None,
    ) -> ToolResultContent:
        """
        Execute a tool from a ToolUseContent block.

        Args:
            tool_use: The tool use content
            timeout: Execution timeout

        Returns:
            ToolResultContent with the result
        """
        result = await self.execute(
            tool_name=tool_use.name,
            arguments=tool_use.input,
            timeout=timeout,
        )

        return ToolResultContent(
            tool_use_id=tool_use.id,
            content=str(result.get("result", "")),
            is_error=result.get("is_error", False),
        )

    async def execute_parallel(
        self,
        tool_calls: list[tuple[str, dict[str, Any]]],
        timeout: Optional[float] = None,
    ) -> list[dict[str, Any]]:
        """
        Execute multiple tools in parallel.

        Args:
            tool_calls: List of (tool_name, arguments) tuples
            timeout: Execution timeout per tool

        Returns:
            List of results in the same order as tool_calls
        """
        tasks = [
            self.execute(name, args, timeout)
            for name, args in tool_calls
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "result": f"Error: {str(result)}",
                    "is_error": True,
                })
            else:
                processed_results.append(result)

        return processed_results

    def get_stats(self) -> dict[str, Any]:
        """Get executor statistics."""
        avg_time = (
            self._total_execution_time / self._execution_count
            if self._execution_count > 0
            else 0.0
        )

        return {
            "total_executions": self._execution_count,
            "error_count": self._error_count,
            "error_rate": (
                self._error_count / self._execution_count
                if self._execution_count > 0
                else 0.0
            ),
            "total_execution_time": self._total_execution_time,
            "avg_execution_time": avg_time,
        }

    async def _execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> Any:
        """Internal method to execute a tool."""
        if hasattr(self._orchestrator, "execute"):
            return await self._orchestrator.execute(tool_name, arguments)
        elif hasattr(self._orchestrator, "run"):
            return await self._orchestrator.run(tool_name, **arguments)
        elif hasattr(self._orchestrator, "call"):
            return await self._orchestrator.call(tool_name, arguments)
        else:
            raise RuntimeError(f"Orchestrator does not support tool execution")

    def _format_tool_definition(self, tool: Any) -> dict[str, Any]:
        """Format a tool for Claude's API."""
        if isinstance(tool, dict):
            return {
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "input_schema": tool.get("parameters", tool.get("input_schema", {
                    "type": "object",
                    "properties": {},
                    "required": [],
                })),
            }

        name = getattr(tool, "name", "")
        description = getattr(tool, "description", "")

        parameters = getattr(tool, "parameters", None)
        if parameters is None:
            parameters = getattr(tool, "input_schema", None)
        if parameters is None:
            parameters = {
                "type": "object",
                "properties": {},
                "required": [],
            }

        if isinstance(parameters, dict) and "type" not in parameters:
            parameters = {
                "type": "object",
                "properties": parameters,
                "required": list(parameters.keys()),
            }

        return {
            "name": name,
            "description": description,
            "input_schema": parameters,
        }


class ToolCallTracker:
    """
    Tracks tool calls within a conversation turn.
    """

    def __init__(self):
        self._calls: list[dict[str, Any]] = []
        self._results: dict[str, Any] = {}

    def add_call(
        self,
        tool_id: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> None:
        """Record a tool call."""
        self._calls.append({
            "id": tool_id,
            "name": tool_name,
            "arguments": arguments,
            "status": "pending",
        })

    def add_result(
        self,
        tool_id: str,
        result: Any,
        is_error: bool = False,
    ) -> None:
        """Record a tool result."""
        self._results[tool_id] = {
            "result": result,
            "is_error": is_error,
        }

        for call in self._calls:
            if call["id"] == tool_id:
                call["status"] = "error" if is_error else "completed"
                break

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of tool calls."""
        completed = sum(1 for c in self._calls if c["status"] == "completed")
        errors = sum(1 for c in self._calls if c["status"] == "error")

        return {
            "total_calls": len(self._calls),
            "completed": completed,
            "errors": errors,
            "pending": len(self._calls) - completed - errors,
            "tools_used": list(set(c["name"] for c in self._calls)),
        }

    def reset(self) -> None:
        """Reset the tracker."""
        self._calls.clear()
        self._results.clear()
