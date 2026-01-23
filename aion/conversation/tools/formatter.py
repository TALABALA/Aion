"""
AION Tool Result Formatter

Formats tool results for display and LLM consumption.
"""

from __future__ import annotations

import json
from typing import Any, Optional
from datetime import datetime
import structlog

logger = structlog.get_logger(__name__)


class ToolResultFormatter:
    """
    Formats tool results for various output contexts.

    Features:
    - JSON formatting with truncation
    - Human-readable formatting
    - Error message formatting
    - Streaming-friendly output
    """

    def __init__(
        self,
        max_result_length: int = 10000,
        max_display_length: int = 500,
        truncation_indicator: str = "... [truncated]",
    ):
        self.max_result_length = max_result_length
        self.max_display_length = max_display_length
        self.truncation_indicator = truncation_indicator

    def format_for_llm(
        self,
        result: Any,
        tool_name: str,
        is_error: bool = False,
    ) -> str:
        """
        Format a tool result for LLM consumption.

        Args:
            result: The tool result
            tool_name: Name of the tool
            is_error: Whether this is an error result

        Returns:
            Formatted string for LLM
        """
        if is_error:
            return f"Error from {tool_name}: {str(result)}"

        if result is None:
            return f"{tool_name} completed successfully (no output)"

        formatted = self._format_value(result)

        if len(formatted) > self.max_result_length:
            formatted = formatted[: self.max_result_length - len(self.truncation_indicator)]
            formatted += self.truncation_indicator

        return formatted

    def format_for_display(
        self,
        result: Any,
        tool_name: str,
        is_error: bool = False,
        include_tool_name: bool = True,
    ) -> str:
        """
        Format a tool result for user display.

        Args:
            result: The tool result
            tool_name: Name of the tool
            is_error: Whether this is an error result
            include_tool_name: Whether to include tool name in output

        Returns:
            Formatted string for display
        """
        prefix = ""
        if include_tool_name:
            status = "Error" if is_error else "Result"
            prefix = f"[{tool_name} {status}] "

        if is_error:
            return f"{prefix}{str(result)}"

        if result is None:
            return f"{prefix}Completed successfully"

        formatted = self._format_value_compact(result)

        max_len = self.max_display_length - len(prefix)
        if len(formatted) > max_len:
            formatted = formatted[: max_len - len(self.truncation_indicator)]
            formatted += self.truncation_indicator

        return f"{prefix}{formatted}"

    def format_for_stream(
        self,
        result: Any,
        tool_name: str,
        is_error: bool = False,
    ) -> dict[str, Any]:
        """
        Format a tool result for streaming output.

        Args:
            result: The tool result
            tool_name: Name of the tool
            is_error: Whether this is an error result

        Returns:
            Dict suitable for streaming event
        """
        return {
            "tool": tool_name,
            "result": self.format_for_display(
                result, tool_name, is_error, include_tool_name=False
            ),
            "is_error": is_error,
            "timestamp": datetime.now().isoformat(),
        }

    def format_error(
        self,
        error: Exception,
        tool_name: str,
        include_traceback: bool = False,
    ) -> str:
        """
        Format an error for output.

        Args:
            error: The exception
            tool_name: Name of the tool
            include_traceback: Whether to include traceback

        Returns:
            Formatted error string
        """
        error_type = type(error).__name__
        error_msg = str(error)

        formatted = f"Error executing {tool_name}: {error_type}: {error_msg}"

        if include_traceback:
            import traceback
            tb = traceback.format_exc()
            formatted += f"\n\nTraceback:\n{tb}"

        return formatted

    def format_multiple_results(
        self,
        results: list[tuple[str, Any, bool]],
    ) -> str:
        """
        Format multiple tool results.

        Args:
            results: List of (tool_name, result, is_error) tuples

        Returns:
            Formatted string with all results
        """
        parts = []

        for tool_name, result, is_error in results:
            formatted = self.format_for_display(
                result, tool_name, is_error, include_tool_name=True
            )
            parts.append(formatted)

        return "\n\n".join(parts)

    def _format_value(self, value: Any) -> str:
        """Format a value to string."""
        if isinstance(value, str):
            return value

        if isinstance(value, (dict, list)):
            try:
                return json.dumps(value, indent=2, default=str)
            except (TypeError, ValueError):
                return str(value)

        if isinstance(value, (int, float, bool)):
            return str(value)

        if hasattr(value, "to_dict"):
            return self._format_value(value.to_dict())

        if hasattr(value, "__dict__"):
            return self._format_value(vars(value))

        return str(value)

    def _format_value_compact(self, value: Any) -> str:
        """Format a value compactly."""
        if isinstance(value, str):
            lines = value.split("\n")
            if len(lines) > 3:
                return "\n".join(lines[:3]) + f"\n... ({len(lines) - 3} more lines)"
            return value

        if isinstance(value, dict):
            if len(value) > 5:
                preview = dict(list(value.items())[:5])
                return json.dumps(preview, default=str) + f" ... ({len(value) - 5} more keys)"
            return json.dumps(value, default=str)

        if isinstance(value, list):
            if len(value) > 5:
                preview = value[:5]
                return json.dumps(preview, default=str) + f" ... ({len(value) - 5} more items)"
            return json.dumps(value, default=str)

        return self._format_value(value)


def format_tool_call_for_log(
    tool_name: str,
    arguments: dict[str, Any],
    result: Optional[Any] = None,
    is_error: bool = False,
    execution_time: Optional[float] = None,
) -> dict[str, Any]:
    """
    Format a tool call for logging.

    Returns a dict suitable for structured logging.
    """
    log_data = {
        "tool_name": tool_name,
        "arguments": {
            k: str(v)[:100] if len(str(v)) > 100 else v
            for k, v in arguments.items()
        },
        "timestamp": datetime.now().isoformat(),
    }

    if result is not None:
        result_str = str(result)
        log_data["result_preview"] = result_str[:200] if len(result_str) > 200 else result_str
        log_data["result_length"] = len(result_str)

    log_data["is_error"] = is_error

    if execution_time is not None:
        log_data["execution_time_ms"] = execution_time * 1000

    return log_data
