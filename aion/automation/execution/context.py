"""
AION Execution Context

Context management for workflow execution.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from aion.automation.types import Workflow, WorkflowExecution

logger = structlog.get_logger(__name__)


class ExecutionContext:
    """
    Execution context for workflows.

    Features:
    - Variable storage and retrieval
    - Expression resolution with {{ }} syntax
    - Step output access
    - Nested path support
    - Built-in variables (now, trigger, inputs, etc.)
    """

    # Expression pattern for {{ variable }}
    EXPRESSION_PATTERN = re.compile(r'\{\{\s*([^}]+)\s*\}\}')

    def __init__(
        self,
        execution: "WorkflowExecution",
        workflow: "Workflow",
    ):
        self.execution = execution
        self.workflow = workflow
        self.current_step_id: Optional[str] = None

        # Context data
        self._data: Dict[str, Any] = {}

        # Initialize built-in variables
        self._initialize_builtins()

    def _initialize_builtins(self) -> None:
        """Initialize built-in context variables."""
        # Workflow info
        self._data["workflow"] = {
            "id": self.workflow.id,
            "name": self.workflow.name,
            "version": self.workflow.version,
        }

        # Execution info
        self._data["execution"] = {
            "id": self.execution.id,
            "attempt": self.execution.attempt,
            "initiated_by": self.execution.initiated_by,
        }

        # Trigger data
        self._data["trigger"] = self.execution.trigger_data or {}

        # Inputs
        self._data["inputs"] = self.execution.inputs or {}

        # Steps (will be populated during execution)
        self._data["steps"] = {}

        # Outputs (will be populated during execution)
        self._data["outputs"] = {}

        # Environment (placeholder for env vars)
        self._data["env"] = {}

        # Secrets (placeholder for secure values)
        self._data["secrets"] = {}

    # === Data Access ===

    def set(self, key: str, value: Any) -> None:
        """Set a context value using dot notation."""
        self._set_nested(self._data, key.split("."), value)
        logger.debug("context_set", key=key)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a context value using dot notation."""
        return self._get_nested(self._data, key.split("."), default)

    def delete(self, key: str) -> None:
        """Delete a context value."""
        parts = key.split(".")
        if len(parts) == 1:
            self._data.pop(key, None)
        else:
            parent = self._get_nested(self._data, parts[:-1])
            if isinstance(parent, dict):
                parent.pop(parts[-1], None)

    def has(self, key: str) -> bool:
        """Check if a key exists."""
        value = self.get(key)
        return value is not None

    def update(self, data: Dict[str, Any]) -> None:
        """Update context with multiple values."""
        for key, value in data.items():
            self.set(key, value)

    # === Expression Resolution ===

    def resolve(self, value: Any) -> Any:
        """
        Resolve expressions in a value.

        Supports:
        - {{ variable }} - Simple variable
        - {{ inputs.name }} - Nested access
        - {{ steps.step_id.output }} - Step output
        - {{ now }} - Current datetime
        - {{ trigger.body.key }} - Trigger data
        """
        if value is None:
            return None

        if isinstance(value, str):
            return self._resolve_string(value)

        if isinstance(value, dict):
            return {k: self.resolve(v) for k, v in value.items()}

        if isinstance(value, list):
            return [self.resolve(item) for item in value]

        return value

    def _resolve_string(self, value: str) -> Any:
        """Resolve expressions in a string."""
        # Check for expression pattern
        # If entire value is a single expression, return resolved value directly
        match = self.EXPRESSION_PATTERN.fullmatch(value.strip())
        if match:
            return self._evaluate_expression(match.group(1).strip())

        # Otherwise, do string interpolation
        def replace(match):
            expr = match.group(1).strip()
            resolved = self._evaluate_expression(expr)
            return str(resolved) if resolved is not None else ""

        return self.EXPRESSION_PATTERN.sub(replace, value)

    def _evaluate_expression(self, expr: str) -> Any:
        """Evaluate a single expression."""
        # Handle special expressions
        if expr == "now":
            return datetime.now().isoformat()

        if expr == "today":
            return datetime.now().date().isoformat()

        if expr.startswith("now."):
            # Access datetime properties
            now = datetime.now()
            prop = expr[4:]
            if prop == "year":
                return now.year
            elif prop == "month":
                return now.month
            elif prop == "day":
                return now.day
            elif prop == "hour":
                return now.hour
            elif prop == "minute":
                return now.minute
            elif prop == "timestamp":
                return now.timestamp()

        # Handle pipe operations (simple filters)
        if "|" in expr:
            parts = expr.split("|")
            value = self.get(parts[0].strip())
            for filter_expr in parts[1:]:
                value = self._apply_filter(value, filter_expr.strip())
            return value

        # Simple variable lookup
        return self.get(expr)

    def _apply_filter(self, value: Any, filter_expr: str) -> Any:
        """Apply a filter to a value."""
        if filter_expr == "upper":
            return str(value).upper() if value else ""

        if filter_expr == "lower":
            return str(value).lower() if value else ""

        if filter_expr == "trim":
            return str(value).strip() if value else ""

        if filter_expr == "length":
            return len(value) if value else 0

        if filter_expr == "first":
            return value[0] if value and len(value) > 0 else None

        if filter_expr == "last":
            return value[-1] if value and len(value) > 0 else None

        if filter_expr == "json":
            import json
            return json.dumps(value)

        if filter_expr == "keys":
            return list(value.keys()) if isinstance(value, dict) else []

        if filter_expr == "values":
            return list(value.values()) if isinstance(value, dict) else []

        if filter_expr.startswith("default("):
            # default(fallback_value)
            match = re.match(r'default\(([^)]+)\)', filter_expr)
            if match and value is None:
                return match.group(1).strip("'\"")
            return value

        if filter_expr.startswith("slice("):
            # slice(start, end)
            match = re.match(r'slice\((\d+),\s*(\d+)\)', filter_expr)
            if match and isinstance(value, (list, str)):
                start, end = int(match.group(1)), int(match.group(2))
                return value[start:end]
            return value

        return value

    # === Step Output Management ===

    def set_step_output(self, step_id: str, output: Any) -> None:
        """Set output for a step."""
        self._data["steps"][step_id] = {"output": output}

    def get_step_output(self, step_id: str) -> Any:
        """Get output for a step."""
        step_data = self._data["steps"].get(step_id, {})
        return step_data.get("output")

    def set_output(self, key: str, value: Any) -> None:
        """Set a workflow output."""
        self._data["outputs"][key] = value

    def get_outputs(self) -> Dict[str, Any]:
        """Get all workflow outputs."""
        return self._data.get("outputs", {})

    # === Loop Support ===

    def enter_loop(self, variable: str, index_variable: str) -> None:
        """Enter a loop context."""
        if "loops" not in self._data:
            self._data["loops"] = []
        self._data["loops"].append({
            "variable": variable,
            "index_variable": index_variable,
        })

    def set_loop_item(self, variable: str, index: int, item: Any) -> None:
        """Set current loop item."""
        self._data[variable] = item
        self._data["index"] = index  # Also set generic 'index'

    def exit_loop(self) -> None:
        """Exit a loop context."""
        if "loops" in self._data and self._data["loops"]:
            loop = self._data["loops"].pop()
            self._data.pop(loop["variable"], None)
            self._data.pop(loop["index_variable"], None)

    # === Nested Access Helpers ===

    def _set_nested(
        self,
        data: Dict[str, Any],
        parts: List[str],
        value: Any,
    ) -> None:
        """Set a nested value."""
        if len(parts) == 1:
            data[parts[0]] = value
        else:
            if parts[0] not in data:
                data[parts[0]] = {}
            self._set_nested(data[parts[0]], parts[1:], value)

    def _get_nested(
        self,
        data: Any,
        parts: List[str],
        default: Any = None,
    ) -> Any:
        """Get a nested value."""
        if not parts:
            return data

        if data is None:
            return default

        if isinstance(data, dict):
            if parts[0] not in data:
                return default
            return self._get_nested(data[parts[0]], parts[1:], default)

        if isinstance(data, list):
            try:
                index = int(parts[0])
                if 0 <= index < len(data):
                    return self._get_nested(data[index], parts[1:], default)
            except ValueError:
                pass
            return default

        # Try attribute access for objects
        if hasattr(data, parts[0]):
            return self._get_nested(getattr(data, parts[0]), parts[1:], default)

        return default

    # === Serialization ===

    def to_dict(self) -> Dict[str, Any]:
        """Export context as dictionary."""
        return self._data.copy()

    def to_safe_dict(self) -> Dict[str, Any]:
        """Export context without sensitive data."""
        data = self._data.copy()
        data.pop("secrets", None)
        return data

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        execution: "WorkflowExecution",
        workflow: "Workflow",
    ) -> "ExecutionContext":
        """Create context from dictionary."""
        context = cls(execution, workflow)
        context._data.update(data)
        return context

    def __repr__(self) -> str:
        """String representation."""
        return f"ExecutionContext(execution={self.execution.id}, keys={list(self._data.keys())})"
