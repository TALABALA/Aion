"""
AION Tool Plugin Interface

Interface for plugins that provide tools.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union
from enum import Enum

from aion.plugins.interfaces.base import BasePlugin
from aion.plugins.types import PluginManifest, PluginType, SemanticVersion


class ToolParameterType(str, Enum):
    """Supported tool parameter types."""

    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"
    FILE = "file"
    ANY = "any"


@dataclass
class ToolParameter:
    """Definition of a tool parameter."""

    name: str
    type: Union[ToolParameterType, str] = ToolParameterType.STRING
    description: str = ""
    required: bool = False
    default: Any = None
    enum: Optional[List[Any]] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    pattern: Optional[str] = None  # Regex pattern for strings
    items_type: Optional[str] = None  # Type for array items
    properties: Optional[Dict[str, "ToolParameter"]] = None  # For object types

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result: Dict[str, Any] = {
            "name": self.name,
            "type": self.type.value if isinstance(self.type, ToolParameterType) else self.type,
            "description": self.description,
            "required": self.required,
        }
        if self.default is not None:
            result["default"] = self.default
        if self.enum:
            result["enum"] = self.enum
        if self.min_value is not None:
            result["minimum"] = self.min_value
        if self.max_value is not None:
            result["maximum"] = self.max_value
        if self.min_length is not None:
            result["minLength"] = self.min_length
        if self.max_length is not None:
            result["maxLength"] = self.max_length
        if self.pattern:
            result["pattern"] = self.pattern
        return result

    def to_json_schema(self) -> Dict[str, Any]:
        """Convert to JSON Schema format."""
        type_mapping = {
            ToolParameterType.STRING: "string",
            ToolParameterType.INTEGER: "integer",
            ToolParameterType.FLOAT: "number",
            ToolParameterType.BOOLEAN: "boolean",
            ToolParameterType.ARRAY: "array",
            ToolParameterType.OBJECT: "object",
            ToolParameterType.FILE: "string",
            ToolParameterType.ANY: {},
        }

        param_type = self.type if isinstance(self.type, ToolParameterType) else ToolParameterType(self.type)
        schema: Dict[str, Any] = {"type": type_mapping.get(param_type, "string")}

        if self.description:
            schema["description"] = self.description
        if self.default is not None:
            schema["default"] = self.default
        if self.enum:
            schema["enum"] = self.enum
        if self.min_value is not None:
            schema["minimum"] = self.min_value
        if self.max_value is not None:
            schema["maximum"] = self.max_value
        if self.min_length is not None:
            schema["minLength"] = self.min_length
        if self.max_length is not None:
            schema["maxLength"] = self.max_length
        if self.pattern:
            schema["pattern"] = self.pattern
        if self.items_type:
            schema["items"] = {"type": self.items_type}

        return schema


@dataclass
class ToolResult:
    """Result of a tool execution."""

    success: bool
    output: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "metadata": self.metadata,
            "execution_time_ms": self.execution_time_ms,
        }


@dataclass
class Tool:
    """
    Definition of a tool provided by a plugin.

    Tools are callable units of functionality that can be
    invoked by agents, planners, and workflows.
    """

    name: str
    description: str
    parameters: List[ToolParameter] = field(default_factory=list)
    returns: Optional[ToolParameter] = None
    handler: Optional[Callable] = None
    category: str = "general"
    tags: List[str] = field(default_factory=list)
    examples: List[Dict[str, Any]] = field(default_factory=list)
    deprecated: bool = False
    deprecation_message: str = ""
    requires_confirmation: bool = False
    is_async: bool = True
    timeout_seconds: float = 30.0
    rate_limit: Optional[int] = None  # Max calls per minute
    cost_estimate: float = 0.0  # Estimated cost per call

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": [p.to_dict() for p in self.parameters],
            "returns": self.returns.to_dict() if self.returns else None,
            "category": self.category,
            "tags": self.tags,
            "examples": self.examples,
            "deprecated": self.deprecated,
            "requires_confirmation": self.requires_confirmation,
            "timeout_seconds": self.timeout_seconds,
        }

    def to_openai_function(self) -> Dict[str, Any]:
        """Convert to OpenAI function calling format."""
        properties = {}
        required = []

        for param in self.parameters:
            properties[param.name] = param.to_json_schema()
            if param.required:
                required.append(param.name)

        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }

    def validate_params(self, params: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        Validate parameters against tool definition.

        Returns:
            (is_valid, error_messages)
        """
        errors: List[str] = []
        param_defs = {p.name: p for p in self.parameters}

        # Check required parameters
        for param in self.parameters:
            if param.required and param.name not in params:
                errors.append(f"Missing required parameter: {param.name}")

        # Validate provided parameters
        for name, value in params.items():
            if name not in param_defs:
                errors.append(f"Unknown parameter: {name}")
                continue

            param = param_defs[name]

            # Type validation (basic)
            if param.type == ToolParameterType.STRING and not isinstance(value, str):
                errors.append(f"Parameter '{name}' must be a string")
            elif param.type == ToolParameterType.INTEGER and not isinstance(value, int):
                errors.append(f"Parameter '{name}' must be an integer")
            elif param.type == ToolParameterType.FLOAT and not isinstance(value, (int, float)):
                errors.append(f"Parameter '{name}' must be a number")
            elif param.type == ToolParameterType.BOOLEAN and not isinstance(value, bool):
                errors.append(f"Parameter '{name}' must be a boolean")
            elif param.type == ToolParameterType.ARRAY and not isinstance(value, list):
                errors.append(f"Parameter '{name}' must be an array")
            elif param.type == ToolParameterType.OBJECT and not isinstance(value, dict):
                errors.append(f"Parameter '{name}' must be an object")

            # Enum validation
            if param.enum and value not in param.enum:
                errors.append(f"Parameter '{name}' must be one of: {param.enum}")

            # Range validation
            if isinstance(value, (int, float)):
                if param.min_value is not None and value < param.min_value:
                    errors.append(f"Parameter '{name}' must be >= {param.min_value}")
                if param.max_value is not None and value > param.max_value:
                    errors.append(f"Parameter '{name}' must be <= {param.max_value}")

            # String length validation
            if isinstance(value, str):
                if param.min_length is not None and len(value) < param.min_length:
                    errors.append(f"Parameter '{name}' must be at least {param.min_length} characters")
                if param.max_length is not None and len(value) > param.max_length:
                    errors.append(f"Parameter '{name}' must be at most {param.max_length} characters")

        return len(errors) == 0, errors


class ToolPlugin(BasePlugin):
    """
    Interface for tool plugins.

    Tool plugins provide one or more tools that can be
    used by agents, planners, and workflows.

    Implement:
    - get_tools(): Return list of Tool definitions
    - execute(): Execute a tool by name
    """

    @abstractmethod
    def get_tools(self) -> List[Tool]:
        """
        Return list of tools provided by this plugin.

        Returns:
            List of Tool definitions
        """
        pass

    @abstractmethod
    async def execute(
        self,
        tool_name: str,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Execute a tool.

        Args:
            tool_name: Name of the tool to execute
            params: Tool parameters
            context: Execution context (optional)

        Returns:
            Tool execution result
        """
        pass

    def validate_params(
        self,
        tool_name: str,
        params: Dict[str, Any],
    ) -> tuple[bool, Optional[str]]:
        """
        Validate tool parameters.

        Args:
            tool_name: Tool to validate for
            params: Parameters to validate

        Returns:
            (is_valid, error_message)
        """
        tools = {t.name: t for t in self.get_tools()}
        tool = tools.get(tool_name)

        if not tool:
            return False, f"Unknown tool: {tool_name}"

        valid, errors = tool.validate_params(params)
        if not valid:
            return False, "; ".join(errors)

        return True, None

    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a specific tool by name."""
        for tool in self.get_tools():
            if tool.name == name:
                return tool
        return None

    async def execute_with_result(
        self,
        tool_name: str,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        """
        Execute a tool and return a structured result.

        Args:
            tool_name: Name of the tool to execute
            params: Tool parameters
            context: Execution context (optional)

        Returns:
            ToolResult with output and metadata
        """
        import time

        start_time = time.time()

        try:
            # Validate parameters
            valid, error = self.validate_params(tool_name, params)
            if not valid:
                return ToolResult(
                    success=False,
                    error=error,
                )

            # Execute
            output = await self.execute(tool_name, params, context)

            return ToolResult(
                success=True,
                output=output,
                execution_time_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e),
                execution_time_ms=(time.time() - start_time) * 1000,
            )


# === Example Implementation ===


class ExampleToolPlugin(ToolPlugin):
    """Example tool plugin implementation."""

    @classmethod
    def get_manifest(cls) -> PluginManifest:
        return PluginManifest(
            id="example-tool-plugin",
            name="Example Tool Plugin",
            version=SemanticVersion(1, 0, 0),
            description="An example tool plugin demonstrating the interface",
            plugin_type=PluginType.TOOL,
            entry_point="example_tool_plugin:ExampleToolPlugin",
            tags=["example", "demo"],
        )

    def get_tools(self) -> List[Tool]:
        return [
            Tool(
                name="echo",
                description="Echo back the input message",
                parameters=[
                    ToolParameter(
                        name="message",
                        type=ToolParameterType.STRING,
                        description="Message to echo",
                        required=True,
                    ),
                    ToolParameter(
                        name="uppercase",
                        type=ToolParameterType.BOOLEAN,
                        description="Convert to uppercase",
                        default=False,
                    ),
                ],
                returns=ToolParameter(
                    name="result",
                    type=ToolParameterType.STRING,
                    description="Echoed message",
                ),
                category="utility",
                tags=["echo", "debug"],
                examples=[
                    {"input": {"message": "Hello"}, "output": "Echo: Hello"},
                ],
            ),
            Tool(
                name="add_numbers",
                description="Add two numbers together",
                parameters=[
                    ToolParameter(
                        name="a",
                        type=ToolParameterType.FLOAT,
                        description="First number",
                        required=True,
                    ),
                    ToolParameter(
                        name="b",
                        type=ToolParameterType.FLOAT,
                        description="Second number",
                        required=True,
                    ),
                ],
                returns=ToolParameter(
                    name="sum",
                    type=ToolParameterType.FLOAT,
                    description="Sum of the numbers",
                ),
                category="math",
                tags=["math", "arithmetic"],
            ),
        ]

    async def execute(
        self,
        tool_name: str,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Any:
        if tool_name == "echo":
            message = params["message"]
            if params.get("uppercase", False):
                message = message.upper()
            return f"Echo: {message}"

        elif tool_name == "add_numbers":
            return params["a"] + params["b"]

        raise ValueError(f"Unknown tool: {tool_name}")

    async def initialize(self, kernel, config: Dict[str, Any]) -> None:
        self._kernel = kernel
        self._config = config
        self._initialized = True

    async def shutdown(self) -> None:
        self._initialized = False
