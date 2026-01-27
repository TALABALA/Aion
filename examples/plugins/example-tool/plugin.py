"""
Example Tool Plugin for AION

This plugin demonstrates how to create a tool plugin for AION,
including:
- Tool definitions with parameters
- Hook handlers
- Configuration handling
- Health checks
- Proper lifecycle management
"""

from typing import Any, Dict, List, Optional
import time
import hashlib

from aion.plugins import (
    ToolPlugin,
    PluginManifest,
    PluginType,
    SemanticVersion,
    Tool,
    ToolParameter,
    ToolParameterType,
    ToolResult,
    PluginPermissions,
    PermissionLevel,
)


class ExampleToolPlugin(ToolPlugin):
    """
    Example tool plugin demonstrating AION plugin development.

    This plugin provides several example tools:
    - greet: Generate personalized greetings
    - calculate: Perform basic calculations
    - hash_text: Generate hashes of text
    - reverse_text: Reverse a string
    """

    def __init__(self):
        super().__init__()
        self._call_count = 0
        self._total_execution_time = 0.0

    @classmethod
    def get_manifest(cls) -> PluginManifest:
        """Return the plugin manifest."""
        return PluginManifest(
            id="example-tool",
            name="Example Tool Plugin",
            version=SemanticVersion(1, 0, 0),
            description="A comprehensive example demonstrating AION plugin development",
            plugin_type=PluginType.TOOL,
            entry_point="plugin:ExampleToolPlugin",
            tags=["example", "demo", "tools", "getting-started"],
            features=["greeting", "calculation", "text-processing"],
            hooks=["tool.before_execute", "request.before"],
            permissions=PluginPermissions(
                level=PermissionLevel.STANDARD,
                memory_access=True,
            ),
            config_schema={
                "type": "object",
                "properties": {
                    "greeting_prefix": {
                        "type": "string",
                        "description": "Prefix for greeting messages",
                        "default": "Hello",
                    },
                    "enable_logging": {
                        "type": "boolean",
                        "description": "Enable verbose logging",
                        "default": False,
                    },
                    "max_retries": {
                        "type": "integer",
                        "description": "Maximum retry attempts",
                        "minimum": 0,
                        "maximum": 10,
                        "default": 3,
                    },
                },
            },
            default_config={
                "greeting_prefix": "Hello",
                "enable_logging": False,
                "max_retries": 3,
            },
        )

    def get_tools(self) -> List[Tool]:
        """Return list of tools provided by this plugin."""
        return [
            Tool(
                name="greet",
                description="Generate a personalized greeting message",
                parameters=[
                    ToolParameter(
                        name="name",
                        type=ToolParameterType.STRING,
                        description="Name of the person to greet",
                        required=True,
                        min_length=1,
                        max_length=100,
                    ),
                    ToolParameter(
                        name="formal",
                        type=ToolParameterType.BOOLEAN,
                        description="Use formal greeting style",
                        default=False,
                    ),
                ],
                returns=ToolParameter(
                    name="greeting",
                    type=ToolParameterType.STRING,
                    description="The greeting message",
                ),
                category="communication",
                tags=["greeting", "text"],
                examples=[
                    {
                        "input": {"name": "Alice", "formal": False},
                        "output": "Hello, Alice!",
                    },
                    {
                        "input": {"name": "Dr. Smith", "formal": True},
                        "output": "Good day, Dr. Smith. It is a pleasure to meet you.",
                    },
                ],
            ),
            Tool(
                name="calculate",
                description="Perform basic arithmetic calculations",
                parameters=[
                    ToolParameter(
                        name="operation",
                        type=ToolParameterType.STRING,
                        description="Operation to perform",
                        required=True,
                        enum=["add", "subtract", "multiply", "divide"],
                    ),
                    ToolParameter(
                        name="a",
                        type=ToolParameterType.FLOAT,
                        description="First operand",
                        required=True,
                    ),
                    ToolParameter(
                        name="b",
                        type=ToolParameterType.FLOAT,
                        description="Second operand",
                        required=True,
                    ),
                ],
                returns=ToolParameter(
                    name="result",
                    type=ToolParameterType.FLOAT,
                    description="Calculation result",
                ),
                category="math",
                tags=["math", "calculation"],
            ),
            Tool(
                name="hash_text",
                description="Generate a hash of the input text",
                parameters=[
                    ToolParameter(
                        name="text",
                        type=ToolParameterType.STRING,
                        description="Text to hash",
                        required=True,
                    ),
                    ToolParameter(
                        name="algorithm",
                        type=ToolParameterType.STRING,
                        description="Hash algorithm to use",
                        default="sha256",
                        enum=["md5", "sha1", "sha256", "sha512"],
                    ),
                ],
                returns=ToolParameter(
                    name="hash",
                    type=ToolParameterType.STRING,
                    description="Hexadecimal hash string",
                ),
                category="security",
                tags=["hash", "security", "crypto"],
            ),
            Tool(
                name="reverse_text",
                description="Reverse a string",
                parameters=[
                    ToolParameter(
                        name="text",
                        type=ToolParameterType.STRING,
                        description="Text to reverse",
                        required=True,
                    ),
                    ToolParameter(
                        name="by_word",
                        type=ToolParameterType.BOOLEAN,
                        description="Reverse word order instead of characters",
                        default=False,
                    ),
                ],
                returns=ToolParameter(
                    name="reversed",
                    type=ToolParameterType.STRING,
                    description="Reversed text",
                ),
                category="text",
                tags=["text", "string", "manipulation"],
            ),
        ]

    async def execute(
        self,
        tool_name: str,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Execute a tool by name."""
        start_time = time.time()
        self._call_count += 1

        try:
            if tool_name == "greet":
                return await self._greet(params)
            elif tool_name == "calculate":
                return await self._calculate(params)
            elif tool_name == "hash_text":
                return await self._hash_text(params)
            elif tool_name == "reverse_text":
                return await self._reverse_text(params)
            else:
                raise ValueError(f"Unknown tool: {tool_name}")
        finally:
            self._total_execution_time += (time.time() - start_time) * 1000

    async def _greet(self, params: Dict[str, Any]) -> str:
        """Generate a greeting."""
        name = params["name"]
        formal = params.get("formal", False)
        prefix = self._config.get("greeting_prefix", "Hello")

        if formal:
            return f"Good day, {name}. It is a pleasure to meet you."
        return f"{prefix}, {name}!"

    async def _calculate(self, params: Dict[str, Any]) -> float:
        """Perform calculation."""
        operation = params["operation"]
        a = float(params["a"])
        b = float(params["b"])

        if operation == "add":
            return a + b
        elif operation == "subtract":
            return a - b
        elif operation == "multiply":
            return a * b
        elif operation == "divide":
            if b == 0:
                raise ValueError("Division by zero")
            return a / b
        else:
            raise ValueError(f"Unknown operation: {operation}")

    async def _hash_text(self, params: Dict[str, Any]) -> str:
        """Generate hash of text."""
        text = params["text"]
        algorithm = params.get("algorithm", "sha256")

        if algorithm == "md5":
            return hashlib.md5(text.encode()).hexdigest()
        elif algorithm == "sha1":
            return hashlib.sha1(text.encode()).hexdigest()
        elif algorithm == "sha256":
            return hashlib.sha256(text.encode()).hexdigest()
        elif algorithm == "sha512":
            return hashlib.sha512(text.encode()).hexdigest()
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    async def _reverse_text(self, params: Dict[str, Any]) -> str:
        """Reverse text."""
        text = params["text"]
        by_word = params.get("by_word", False)

        if by_word:
            return " ".join(text.split()[::-1])
        return text[::-1]

    # === Lifecycle ===

    async def initialize(self, kernel, config: Dict[str, Any]) -> None:
        """Initialize the plugin."""
        self._kernel = kernel
        self._config = config
        self._initialized = True
        self._call_count = 0
        self._total_execution_time = 0.0

        if config.get("enable_logging"):
            import structlog
            logger = structlog.get_logger(__name__)
            logger.info(f"Example tool plugin initialized with config: {config}")

    async def shutdown(self) -> None:
        """Shutdown the plugin."""
        self._initialized = False

        if self._config.get("enable_logging"):
            import structlog
            logger = structlog.get_logger(__name__)
            logger.info(
                f"Example tool plugin shutting down. "
                f"Total calls: {self._call_count}, "
                f"Total time: {self._total_execution_time:.2f}ms"
            )

    async def health_check(self) -> Dict[str, Any]:
        """Return health status."""
        return {
            "healthy": self._initialized,
            "initialized": self._initialized,
            "active": self._active,
            "stats": {
                "call_count": self._call_count,
                "total_execution_time_ms": self._total_execution_time,
                "average_execution_time_ms": (
                    self._total_execution_time / self._call_count
                    if self._call_count > 0
                    else 0
                ),
            },
        }

    # === Hook Handlers ===

    async def hook_tool_before_execute(
        self,
        tool_name: str,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Hook handler called before any tool executes.

        This can modify the parameters before execution.
        """
        if self._config.get("enable_logging"):
            import structlog
            logger = structlog.get_logger(__name__)
            logger.debug(f"Tool executing: {tool_name} with params: {params}")

        # Return params (can be modified)
        return params

    async def hook_request_before(
        self,
        request: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Hook handler called before request processing."""
        # Example: Add metadata to requests
        if "metadata" not in request:
            request["metadata"] = {}
        request["metadata"]["example_plugin_processed"] = True
        return request
