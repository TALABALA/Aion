"""
AION Tool Registry

Extensible tool registry with:
- Tool definition and validation
- Parameter schemas
- Rate limiting
- Performance tracking
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional, Union

import structlog

logger = structlog.get_logger(__name__)


class ToolCategory(str, Enum):
    """Categories of tools."""
    WEB = "web"           # Web browsing, APIs
    FILE = "file"         # File operations
    CODE = "code"         # Code execution
    DATA = "data"         # Data processing
    SEARCH = "search"     # Search operations
    COMPUTE = "compute"   # Computation/math
    SYSTEM = "system"     # System operations
    CUSTOM = "custom"     # Custom tools


class ParameterType(str, Enum):
    """Parameter types for tools."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"
    FILE = "file"


@dataclass
class ToolParameter:
    """Definition of a tool parameter."""
    name: str
    type: ParameterType
    description: str
    required: bool = True
    default: Any = None
    enum: Optional[list[Any]] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    pattern: Optional[str] = None  # Regex pattern for strings

    def validate(self, value: Any) -> tuple[bool, Optional[str]]:
        """
        Validate a parameter value.

        Returns:
            Tuple of (is_valid, error_message)
        """
        if value is None:
            if self.required and self.default is None:
                return False, f"Required parameter '{self.name}' is missing"
            return True, None

        # Type validation
        type_map = {
            ParameterType.STRING: str,
            ParameterType.INTEGER: int,
            ParameterType.FLOAT: (int, float),
            ParameterType.BOOLEAN: bool,
            ParameterType.ARRAY: (list, tuple),
            ParameterType.OBJECT: dict,
        }

        expected_type = type_map.get(self.type)
        if expected_type and not isinstance(value, expected_type):
            return False, f"Parameter '{self.name}' must be {self.type.value}"

        # Enum validation
        if self.enum and value not in self.enum:
            return False, f"Parameter '{self.name}' must be one of {self.enum}"

        # Range validation
        if self.type in (ParameterType.INTEGER, ParameterType.FLOAT):
            if self.min_value is not None and value < self.min_value:
                return False, f"Parameter '{self.name}' must be >= {self.min_value}"
            if self.max_value is not None and value > self.max_value:
                return False, f"Parameter '{self.name}' must be <= {self.max_value}"

        # Pattern validation
        if self.pattern and self.type == ParameterType.STRING:
            import re
            if not re.match(self.pattern, value):
                return False, f"Parameter '{self.name}' must match pattern {self.pattern}"

        return True, None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "type": self.type.value,
            "description": self.description,
            "required": self.required,
            "default": self.default,
            "enum": self.enum,
        }


@dataclass
class ToolStats:
    """Statistics for a tool."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_latency_ms: float = 0.0
    last_call: Optional[datetime] = None
    last_error: Optional[str] = None

    @property
    def avg_latency_ms(self) -> float:
        """Get average latency."""
        if self.total_calls == 0:
            return 0.0
        return self.total_latency_ms / self.total_calls

    @property
    def success_rate(self) -> float:
        """Get success rate."""
        if self.total_calls == 0:
            return 1.0
        return self.successful_calls / self.total_calls


@dataclass
class Tool:
    """Definition of a tool."""
    name: str
    description: str
    handler: Callable
    parameters: list[ToolParameter] = field(default_factory=list)
    category: ToolCategory = ToolCategory.CUSTOM
    version: str = "1.0.0"
    rate_limit: float = 10.0  # Max calls per second
    timeout: float = 60.0
    requires_approval: bool = False
    enabled: bool = True
    tags: list[str] = field(default_factory=list)
    stats: ToolStats = field(default_factory=ToolStats)

    def validate_params(
        self, params: dict[str, Any]
    ) -> tuple[bool, list[str]]:
        """
        Validate parameters for this tool.

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []

        # Check each defined parameter
        for param in self.parameters:
            value = params.get(param.name, param.default)
            is_valid, error = param.validate(value)
            if not is_valid:
                errors.append(error)

        # Check for unknown parameters
        known_params = {p.name for p in self.parameters}
        unknown = set(params.keys()) - known_params
        if unknown:
            errors.append(f"Unknown parameters: {unknown}")

        return len(errors) == 0, errors

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for LLM consumption."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    p.name: {
                        "type": p.type.value,
                        "description": p.description,
                        **({"enum": p.enum} if p.enum else {}),
                        **({"default": p.default} if p.default is not None else {}),
                    }
                    for p in self.parameters
                },
                "required": [p.name for p in self.parameters if p.required],
            },
        }

    def to_openai_format(self) -> dict[str, Any]:
        """Convert to OpenAI function calling format."""
        return {
            "type": "function",
            "function": self.to_dict(),
        }


class RateLimiter:
    """Token bucket rate limiter for tools."""

    def __init__(self, rate: float, burst: int = 5):
        self.rate = rate
        self.burst = burst
        self.tokens = float(burst)
        self.last_update = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> bool:
        """Try to acquire a token."""
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_update
            self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
            self.last_update = now

            if self.tokens >= 1:
                self.tokens -= 1
                return True
            return False

    async def wait(self) -> None:
        """Wait until a token is available."""
        while not await self.acquire():
            await asyncio.sleep(1.0 / self.rate)


class ToolRegistry:
    """
    Registry for managing tools.

    Features:
    - Tool registration and discovery
    - Parameter validation
    - Rate limiting
    - Usage tracking
    """

    def __init__(self):
        self._tools: dict[str, Tool] = {}
        self._rate_limiters: dict[str, RateLimiter] = {}
        self._disabled_tools: set[str] = set()

    def register(self, tool: Tool) -> None:
        """
        Register a tool.

        Args:
            tool: Tool to register
        """
        if tool.name in self._tools:
            logger.warning(f"Overwriting existing tool: {tool.name}")

        self._tools[tool.name] = tool
        self._rate_limiters[tool.name] = RateLimiter(tool.rate_limit)

        logger.info(
            "Tool registered",
            name=tool.name,
            category=tool.category.value,
        )

    def register_function(
        self,
        name: str,
        handler: Callable,
        description: str,
        parameters: Optional[list[ToolParameter]] = None,
        **kwargs,
    ) -> Tool:
        """
        Register a function as a tool.

        Args:
            name: Tool name
            handler: Function to execute
            description: Tool description
            parameters: Tool parameters
            **kwargs: Additional tool options

        Returns:
            Created Tool
        """
        tool = Tool(
            name=name,
            description=description,
            handler=handler,
            parameters=parameters or [],
            **kwargs,
        )
        self.register(tool)
        return tool

    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(
        self,
        category: Optional[ToolCategory] = None,
        enabled_only: bool = True,
    ) -> list[Tool]:
        """
        List registered tools.

        Args:
            category: Filter by category
            enabled_only: Only return enabled tools

        Returns:
            List of tools
        """
        tools = list(self._tools.values())

        if category:
            tools = [t for t in tools if t.category == category]

        if enabled_only:
            tools = [t for t in tools if t.enabled and t.name not in self._disabled_tools]

        return tools

    def disable(self, name: str) -> bool:
        """Disable a tool."""
        if name in self._tools:
            self._disabled_tools.add(name)
            return True
        return False

    def enable(self, name: str) -> bool:
        """Enable a tool."""
        if name in self._tools:
            self._disabled_tools.discard(name)
            return True
        return False

    def is_enabled(self, name: str) -> bool:
        """Check if a tool is enabled."""
        return name in self._tools and name not in self._disabled_tools

    async def check_rate_limit(self, name: str) -> bool:
        """Check if a tool call is allowed by rate limit."""
        limiter = self._rate_limiters.get(name)
        if limiter is None:
            return True
        return await limiter.acquire()

    async def wait_for_rate_limit(self, name: str) -> None:
        """Wait for rate limit to allow a call."""
        limiter = self._rate_limiters.get(name)
        if limiter:
            await limiter.wait()

    def get_stats(self, name: str) -> Optional[ToolStats]:
        """Get statistics for a tool."""
        tool = self._tools.get(name)
        return tool.stats if tool else None

    def record_call(
        self,
        name: str,
        success: bool,
        latency_ms: float,
        error: Optional[str] = None,
    ) -> None:
        """Record a tool call for statistics."""
        tool = self._tools.get(name)
        if tool is None:
            return

        tool.stats.total_calls += 1
        tool.stats.total_latency_ms += latency_ms
        tool.stats.last_call = datetime.now()

        if success:
            tool.stats.successful_calls += 1
        else:
            tool.stats.failed_calls += 1
            tool.stats.last_error = error

    def to_openai_tools(self) -> list[dict[str, Any]]:
        """Get all tools in OpenAI function format."""
        return [
            tool.to_openai_format()
            for tool in self.list_tools(enabled_only=True)
        ]
