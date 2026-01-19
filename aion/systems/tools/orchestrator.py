"""
AION Tool Orchestrator

Intelligent multi-tool coordination with:
- Automatic tool selection
- Dependency resolution
- Parallel execution optimization
- Tool composition and chaining
- Learning optimal tool combinations
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional

import httpx
from bs4 import BeautifulSoup
import structlog

from aion.systems.tools.registry import (
    ToolRegistry,
    Tool,
    ToolParameter,
    ToolCategory,
    ParameterType,
)
from aion.systems.tools.executor import ToolExecutor, ExecutionResult

logger = structlog.get_logger(__name__)


@dataclass
class ToolChain:
    """A chain of tools to execute in sequence."""
    id: str
    name: str
    steps: list[tuple[str, dict[str, Any]]]  # (tool_name, params_template)
    description: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    execution_count: int = 0
    success_count: int = 0


@dataclass
class OrchestrationResult:
    """Result of orchestrated tool execution."""
    success: bool
    results: list[ExecutionResult]
    total_latency_ms: float
    tools_executed: list[str]
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "results": [r.to_dict() for r in self.results],
            "total_latency_ms": self.total_latency_ms,
            "tools_executed": self.tools_executed,
            "errors": self.errors,
        }


class ToolOrchestrator:
    """
    AION Tool Orchestrator

    Coordinates multiple tools to accomplish complex tasks.

    Features:
    - Built-in essential tools (web, file, compute)
    - Automatic tool selection based on task
    - Parallel execution optimization
    - Tool chaining and composition
    - Learning from execution patterns
    """

    def __init__(
        self,
        max_parallel: int = 10,
        default_timeout: float = 60.0,
        enable_learning: bool = True,
    ):
        self.max_parallel = max_parallel
        self.default_timeout = default_timeout
        self.enable_learning = enable_learning

        # Core components
        self.registry = ToolRegistry()
        self.executor = ToolExecutor(self.registry)

        # Tool chains
        self._chains: dict[str, ToolChain] = {}

        # Learning data
        self._tool_cooccurrence: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self._task_tool_mapping: dict[str, list[str]] = defaultdict(list)

        # Statistics
        self._stats = {
            "total_orchestrations": 0,
            "total_tools_executed": 0,
            "successful_orchestrations": 0,
        }

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the orchestrator and register built-in tools."""
        if self._initialized:
            return

        logger.info("Initializing Tool Orchestrator")

        # Register built-in tools
        self._register_builtin_tools()

        self._initialized = True
        logger.info(
            "Tool Orchestrator initialized",
            tools_registered=len(self.registry.list_tools()),
        )

    async def shutdown(self) -> None:
        """Shutdown the orchestrator."""
        logger.info("Shutting down Tool Orchestrator")
        self._initialized = False

    def _register_builtin_tools(self) -> None:
        """Register built-in essential tools."""

        # Web Fetch Tool
        self.registry.register(Tool(
            name="web_fetch",
            description="Fetch content from a URL and return the text content",
            handler=self._tool_web_fetch,
            parameters=[
                ToolParameter(
                    name="url",
                    type=ParameterType.STRING,
                    description="The URL to fetch",
                    required=True,
                ),
                ToolParameter(
                    name="extract_text",
                    type=ParameterType.BOOLEAN,
                    description="Extract text content from HTML",
                    default=True,
                ),
            ],
            category=ToolCategory.WEB,
            rate_limit=5.0,
        ))

        # Web Search Tool
        self.registry.register(Tool(
            name="web_search",
            description="Search the web using a search engine",
            handler=self._tool_web_search,
            parameters=[
                ToolParameter(
                    name="query",
                    type=ParameterType.STRING,
                    description="Search query",
                    required=True,
                ),
                ToolParameter(
                    name="num_results",
                    type=ParameterType.INTEGER,
                    description="Number of results to return",
                    default=10,
                    min_value=1,
                    max_value=50,
                ),
            ],
            category=ToolCategory.SEARCH,
            rate_limit=2.0,
        ))

        # Calculator Tool
        self.registry.register(Tool(
            name="calculator",
            description="Perform mathematical calculations",
            handler=self._tool_calculator,
            parameters=[
                ToolParameter(
                    name="expression",
                    type=ParameterType.STRING,
                    description="Mathematical expression to evaluate",
                    required=True,
                ),
            ],
            category=ToolCategory.COMPUTE,
            rate_limit=100.0,
        ))

        # JSON Parser Tool
        self.registry.register(Tool(
            name="json_parse",
            description="Parse and query JSON data",
            handler=self._tool_json_parse,
            parameters=[
                ToolParameter(
                    name="json_str",
                    type=ParameterType.STRING,
                    description="JSON string to parse",
                    required=True,
                ),
                ToolParameter(
                    name="path",
                    type=ParameterType.STRING,
                    description="JSONPath expression to extract data",
                    default=None,
                ),
            ],
            category=ToolCategory.DATA,
        ))

        # Text Transform Tool
        self.registry.register(Tool(
            name="text_transform",
            description="Transform text with various operations",
            handler=self._tool_text_transform,
            parameters=[
                ToolParameter(
                    name="text",
                    type=ParameterType.STRING,
                    description="Text to transform",
                    required=True,
                ),
                ToolParameter(
                    name="operation",
                    type=ParameterType.STRING,
                    description="Transformation operation",
                    enum=["uppercase", "lowercase", "title", "reverse", "count_words"],
                    required=True,
                ),
            ],
            category=ToolCategory.DATA,
        ))

        # DateTime Tool
        self.registry.register(Tool(
            name="datetime",
            description="Get current date/time or parse/format dates",
            handler=self._tool_datetime,
            parameters=[
                ToolParameter(
                    name="operation",
                    type=ParameterType.STRING,
                    description="DateTime operation",
                    enum=["now", "parse", "format", "diff"],
                    default="now",
                ),
                ToolParameter(
                    name="value",
                    type=ParameterType.STRING,
                    description="Date/time value for parse/format",
                    default=None,
                ),
                ToolParameter(
                    name="format",
                    type=ParameterType.STRING,
                    description="Date format string",
                    default="%Y-%m-%d %H:%M:%S",
                ),
            ],
            category=ToolCategory.COMPUTE,
        ))

        # Random Generator Tool
        self.registry.register(Tool(
            name="random",
            description="Generate random values",
            handler=self._tool_random,
            parameters=[
                ToolParameter(
                    name="type",
                    type=ParameterType.STRING,
                    description="Type of random value",
                    enum=["int", "float", "choice", "uuid"],
                    default="int",
                ),
                ToolParameter(
                    name="min",
                    type=ParameterType.INTEGER,
                    description="Minimum value for int/float",
                    default=0,
                ),
                ToolParameter(
                    name="max",
                    type=ParameterType.INTEGER,
                    description="Maximum value for int/float",
                    default=100,
                ),
                ToolParameter(
                    name="choices",
                    type=ParameterType.ARRAY,
                    description="Array of choices for choice type",
                    default=None,
                ),
            ],
            category=ToolCategory.COMPUTE,
        ))

    # ==================== Built-in Tool Handlers ====================

    async def _tool_web_fetch(self, params: dict[str, Any]) -> dict[str, Any]:
        """Fetch content from a URL."""
        url = params["url"]
        extract_text = params.get("extract_text", True)

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, follow_redirects=True)
            response.raise_for_status()

            content = response.text

            if extract_text and "text/html" in response.headers.get("content-type", ""):
                soup = BeautifulSoup(content, "html.parser")
                # Remove script and style elements
                for element in soup(["script", "style", "nav", "footer"]):
                    element.decompose()
                content = soup.get_text(separator="\n", strip=True)

            return {
                "url": str(response.url),
                "status_code": response.status_code,
                "content": content[:10000],  # Limit content size
                "content_type": response.headers.get("content-type"),
            }

    async def _tool_web_search(self, params: dict[str, Any]) -> dict[str, Any]:
        """Perform a web search (mock implementation)."""
        query = params["query"]
        num_results = params.get("num_results", 10)

        # Note: In production, integrate with a real search API
        # This is a mock implementation
        return {
            "query": query,
            "results": [
                {
                    "title": f"Result {i+1} for: {query}",
                    "url": f"https://example.com/result/{i+1}",
                    "snippet": f"This is a mock search result for '{query}'",
                }
                for i in range(min(num_results, 10))
            ],
            "total_results": num_results,
            "note": "Mock search results - integrate with real search API in production",
        }

    async def _tool_calculator(self, params: dict[str, Any]) -> dict[str, Any]:
        """Evaluate a mathematical expression."""
        import math
        import operator

        expression = params["expression"]

        # Safe evaluation with limited operations
        allowed_operations = {
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum,
            "pow": pow,
            "sqrt": math.sqrt,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "log": math.log,
            "log10": math.log10,
            "exp": math.exp,
            "pi": math.pi,
            "e": math.e,
        }

        try:
            # Use eval with restricted globals for safety
            result = eval(expression, {"__builtins__": {}}, allowed_operations)
            return {
                "expression": expression,
                "result": result,
            }
        except Exception as e:
            return {
                "expression": expression,
                "error": str(e),
            }

    async def _tool_json_parse(self, params: dict[str, Any]) -> dict[str, Any]:
        """Parse and optionally query JSON data."""
        json_str = params["json_str"]
        path = params.get("path")

        data = json.loads(json_str)

        if path:
            # Simple JSONPath-like extraction
            parts = path.strip("$.").split(".")
            result = data
            for part in parts:
                if isinstance(result, dict):
                    result = result.get(part)
                elif isinstance(result, list):
                    try:
                        idx = int(part)
                        result = result[idx]
                    except (ValueError, IndexError):
                        result = None
                else:
                    result = None
            return {"path": path, "result": result}

        return {"data": data}

    async def _tool_text_transform(self, params: dict[str, Any]) -> dict[str, Any]:
        """Transform text with various operations."""
        text = params["text"]
        operation = params["operation"]

        if operation == "uppercase":
            result = text.upper()
        elif operation == "lowercase":
            result = text.lower()
        elif operation == "title":
            result = text.title()
        elif operation == "reverse":
            result = text[::-1]
        elif operation == "count_words":
            result = len(text.split())
        else:
            result = text

        return {
            "original": text[:100],
            "operation": operation,
            "result": result if not isinstance(result, str) else result[:1000],
        }

    async def _tool_datetime(self, params: dict[str, Any]) -> dict[str, Any]:
        """DateTime operations."""
        from datetime import datetime, timedelta

        operation = params.get("operation", "now")
        value = params.get("value")
        fmt = params.get("format", "%Y-%m-%d %H:%M:%S")

        if operation == "now":
            now = datetime.now()
            return {
                "datetime": now.strftime(fmt),
                "timestamp": now.timestamp(),
                "iso": now.isoformat(),
            }
        elif operation == "parse" and value:
            dt = datetime.strptime(value, fmt)
            return {
                "datetime": dt.strftime(fmt),
                "timestamp": dt.timestamp(),
                "iso": dt.isoformat(),
            }
        elif operation == "format" and value:
            # Parse ISO format and reformat
            dt = datetime.fromisoformat(value)
            return {"formatted": dt.strftime(fmt)}
        else:
            return {"error": "Invalid operation or missing value"}

    async def _tool_random(self, params: dict[str, Any]) -> dict[str, Any]:
        """Generate random values."""
        import random
        import uuid as uuid_module

        rand_type = params.get("type", "int")
        min_val = params.get("min", 0)
        max_val = params.get("max", 100)
        choices = params.get("choices")

        if rand_type == "int":
            result = random.randint(min_val, max_val)
        elif rand_type == "float":
            result = random.uniform(min_val, max_val)
        elif rand_type == "choice" and choices:
            result = random.choice(choices)
        elif rand_type == "uuid":
            result = str(uuid_module.uuid4())
        else:
            result = random.random()

        return {"type": rand_type, "result": result}

    # ==================== Orchestration Methods ====================

    async def execute(
        self,
        tool_name: str,
        params: dict[str, Any],
        timeout: Optional[float] = None,
    ) -> ExecutionResult:
        """
        Execute a single tool.

        Args:
            tool_name: Name of the tool
            params: Tool parameters
            timeout: Execution timeout

        Returns:
            ExecutionResult
        """
        if not self._initialized:
            await self.initialize()

        result = await self.executor.execute(
            tool_name,
            params,
            timeout=timeout or self.default_timeout,
        )

        self._stats["total_tools_executed"] += 1
        return result

    async def execute_parallel(
        self,
        calls: list[tuple[str, dict[str, Any]]],
    ) -> list[ExecutionResult]:
        """
        Execute multiple tools in parallel.

        Args:
            calls: List of (tool_name, params) tuples

        Returns:
            List of ExecutionResult
        """
        if not self._initialized:
            await self.initialize()

        results = await self.executor.execute_many(calls, self.max_parallel)
        self._stats["total_tools_executed"] += len(calls)

        # Record co-occurrence for learning
        if self.enable_learning and len(calls) > 1:
            tool_names = [c[0] for c in calls]
            for i, t1 in enumerate(tool_names):
                for t2 in tool_names[i+1:]:
                    self._tool_cooccurrence[t1][t2] += 1
                    self._tool_cooccurrence[t2][t1] += 1

        return results

    async def execute_chain(
        self,
        chain_id: str,
        initial_context: Optional[dict[str, Any]] = None,
    ) -> OrchestrationResult:
        """
        Execute a predefined tool chain.

        Args:
            chain_id: ID of the chain to execute
            initial_context: Initial context for parameter substitution

        Returns:
            OrchestrationResult
        """
        if chain_id not in self._chains:
            return OrchestrationResult(
                success=False,
                results=[],
                total_latency_ms=0,
                tools_executed=[],
                errors=[f"Chain not found: {chain_id}"],
            )

        chain = self._chains[chain_id]
        context = initial_context or {}
        results = []
        errors = []
        start_time = time.monotonic()

        for tool_name, params_template in chain.steps:
            # Substitute context values in parameters
            params = self._substitute_params(params_template, context)

            result = await self.execute(tool_name, params)
            results.append(result)

            if not result.success:
                errors.append(f"{tool_name}: {result.error}")
                break

            # Add result to context for next step
            context[f"step_{len(results)}"] = result.result
            context["last_result"] = result.result

        total_latency = (time.monotonic() - start_time) * 1000
        success = len(errors) == 0

        chain.execution_count += 1
        if success:
            chain.success_count += 1

        self._stats["total_orchestrations"] += 1
        if success:
            self._stats["successful_orchestrations"] += 1

        return OrchestrationResult(
            success=success,
            results=results,
            total_latency_ms=total_latency,
            tools_executed=[r.tool_name for r in results],
            errors=errors,
        )

    def _substitute_params(
        self,
        params_template: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Substitute context values in parameter template."""
        result = {}

        for key, value in params_template.items():
            if isinstance(value, str) and value.startswith("$"):
                # Context variable reference
                var_name = value[1:]
                result[key] = context.get(var_name, value)
            elif isinstance(value, dict):
                result[key] = self._substitute_params(value, context)
            else:
                result[key] = value

        return result

    def create_chain(
        self,
        name: str,
        steps: list[tuple[str, dict[str, Any]]],
        description: str = "",
    ) -> ToolChain:
        """
        Create a tool chain.

        Args:
            name: Chain name
            steps: List of (tool_name, params_template) tuples
            description: Chain description

        Returns:
            Created ToolChain
        """
        import uuid
        chain_id = str(uuid.uuid4())

        chain = ToolChain(
            id=chain_id,
            name=name,
            steps=steps,
            description=description,
        )

        self._chains[chain_id] = chain
        return chain

    def register_tool(self, tool: Tool) -> None:
        """Register a custom tool."""
        self.registry.register(tool)

    def list_tools(
        self,
        category: Optional[ToolCategory] = None,
    ) -> list[dict[str, Any]]:
        """List available tools."""
        tools = self.registry.list_tools(category=category)
        return [tool.to_dict() for tool in tools]

    def get_stats(self) -> dict[str, Any]:
        """Get orchestrator statistics."""
        return {
            **self._stats,
            "registered_tools": len(self.registry.list_tools()),
            "chains": len(self._chains),
        }

    def suggest_tools(
        self,
        task_description: str,
        max_suggestions: int = 5,
    ) -> list[str]:
        """
        Suggest tools for a task based on learning.

        Args:
            task_description: Description of the task
            max_suggestions: Maximum number of suggestions

        Returns:
            List of suggested tool names
        """
        # Simple keyword-based suggestion
        keywords = task_description.lower().split()
        tool_scores: dict[str, int] = defaultdict(int)

        for tool in self.registry.list_tools():
            tool_name_lower = tool.name.lower()
            tool_desc_lower = tool.description.lower()

            for keyword in keywords:
                if keyword in tool_name_lower:
                    tool_scores[tool.name] += 3
                if keyword in tool_desc_lower:
                    tool_scores[tool.name] += 1

        # Sort by score
        sorted_tools = sorted(
            tool_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        return [t[0] for t in sorted_tools[:max_suggestions] if t[1] > 0]
