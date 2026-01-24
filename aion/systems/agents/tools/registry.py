"""
Tool Registry

Centralized registry for managing tools across agents.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
import structlog

from aion.systems.agents.tools.mcp import (
    MCPTool,
    MCPToolResult,
    ToolCapability,
)

logger = structlog.get_logger()


@dataclass
class ToolUsageStats:
    """Statistics for tool usage."""

    tool_name: str
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_execution_time: float = 0.0
    last_used: Optional[datetime] = None
    errors: list[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Get success rate."""
        if self.total_calls == 0:
            return 0.0
        return self.successful_calls / self.total_calls

    @property
    def avg_execution_time(self) -> float:
        """Get average execution time."""
        if self.successful_calls == 0:
            return 0.0
        return self.total_execution_time / self.successful_calls

    def record_call(self, result: MCPToolResult) -> None:
        """Record a tool call."""
        self.total_calls += 1
        self.last_used = datetime.now()

        if result.success:
            self.successful_calls += 1
            self.total_execution_time += result.execution_time
        else:
            self.failed_calls += 1
            if result.error:
                self.errors.append(result.error)
                # Keep only last 10 errors
                self.errors = self.errors[-10:]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tool_name": self.tool_name,
            "total_calls": self.total_calls,
            "success_rate": self.success_rate,
            "avg_execution_time": self.avg_execution_time,
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "recent_errors": self.errors[-5:],
        }


@dataclass
class ToolPermission:
    """Permission settings for a tool."""

    tool_name: str
    allowed_agents: set[str] = field(default_factory=set)
    denied_agents: set[str] = field(default_factory=set)
    requires_approval: bool = False
    max_calls_per_minute: Optional[int] = None
    max_calls_per_agent: Optional[int] = None

    def is_allowed(self, agent_id: str) -> bool:
        """Check if agent is allowed to use tool."""
        if agent_id in self.denied_agents:
            return False
        if self.allowed_agents and agent_id not in self.allowed_agents:
            return False
        return True


class ToolRegistry:
    """
    Centralized tool registry.

    Features:
    - Tool registration and discovery
    - Usage tracking and statistics
    - Permission management
    - Tool search and filtering
    """

    def __init__(self):
        self._tools: dict[str, MCPTool] = {}
        self._stats: dict[str, ToolUsageStats] = {}
        self._permissions: dict[str, ToolPermission] = {}
        self._tags: dict[str, set[str]] = {}  # tag -> tool names
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize registry."""
        self._initialized = True
        logger.info("tool_registry_initialized")

    async def shutdown(self) -> None:
        """Shutdown registry."""
        self._initialized = False
        logger.info("tool_registry_shutdown")

    def register(
        self,
        tool: MCPTool,
        tags: Optional[list[str]] = None,
    ) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool
        self._stats[tool.name] = ToolUsageStats(tool_name=tool.name)
        self._permissions[tool.name] = ToolPermission(tool_name=tool.name)

        # Add tags
        for tag in tags or []:
            if tag not in self._tags:
                self._tags[tag] = set()
            self._tags[tag].add(tool.name)

        logger.info("tool_registered", tool=tool.name, tags=tags)

    def unregister(self, name: str) -> bool:
        """Unregister a tool."""
        if name not in self._tools:
            return False

        del self._tools[name]
        del self._stats[name]
        del self._permissions[name]

        # Remove from tags
        for tag_tools in self._tags.values():
            tag_tools.discard(name)

        return True

    def get(self, name: str) -> Optional[MCPTool]:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_all(self) -> list[MCPTool]:
        """List all registered tools."""
        return list(self._tools.values())

    def search(
        self,
        query: Optional[str] = None,
        capabilities: Optional[list[ToolCapability]] = None,
        tags: Optional[list[str]] = None,
    ) -> list[MCPTool]:
        """Search for tools."""
        results = list(self._tools.values())

        # Filter by query
        if query:
            query_lower = query.lower()
            results = [
                t for t in results
                if query_lower in t.name.lower()
                or query_lower in t.description.lower()
            ]

        # Filter by capabilities
        if capabilities:
            results = [
                t for t in results
                if any(cap in t.capabilities for cap in capabilities)
            ]

        # Filter by tags
        if tags:
            tagged_tools = set()
            for tag in tags:
                if tag in self._tags:
                    tagged_tools.update(self._tags[tag])
            results = [t for t in results if t.name in tagged_tools]

        return results

    def record_usage(self, result: MCPToolResult) -> None:
        """Record tool usage."""
        if result.tool_name in self._stats:
            self._stats[result.tool_name].record_call(result)

    def get_stats(self, name: str) -> Optional[ToolUsageStats]:
        """Get usage stats for a tool."""
        return self._stats.get(name)

    def get_all_stats(self) -> dict[str, dict[str, Any]]:
        """Get stats for all tools."""
        return {
            name: stats.to_dict()
            for name, stats in self._stats.items()
        }

    def set_permission(
        self,
        tool_name: str,
        permission: ToolPermission,
    ) -> bool:
        """Set permission for a tool."""
        if tool_name not in self._tools:
            return False
        self._permissions[tool_name] = permission
        return True

    def check_permission(self, tool_name: str, agent_id: str) -> bool:
        """Check if agent can use tool."""
        if tool_name not in self._permissions:
            return True  # Default allow
        return self._permissions[tool_name].is_allowed(agent_id)

    def add_tag(self, tool_name: str, tag: str) -> bool:
        """Add a tag to a tool."""
        if tool_name not in self._tools:
            return False

        if tag not in self._tags:
            self._tags[tag] = set()
        self._tags[tag].add(tool_name)
        return True

    def remove_tag(self, tool_name: str, tag: str) -> bool:
        """Remove a tag from a tool."""
        if tag in self._tags:
            self._tags[tag].discard(tool_name)
            return True
        return False

    def get_tools_by_tag(self, tag: str) -> list[MCPTool]:
        """Get tools with a specific tag."""
        if tag not in self._tags:
            return []
        return [
            self._tools[name]
            for name in self._tags[tag]
            if name in self._tools
        ]

    def get_popular_tools(self, limit: int = 10) -> list[tuple[MCPTool, ToolUsageStats]]:
        """Get most popular tools by usage."""
        sorted_stats = sorted(
            self._stats.values(),
            key=lambda s: s.total_calls,
            reverse=True,
        )

        return [
            (self._tools[s.tool_name], s)
            for s in sorted_stats[:limit]
            if s.tool_name in self._tools
        ]

    def get_reliable_tools(
        self,
        min_calls: int = 10,
        min_success_rate: float = 0.9,
    ) -> list[tuple[MCPTool, ToolUsageStats]]:
        """Get reliable tools based on success rate."""
        reliable = [
            s for s in self._stats.values()
            if s.total_calls >= min_calls and s.success_rate >= min_success_rate
        ]

        return [
            (self._tools[s.tool_name], s)
            for s in sorted(reliable, key=lambda s: s.success_rate, reverse=True)
            if s.tool_name in self._tools
        ]

    def get_registry_stats(self) -> dict[str, Any]:
        """Get overall registry statistics."""
        total_calls = sum(s.total_calls for s in self._stats.values())
        total_successes = sum(s.successful_calls for s in self._stats.values())

        return {
            "total_tools": len(self._tools),
            "total_tags": len(self._tags),
            "total_calls": total_calls,
            "overall_success_rate": total_successes / max(1, total_calls),
            "tools_by_capability": {
                cap.value: len([
                    t for t in self._tools.values()
                    if cap in t.capabilities
                ])
                for cap in ToolCapability
            },
        }
