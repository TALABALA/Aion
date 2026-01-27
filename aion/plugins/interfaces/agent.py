"""
AION Agent Plugin Interface

Interface for plugins that provide custom agent types.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type
from enum import Enum

from aion.plugins.interfaces.base import BasePlugin
from aion.plugins.types import PluginManifest, PluginType, SemanticVersion


class AgentCapability(str, Enum):
    """Capabilities an agent can have."""

    TOOL_USE = "tool_use"
    CODE_EXECUTION = "code_execution"
    WEB_BROWSING = "web_browsing"
    FILE_OPERATIONS = "file_operations"
    MEMORY_ACCESS = "memory_access"
    MULTI_MODAL = "multi_modal"
    PLANNING = "planning"
    DELEGATION = "delegation"
    LEARNING = "learning"
    REASONING = "reasoning"


@dataclass
class AgentConfig:
    """Configuration for an agent instance."""

    name: str
    agent_type: str
    description: str = ""
    model: str = "default"
    temperature: float = 0.7
    max_tokens: int = 4096
    system_prompt: Optional[str] = None
    capabilities: List[AgentCapability] = field(default_factory=list)
    allowed_tools: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: float = 300.0
    max_iterations: int = 100
    auto_save: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "agent_type": self.agent_type,
            "description": self.description,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "system_prompt": self.system_prompt,
            "capabilities": [c.value for c in self.capabilities],
            "allowed_tools": self.allowed_tools,
            "metadata": self.metadata,
            "timeout_seconds": self.timeout_seconds,
            "max_iterations": self.max_iterations,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentConfig":
        capabilities = [
            AgentCapability(c) for c in data.get("capabilities", [])
        ]
        return cls(
            name=data["name"],
            agent_type=data["agent_type"],
            description=data.get("description", ""),
            model=data.get("model", "default"),
            temperature=data.get("temperature", 0.7),
            max_tokens=data.get("max_tokens", 4096),
            system_prompt=data.get("system_prompt"),
            capabilities=capabilities,
            allowed_tools=data.get("allowed_tools", []),
            metadata=data.get("metadata", {}),
            timeout_seconds=data.get("timeout_seconds", 300.0),
            max_iterations=data.get("max_iterations", 100),
        )


@dataclass
class AgentTypeDefinition:
    """Definition of an agent type provided by a plugin."""

    type_id: str
    name: str
    description: str
    default_config: AgentConfig
    capabilities: List[AgentCapability] = field(default_factory=list)
    required_tools: List[str] = field(default_factory=list)
    optional_tools: List[str] = field(default_factory=list)
    system_prompt_template: str = ""
    examples: List[Dict[str, Any]] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type_id": self.type_id,
            "name": self.name,
            "description": self.description,
            "default_config": self.default_config.to_dict(),
            "capabilities": [c.value for c in self.capabilities],
            "required_tools": self.required_tools,
            "optional_tools": self.optional_tools,
            "tags": self.tags,
        }


class AgentInstance:
    """
    Base class for agent instances created by plugins.

    Plugins should subclass this to implement custom agent behavior.
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        self._running = False
        self._paused = False

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def agent_type(self) -> str:
        return self.config.agent_type

    @property
    def is_running(self) -> bool:
        return self._running

    async def start(self) -> None:
        """Start the agent."""
        self._running = True

    async def stop(self) -> None:
        """Stop the agent."""
        self._running = False

    async def pause(self) -> None:
        """Pause the agent."""
        self._paused = True

    async def resume(self) -> None:
        """Resume the agent."""
        self._paused = False

    async def process_message(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Process a message and return a response.

        Args:
            message: Input message
            context: Additional context

        Returns:
            Agent response
        """
        raise NotImplementedError

    async def execute_task(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute a task.

        Args:
            task: Task description
            context: Additional context

        Returns:
            Task result with output and metadata
        """
        raise NotImplementedError

    def get_state(self) -> Dict[str, Any]:
        """Get agent state for persistence."""
        return {
            "config": self.config.to_dict(),
            "running": self._running,
            "paused": self._paused,
        }

    def restore_state(self, state: Dict[str, Any]) -> None:
        """Restore agent from saved state."""
        self._running = state.get("running", False)
        self._paused = state.get("paused", False)


class AgentPlugin(BasePlugin):
    """
    Interface for agent plugins.

    Agent plugins provide custom agent types that can be
    spawned by the process manager.

    Implement:
    - get_agent_types(): Return list of agent types
    - create_agent(): Create an agent instance
    """

    @abstractmethod
    def get_agent_types(self) -> List[AgentTypeDefinition]:
        """
        Return list of agent types provided by this plugin.

        Returns:
            List of AgentTypeDefinition
        """
        pass

    @abstractmethod
    def create_agent(
        self,
        agent_type: str,
        config: AgentConfig,
    ) -> AgentInstance:
        """
        Create an agent instance.

        Args:
            agent_type: Type of agent to create
            config: Agent configuration

        Returns:
            AgentInstance

        Raises:
            ValueError: If agent type is unknown
        """
        pass

    def get_agent_type(self, type_id: str) -> Optional[AgentTypeDefinition]:
        """Get a specific agent type definition."""
        for agent_type in self.get_agent_types():
            if agent_type.type_id == type_id:
                return agent_type
        return None

    def get_default_config(self, agent_type: str) -> Optional[AgentConfig]:
        """
        Get default configuration for an agent type.

        Args:
            agent_type: Agent type identifier

        Returns:
            Default AgentConfig or None
        """
        type_def = self.get_agent_type(agent_type)
        return type_def.default_config if type_def else None

    def validate_config(
        self,
        agent_type: str,
        config: AgentConfig,
    ) -> tuple[bool, Optional[str]]:
        """
        Validate agent configuration.

        Args:
            agent_type: Agent type identifier
            config: Configuration to validate

        Returns:
            (is_valid, error_message)
        """
        type_def = self.get_agent_type(agent_type)
        if not type_def:
            return False, f"Unknown agent type: {agent_type}"

        # Check required tools are available
        missing_tools = [
            t for t in type_def.required_tools
            if t not in config.allowed_tools
        ]
        if missing_tools:
            return False, f"Missing required tools: {missing_tools}"

        return True, None

    def list_agent_type_ids(self) -> List[str]:
        """List all agent type IDs."""
        return [t.type_id for t in self.get_agent_types()]


# === Example Implementation ===


class SimpleAgent(AgentInstance):
    """Simple example agent implementation."""

    async def process_message(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        return f"[{self.name}] Received: {message}"

    async def execute_task(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return {
            "success": True,
            "output": f"Task '{task}' processed by {self.name}",
            "agent": self.name,
        }


class ExampleAgentPlugin(AgentPlugin):
    """Example agent plugin implementation."""

    @classmethod
    def get_manifest(cls) -> PluginManifest:
        return PluginManifest(
            id="example-agent-plugin",
            name="Example Agent Plugin",
            version=SemanticVersion(1, 0, 0),
            description="An example agent plugin demonstrating the interface",
            plugin_type=PluginType.AGENT,
            entry_point="example_agent_plugin:ExampleAgentPlugin",
            tags=["example", "demo"],
        )

    def get_agent_types(self) -> List[AgentTypeDefinition]:
        return [
            AgentTypeDefinition(
                type_id="simple-assistant",
                name="Simple Assistant",
                description="A simple conversational assistant",
                default_config=AgentConfig(
                    name="assistant",
                    agent_type="simple-assistant",
                    description="A helpful assistant",
                    capabilities=[
                        AgentCapability.TOOL_USE,
                        AgentCapability.MEMORY_ACCESS,
                    ],
                ),
                capabilities=[
                    AgentCapability.TOOL_USE,
                    AgentCapability.MEMORY_ACCESS,
                    AgentCapability.REASONING,
                ],
                tags=["assistant", "general-purpose"],
            ),
        ]

    def create_agent(
        self,
        agent_type: str,
        config: AgentConfig,
    ) -> AgentInstance:
        if agent_type == "simple-assistant":
            return SimpleAgent(config)
        raise ValueError(f"Unknown agent type: {agent_type}")

    async def initialize(self, kernel, config: Dict[str, Any]) -> None:
        self._kernel = kernel
        self._config = config
        self._initialized = True

    async def shutdown(self) -> None:
        self._initialized = False
