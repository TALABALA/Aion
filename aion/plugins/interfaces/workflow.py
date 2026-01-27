"""
AION Workflow Plugin Interfaces

Interfaces for workflow triggers and actions.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union
from datetime import datetime
from enum import Enum

from aion.plugins.interfaces.base import BasePlugin
from aion.plugins.types import PluginManifest, PluginType, SemanticVersion


class TriggerType(str, Enum):
    """Types of workflow triggers."""

    EVENT = "event"           # Triggered by events
    SCHEDULE = "schedule"     # Time-based (cron)
    WEBHOOK = "webhook"       # HTTP webhook
    FILE_WATCH = "file_watch"  # File system changes
    CONDITION = "condition"   # Conditional trigger
    MANUAL = "manual"         # Manual activation
    CHAIN = "chain"           # Triggered by other workflows


class ActionType(str, Enum):
    """Types of workflow actions."""

    EXECUTE = "execute"       # Execute code/tool
    HTTP = "http"             # HTTP request
    EMAIL = "email"           # Send email
    NOTIFY = "notify"         # Send notification
    TRANSFORM = "transform"   # Data transformation
    BRANCH = "branch"         # Conditional branching
    LOOP = "loop"             # Iteration
    WAIT = "wait"             # Delay/wait
    PARALLEL = "parallel"     # Parallel execution
    INVOKE_AGENT = "invoke_agent"  # Call an agent


@dataclass
class TriggerConfig:
    """Configuration for a trigger."""

    trigger_type: TriggerType
    name: str
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    filters: Dict[str, Any] = field(default_factory=dict)
    debounce_seconds: float = 0.0
    max_frequency_per_minute: int = 0
    enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.trigger_type.value,
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "filters": self.filters,
            "debounce_seconds": self.debounce_seconds,
            "max_frequency_per_minute": self.max_frequency_per_minute,
            "enabled": self.enabled,
        }


@dataclass
class TriggerEvent:
    """An event from a trigger."""

    trigger_id: str
    trigger_type: TriggerType
    timestamp: datetime = field(default_factory=datetime.now)
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trigger_id": self.trigger_id,
            "trigger_type": self.trigger_type.value,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
            "metadata": self.metadata,
            "source": self.source,
        }


@dataclass
class ActionConfig:
    """Configuration for an action."""

    action_type: ActionType
    name: str
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: float = 60.0
    retry_count: int = 0
    retry_delay_seconds: float = 1.0
    on_error: str = "fail"  # fail, continue, retry

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.action_type.value,
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "timeout_seconds": self.timeout_seconds,
            "retry_count": self.retry_count,
            "on_error": self.on_error,
        }


@dataclass
class ActionResult:
    """Result of executing an action."""

    success: bool
    output: Any = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "execution_time_ms": self.execution_time_ms,
            "metadata": self.metadata,
        }


@dataclass
class TriggerDefinition:
    """Definition of a trigger type provided by a plugin."""

    trigger_id: str
    name: str
    description: str
    trigger_type: TriggerType
    config_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    examples: List[Dict[str, Any]] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trigger_id": self.trigger_id,
            "name": self.name,
            "description": self.description,
            "trigger_type": self.trigger_type.value,
            "config_schema": self.config_schema,
            "output_schema": self.output_schema,
            "tags": self.tags,
        }


@dataclass
class ActionDefinition:
    """Definition of an action type provided by a plugin."""

    action_id: str
    name: str
    description: str
    action_type: ActionType
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    examples: List[Dict[str, Any]] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_id": self.action_id,
            "name": self.name,
            "description": self.description,
            "action_type": self.action_type.value,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "tags": self.tags,
        }


class WorkflowTriggerPlugin(BasePlugin):
    """
    Interface for workflow trigger plugins.

    Trigger plugins provide event sources that can
    start workflow executions.

    Implement:
    - get_trigger_types(): Return available trigger types
    - create_trigger(): Create a trigger instance
    - start_trigger()/stop_trigger(): Control trigger
    """

    def __init__(self):
        super().__init__()
        self._active_triggers: Dict[str, Any] = {}
        self._callbacks: Dict[str, Callable] = {}

    @abstractmethod
    def get_trigger_types(self) -> List[TriggerDefinition]:
        """
        Return list of trigger types provided.

        Returns:
            List of TriggerDefinition
        """
        pass

    @abstractmethod
    async def create_trigger(
        self,
        trigger_id: str,
        config: TriggerConfig,
        callback: Callable[[TriggerEvent], None],
    ) -> bool:
        """
        Create and register a trigger.

        Args:
            trigger_id: Unique identifier for this trigger instance
            config: Trigger configuration
            callback: Function to call when trigger fires

        Returns:
            True if created successfully
        """
        pass

    @abstractmethod
    async def start_trigger(self, trigger_id: str) -> bool:
        """
        Start a trigger.

        Args:
            trigger_id: Trigger identifier

        Returns:
            True if started successfully
        """
        pass

    @abstractmethod
    async def stop_trigger(self, trigger_id: str) -> bool:
        """
        Stop a trigger.

        Args:
            trigger_id: Trigger identifier

        Returns:
            True if stopped successfully
        """
        pass

    async def destroy_trigger(self, trigger_id: str) -> bool:
        """
        Destroy a trigger.

        Args:
            trigger_id: Trigger identifier

        Returns:
            True if destroyed successfully
        """
        await self.stop_trigger(trigger_id)
        self._active_triggers.pop(trigger_id, None)
        self._callbacks.pop(trigger_id, None)
        return True

    def get_active_triggers(self) -> List[str]:
        """Get list of active trigger IDs."""
        return list(self._active_triggers.keys())

    def is_trigger_active(self, trigger_id: str) -> bool:
        """Check if trigger is active."""
        return trigger_id in self._active_triggers


class WorkflowActionPlugin(BasePlugin):
    """
    Interface for workflow action plugins.

    Action plugins provide executable steps for workflows.

    Implement:
    - get_action_types(): Return available action types
    - execute_action(): Execute an action
    """

    @abstractmethod
    def get_action_types(self) -> List[ActionDefinition]:
        """
        Return list of action types provided.

        Returns:
            List of ActionDefinition
        """
        pass

    @abstractmethod
    async def execute_action(
        self,
        action_type: str,
        config: ActionConfig,
        input_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> ActionResult:
        """
        Execute an action.

        Args:
            action_type: Type of action to execute
            config: Action configuration
            input_data: Input data for the action
            context: Execution context

        Returns:
            ActionResult with output
        """
        pass

    def get_action_type(self, action_id: str) -> Optional[ActionDefinition]:
        """Get a specific action type definition."""
        for action in self.get_action_types():
            if action.action_id == action_id:
                return action
        return None

    async def validate_input(
        self,
        action_type: str,
        input_data: Dict[str, Any],
    ) -> tuple[bool, List[str]]:
        """
        Validate action input data.

        Args:
            action_type: Type of action
            input_data: Input to validate

        Returns:
            (is_valid, error_messages)
        """
        action_def = self.get_action_type(action_type)
        if not action_def:
            return False, [f"Unknown action type: {action_type}"]

        # Basic validation - could use JSON Schema
        return True, []


# === Example Implementations ===


class ExampleTriggerPlugin(WorkflowTriggerPlugin):
    """Example trigger plugin implementation."""

    @classmethod
    def get_manifest(cls) -> PluginManifest:
        return PluginManifest(
            id="example-trigger-plugin",
            name="Example Trigger Plugin",
            version=SemanticVersion(1, 0, 0),
            description="Example workflow trigger plugin",
            plugin_type=PluginType.WORKFLOW_TRIGGER,
            entry_point="example_trigger:ExampleTriggerPlugin",
            tags=["workflow", "trigger", "example"],
        )

    def get_trigger_types(self) -> List[TriggerDefinition]:
        return [
            TriggerDefinition(
                trigger_id="interval",
                name="Interval Trigger",
                description="Fires at regular intervals",
                trigger_type=TriggerType.SCHEDULE,
                config_schema={
                    "type": "object",
                    "properties": {
                        "interval_seconds": {"type": "integer", "minimum": 1},
                    },
                    "required": ["interval_seconds"],
                },
                tags=["schedule", "interval"],
            ),
        ]

    async def create_trigger(
        self,
        trigger_id: str,
        config: TriggerConfig,
        callback: Callable[[TriggerEvent], None],
    ) -> bool:
        self._active_triggers[trigger_id] = {
            "config": config,
            "running": False,
        }
        self._callbacks[trigger_id] = callback
        return True

    async def start_trigger(self, trigger_id: str) -> bool:
        if trigger_id in self._active_triggers:
            self._active_triggers[trigger_id]["running"] = True
            # In a real implementation, start an async task here
            return True
        return False

    async def stop_trigger(self, trigger_id: str) -> bool:
        if trigger_id in self._active_triggers:
            self._active_triggers[trigger_id]["running"] = False
            return True
        return False

    async def initialize(self, kernel, config: Dict[str, Any]) -> None:
        self._kernel = kernel
        self._config = config
        self._initialized = True

    async def shutdown(self) -> None:
        # Stop all triggers
        for trigger_id in list(self._active_triggers.keys()):
            await self.destroy_trigger(trigger_id)
        self._initialized = False


class ExampleActionPlugin(WorkflowActionPlugin):
    """Example action plugin implementation."""

    @classmethod
    def get_manifest(cls) -> PluginManifest:
        return PluginManifest(
            id="example-action-plugin",
            name="Example Action Plugin",
            version=SemanticVersion(1, 0, 0),
            description="Example workflow action plugin",
            plugin_type=PluginType.WORKFLOW_ACTION,
            entry_point="example_action:ExampleActionPlugin",
            tags=["workflow", "action", "example"],
        )

    def get_action_types(self) -> List[ActionDefinition]:
        return [
            ActionDefinition(
                action_id="log",
                name="Log Action",
                description="Log a message",
                action_type=ActionType.EXECUTE,
                input_schema={
                    "type": "object",
                    "properties": {
                        "message": {"type": "string"},
                        "level": {"type": "string", "enum": ["info", "warning", "error"]},
                    },
                    "required": ["message"],
                },
                tags=["logging", "debug"],
            ),
            ActionDefinition(
                action_id="transform",
                name="Transform Data",
                description="Transform input data",
                action_type=ActionType.TRANSFORM,
                input_schema={
                    "type": "object",
                    "properties": {
                        "data": {"type": "object"},
                        "template": {"type": "string"},
                    },
                },
                tags=["transform", "data"],
            ),
        ]

    async def execute_action(
        self,
        action_type: str,
        config: ActionConfig,
        input_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> ActionResult:
        import time

        start_time = time.time()

        try:
            if action_type == "log":
                message = input_data.get("message", "")
                level = input_data.get("level", "info")
                # In real implementation, use proper logging
                print(f"[{level.upper()}] {message}")
                return ActionResult(
                    success=True,
                    output={"logged": True, "message": message},
                    execution_time_ms=(time.time() - start_time) * 1000,
                )

            elif action_type == "transform":
                data = input_data.get("data", {})
                template = input_data.get("template", "")
                # Simple template processing
                result = template
                for key, value in data.items():
                    result = result.replace(f"{{{key}}}", str(value))
                return ActionResult(
                    success=True,
                    output={"result": result},
                    execution_time_ms=(time.time() - start_time) * 1000,
                )

            return ActionResult(
                success=False,
                error=f"Unknown action type: {action_type}",
            )

        except Exception as e:
            return ActionResult(
                success=False,
                error=str(e),
                execution_time_ms=(time.time() - start_time) * 1000,
            )

    async def initialize(self, kernel, config: Dict[str, Any]) -> None:
        self._kernel = kernel
        self._config = config
        self._initialized = True

    async def shutdown(self) -> None:
        self._initialized = False
