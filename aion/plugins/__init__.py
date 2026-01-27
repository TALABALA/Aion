"""
AION Plugin System

A comprehensive extensibility framework for AION, enabling third-party
plugins to extend functionality without modifying the core codebase.

Features:
- Plugin discovery and loading from multiple sources
- Full lifecycle management (load/unload/activate/suspend)
- Dependency resolution with topological ordering
- Hook system for extensibility points
- Sandboxed execution with permission control
- Hot reload support
- Event emission for monitoring
- REST API for management

Basic Usage:
    from aion.plugins import PluginManager, PluginType

    # Create manager
    manager = PluginManager(kernel=kernel)

    # Initialize (discovers plugins)
    await manager.initialize()

    # Load and activate a plugin
    await manager.load("my-plugin", config={"key": "value"})
    await manager.activate("my-plugin")

    # Get active tools from plugins
    tools = manager.get_plugin_tools()

    # Shutdown
    await manager.shutdown()

Plugin Development:
    from aion.plugins import BasePlugin, PluginManifest, PluginType

    class MyPlugin(BasePlugin):
        @classmethod
        def get_manifest(cls):
            return PluginManifest(
                id="my-plugin",
                name="My Plugin",
                version=SemanticVersion(1, 0, 0),
                plugin_type=PluginType.TOOL,
            )

        async def initialize(self, kernel, config):
            self._kernel = kernel
            self._config = config

        async def shutdown(self):
            pass
"""

from aion.plugins.types import (
    # Enums
    PluginType,
    PluginState,
    PluginPriority,
    PermissionLevel,
    PluginEvent,
    # Version handling
    SemanticVersion,
    VersionConstraint,
    # Plugin definitions
    PluginDependency,
    PluginPermissions,
    PluginManifest,
    PluginInfo,
    PluginMetrics,
    ResourceLimit,
    # Hooks
    HookDefinition,
    HookRegistration,
    # Events
    PluginEventData,
    # Validation
    ValidationResult,
)

from aion.plugins.interfaces.base import BasePlugin, PluginContext, PluginMixin
from aion.plugins.interfaces.tool import (
    ToolPlugin,
    Tool,
    ToolParameter,
    ToolParameterType,
    ToolResult,
)
from aion.plugins.interfaces.agent import (
    AgentPlugin,
    AgentConfig,
    AgentCapability,
    AgentInstance,
    AgentTypeDefinition,
)
from aion.plugins.interfaces.storage import (
    StoragePlugin,
    StorageItem,
    StorageCapabilities,
    StorageStats,
)
from aion.plugins.interfaces.workflow import (
    WorkflowTriggerPlugin,
    WorkflowActionPlugin,
    TriggerConfig,
    TriggerEvent,
    TriggerDefinition,
    TriggerType,
    ActionConfig,
    ActionResult,
    ActionDefinition,
    ActionType,
)

from aion.plugins.registry import PluginRegistry
from aion.plugins.loader import PluginLoader, PluginLoadError
from aion.plugins.lifecycle import LifecycleManager, LifecycleError
from aion.plugins.manager import PluginManager
from aion.plugins.validation import PluginValidator
from aion.plugins.config import PluginSystemConfig, PluginRuntimeConfig, ConfigManager

from aion.plugins.hooks.system import HookSystem, BUILTIN_HOOKS, hook_handler
from aion.plugins.hooks.events import PluginEventEmitter, get_event_emitter

from aion.plugins.dependencies.resolver import DependencyResolver, ResolutionResult
from aion.plugins.dependencies.graph import DependencyGraph

from aion.plugins.sandbox.runtime import SandboxRuntime, SandboxedPlugin, ExecutionResult
from aion.plugins.sandbox.permissions import (
    PermissionChecker,
    PermissionViolation,
    PermissionBuilder,
)

from aion.plugins.discovery.local import LocalDiscovery
from aion.plugins.discovery.marketplace import MarketplaceClient, MarketplacePlugin

from aion.plugins.api import create_plugin_routes, setup_plugin_routes


__all__ = [
    # === Core Manager ===
    "PluginManager",

    # === Types & Enums ===
    "PluginType",
    "PluginState",
    "PluginPriority",
    "PermissionLevel",
    "PluginEvent",

    # === Version ===
    "SemanticVersion",
    "VersionConstraint",

    # === Plugin Definitions ===
    "PluginDependency",
    "PluginPermissions",
    "PluginManifest",
    "PluginInfo",
    "PluginMetrics",
    "ResourceLimit",

    # === Base Interfaces ===
    "BasePlugin",
    "PluginContext",
    "PluginMixin",

    # === Plugin Types ===
    "ToolPlugin",
    "Tool",
    "ToolParameter",
    "ToolParameterType",
    "ToolResult",

    "AgentPlugin",
    "AgentConfig",
    "AgentCapability",
    "AgentInstance",
    "AgentTypeDefinition",

    "StoragePlugin",
    "StorageItem",
    "StorageCapabilities",
    "StorageStats",

    "WorkflowTriggerPlugin",
    "WorkflowActionPlugin",
    "TriggerConfig",
    "TriggerEvent",
    "TriggerDefinition",
    "TriggerType",
    "ActionConfig",
    "ActionResult",
    "ActionDefinition",
    "ActionType",

    # === Registry & Loading ===
    "PluginRegistry",
    "PluginLoader",
    "PluginLoadError",

    # === Lifecycle ===
    "LifecycleManager",
    "LifecycleError",

    # === Validation ===
    "PluginValidator",
    "ValidationResult",

    # === Configuration ===
    "PluginSystemConfig",
    "PluginRuntimeConfig",
    "ConfigManager",

    # === Hooks ===
    "HookSystem",
    "HookDefinition",
    "HookRegistration",
    "BUILTIN_HOOKS",
    "hook_handler",

    # === Events ===
    "PluginEventEmitter",
    "PluginEventData",
    "get_event_emitter",

    # === Dependencies ===
    "DependencyResolver",
    "DependencyGraph",
    "ResolutionResult",

    # === Sandbox ===
    "SandboxRuntime",
    "SandboxedPlugin",
    "ExecutionResult",
    "PermissionChecker",
    "PermissionViolation",
    "PermissionBuilder",

    # === Discovery ===
    "LocalDiscovery",
    "MarketplaceClient",
    "MarketplacePlugin",

    # === API ===
    "create_plugin_routes",
    "setup_plugin_routes",
]

# Version
__version__ = "1.0.0"
