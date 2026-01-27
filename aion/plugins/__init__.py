"""
AION Plugin System - State-of-the-Art Extensibility Framework

A production-grade, SOTA plugin architecture for AION, enabling third-party
plugins to extend functionality with enterprise-level security and reliability.

SOTA Features:
- Process isolation (subprocess-based sandboxing like VS Code Extension Host)
- Capability-based security (Deno-style permission model)
- Circuit breaker & bulkhead patterns (fault isolation)
- Cryptographic code signing (ED25519 verification)
- OpenTelemetry integration (distributed tracing & metrics)
- Hot module replacement with state preservation (Webpack/Vite-style HMR)
- Resource quotas & limits enforcement
- Full lifecycle management (load/unload/activate/suspend)
- Dependency resolution with topological ordering
- Hook system for extensibility points
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

Advanced Features:

    # Process Isolation
    from aion.plugins import ProcessIsolator
    isolator = ProcessIsolator()
    proxy = await isolator.create_isolated_plugin("my-plugin", path, entry_point)

    # Circuit Breaker
    from aion.plugins import CircuitBreaker
    breaker = CircuitBreaker("my-service")
    async with breaker:
        result = await risky_operation()

    # Capability Security
    from aion.plugins import CapabilityChecker, Capability
    checker = CapabilityChecker(plugin_id)
    checker.grant(Capability.READ_FILE, allowed_paths=["/data/*"])
    checker.check(Capability.READ_FILE, "/data/file.txt")

    # Code Signing
    from aion.plugins import PluginSigner, PluginVerifier
    signer = PluginSigner(signing_key)
    signature = signer.sign_plugin(path, plugin_id, version)

    # Distributed Tracing
    from aion.plugins import PluginTracer
    tracer = PluginTracer(plugin_id)
    with tracer.start_span("operation"):
        # traced operation

    # Hot Module Replacement
    from aion.plugins import HotModuleReplacer
    hmr = HotModuleReplacer(plugin_id, plugin_path)
    hmr.state_transfer.register_dispose(module_id, dispose_handler)
    hmr.state_transfer.register_accept(module_id, accept_handler)
    await hmr.start()

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

# === SOTA Features ===

# Process Isolation
from aion.plugins.isolation import (
    ProcessIsolator,
    IsolatedPluginProxy,
    PluginProcess,
    ProcessConfig,
    IPCMessage,
    IPCMessageType,
    ResourceQuota,
    ResourceEnforcer,
    ResourceViolation,
)

# Resilience (Circuit Breaker, Bulkhead, Retry)
from aion.plugins.resilience import (
    CircuitBreaker,
    CircuitState,
    CircuitBreakerConfig,
    CircuitBreakerRegistry,
    Bulkhead,
    BulkheadConfig,
    BulkheadRegistry,
    BulkheadFullError,
    RetryPolicy,
    RetryConfig,
    ExponentialBackoff,
)

# Security (Code Signing, Capabilities)
from aion.plugins.security import (
    PluginSigner,
    PluginVerifier,
    SignatureInfo,
    SigningKey,
    VerifyingKey,
    SignatureError,
    InvalidSignatureError,
    ExpiredSignatureError,
    Capability,
    CapabilityGrant,
    CapabilityChecker,
    CapabilityDeniedError,
    CapabilityPrompt,
)

# Telemetry (Tracing, Metrics)
from aion.plugins.telemetry import (
    PluginTracer,
    SpanContext,
    TracingConfig,
    trace_plugin_operation,
    PluginMetrics as TelemetryMetrics,
    MetricsConfig,
    Counter,
    Gauge,
    Histogram,
)

# Hot Module Replacement
from aion.plugins.hotreload import (
    HotModuleReplacer,
    HMRConfig,
    ModuleState,
    StateTransfer,
    HMREvent,
    HMREventType,
)


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

    # === SOTA: Process Isolation ===
    "ProcessIsolator",
    "IsolatedPluginProxy",
    "PluginProcess",
    "ProcessConfig",
    "IPCMessage",
    "IPCMessageType",
    "ResourceQuota",
    "ResourceEnforcer",
    "ResourceViolation",

    # === SOTA: Resilience ===
    "CircuitBreaker",
    "CircuitState",
    "CircuitBreakerConfig",
    "CircuitBreakerRegistry",
    "Bulkhead",
    "BulkheadConfig",
    "BulkheadRegistry",
    "BulkheadFullError",
    "RetryPolicy",
    "RetryConfig",
    "ExponentialBackoff",

    # === SOTA: Security ===
    "PluginSigner",
    "PluginVerifier",
    "SignatureInfo",
    "SigningKey",
    "VerifyingKey",
    "SignatureError",
    "InvalidSignatureError",
    "ExpiredSignatureError",
    "Capability",
    "CapabilityGrant",
    "CapabilityChecker",
    "CapabilityDeniedError",
    "CapabilityPrompt",

    # === SOTA: Telemetry ===
    "PluginTracer",
    "SpanContext",
    "TracingConfig",
    "trace_plugin_operation",
    "TelemetryMetrics",
    "MetricsConfig",
    "Counter",
    "Gauge",
    "Histogram",

    # === SOTA: Hot Module Replacement ===
    "HotModuleReplacer",
    "HMRConfig",
    "ModuleState",
    "StateTransfer",
    "HMREvent",
    "HMREventType",
]

# Version
__version__ = "2.0.0"  # Major version bump for SOTA features
