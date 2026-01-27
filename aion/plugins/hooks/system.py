"""
AION Plugin Hook System

Manages hook registration and dispatch for plugin extensibility.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Callable, Dict, List, Optional, Set
from functools import wraps
from contextlib import asynccontextmanager

import structlog

from aion.plugins.types import HookDefinition, HookRegistration

logger = structlog.get_logger(__name__)


class HookExecutionError(Exception):
    """Error during hook execution."""

    def __init__(self, hook_name: str, plugin_id: str, message: str):
        self.hook_name = hook_name
        self.plugin_id = plugin_id
        super().__init__(f"Hook '{hook_name}' error from plugin '{plugin_id}': {message}")


class HookSystem:
    """
    Plugin hook system.

    Features:
    - Hook point definition
    - Handler registration with priority
    - Action hooks (side effects)
    - Filter hooks (data transformation)
    - Async and sync handler support
    - Timeout and error handling
    - Hook statistics
    """

    def __init__(self):
        self._hooks: Dict[str, List[HookRegistration]] = {}
        self._definitions: Dict[str, HookDefinition] = {}
        self._disabled_hooks: Set[str] = set()
        self._plugin_hooks: Dict[str, Set[str]] = {}  # plugin_id -> hook names

        # Statistics
        self._stats: Dict[str, Dict[str, Any]] = {}

        # Register built-in hooks
        for hook_def in BUILTIN_HOOKS:
            self.define(hook_def)

    # === Hook Definition ===

    def define(self, definition: HookDefinition) -> None:
        """
        Define a hook point.

        Args:
            definition: Hook definition
        """
        self._definitions[definition.name] = definition
        if definition.name not in self._hooks:
            self._hooks[definition.name] = []
            self._stats[definition.name] = {
                "calls": 0,
                "total_time_ms": 0.0,
                "errors": 0,
            }
        logger.debug(f"Defined hook: {definition.name}")

    def undefine(self, hook_name: str) -> None:
        """Remove a hook definition."""
        self._definitions.pop(hook_name, None)
        self._hooks.pop(hook_name, None)
        self._stats.pop(hook_name, None)

    def is_defined(self, hook_name: str) -> bool:
        """Check if hook is defined."""
        return hook_name in self._definitions

    def get_definition(self, hook_name: str) -> Optional[HookDefinition]:
        """Get hook definition."""
        return self._definitions.get(hook_name)

    # === Registration ===

    def register(
        self,
        hook_name: str,
        plugin_id: str,
        handler: Callable,
        priority: int = 100,
        timeout_seconds: float = 30.0,
    ) -> bool:
        """
        Register a hook handler.

        Args:
            hook_name: Name of hook to register for
            plugin_id: Plugin registering the hook
            handler: Handler function (sync or async)
            priority: Execution priority (lower = earlier)
            timeout_seconds: Handler timeout

        Returns:
            True if registered successfully
        """
        # Create hook list if it doesn't exist (allows dynamic hooks)
        if hook_name not in self._hooks:
            self._hooks[hook_name] = []
            self._stats[hook_name] = {"calls": 0, "total_time_ms": 0.0, "errors": 0}

        # Check for duplicate
        for existing in self._hooks[hook_name]:
            if existing.plugin_id == plugin_id and existing.handler == handler:
                logger.warning(f"Handler already registered for {hook_name} from {plugin_id}")
                return False

        registration = HookRegistration(
            hook_name=hook_name,
            plugin_id=plugin_id,
            handler=handler,
            priority=priority,
            enabled=True,
            timeout_seconds=timeout_seconds,
        )

        self._hooks[hook_name].append(registration)
        self._hooks[hook_name].sort()  # Sort by priority

        # Track plugin's hooks
        if plugin_id not in self._plugin_hooks:
            self._plugin_hooks[plugin_id] = set()
        self._plugin_hooks[plugin_id].add(hook_name)

        logger.debug(f"Registered hook handler: {hook_name} from {plugin_id} (priority={priority})")
        return True

    def unregister(self, hook_name: str, plugin_id: str) -> bool:
        """
        Unregister a plugin's hook handler.

        Args:
            hook_name: Hook name
            plugin_id: Plugin identifier

        Returns:
            True if unregistered
        """
        if hook_name not in self._hooks:
            return False

        original_count = len(self._hooks[hook_name])
        self._hooks[hook_name] = [
            h for h in self._hooks[hook_name]
            if h.plugin_id != plugin_id
        ]

        if plugin_id in self._plugin_hooks:
            self._plugin_hooks[plugin_id].discard(hook_name)

        return len(self._hooks[hook_name]) < original_count

    def unregister_plugin(self, plugin_id: str) -> int:
        """
        Unregister all hooks for a plugin.

        Args:
            plugin_id: Plugin identifier

        Returns:
            Number of hooks unregistered
        """
        count = 0
        hook_names = self._plugin_hooks.get(plugin_id, set()).copy()

        for hook_name in hook_names:
            if self.unregister(hook_name, plugin_id):
                count += 1

        self._plugin_hooks.pop(plugin_id, None)
        logger.debug(f"Unregistered {count} hooks for plugin {plugin_id}")
        return count

    # === Dispatch ===

    async def dispatch(
        self,
        hook_name: str,
        *args,
        **kwargs,
    ) -> List[Any]:
        """
        Dispatch an action hook.

        Calls all handlers and returns their results.
        Handlers are called in priority order.

        Args:
            hook_name: Hook to dispatch
            *args, **kwargs: Arguments to pass to handlers

        Returns:
            List of handler results
        """
        if hook_name in self._disabled_hooks:
            return []

        if hook_name not in self._hooks:
            return []

        results = []
        start_time = time.time()

        definition = self._definitions.get(hook_name)
        fail_fast = definition.fail_fast if definition else False

        for registration in self._hooks[hook_name]:
            if not registration.enabled:
                continue

            try:
                result = await self._call_handler(registration, *args, **kwargs)
                results.append(result)
            except asyncio.TimeoutError:
                logger.warning(
                    f"Hook handler timeout: {hook_name} from {registration.plugin_id}"
                )
                self._stats[hook_name]["errors"] += 1
                if fail_fast:
                    raise
            except Exception as e:
                logger.error(
                    f"Hook handler error: {hook_name} from {registration.plugin_id}: {e}"
                )
                self._stats[hook_name]["errors"] += 1
                if fail_fast:
                    raise

        # Update stats
        self._stats[hook_name]["calls"] += 1
        self._stats[hook_name]["total_time_ms"] += (time.time() - start_time) * 1000

        return results

    async def filter(
        self,
        hook_name: str,
        value: Any,
        *args,
        **kwargs,
    ) -> Any:
        """
        Apply a filter hook.

        Each handler can modify the value, which is passed to the next.
        Used for data transformation pipelines.

        Args:
            hook_name: Hook to apply
            value: Initial value to filter
            *args, **kwargs: Additional arguments

        Returns:
            Filtered value
        """
        if hook_name in self._disabled_hooks:
            return value

        if hook_name not in self._hooks:
            return value

        start_time = time.time()

        for registration in self._hooks[hook_name]:
            if not registration.enabled:
                continue

            try:
                result = await self._call_handler(registration, value, *args, **kwargs)
                value = result
            except asyncio.TimeoutError:
                logger.warning(
                    f"Filter handler timeout: {hook_name} from {registration.plugin_id}"
                )
                self._stats[hook_name]["errors"] += 1
            except Exception as e:
                logger.error(
                    f"Filter handler error: {hook_name} from {registration.plugin_id}: {e}"
                )
                self._stats[hook_name]["errors"] += 1

        # Update stats
        self._stats[hook_name]["calls"] += 1
        self._stats[hook_name]["total_time_ms"] += (time.time() - start_time) * 1000

        return value

    async def _call_handler(
        self,
        registration: HookRegistration,
        *args,
        **kwargs,
    ) -> Any:
        """Call a handler with timeout."""
        handler = registration.handler

        if asyncio.iscoroutinefunction(handler):
            coro = handler(*args, **kwargs)
        else:
            # Wrap sync handler
            coro = asyncio.get_event_loop().run_in_executor(
                None, lambda: handler(*args, **kwargs)
            )

        if registration.timeout_seconds > 0:
            return await asyncio.wait_for(coro, timeout=registration.timeout_seconds)
        return await coro

    # === Control ===

    def enable_hook(self, hook_name: str) -> None:
        """Enable a hook."""
        self._disabled_hooks.discard(hook_name)

    def disable_hook(self, hook_name: str) -> None:
        """Disable a hook (all handlers)."""
        self._disabled_hooks.add(hook_name)

    def enable_handler(self, hook_name: str, plugin_id: str) -> None:
        """Enable a specific handler."""
        if hook_name in self._hooks:
            for registration in self._hooks[hook_name]:
                if registration.plugin_id == plugin_id:
                    registration.enabled = True

    def disable_handler(self, hook_name: str, plugin_id: str) -> None:
        """Disable a specific handler."""
        if hook_name in self._hooks:
            for registration in self._hooks[hook_name]:
                if registration.plugin_id == plugin_id:
                    registration.enabled = False

    # === Queries ===

    def get_handlers(self, hook_name: str) -> List[HookRegistration]:
        """Get all handlers for a hook."""
        return self._hooks.get(hook_name, [])

    def get_plugin_hooks(self, plugin_id: str) -> List[str]:
        """Get all hooks a plugin has registered for."""
        return list(self._plugin_hooks.get(plugin_id, set()))

    def list_hooks(self) -> List[str]:
        """List all defined hook names."""
        return list(self._definitions.keys())

    def list_active_hooks(self) -> List[str]:
        """List hooks with registered handlers."""
        return [name for name, handlers in self._hooks.items() if handlers]

    def get_stats(self, hook_name: Optional[str] = None) -> Dict[str, Any]:
        """Get hook statistics."""
        if hook_name:
            return self._stats.get(hook_name, {})
        return self._stats.copy()

    # === Context Manager ===

    @asynccontextmanager
    async def hook_context(self, hook_name: str, *args, **kwargs):
        """
        Context manager for before/after hooks.

        Usage:
            async with hooks.hook_context("request", request) as ctx:
                # Do work
                ctx["response"] = response
        """
        context: Dict[str, Any] = {"args": args, "kwargs": kwargs}

        # Call before hooks
        await self.dispatch(f"{hook_name}.before", context)

        try:
            yield context
        finally:
            # Call after hooks
            await self.dispatch(f"{hook_name}.after", context)


# === Decorator ===


def hook_handler(hook_name: str, priority: int = 100):
    """
    Decorator to mark a method as a hook handler.

    Usage:
        class MyPlugin(BasePlugin):
            @hook_handler("request.before", priority=50)
            async def on_request(self, context):
                pass
    """
    def decorator(func: Callable) -> Callable:
        func._hook_name = hook_name
        func._hook_priority = priority
        return func
    return decorator


# === Built-in Hooks ===


BUILTIN_HOOKS = [
    # === Request Processing ===
    HookDefinition(
        name="request.before",
        description="Called before processing a request",
        is_filter=True,
        is_async=True,
    ),
    HookDefinition(
        name="request.after",
        description="Called after processing a request",
        is_filter=True,
        is_async=True,
    ),
    HookDefinition(
        name="request.error",
        description="Called when request processing fails",
        is_async=True,
    ),

    # === Tool Execution ===
    HookDefinition(
        name="tool.before_execute",
        description="Called before tool execution, can modify parameters",
        is_filter=True,
        is_async=True,
    ),
    HookDefinition(
        name="tool.after_execute",
        description="Called after tool execution, can modify result",
        is_filter=True,
        is_async=True,
    ),
    HookDefinition(
        name="tool.registered",
        description="Called when a new tool is registered",
        is_async=True,
    ),

    # === Memory Operations ===
    HookDefinition(
        name="memory.before_store",
        description="Called before storing a memory",
        is_filter=True,
        is_async=True,
    ),
    HookDefinition(
        name="memory.after_store",
        description="Called after storing a memory",
        is_async=True,
    ),
    HookDefinition(
        name="memory.before_retrieve",
        description="Called before retrieving memories",
        is_filter=True,
        is_async=True,
    ),
    HookDefinition(
        name="memory.after_retrieve",
        description="Called after retrieving memories, can modify results",
        is_filter=True,
        is_async=True,
    ),

    # === Agent Lifecycle ===
    HookDefinition(
        name="agent.spawning",
        description="Called before an agent is spawned",
        is_filter=True,
        is_async=True,
    ),
    HookDefinition(
        name="agent.spawned",
        description="Called when an agent is spawned",
        is_async=True,
    ),
    HookDefinition(
        name="agent.terminating",
        description="Called before an agent terminates",
        is_async=True,
    ),
    HookDefinition(
        name="agent.terminated",
        description="Called when an agent is terminated",
        is_async=True,
    ),
    HookDefinition(
        name="agent.message",
        description="Called when an agent sends a message",
        is_filter=True,
        is_async=True,
    ),

    # === Planning ===
    HookDefinition(
        name="plan.creating",
        description="Called before a plan is created",
        is_filter=True,
        is_async=True,
    ),
    HookDefinition(
        name="plan.created",
        description="Called when a plan is created",
        is_async=True,
    ),
    HookDefinition(
        name="plan.step_starting",
        description="Called before a plan step starts",
        is_filter=True,
        is_async=True,
    ),
    HookDefinition(
        name="plan.step_completed",
        description="Called when a plan step completes",
        is_async=True,
    ),
    HookDefinition(
        name="plan.completed",
        description="Called when a plan completes",
        is_async=True,
    ),

    # === Workflow ===
    HookDefinition(
        name="workflow.triggered",
        description="Called when a workflow is triggered",
        is_async=True,
    ),
    HookDefinition(
        name="workflow.started",
        description="Called when a workflow starts execution",
        is_async=True,
    ),
    HookDefinition(
        name="workflow.completed",
        description="Called when a workflow completes",
        is_async=True,
    ),
    HookDefinition(
        name="workflow.failed",
        description="Called when a workflow fails",
        is_async=True,
    ),

    # === LLM ===
    HookDefinition(
        name="llm.before_call",
        description="Called before LLM API call",
        is_filter=True,
        is_async=True,
    ),
    HookDefinition(
        name="llm.after_call",
        description="Called after LLM API call",
        is_filter=True,
        is_async=True,
    ),

    # === Security ===
    HookDefinition(
        name="auth.login",
        description="Called after user login",
        is_async=True,
    ),
    HookDefinition(
        name="auth.logout",
        description="Called after user logout",
        is_async=True,
    ),
    HookDefinition(
        name="auth.permission_check",
        description="Called to check permissions",
        is_filter=True,
        is_async=True,
    ),

    # === Plugin Lifecycle ===
    HookDefinition(
        name="plugin.loaded",
        description="Called when a plugin is loaded",
        is_async=True,
    ),
    HookDefinition(
        name="plugin.activated",
        description="Called when a plugin is activated",
        is_async=True,
    ),
    HookDefinition(
        name="plugin.deactivated",
        description="Called when a plugin is deactivated",
        is_async=True,
    ),
    HookDefinition(
        name="plugin.unloaded",
        description="Called when a plugin is unloaded",
        is_async=True,
    ),

    # === System ===
    HookDefinition(
        name="system.startup",
        description="Called during system startup",
        is_async=True,
    ),
    HookDefinition(
        name="system.shutdown",
        description="Called during system shutdown",
        is_async=True,
    ),
    HookDefinition(
        name="system.health_check",
        description="Called during health checks",
        is_filter=True,
        is_async=True,
    ),
]
