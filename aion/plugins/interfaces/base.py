"""
AION Base Plugin Interface

All plugins must implement this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TYPE_CHECKING
import asyncio
import time

import structlog

if TYPE_CHECKING:
    from aion.core.kernel import AIONKernel
    from aion.plugins.types import PluginManifest, PluginPermissions

logger = structlog.get_logger(__name__)


class PluginContext:
    """
    Context provided to plugins for accessing AION services.

    This provides a controlled interface to AION functionality
    based on the plugin's permissions.
    """

    def __init__(
        self,
        kernel: "AIONKernel",
        plugin_id: str,
        permissions: "PluginPermissions",
    ):
        self._kernel = kernel
        self._plugin_id = plugin_id
        self._permissions = permissions
        self._logger = structlog.get_logger(f"plugin.{plugin_id}")

    @property
    def plugin_id(self) -> str:
        """Get plugin identifier."""
        return self._plugin_id

    @property
    def logger(self) -> structlog.stdlib.BoundLogger:
        """Get plugin-specific logger."""
        return self._logger

    # === Memory Access ===

    async def store_memory(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        namespace: Optional[str] = None,
    ) -> Optional[str]:
        """Store a memory (if permitted)."""
        if not self._permissions.memory_access:
            raise PermissionError("Plugin does not have memory access")

        ns = namespace or self._plugin_id
        if (
            self._permissions.allowed_memory_namespaces
            and ns not in self._permissions.allowed_memory_namespaces
        ):
            raise PermissionError(f"Access to namespace '{ns}' not permitted")

        if hasattr(self._kernel, "_memory_system") and self._kernel._memory_system:
            return await self._kernel._memory_system.store(
                content=content,
                metadata={**(metadata or {}), "plugin_id": self._plugin_id},
            )
        return None

    async def search_memory(
        self,
        query: str,
        limit: int = 10,
        namespace: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search memories (if permitted)."""
        if not self._permissions.memory_access:
            raise PermissionError("Plugin does not have memory access")

        if hasattr(self._kernel, "_memory_system") and self._kernel._memory_system:
            results = await self._kernel._memory_system.search(
                query=query,
                limit=limit,
            )
            return [r.to_dict() if hasattr(r, "to_dict") else r for r in results]
        return []

    # === Tool Execution ===

    async def execute_tool(
        self,
        tool_name: str,
        params: Dict[str, Any],
    ) -> Any:
        """Execute another tool (if permitted)."""
        api_endpoint = f"tools/{tool_name}"
        if (
            self._permissions.api_access
            and api_endpoint not in self._permissions.api_access
            and "tools/*" not in self._permissions.api_access
        ):
            raise PermissionError(f"Access to tool '{tool_name}' not permitted")

        if hasattr(self._kernel, "_tool_orchestrator") and self._kernel._tool_orchestrator:
            return await self._kernel._tool_orchestrator.execute(
                tool_name=tool_name,
                params=params,
            )
        raise RuntimeError("Tool orchestrator not available")

    # === HTTP Requests ===

    async def http_request(
        self,
        method: str,
        url: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """Make HTTP request (if permitted)."""
        if not self._permissions.network_access:
            raise PermissionError("Plugin does not have network access")

        # Extract domain from URL
        from urllib.parse import urlparse

        parsed = urlparse(url)
        domain = parsed.netloc

        if not self._permissions.is_allowed_domain(domain):
            raise PermissionError(f"Access to domain '{domain}' not permitted")

        # Use aiohttp for requests
        import aiohttp

        async with aiohttp.ClientSession() as session:
            async with session.request(method, url, **kwargs) as response:
                return {
                    "status": response.status,
                    "headers": dict(response.headers),
                    "body": await response.text(),
                }

    # === Configuration ===

    def get_config(self, key: str, default: Any = None) -> Any:
        """Get plugin configuration value."""
        # This should be set by the plugin manager
        return default

    # === Events ===

    async def emit_event(self, event_name: str, data: Dict[str, Any]) -> None:
        """Emit a plugin event."""
        if hasattr(self._kernel, "_event_bus") and self._kernel._event_bus:
            await self._kernel._event_bus.publish(
                f"plugin.{self._plugin_id}.{event_name}",
                {"plugin_id": self._plugin_id, **data},
            )


class BasePlugin(ABC):
    """
    Base interface for all AION plugins.

    Plugins must implement:
    - get_manifest(): Return plugin metadata
    - initialize(): Called when plugin is loaded
    - shutdown(): Called when plugin is unloaded

    Optionally implement:
    - configure(): Handle configuration changes
    - health_check(): Return plugin health status
    - on_activate()/on_suspend(): Lifecycle hooks
    """

    def __init__(self):
        self._kernel: Optional["AIONKernel"] = None
        self._context: Optional[PluginContext] = None
        self._config: Dict[str, Any] = {}
        self._initialized = False
        self._active = False
        self._start_time: Optional[float] = None

    @property
    def kernel(self) -> Optional["AIONKernel"]:
        """Get AION kernel reference."""
        return self._kernel

    @property
    def context(self) -> Optional[PluginContext]:
        """Get plugin context."""
        return self._context

    @property
    def config(self) -> Dict[str, Any]:
        """Get plugin configuration."""
        return self._config

    @property
    def is_initialized(self) -> bool:
        """Check if plugin is initialized."""
        return self._initialized

    @property
    def is_active(self) -> bool:
        """Check if plugin is active."""
        return self._active

    @property
    def uptime(self) -> float:
        """Get plugin uptime in seconds."""
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time

    # === Required Methods ===

    @classmethod
    @abstractmethod
    def get_manifest(cls) -> "PluginManifest":
        """
        Return the plugin manifest.

        This is called before instantiation to get plugin metadata.
        Must be a classmethod as it's called without an instance.

        Returns:
            PluginManifest with plugin metadata
        """
        pass

    @abstractmethod
    async def initialize(
        self,
        kernel: "AIONKernel",
        config: Dict[str, Any],
    ) -> None:
        """
        Initialize the plugin.

        Called when the plugin is loaded. Setup resources,
        connections, and internal state here.

        Args:
            kernel: AION kernel reference
            config: Plugin configuration dictionary
        """
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """
        Shutdown the plugin.

        Called when the plugin is unloaded. Clean up resources,
        close connections, save state, etc.
        """
        pass

    # === Optional Methods ===

    async def configure(self, config: Dict[str, Any]) -> None:
        """
        Handle configuration update.

        Called when plugin configuration changes at runtime.
        Default implementation just stores the config.

        Args:
            config: New configuration dictionary
        """
        old_config = self._config.copy()
        self._config = config

        # Call on_config_changed if config actually changed
        if old_config != config:
            await self.on_config_changed(old_config, config)

    async def on_config_changed(
        self,
        old_config: Dict[str, Any],
        new_config: Dict[str, Any],
    ) -> None:
        """
        Called when configuration changes.

        Override to handle configuration changes.

        Args:
            old_config: Previous configuration
            new_config: New configuration
        """
        pass

    async def health_check(self) -> Dict[str, Any]:
        """
        Return plugin health status.

        Override to provide detailed health information.

        Returns:
            Health status dict with at least {"healthy": bool}
        """
        return {
            "healthy": self._initialized and self._active,
            "initialized": self._initialized,
            "active": self._active,
            "uptime_seconds": self.uptime,
        }

    def validate_permissions(self, requested: "PluginPermissions") -> bool:
        """
        Validate that plugin has required permissions.

        Override to implement custom permission validation.

        Args:
            requested: Requested permissions

        Returns:
            True if permissions are acceptable
        """
        return True

    # === Lifecycle Hooks ===

    async def on_activate(self) -> None:
        """
        Called when plugin is activated.

        Override to perform actions when plugin becomes active.
        """
        self._active = True
        self._start_time = time.time()

    async def on_suspend(self) -> None:
        """
        Called when plugin is suspended.

        Override to pause operations, close connections, etc.
        """
        self._active = False

    async def on_resume(self) -> None:
        """
        Called when plugin is resumed from suspension.

        Override to restore operations.
        """
        self._active = True

    async def on_error(self, error: Exception) -> None:
        """
        Called when an error occurs in the plugin.

        Override to handle errors, potentially recovering.

        Args:
            error: The exception that occurred
        """
        logger.error(f"Plugin error: {error}", exc_info=error)

    async def on_reload(self) -> None:
        """
        Called before plugin is hot-reloaded.

        Override to prepare for reload (save state, etc.).
        """
        pass

    # === Utility Methods ===

    def _set_context(
        self,
        kernel: "AIONKernel",
        permissions: "PluginPermissions",
    ) -> None:
        """Set plugin context (called by plugin manager)."""
        manifest = self.get_manifest()
        self._kernel = kernel
        self._context = PluginContext(kernel, manifest.id, permissions)

    def _mark_initialized(self) -> None:
        """Mark plugin as initialized (called by plugin manager)."""
        self._initialized = True

    def _mark_shutdown(self) -> None:
        """Mark plugin as shutdown (called by plugin manager)."""
        self._initialized = False
        self._active = False
        self._kernel = None
        self._context = None


class PluginMixin:
    """
    Mixin class providing common plugin functionality.

    Can be used with BasePlugin to add shared behavior.
    """

    _retry_count: int = 3
    _retry_delay: float = 1.0

    async def with_retry(
        self,
        func,
        *args,
        retries: Optional[int] = None,
        delay: Optional[float] = None,
        **kwargs,
    ) -> Any:
        """
        Execute function with retry logic.

        Args:
            func: Async function to execute
            *args: Function arguments
            retries: Number of retries (default: 3)
            delay: Delay between retries in seconds
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            Last exception if all retries fail
        """
        retries = retries or self._retry_count
        delay = delay or self._retry_delay
        last_error: Optional[Exception] = None

        for attempt in range(retries + 1):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                return func(*args, **kwargs)
            except Exception as e:
                last_error = e
                if attempt < retries:
                    await asyncio.sleep(delay * (2 ** attempt))

        raise last_error  # type: ignore

    def validate_config_schema(
        self,
        config: Dict[str, Any],
        schema: Dict[str, Any],
    ) -> tuple[bool, List[str]]:
        """
        Validate configuration against JSON schema.

        Args:
            config: Configuration to validate
            schema: JSON Schema

        Returns:
            (is_valid, error_messages)
        """
        try:
            import jsonschema

            jsonschema.validate(config, schema)
            return True, []
        except ImportError:
            # jsonschema not available, skip validation
            return True, []
        except jsonschema.ValidationError as e:
            return False, [str(e)]
        except Exception as e:
            return False, [f"Validation error: {e}"]
