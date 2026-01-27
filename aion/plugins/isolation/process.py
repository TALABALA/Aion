"""
Process Isolation for Plugins

Implements subprocess-based plugin execution with IPC communication,
providing true isolation similar to VS Code's Extension Host model.
"""

from __future__ import annotations

import asyncio
import json
import multiprocessing
import os
import pickle
import signal
import sys
import time
import traceback
import uuid
from dataclasses import dataclass, field
from enum import Enum
from multiprocessing import Process, Queue
from pathlib import Path
from typing import Any, Callable, Optional, Type

import structlog

logger = structlog.get_logger(__name__)


class IPCMessageType(str, Enum):
    """Types of IPC messages between host and plugin process."""

    # Lifecycle
    INIT = "init"
    INIT_RESPONSE = "init_response"
    SHUTDOWN = "shutdown"
    SHUTDOWN_RESPONSE = "shutdown_response"

    # Method calls
    CALL = "call"
    CALL_RESPONSE = "call_response"

    # Events
    EVENT = "event"
    HOOK = "hook"
    HOOK_RESPONSE = "hook_response"

    # Health
    HEARTBEAT = "heartbeat"
    HEARTBEAT_RESPONSE = "heartbeat_response"

    # Errors
    ERROR = "error"
    FATAL = "fatal"

    # Resource monitoring
    RESOURCE_REPORT = "resource_report"
    RESOURCE_LIMIT_EXCEEDED = "resource_limit_exceeded"


@dataclass
class IPCMessage:
    """Message for inter-process communication."""

    type: IPCMessageType
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    plugin_id: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "type": self.type.value,
            "id": self.id,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "plugin_id": self.plugin_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IPCMessage":
        """Create from dictionary."""
        return cls(
            type=IPCMessageType(data["type"]),
            id=data["id"],
            payload=data.get("payload", {}),
            timestamp=data.get("timestamp", time.time()),
            plugin_id=data.get("plugin_id"),
        )


@dataclass
class ProcessConfig:
    """Configuration for isolated plugin process."""

    plugin_id: str
    plugin_path: str
    entry_point: str

    # Resource limits
    max_memory_mb: int = 512
    max_cpu_percent: float = 50.0
    max_file_descriptors: int = 100
    max_threads: int = 10

    # Timeouts
    init_timeout: float = 30.0
    call_timeout: float = 60.0
    shutdown_timeout: float = 10.0
    heartbeat_interval: float = 5.0
    heartbeat_timeout: float = 15.0

    # Permissions (capability-based)
    allowed_paths: list[str] = field(default_factory=list)
    allowed_network: bool = False
    allowed_env_vars: list[str] = field(default_factory=list)
    allowed_subprocesses: bool = False

    # Environment
    env_vars: dict[str, str] = field(default_factory=dict)
    working_dir: Optional[str] = None


class PluginProcessWorker:
    """
    Worker that runs inside the isolated subprocess.

    This class is instantiated in the child process and handles
    all plugin operations in isolation.
    """

    def __init__(
        self,
        config: ProcessConfig,
        to_host: Queue,
        from_host: Queue,
    ):
        self.config = config
        self.to_host = to_host
        self.from_host = from_host
        self.plugin_instance = None
        self.running = True
        self._start_time = time.time()

    def run(self) -> None:
        """Main loop for the plugin process."""
        try:
            # Set up resource limits
            self._setup_resource_limits()

            # Set up signal handlers
            signal.signal(signal.SIGTERM, self._handle_sigterm)
            signal.signal(signal.SIGINT, self._handle_sigint)

            # Main message loop
            while self.running:
                try:
                    # Check for messages with timeout
                    if not self.from_host.empty():
                        msg_data = self.from_host.get(timeout=0.1)
                        msg = IPCMessage.from_dict(msg_data)
                        self._handle_message(msg)
                    else:
                        time.sleep(0.01)

                except Exception as e:
                    self._send_error(str(e), traceback.format_exc())

        except Exception as e:
            self._send_fatal(str(e), traceback.format_exc())

    def _setup_resource_limits(self) -> None:
        """Set up resource limits for the process."""
        try:
            import resource

            # Memory limit
            memory_bytes = self.config.max_memory_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))

            # File descriptor limit
            resource.setrlimit(
                resource.RLIMIT_NOFILE,
                (self.config.max_file_descriptors, self.config.max_file_descriptors)
            )

            # CPU time limit (soft limit for warning)
            # Note: This is cumulative CPU time, not percentage
            resource.setrlimit(resource.RLIMIT_CPU, (3600, 7200))  # 1hr soft, 2hr hard

        except (ImportError, ValueError, OSError) as e:
            # resource module not available on all platforms
            logger.warning(f"Could not set resource limits: {e}")

    def _handle_sigterm(self, signum, frame) -> None:
        """Handle SIGTERM signal."""
        self.running = False

    def _handle_sigint(self, signum, frame) -> None:
        """Handle SIGINT signal."""
        self.running = False

    def _handle_message(self, msg: IPCMessage) -> None:
        """Handle an incoming IPC message."""
        handlers = {
            IPCMessageType.INIT: self._handle_init,
            IPCMessageType.SHUTDOWN: self._handle_shutdown,
            IPCMessageType.CALL: self._handle_call,
            IPCMessageType.HEARTBEAT: self._handle_heartbeat,
            IPCMessageType.HOOK: self._handle_hook,
        }

        handler = handlers.get(msg.type)
        if handler:
            handler(msg)
        else:
            self._send_error(f"Unknown message type: {msg.type}")

    def _handle_init(self, msg: IPCMessage) -> None:
        """Initialize the plugin."""
        try:
            # Load the plugin module
            plugin_path = self.config.plugin_path
            entry_point = self.config.entry_point

            # Add plugin path to sys.path
            if plugin_path not in sys.path:
                sys.path.insert(0, plugin_path)

            # Parse entry point (module:class)
            module_name, class_name = entry_point.split(":")

            # Import module
            import importlib
            module = importlib.import_module(module_name)

            # Get plugin class
            plugin_class = getattr(module, class_name)

            # Instantiate plugin
            self.plugin_instance = plugin_class()

            # Call initialize if available
            if hasattr(self.plugin_instance, "initialize"):
                init_result = self.plugin_instance.initialize()
                # Handle async initialization
                if asyncio.iscoroutine(init_result):
                    asyncio.get_event_loop().run_until_complete(init_result)

            self._send_response(msg.id, IPCMessageType.INIT_RESPONSE, {
                "success": True,
                "plugin_id": self.config.plugin_id,
            })

        except Exception as e:
            self._send_response(msg.id, IPCMessageType.INIT_RESPONSE, {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
            })

    def _handle_shutdown(self, msg: IPCMessage) -> None:
        """Shutdown the plugin."""
        try:
            if self.plugin_instance and hasattr(self.plugin_instance, "shutdown"):
                shutdown_result = self.plugin_instance.shutdown()
                if asyncio.iscoroutine(shutdown_result):
                    asyncio.get_event_loop().run_until_complete(shutdown_result)

            self._send_response(msg.id, IPCMessageType.SHUTDOWN_RESPONSE, {
                "success": True,
            })

            self.running = False

        except Exception as e:
            self._send_response(msg.id, IPCMessageType.SHUTDOWN_RESPONSE, {
                "success": False,
                "error": str(e),
            })
            self.running = False

    def _handle_call(self, msg: IPCMessage) -> None:
        """Handle a method call on the plugin."""
        try:
            method_name = msg.payload.get("method")
            args = msg.payload.get("args", [])
            kwargs = msg.payload.get("kwargs", {})

            if not self.plugin_instance:
                raise RuntimeError("Plugin not initialized")

            if not hasattr(self.plugin_instance, method_name):
                raise AttributeError(f"Plugin has no method: {method_name}")

            method = getattr(self.plugin_instance, method_name)
            result = method(*args, **kwargs)

            # Handle async methods
            if asyncio.iscoroutine(result):
                result = asyncio.get_event_loop().run_until_complete(result)

            # Serialize result
            try:
                serialized = json.dumps(result)
                result_data = json.loads(serialized)
            except (TypeError, ValueError):
                # Fall back to string representation
                result_data = str(result)

            self._send_response(msg.id, IPCMessageType.CALL_RESPONSE, {
                "success": True,
                "result": result_data,
            })

        except Exception as e:
            self._send_response(msg.id, IPCMessageType.CALL_RESPONSE, {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
            })

    def _handle_heartbeat(self, msg: IPCMessage) -> None:
        """Respond to heartbeat."""
        self._send_response(msg.id, IPCMessageType.HEARTBEAT_RESPONSE, {
            "uptime": time.time() - self._start_time,
            "memory_mb": self._get_memory_usage(),
        })

    def _handle_hook(self, msg: IPCMessage) -> None:
        """Handle a hook invocation."""
        try:
            hook_name = msg.payload.get("hook")
            args = msg.payload.get("args", [])
            kwargs = msg.payload.get("kwargs", {})

            if not self.plugin_instance:
                raise RuntimeError("Plugin not initialized")

            # Check if plugin has hook handler
            handler_name = f"on_{hook_name.replace('.', '_')}"
            if hasattr(self.plugin_instance, handler_name):
                handler = getattr(self.plugin_instance, handler_name)
                result = handler(*args, **kwargs)

                if asyncio.iscoroutine(result):
                    result = asyncio.get_event_loop().run_until_complete(result)
            else:
                result = None

            self._send_response(msg.id, IPCMessageType.HOOK_RESPONSE, {
                "success": True,
                "result": result,
            })

        except Exception as e:
            self._send_response(msg.id, IPCMessageType.HOOK_RESPONSE, {
                "success": False,
                "error": str(e),
            })

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import resource
            usage = resource.getrusage(resource.RUSAGE_SELF)
            return usage.ru_maxrss / 1024  # Convert KB to MB
        except ImportError:
            return 0.0

    def _send_response(
        self,
        request_id: str,
        msg_type: IPCMessageType,
        payload: dict[str, Any],
    ) -> None:
        """Send a response message to the host."""
        msg = IPCMessage(
            type=msg_type,
            id=request_id,
            payload=payload,
            plugin_id=self.config.plugin_id,
        )
        self.to_host.put(msg.to_dict())

    def _send_error(self, error: str, tb: Optional[str] = None) -> None:
        """Send an error message to the host."""
        msg = IPCMessage(
            type=IPCMessageType.ERROR,
            payload={"error": error, "traceback": tb},
            plugin_id=self.config.plugin_id,
        )
        self.to_host.put(msg.to_dict())

    def _send_fatal(self, error: str, tb: Optional[str] = None) -> None:
        """Send a fatal error message to the host."""
        msg = IPCMessage(
            type=IPCMessageType.FATAL,
            payload={"error": error, "traceback": tb},
            plugin_id=self.config.plugin_id,
        )
        self.to_host.put(msg.to_dict())


def _plugin_process_entry(
    config_dict: dict,
    to_host: Queue,
    from_host: Queue,
) -> None:
    """Entry point for the plugin subprocess."""
    config = ProcessConfig(**config_dict)
    worker = PluginProcessWorker(config, to_host, from_host)
    worker.run()


class PluginProcess:
    """
    Manages a single isolated plugin process.

    Handles process lifecycle, IPC communication, and health monitoring.
    """

    def __init__(self, config: ProcessConfig):
        self.config = config
        self.process: Optional[Process] = None
        self.to_plugin: Optional[Queue] = None
        self.from_plugin: Optional[Queue] = None
        self._pending_calls: dict[str, asyncio.Future] = {}
        self._message_handler_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._last_heartbeat: float = 0
        self._initialized = False

    async def start(self) -> bool:
        """Start the plugin process."""
        try:
            # Create IPC queues
            ctx = multiprocessing.get_context("spawn")
            self.to_plugin = ctx.Queue()
            self.from_plugin = ctx.Queue()

            # Convert config to dict for pickling
            config_dict = {
                "plugin_id": self.config.plugin_id,
                "plugin_path": self.config.plugin_path,
                "entry_point": self.config.entry_point,
                "max_memory_mb": self.config.max_memory_mb,
                "max_cpu_percent": self.config.max_cpu_percent,
                "max_file_descriptors": self.config.max_file_descriptors,
                "max_threads": self.config.max_threads,
                "allowed_paths": self.config.allowed_paths,
                "allowed_network": self.config.allowed_network,
                "allowed_env_vars": self.config.allowed_env_vars,
                "allowed_subprocesses": self.config.allowed_subprocesses,
            }

            # Start process
            self.process = ctx.Process(
                target=_plugin_process_entry,
                args=(config_dict, self.from_plugin, self.to_plugin),
                daemon=True,
            )
            self.process.start()

            # Start message handler
            self._message_handler_task = asyncio.create_task(
                self._message_handler_loop()
            )

            # Initialize plugin
            response = await self._send_and_wait(
                IPCMessageType.INIT,
                {},
                timeout=self.config.init_timeout,
            )

            if not response.payload.get("success"):
                error = response.payload.get("error", "Unknown error")
                raise RuntimeError(f"Plugin initialization failed: {error}")

            self._initialized = True

            # Start heartbeat monitor
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            self._last_heartbeat = time.time()

            logger.info(
                "Plugin process started",
                plugin_id=self.config.plugin_id,
                pid=self.process.pid,
            )

            return True

        except Exception as e:
            logger.error(
                "Failed to start plugin process",
                plugin_id=self.config.plugin_id,
                error=str(e),
            )
            await self.stop()
            raise

    async def stop(self) -> None:
        """Stop the plugin process."""
        try:
            # Cancel tasks
            if self._heartbeat_task:
                self._heartbeat_task.cancel()
                try:
                    await self._heartbeat_task
                except asyncio.CancelledError:
                    pass

            if self._message_handler_task:
                self._message_handler_task.cancel()
                try:
                    await self._message_handler_task
                except asyncio.CancelledError:
                    pass

            # Send shutdown if initialized
            if self._initialized and self.process and self.process.is_alive():
                try:
                    await self._send_and_wait(
                        IPCMessageType.SHUTDOWN,
                        {},
                        timeout=self.config.shutdown_timeout,
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        "Plugin shutdown timed out",
                        plugin_id=self.config.plugin_id,
                    )

            # Terminate process if still running
            if self.process and self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=5)

                if self.process.is_alive():
                    self.process.kill()
                    self.process.join(timeout=1)

            # Cancel pending calls
            for future in self._pending_calls.values():
                if not future.done():
                    future.cancel()

            self._pending_calls.clear()
            self._initialized = False

            logger.info(
                "Plugin process stopped",
                plugin_id=self.config.plugin_id,
            )

        except Exception as e:
            logger.error(
                "Error stopping plugin process",
                plugin_id=self.config.plugin_id,
                error=str(e),
            )

    async def call(
        self,
        method: str,
        *args,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Any:
        """Call a method on the plugin."""
        if not self._initialized:
            raise RuntimeError("Plugin not initialized")

        response = await self._send_and_wait(
            IPCMessageType.CALL,
            {
                "method": method,
                "args": args,
                "kwargs": kwargs,
            },
            timeout=timeout or self.config.call_timeout,
        )

        if not response.payload.get("success"):
            error = response.payload.get("error", "Unknown error")
            raise RuntimeError(f"Plugin method call failed: {error}")

        return response.payload.get("result")

    async def invoke_hook(
        self,
        hook_name: str,
        *args,
        **kwargs,
    ) -> Any:
        """Invoke a hook on the plugin."""
        if not self._initialized:
            return None

        try:
            response = await self._send_and_wait(
                IPCMessageType.HOOK,
                {
                    "hook": hook_name,
                    "args": args,
                    "kwargs": kwargs,
                },
                timeout=self.config.call_timeout,
            )

            if response.payload.get("success"):
                return response.payload.get("result")
            return None

        except asyncio.TimeoutError:
            logger.warning(
                "Hook invocation timed out",
                plugin_id=self.config.plugin_id,
                hook=hook_name,
            )
            return None

    def is_alive(self) -> bool:
        """Check if the process is alive."""
        return self.process is not None and self.process.is_alive()

    def is_healthy(self) -> bool:
        """Check if the plugin is healthy based on heartbeat."""
        if not self.is_alive():
            return False
        return (time.time() - self._last_heartbeat) < self.config.heartbeat_timeout

    async def _send_and_wait(
        self,
        msg_type: IPCMessageType,
        payload: dict[str, Any],
        timeout: float,
    ) -> IPCMessage:
        """Send a message and wait for response."""
        msg = IPCMessage(
            type=msg_type,
            payload=payload,
            plugin_id=self.config.plugin_id,
        )

        # Create future for response
        future = asyncio.get_event_loop().create_future()
        self._pending_calls[msg.id] = future

        try:
            # Send message
            self.to_plugin.put(msg.to_dict())

            # Wait for response
            return await asyncio.wait_for(future, timeout=timeout)

        finally:
            self._pending_calls.pop(msg.id, None)

    async def _message_handler_loop(self) -> None:
        """Handle incoming messages from the plugin process."""
        while True:
            try:
                await asyncio.sleep(0.01)

                while not self.from_plugin.empty():
                    msg_data = self.from_plugin.get_nowait()
                    msg = IPCMessage.from_dict(msg_data)

                    # Check if this is a response to a pending call
                    if msg.id in self._pending_calls:
                        future = self._pending_calls[msg.id]
                        if not future.done():
                            future.set_result(msg)
                    elif msg.type == IPCMessageType.ERROR:
                        logger.error(
                            "Plugin error",
                            plugin_id=self.config.plugin_id,
                            error=msg.payload.get("error"),
                        )
                    elif msg.type == IPCMessageType.FATAL:
                        logger.error(
                            "Plugin fatal error",
                            plugin_id=self.config.plugin_id,
                            error=msg.payload.get("error"),
                        )
                        # Process will likely terminate

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    "Message handler error",
                    plugin_id=self.config.plugin_id,
                    error=str(e),
                )

    async def _heartbeat_loop(self) -> None:
        """Monitor plugin health via heartbeats."""
        while True:
            try:
                await asyncio.sleep(self.config.heartbeat_interval)

                try:
                    response = await self._send_and_wait(
                        IPCMessageType.HEARTBEAT,
                        {},
                        timeout=self.config.heartbeat_timeout / 2,
                    )
                    self._last_heartbeat = time.time()

                except asyncio.TimeoutError:
                    logger.warning(
                        "Plugin heartbeat timeout",
                        plugin_id=self.config.plugin_id,
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    "Heartbeat error",
                    plugin_id=self.config.plugin_id,
                    error=str(e),
                )


class IsolatedPluginProxy:
    """
    Proxy for an isolated plugin that forwards calls to the subprocess.

    This class provides a transparent interface to the plugin running
    in the isolated process.
    """

    def __init__(self, plugin_process: PluginProcess):
        self._process = plugin_process

    def __getattr__(self, name: str):
        """Forward attribute access to plugin process."""
        async def method_proxy(*args, **kwargs):
            return await self._process.call(name, *args, **kwargs)
        return method_proxy

    async def initialize(self) -> None:
        """Initialize is handled by process start."""
        pass

    async def shutdown(self) -> None:
        """Shutdown is handled by process stop."""
        pass


class ProcessIsolator:
    """
    Manages multiple isolated plugin processes.

    Provides a unified interface for creating, managing, and destroying
    isolated plugin environments.
    """

    def __init__(
        self,
        default_memory_mb: int = 512,
        default_timeout: float = 60.0,
    ):
        self.default_memory_mb = default_memory_mb
        self.default_timeout = default_timeout
        self._processes: dict[str, PluginProcess] = {}
        self._lock = asyncio.Lock()

    async def create_isolated_plugin(
        self,
        plugin_id: str,
        plugin_path: str,
        entry_point: str,
        permissions: Optional[dict[str, Any]] = None,
        resource_limits: Optional[dict[str, Any]] = None,
    ) -> IsolatedPluginProxy:
        """
        Create an isolated plugin process.

        Args:
            plugin_id: Unique plugin identifier
            plugin_path: Path to plugin directory
            entry_point: Module:class entry point
            permissions: Capability permissions
            resource_limits: Resource limits

        Returns:
            Proxy to the isolated plugin
        """
        async with self._lock:
            if plugin_id in self._processes:
                raise ValueError(f"Plugin already isolated: {plugin_id}")

            permissions = permissions or {}
            resource_limits = resource_limits or {}

            config = ProcessConfig(
                plugin_id=plugin_id,
                plugin_path=plugin_path,
                entry_point=entry_point,
                max_memory_mb=resource_limits.get("memory_mb", self.default_memory_mb),
                max_cpu_percent=resource_limits.get("cpu_percent", 50.0),
                call_timeout=resource_limits.get("timeout", self.default_timeout),
                allowed_paths=permissions.get("paths", []),
                allowed_network=permissions.get("network", False),
                allowed_env_vars=permissions.get("env_vars", []),
                allowed_subprocesses=permissions.get("subprocesses", False),
            )

            process = PluginProcess(config)
            await process.start()

            self._processes[plugin_id] = process

            return IsolatedPluginProxy(process)

    async def destroy_isolated_plugin(self, plugin_id: str) -> None:
        """Destroy an isolated plugin process."""
        async with self._lock:
            process = self._processes.pop(plugin_id, None)
            if process:
                await process.stop()

    async def invoke_hook(
        self,
        hook_name: str,
        *args,
        **kwargs,
    ) -> list[Any]:
        """Invoke a hook on all isolated plugins."""
        results = []
        for process in self._processes.values():
            try:
                result = await process.invoke_hook(hook_name, *args, **kwargs)
                if result is not None:
                    results.append(result)
            except Exception as e:
                logger.error(
                    "Hook invocation failed",
                    plugin_id=process.config.plugin_id,
                    hook=hook_name,
                    error=str(e),
                )
        return results

    def get_process(self, plugin_id: str) -> Optional[PluginProcess]:
        """Get a plugin process by ID."""
        return self._processes.get(plugin_id)

    def list_processes(self) -> list[str]:
        """List all isolated plugin IDs."""
        return list(self._processes.keys())

    async def shutdown(self) -> None:
        """Shutdown all isolated plugins."""
        async with self._lock:
            for process in self._processes.values():
                try:
                    await process.stop()
                except Exception as e:
                    logger.error(
                        "Error stopping process",
                        plugin_id=process.config.plugin_id,
                        error=str(e),
                    )
            self._processes.clear()

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about isolated processes."""
        return {
            "total_processes": len(self._processes),
            "processes": {
                pid: {
                    "alive": proc.is_alive(),
                    "healthy": proc.is_healthy(),
                }
                for pid, proc in self._processes.items()
            },
        }
