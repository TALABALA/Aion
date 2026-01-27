"""
AION Distributed Task Executor

Production-grade local task executor with handler registry and metrics.
Implements SOTA patterns including:
- Typed handler registry mapping task types to async callables
- Timeout enforcement via asyncio.wait_for
- Resource tracking with node capacity increment/decrement
- Full error capture with traceback preservation
- Execution latency histogram for performance monitoring
- Built-in handlers for core AION task types
"""

from __future__ import annotations

import asyncio
import time
import traceback
from collections import defaultdict
from datetime import datetime
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Coroutine,
    Dict,
    List,
    Optional,
    Set,
)

import structlog

from aion.distributed.types import (
    DistributedTask,
    TaskStatus,
    TaskType,
)

if TYPE_CHECKING:
    from aion.distributed.cluster.manager import ClusterManager

logger = structlog.get_logger(__name__)

# Type alias for task handler functions
TaskHandler = Callable[[DistributedTask], Coroutine[Any, Any, Any]]

# Default timeout for tasks without explicit timeout
DEFAULT_TASK_TIMEOUT_SECONDS = 300

# Latency histogram bucket boundaries in milliseconds
LATENCY_BUCKETS = [10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 30000]


class TaskExecutor:
    """
    Executes distributed tasks locally on a single node.

    Maintains a registry of task handlers keyed by task type, enforces
    execution timeouts, tracks resource usage on the local node, and
    captures comprehensive error information including tracebacks.

    The executor registers built-in handlers for all core AION task types
    on initialization. Custom handlers can be registered at any time.

    Attributes:
        cluster_manager: Reference to the cluster manager for node state.
        default_timeout: Default timeout in seconds for task execution.
    """

    def __init__(
        self,
        cluster_manager: ClusterManager,
        *,
        default_timeout: int = DEFAULT_TASK_TIMEOUT_SECONDS,
    ) -> None:
        self._cluster_manager = cluster_manager
        self.default_timeout = default_timeout

        # Handler registry: task_type -> async callable
        self._handlers: Dict[str, TaskHandler] = {}

        # Currently executing tasks for cancellation support
        self._running_tasks: Dict[str, asyncio.Task[Any]] = {}

        # Execution metrics
        self._execution_count: int = 0
        self._error_count: int = 0
        self._timeout_count: int = 0
        self._latency_histogram: Dict[str, int] = {
            f"le_{b}ms": 0 for b in LATENCY_BUCKETS
        }
        self._latency_histogram["le_inf"] = 0
        self._total_latency_ms: float = 0.0
        self._latencies_by_type: Dict[str, List[float]] = defaultdict(list)

        # Register built-in handlers
        self._register_builtin_handlers()

        logger.info(
            "task_executor_initialized",
            default_timeout=default_timeout,
            registered_handlers=list(self._handlers.keys()),
        )

    # -------------------------------------------------------------------------
    # Handler Registry
    # -------------------------------------------------------------------------

    def register_handler(
        self,
        task_type: str,
        handler: TaskHandler,
    ) -> None:
        """
        Register an async handler for a specific task type.

        If a handler is already registered for the given type, it is
        replaced with a warning log.

        Args:
            task_type: The task type string this handler processes.
            handler: An async callable that accepts a DistributedTask
                     and returns the result value.
        """
        if task_type in self._handlers:
            logger.warning(
                "handler_replaced",
                task_type=task_type,
            )
        self._handlers[task_type] = handler
        logger.debug("handler_registered", task_type=task_type)

    def has_handler(self, task_type: str) -> bool:
        """Check if a handler is registered for the given task type."""
        return task_type in self._handlers

    def get_registered_types(self) -> List[str]:
        """Return all task types with registered handlers."""
        return list(self._handlers.keys())

    # -------------------------------------------------------------------------
    # Task Execution
    # -------------------------------------------------------------------------

    async def execute(self, task: DistributedTask) -> Any:
        """
        Execute a task using its registered handler.

        Updates task lifecycle status, enforces timeouts, tracks resource
        usage on the local node, and captures errors with full traceback.

        Args:
            task: The task to execute.

        Returns:
            The result value returned by the handler.

        Raises:
            ValueError: If no handler is registered for the task type.
            asyncio.TimeoutError: If execution exceeds the timeout.
            Exception: Re-raises any handler exception after recording it.
        """
        handler = self._handlers.get(task.task_type)
        if handler is None:
            error_msg = f"No handler registered for task type: {task.task_type}"
            task.status = TaskStatus.FAILED
            task.error = error_msg
            logger.error(
                "no_handler_for_task_type",
                task_id=task.id,
                task_type=task.task_type,
            )
            raise ValueError(error_msg)

        # Update task state to running
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now()

        # Increment resource usage on the local node
        self._increment_node_tasks()

        timeout = task.timeout_seconds or self.default_timeout
        start_time = time.monotonic()

        logger.info(
            "task_execution_started",
            task_id=task.id,
            task_name=task.name,
            task_type=task.task_type,
            timeout_seconds=timeout,
        )

        try:
            # Create a tracked asyncio task for cancellation support
            coro = handler(task)
            async_task = asyncio.ensure_future(coro)
            self._running_tasks[task.id] = async_task

            # Execute with timeout enforcement
            result = await asyncio.wait_for(async_task, timeout=timeout)

            # Mark as completed
            task.status = TaskStatus.COMPLETED
            task.result = result
            task.completed_at = datetime.now()

            elapsed_ms = (time.monotonic() - start_time) * 1000
            self._record_latency(task.task_type, elapsed_ms)
            self._execution_count += 1

            logger.info(
                "task_execution_completed",
                task_id=task.id,
                task_name=task.name,
                elapsed_ms=round(elapsed_ms, 2),
            )
            return result

        except asyncio.TimeoutError:
            task.status = TaskStatus.TIMEOUT
            task.error = f"Task timed out after {timeout} seconds"
            task.completed_at = datetime.now()
            self._timeout_count += 1

            elapsed_ms = (time.monotonic() - start_time) * 1000
            self._record_latency(task.task_type, elapsed_ms)

            logger.error(
                "task_execution_timeout",
                task_id=task.id,
                task_name=task.name,
                timeout_seconds=timeout,
            )
            raise

        except asyncio.CancelledError:
            task.status = TaskStatus.CANCELLED
            task.completed_at = datetime.now()

            logger.info(
                "task_execution_cancelled",
                task_id=task.id,
                task_name=task.name,
            )
            raise

        except Exception as exc:
            task.status = TaskStatus.FAILED
            task.error = str(exc)
            task.error_traceback = traceback.format_exc()
            task.completed_at = datetime.now()
            self._error_count += 1

            elapsed_ms = (time.monotonic() - start_time) * 1000
            self._record_latency(task.task_type, elapsed_ms)

            logger.error(
                "task_execution_failed",
                task_id=task.id,
                task_name=task.name,
                error=str(exc),
                elapsed_ms=round(elapsed_ms, 2),
            )
            raise

        finally:
            # Always decrement resource usage and clean up
            self._decrement_node_tasks()
            self._running_tasks.pop(task.id, None)

    async def cancel(self, task_id: str) -> bool:
        """
        Cancel a currently executing task.

        Sends a cancellation signal to the asyncio task. The execute
        method handles the CancelledError and updates status.

        Args:
            task_id: The ID of the running task to cancel.

        Returns:
            True if the task was found and cancellation was requested.
        """
        async_task = self._running_tasks.get(task_id)
        if async_task is None:
            logger.warning(
                "cancel_task_not_running",
                task_id=task_id,
            )
            return False

        async_task.cancel()
        logger.info("task_cancellation_requested", task_id=task_id)
        return True

    # -------------------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------------------

    async def get_metrics(self) -> Dict[str, Any]:
        """
        Return comprehensive execution metrics.

        Returns:
            Dictionary containing execution counts, error rates,
            latency histogram, and per-type latency summaries.
        """
        type_summaries: Dict[str, Dict[str, float]] = {}
        for task_type, latencies in self._latencies_by_type.items():
            if latencies:
                sorted_lat = sorted(latencies)
                count = len(sorted_lat)
                type_summaries[task_type] = {
                    "count": count,
                    "avg_ms": round(sum(sorted_lat) / count, 2),
                    "min_ms": round(sorted_lat[0], 2),
                    "max_ms": round(sorted_lat[-1], 2),
                    "p50_ms": round(sorted_lat[count // 2], 2),
                    "p95_ms": round(sorted_lat[int(count * 0.95)], 2),
                    "p99_ms": round(sorted_lat[int(count * 0.99)], 2),
                }

        total = self._execution_count + self._error_count + self._timeout_count
        avg_latency = (
            round(self._total_latency_ms / total, 2) if total > 0 else 0.0
        )

        return {
            "execution_count": self._execution_count,
            "error_count": self._error_count,
            "timeout_count": self._timeout_count,
            "running_count": len(self._running_tasks),
            "average_latency_ms": avg_latency,
            "latency_histogram": dict(self._latency_histogram),
            "type_summaries": type_summaries,
            "registered_handlers": list(self._handlers.keys()),
        }

    # -------------------------------------------------------------------------
    # Resource Tracking
    # -------------------------------------------------------------------------

    def _increment_node_tasks(self) -> None:
        """Increment the current_tasks counter on the local node."""
        try:
            local_node = self._cluster_manager.local_node
            if local_node is not None:
                local_node.current_tasks += 1
        except Exception:
            logger.debug("increment_node_tasks_error", exc_info=True)

    def _decrement_node_tasks(self) -> None:
        """Decrement the current_tasks counter on the local node."""
        try:
            local_node = self._cluster_manager.local_node
            if local_node is not None:
                local_node.current_tasks = max(0, local_node.current_tasks - 1)
        except Exception:
            logger.debug("decrement_node_tasks_error", exc_info=True)

    # -------------------------------------------------------------------------
    # Latency Recording
    # -------------------------------------------------------------------------

    def _record_latency(self, task_type: str, elapsed_ms: float) -> None:
        """
        Record an execution latency sample into the histogram and
        per-type latency lists.

        Args:
            task_type: The type of the executed task.
            elapsed_ms: The execution duration in milliseconds.
        """
        self._total_latency_ms += elapsed_ms
        self._latencies_by_type[task_type].append(elapsed_ms)

        # Trim per-type list to prevent unbounded growth (keep last 1000)
        if len(self._latencies_by_type[task_type]) > 1000:
            self._latencies_by_type[task_type] = self._latencies_by_type[task_type][-1000:]

        # Update histogram buckets
        recorded = False
        for bucket in LATENCY_BUCKETS:
            if elapsed_ms <= bucket:
                self._latency_histogram[f"le_{bucket}ms"] += 1
                recorded = True
                break
        if not recorded:
            self._latency_histogram["le_inf"] += 1

    # -------------------------------------------------------------------------
    # Built-in Handlers
    # -------------------------------------------------------------------------

    def _register_builtin_handlers(self) -> None:
        """Register handlers for all core AION task types."""
        self.register_handler(
            TaskType.TOOL_EXECUTION.value,
            self._handle_tool_execution,
        )
        self.register_handler(
            TaskType.MEMORY_OPERATION.value,
            self._handle_memory_operation,
        )
        self.register_handler(
            TaskType.AGENT_OPERATION.value,
            self._handle_agent_operation,
        )
        self.register_handler(
            TaskType.PLANNING_OPERATION.value,
            self._handle_planning_operation,
        )

    async def _handle_tool_execution(self, task: DistributedTask) -> Any:
        """
        Built-in handler for tool execution tasks.

        Expects payload keys:
            - tool_name (str): Name of the tool to invoke.
            - tool_args (dict): Arguments to pass to the tool.

        Returns:
            The tool execution result.
        """
        tool_name = task.payload.get("tool_name", "")
        tool_args = task.payload.get("tool_args", {})

        logger.info(
            "executing_tool",
            task_id=task.id,
            tool_name=tool_name,
        )

        # Delegate to the cluster manager's tool execution subsystem
        result = await self._cluster_manager.execute_tool(
            tool_name,
            **tool_args,
        )
        return result

    async def _handle_memory_operation(self, task: DistributedTask) -> Any:
        """
        Built-in handler for memory operation tasks.

        Expects payload keys:
            - operation (str): "read", "write", or "delete".
            - key (str): The memory key.
            - value (any, optional): Value for write operations.

        Returns:
            The memory operation result.
        """
        operation = task.payload.get("operation", "read")
        key = task.payload.get("key", "")
        value = task.payload.get("value")

        logger.info(
            "executing_memory_operation",
            task_id=task.id,
            operation=operation,
            key=key,
        )

        result = await self._cluster_manager.memory_operation(
            operation=operation,
            key=key,
            value=value,
        )
        return result

    async def _handle_agent_operation(self, task: DistributedTask) -> Any:
        """
        Built-in handler for agent operation tasks.

        Expects payload keys:
            - agent_id (str): ID of the agent to invoke.
            - action (str): The action to perform.
            - params (dict): Parameters for the action.

        Returns:
            The agent operation result.
        """
        agent_id = task.payload.get("agent_id", "")
        action = task.payload.get("action", "")
        params = task.payload.get("params", {})

        logger.info(
            "executing_agent_operation",
            task_id=task.id,
            agent_id=agent_id,
            action=action,
        )

        result = await self._cluster_manager.agent_operation(
            agent_id=agent_id,
            action=action,
            params=params,
        )
        return result

    async def _handle_planning_operation(self, task: DistributedTask) -> Any:
        """
        Built-in handler for planning operation tasks.

        Expects payload keys:
            - plan_id (str): ID of the plan.
            - operation (str): The planning operation to perform.
            - context (dict): Context for the planning operation.

        Returns:
            The planning operation result.
        """
        plan_id = task.payload.get("plan_id", "")
        operation = task.payload.get("operation", "")
        context = task.payload.get("context", {})

        logger.info(
            "executing_planning_operation",
            task_id=task.id,
            plan_id=plan_id,
            operation=operation,
        )

        result = await self._cluster_manager.planning_operation(
            plan_id=plan_id,
            operation=operation,
            context=context,
        )
        return result
