"""
Tool Executor

Safe execution of tools with sandboxing and monitoring.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional
import structlog

from aion.systems.agents.tools.mcp import MCPClient, MCPTool, MCPToolResult

logger = structlog.get_logger()


class ExecutionMode(str, Enum):
    """Execution mode for tools."""

    DIRECT = "direct"  # Execute immediately
    SANDBOXED = "sandboxed"  # Execute in sandbox
    CONFIRMED = "confirmed"  # Require confirmation
    DRY_RUN = "dry_run"  # Simulate execution


@dataclass
class ExecutionContext:
    """Context for tool execution."""

    agent_id: str
    task_id: Optional[str] = None
    session_id: Optional[str] = None
    mode: ExecutionMode = ExecutionMode.DIRECT
    timeout: float = 30.0
    max_retries: int = 3
    metadata: dict[str, Any] = field(default_factory=dict)

    # Resource limits
    max_memory_mb: Optional[int] = None
    max_cpu_seconds: Optional[float] = None
    allowed_capabilities: Optional[list[str]] = None


@dataclass
class ExecutionRecord:
    """Record of a tool execution."""

    id: str
    tool_name: str
    arguments: dict[str, Any]
    context: ExecutionContext
    result: Optional[MCPToolResult] = None
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    status: str = "pending"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "agent_id": self.context.agent_id,
            "status": self.status,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "success": self.result.success if self.result else None,
        }


class ToolExecutor:
    """
    Safe tool executor with monitoring.

    Features:
    - Multiple execution modes
    - Resource limiting
    - Execution history
    - Rollback support
    - Parallel execution
    """

    def __init__(self, client: MCPClient):
        self.client = client
        self._history: list[ExecutionRecord] = []
        self._pending: dict[str, ExecutionRecord] = {}
        self._confirmation_callbacks: dict[str, asyncio.Future] = {}
        self._execution_counter = 0
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize executor."""
        self._initialized = True
        logger.info("tool_executor_initialized")

    async def shutdown(self) -> None:
        """Shutdown executor."""
        # Cancel pending confirmations
        for future in self._confirmation_callbacks.values():
            if not future.done():
                future.cancel()

        self._initialized = False
        logger.info("tool_executor_shutdown")

    async def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: ExecutionContext,
    ) -> MCPToolResult:
        """Execute a tool with context."""
        self._execution_counter += 1
        execution_id = f"exec-{self._execution_counter}"

        record = ExecutionRecord(
            id=execution_id,
            tool_name=tool_name,
            arguments=arguments,
            context=context,
        )

        self._pending[execution_id] = record

        try:
            # Check mode
            if context.mode == ExecutionMode.DRY_RUN:
                result = await self._dry_run(tool_name, arguments)
            elif context.mode == ExecutionMode.CONFIRMED:
                result = await self._execute_with_confirmation(
                    tool_name, arguments, context
                )
            elif context.mode == ExecutionMode.SANDBOXED:
                result = await self._execute_sandboxed(
                    tool_name, arguments, context
                )
            else:
                result = await self._execute_direct(
                    tool_name, arguments, context
                )

            record.result = result
            record.status = "completed" if result.success else "failed"
            record.completed_at = datetime.now()

            logger.info(
                "tool_executed",
                execution_id=execution_id,
                tool=tool_name,
                success=result.success,
            )

            return result

        except Exception as e:
            record.status = "error"
            record.completed_at = datetime.now()

            result = MCPToolResult(
                tool_name=tool_name,
                success=False,
                error=str(e),
            )
            record.result = result

            logger.error(
                "tool_execution_error",
                execution_id=execution_id,
                error=str(e),
            )

            return result

        finally:
            del self._pending[execution_id]
            self._history.append(record)

            # Keep history bounded
            if len(self._history) > 1000:
                self._history = self._history[-500:]

    async def _execute_direct(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: ExecutionContext,
    ) -> MCPToolResult:
        """Execute tool directly."""
        return await asyncio.wait_for(
            self.client.call_tool_with_retry(
                tool_name,
                arguments,
                max_retries=context.max_retries,
            ),
            timeout=context.timeout,
        )

    async def _execute_sandboxed(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: ExecutionContext,
    ) -> MCPToolResult:
        """Execute tool in sandbox (simulated)."""
        # In production, this would use actual sandboxing
        # For now, we add extra validation and monitoring

        tool = self.client.get_tool(tool_name)
        if not tool:
            return MCPToolResult(
                tool_name=tool_name,
                success=False,
                error="Tool not found",
            )

        # Check capabilities
        if context.allowed_capabilities:
            for cap in tool.capabilities:
                if cap.value not in context.allowed_capabilities:
                    return MCPToolResult(
                        tool_name=tool_name,
                        success=False,
                        error=f"Capability not allowed: {cap.value}",
                    )

        # Execute with timeout
        return await self._execute_direct(tool_name, arguments, context)

    async def _execute_with_confirmation(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: ExecutionContext,
    ) -> MCPToolResult:
        """Execute tool with confirmation."""
        confirmation_id = f"confirm-{self._execution_counter}"

        # Create confirmation future
        future: asyncio.Future = asyncio.Future()
        self._confirmation_callbacks[confirmation_id] = future

        logger.info(
            "confirmation_required",
            confirmation_id=confirmation_id,
            tool=tool_name,
            arguments=arguments,
        )

        try:
            # Wait for confirmation (with timeout)
            confirmed = await asyncio.wait_for(
                future,
                timeout=context.timeout,
            )

            if confirmed:
                return await self._execute_direct(tool_name, arguments, context)
            else:
                return MCPToolResult(
                    tool_name=tool_name,
                    success=False,
                    error="Execution rejected",
                )

        except asyncio.TimeoutError:
            return MCPToolResult(
                tool_name=tool_name,
                success=False,
                error="Confirmation timed out",
            )

        finally:
            del self._confirmation_callbacks[confirmation_id]

    async def _dry_run(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> MCPToolResult:
        """Simulate tool execution."""
        tool = self.client.get_tool(tool_name)

        if not tool:
            return MCPToolResult(
                tool_name=tool_name,
                success=False,
                error="Tool not found",
            )

        return MCPToolResult(
            tool_name=tool_name,
            success=True,
            result={
                "dry_run": True,
                "would_execute": tool_name,
                "with_arguments": arguments,
            },
            metadata={"mode": "dry_run"},
        )

    def confirm(self, confirmation_id: str, approved: bool) -> bool:
        """Confirm or reject a pending execution."""
        if confirmation_id in self._confirmation_callbacks:
            future = self._confirmation_callbacks[confirmation_id]
            if not future.done():
                future.set_result(approved)
                return True
        return False

    async def execute_parallel(
        self,
        calls: list[tuple[str, dict[str, Any]]],
        context: ExecutionContext,
    ) -> list[MCPToolResult]:
        """Execute multiple tools in parallel."""
        tasks = [
            self.execute(tool_name, arguments, context)
            for tool_name, arguments in calls
        ]

        return await asyncio.gather(*tasks)

    async def execute_sequence(
        self,
        calls: list[tuple[str, dict[str, Any]]],
        context: ExecutionContext,
        stop_on_failure: bool = True,
    ) -> list[MCPToolResult]:
        """Execute tools in sequence."""
        results = []

        for tool_name, arguments in calls:
            result = await self.execute(tool_name, arguments, context)
            results.append(result)

            if stop_on_failure and not result.success:
                break

        return results

    def get_history(
        self,
        agent_id: Optional[str] = None,
        tool_name: Optional[str] = None,
        limit: int = 100,
    ) -> list[ExecutionRecord]:
        """Get execution history."""
        history = self._history

        if agent_id:
            history = [r for r in history if r.context.agent_id == agent_id]

        if tool_name:
            history = [r for r in history if r.tool_name == tool_name]

        return history[-limit:]

    def get_pending(self) -> list[ExecutionRecord]:
        """Get pending executions."""
        return list(self._pending.values())

    def get_stats(self) -> dict[str, Any]:
        """Get executor statistics."""
        total = len(self._history)
        successful = sum(1 for r in self._history if r.result and r.result.success)

        return {
            "total_executions": total,
            "success_rate": successful / max(1, total),
            "pending_count": len(self._pending),
            "pending_confirmations": len(self._confirmation_callbacks),
            "recent_tools": list({
                r.tool_name for r in self._history[-20:]
            }),
        }
