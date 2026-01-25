"""
AION Workflow Action Handler

Invoke sub-workflows from workflows.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, TYPE_CHECKING

import structlog

from aion.automation.types import ActionConfig, ExecutionStatus
from aion.automation.actions.executor import BaseActionHandler

if TYPE_CHECKING:
    from aion.automation.execution.context import ExecutionContext

logger = structlog.get_logger(__name__)


class WorkflowActionHandler(BaseActionHandler):
    """
    Handler for sub-workflow actions.

    Allows workflows to invoke other workflows.
    """

    async def execute(
        self,
        action: ActionConfig,
        context: "ExecutionContext",
    ) -> Any:
        """Execute a sub-workflow action."""
        sub_workflow_id = context.resolve(action.sub_workflow_id or "")
        if not sub_workflow_id:
            return {"error": "No sub_workflow_id specified", "success": False}

        # Resolve inputs
        inputs = {}
        if action.sub_workflow_inputs:
            inputs = self.resolve_params(action.sub_workflow_inputs, context)

        wait_for_completion = action.wait_for_completion

        logger.info(
            "invoking_sub_workflow",
            sub_workflow_id=sub_workflow_id,
            wait=wait_for_completion,
        )

        try:
            # Get workflow engine from context or import
            from aion.automation.engine import WorkflowEngine

            engine = WorkflowEngine()
            await engine.initialize()

            # Execute sub-workflow
            execution = await engine.execute(
                workflow_id=sub_workflow_id,
                inputs=inputs,
                initiated_by=f"workflow:{context.execution.workflow_id}",
            )

            if wait_for_completion:
                # Wait for completion
                result = await self._wait_for_completion(
                    engine,
                    execution.id,
                    timeout=action.timeout_seconds,
                )

                return {
                    "sub_workflow_id": sub_workflow_id,
                    "execution_id": execution.id,
                    "status": result["status"],
                    "outputs": result.get("outputs", {}),
                    "success": result["status"] == "completed",
                }
            else:
                # Return immediately
                return {
                    "sub_workflow_id": sub_workflow_id,
                    "execution_id": execution.id,
                    "status": "started",
                    "success": True,
                }

        except ImportError:
            logger.warning("workflow_engine_not_available")
            return {
                "sub_workflow_id": sub_workflow_id,
                "execution_id": f"sim-exec-{sub_workflow_id[:8]}",
                "status": "completed" if wait_for_completion else "started",
                "outputs": {},
                "success": True,
                "simulated": True,
            }

        except Exception as e:
            logger.error("sub_workflow_error", error=str(e))
            return {
                "sub_workflow_id": sub_workflow_id,
                "error": str(e),
                "success": False,
            }

    async def _wait_for_completion(
        self,
        engine: "WorkflowEngine",
        execution_id: str,
        timeout: float,
    ) -> Dict[str, Any]:
        """Wait for a sub-workflow to complete."""
        poll_interval = 1.0  # seconds
        elapsed = 0.0

        while elapsed < timeout:
            execution = engine.get_execution(execution_id)

            if not execution:
                return {
                    "status": "not_found",
                    "error": "Execution not found",
                }

            if execution.is_terminal():
                return {
                    "status": execution.status.value,
                    "outputs": execution.outputs,
                    "error": execution.error,
                }

            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        return {
            "status": "timeout",
            "error": f"Timeout after {timeout}s",
        }


class ParallelWorkflowHandler(BaseActionHandler):
    """
    Handler for parallel workflow execution.

    Executes multiple workflows or steps in parallel.
    """

    async def execute(
        self,
        action: ActionConfig,
        context: "ExecutionContext",
    ) -> Any:
        """Execute workflows in parallel."""
        parallel_configs = action.parallel_steps or []

        if not parallel_configs:
            return {"error": "No parallel configurations specified", "success": False}

        logger.info("executing_parallel", count=len(parallel_configs))

        try:
            from aion.automation.engine import WorkflowEngine

            engine = WorkflowEngine()
            await engine.initialize()

            # Start all executions
            tasks = []
            for config in parallel_configs:
                if isinstance(config, str):
                    # Assume it's a workflow ID
                    task = engine.execute(
                        workflow_id=config,
                        initiated_by=f"parallel:{context.execution.id}",
                    )
                elif isinstance(config, dict):
                    # Full configuration
                    task = engine.execute(
                        workflow_id=config.get("workflow_id", ""),
                        inputs=config.get("inputs", {}),
                        initiated_by=f"parallel:{context.execution.id}",
                    )
                else:
                    continue

                tasks.append(task)

            # Wait for all to start
            executions = await asyncio.gather(*tasks, return_exceptions=True)

            if action.parallel_wait_all:
                # Wait for all to complete
                results = []
                for execution in executions:
                    if isinstance(execution, Exception):
                        results.append({
                            "status": "error",
                            "error": str(execution),
                        })
                    else:
                        result = await self._wait_for_execution(
                            engine,
                            execution.id,
                            timeout=action.timeout_seconds,
                        )
                        results.append(result)

                return {
                    "parallel": True,
                    "count": len(results),
                    "results": results,
                    "all_success": all(r.get("status") == "completed" for r in results),
                    "success": True,
                }
            else:
                # Return immediately with execution IDs
                execution_ids = [
                    e.id if not isinstance(e, Exception) else str(e)
                    for e in executions
                ]

                return {
                    "parallel": True,
                    "count": len(execution_ids),
                    "execution_ids": execution_ids,
                    "success": True,
                }

        except ImportError:
            return {
                "parallel": True,
                "count": len(parallel_configs),
                "results": [{"status": "completed", "simulated": True}] * len(parallel_configs),
                "all_success": True,
                "success": True,
                "simulated": True,
            }

        except Exception as e:
            return {
                "parallel": True,
                "error": str(e),
                "success": False,
            }

    async def _wait_for_execution(
        self,
        engine: "WorkflowEngine",
        execution_id: str,
        timeout: float,
    ) -> Dict[str, Any]:
        """Wait for an execution to complete."""
        poll_interval = 1.0
        elapsed = 0.0

        while elapsed < timeout:
            execution = engine.get_execution(execution_id)

            if not execution:
                return {"status": "not_found"}

            if execution.is_terminal():
                return {
                    "execution_id": execution_id,
                    "status": execution.status.value,
                    "outputs": execution.outputs,
                }

            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        return {
            "execution_id": execution_id,
            "status": "timeout",
        }
