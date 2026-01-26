"""
AION Workflow Engine

Main execution engine for workflows.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import structlog

from aion.automation.types import (
    Workflow,
    WorkflowStep,
    WorkflowExecution,
    StepResult,
    ExecutionStatus,
    WorkflowStatus,
    ActionType,
    TriggerType,
)
from aion.automation.registry import WorkflowRegistry
from aion.automation.triggers.manager import TriggerManager
from aion.automation.actions.executor import ActionExecutor
from aion.automation.conditions.evaluator import ConditionEvaluator
from aion.automation.approval.manager import ApprovalManager
from aion.automation.execution.context import ExecutionContext
from aion.automation.execution.history import ExecutionHistoryManager

logger = structlog.get_logger(__name__)


class WorkflowEngine:
    """
    Main workflow execution engine.

    Features:
    - Workflow execution with step orchestration
    - Trigger management
    - Condition evaluation
    - Error handling and retries
    - State persistence
    - Approval gates
    - Sub-workflow support
    - Parallel execution
    """

    def __init__(
        self,
        registry: Optional[WorkflowRegistry] = None,
        trigger_manager: Optional[TriggerManager] = None,
        max_concurrent_executions: int = 100,
    ):
        self.registry = registry or WorkflowRegistry()
        self.max_concurrent = max_concurrent_executions

        # Components
        self.triggers = trigger_manager or TriggerManager(self)
        if trigger_manager:
            trigger_manager.set_engine(self)
        self.executor = ActionExecutor()
        self.evaluator = ConditionEvaluator()
        self.approvals = ApprovalManager()
        self.history = ExecutionHistoryManager()

        # Event bus reference (optional, for kernel integration)
        self._event_bus = None

        # Active executions
        self._executions: Dict[str, WorkflowExecution] = {}
        self._execution_tasks: Dict[str, asyncio.Task] = {}

        # Semaphore for concurrency control
        self._semaphore = asyncio.Semaphore(max_concurrent_executions)

        # Event callbacks
        self._on_execution_started: List[Callable] = []
        self._on_execution_completed: List[Callable] = []
        self._on_step_started: List[Callable] = []
        self._on_step_completed: List[Callable] = []

        self._initialized = False

    def set_event_bus(self, event_bus: Any) -> None:
        """Set the event bus for kernel integration."""
        self._event_bus = event_bus
        # Also connect to trigger manager
        if hasattr(self.triggers, 'set_event_bus'):
            self.triggers.set_event_bus(event_bus)

    async def initialize(self) -> None:
        """Initialize the workflow engine."""
        if self._initialized:
            return

        logger.info("Initializing Workflow Engine")

        await self.registry.initialize()
        await self.triggers.initialize()
        await self.executor.initialize()
        await self.approvals.initialize()
        await self.history.initialize()

        self._initialized = True
        logger.info("Workflow Engine initialized")

    async def shutdown(self) -> None:
        """Shutdown the workflow engine."""
        logger.info("Shutting down Workflow Engine")

        # Cancel active executions
        for task in self._execution_tasks.values():
            task.cancel()

        # Wait for tasks to complete
        if self._execution_tasks:
            await asyncio.gather(*self._execution_tasks.values(), return_exceptions=True)

        await self.triggers.shutdown()
        await self.approvals.shutdown()
        await self.history.shutdown()
        await self.registry.shutdown()

        self._initialized = False
        logger.info("Workflow Engine shutdown complete")

    # === Workflow Management ===

    async def register_workflow(self, workflow: Workflow) -> str:
        """Register a workflow."""
        # Validate
        errors = workflow.validate()
        if errors:
            raise ValueError(f"Invalid workflow: {errors}")

        # Save to registry
        await self.registry.save(workflow)

        # Register triggers
        for trigger_config in workflow.triggers:
            await self.triggers.register(workflow.id, trigger_config)

        logger.info(
            "workflow_registered",
            workflow_id=workflow.id,
            name=workflow.name,
            triggers=len(workflow.triggers),
            steps=len(workflow.steps),
        )

        return workflow.id

    async def get_workflow(self, workflow_id: str) -> Optional[Workflow]:
        """Get a workflow by ID."""
        return await self.registry.get(workflow_id)

    async def update_workflow(self, workflow: Workflow) -> str:
        """Update a workflow."""
        existing = await self.registry.get(workflow.id)
        if not existing:
            raise ValueError(f"Workflow not found: {workflow.id}")

        # Increment version
        workflow.version = existing.version + 1
        workflow.updated_at = datetime.now()

        # Re-register triggers
        await self.triggers.unregister_all(workflow.id)
        for trigger_config in workflow.triggers:
            await self.triggers.register(workflow.id, trigger_config)

        # Save
        await self.registry.save(workflow)

        logger.info(
            "workflow_updated",
            workflow_id=workflow.id,
            version=workflow.version,
        )

        return workflow.id

    async def delete_workflow(self, workflow_id: str) -> bool:
        """Delete a workflow."""
        # Remove triggers
        await self.triggers.unregister_all(workflow_id)

        # Delete from registry
        deleted = await self.registry.delete(workflow_id)

        if deleted:
            logger.info("workflow_deleted", workflow_id=workflow_id)

        return deleted

    async def activate_workflow(self, workflow_id: str) -> bool:
        """Activate a workflow."""
        workflow = await self.registry.get(workflow_id)
        if not workflow:
            return False

        workflow.status = WorkflowStatus.ACTIVE
        await self.registry.save(workflow)

        logger.info("workflow_activated", workflow_id=workflow_id)
        return True

    async def pause_workflow(self, workflow_id: str) -> bool:
        """Pause a workflow."""
        workflow = await self.registry.get(workflow_id)
        if not workflow:
            return False

        workflow.status = WorkflowStatus.PAUSED
        await self.registry.save(workflow)

        logger.info("workflow_paused", workflow_id=workflow_id)
        return True

    # === Execution ===

    async def execute(
        self,
        workflow_id: str,
        inputs: Dict[str, Any] = None,
        trigger_type: TriggerType = TriggerType.MANUAL,
        trigger_data: Dict[str, Any] = None,
        trigger_id: str = None,
        initiated_by: str = None,
    ) -> WorkflowExecution:
        """
        Execute a workflow.

        Args:
            workflow_id: Workflow to execute
            inputs: Input data
            trigger_type: What triggered the execution
            trigger_data: Data from the trigger
            trigger_id: ID of the trigger
            initiated_by: User or system ID

        Returns:
            Workflow execution record
        """
        workflow = await self.registry.get(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow not found: {workflow_id}")

        # Check workflow status
        if workflow.status not in (WorkflowStatus.ACTIVE, WorkflowStatus.DRAFT):
            raise ValueError(f"Workflow is {workflow.status.value}")

        # Check concurrent execution limit
        active_count = len([
            e for e in self._executions.values()
            if e.workflow_id == workflow_id and not e.is_terminal()
        ])
        if active_count >= workflow.max_concurrent_executions:
            raise ValueError("Max concurrent executions reached")

        # Merge default inputs
        final_inputs = {**(workflow.default_inputs or {}), **(inputs or {})}

        # Create execution record
        execution = WorkflowExecution(
            workflow_id=workflow_id,
            workflow_name=workflow.name,
            workflow_version=workflow.version,
            trigger_type=trigger_type,
            trigger_id=trigger_id,
            trigger_data=trigger_data or {},
            inputs=final_inputs,
            initiated_by=initiated_by or "system",
            tenant_id=workflow.tenant_id,
        )

        # Store execution
        self._executions[execution.id] = execution
        await self.registry.save_execution(execution)

        # Start execution task
        task = asyncio.create_task(
            self._run_workflow(workflow, execution)
        )
        self._execution_tasks[execution.id] = task

        logger.info(
            "execution_started",
            execution_id=execution.id,
            workflow_id=workflow_id,
            trigger_type=trigger_type.value,
        )

        # Fire callbacks
        await self._fire_callbacks(self._on_execution_started, execution)

        return execution

    async def _run_workflow(
        self,
        workflow: Workflow,
        execution: WorkflowExecution,
    ) -> None:
        """Run a workflow execution."""
        async with self._semaphore:
            execution.start()
            await self.registry.save_execution(execution)

            # Create execution context
            context = ExecutionContext(
                execution=execution,
                workflow=workflow,
            )

            try:
                # Get entry step
                entry_step = workflow.get_entry_step()
                if not entry_step:
                    raise ValueError("No entry step defined")

                # Execute steps
                next_step_id = entry_step.id

                while next_step_id:
                    step = workflow.get_step(next_step_id)
                    if not step:
                        raise ValueError(f"Step not found: {next_step_id}")

                    execution.current_step_id = next_step_id
                    execution.current_step_name = step.name
                    await self.registry.save_execution(execution)

                    # Execute step
                    result, next_step_id = await self._execute_step(
                        step, context, execution, workflow
                    )

                    # Store result
                    execution.step_results[step.id] = result

                    # Check for failure
                    if result.status == ExecutionStatus.FAILED:
                        if not step.continue_on_error:
                            if step.error_handler_step:
                                next_step_id = step.error_handler_step
                            else:
                                raise Exception(result.error)

                    # Check workflow timeout
                    if execution.started_at:
                        elapsed = (datetime.now() - execution.started_at).total_seconds()
                        if elapsed > workflow.timeout_seconds:
                            raise TimeoutError("Workflow timeout exceeded")

                # Workflow completed successfully
                execution.complete(context.get_outputs())

                # Update workflow stats
                workflow.execution_count += 1
                workflow.success_count += 1
                await self.registry.save(workflow)

            except asyncio.CancelledError:
                execution.cancel()
                raise

            except TimeoutError as e:
                execution.status = ExecutionStatus.TIMED_OUT
                execution.error = str(e)
                execution.completed_at = datetime.now()

            except Exception as e:
                logger.error(
                    "workflow_execution_failed",
                    execution_id=execution.id,
                    error=str(e),
                )
                execution.fail(str(e), execution.current_step_id)

                # Update workflow stats
                workflow.execution_count += 1
                workflow.failure_count += 1
                await self.registry.save(workflow)

            finally:
                # Save final state
                await self.registry.save_execution(execution)

                # Record in history
                await self.history.record(execution)

                # Cleanup
                self._execution_tasks.pop(execution.id, None)

                # Fire callbacks
                await self._fire_callbacks(self._on_execution_completed, execution)

                logger.info(
                    "execution_completed",
                    execution_id=execution.id,
                    status=execution.status.value,
                    duration_ms=execution.duration_ms,
                )

    async def _execute_step(
        self,
        step: WorkflowStep,
        context: ExecutionContext,
        execution: WorkflowExecution,
        workflow: Workflow,
    ) -> tuple[StepResult, Optional[str]]:
        """Execute a single workflow step."""
        context.current_step_id = step.id
        result = StepResult(step_id=step.id, step_name=step.name)
        next_step_id = None

        logger.debug(
            "executing_step",
            step_id=step.id,
            step_name=step.name,
            action_type=step.action.action_type.value,
        )

        # Fire step started callback
        await self._fire_callbacks(self._on_step_started, execution, step)

        try:
            # Check condition
            if step.condition:
                condition_met = await self.evaluator.evaluate(
                    step.condition, context
                )
                if not condition_met:
                    result.complete({"skipped": True, "reason": "condition_not_met"})
                    next_step_id = step.on_success
                    return result, next_step_id

            # Handle loop
            if step.loop_over:
                result, next_step_id = await self._execute_loop(
                    step, context, execution
                )
            # Handle approval action specially
            elif step.action.action_type == ActionType.APPROVAL:
                result, next_step_id = await self._execute_approval_step(
                    step, context, execution
                )
            else:
                # Execute action
                output = await self.executor.execute(step.action, context)
                result.complete(output)

                # Store output in context
                context.set_step_output(step.id, output)

                # Store in variable if specified
                if step.output_variable:
                    context.set(step.output_variable, output)

                # Determine next step
                next_step_id = step.on_success

        except Exception as e:
            result.fail(str(e), type(e).__name__)

            # Retry logic
            if step.action.retry_count > 0 and result.attempt <= step.action.retry_count:
                await asyncio.sleep(
                    step.action.retry_delay_seconds *
                    (step.action.retry_backoff_multiplier ** (result.attempt - 1))
                )
                result.attempt += 1
                return await self._execute_step(step, context, execution, workflow)

            next_step_id = step.on_failure

        # Fire step completed callback
        await self._fire_callbacks(self._on_step_completed, execution, step, result)

        return result, next_step_id

    async def _execute_loop(
        self,
        step: WorkflowStep,
        context: ExecutionContext,
        execution: WorkflowExecution,
    ) -> tuple[StepResult, Optional[str]]:
        """Execute a loop step."""
        result = StepResult(step_id=step.id, step_name=step.name)

        # Resolve loop items
        items = context.resolve(step.loop_over)
        if not isinstance(items, (list, tuple)):
            items = [items] if items is not None else []

        # Enforce iteration limit
        items = items[:step.max_iterations]

        loop_results = []
        context.enter_loop(step.loop_variable, step.loop_index_variable)

        try:
            for index, item in enumerate(items):
                context.set_loop_item(step.loop_variable, index, item)

                # Execute action for this item
                item_output = await self.executor.execute(step.action, context)
                loop_results.append({
                    "index": index,
                    "item": item,
                    "output": item_output,
                })

        finally:
            context.exit_loop()

        result.complete(loop_results)
        context.set_step_output(step.id, loop_results)

        return result, step.on_success

    async def _execute_approval_step(
        self,
        step: WorkflowStep,
        context: ExecutionContext,
        execution: WorkflowExecution,
    ) -> tuple[StepResult, Optional[str]]:
        """Execute an approval step."""
        result = StepResult(step_id=step.id, step_name=step.name)
        action = step.action

        # Check for auto-approve
        if action.auto_approve:
            result.complete({"approved": True, "auto": True})
            return result, step.on_success

        # Create approval request
        message = context.resolve(action.approval_message or "Approval required")
        title = context.resolve(action.approval_title or "")
        approvers = action.approvers or []

        request = await self.approvals.create_request(
            execution_id=execution.id,
            step_id=step.id,
            workflow_id=execution.workflow_id,
            workflow_name=execution.workflow_name,
            title=title,
            message=message,
            approvers=approvers,
            timeout_hours=action.approval_timeout_hours,
            requires_all=action.require_all_approvers,
            details=context.to_safe_dict(),
        )

        # Mark execution as waiting
        execution.status = ExecutionStatus.WAITING
        await self.registry.save_execution(execution)

        # Wait for decision
        decision = await self.approvals.wait_for_decision(
            request.id,
            timeout_hours=action.approval_timeout_hours,
        )

        # Resume execution
        execution.status = ExecutionStatus.RUNNING

        if decision.approved:
            result.complete({
                "approved": True,
                "approved_by": decision.approved_by,
                "message": decision.message,
            })
            return result, step.on_success
        else:
            result.fail(f"Approval rejected: {decision.message}")
            return result, step.on_failure

    # === Execution Management ===

    def get_execution(self, execution_id: str) -> Optional[WorkflowExecution]:
        """Get an execution by ID."""
        return self._executions.get(execution_id)

    async def cancel_execution(self, execution_id: str) -> bool:
        """Cancel an execution."""
        task = self._execution_tasks.get(execution_id)
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            return True
        return False

    def list_executions(
        self,
        workflow_id: str = None,
        status: ExecutionStatus = None,
        limit: int = 100,
    ) -> List[WorkflowExecution]:
        """List executions with filters."""
        executions = list(self._executions.values())

        if workflow_id:
            executions = [e for e in executions if e.workflow_id == workflow_id]

        if status:
            executions = [e for e in executions if e.status == status]

        # Sort by start time descending
        executions.sort(
            key=lambda e: e.started_at or datetime.min,
            reverse=True,
        )

        return executions[:limit]

    # === Event Callbacks ===

    def on_execution_started(self, callback: Callable) -> None:
        """Register callback for execution start."""
        self._on_execution_started.append(callback)

    def on_execution_completed(self, callback: Callable) -> None:
        """Register callback for execution completion."""
        self._on_execution_completed.append(callback)

    def on_step_started(self, callback: Callable) -> None:
        """Register callback for step start."""
        self._on_step_started.append(callback)

    def on_step_completed(self, callback: Callable) -> None:
        """Register callback for step completion."""
        self._on_step_completed.append(callback)

    async def _fire_callbacks(
        self,
        callbacks: List[Callable],
        *args,
    ) -> None:
        """Fire callbacks."""
        for callback in callbacks:
            try:
                result = callback(*args)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error("callback_error", error=str(e))

    # === Statistics ===

    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        status_counts = {}
        for status in ExecutionStatus:
            status_counts[status.value] = len([
                e for e in self._executions.values()
                if e.status == status
            ])

        return {
            "active_executions": len(self._execution_tasks),
            "total_executions": len(self._executions),
            "by_status": status_counts,
            "max_concurrent": self.max_concurrent,
            "triggers": self.triggers.get_stats(),
            "approvals": self.approvals.get_stats(),
        }
