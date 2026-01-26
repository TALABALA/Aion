"""
AION Workflow Automation API Routes

FastAPI routes for workflow automation system.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request, Body, Query
from pydantic import BaseModel, Field

import structlog

from aion.automation.types import (
    Workflow,
    WorkflowStep,
    TriggerConfig,
    ActionConfig,
    WorkflowStatus,
    ExecutionStatus,
    TriggerType,
    ActionType,
    ApprovalStatus,
)
from aion.automation.engine import WorkflowEngine
from aion.automation.templates.manager import TemplateManager

logger = structlog.get_logger(__name__)


# === Request/Response Models ===


class TriggerConfigRequest(BaseModel):
    """Trigger configuration request."""
    type: str = Field(..., description="Trigger type")
    cron_expression: Optional[str] = None
    webhook_path: Optional[str] = None
    event_type: Optional[str] = None
    event_source: Optional[str] = None
    data_source: Optional[str] = None
    enabled: bool = True


class ActionConfigRequest(BaseModel):
    """Action configuration request."""
    type: str = Field(..., description="Action type")
    tool_name: Optional[str] = None
    tool_params: Optional[Dict[str, Any]] = None
    webhook_url: Optional[str] = None
    webhook_method: str = "POST"
    agent_operation: Optional[str] = None
    agent_role: Optional[str] = None
    goal_operation: Optional[str] = None
    goal_title: Optional[str] = None
    notification_channel: Optional[str] = None
    notification_message: Optional[str] = None
    llm_prompt: Optional[str] = None
    delay_seconds: Optional[float] = None
    transform_expression: Optional[str] = None
    timeout_seconds: float = 300.0


class StepRequest(BaseModel):
    """Workflow step request."""
    id: Optional[str] = None
    name: str = Field(..., description="Step name")
    description: str = ""
    action: ActionConfigRequest
    on_success: Optional[str] = None
    on_failure: Optional[str] = None
    condition: Optional[Dict[str, Any]] = None
    loop_over: Optional[str] = None


class CreateWorkflowRequest(BaseModel):
    """Create workflow request."""
    name: str = Field(..., description="Workflow name")
    description: str = ""
    triggers: List[TriggerConfigRequest] = Field(default_factory=list)
    steps: List[StepRequest] = Field(default_factory=list)
    entry_step_id: Optional[str] = None
    input_schema: Optional[Dict[str, Any]] = None
    timeout_seconds: float = 3600.0


class UpdateWorkflowRequest(BaseModel):
    """Update workflow request."""
    name: Optional[str] = None
    description: Optional[str] = None
    triggers: Optional[List[TriggerConfigRequest]] = None
    steps: Optional[List[StepRequest]] = None
    status: Optional[str] = None


class ExecuteWorkflowRequest(BaseModel):
    """Execute workflow request."""
    inputs: Dict[str, Any] = Field(default_factory=dict)
    initiated_by: Optional[str] = None


class ApprovalDecisionRequest(BaseModel):
    """Approval decision request."""
    decision: str = Field(..., description="'approve' or 'reject'")
    message: str = ""
    approver: str = Field(..., description="Approver identifier")


class InstantiateTemplateRequest(BaseModel):
    """Instantiate template request."""
    name: str = Field(..., description="Name for new workflow")
    parameters: Dict[str, Any] = Field(default_factory=dict)
    owner_id: str = ""


# === Route Setup ===


def setup_automation_routes(app, engine: WorkflowEngine) -> None:
    """
    Setup workflow automation routes.

    Args:
        app: FastAPI application
        engine: Workflow engine instance
    """
    router = APIRouter(prefix="/automation", tags=["Automation"])
    template_manager = TemplateManager()

    # Initialize template manager on startup
    @app.on_event("startup")
    async def init_templates():
        await template_manager.initialize()

    # === Workflow Routes ===

    @router.post("/workflows", response_model=Dict[str, Any])
    async def create_workflow(request: CreateWorkflowRequest):
        """Create a new workflow."""
        try:
            # Convert request to Workflow
            workflow = _request_to_workflow(request)

            # Register workflow
            workflow_id = await engine.register_workflow(workflow)

            return {
                "workflow_id": workflow_id,
                "name": workflow.name,
                "status": workflow.status.value,
            }

        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error("create_workflow_error", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/workflows", response_model=Dict[str, Any])
    async def list_workflows(
        status: Optional[str] = None,
        tag: Optional[str] = None,
        search: Optional[str] = None,
        limit: int = Query(default=100, le=1000),
        offset: int = 0,
    ):
        """List all workflows."""
        try:
            status_filter = WorkflowStatus(status) if status else None
            workflows = await engine.registry.list(
                status=status_filter,
                tag=tag,
                search=search,
                limit=limit,
                offset=offset,
            )

            return {
                "workflows": [w.to_dict() for w in workflows],
                "count": len(workflows),
            }

        except Exception as e:
            logger.error("list_workflows_error", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/workflows/{workflow_id}", response_model=Dict[str, Any])
    async def get_workflow(workflow_id: str):
        """Get a workflow by ID."""
        workflow = await engine.get_workflow(workflow_id)
        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")
        return workflow.to_dict()

    @router.put("/workflows/{workflow_id}", response_model=Dict[str, Any])
    async def update_workflow(workflow_id: str, request: UpdateWorkflowRequest):
        """Update a workflow."""
        try:
            workflow = await engine.get_workflow(workflow_id)
            if not workflow:
                raise HTTPException(status_code=404, detail="Workflow not found")

            # Apply updates
            if request.name is not None:
                workflow.name = request.name
            if request.description is not None:
                workflow.description = request.description
            if request.status is not None:
                workflow.status = WorkflowStatus(request.status)
            if request.triggers is not None:
                workflow.triggers = [_request_to_trigger(t) for t in request.triggers]
            if request.steps is not None:
                workflow.steps = [_request_to_step(s) for s in request.steps]

            await engine.update_workflow(workflow)

            return {
                "workflow_id": workflow.id,
                "version": workflow.version,
            }

        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @router.delete("/workflows/{workflow_id}")
    async def delete_workflow(workflow_id: str):
        """Delete a workflow."""
        deleted = await engine.delete_workflow(workflow_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Workflow not found")
        return {"deleted": True}

    @router.post("/workflows/{workflow_id}/activate")
    async def activate_workflow(workflow_id: str):
        """Activate a workflow."""
        success = await engine.activate_workflow(workflow_id)
        if not success:
            raise HTTPException(status_code=404, detail="Workflow not found")
        return {"activated": True}

    @router.post("/workflows/{workflow_id}/pause")
    async def pause_workflow(workflow_id: str):
        """Pause a workflow."""
        success = await engine.pause_workflow(workflow_id)
        if not success:
            raise HTTPException(status_code=404, detail="Workflow not found")
        return {"paused": True}

    # === Execution Routes ===

    @router.post("/workflows/{workflow_id}/execute", response_model=Dict[str, Any])
    async def execute_workflow(workflow_id: str, request: ExecuteWorkflowRequest):
        """Execute a workflow."""
        try:
            execution = await engine.execute(
                workflow_id=workflow_id,
                inputs=request.inputs,
                initiated_by=request.initiated_by,
            )

            return execution.to_dict()

        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error("execute_workflow_error", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/executions", response_model=Dict[str, Any])
    async def list_executions(
        workflow_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = Query(default=100, le=1000),
    ):
        """List workflow executions."""
        status_filter = ExecutionStatus(status) if status else None
        executions = engine.list_executions(
            workflow_id=workflow_id,
            status=status_filter,
            limit=limit,
        )

        return {
            "executions": [e.to_dict() for e in executions],
            "count": len(executions),
        }

    @router.get("/executions/{execution_id}", response_model=Dict[str, Any])
    async def get_execution(execution_id: str):
        """Get an execution by ID."""
        execution = engine.get_execution(execution_id)
        if not execution:
            raise HTTPException(status_code=404, detail="Execution not found")
        return execution.to_dict()

    @router.post("/executions/{execution_id}/cancel")
    async def cancel_execution(execution_id: str):
        """Cancel an execution."""
        success = await engine.cancel_execution(execution_id)
        if not success:
            raise HTTPException(status_code=404, detail="Execution not found or already completed")
        return {"cancelled": True}

    # === Webhook Routes ===

    @router.post("/webhooks/{path:path}")
    async def handle_webhook(path: str, request: Request):
        """Handle incoming webhooks."""
        try:
            body = await request.json()
        except Exception:
            body = {}

        headers = dict(request.headers)
        method = request.method

        execution_id = await engine.triggers.handle_webhook(
            f"/webhooks/{path}",
            body,
            headers,
            method,
        )

        if execution_id:
            return {"execution_id": execution_id}

        raise HTTPException(status_code=404, detail="No matching webhook")

    # === Approval Routes ===

    @router.get("/approvals", response_model=Dict[str, Any])
    async def list_approvals(
        status: Optional[str] = None,
        approver: Optional[str] = None,
        limit: int = Query(default=100, le=1000),
    ):
        """List approval requests."""
        status_filter = ApprovalStatus(status) if status else None
        requests = await engine.approvals.list_requests(
            status=status_filter,
            approver=approver,
            limit=limit,
        )

        return {
            "approvals": [r.to_dict() for r in requests],
            "count": len(requests),
        }

    @router.get("/approvals/{request_id}", response_model=Dict[str, Any])
    async def get_approval(request_id: str):
        """Get an approval request."""
        request = await engine.approvals.get_request(request_id)
        if not request:
            raise HTTPException(status_code=404, detail="Approval request not found")
        return request.to_dict()

    @router.post("/approvals/{request_id}/decide")
    async def decide_approval(request_id: str, decision: ApprovalDecisionRequest):
        """Approve or reject a request."""
        if decision.decision == "approve":
            success = await engine.approvals.approve(
                request_id,
                decision.approver,
                decision.message,
            )
        elif decision.decision == "reject":
            success = await engine.approvals.reject(
                request_id,
                decision.approver,
                decision.message,
            )
        else:
            raise HTTPException(status_code=400, detail="Invalid decision")

        if not success:
            raise HTTPException(status_code=400, detail="Could not process decision")

        return {"success": True}

    # === Template Routes ===

    @router.get("/templates", response_model=Dict[str, Any])
    async def list_templates(
        category: Optional[str] = None,
        search: Optional[str] = None,
    ):
        """List workflow templates."""
        templates = await template_manager.list(
            category=category,
            search=search,
        )

        return {
            "templates": [
                {
                    "id": t.id,
                    "name": t.name,
                    "description": t.description,
                    "category": t.category,
                    "tags": t.tags,
                    "usage_count": t.usage_count,
                }
                for t in templates
            ],
            "count": len(templates),
        }

    @router.get("/templates/categories")
    async def list_template_categories():
        """List template categories."""
        categories = await template_manager.list_categories()
        return {"categories": categories}

    @router.get("/templates/{template_id}", response_model=Dict[str, Any])
    async def get_template(template_id: str):
        """Get a template by ID."""
        template = await template_manager.get(template_id)
        if not template:
            raise HTTPException(status_code=404, detail="Template not found")
        return template.to_dict()

    @router.post("/templates/{template_id}/preview", response_model=Dict[str, Any])
    async def preview_template(
        template_id: str,
        parameters: Dict[str, Any] = Body(default={}),
    ):
        """Preview a template with parameters."""
        try:
            preview = await template_manager.preview(template_id, parameters)
            return preview
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))

    @router.post("/templates/{template_id}/instantiate", response_model=Dict[str, Any])
    async def instantiate_template(
        template_id: str,
        request: InstantiateTemplateRequest,
    ):
        """Create a workflow from a template."""
        try:
            workflow = await template_manager.instantiate(
                template_id,
                request.name,
                request.parameters,
                request.owner_id,
            )

            # Register the workflow
            workflow_id = await engine.register_workflow(workflow)

            return {
                "workflow_id": workflow_id,
                "name": workflow.name,
                "template_id": template_id,
            }

        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    # === Event Routes ===

    @router.post("/events")
    async def emit_event(
        event_type: str = Body(...),
        data: Dict[str, Any] = Body(default={}),
        source: str = Body(default="api"),
    ):
        """Emit an event to trigger workflows."""
        from aion.automation.triggers.event import EventTriggerHandler

        handler = engine.triggers._handlers.get(TriggerType.EVENT)
        if isinstance(handler, EventTriggerHandler):
            execution_ids = await handler.emit(
                event_type=event_type,
                data=data,
                source=source,
            )
        else:
            execution_ids = await engine.triggers.handle_event(
                event_type=event_type,
                event_data=data,
                source=source,
            )

        return {
            "event_type": event_type,
            "triggered": len(execution_ids),
            "execution_ids": execution_ids,
        }

    # === Statistics Routes ===

    @router.get("/stats")
    async def get_stats():
        """Get automation system statistics."""
        return engine.get_stats()

    @router.get("/history")
    async def get_history(
        workflow_id: Optional[str] = None,
        days: int = 30,
    ):
        """Get execution history statistics."""
        stats = await engine.history.get_daily_stats(
            days=days,
            workflow_id=workflow_id,
        )
        return {"history": stats}

    # Register router
    app.include_router(router)


# === Helper Functions ===


def _request_to_workflow(request: CreateWorkflowRequest) -> Workflow:
    """Convert request to Workflow object."""
    workflow = Workflow(
        name=request.name,
        description=request.description,
        timeout_seconds=request.timeout_seconds,
        input_schema=request.input_schema or {},
    )

    # Convert triggers
    for trigger_req in request.triggers:
        workflow.triggers.append(_request_to_trigger(trigger_req))

    # Convert steps
    for step_req in request.steps:
        workflow.steps.append(_request_to_step(step_req))

    # Set entry step
    if request.entry_step_id:
        workflow.entry_step_id = request.entry_step_id
    elif workflow.steps:
        workflow.entry_step_id = workflow.steps[0].id

    return workflow


def _request_to_trigger(request: TriggerConfigRequest) -> TriggerConfig:
    """Convert request to TriggerConfig."""
    return TriggerConfig(
        trigger_type=TriggerType(request.type),
        cron_expression=request.cron_expression,
        webhook_path=request.webhook_path,
        event_type=request.event_type,
        event_source=request.event_source,
        data_source=request.data_source,
        enabled=request.enabled,
    )


def _request_to_step(request: StepRequest) -> WorkflowStep:
    """Convert request to WorkflowStep."""
    from aion.automation.types import Condition

    step = WorkflowStep(
        name=request.name,
        description=request.description,
        action=_request_to_action(request.action),
        on_success=request.on_success,
        on_failure=request.on_failure,
        loop_over=request.loop_over,
    )

    if request.id:
        step.id = request.id

    if request.condition:
        step.condition = Condition.from_dict(request.condition)

    return step


def _request_to_action(request: ActionConfigRequest) -> ActionConfig:
    """Convert request to ActionConfig."""
    return ActionConfig(
        action_type=ActionType(request.type),
        tool_name=request.tool_name,
        tool_params=request.tool_params,
        webhook_url=request.webhook_url,
        webhook_method=request.webhook_method,
        agent_operation=request.agent_operation,
        agent_role=request.agent_role,
        goal_operation=request.goal_operation,
        goal_title=request.goal_title,
        notification_channel=request.notification_channel,
        notification_message=request.notification_message,
        llm_prompt=request.llm_prompt,
        delay_seconds=request.delay_seconds,
        transform_expression=request.transform_expression,
        timeout_seconds=request.timeout_seconds,
    )
