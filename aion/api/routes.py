"""
AION API Routes

FastAPI routes for all AION subsystems.
"""

from __future__ import annotations

from typing import Any, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, UploadFile, File, Query, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import structlog

logger = structlog.get_logger(__name__)


# ==================== Request/Response Models ====================

class ProcessRequest(BaseModel):
    """Request to process a user query."""
    query: str = Field(..., description="The user's query or request")
    context: Optional[dict] = Field(default=None, description="Additional context")
    user_id: Optional[str] = Field(default=None, description="User identifier")
    session_id: Optional[str] = Field(default=None, description="Session identifier")


class ProcessResponse(BaseModel):
    """Response from processing a query."""
    request_id: str
    success: bool
    result: Any
    error: Optional[str] = None
    execution_time_ms: float


class MemoryStoreRequest(BaseModel):
    """Request to store a memory."""
    content: str = Field(..., description="Memory content")
    memory_type: str = Field(default="episodic", description="Type of memory")
    importance: float = Field(default=0.5, ge=0, le=1)
    metadata: Optional[dict] = None


class MemorySearchRequest(BaseModel):
    """Request to search memories."""
    query: str = Field(..., description="Search query")
    limit: int = Field(default=10, ge=1, le=100)
    memory_type: Optional[str] = None
    min_importance: float = Field(default=0.0, ge=0, le=1)


class PlanCreateRequest(BaseModel):
    """Request to create an execution plan."""
    name: str
    description: str = ""
    steps: list[dict] = Field(default_factory=list)


class ToolExecuteRequest(BaseModel):
    """Request to execute a tool."""
    tool_name: str
    params: dict = Field(default_factory=dict)
    timeout: Optional[float] = None


class VisionAnalyzeRequest(BaseModel):
    """Request to analyze an image."""
    image_url: Optional[str] = None
    question: Optional[str] = None
    store_in_memory: bool = True


class ApprovalRequest(BaseModel):
    """Request to approve/deny an operation."""
    request_id: str
    approved: bool
    reason: Optional[str] = None


# Process Manager Request/Response Models
class SpawnAgentRequest(BaseModel):
    """Request to spawn a new agent."""
    name: str = Field(..., description="Agent name")
    agent_class: str = Field(..., description="Registered agent class name")
    priority: str = Field(default="NORMAL", description="Process priority")
    restart_policy: str = Field(default="on_failure", description="Restart policy")
    max_restarts: int = Field(default=5, description="Maximum restarts")
    system_prompt: Optional[str] = None
    tools: list[str] = Field(default_factory=list)
    initial_goal: Optional[str] = None
    instructions: list[str] = Field(default_factory=list)
    input_channels: list[str] = Field(default_factory=list)
    output_channels: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)


class ScheduleTaskRequest(BaseModel):
    """Request to schedule a task."""
    name: str = Field(..., description="Task name")
    handler: str = Field(..., description="Handler function name")
    schedule_type: str = Field(..., description="once, interval, or cron")
    params: dict = Field(default_factory=dict)
    cron_expression: Optional[str] = None
    interval_seconds: Optional[int] = None
    run_at: Optional[datetime] = None
    priority: str = Field(default="NORMAL")
    timeout_seconds: int = Field(default=300)


class ProcessSignalRequest(BaseModel):
    """Request to send a signal to a process."""
    signal: str = Field(..., description="Signal type")
    payload: dict = Field(default_factory=dict)


class EmitEventRequest(BaseModel):
    """Request to emit an event."""
    type: str = Field(..., description="Event type/channel")
    payload: dict = Field(..., description="Event payload")


# ==================== Route Setup Functions ====================

def setup_routes(app: FastAPI, kernel) -> None:
    """Setup all routes for the AION API."""

    @app.get("/")
    async def root():
        """API root - returns system information."""
        return {
            "name": "AION",
            "version": "1.0.0",
            "description": "Artificial Intelligence Operating Nexus",
            "status": kernel.get_status(),
        }

    @app.get("/health")
    async def health():
        """Health check endpoint."""
        health = kernel.get_health()
        all_ready = all(h.status.value == "ready" for h in health.values())
        return {
            "healthy": all_ready,
            "systems": {
                name: {"status": h.status.value, "message": h.message}
                for name, h in health.items()
            },
        }

    @app.get("/status")
    async def status():
        """Get detailed system status."""
        return kernel.get_status()

    @app.post("/process", response_model=ProcessResponse)
    async def process_request(request: ProcessRequest):
        """Process a user request through the cognitive pipeline."""
        try:
            result = await kernel.process_request(
                request=request.query,
                context=request.context,
                user_id=request.user_id,
                session_id=request.session_id,
            )
            return ProcessResponse(
                request_id=result.request_id,
                success=result.success,
                result=result.result,
                error=result.error,
                execution_time_ms=result.execution_time_ms,
            )
        except Exception as e:
            logger.error("Process request failed", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/emergency-stop")
    async def emergency_stop(reason: str = Body(..., embed=True)):
        """Activate emergency stop."""
        kernel.security.emergency_stop(reason)
        return {"status": "emergency_stop_activated", "reason": reason}

    @app.post("/clear-emergency-stop")
    async def clear_emergency_stop():
        """Clear emergency stop."""
        kernel.security.clear_emergency_stop()
        return {"status": "emergency_stop_cleared"}


def setup_planning_routes(app: FastAPI, planning_graph) -> None:
    """Setup routes for the Planning Graph system."""

    @app.get("/planning/plans")
    async def list_plans():
        """List all execution plans."""
        plans = planning_graph.list_plans()
        return {"plans": [p.to_dict() for p in plans]}

    @app.post("/planning/plans")
    async def create_plan(request: PlanCreateRequest):
        """Create a new execution plan."""
        plan = planning_graph.create_plan(
            name=request.name,
            description=request.description,
        )

        # Add steps as nodes
        prev_node_id = None
        for i, step in enumerate(request.steps):
            node = planning_graph.add_node(
                plan_id=plan.id,
                name=step.get("name", f"Step {i+1}"),
                action=step.get("action", ""),
                params=step.get("params", {}),
            )

            # Connect to previous node
            if prev_node_id:
                planning_graph.add_edge(plan.id, prev_node_id, node.id)
            else:
                # Connect start to first node
                start_id = f"{plan.id}_start"
                planning_graph.add_edge(plan.id, start_id, node.id)

            prev_node_id = node.id

        # Connect last node to end
        if prev_node_id:
            end_id = f"{plan.id}_end"
            planning_graph.add_edge(plan.id, prev_node_id, end_id)

        return {"plan": plan.to_dict()}

    @app.get("/planning/plans/{plan_id}")
    async def get_plan(plan_id: str):
        """Get a specific plan."""
        plan = planning_graph.get_plan(plan_id)
        if not plan:
            raise HTTPException(status_code=404, detail="Plan not found")
        return {"plan": plan.to_dict()}

    @app.post("/planning/plans/{plan_id}/execute")
    async def execute_plan(plan_id: str, context: Optional[dict] = Body(default=None)):
        """Execute a plan."""
        try:
            result = await planning_graph.execute_plan(plan_id, context=context)
            return result
        except KeyError:
            raise HTTPException(status_code=404, detail="Plan not found")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.get("/planning/plans/{plan_id}/visualize")
    async def visualize_plan(plan_id: str, format: str = "ascii"):
        """Get plan visualization."""
        from aion.systems.planning import PlanVisualizer

        visualizer = PlanVisualizer(planning_graph)

        if format == "ascii":
            return {"visualization": visualizer.to_ascii(plan_id)}
        elif format == "mermaid":
            return {"visualization": visualizer.to_mermaid(plan_id)}
        elif format == "graphviz":
            return {"visualization": visualizer.to_graphviz(plan_id)}
        elif format == "html":
            return JSONResponse(
                content=visualizer.to_html(plan_id),
                media_type="text/html",
            )
        else:
            raise HTTPException(status_code=400, detail=f"Unknown format: {format}")

    @app.get("/planning/stats")
    async def planning_stats():
        """Get planning system statistics."""
        return planning_graph.get_stats()


def setup_memory_routes(app: FastAPI, memory_system) -> None:
    """Setup routes for the Cognitive Memory system."""

    @app.post("/memory/store")
    async def store_memory(request: MemoryStoreRequest):
        """Store a new memory."""
        from aion.systems.memory import CognitiveMemorySystem

        memory_type_enum = CognitiveMemorySystem.MemoryType if hasattr(
            CognitiveMemorySystem, 'MemoryType'
        ) else None

        memory = await memory_system.store(
            content=request.content,
            metadata=request.metadata,
            importance=request.importance,
        )

        return {"memory": memory.to_dict()}

    @app.post("/memory/search")
    async def search_memories(request: MemorySearchRequest):
        """Search for memories."""
        results = await memory_system.search(
            query=request.query,
            limit=request.limit,
            min_importance=request.min_importance,
        )

        return {
            "results": [
                {
                    "memory": r.memory.to_dict(),
                    "relevance": r.relevance,
                    "combined_score": r.combined_score,
                }
                for r in results
            ]
        }

    @app.post("/memory/recall")
    async def recall_memory(query: str = Body(..., embed=True)):
        """Recall and synthesize information from memory."""
        result = await memory_system.recall(query)
        return {"recall": result}

    @app.post("/memory/consolidate")
    async def consolidate_memories():
        """Trigger memory consolidation."""
        stats = await memory_system.consolidate()
        return {"consolidation": stats}

    @app.get("/memory/stats")
    async def memory_stats():
        """Get memory system statistics."""
        return memory_system.get_stats()

    @app.get("/memory/working")
    async def get_working_memory():
        """Get current working memory contents."""
        wm = memory_system.get_working_memory()
        return {
            "items": [m.to_dict() for m in wm.get_all()],
            "summary": wm.get_summary(),
        }


def setup_tool_routes(app: FastAPI, tool_orchestrator) -> None:
    """Setup routes for the Tool Orchestration system."""

    @app.get("/tools")
    async def list_tools(category: Optional[str] = None):
        """List available tools."""
        from aion.systems.tools.registry import ToolCategory

        cat = ToolCategory(category) if category else None
        tools = tool_orchestrator.list_tools(category=cat)
        return {"tools": tools}

    @app.post("/tools/execute")
    async def execute_tool(request: ToolExecuteRequest):
        """Execute a single tool."""
        result = await tool_orchestrator.execute(
            tool_name=request.tool_name,
            params=request.params,
            timeout=request.timeout,
        )
        return {"result": result.to_dict()}

    @app.post("/tools/execute-parallel")
    async def execute_tools_parallel(calls: list[ToolExecuteRequest]):
        """Execute multiple tools in parallel."""
        call_list = [(c.tool_name, c.params) for c in calls]
        results = await tool_orchestrator.execute_parallel(call_list)
        return {"results": [r.to_dict() for r in results]}

    @app.post("/tools/suggest")
    async def suggest_tools(task_description: str = Body(..., embed=True)):
        """Suggest tools for a task."""
        suggestions = tool_orchestrator.suggest_tools(task_description)
        return {"suggestions": suggestions}

    @app.get("/tools/stats")
    async def tool_stats():
        """Get tool orchestration statistics."""
        return tool_orchestrator.get_stats()


def setup_evolution_routes(app: FastAPI, evolution_engine) -> None:
    """Setup routes for the Self-Improvement Engine."""

    @app.get("/evolution/status")
    async def evolution_status():
        """Get evolution engine status."""
        return evolution_engine.get_stats()

    @app.get("/evolution/parameters")
    async def get_parameters():
        """Get current parameter values."""
        return {"parameters": evolution_engine.get_current_parameters()}

    @app.get("/evolution/approvals")
    async def get_pending_approvals():
        """Get pending approval requests."""
        return {"pending": evolution_engine.get_pending_approvals()}

    @app.post("/evolution/approve")
    async def approve_hypothesis(request: ApprovalRequest):
        """Approve or reject a hypothesis."""
        if request.approved:
            success = evolution_engine.approve_hypothesis(request.request_id)
        else:
            success = evolution_engine.reject_hypothesis(request.request_id)

        return {"success": success}

    @app.post("/evolution/start")
    async def start_evolution():
        """Start the improvement loop."""
        await evolution_engine.start_improvement_loop()
        return {"status": "started"}

    @app.post("/evolution/stop")
    async def stop_evolution():
        """Stop the improvement loop."""
        await evolution_engine.stop_improvement_loop()
        return {"status": "stopped"}

    @app.post("/evolution/emergency-rollback")
    async def emergency_rollback():
        """Perform emergency rollback."""
        success = evolution_engine.emergency_rollback()
        return {"success": success}


def setup_vision_routes(app: FastAPI, visual_cortex) -> None:
    """Setup routes for the Visual Cortex system."""

    @app.post("/vision/analyze")
    async def analyze_image(request: VisionAnalyzeRequest):
        """Analyze an image."""
        if not request.image_url:
            raise HTTPException(status_code=400, detail="image_url is required")

        result = await visual_cortex.process(
            image_path=request.image_url,
            query=request.question,
            store_in_memory=request.store_in_memory,
        )

        return {"analysis": result.to_dict()}

    @app.post("/vision/analyze-upload")
    async def analyze_uploaded_image(
        file: UploadFile = File(...),
        question: Optional[str] = Query(default=None),
    ):
        """Analyze an uploaded image."""
        image_bytes = await file.read()
        result = await visual_cortex.process(
            image_path=image_bytes,
            query=question,
        )
        return {"analysis": result.to_dict()}

    @app.post("/vision/detect")
    async def detect_objects(
        image_url: str = Body(..., embed=True),
        threshold: float = Body(default=0.5, embed=True),
    ):
        """Detect objects in an image."""
        objects = await visual_cortex.detect_objects(image_url, threshold)
        return {"objects": [o.to_dict() for o in objects]}

    @app.post("/vision/describe")
    async def describe_image(image_url: str = Body(..., embed=True)):
        """Generate a description of an image."""
        description = await visual_cortex.describe(image_url)
        return {"description": description}

    @app.post("/vision/answer")
    async def answer_visual_question(
        image_url: str = Body(...),
        question: str = Body(...),
    ):
        """Answer a question about an image."""
        answer = await visual_cortex.answer(image_url, question)
        return {"answer": answer}

    @app.post("/vision/compare")
    async def compare_images(
        image1_url: str = Body(...),
        image2_url: str = Body(...),
    ):
        """Compare two images."""
        comparison = await visual_cortex.compare(image1_url, image2_url)
        return {"comparison": comparison}

    @app.post("/vision/imagine")
    async def imagine_scene(description: str = Body(..., embed=True)):
        """Imagine a scene from a description."""
        scene = await visual_cortex.imagine(description)
        return {"scene": scene.to_dict()}

    @app.get("/vision/stats")
    async def vision_stats():
        """Get visual cortex statistics."""
        return visual_cortex.get_stats()


def setup_process_routes(app: FastAPI, kernel) -> None:
    """Setup routes for the Process & Agent Manager system."""
    import uuid

    # Conditional imports
    try:
        from aion.systems.process import (
            AgentConfig,
            ProcessPriority,
            RestartPolicy,
            Event,
        )
        process_available = True
    except ImportError:
        process_available = False

    if not process_available:
        logger.warning("Process manager not available, skipping routes")
        return

    # ==================== Process Management ====================

    @app.get("/processes")
    async def list_processes(
        state: Optional[str] = None,
        process_type: Optional[str] = None,
        priority: Optional[str] = None,
        tag: Optional[str] = None,
    ):
        """List all processes with optional filtering."""
        if not kernel.supervisor:
            raise HTTPException(status_code=503, detail="Process supervisor not available")

        processes = kernel.supervisor.get_all_processes()

        # Apply filters
        if state:
            processes = [p for p in processes if p.state.value == state]
        if process_type:
            processes = [p for p in processes if p.type.value == process_type]
        if priority:
            processes = [p for p in processes if p.priority.name == priority]
        if tag:
            processes = [p for p in processes if tag in p.tags]

        return {"processes": [p.to_dict() for p in processes]}

    @app.get("/processes/{process_id}")
    async def get_process(process_id: str):
        """Get process details by ID."""
        if not kernel.supervisor:
            raise HTTPException(status_code=503, detail="Process supervisor not available")

        process = kernel.supervisor.get_process(process_id)
        if not process:
            raise HTTPException(status_code=404, detail="Process not found")

        return {"process": process.to_dict()}

    @app.post("/processes/spawn")
    async def spawn_agent(request: SpawnAgentRequest):
        """Spawn a new agent process."""
        if not kernel.supervisor:
            raise HTTPException(status_code=503, detail="Process supervisor not available")

        try:
            config = AgentConfig(
                name=request.name,
                agent_class=request.agent_class,
                priority=ProcessPriority[request.priority],
                restart_policy=RestartPolicy(request.restart_policy),
                max_restarts=request.max_restarts,
                system_prompt=request.system_prompt,
                tools=request.tools,
                initial_goal=request.initial_goal,
                instructions=request.instructions,
                input_channels=request.input_channels,
                output_channels=request.output_channels,
                metadata=request.metadata,
                tags=request.tags,
            )

            process_id = await kernel.supervisor.spawn_agent(config)
            return {"process_id": process_id, "status": "spawned"}

        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))

    @app.post("/processes/{process_id}/stop")
    async def stop_process(
        process_id: str,
        graceful: bool = True,
        timeout: float = 30.0,
    ):
        """Stop a process."""
        if not kernel.supervisor:
            raise HTTPException(status_code=503, detail="Process supervisor not available")

        success = await kernel.supervisor.stop_process(process_id, graceful, timeout)
        return {"success": success}

    @app.post("/processes/{process_id}/kill")
    async def kill_process(process_id: str):
        """Force terminate a process."""
        if not kernel.supervisor:
            raise HTTPException(status_code=503, detail="Process supervisor not available")

        success = await kernel.supervisor.kill_process(process_id)
        return {"success": success}

    @app.post("/processes/{process_id}/pause")
    async def pause_process(process_id: str):
        """Pause a running process."""
        if not kernel.supervisor:
            raise HTTPException(status_code=503, detail="Process supervisor not available")

        success = await kernel.supervisor.pause_process(process_id)
        return {"success": success}

    @app.post("/processes/{process_id}/resume")
    async def resume_process(process_id: str):
        """Resume a paused process."""
        if not kernel.supervisor:
            raise HTTPException(status_code=503, detail="Process supervisor not available")

        success = await kernel.supervisor.resume_process(process_id)
        return {"success": success}

    @app.post("/processes/{process_id}/restart")
    async def restart_process(process_id: str):
        """Restart a process."""
        if not kernel.supervisor:
            raise HTTPException(status_code=503, detail="Process supervisor not available")

        success = await kernel.supervisor.restart_process(process_id)
        return {"success": success}

    @app.post("/processes/{process_id}/signal")
    async def send_signal(process_id: str, request: ProcessSignalRequest):
        """Send a signal to a process."""
        if not kernel.supervisor:
            raise HTTPException(status_code=503, detail="Process supervisor not available")

        from aion.systems.process import SignalType
        try:
            signal = SignalType(request.signal)
        except ValueError:
            signal = request.signal

        success = await kernel.supervisor.send_signal(process_id, signal, request.payload)
        return {"success": success}

    @app.get("/processes/{process_id}/children")
    async def get_process_children(process_id: str):
        """Get child processes of a process."""
        if not kernel.supervisor:
            raise HTTPException(status_code=503, detail="Process supervisor not available")

        children = kernel.supervisor.get_children(process_id)
        return {"children": [c.to_dict() for c in children]}

    @app.get("/processes/{process_id}/checkpoints")
    async def get_process_checkpoints(process_id: str):
        """Get checkpoints for a process."""
        if not kernel.supervisor:
            raise HTTPException(status_code=503, detail="Process supervisor not available")

        checkpoints = kernel.supervisor.get_checkpoints(process_id)
        return {"checkpoints": [c.to_dict() for c in checkpoints]}

    @app.post("/processes/{process_id}/checkpoint")
    async def create_checkpoint(process_id: str, reason: str = Body(default="manual", embed=True)):
        """Create a checkpoint for a process."""
        if not kernel.supervisor:
            raise HTTPException(status_code=503, detail="Process supervisor not available")

        checkpoint = await kernel.supervisor.create_checkpoint(process_id, reason)
        if not checkpoint:
            raise HTTPException(status_code=404, detail="Process not found or not running")

        return {"checkpoint": checkpoint.to_dict()}

    @app.get("/processes/stats")
    async def get_process_stats():
        """Get process supervisor statistics."""
        if not kernel.supervisor:
            raise HTTPException(status_code=503, detail="Process supervisor not available")

        return kernel.supervisor.get_stats()

    @app.get("/processes/registered-classes")
    async def get_registered_classes():
        """Get list of registered agent classes."""
        if not kernel.supervisor:
            raise HTTPException(status_code=503, detail="Process supervisor not available")

        return {"classes": kernel.supervisor.get_registered_classes()}

    # ==================== Task Scheduling ====================

    @app.get("/tasks")
    async def list_tasks(
        enabled: Optional[bool] = None,
        schedule_type: Optional[str] = None,
        tag: Optional[str] = None,
    ):
        """List scheduled tasks."""
        if not kernel.scheduler:
            raise HTTPException(status_code=503, detail="Task scheduler not available")

        tasks = kernel.scheduler.get_all_tasks()

        if enabled is not None:
            tasks = [t for t in tasks if t.enabled == enabled]
        if schedule_type:
            tasks = [t for t in tasks if t.schedule_type == schedule_type]
        if tag:
            tasks = [t for t in tasks if tag in t.tags]

        return {"tasks": [t.to_dict() for t in tasks]}

    @app.get("/tasks/{task_id}")
    async def get_task(task_id: str):
        """Get task details."""
        if not kernel.scheduler:
            raise HTTPException(status_code=503, detail="Task scheduler not available")

        task = kernel.scheduler.get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

        return {"task": task.to_dict()}

    @app.post("/tasks/schedule")
    async def schedule_task(request: ScheduleTaskRequest):
        """Schedule a new task."""
        if not kernel.scheduler:
            raise HTTPException(status_code=503, detail="Task scheduler not available")

        try:
            if request.schedule_type == "once":
                if not request.run_at:
                    raise HTTPException(status_code=400, detail="run_at required for once")
                task_id = await kernel.scheduler.schedule_once(
                    name=request.name,
                    handler=request.handler,
                    run_at=request.run_at,
                    params=request.params,
                    priority=ProcessPriority[request.priority],
                    timeout_seconds=request.timeout_seconds,
                )
            elif request.schedule_type == "interval":
                if not request.interval_seconds:
                    raise HTTPException(status_code=400, detail="interval_seconds required")
                task_id = await kernel.scheduler.schedule_interval(
                    name=request.name,
                    handler=request.handler,
                    interval_seconds=request.interval_seconds,
                    params=request.params,
                    priority=ProcessPriority[request.priority],
                    timeout_seconds=request.timeout_seconds,
                )
            elif request.schedule_type == "cron":
                if not request.cron_expression:
                    raise HTTPException(status_code=400, detail="cron_expression required")
                task_id = await kernel.scheduler.schedule_cron(
                    name=request.name,
                    handler=request.handler,
                    cron_expression=request.cron_expression,
                    params=request.params,
                    priority=ProcessPriority[request.priority],
                    timeout_seconds=request.timeout_seconds,
                )
            else:
                raise HTTPException(status_code=400, detail=f"Unknown schedule_type: {request.schedule_type}")

            return {"task_id": task_id, "status": "scheduled"}

        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.delete("/tasks/{task_id}")
    async def cancel_task(task_id: str):
        """Cancel a scheduled task."""
        if not kernel.scheduler:
            raise HTTPException(status_code=503, detail="Task scheduler not available")

        success = await kernel.scheduler.cancel_task(task_id)
        return {"success": success}

    @app.post("/tasks/{task_id}/pause")
    async def pause_task(task_id: str):
        """Pause a scheduled task."""
        if not kernel.scheduler:
            raise HTTPException(status_code=503, detail="Task scheduler not available")

        success = await kernel.scheduler.pause_task(task_id)
        return {"success": success}

    @app.post("/tasks/{task_id}/resume")
    async def resume_task(task_id: str):
        """Resume a paused task."""
        if not kernel.scheduler:
            raise HTTPException(status_code=503, detail="Task scheduler not available")

        success = await kernel.scheduler.resume_task(task_id)
        return {"success": success}

    @app.post("/tasks/{task_id}/trigger")
    async def trigger_task(task_id: str):
        """Manually trigger a task immediately."""
        if not kernel.scheduler:
            raise HTTPException(status_code=503, detail="Task scheduler not available")

        process_id = await kernel.scheduler.trigger_task(task_id)
        if not process_id:
            raise HTTPException(status_code=404, detail="Task not found")

        return {"process_id": process_id}

    @app.get("/tasks/stats")
    async def get_scheduler_stats():
        """Get scheduler statistics."""
        if not kernel.scheduler:
            raise HTTPException(status_code=503, detail="Task scheduler not available")

        return kernel.scheduler.get_stats()

    # ==================== Event Bus ====================

    @app.get("/events/history")
    async def get_event_history(
        pattern: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ):
        """Get event history."""
        if not kernel.event_bus:
            raise HTTPException(status_code=503, detail="Event bus not available")

        events = kernel.event_bus.get_history(pattern, limit)
        return {"events": [e.to_dict() for e in events]}

    @app.post("/events/emit")
    async def emit_event(request: EmitEventRequest):
        """Emit an event to the event bus."""
        if not kernel.event_bus:
            raise HTTPException(status_code=503, detail="Event bus not available")

        event = Event(
            id=str(uuid.uuid4()),
            type=request.type,
            source="api",
            payload=request.payload,
        )

        await kernel.event_bus.emit(event)
        return {"event_id": event.id, "status": "emitted"}

    @app.get("/events/stats")
    async def get_event_bus_stats():
        """Get event bus statistics."""
        if not kernel.event_bus:
            raise HTTPException(status_code=503, detail="Event bus not available")

        return kernel.event_bus.get_stats()

    @app.get("/events/patterns")
    async def get_subscribed_patterns():
        """Get all subscribed patterns."""
        if not kernel.event_bus:
            raise HTTPException(status_code=503, detail="Event bus not available")

        return {"patterns": kernel.event_bus.get_all_patterns()}

    # ==================== Worker Pool ====================

    @app.get("/workers")
    async def get_workers():
        """Get worker pool information."""
        if not kernel.worker_pool:
            raise HTTPException(status_code=503, detail="Worker pool not available")

        workers = kernel.worker_pool.get_workers()
        return {
            "workers": [w.to_dict() for w in workers],
            "stats": kernel.worker_pool.get_stats(),
        }

    @app.get("/workers/stats")
    async def get_worker_pool_stats():
        """Get worker pool statistics."""
        if not kernel.worker_pool:
            raise HTTPException(status_code=503, detail="Worker pool not available")

        return kernel.worker_pool.get_stats()

    # ==================== Combined Stats ====================

    @app.get("/process-manager/stats")
    async def get_all_process_stats():
        """Get comprehensive process manager statistics."""
        return kernel.get_process_stats()
# ==================== Audio Request/Response Models ====================

class AudioTranscribeRequest(BaseModel):
    """Request to transcribe audio from URL."""
    audio_url: str
    language: Optional[str] = None
    enable_diarization: bool = True
    enable_timestamps: bool = True


class AudioSynthesizeRequest(BaseModel):
    """Request to synthesize speech."""
    text: str = Field(..., description="Text to synthesize")
    voice: Optional[str] = Field(default=None, description="Voice preset or speaker ID")
    language: str = Field(default="en")
    speed: float = Field(default=1.0, ge=0.5, le=2.0)


class AudioQuestionRequest(BaseModel):
    """Request to answer a question about audio."""
    audio_url: str
    question: str


class AudioMemoryRequest(BaseModel):
    """Request to store audio in memory."""
    audio_url: str
    context: Optional[str] = None
    importance: float = Field(default=0.5, ge=0, le=1)
    tags: Optional[list[str]] = None


class AudioSearchRequest(BaseModel):
    """Request to search audio memory."""
    query: str
    limit: int = Field(default=5, ge=1, le=50)


class SpeakerRegisterRequest(BaseModel):
    """Request to register a speaker."""
    audio_url: str
    name: str


def setup_audio_routes(app: FastAPI, audio_cortex) -> None:
    """Setup routes for the Auditory Cortex system."""

    @app.post("/audio/transcribe")
    async def transcribe_audio(request: AudioTranscribeRequest):
        """
        Transcribe audio to text with optional speaker diarization.

        Returns transcript with word-level timestamps and speaker attribution.
        """
        transcript = await audio_cortex.transcribe(
            audio=request.audio_url,
            language=request.language,
            enable_diarization=request.enable_diarization,
            enable_timestamps=request.enable_timestamps,
        )
        return {"transcript": transcript.to_dict()}

    @app.post("/audio/transcribe-upload")
    async def transcribe_uploaded_audio(
        file: UploadFile = File(...),
        language: Optional[str] = Query(default=None),
        enable_diarization: bool = Query(default=True),
    ):
        """Transcribe an uploaded audio file."""
        audio_bytes = await file.read()
        transcript = await audio_cortex.transcribe(
            audio=audio_bytes,
            language=language,
            enable_diarization=enable_diarization,
        )
        return {"transcript": transcript.to_dict()}

    @app.post("/audio/detect-events")
    async def detect_audio_events(
        audio_url: str = Body(..., embed=True),
        threshold: float = Body(default=0.3, embed=True),
    ):
        """
        Detect audio events (speech, music, environmental sounds).

        Returns list of detected events with timestamps and confidence.
        """
        events = await audio_cortex.detect_events(audio_url, threshold=threshold)
        return {"events": [e.to_dict() for e in events]}

    @app.post("/audio/detect-events-upload")
    async def detect_events_upload(
        file: UploadFile = File(...),
        threshold: float = Query(default=0.3),
    ):
        """Detect events in uploaded audio."""
        audio_bytes = await file.read()
        events = await audio_cortex.detect_events(audio_bytes, threshold=threshold)
        return {"events": [e.to_dict() for e in events]}

    @app.post("/audio/understand")
    async def understand_audio(
        audio_url: str = Body(..., embed=True),
        store_in_memory: bool = Body(default=False, embed=True),
        context: Optional[str] = Body(default=None, embed=True),
    ):
        """
        Full audio scene understanding.

        Combines transcription, event detection, speaker diarization,
        and music analysis for comprehensive audio understanding.
        """
        scene = await audio_cortex.understand_scene(
            audio=audio_url,
            store_in_memory=store_in_memory,
            context=context or "",
        )
        return {"scene": scene.to_dict()}

    @app.post("/audio/understand-upload")
    async def understand_audio_upload(
        file: UploadFile = File(...),
        store_in_memory: bool = Query(default=False),
    ):
        """Full understanding of uploaded audio."""
        audio_bytes = await file.read()
        scene = await audio_cortex.understand_scene(
            audio=audio_bytes,
            store_in_memory=store_in_memory,
        )
        return {"scene": scene.to_dict()}

    @app.post("/audio/synthesize")
    async def synthesize_speech(request: AudioSynthesizeRequest):
        """
        Generate speech from text.

        Returns audio segment with waveform data encoded as base64.
        """
        import base64

        audio = await audio_cortex.synthesize(
            text=request.text,
            voice=request.voice,
            language=request.language,
            speed=request.speed,
        )

        # Encode waveform as base64 for JSON transport
        waveform_b64 = None
        if audio.waveform is not None:
            import io
            import soundfile as sf
            buf = io.BytesIO()
            sf.write(buf, audio.waveform, audio.sample_rate, format='WAV')
            waveform_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

        return {
            "audio": {
                "id": audio.id,
                "duration": audio.duration,
                "sample_rate": audio.sample_rate,
                "waveform_base64": waveform_b64,
                "metadata": audio.metadata,
            }
        }

    @app.post("/audio/identify-speaker")
    async def identify_speaker(audio_url: str = Body(..., embed=True)):
        """
        Identify speaker in audio.

        Matches against registered speakers and returns identification result.
        """
        speaker, confidence = await audio_cortex.identify_speaker(audio_url)
        return {
            "speaker": speaker.to_dict() if speaker else None,
            "confidence": confidence,
            "identified": speaker is not None,
        }

    @app.post("/audio/identify-speaker-upload")
    async def identify_speaker_upload(file: UploadFile = File(...)):
        """Identify speaker from uploaded audio."""
        audio_bytes = await file.read()
        speaker, confidence = await audio_cortex.identify_speaker(audio_bytes)
        return {
            "speaker": speaker.to_dict() if speaker else None,
            "confidence": confidence,
            "identified": speaker is not None,
        }

    @app.post("/audio/register-speaker")
    async def register_speaker(request: SpeakerRegisterRequest):
        """
        Register a new speaker from sample audio.

        The audio should contain clear speech from a single speaker.
        """
        speaker = await audio_cortex.register_speaker(
            audio=request.audio_url,
            name=request.name,
        )
        return {"speaker": speaker.to_dict()}

    @app.post("/audio/register-speaker-upload")
    async def register_speaker_upload(
        file: UploadFile = File(...),
        name: str = Query(...),
    ):
        """Register a speaker from uploaded audio."""
        audio_bytes = await file.read()
        speaker = await audio_cortex.register_speaker(audio=audio_bytes, name=name)
        return {"speaker": speaker.to_dict()}

    @app.post("/audio/verify-speaker")
    async def verify_speaker(
        audio_url: str = Body(...),
        speaker_id: str = Body(...),
    ):
        """Verify if audio matches a claimed speaker."""
        # Get registered speaker
        if audio_cortex.memory:
            registered_speakers = await audio_cortex.memory.get_registered_speakers()
            speaker = next((s for s in registered_speakers if s.id == speaker_id), None)
            if speaker:
                is_verified, similarity = await audio_cortex.verify_speaker(
                    audio=audio_url,
                    claimed_speaker=speaker,
                )
                return {
                    "verified": is_verified,
                    "similarity": similarity,
                    "threshold": 0.75,
                }
        return {"verified": False, "similarity": 0.0, "error": "Speaker not found"}

    @app.post("/audio/compare-speakers")
    async def compare_speakers(
        audio1_url: str = Body(...),
        audio2_url: str = Body(...),
    ):
        """Compare two audio samples for speaker similarity."""
        is_same, similarity = await audio_cortex.compare_speakers(
            audio1=audio1_url,
            audio2=audio2_url,
        )
        return {
            "same_speaker": is_same,
            "similarity": similarity,
            "threshold": 0.75,
        }

    @app.post("/audio/answer")
    async def answer_audio_question(request: AudioQuestionRequest):
        """
        Answer a question about audio content.

        Uses audio understanding and LLM reasoning to answer questions.
        """
        answer = await audio_cortex.answer_question(
            audio=request.audio_url,
            question=request.question,
        )
        return {"answer": answer, "question": request.question}

    @app.post("/audio/summarize")
    async def summarize_audio(audio_url: str = Body(..., embed=True)):
        """Generate a summary of audio content."""
        summary = await audio_cortex.summarize(audio_url)
        return {"summary": summary}

    @app.post("/audio/compare")
    async def compare_audio(
        audio1_url: str = Body(...),
        audio2_url: str = Body(...),
    ):
        """Compare two audio samples."""
        comparison = await audio_cortex.compare(audio1_url, audio2_url)
        return {"comparison": comparison.to_dict()}

    @app.post("/audio/analyze-music")
    async def analyze_music(audio_url: str = Body(..., embed=True)):
        """Analyze music content (tempo, key, mood, etc.)."""
        analysis = await audio_cortex.analyze_music(audio_url)
        return {"analysis": analysis.to_dict()}

    @app.post("/audio/remember")
    async def remember_audio(request: AudioMemoryRequest):
        """Store audio in memory for later retrieval."""
        memory_id = await audio_cortex.remember(
            audio=request.audio_url,
            context=request.context,
            importance=request.importance,
            tags=request.tags,
        )
        return {"memory_id": memory_id, "stored": True}

    @app.post("/audio/recall")
    async def recall_audio(request: AudioSearchRequest):
        """Retrieve similar audio from memory."""
        results = await audio_cortex.recall_similar(
            query=request.query,
            limit=request.limit,
        )
        return {"results": [r.to_dict() for r in results]}

    @app.post("/audio/recall-by-tags")
    async def recall_by_tags(
        tags: list[str] = Body(...),
        require_all: bool = Body(default=False),
        limit: int = Body(default=10),
    ):
        """Retrieve audio by tags."""
        results = await audio_cortex.recall_by_tags(
            tags=tags,
            require_all=require_all,
            limit=limit,
        )
        return {"results": [r.to_dict() for r in results]}

    @app.get("/audio/stats")
    async def audio_stats():
        """Get audio system statistics."""
        return audio_cortex.get_stats()

    @app.get("/audio/speakers")
    async def list_speakers():
        """List all registered speakers."""
        if audio_cortex.memory:
            speakers = await audio_cortex.memory.get_registered_speakers()
            return {"speakers": [s.to_dict() for s in speakers]}
        return {"speakers": []}


# ==================== MCP Integration Routes ====================

class MCPServerConnectRequest(BaseModel):
    """Request to connect to an MCP server."""
    name: str = Field(..., description="Server name from registry")


class MCPToolCallRequest(BaseModel):
    """Request to call an MCP tool."""
    server: str = Field(..., description="Server name")
    tool: str = Field(..., description="Tool name")
    arguments: dict = Field(default_factory=dict, description="Tool arguments")


class MCPResourceReadRequest(BaseModel):
    """Request to read an MCP resource."""
    server: str = Field(..., description="Server name")
    uri: str = Field(..., description="Resource URI")


class MCPPromptGetRequest(BaseModel):
    """Request to get an MCP prompt."""
    server: str = Field(..., description="Server name")
    name: str = Field(..., description="Prompt name")
    arguments: Optional[dict] = Field(default=None, description="Prompt arguments")


class MCPServerRegisterRequest(BaseModel):
    """Request to register a new MCP server."""
    name: str = Field(..., description="Server name")
    transport: str = Field(default="stdio", description="Transport type")
    command: Optional[str] = None
    args: list[str] = Field(default_factory=list)
    env: dict = Field(default_factory=dict)
    url: Optional[str] = None
    ws_url: Optional[str] = None
    headers: dict = Field(default_factory=dict)
    description: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    enabled: bool = True


def setup_mcp_routes(app: FastAPI, mcp_manager) -> None:
    """Setup routes for MCP integration."""

    @app.get("/mcp/status")
    async def mcp_status():
        """Get MCP integration status."""
        return {
            "enabled": True,
            "stats": mcp_manager.get_stats(),
        }

    @app.get("/mcp/servers")
    async def list_mcp_servers():
        """List all configured MCP servers."""
        states = mcp_manager.get_server_states()
        return {
            "servers": [
                {
                    "name": name,
                    "connected": state.connected,
                    "tools_count": len(state.tools),
                    "resources_count": len(state.resources),
                    "prompts_count": len(state.prompts),
                    "last_connected": state.last_connected.isoformat() if state.last_connected else None,
                    "last_error": state.last_error,
                }
                for name, state in states.items()
            ]
        }

    @app.get("/mcp/servers/available")
    async def list_available_servers():
        """List all servers in the registry (including not connected)."""
        all_servers = mcp_manager.registry.get_all_servers()
        connected = set(mcp_manager.get_connected_servers())

        return {
            "servers": [
                {
                    "name": s.name,
                    "transport": s.transport.value,
                    "description": s.description,
                    "enabled": s.enabled,
                    "connected": s.name in connected,
                    "tags": s.tags,
                }
                for s in all_servers
            ]
        }

    @app.post("/mcp/servers/connect")
    async def connect_mcp_server(request: MCPServerConnectRequest):
        """Connect to an MCP server."""
        try:
            success = await mcp_manager.connect_server(request.name)
            state = mcp_manager.get_server_state(request.name)
            return {
                "success": success,
                "server": request.name,
                "tools_count": len(state.tools) if state else 0,
                "resources_count": len(state.resources) if state else 0,
                "prompts_count": len(state.prompts) if state else 0,
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/mcp/servers/{name}/disconnect")
    async def disconnect_mcp_server(name: str):
        """Disconnect from an MCP server."""
        success = await mcp_manager.disconnect_server(name)
        return {"success": success, "server": name}

    @app.post("/mcp/servers/{name}/reconnect")
    async def reconnect_mcp_server(name: str):
        """Reconnect to an MCP server."""
        try:
            success = await mcp_manager.reconnect_server(name)
            return {"success": success, "server": name}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/mcp/servers/{name}/ping")
    async def ping_mcp_server(name: str):
        """Ping an MCP server."""
        result = await mcp_manager.ping_server(name)
        return {"server": name, "reachable": result}

    @app.get("/mcp/servers/{name}/state")
    async def get_server_state(name: str):
        """Get detailed state of a specific server."""
        state = mcp_manager.get_server_state(name)
        if not state:
            raise HTTPException(status_code=404, detail=f"Server not found: {name}")
        return state.to_dict()

    @app.post("/mcp/servers/register")
    async def register_mcp_server(request: MCPServerRegisterRequest):
        """Register a new MCP server configuration."""
        from aion.mcp.types import ServerConfig, TransportType

        config = ServerConfig(
            name=request.name,
            transport=TransportType(request.transport),
            command=request.command,
            args=request.args,
            env=request.env,
            url=request.url,
            ws_url=request.ws_url,
            headers=request.headers,
            description=request.description,
            tags=request.tags,
            enabled=request.enabled,
        )
        mcp_manager.registry.register(config)
        return {"success": True, "server": request.name}

    @app.delete("/mcp/servers/{name}")
    async def unregister_mcp_server(name: str):
        """Unregister an MCP server."""
        await mcp_manager.disconnect_server(name)
        success = mcp_manager.registry.unregister(name)
        return {"success": success, "server": name}

    @app.post("/mcp/servers/{name}/enable")
    async def enable_mcp_server(name: str):
        """Enable an MCP server."""
        success = mcp_manager.registry.enable_server(name)
        return {"success": success, "server": name}

    @app.post("/mcp/servers/{name}/disable")
    async def disable_mcp_server(name: str):
        """Disable an MCP server."""
        success = mcp_manager.registry.disable_server(name)
        return {"success": success, "server": name}

    # === Tool Routes ===

    @app.get("/mcp/tools")
    async def list_mcp_tools(server: Optional[str] = None):
        """List all MCP tools (optionally filter by server)."""
        all_tools = mcp_manager.list_all_tools()

        if server:
            tools = all_tools.get(server, [])
            return {
                "server": server,
                "tools": [t.to_dict() for t in tools],
            }

        return {
            "tools": {
                name: [t.to_dict() for t in tools]
                for name, tools in all_tools.items()
            }
        }

    @app.get("/mcp/tools/flat")
    async def list_mcp_tools_flat():
        """List all MCP tools as a flat list."""
        tools = mcp_manager.get_tools_flat()
        return {
            "tools": [
                {
                    "server": server_name,
                    "name": tool.name,
                    "description": tool.description,
                    "full_name": f"{server_name}:{tool.name}",
                }
                for server_name, tool in tools
            ]
        }

    @app.post("/mcp/tools/call")
    async def call_mcp_tool(request: MCPToolCallRequest):
        """Call an MCP tool."""
        try:
            result = await mcp_manager.call_tool(
                request.server,
                request.tool,
                request.arguments,
            )
            return result.to_dict()
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/mcp/tools/call-by-name")
    async def call_mcp_tool_by_name(
        full_name: str = Body(..., description="Full tool name (server:tool or just tool)"),
        arguments: dict = Body(default_factory=dict),
    ):
        """Call an MCP tool by full name."""
        try:
            result = await mcp_manager.call_tool_by_name(full_name, arguments)
            return result.to_dict()
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    # === Resource Routes ===

    @app.get("/mcp/resources")
    async def list_mcp_resources(server: Optional[str] = None):
        """List all MCP resources."""
        all_resources = mcp_manager.list_all_resources()

        if server:
            resources = all_resources.get(server, [])
            return {
                "server": server,
                "resources": [r.to_dict() for r in resources],
            }

        return {
            "resources": {
                name: [r.to_dict() for r in resources]
                for name, resources in all_resources.items()
            }
        }

    @app.post("/mcp/resources/read")
    async def read_mcp_resource(request: MCPResourceReadRequest):
        """Read an MCP resource."""
        try:
            content = await mcp_manager.read_resource(request.server, request.uri)
            return content.to_dict()
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    # === Prompt Routes ===

    @app.get("/mcp/prompts")
    async def list_mcp_prompts(server: Optional[str] = None):
        """List all MCP prompts."""
        all_prompts = mcp_manager.list_all_prompts()

        if server:
            prompts = all_prompts.get(server, [])
            return {
                "server": server,
                "prompts": [p.to_dict() for p in prompts],
            }

        return {
            "prompts": {
                name: [p.to_dict() for p in prompts]
                for name, prompts in all_prompts.items()
            }
        }

    @app.post("/mcp/prompts/get")
    async def get_mcp_prompt(request: MCPPromptGetRequest):
        """Get an MCP prompt."""
        try:
            messages = await mcp_manager.get_prompt(
                request.server,
                request.name,
                request.arguments,
            )
            return {
                "messages": [m.to_dict() for m in messages],
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    # === Statistics ===

    @app.get("/mcp/stats")
    async def mcp_stats():
        """Get MCP manager statistics."""
        return mcp_manager.get_stats()


# ==================== Conversation Routes ====================

def setup_conversation_routes(app: FastAPI, kernel) -> None:
    """
    Setup routes for the Conversation Interface.

    The conversation system provides natural language interaction with all AION
    capabilities including memory, tools, planning, and vision.
    """
    try:
        from aion.conversation.transports.rest import (
            create_conversation_router,
            create_health_router,
        )
        from aion.conversation.transports.websocket import create_websocket_router

        conversation_available = True
    except ImportError:
        conversation_available = False
        logger.warning("Conversation module not available")
        return

    if not kernel.conversation:
        logger.warning("Conversation manager not initialized, skipping routes")
        return

    # Create and include REST API router
    conversation_router = create_conversation_router(kernel.conversation)
    app.include_router(conversation_router, prefix="/api/v1")

    # Create and include WebSocket router
    ws_router = create_websocket_router(kernel.conversation)
    app.include_router(ws_router, prefix="/api/v1")

    # Add conversation stats to main stats endpoint
    @app.get("/conversation/stats")
    async def conversation_stats():
        """Get conversation system statistics."""
        return kernel.get_conversation_stats()

    @app.get("/conversation/health")
    async def conversation_health():
        """Get conversation system health."""
        if not kernel.conversation:
            return {"status": "unavailable"}

        return {
            "status": "ready" if kernel.conversation.is_initialized else "initializing",
            "active_sessions": kernel.conversation.sessions.active_count(),
        }

    logger.info("Conversation routes initialized")


# ==================== Goal System Routes ====================

class GoalCreateRequest(BaseModel):
    """Request to create a goal."""
    title: str = Field(..., description="Goal title")
    description: str = Field(..., description="Goal description")
    success_criteria: list[str] = Field(default_factory=list, description="Success criteria")
    priority: str = Field(default="medium", description="Priority level")
    goal_type: str = Field(default="achievement", description="Type of goal")
    deadline: Optional[datetime] = Field(default=None, description="Optional deadline")
    tags: list[str] = Field(default_factory=list, description="Tags")
    auto_decompose: bool = Field(default=False, description="Auto-decompose into subgoals")


class GoalAbandonRequest(BaseModel):
    """Request to abandon a goal."""
    reason: str = Field(..., description="Reason for abandonment")


class ApprovalActionRequest(BaseModel):
    """Request to approve or deny a safety request."""
    approver: str = Field(..., description="Approver identifier")
    reason: str = Field(default="", description="Optional reason")


class ObjectiveCreateRequest(BaseModel):
    """Request to create an objective."""
    name: str = Field(..., description="Objective name")
    description: str = Field(..., description="Objective description")
    rationale: str = Field(default="", description="Why this objective matters")


def setup_goal_routes(app: FastAPI, kernel) -> None:
    """Setup routes for the Autonomous Goal System."""

    # Check if goal system is available
    if not kernel.goals:
        logger.warning("Goal system not initialized, skipping routes")
        return

    goal_manager = kernel.goals

    # === Goal CRUD ===

    @app.get("/goals")
    async def list_goals(
        status: Optional[str] = Query(default=None, description="Filter by status"),
        priority: Optional[str] = Query(default=None, description="Filter by priority"),
        goal_type: Optional[str] = Query(default=None, description="Filter by type"),
        tag: Optional[str] = Query(default=None, description="Filter by tag"),
        limit: int = Query(default=50, ge=1, le=200),
    ):
        """List goals with optional filtering."""
        from aion.systems.goals.types import GoalStatus, GoalPriority, GoalType

        goals = await goal_manager.registry.get_all()

        if status:
            try:
                status_enum = GoalStatus(status)
                goals = [g for g in goals if g.status == status_enum]
            except ValueError:
                pass

        if priority:
            try:
                priority_enum = GoalPriority[priority.upper()]
                goals = [g for g in goals if g.priority == priority_enum]
            except (KeyError, ValueError):
                pass

        if goal_type:
            try:
                type_enum = GoalType(goal_type)
                goals = [g for g in goals if g.goal_type == type_enum]
            except ValueError:
                pass

        if tag:
            goals = [g for g in goals if tag in g.tags]

        return {"goals": [g.to_dict() for g in goals[:limit]]}

    @app.post("/goals")
    async def create_goal(request: GoalCreateRequest):
        """Create a new goal."""
        try:
            goal = await goal_manager.submit_goal(
                title=request.title,
                description=request.description,
                success_criteria=request.success_criteria,
                priority=request.priority,
                goal_type=request.goal_type,
                deadline=request.deadline,
                tags=request.tags,
                auto_decompose=request.auto_decompose,
            )
            return {"goal": goal.to_dict()}
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.get("/goals/{goal_id}")
    async def get_goal(goal_id: str):
        """Get goal details."""
        goal = await goal_manager.get_goal(goal_id)
        if not goal:
            raise HTTPException(status_code=404, detail="Goal not found")
        return {"goal": goal.to_dict()}

    @app.get("/goals/{goal_id}/progress")
    async def get_goal_progress(goal_id: str):
        """Get detailed goal progress."""
        progress = await goal_manager.get_progress(goal_id)
        if not progress:
            raise HTTPException(status_code=404, detail="Goal not found")
        return progress

    @app.get("/goals/{goal_id}/health")
    async def get_goal_health(goal_id: str):
        """Get goal health status."""
        health = await goal_manager.get_health(goal_id)
        if "error" in health:
            raise HTTPException(status_code=404, detail=health["error"])
        return health

    @app.post("/goals/{goal_id}/decompose")
    async def decompose_goal(goal_id: str):
        """Decompose a goal into subgoals."""
        try:
            subgoals = await goal_manager.decompose_goal(goal_id)
            return {"subgoals": [g.to_dict() for g in subgoals]}
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))

    @app.post("/goals/{goal_id}/pause")
    async def pause_goal(goal_id: str):
        """Pause a goal."""
        success = await goal_manager.pause_goal(goal_id)
        return {"success": success}

    @app.post("/goals/{goal_id}/resume")
    async def resume_goal(goal_id: str):
        """Resume a paused goal."""
        success = await goal_manager.resume_goal(goal_id)
        return {"success": success}

    @app.post("/goals/{goal_id}/abandon")
    async def abandon_goal(goal_id: str, request: GoalAbandonRequest):
        """Abandon a goal."""
        success = await goal_manager.abandon_goal(goal_id, request.reason)
        return {"success": success}

    # === Goal Generation ===

    @app.post("/goals/generate")
    async def generate_goals(max_goals: int = Query(default=2, ge=1, le=5)):
        """Manually trigger goal generation."""
        goals = await goal_manager.generate_goals(max_goals=max_goals)
        return {"generated": [g.to_dict() for g in goals]}

    @app.post("/goals/prioritize")
    async def prioritize_goals():
        """Reprioritize all pending goals."""
        prioritized = await goal_manager.prioritize_goals()
        return {"prioritized": [g.to_dict() for g in prioritized]}

    # === Objectives ===

    @app.get("/goals/objectives")
    async def list_objectives():
        """List all objectives."""
        objectives = await goal_manager.get_objectives()
        return {"objectives": [o.to_dict() for o in objectives]}

    @app.post("/goals/objectives")
    async def create_objective(request: ObjectiveCreateRequest):
        """Create a new objective."""
        objective = await goal_manager.create_objective(
            name=request.name,
            description=request.description,
            rationale=request.rationale,
        )
        return {"objective": objective.to_dict()}

    # === Safety ===

    @app.get("/goals/safety/approvals")
    async def get_pending_approvals():
        """Get pending safety approval requests."""
        approvals = goal_manager.get_pending_approvals()
        return {"approvals": [a.to_dict() for a in approvals]}

    @app.post("/goals/safety/approve/{request_id}")
    async def approve_request(request_id: str, request: ApprovalActionRequest):
        """Approve a safety request."""
        success = await goal_manager.approve_action(request_id, request.approver)
        return {"success": success}

    @app.post("/goals/safety/deny/{request_id}")
    async def deny_request(request_id: str, request: ApprovalActionRequest):
        """Deny a safety request."""
        success = await goal_manager.deny_action(request_id, request.approver, request.reason)
        return {"success": success}

    @app.post("/goals/safety/emergency-stop")
    async def emergency_stop(reason: str = Body(default="Manual emergency stop", embed=True)):
        """Activate emergency stop for the goal system."""
        goal_manager.emergency_stop(reason)
        return {"status": "emergency_stop_activated", "reason": reason}

    @app.post("/goals/safety/clear-emergency-stop")
    async def clear_emergency_stop():
        """Clear emergency stop."""
        goal_manager.clear_emergency_stop()
        return {"status": "emergency_stop_cleared"}

    # === Statistics ===

    @app.get("/goals/stats")
    async def get_goal_stats():
        """Get comprehensive goal system statistics."""
        return goal_manager.get_stats()

    @app.get("/goals/health")
    async def get_system_health():
        """Get overall goal system health."""
        return await goal_manager.get_system_health()

    # === Events ===

    @app.get("/goals/events")
    async def get_goal_events(
        goal_id: Optional[str] = Query(default=None),
        event_type: Optional[str] = Query(default=None),
        limit: int = Query(default=50, ge=1, le=200),
    ):
        """Get goal events."""
        events = await goal_manager.registry.get_events(
            goal_id=goal_id,
            event_type=event_type,
            limit=limit,
        )
        return {"events": [e.to_dict() for e in events]}

    logger.info("Goal routes initialized")
