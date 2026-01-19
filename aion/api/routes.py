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


# ==================== Audio Request/Response Models ====================

class AudioTranscribeRequest(BaseModel):
    """Request to transcribe audio."""
    audio_url: Optional[str] = None
    language: Optional[str] = None
    enable_diarization: bool = True


class AudioSynthesizeRequest(BaseModel):
    """Request to synthesize speech."""
    text: str = Field(..., description="Text to synthesize")
    voice: Optional[str] = None
    language: str = "en"
    speed: float = 1.0


class AudioAnswerRequest(BaseModel):
    """Request to answer a question about audio."""
    audio_url: Optional[str] = None
    question: str = Field(..., description="Question about the audio")


class SpeakerRegisterRequest(BaseModel):
    """Request to register a speaker."""
    name: str = Field(..., description="Speaker name")


def setup_audio_routes(app: FastAPI, audio_cortex) -> None:
    """Setup routes for the Auditory Cortex system."""

    @app.post("/audio/transcribe")
    async def transcribe_audio(
        file: UploadFile = File(None),
        request: Optional[AudioTranscribeRequest] = None,
    ):
        """Transcribe audio to text with optional speaker diarization."""
        if file:
            audio_bytes = await file.read()
            transcript = await audio_cortex.transcribe(
                audio_bytes,
                language=request.language if request else None,
                enable_diarization=request.enable_diarization if request else True,
            )
        elif request and request.audio_url:
            transcript = await audio_cortex.transcribe(
                request.audio_url,
                language=request.language,
                enable_diarization=request.enable_diarization,
            )
        else:
            raise HTTPException(status_code=400, detail="Audio file or URL required")

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
            audio_bytes,
            language=language,
            enable_diarization=enable_diarization,
        )
        return {"transcript": transcript.to_dict()}

    @app.post("/audio/detect-events")
    async def detect_audio_events(
        file: UploadFile = File(None),
        audio_url: Optional[str] = Body(default=None),
        threshold: float = Body(default=0.5),
    ):
        """Detect events in audio (speech, music, environmental sounds)."""
        if file:
            audio_bytes = await file.read()
            events = await audio_cortex.detect_events(audio_bytes, threshold=threshold)
        elif audio_url:
            events = await audio_cortex.detect_events(audio_url, threshold=threshold)
        else:
            raise HTTPException(status_code=400, detail="Audio file or URL required")

        return {"events": [e.to_dict() for e in events]}

    @app.post("/audio/understand")
    async def understand_audio(
        file: UploadFile = File(None),
        audio_url: Optional[str] = Body(default=None),
    ):
        """Full audio scene understanding."""
        if file:
            audio_bytes = await file.read()
            scene = await audio_cortex.understand_scene(audio_bytes)
        elif audio_url:
            scene = await audio_cortex.understand_scene(audio_url)
        else:
            raise HTTPException(status_code=400, detail="Audio file or URL required")

        return {"scene": scene.to_dict()}

    @app.post("/audio/analyze")
    async def analyze_audio(
        file: UploadFile = File(None),
        audio_url: Optional[str] = Body(default=None),
        question: Optional[str] = Body(default=None),
        store_in_memory: bool = Body(default=True),
    ):
        """Full audio analysis with optional question."""
        if file:
            audio_bytes = await file.read()
            result = await audio_cortex.process(
                audio_bytes,
                query=question,
                store_in_memory=store_in_memory,
            )
        elif audio_url:
            result = await audio_cortex.process(
                audio_url,
                query=question,
                store_in_memory=store_in_memory,
            )
        else:
            raise HTTPException(status_code=400, detail="Audio file or URL required")

        return {"analysis": result.to_dict()}

    @app.post("/audio/synthesize")
    async def synthesize_speech(request: AudioSynthesizeRequest):
        """Generate speech from text."""
        audio_segment = await audio_cortex.synthesize(
            text=request.text,
            voice=request.voice,
            language=request.language,
            speed=request.speed,
        )

        return {
            "audio": audio_segment.to_dict(),
            "duration": audio_segment.duration,
        }

    @app.post("/audio/identify-speaker")
    async def identify_speaker(
        file: UploadFile = File(None),
        audio_url: Optional[str] = Body(default=None),
    ):
        """Identify speaker in audio."""
        if file:
            audio_bytes = await file.read()
            profile, confidence = await audio_cortex.identify_speaker(audio_bytes)
        elif audio_url:
            profile, confidence = await audio_cortex.identify_speaker(audio_url)
        else:
            raise HTTPException(status_code=400, detail="Audio file or URL required")

        return {
            "speaker": profile.to_dict() if profile else None,
            "confidence": confidence,
        }

    @app.post("/audio/register-speaker")
    async def register_speaker(
        file: UploadFile = File(...),
        name: str = Body(...),
    ):
        """Register a new speaker with audio sample."""
        audio_bytes = await file.read()
        profile = await audio_cortex.register_speaker(audio_bytes, name)
        return {"speaker": profile.to_dict()}

    @app.post("/audio/verify-speaker")
    async def verify_speaker(
        file: UploadFile = File(...),
        speaker_id: str = Body(...),
    ):
        """Verify if audio matches a registered speaker."""
        audio_bytes = await file.read()

        # Get speaker profile from memory
        if audio_cortex.memory:
            profile = audio_cortex.memory.get_voice_profile(speaker_id)
            if not profile:
                raise HTTPException(status_code=404, detail="Speaker not found")

            is_verified, confidence = await audio_cortex.verify_speaker(
                audio_bytes, profile
            )
            return {
                "verified": is_verified,
                "confidence": confidence,
            }
        else:
            raise HTTPException(status_code=400, detail="Memory not enabled")

    @app.post("/audio/answer")
    async def answer_audio_question(
        file: UploadFile = File(None),
        audio_url: Optional[str] = Body(default=None),
        question: str = Body(...),
    ):
        """Answer a question about audio content."""
        if file:
            audio_bytes = await file.read()
            answer = await audio_cortex.answer_question(audio_bytes, question)
        elif audio_url:
            answer = await audio_cortex.answer_question(audio_url, question)
        else:
            raise HTTPException(status_code=400, detail="Audio file or URL required")

        return {"answer": answer}

    @app.post("/audio/summarize")
    async def summarize_audio(
        file: UploadFile = File(None),
        audio_url: Optional[str] = Body(default=None),
    ):
        """Generate a summary of audio content."""
        if file:
            audio_bytes = await file.read()
            summary = await audio_cortex.summarize(audio_bytes)
        elif audio_url:
            summary = await audio_cortex.summarize(audio_url)
        else:
            raise HTTPException(status_code=400, detail="Audio file or URL required")

        return {"summary": summary}

    @app.post("/audio/compare")
    async def compare_audio(
        file1: UploadFile = File(None),
        file2: UploadFile = File(None),
        audio1_url: Optional[str] = Body(default=None),
        audio2_url: Optional[str] = Body(default=None),
    ):
        """Compare two audio samples."""
        # Get audio sources
        audio1 = await file1.read() if file1 else audio1_url
        audio2 = await file2.read() if file2 else audio2_url

        if not audio1 or not audio2:
            raise HTTPException(
                status_code=400,
                detail="Two audio files or URLs required",
            )

        comparison = await audio_cortex.compare(audio1, audio2)
        return {"comparison": comparison}

    @app.post("/audio/remember")
    async def remember_audio(
        file: UploadFile = File(...),
        context: Optional[str] = Body(default=None),
        importance: float = Body(default=0.5),
    ):
        """Store audio in memory."""
        audio_bytes = await file.read()
        memory_id = await audio_cortex.remember(
            audio_bytes,
            context=context,
            importance=importance,
        )
        return {"memory_id": memory_id}

    @app.post("/audio/recall")
    async def recall_audio(
        query: str = Body(...),
        limit: int = Body(default=5),
    ):
        """Recall similar audio from memory."""
        memories = await audio_cortex.recall_similar(query, limit=limit)
        return {"memories": [m.to_dict() for m in memories]}

    @app.get("/audio/speakers")
    async def list_speakers():
        """List all registered voice profiles."""
        if audio_cortex.memory:
            profiles = audio_cortex.memory.get_voice_profiles()
            return {"speakers": [p.to_dict() for p in profiles]}
        return {"speakers": []}

    @app.get("/audio/stats")
    async def audio_stats():
        """Get auditory cortex statistics."""
        return audio_cortex.get_stats()
