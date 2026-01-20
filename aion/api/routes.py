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
