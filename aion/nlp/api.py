"""
AION NLP Programming API Routes.

FastAPI routes for the natural language programming system,
enabling programmatic and UI access to all NLP capabilities.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, HTTPException

from aion.nlp.engine import NLProgrammingEngine

router = APIRouter(prefix="/nlp", tags=["natural-language-programming"])


def setup_nlp_routes(app: Any, engine: NLProgrammingEngine) -> None:
    """Setup NLP programming API routes."""

    @router.post("/process")
    async def process_request(
        input: str = Body(..., embed=True, description="Natural language request"),
        session_id: Optional[str] = Body(None, description="Session ID for continuity"),
        user_id: str = Body("", description="User identifier"),
    ) -> Dict[str, Any]:
        """
        Process a natural language programming request.

        This is the main endpoint for creating tools, workflows,
        agents, APIs, and integrations through natural language.
        """
        return await engine.process(input, session_id, user_id)

    @router.post("/sessions/{session_id}/confirm")
    async def confirm_deployment(
        session_id: str,
        confirmed: bool = Body(True, description="Whether to proceed with deployment"),
    ) -> Dict[str, Any]:
        """Confirm or cancel deployment of generated code."""
        return await engine.confirm_deploy(session_id, confirmed)

    @router.post("/sessions/{session_id}/refine")
    async def refine_system(
        session_id: str,
        feedback: str = Body(..., embed=True, description="Feedback for refinement"),
    ) -> Dict[str, Any]:
        """Refine the current system based on feedback."""
        return await engine.refine(session_id, feedback)

    @router.get("/sessions/{session_id}")
    async def get_session(session_id: str) -> Dict[str, Any]:
        """Get session details."""
        session = engine.get_session(session_id)
        if not session:
            raise HTTPException(404, "Session not found")
        return {
            "id": session.id,
            "user_id": session.user_id,
            "state": session.state,
            "iterations": session.iterations,
            "messages": [m.to_dict() for m in session.messages[-20:]],
            "current_intent": session.current_intent.type.value if session.current_intent else None,
            "current_spec": (
                session.current_spec.to_dict()
                if session.current_spec and hasattr(session.current_spec, "to_dict")
                else None
            ),
            "has_code": session.current_code is not None,
            "referenced_systems": session.referenced_systems,
            "started_at": session.started_at.isoformat(),
            "last_activity": session.last_activity.isoformat(),
            "duration_seconds": session.duration_seconds,
        }

    @router.get("/deployed")
    async def list_deployed() -> List[Dict[str, Any]]:
        """List all deployed systems."""
        return [
            {
                "id": d.id,
                "name": d.name,
                "type": d.system_type.value,
                "status": d.status.value,
                "version": d.version,
                "invocations": d.invocation_count,
                "error_count": d.error_count,
                "error_rate": round(d.error_rate, 4),
                "avg_latency_ms": round(d.avg_latency_ms, 2),
                "created_at": d.created_at.isoformat(),
                "updated_at": d.updated_at.isoformat(),
                "created_by": d.created_by,
                "tags": d.tags,
            }
            for d in engine.list_deployed()
        ]

    @router.get("/deployed/{system_id}")
    async def get_deployed_system(system_id: str) -> Dict[str, Any]:
        """Get details of a deployed system."""
        system = engine.deployer.get_deployed(system_id)
        if not system:
            raise HTTPException(404, "System not found")
        return {
            "id": system.id,
            "name": system.name,
            "type": system.system_type.value,
            "status": system.status.value,
            "version": system.version,
            "specification": (
                system.specification.to_dict()
                if system.specification and hasattr(system.specification, "to_dict")
                else None
            ),
            "code": system.generated_code.code if system.generated_code else None,
            "invocations": system.invocation_count,
            "error_count": system.error_count,
            "error_rate": round(system.error_rate, 4),
            "created_at": system.created_at.isoformat(),
            "deployment_history": [
                {
                    "version": r.version,
                    "deployed_at": r.deployed_at.isoformat(),
                    "deployed_by": r.deployed_by,
                    "summary": r.change_summary,
                }
                for r in system.deployment_history
            ],
        }

    @router.delete("/deployed/{system_id}")
    async def undeploy_system(system_id: str) -> Dict[str, Any]:
        """Undeploy a system."""
        success = await engine.deployer.undeploy(system_id)
        if not success:
            raise HTTPException(404, "System not found or undeploy failed")
        return {"status": "undeployed", "system_id": system_id}

    @router.post("/deployed/{system_id}/rollback")
    async def rollback_system(
        system_id: str,
        target_version: Optional[int] = Body(None, description="Target version (default: previous)"),
    ) -> Dict[str, Any]:
        """Rollback a deployed system to a previous version."""
        success = await engine.deployer.rollback(system_id, target_version)
        if not success:
            raise HTTPException(400, "Rollback failed")
        return {"status": "rolled_back", "system_id": system_id}

    @router.get("/stats")
    async def get_stats() -> Dict[str, Any]:
        """Get NLP programming engine statistics."""
        return engine.get_stats()

    app.include_router(router)
