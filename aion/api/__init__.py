"""AION API Module - FastAPI routes for all subsystems."""

from aion.api.routes import (
    setup_routes,
    setup_planning_routes,
    setup_memory_routes,
    setup_tool_routes,
    setup_evolution_routes,
    setup_vision_routes,
)

__all__ = [
    "setup_routes",
    "setup_planning_routes",
    "setup_memory_routes",
    "setup_tool_routes",
    "setup_evolution_routes",
    "setup_vision_routes",
]
