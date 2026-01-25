"""
AION Agent Security Module

Permission boundaries and sandboxing for AI agents.
"""

from aion.security.agent_security.boundaries import (
    AgentBoundaryEnforcer,
    BoundaryViolation,
)

__all__ = [
    "AgentBoundaryEnforcer",
    "BoundaryViolation",
]
