"""
AION Natural Language Programming System.

Enables users to create tools, workflows, agents, APIs, and integrations
through natural language conversation. This is AION's core AGI capability
that eliminates the barrier between intention and implementation.

Usage:
    from aion.nlp import NLProgrammingEngine, NLProgrammingConfig

    engine = NLProgrammingEngine(kernel, config)
    result = await engine.process("Create a tool that fetches weather data")
"""

from aion.nlp.engine import NLProgrammingEngine
from aion.nlp.config import NLProgrammingConfig
from aion.nlp.types import (
    Intent,
    IntentType,
    Entity,
    EntityType,
    ToolSpecification,
    WorkflowSpecification,
    AgentSpecification,
    APISpecification,
    IntegrationSpecification,
    GeneratedCode,
    ValidationResult,
    DeployedSystem,
    ProgrammingSession,
    SpecificationType,
    DeploymentStatus,
    ValidationStatus,
    SafetyLevel,
    Complexity,
)

__all__ = [
    # Engine
    "NLProgrammingEngine",
    "NLProgrammingConfig",
    # Types
    "Intent",
    "IntentType",
    "Entity",
    "EntityType",
    "ToolSpecification",
    "WorkflowSpecification",
    "AgentSpecification",
    "APISpecification",
    "IntegrationSpecification",
    "GeneratedCode",
    "ValidationResult",
    "DeployedSystem",
    "ProgrammingSession",
    "SpecificationType",
    "DeploymentStatus",
    "ValidationStatus",
    "SafetyLevel",
    "Complexity",
]
