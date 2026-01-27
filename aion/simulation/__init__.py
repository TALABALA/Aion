"""AION Simulation Environment.

Provides sandboxed virtual worlds for testing, training, and evaluating
AI agents safely before deployment.

Core Components:
- SimulationEnvironment: Main coordinator for simulation lifecycle.
- SimulationAPI: High-level ergonomic API.
- WorldEngine: Entity/state/event management with ECS pattern.
- ScenarioGenerator: Procedural and template-based scenario creation.
- AgentSandbox: Isolated agent execution with tool mocking.
- TimelineManager: Snapshots, branching, and deterministic replay.
- SimulationEvaluator: Statistical evaluation with A/B comparison.
- AdversarialGenerator: Fuzzing, edge cases, and stress testing.

Quick Start:
    from aion.simulation import SimulationAPI, SimulationConfig

    api = SimulationAPI()
    result = await api.run_scenario("customer_support_basic")
    evaluation = await api.evaluate_result(result)
"""

from aion.simulation.types import (
    AgentInSimulation,
    Assertion,
    Constraint,
    ConstraintType,
    Entity,
    EntityType,
    EvaluationMetric,
    EventType,
    FuzzStrategy,
    Scenario,
    ScenarioType,
    SimulationConfig,
    SimulationEvent,
    SimulationResult,
    SimulationStatus,
    TimelineSnapshot,
    TimeMode,
    WorldState,
)
from aion.simulation.config import (
    SimulationEnvironmentConfig,
    default_config,
    test_config,
    performance_config,
)
from aion.simulation.environment import SimulationEnvironment
from aion.simulation.api import SimulationAPI

__all__ = [
    # Main classes
    "SimulationEnvironment",
    "SimulationAPI",
    # Config
    "SimulationConfig",
    "SimulationEnvironmentConfig",
    "default_config",
    "test_config",
    "performance_config",
    # Types
    "AgentInSimulation",
    "Assertion",
    "Constraint",
    "ConstraintType",
    "Entity",
    "EntityType",
    "EvaluationMetric",
    "EventType",
    "FuzzStrategy",
    "Scenario",
    "ScenarioType",
    "SimulationEvent",
    "SimulationResult",
    "SimulationStatus",
    "TimelineSnapshot",
    "TimeMode",
    "WorldState",
]
