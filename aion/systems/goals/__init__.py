"""
AION Autonomous Goal System

The cognitive crown jewel that transforms AION from a reactive assistant
into a proactive AGI-level system capable of setting, pursuing, and achieving
goals autonomously over extended timeframes.

Components:
- GoalReasoner: LLM-powered goal cognition
- GoalRegistry: Goal storage and retrieval
- GoalScheduler: Execution scheduling
- GoalExecutor: Plan execution
- GoalMonitor: Progress tracking
- GoalDecomposer: Break down complex goals
- GoalPrioritizer: Priority management
- SafetyBoundary: Constraints and limits
- ValueSystem: Core values and alignment
- GoalTriggers: Event-based goal activation
- AutonomousGoalManager: Central coordinator
"""

from aion.systems.goals.types import (
    Goal,
    GoalStatus,
    GoalPriority,
    GoalType,
    GoalSource,
    GoalConstraint,
    GoalMetrics,
    GoalEvent,
    GoalProposal,
    Objective,
    ValuePrinciple,
)
from aion.systems.goals.registry import GoalRegistry
from aion.systems.goals.reasoner import GoalReasoner
from aion.systems.goals.scheduler import GoalScheduler
from aion.systems.goals.executor import GoalExecutor
from aion.systems.goals.monitor import GoalMonitor
from aion.systems.goals.decomposer import GoalDecomposer
from aion.systems.goals.prioritizer import GoalPrioritizer
from aion.systems.goals.safety import SafetyBoundary, SafetyRule, ApprovalRequest
from aion.systems.goals.values import ValueSystem
from aion.systems.goals.triggers import GoalTriggers, TriggerCondition, TriggerAction
from aion.systems.goals.persistence import GoalPersistence
from aion.systems.goals.manager import AutonomousGoalManager

__all__ = [
    # Core types
    "Goal",
    "GoalStatus",
    "GoalPriority",
    "GoalType",
    "GoalSource",
    "GoalConstraint",
    "GoalMetrics",
    "GoalEvent",
    "GoalProposal",
    "Objective",
    "ValuePrinciple",
    # Components
    "GoalRegistry",
    "GoalReasoner",
    "GoalScheduler",
    "GoalExecutor",
    "GoalMonitor",
    "GoalDecomposer",
    "GoalPrioritizer",
    "SafetyBoundary",
    "SafetyRule",
    "ApprovalRequest",
    "ValueSystem",
    "GoalTriggers",
    "TriggerCondition",
    "TriggerAction",
    "GoalPersistence",
    # Main manager
    "AutonomousGoalManager",
]
