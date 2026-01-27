"""
AION Reinforcement Learning Loop

System-wide adaptive learning through reward-driven optimization.
Enables AION to learn from user feedback, outcome signals, and
implicit behavioural cues — continuously improving tool selection,
planning strategies, and agent behaviors.

Key Components:
- ReinforcementLearningLoop: Main coordinator
- RewardCollector: Multi-source reward aggregation
- ExperienceBuffer: Prioritized experience replay
- PolicyOptimizer: Background policy training
- ThompsonSampling / UCB1 / LinUCB: Bandit algorithms
- ABTestingFramework: Experiment management
- LearningConfig: Centralized configuration
"""

from aion.learning.loop import ReinforcementLearningLoop
from aion.learning.config import LearningConfig
from aion.learning.types import (
    Action,
    ActionType,
    ArmStatistics,
    Experience,
    Experiment,
    ExperimentStatus,
    ExperimentVariant,
    PolicyConfig,
    RewardSignal,
    RewardSource,
    RewardType,
    StateRepresentation,
)
from aion.learning.rewards.collector import RewardCollector
from aion.learning.experience.buffer import ExperienceBuffer
from aion.learning.policies.optimizer import PolicyOptimizer
from aion.learning.policies.base import BasePolicy
from aion.learning.policies.tool_policy import ToolSelectionPolicy
from aion.learning.policies.planning_policy import PlanningStrategyPolicy
from aion.learning.policies.agent_policy import AgentBehaviorPolicy
from aion.learning.bandits.thompson import ThompsonSampling, ContextualThompsonSampling
from aion.learning.bandits.ucb import UCB1, SlidingWindowUCB
from aion.learning.bandits.contextual import LinUCB, HybridLinUCB
from aion.learning.experiments.framework import ABTestingFramework
from aion.learning.integration.evolution import EvolutionIntegration
from aion.learning.integration.tools import ToolLearningIntegration
from aion.learning.persistence.repository import LearningStateRepository

__all__ = [
    # Core
    "ReinforcementLearningLoop",
    "LearningConfig",
    # Types
    "Action",
    "ActionType",
    "ArmStatistics",
    "Experience",
    "Experiment",
    "ExperimentStatus",
    "ExperimentVariant",
    "PolicyConfig",
    "RewardSignal",
    "RewardSource",
    "RewardType",
    "StateRepresentation",
    # Rewards
    "RewardCollector",
    # Experience
    "ExperienceBuffer",
    # Policies
    "PolicyOptimizer",
    "BasePolicy",
    "ToolSelectionPolicy",
    "PlanningStrategyPolicy",
    "AgentBehaviorPolicy",
    # Bandits
    "ThompsonSampling",
    "ContextualThompsonSampling",
    "UCB1",
    "SlidingWindowUCB",
    "LinUCB",
    "HybridLinUCB",
    # Experiments
    "ABTestingFramework",
    # Integration
    "EvolutionIntegration",
    "ToolLearningIntegration",
    # Persistence
    "LearningStateRepository",
]
