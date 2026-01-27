"""
AION Reinforcement Learning Loop

System-wide adaptive learning through actor-critic optimization.
Enables AION to learn from user feedback, outcome signals, and
implicit behavioural cues — continuously improving tool selection,
planning strategies, and agent behaviors.

Architecture:
- Actor-Critic: Shared V(s) critic with Polyak-averaged target network,
  per-domain actor policies with advantage-based gradients (GAE)
- RND Curiosity: Random Network Distillation for intrinsic motivation
- Prioritized Replay: Sum-tree sampling with importance weight correction
- Multi-armed Bandits: Thompson Sampling, UCB1, LinUCB, Hybrid LinUCB
- A/B Testing: Sequential testing with alpha-spending functions
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
from aion.learning.rewards.rnd import RNDCuriosityShaper
from aion.learning.experience.buffer import ExperienceBuffer
from aion.learning.policies.optimizer import PolicyOptimizer
from aion.learning.policies.base import BasePolicy
from aion.learning.policies.tool_policy import ToolSelectionPolicy
from aion.learning.policies.planning_policy import PlanningStrategyPolicy
from aion.learning.policies.agent_policy import AgentBehaviorPolicy
from aion.learning.nn import MLP
from aion.learning.policies.value_function import TargetNetwork, StateValueFunction
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
    "RNDCuriosityShaper",
    # Experience
    "ExperienceBuffer",
    # Policies (Actor-Critic)
    "PolicyOptimizer",
    "BasePolicy",
    "ToolSelectionPolicy",
    "PlanningStrategyPolicy",
    "AgentBehaviorPolicy",
    "MLP",
    "TargetNetwork",
    "StateValueFunction",
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
