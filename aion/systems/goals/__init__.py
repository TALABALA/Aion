"""
AION Autonomous Goal System

The cognitive crown jewel that transforms AION from a reactive assistant
into a proactive AGI-level system capable of setting, pursuing, and achieving
goals autonomously over extended timeframes.

Core Components:
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

SOTA Components:
- GoalLearningSystem: Neural goal evaluation and learned components
- UncertaintyQuantifier: Bayesian reasoning and confidence estimation
- WorldModel: Outcome simulation and planning
- MetaLearningSystem: Adaptive strategies and transfer learning
- FormalVerificationSystem: Provable safety guarantees
- MultiAgentCoordinator: Distributed goal pursuit
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

# SOTA Components
from aion.systems.goals.learning import (
    GoalLearningSystem,
    GoalEmbedding,
    GoalOutcome,
    NeuralGoalEncoder,
    SuccessPredictor,
    AdaptivePriorityLearner,
)
from aion.systems.goals.uncertainty import (
    UncertaintyQuantifier,
    UncertaintyEstimate,
    BayesianGoalEstimator,
    ThompsonSampler,
    ConfidenceCalibrator,
)
from aion.systems.goals.world_model import (
    WorldModel,
    WorldState,
    StateVariable,
    Action,
    Transition,
    TransitionModel,
    RewardModel,
    MonteCarloTreeSearch,
    CausalModel,
)
from aion.systems.goals.meta_learning import (
    MetaLearningSystem,
    Strategy,
    StrategyPortfolio,
    MAMLAdapter,
    CurriculumLearner,
    TransferLearner,
    HyperparameterTuner,
)
from aion.systems.goals.formal_verification import (
    FormalVerificationSystem,
    Formula,
    SafetyProperty,
    VerificationResult,
    Contract,
    ModelChecker,
    RuntimeMonitor,
    BoundsChecker,
    InvariantChecker,
    SafetyShield,
)
from aion.systems.goals.multi_agent import (
    MultiAgentCoordinator,
    Agent,
    AgentRole,
    AgentStatus,
    AgentCapability,
    Message,
    MessageType,
    AgentRegistry,
    AuctionAllocator,
    ConsensusProtocol,
    ConflictResolver,
    CoalitionFormation,
)

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
    # Core Components
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
    # SOTA - Learning
    "GoalLearningSystem",
    "GoalEmbedding",
    "GoalOutcome",
    "NeuralGoalEncoder",
    "SuccessPredictor",
    "AdaptivePriorityLearner",
    # SOTA - Uncertainty
    "UncertaintyQuantifier",
    "UncertaintyEstimate",
    "BayesianGoalEstimator",
    "ThompsonSampler",
    "ConfidenceCalibrator",
    # SOTA - World Model
    "WorldModel",
    "WorldState",
    "StateVariable",
    "Action",
    "Transition",
    "TransitionModel",
    "RewardModel",
    "MonteCarloTreeSearch",
    "CausalModel",
    # SOTA - Meta Learning
    "MetaLearningSystem",
    "Strategy",
    "StrategyPortfolio",
    "MAMLAdapter",
    "CurriculumLearner",
    "TransferLearner",
    "HyperparameterTuner",
    # SOTA - Formal Verification
    "FormalVerificationSystem",
    "Formula",
    "SafetyProperty",
    "VerificationResult",
    "Contract",
    "ModelChecker",
    "RuntimeMonitor",
    "BoundsChecker",
    "InvariantChecker",
    "SafetyShield",
    # SOTA - Multi Agent
    "MultiAgentCoordinator",
    "Agent",
    "AgentRole",
    "AgentStatus",
    "AgentCapability",
    "Message",
    "MessageType",
    "AgentRegistry",
    "AuctionAllocator",
    "ConsensusProtocol",
    "ConflictResolver",
    "CoalitionFormation",
]
