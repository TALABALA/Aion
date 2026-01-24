"""
AION Multi-Agent Learning Systems

State-of-the-art learning capabilities for agents including:
- Reinforcement learning from feedback
- Skill libraries and transfer
- In-context learning
- Meta-learning
- Continuous adaptation
"""

from .reinforcement import (
    ReinforcementLearner,
    Experience,
    Policy,
    ValueFunction,
    RLConfig,
)
from .skills import (
    SkillLibrary,
    Skill,
    SkillTemplate,
    SkillExecution,
)
from .adaptation import (
    ContinuousAdapter,
    AdaptationStrategy,
    PerformanceTracker,
)
from .meta import (
    MetaLearner,
    TaskDistribution,
    LearningCurve,
)

__all__ = [
    # Reinforcement Learning
    "ReinforcementLearner",
    "Experience",
    "Policy",
    "ValueFunction",
    "RLConfig",
    # Skills
    "SkillLibrary",
    "Skill",
    "SkillTemplate",
    "SkillExecution",
    # Adaptation
    "ContinuousAdapter",
    "AdaptationStrategy",
    "PerformanceTracker",
    # Meta-Learning
    "MetaLearner",
    "TaskDistribution",
    "LearningCurve",
]
