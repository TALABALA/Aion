"""AION Reward Collection Subsystem."""

from .collector import RewardCollector
from .signals import SignalProcessor, SignalRegistry
from .explicit import ExplicitFeedbackProcessor
from .implicit import ImplicitSignalExtractor
from .shaping import RewardShaper, PotentialBasedShaping

__all__ = [
    "RewardCollector",
    "SignalProcessor",
    "SignalRegistry",
    "ExplicitFeedbackProcessor",
    "ImplicitSignalExtractor",
    "RewardShaper",
    "PotentialBasedShaping",
]
