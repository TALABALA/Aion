"""
AION Multi-Agent Reasoning Systems

State-of-the-art reasoning capabilities for agents including:
- Tree-of-Thought reasoning
- Chain-of-Thought prompting
- Self-reflection and critique
- Metacognitive monitoring
- Analogical reasoning
"""

from .tree_of_thought import TreeOfThought, ThoughtNode, ThoughtTree, ToTConfig
from .chain_of_thought import ChainOfThought, ReasoningStep, CoTConfig
from .reflection import SelfReflection, ReflectionResult, CritiqueType
from .metacognition import MetacognitiveMonitor, CognitiveState, ConfidenceEstimate
from .analogical import AnalogicalReasoner, Analogy, StructuralMapping

__all__ = [
    # Tree of Thought
    "TreeOfThought",
    "ThoughtNode",
    "ThoughtTree",
    "ToTConfig",
    # Chain of Thought
    "ChainOfThought",
    "ReasoningStep",
    "CoTConfig",
    # Reflection
    "SelfReflection",
    "ReflectionResult",
    "CritiqueType",
    # Metacognition
    "MetacognitiveMonitor",
    "CognitiveState",
    "ConfidenceEstimate",
    # Analogical
    "AnalogicalReasoner",
    "Analogy",
    "StructuralMapping",
]
