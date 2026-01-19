"""AION Core Module - Foundation components for the cognitive architecture."""

from aion.core.config import AIONConfig
from aion.core.kernel import AIONKernel
from aion.core.security import SecurityManager, ApprovalGate, RiskLevel
from aion.core.llm import LLMAdapter, LLMProvider

__all__ = [
    "AIONConfig",
    "AIONKernel",
    "SecurityManager",
    "ApprovalGate",
    "RiskLevel",
    "LLMAdapter",
    "LLMProvider",
]
