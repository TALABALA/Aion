"""
AION Agent Archetypes

Specialized agent implementations with domain-specific behavior.
"""

from aion.systems.agents.archetypes.base import BaseSpecialist
from aion.systems.agents.archetypes.researcher import ResearcherAgent
from aion.systems.agents.archetypes.coder import CoderAgent
from aion.systems.agents.archetypes.analyst import AnalystAgent
from aion.systems.agents.archetypes.writer import WriterAgent
from aion.systems.agents.archetypes.reviewer import ReviewerAgent
from aion.systems.agents.archetypes.planner import PlannerAgent
from aion.systems.agents.archetypes.executor import ExecutorAgent

__all__ = [
    "BaseSpecialist",
    "ResearcherAgent",
    "CoderAgent",
    "AnalystAgent",
    "WriterAgent",
    "ReviewerAgent",
    "PlannerAgent",
    "ExecutorAgent",
]
