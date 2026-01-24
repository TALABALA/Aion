"""
AION Multi-Agent Workflows

Workflow patterns for coordinating multi-agent task execution.
"""

from aion.systems.agents.workflows.base import (
    WorkflowExecutor,
    get_workflow_executor,
)
from aion.systems.agents.workflows.sequential import SequentialWorkflow
from aion.systems.agents.workflows.parallel import ParallelWorkflow
from aion.systems.agents.workflows.hierarchical import HierarchicalWorkflow
from aion.systems.agents.workflows.debate import DebateWorkflow
from aion.systems.agents.workflows.swarm import SwarmWorkflow

__all__ = [
    "WorkflowExecutor",
    "get_workflow_executor",
    "SequentialWorkflow",
    "ParallelWorkflow",
    "HierarchicalWorkflow",
    "DebateWorkflow",
    "SwarmWorkflow",
]
