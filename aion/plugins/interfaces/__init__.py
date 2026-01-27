"""
AION Plugin Interfaces

Defines the contracts that plugins must implement.
"""

from aion.plugins.interfaces.base import BasePlugin
from aion.plugins.interfaces.tool import ToolPlugin
from aion.plugins.interfaces.agent import AgentPlugin
from aion.plugins.interfaces.storage import StoragePlugin
from aion.plugins.interfaces.workflow import WorkflowTriggerPlugin, WorkflowActionPlugin

__all__ = [
    "BasePlugin",
    "ToolPlugin",
    "AgentPlugin",
    "StoragePlugin",
    "WorkflowTriggerPlugin",
    "WorkflowActionPlugin",
]
