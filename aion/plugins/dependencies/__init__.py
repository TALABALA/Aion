"""
AION Plugin Dependencies

Dependency resolution and management for plugins.
"""

from aion.plugins.dependencies.resolver import DependencyResolver
from aion.plugins.dependencies.graph import DependencyGraph

__all__ = [
    "DependencyResolver",
    "DependencyGraph",
]
