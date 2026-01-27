"""
AION Plugin Dependencies

Dependency resolution and management for plugins.

Includes:
- Topological sort-based resolver (fast, simple)
- SAT-based resolver (handles complex constraints like npm/pip)
"""

from aion.plugins.dependencies.resolver import DependencyResolver, ResolutionResult
from aion.plugins.dependencies.graph import DependencyGraph
from aion.plugins.dependencies.sat_solver import (
    SATDependencyResolver,
    DPLLSolver,
    SolutionSet,
    Clause,
    Conflict,
    ConflictType,
    PackageVersion,
    resolve_dependencies,
)

__all__ = [
    # Original resolver
    "DependencyResolver",
    "ResolutionResult",
    "DependencyGraph",

    # SAT-based resolver
    "SATDependencyResolver",
    "DPLLSolver",
    "SolutionSet",
    "Clause",
    "Conflict",
    "ConflictType",
    "PackageVersion",
    "resolve_dependencies",
]
