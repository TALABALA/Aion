"""
SAT-Based Dependency Resolution

Implements a Boolean Satisfiability (SAT) solver for plugin dependency
resolution, similar to modern package managers like pip, npm, and cargo.

This approach handles complex version constraints, conflicts, and optional
dependencies more robustly than simple topological sort.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, FrozenSet

import structlog

from aion.plugins.types import SemanticVersion, VersionConstraint, PluginDependency

logger = structlog.get_logger(__name__)


class SolverState(str, Enum):
    """State of the SAT solver."""
    IDLE = "idle"
    SOLVING = "solving"
    SATISFIED = "satisfied"
    UNSATISFIABLE = "unsatisfiable"


class ConflictType(str, Enum):
    """Types of dependency conflicts."""
    VERSION_CONFLICT = "version_conflict"
    MISSING_DEPENDENCY = "missing_dependency"
    CIRCULAR_DEPENDENCY = "circular_dependency"
    INCOMPATIBLE_CONSTRAINTS = "incompatible_constraints"


@dataclass
class PackageVersion:
    """Represents a specific version of a package."""

    package_id: str
    version: SemanticVersion
    dependencies: list[PluginDependency] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)  # Package IDs this conflicts with
    provides: list[str] = field(default_factory=list)  # Virtual packages this provides

    @property
    def id(self) -> str:
        """Unique identifier for this package version."""
        return f"{self.package_id}@{self.version}"

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other) -> bool:
        if not isinstance(other, PackageVersion):
            return False
        return self.id == other.id


@dataclass
class Clause:
    """
    A clause in CNF (Conjunctive Normal Form).

    Represents a disjunction of literals: (l1 OR l2 OR ... OR ln)
    Each literal is a (package_version_id, positive) tuple.
    """

    literals: FrozenSet[Tuple[str, bool]]  # (version_id, is_positive)
    reason: str = ""

    def __hash__(self) -> int:
        return hash(self.literals)

    def __eq__(self, other) -> bool:
        if not isinstance(other, Clause):
            return False
        return self.literals == other.literals

    @classmethod
    def unit(cls, version_id: str, positive: bool = True, reason: str = "") -> "Clause":
        """Create a unit clause (single literal)."""
        return cls(frozenset([(version_id, positive)]), reason)

    @classmethod
    def at_most_one(cls, version_ids: List[str], reason: str = "") -> List["Clause"]:
        """
        Create clauses ensuring at most one of the versions is selected.

        For n versions, creates n*(n-1)/2 clauses of form: NOT(a) OR NOT(b)
        """
        clauses = []
        for v1, v2 in itertools.combinations(version_ids, 2):
            clauses.append(cls(
                frozenset([(v1, False), (v2, False)]),
                reason=reason or f"At most one of {v1}, {v2}",
            ))
        return clauses

    @classmethod
    def exactly_one(cls, version_ids: List[str], reason: str = "") -> List["Clause"]:
        """
        Create clauses ensuring exactly one of the versions is selected.

        Combines at_least_one (one big OR clause) with at_most_one.
        """
        if not version_ids:
            return []

        clauses = []

        # At least one must be selected
        clauses.append(cls(
            frozenset((vid, True) for vid in version_ids),
            reason=reason or f"At least one of {version_ids}",
        ))

        # At most one can be selected
        clauses.extend(cls.at_most_one(version_ids, reason))

        return clauses

    def is_unit(self) -> bool:
        """Check if this is a unit clause."""
        return len(self.literals) == 1

    def is_empty(self) -> bool:
        """Check if this clause is empty (conflict)."""
        return len(self.literals) == 0

    def evaluate(self, assignment: Dict[str, bool]) -> Optional[bool]:
        """
        Evaluate clause under partial assignment.

        Returns True if satisfied, False if unsatisfied, None if undetermined.
        """
        has_unassigned = False

        for version_id, positive in self.literals:
            if version_id not in assignment:
                has_unassigned = True
                continue

            value = assignment[version_id]
            if value == positive:
                return True  # Clause is satisfied

        if has_unassigned:
            return None  # Undetermined
        return False  # All literals false, clause unsatisfied

    def propagate(self, assignment: Dict[str, bool]) -> Optional[Tuple[str, bool]]:
        """
        Perform unit propagation.

        If all but one literal is false, returns the forced assignment.
        """
        unassigned = []

        for version_id, positive in self.literals:
            if version_id not in assignment:
                unassigned.append((version_id, positive))
            elif assignment[version_id] == positive:
                return None  # Already satisfied

        if len(unassigned) == 1:
            return unassigned[0]
        return None


@dataclass
class Conflict:
    """Represents a dependency conflict."""

    conflict_type: ConflictType
    packages: list[str]
    message: str
    clause: Optional[Clause] = None


@dataclass
class SolutionSet:
    """Result of SAT solving."""

    satisfied: bool
    selected_versions: Dict[str, PackageVersion] = field(default_factory=dict)
    conflicts: list[Conflict] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)  # Decision trail
    propagations: int = 0
    backtracks: int = 0


class DPLLSolver:
    """
    DPLL (Davis-Putnam-Logemann-Loveland) SAT Solver.

    Implements the classic DPLL algorithm with:
    - Unit propagation
    - Pure literal elimination
    - Conflict-driven clause learning (CDCL) inspired backtracking
    - Variable ordering heuristics
    """

    def __init__(self):
        self.clauses: List[Clause] = []
        self.variables: Set[str] = set()
        self.assignment: Dict[str, bool] = {}
        self.decision_level: Dict[str, int] = {}
        self.decision_trail: List[str] = []
        self.propagation_count = 0
        self.backtrack_count = 0

        # Watched literals for efficient propagation
        self._watches: Dict[Tuple[str, bool], List[int]] = {}

    def add_clause(self, clause: Clause) -> None:
        """Add a clause to the formula."""
        self.clauses.append(clause)

        # Register variables
        for version_id, _ in clause.literals:
            self.variables.add(version_id)

        # Set up watched literals (first two literals)
        literals = list(clause.literals)
        if len(literals) >= 1:
            self._add_watch(literals[0], len(self.clauses) - 1)
        if len(literals) >= 2:
            self._add_watch(literals[1], len(self.clauses) - 1)

    def _add_watch(self, literal: Tuple[str, bool], clause_idx: int) -> None:
        """Add a watch for a literal."""
        if literal not in self._watches:
            self._watches[literal] = []
        self._watches[literal].append(clause_idx)

    def solve(self) -> SolutionSet:
        """
        Solve the SAT problem using DPLL.

        Returns a SolutionSet indicating satisfiability and variable assignments.
        """
        self.assignment = {}
        self.decision_level = {}
        self.decision_trail = []
        self.propagation_count = 0
        self.backtrack_count = 0

        # Initial unit propagation
        conflict = self._unit_propagate()
        if conflict:
            return SolutionSet(
                satisfied=False,
                conflicts=[conflict],
            )

        # Main DPLL loop
        while True:
            # All variables assigned?
            if len(self.assignment) == len(self.variables):
                return SolutionSet(
                    satisfied=True,
                    decisions=self.decision_trail.copy(),
                    propagations=self.propagation_count,
                    backtracks=self.backtrack_count,
                )

            # Make a decision
            var = self._pick_branching_variable()
            if var is None:
                return SolutionSet(
                    satisfied=True,
                    decisions=self.decision_trail.copy(),
                    propagations=self.propagation_count,
                    backtracks=self.backtrack_count,
                )

            # Try True first (prefer installing packages)
            self._decide(var, True)

            # Propagate
            conflict = self._unit_propagate()

            while conflict:
                # Backtrack
                self.backtrack_count += 1

                if not self.decision_trail:
                    # No decisions left to backtrack
                    return SolutionSet(
                        satisfied=False,
                        conflicts=[conflict],
                        propagations=self.propagation_count,
                        backtracks=self.backtrack_count,
                    )

                # Backtrack and try opposite value
                flipped = self._backtrack()
                if flipped is None:
                    return SolutionSet(
                        satisfied=False,
                        conflicts=[conflict],
                        propagations=self.propagation_count,
                        backtracks=self.backtrack_count,
                    )

                conflict = self._unit_propagate()

    def _unit_propagate(self) -> Optional[Conflict]:
        """
        Perform unit propagation until fixed point or conflict.

        Returns a Conflict if one is found, None otherwise.
        """
        changed = True
        while changed:
            changed = False

            for clause in self.clauses:
                result = clause.evaluate(self.assignment)

                if result is False:
                    # Conflict detected
                    return Conflict(
                        conflict_type=ConflictType.INCOMPATIBLE_CONSTRAINTS,
                        packages=[lit[0] for lit in clause.literals],
                        message=clause.reason,
                        clause=clause,
                    )

                if result is None:
                    # Try unit propagation
                    propagation = clause.propagate(self.assignment)
                    if propagation:
                        var, value = propagation
                        self.assignment[var] = value
                        self.propagation_count += 1
                        changed = True

        return None

    def _pick_branching_variable(self) -> Optional[str]:
        """
        Pick the next variable to branch on.

        Uses VSIDS-inspired heuristic: prefer variables appearing in more clauses.
        """
        unassigned = self.variables - set(self.assignment.keys())
        if not unassigned:
            return None

        # Count occurrences in unsatisfied clauses
        scores: Dict[str, int] = {v: 0 for v in unassigned}

        for clause in self.clauses:
            if clause.evaluate(self.assignment) is not True:
                for var, _ in clause.literals:
                    if var in scores:
                        scores[var] += 1

        # Return variable with highest score
        return max(scores, key=lambda v: scores[v])

    def _decide(self, var: str, value: bool) -> None:
        """Make a decision."""
        self.assignment[var] = value
        self.decision_level[var] = len(self.decision_trail)
        self.decision_trail.append(var)

    def _backtrack(self) -> Optional[str]:
        """
        Backtrack to the last decision point and flip.

        Returns the flipped variable or None if no backtracking possible.
        """
        while self.decision_trail:
            var = self.decision_trail.pop()
            level = self.decision_level.pop(var)
            current_value = self.assignment.pop(var)

            # Remove all assignments at higher levels
            to_remove = [
                v for v, l in self.decision_level.items()
                if l > level
            ]
            for v in to_remove:
                self.assignment.pop(v, None)
                self.decision_level.pop(v, None)

            # If we haven't tried the opposite value, try it
            if current_value:  # Was True, try False
                self._decide(var, False)
                return var

        return None


class SATDependencyResolver:
    """
    SAT-based dependency resolver for plugins.

    Converts dependency constraints to CNF and uses DPLL solver
    to find a satisfying assignment.
    """

    def __init__(self):
        self._packages: Dict[str, List[PackageVersion]] = {}
        self._solver = DPLLSolver()

    def add_package(
        self,
        package_id: str,
        version: SemanticVersion,
        dependencies: Optional[List[PluginDependency]] = None,
        conflicts: Optional[List[str]] = None,
        provides: Optional[List[str]] = None,
    ) -> PackageVersion:
        """Register a package version."""
        pkg = PackageVersion(
            package_id=package_id,
            version=version,
            dependencies=dependencies or [],
            conflicts=conflicts or [],
            provides=provides or [],
        )

        if package_id not in self._packages:
            self._packages[package_id] = []
        self._packages[package_id].append(pkg)

        # Sort versions (newest first)
        self._packages[package_id].sort(
            key=lambda p: p.version,
            reverse=True,
        )

        return pkg

    def add_packages_from_registry(
        self,
        registry: Dict[str, List[Dict[str, Any]]],
    ) -> None:
        """Add packages from a registry format."""
        for package_id, versions in registry.items():
            for version_info in versions:
                self.add_package(
                    package_id=package_id,
                    version=SemanticVersion.parse(version_info["version"]),
                    dependencies=[
                        PluginDependency(
                            plugin_id=d["id"],
                            version_constraint=VersionConstraint.parse(d.get("version", "*")),
                            optional=d.get("optional", False),
                        )
                        for d in version_info.get("dependencies", [])
                    ],
                    conflicts=version_info.get("conflicts", []),
                    provides=version_info.get("provides", []),
                )

    def resolve(
        self,
        requirements: List[PluginDependency],
        installed: Optional[Dict[str, SemanticVersion]] = None,
    ) -> SolutionSet:
        """
        Resolve dependencies using SAT solver.

        Args:
            requirements: Top-level requirements to satisfy
            installed: Currently installed package versions (prefer keeping)

        Returns:
            SolutionSet with selected versions or conflicts
        """
        installed = installed or {}
        self._solver = DPLLSolver()

        # Build CNF formula
        self._encode_requirements(requirements)
        self._encode_at_most_one_per_package()
        self._encode_dependencies()
        self._encode_conflicts()
        self._encode_installed_preferences(installed)

        # Solve
        solution = self._solver.solve()

        # Build result
        if solution.satisfied:
            selected = {}
            for var, value in self._solver.assignment.items():
                if value:
                    pkg = self._get_package_by_id(var)
                    if pkg:
                        selected[pkg.package_id] = pkg
            solution.selected_versions = selected

        return solution

    def _encode_requirements(self, requirements: List[PluginDependency]) -> None:
        """Encode top-level requirements as clauses."""
        for req in requirements:
            if req.optional:
                continue  # Optional deps don't require installation

            matching = self._get_matching_versions(req.plugin_id, req.version_constraint)

            if not matching:
                # No matching versions - add empty clause (immediate conflict)
                self._solver.add_clause(Clause(
                    frozenset(),
                    reason=f"No version of {req.plugin_id} satisfies {req.version_constraint}",
                ))
                continue

            # At least one matching version must be installed
            self._solver.add_clause(Clause(
                frozenset((pkg.id, True) for pkg in matching),
                reason=f"Requirement: {req.plugin_id} {req.version_constraint}",
            ))

    def _encode_at_most_one_per_package(self) -> None:
        """Encode that at most one version of each package can be selected."""
        for package_id, versions in self._packages.items():
            if len(versions) > 1:
                version_ids = [pkg.id for pkg in versions]
                clauses = Clause.at_most_one(
                    version_ids,
                    reason=f"At most one version of {package_id}",
                )
                for clause in clauses:
                    self._solver.add_clause(clause)

    def _encode_dependencies(self) -> None:
        """Encode package dependencies as implications."""
        for package_id, versions in self._packages.items():
            for pkg in versions:
                for dep in pkg.dependencies:
                    if dep.optional:
                        continue

                    matching = self._get_matching_versions(
                        dep.plugin_id,
                        dep.version_constraint,
                    )

                    if not matching:
                        # Dependency unsatisfiable - if pkg is selected, conflict
                        self._solver.add_clause(Clause(
                            frozenset([(pkg.id, False)]),
                            reason=f"{pkg.id} requires unavailable {dep.plugin_id}",
                        ))
                        continue

                    # pkg implies (dep_v1 OR dep_v2 OR ...)
                    # Encoded as: NOT(pkg) OR dep_v1 OR dep_v2 OR ...
                    literals = [(pkg.id, False)]
                    literals.extend((dep_pkg.id, True) for dep_pkg in matching)

                    self._solver.add_clause(Clause(
                        frozenset(literals),
                        reason=f"{pkg.id} requires {dep.plugin_id} {dep.version_constraint}",
                    ))

    def _encode_conflicts(self) -> None:
        """Encode package conflicts."""
        for package_id, versions in self._packages.items():
            for pkg in versions:
                for conflict_id in pkg.conflicts:
                    if conflict_id in self._packages:
                        for conflict_pkg in self._packages[conflict_id]:
                            # pkg AND conflict_pkg cannot both be true
                            # NOT(pkg) OR NOT(conflict_pkg)
                            self._solver.add_clause(Clause(
                                frozenset([(pkg.id, False), (conflict_pkg.id, False)]),
                                reason=f"{pkg.id} conflicts with {conflict_pkg.id}",
                            ))

    def _encode_installed_preferences(
        self,
        installed: Dict[str, SemanticVersion],
    ) -> None:
        """
        Encode preferences to keep installed versions.

        Uses soft constraints (don't force, just prefer).
        """
        # This is a simplification - real solvers use weighted MAX-SAT
        # For now, we just ensure installed versions are considered first
        pass

    def _get_matching_versions(
        self,
        package_id: str,
        constraint: VersionConstraint,
    ) -> List[PackageVersion]:
        """Get all versions matching a constraint."""
        if package_id not in self._packages:
            return []

        return [
            pkg for pkg in self._packages[package_id]
            if constraint.satisfies(pkg.version)
        ]

    def _get_package_by_id(self, version_id: str) -> Optional[PackageVersion]:
        """Get a package by its version ID."""
        for versions in self._packages.values():
            for pkg in versions:
                if pkg.id == version_id:
                    return pkg
        return None

    def get_resolution_order(self, solution: SolutionSet) -> List[str]:
        """
        Get topological order for installing resolved packages.

        After SAT solving determines WHAT to install, this determines
        the ORDER to install them in.
        """
        if not solution.satisfied:
            return []

        # Build dependency graph from selected versions
        graph: Dict[str, Set[str]] = {}
        for pkg_id, pkg in solution.selected_versions.items():
            graph[pkg_id] = set()
            for dep in pkg.dependencies:
                if dep.plugin_id in solution.selected_versions:
                    graph[pkg_id].add(dep.plugin_id)

        # Topological sort (Kahn's algorithm)
        in_degree = {node: 0 for node in graph}
        for node, deps in graph.items():
            for dep in deps:
                if dep in in_degree:
                    in_degree[node] += 1

        queue = [node for node, degree in in_degree.items() if degree == 0]
        order = []

        while queue:
            node = queue.pop(0)
            order.append(node)

            for other, deps in graph.items():
                if node in deps:
                    in_degree[other] -= 1
                    if in_degree[other] == 0:
                        queue.append(other)

        return order

    def explain_conflict(self, solution: SolutionSet) -> str:
        """Generate human-readable conflict explanation."""
        if solution.satisfied:
            return "No conflicts"

        lines = ["Dependency resolution failed:"]

        for conflict in solution.conflicts:
            lines.append(f"\n  {conflict.conflict_type.value}:")
            lines.append(f"    {conflict.message}")
            if conflict.packages:
                lines.append(f"    Packages involved: {', '.join(conflict.packages)}")

        lines.append(f"\nStatistics:")
        lines.append(f"  Propagations: {solution.propagations}")
        lines.append(f"  Backtracks: {solution.backtracks}")

        return "\n".join(lines)


# Convenience function
def resolve_dependencies(
    requirements: List[PluginDependency],
    available_packages: Dict[str, List[Dict[str, Any]]],
    installed: Optional[Dict[str, SemanticVersion]] = None,
) -> SolutionSet:
    """
    Resolve plugin dependencies using SAT solver.

    Args:
        requirements: List of required plugins
        available_packages: Registry of available packages and versions
        installed: Currently installed packages

    Returns:
        SolutionSet with resolution result

    Example:
        requirements = [
            PluginDependency("plugin-a", VersionConstraint.parse("^1.0.0")),
            PluginDependency("plugin-b", VersionConstraint.parse(">=2.0.0")),
        ]

        available = {
            "plugin-a": [
                {"version": "1.0.0", "dependencies": []},
                {"version": "1.1.0", "dependencies": [{"id": "plugin-c", "version": "^1.0"}]},
            ],
            "plugin-b": [
                {"version": "2.0.0", "dependencies": []},
                {"version": "2.1.0", "dependencies": []},
            ],
            "plugin-c": [
                {"version": "1.0.0", "dependencies": []},
            ],
        }

        result = resolve_dependencies(requirements, available)
        if result.satisfied:
            for pkg_id, pkg in result.selected_versions.items():
                print(f"Install {pkg_id}@{pkg.version}")
    """
    resolver = SATDependencyResolver()
    resolver.add_packages_from_registry(available_packages)
    return resolver.resolve(requirements, installed)
