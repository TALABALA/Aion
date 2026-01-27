"""
AION Plugin Dependency Resolver

Resolves plugin dependencies and determines load order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import structlog

from aion.plugins.types import (
    PluginDependency,
    PluginInfo,
    PluginManifest,
    SemanticVersion,
    ValidationResult,
    VersionConstraint,
)
from aion.plugins.dependencies.graph import DependencyGraph

logger = structlog.get_logger(__name__)


@dataclass
class ResolutionResult:
    """Result of dependency resolution."""

    success: bool
    load_order: List[str] = field(default_factory=list)
    missing_dependencies: Dict[str, List[str]] = field(default_factory=dict)
    version_conflicts: List[str] = field(default_factory=list)
    circular_dependencies: List[List[str]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "load_order": self.load_order,
            "missing_dependencies": self.missing_dependencies,
            "version_conflicts": self.version_conflicts,
            "circular_dependencies": self.circular_dependencies,
            "warnings": self.warnings,
        }


class DependencyResolver:
    """
    Resolves plugin dependencies using topological sort.

    Features:
    - Dependency graph construction
    - Cycle detection
    - Load order calculation
    - Version compatibility checking
    - Optional dependency handling
    - Conflict detection
    """

    def __init__(self):
        self._graph = DependencyGraph()

    def resolve(self, plugins: List[PluginInfo]) -> ResolutionResult:
        """
        Resolve dependencies for a list of plugins.

        Args:
            plugins: List of plugins to resolve

        Returns:
            ResolutionResult with load order and any issues
        """
        result = ResolutionResult(success=True)

        # Build plugin map
        plugin_map: Dict[str, PluginInfo] = {p.id: p for p in plugins}

        # Build dependency graph
        self._graph.clear()

        for plugin in plugins:
            self._graph.add_node(plugin.id)

            for dep in plugin.manifest.dependencies:
                if not dep.optional:
                    self._graph.add_edge(dep.plugin_id, plugin.id)

        # Check for missing dependencies
        missing = self._check_missing_dependencies(plugins, plugin_map)
        if missing:
            result.missing_dependencies = missing
            result.success = False

        # Check for version conflicts
        conflicts = self._check_version_conflicts(plugins, plugin_map)
        if conflicts:
            result.version_conflicts = conflicts
            result.success = False

        # Check for cycles
        cycles = self._graph.find_cycles()
        if cycles:
            result.circular_dependencies = cycles
            result.success = False

        # Calculate load order
        if result.success:
            try:
                order = self._graph.topological_sort()
                # Filter to only include requested plugins
                result.load_order = [p for p in order if p in plugin_map]
            except ValueError as e:
                result.success = False
                result.warnings.append(str(e))

        # Add warnings for optional dependencies
        for plugin in plugins:
            for dep in plugin.manifest.dependencies:
                if dep.optional and dep.plugin_id not in plugin_map:
                    result.warnings.append(
                        f"Optional dependency '{dep.plugin_id}' for '{plugin.id}' not available"
                    )

        return result

    def resolve_order(self, plugins: List[PluginInfo]) -> List[PluginInfo]:
        """
        Resolve load order based on dependencies.

        Args:
            plugins: List of plugins to order

        Returns:
            Plugins in dependency-safe load order
        """
        result = self.resolve(plugins)
        plugin_map = {p.id: p for p in plugins}

        return [plugin_map[pid] for pid in result.load_order if pid in plugin_map]

    def _check_missing_dependencies(
        self,
        plugins: List[PluginInfo],
        available: Dict[str, PluginInfo],
    ) -> Dict[str, List[str]]:
        """Check for missing required dependencies."""
        missing: Dict[str, List[str]] = {}

        for plugin in plugins:
            plugin_missing = []

            for dep in plugin.manifest.dependencies:
                if dep.optional:
                    continue

                if dep.plugin_id not in available:
                    plugin_missing.append(dep.plugin_id)

            if plugin_missing:
                missing[plugin.id] = plugin_missing

        return missing

    def _check_version_conflicts(
        self,
        plugins: List[PluginInfo],
        available: Dict[str, PluginInfo],
    ) -> List[str]:
        """Check for version conflicts."""
        conflicts: List[str] = []

        for plugin in plugins:
            for dep in plugin.manifest.dependencies:
                if dep.plugin_id not in available:
                    continue

                dep_plugin = available[dep.plugin_id]
                if not dep.version_constraint.satisfies(dep_plugin.version):
                    conflicts.append(
                        f"{plugin.id} requires {dep.plugin_id} "
                        f"{dep.version_constraint}, but {dep_plugin.version} is available"
                    )

        return conflicts

    def check_dependencies(
        self,
        plugin: PluginInfo,
        available: Dict[str, PluginInfo],
    ) -> Tuple[bool, List[str]]:
        """
        Check if all dependencies are satisfied for a single plugin.

        Args:
            plugin: Plugin to check
            available: Available plugins

        Returns:
            (satisfied, missing_deps)
        """
        missing = []

        for dep in plugin.manifest.dependencies:
            if dep.optional:
                continue

            available_plugin = available.get(dep.plugin_id)

            if not available_plugin:
                missing.append(dep.plugin_id)
            elif not dep.version_constraint.satisfies(available_plugin.version):
                missing.append(
                    f"{dep.plugin_id} (need {dep.version_constraint}, "
                    f"have {available_plugin.version})"
                )

        return len(missing) == 0, missing

    def find_dependents(
        self,
        plugin_id: str,
        plugins: List[PluginInfo],
    ) -> List[str]:
        """
        Find plugins that depend on the given plugin.

        Args:
            plugin_id: Plugin to find dependents for
            plugins: All plugins

        Returns:
            List of dependent plugin IDs
        """
        dependents = []

        for plugin in plugins:
            for dep in plugin.manifest.dependencies:
                if dep.plugin_id == plugin_id:
                    dependents.append(plugin.id)
                    break

        return dependents

    def get_dependency_tree(
        self,
        plugin_id: str,
        plugins: List[PluginInfo],
    ) -> Dict[str, Any]:
        """
        Get full dependency tree for a plugin.

        Args:
            plugin_id: Root plugin
            plugins: Available plugins

        Returns:
            Nested dict representing dependency tree
        """
        plugin_map = {p.id: p for p in plugins}

        def build_tree(pid: str, visited: Set[str]) -> Dict[str, Any]:
            if pid in visited:
                return {"id": pid, "circular": True}

            visited.add(pid)
            plugin = plugin_map.get(pid)

            if not plugin:
                return {"id": pid, "missing": True}

            children = {}
            for dep in plugin.manifest.dependencies:
                children[dep.plugin_id] = build_tree(dep.plugin_id, visited.copy())

            return {
                "id": pid,
                "version": str(plugin.version),
                "dependencies": children if children else None,
            }

        return build_tree(plugin_id, set())

    def validate_manifest_dependencies(
        self,
        manifest: PluginManifest,
        available: Dict[str, SemanticVersion],
    ) -> ValidationResult:
        """
        Validate dependencies in a manifest.

        Args:
            manifest: Plugin manifest to validate
            available: Available plugins and their versions

        Returns:
            ValidationResult
        """
        result = ValidationResult(valid=True)

        for dep in manifest.dependencies:
            if dep.plugin_id not in available:
                if dep.optional:
                    result.add_warning(
                        f"Optional dependency '{dep.plugin_id}' not available"
                    )
                else:
                    result.add_error(
                        f"Required dependency '{dep.plugin_id}' not available"
                    )
            else:
                version = available[dep.plugin_id]
                if not dep.version_constraint.satisfies(version):
                    result.add_error(
                        f"Dependency '{dep.plugin_id}' version {version} "
                        f"does not satisfy constraint {dep.version_constraint}"
                    )

        return result

    def suggest_resolution(
        self,
        missing: Dict[str, List[str]],
        available: Dict[str, List[SemanticVersion]],
    ) -> Dict[str, str]:
        """
        Suggest how to resolve missing dependencies.

        Args:
            missing: Missing dependencies per plugin
            available: Available versions per plugin

        Returns:
            Suggestions per plugin
        """
        suggestions: Dict[str, str] = {}

        for plugin_id, deps in missing.items():
            for dep in deps:
                if dep in available:
                    versions = sorted(available[dep])
                    suggestions[dep] = f"Install {dep} (available versions: {', '.join(str(v) for v in versions[-3:])})"
                else:
                    suggestions[dep] = f"Plugin '{dep}' is not available in any repository"

        return suggestions
