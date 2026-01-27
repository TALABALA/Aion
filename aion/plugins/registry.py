"""
AION Plugin Registry

Centralized storage and lookup for plugin information.
Provides efficient querying by type, state, tags, and features.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from datetime import datetime
from typing import Any, Callable, Dict, Iterator, List, Optional, Set

import structlog

from aion.plugins.types import (
    PluginInfo,
    PluginManifest,
    PluginState,
    PluginType,
    SemanticVersion,
)

logger = structlog.get_logger(__name__)


class PluginRegistry:
    """
    Central registry for all plugins.

    Features:
    - Thread-safe plugin storage
    - Multi-index lookups (by type, state, tags, features)
    - Version management
    - Dependency tracking
    - Statistics and monitoring
    """

    def __init__(self):
        # Primary storage
        self._plugins: Dict[str, PluginInfo] = {}

        # Indices for fast lookup
        self._by_type: Dict[PluginType, Set[str]] = defaultdict(set)
        self._by_state: Dict[PluginState, Set[str]] = defaultdict(set)
        self._by_tag: Dict[str, Set[str]] = defaultdict(set)
        self._by_feature: Dict[str, Set[str]] = defaultdict(set)
        self._by_source: Dict[str, Set[str]] = defaultdict(set)

        # Dependency tracking
        self._dependents: Dict[str, Set[str]] = defaultdict(set)  # plugin_id -> plugins that depend on it
        self._dependencies: Dict[str, Set[str]] = defaultdict(set)  # plugin_id -> plugins it depends on

        # Version tracking (plugin_id -> list of versions)
        self._versions: Dict[str, List[SemanticVersion]] = defaultdict(list)

        # Lock for thread safety
        self._lock = asyncio.Lock()

        # Event callbacks
        self._on_register: List[Callable[[PluginInfo], None]] = []
        self._on_unregister: List[Callable[[str], None]] = []
        self._on_state_change: List[Callable[[str, PluginState, PluginState], None]] = []

    # === Registration ===

    def register(self, plugin_info: PluginInfo) -> bool:
        """
        Register a plugin.

        Args:
            plugin_info: Plugin information to register

        Returns:
            True if registered, False if already exists
        """
        plugin_id = plugin_info.id

        if plugin_id in self._plugins:
            logger.warning(f"Plugin already registered: {plugin_id}")
            return False

        # Store in primary storage
        self._plugins[plugin_id] = plugin_info

        # Update indices
        self._index_plugin(plugin_info)

        # Track version
        self._versions[plugin_id].append(plugin_info.version)

        logger.debug(f"Registered plugin: {plugin_id} v{plugin_info.version}")

        # Notify listeners
        for callback in self._on_register:
            try:
                callback(plugin_info)
            except Exception as e:
                logger.error(f"Error in register callback: {e}")

        return True

    def unregister(self, plugin_id: str) -> Optional[PluginInfo]:
        """
        Unregister a plugin.

        Args:
            plugin_id: Plugin identifier to unregister

        Returns:
            The unregistered plugin info, or None if not found
        """
        if plugin_id not in self._plugins:
            return None

        plugin_info = self._plugins.pop(plugin_id)

        # Remove from indices
        self._unindex_plugin(plugin_info)

        # Clean up dependency tracking
        self._clean_dependencies(plugin_id)

        logger.debug(f"Unregistered plugin: {plugin_id}")

        # Notify listeners
        for callback in self._on_unregister:
            try:
                callback(plugin_id)
            except Exception as e:
                logger.error(f"Error in unregister callback: {e}")

        return plugin_info

    def update(self, plugin_info: PluginInfo) -> bool:
        """
        Update a plugin's information.

        Args:
            plugin_info: Updated plugin information

        Returns:
            True if updated, False if not found
        """
        plugin_id = plugin_info.id

        if plugin_id not in self._plugins:
            return False

        old_info = self._plugins[plugin_id]

        # Remove old indices
        self._unindex_plugin(old_info)

        # Update storage
        self._plugins[plugin_id] = plugin_info

        # Add new indices
        self._index_plugin(plugin_info)

        return True

    def _index_plugin(self, plugin_info: PluginInfo) -> None:
        """Add plugin to all indices."""
        plugin_id = plugin_info.id
        manifest = plugin_info.manifest

        # Type index
        self._by_type[manifest.plugin_type].add(plugin_id)

        # State index
        self._by_state[plugin_info.state].add(plugin_id)

        # Tag index
        for tag in manifest.tags:
            self._by_tag[tag.lower()].add(plugin_id)

        # Feature index
        for feature in manifest.features:
            self._by_feature[feature].add(plugin_id)

        # Source index
        self._by_source[plugin_info.source].add(plugin_id)

        # Dependency tracking
        for dep in manifest.dependencies:
            self._dependencies[plugin_id].add(dep.plugin_id)
            self._dependents[dep.plugin_id].add(plugin_id)

    def _unindex_plugin(self, plugin_info: PluginInfo) -> None:
        """Remove plugin from all indices."""
        plugin_id = plugin_info.id
        manifest = plugin_info.manifest

        # Type index
        self._by_type[manifest.plugin_type].discard(plugin_id)

        # State index
        self._by_state[plugin_info.state].discard(plugin_id)

        # Tag index
        for tag in manifest.tags:
            self._by_tag[tag.lower()].discard(plugin_id)

        # Feature index
        for feature in manifest.features:
            self._by_feature[feature].discard(plugin_id)

        # Source index
        self._by_source[plugin_info.source].discard(plugin_id)

    def _clean_dependencies(self, plugin_id: str) -> None:
        """Clean up dependency tracking for a plugin."""
        # Remove from dependents of other plugins
        for dep_id in self._dependencies.get(plugin_id, set()):
            self._dependents[dep_id].discard(plugin_id)

        # Remove from dependencies of other plugins
        for dependent_id in self._dependents.get(plugin_id, set()):
            self._dependencies[dependent_id].discard(plugin_id)

        # Clean up own entries
        self._dependencies.pop(plugin_id, None)
        self._dependents.pop(plugin_id, None)

    # === State Management ===

    def set_state(self, plugin_id: str, state: PluginState) -> bool:
        """
        Update plugin state.

        Args:
            plugin_id: Plugin identifier
            state: New state

        Returns:
            True if state updated, False if plugin not found
        """
        if plugin_id not in self._plugins:
            return False

        plugin_info = self._plugins[plugin_id]
        old_state = plugin_info.state

        if old_state == state:
            return True

        # Update state index
        self._by_state[old_state].discard(plugin_id)
        self._by_state[state].add(plugin_id)

        # Update plugin
        plugin_info.state = state

        logger.debug(f"Plugin {plugin_id} state: {old_state.value} -> {state.value}")

        # Notify listeners
        for callback in self._on_state_change:
            try:
                callback(plugin_id, old_state, state)
            except Exception as e:
                logger.error(f"Error in state change callback: {e}")

        return True

    # === Queries ===

    def get(self, plugin_id: str) -> Optional[PluginInfo]:
        """Get plugin by ID."""
        return self._plugins.get(plugin_id)

    def exists(self, plugin_id: str) -> bool:
        """Check if plugin exists."""
        return plugin_id in self._plugins

    def get_all(self) -> List[PluginInfo]:
        """Get all registered plugins."""
        return list(self._plugins.values())

    def get_by_type(self, plugin_type: PluginType) -> List[PluginInfo]:
        """Get plugins by type."""
        return [
            self._plugins[pid]
            for pid in self._by_type.get(plugin_type, set())
            if pid in self._plugins
        ]

    def get_by_state(self, state: PluginState) -> List[PluginInfo]:
        """Get plugins by state."""
        return [
            self._plugins[pid]
            for pid in self._by_state.get(state, set())
            if pid in self._plugins
        ]

    def get_active(self) -> List[PluginInfo]:
        """Get all active plugins."""
        return self.get_by_state(PluginState.ACTIVE)

    def get_by_tag(self, tag: str) -> List[PluginInfo]:
        """Get plugins by tag."""
        return [
            self._plugins[pid]
            for pid in self._by_tag.get(tag.lower(), set())
            if pid in self._plugins
        ]

    def get_by_feature(self, feature: str) -> List[PluginInfo]:
        """Get plugins that provide a feature."""
        return [
            self._plugins[pid]
            for pid in self._by_feature.get(feature, set())
            if pid in self._plugins
        ]

    def get_by_source(self, source: str) -> List[PluginInfo]:
        """Get plugins by source."""
        return [
            self._plugins[pid]
            for pid in self._by_source.get(source, set())
            if pid in self._plugins
        ]

    def get_dependents(self, plugin_id: str) -> List[str]:
        """Get plugins that depend on the given plugin."""
        return list(self._dependents.get(plugin_id, set()))

    def get_dependencies(self, plugin_id: str) -> List[str]:
        """Get plugins that the given plugin depends on."""
        return list(self._dependencies.get(plugin_id, set()))

    def count(self) -> int:
        """Get total number of registered plugins."""
        return len(self._plugins)

    def count_by_state(self) -> Dict[PluginState, int]:
        """Get count by state."""
        return {state: len(pids) for state, pids in self._by_state.items()}

    def count_by_type(self) -> Dict[PluginType, int]:
        """Get count by type."""
        return {ptype: len(pids) for ptype, pids in self._by_type.items()}

    # === Search ===

    def search(
        self,
        query: Optional[str] = None,
        plugin_type: Optional[PluginType] = None,
        state: Optional[PluginState] = None,
        tags: Optional[List[str]] = None,
        features: Optional[List[str]] = None,
        source: Optional[str] = None,
        active_only: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> List[PluginInfo]:
        """
        Search plugins with filters.

        Args:
            query: Text search in name/description
            plugin_type: Filter by type
            state: Filter by state
            tags: Filter by tags (any match)
            features: Filter by features (all must match)
            source: Filter by source
            active_only: Only return active plugins
            limit: Maximum results
            offset: Skip first N results

        Returns:
            List of matching plugins
        """
        # Start with all plugins or filtered set
        candidates: Set[str] = set(self._plugins.keys())

        # Apply type filter
        if plugin_type:
            candidates &= self._by_type.get(plugin_type, set())

        # Apply state filter
        if state:
            candidates &= self._by_state.get(state, set())
        elif active_only:
            candidates &= self._by_state.get(PluginState.ACTIVE, set())

        # Apply source filter
        if source:
            candidates &= self._by_source.get(source, set())

        # Apply tag filter (any match)
        if tags:
            tag_matches: Set[str] = set()
            for tag in tags:
                tag_matches |= self._by_tag.get(tag.lower(), set())
            candidates &= tag_matches

        # Apply feature filter (all must match)
        if features:
            for feature in features:
                candidates &= self._by_feature.get(feature, set())

        # Get plugin info objects
        results: List[PluginInfo] = []
        for plugin_id in candidates:
            if plugin_id not in self._plugins:
                continue

            plugin_info = self._plugins[plugin_id]

            # Apply text search
            if query:
                query_lower = query.lower()
                if not (
                    query_lower in plugin_info.name.lower()
                    or query_lower in plugin_info.manifest.description.lower()
                    or query_lower in plugin_info.id.lower()
                ):
                    continue

            results.append(plugin_info)

        # Sort by name
        results.sort(key=lambda p: p.name.lower())

        # Apply pagination
        return results[offset : offset + limit]

    # === Version Management ===

    def get_versions(self, plugin_id: str) -> List[SemanticVersion]:
        """Get all known versions of a plugin."""
        return sorted(self._versions.get(plugin_id, []))

    def get_latest_version(self, plugin_id: str) -> Optional[SemanticVersion]:
        """Get latest version of a plugin."""
        versions = self.get_versions(plugin_id)
        return versions[-1] if versions else None

    # === Callbacks ===

    def on_register(self, callback: Callable[[PluginInfo], None]) -> None:
        """Register callback for plugin registration."""
        self._on_register.append(callback)

    def on_unregister(self, callback: Callable[[str], None]) -> None:
        """Register callback for plugin unregistration."""
        self._on_unregister.append(callback)

    def on_state_change(
        self, callback: Callable[[str, PluginState, PluginState], None]
    ) -> None:
        """Register callback for state changes."""
        self._on_state_change.append(callback)

    # === Statistics ===

    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        return {
            "total_plugins": self.count(),
            "by_state": {
                state.value: count
                for state, count in self.count_by_state().items()
                if count > 0
            },
            "by_type": {
                ptype.value: count
                for ptype, count in self.count_by_type().items()
                if count > 0
            },
            "by_source": {
                source: len(pids)
                for source, pids in self._by_source.items()
                if pids
            },
            "active_plugins": len(self.get_active()),
            "total_tags": len(self._by_tag),
            "total_features": len(self._by_feature),
        }

    # === Iteration ===

    def __iter__(self) -> Iterator[PluginInfo]:
        """Iterate over all plugins."""
        return iter(self._plugins.values())

    def __len__(self) -> int:
        """Get number of plugins."""
        return len(self._plugins)

    def __contains__(self, plugin_id: str) -> bool:
        """Check if plugin exists."""
        return plugin_id in self._plugins

    # === Serialization ===

    def to_dict(self) -> Dict[str, Any]:
        """Serialize registry to dictionary."""
        return {
            "plugins": {pid: info.to_dict() for pid, info in self._plugins.items()},
            "stats": self.get_stats(),
        }

    def export_manifests(self) -> List[Dict[str, Any]]:
        """Export all plugin manifests."""
        return [
            plugin_info.manifest.to_dict()
            for plugin_info in self._plugins.values()
        ]
