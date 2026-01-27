"""
AION Local Plugin Discovery

Discovers plugins from local file system.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent

import structlog

from aion.plugins.types import PluginManifest, PluginInfo, PluginState

logger = structlog.get_logger(__name__)


class PluginDirectoryHandler(FileSystemEventHandler):
    """Handles file system events for plugin directories."""

    def __init__(self, callback: Callable[[str, str], None]):
        self._callback = callback

    def on_created(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            return
        self._callback(event.src_path, "created")

    def on_deleted(self, event: FileSystemEvent) -> None:
        self._callback(event.src_path, "deleted")

    def on_modified(self, event: FileSystemEvent) -> None:
        if event.src_path.endswith("manifest.json"):
            self._callback(str(Path(event.src_path).parent), "modified")


class LocalDiscovery:
    """
    Discovers plugins from local file system.

    Features:
    - Directory scanning
    - File watching for changes
    - Manifest parsing
    - Checksum tracking
    """

    MANIFEST_FILE = "manifest.json"

    def __init__(self, plugin_dirs: Optional[List[Path]] = None):
        self._plugin_dirs = plugin_dirs or []
        self._discovered: Dict[str, PluginInfo] = {}
        self._checksums: Dict[str, str] = {}
        self._observer: Optional[Observer] = None
        self._callbacks: List[Callable[[str, str], None]] = []

    def add_directory(self, path: Path) -> None:
        """Add a plugin directory to scan."""
        if path not in self._plugin_dirs:
            self._plugin_dirs.append(path)

    def remove_directory(self, path: Path) -> None:
        """Remove a plugin directory."""
        if path in self._plugin_dirs:
            self._plugin_dirs.remove(path)

    async def scan(self) -> List[PluginManifest]:
        """
        Scan all directories for plugins.

        Returns:
            List of discovered manifests
        """
        manifests = []

        for plugin_dir in self._plugin_dirs:
            if not plugin_dir.exists():
                plugin_dir.mkdir(parents=True, exist_ok=True)
                continue

            dir_manifests = await self._scan_directory(plugin_dir)
            manifests.extend(dir_manifests)

        logger.info(f"Local discovery found {len(manifests)} plugins")
        return manifests

    async def _scan_directory(self, directory: Path) -> List[PluginManifest]:
        """Scan a single directory for plugins."""
        manifests = []

        for item in directory.iterdir():
            manifest = await self._discover_plugin(item)
            if manifest:
                manifests.append(manifest)

        return manifests

    async def _discover_plugin(self, path: Path) -> Optional[PluginManifest]:
        """Discover a plugin from a path."""
        try:
            if path.is_dir():
                manifest_path = path / self.MANIFEST_FILE
                if manifest_path.exists():
                    return await self._load_manifest(manifest_path)

            elif path.suffix == ".py":
                # Single-file plugin - extract manifest from code
                return await self._extract_manifest(path)

        except Exception as e:
            logger.warning(f"Error discovering plugin at {path}: {e}")

        return None

    async def _load_manifest(self, path: Path) -> PluginManifest:
        """Load manifest from JSON file."""
        loop = asyncio.get_event_loop()

        def _load():
            with open(path, encoding="utf-8") as f:
                return json.load(f)

        data = await loop.run_in_executor(None, _load)
        manifest = PluginManifest.from_dict(data)

        # Track checksum
        checksum = self._compute_checksum(path)
        self._checksums[manifest.id] = checksum

        return manifest

    async def _extract_manifest(self, path: Path) -> Optional[PluginManifest]:
        """Extract manifest from Python file."""
        # This would use AST parsing or dynamic import
        # For now, return None (handled by loader)
        return None

    def _compute_checksum(self, path: Path) -> str:
        """Compute SHA256 checksum of a file."""
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def has_changed(self, plugin_id: str, path: Path) -> bool:
        """Check if a plugin's manifest has changed."""
        manifest_path = path / self.MANIFEST_FILE
        if not manifest_path.exists():
            return True

        current_checksum = self._compute_checksum(manifest_path)
        stored_checksum = self._checksums.get(plugin_id)

        return current_checksum != stored_checksum

    # === File Watching ===

    def start_watching(self) -> None:
        """Start watching plugin directories for changes."""
        if self._observer:
            return

        self._observer = Observer()
        handler = PluginDirectoryHandler(self._on_directory_change)

        for plugin_dir in self._plugin_dirs:
            if plugin_dir.exists():
                self._observer.schedule(handler, str(plugin_dir), recursive=False)

        self._observer.start()
        logger.info("Started watching plugin directories")

    def stop_watching(self) -> None:
        """Stop watching plugin directories."""
        if self._observer:
            self._observer.stop()
            self._observer.join()
            self._observer = None
            logger.info("Stopped watching plugin directories")

    def _on_directory_change(self, path: str, event_type: str) -> None:
        """Handle directory change event."""
        logger.debug(f"Plugin directory change: {path} ({event_type})")

        for callback in self._callbacks:
            try:
                callback(path, event_type)
            except Exception as e:
                logger.error(f"Error in directory change callback: {e}")

    def on_change(self, callback: Callable[[str, str], None]) -> None:
        """Register callback for directory changes."""
        self._callbacks.append(callback)

    # === Utilities ===

    def list_directories(self) -> List[Path]:
        """List all configured plugin directories."""
        return self._plugin_dirs.copy()

    def get_plugin_path(self, plugin_id: str) -> Optional[Path]:
        """Find path to a plugin by ID."""
        for plugin_dir in self._plugin_dirs:
            if not plugin_dir.exists():
                continue

            for item in plugin_dir.iterdir():
                if item.is_dir() and item.name == plugin_id:
                    return item

                manifest_path = item / self.MANIFEST_FILE if item.is_dir() else None
                if manifest_path and manifest_path.exists():
                    try:
                        with open(manifest_path) as f:
                            data = json.load(f)
                            if data.get("id") == plugin_id:
                                return item
                    except Exception:
                        pass

        return None
