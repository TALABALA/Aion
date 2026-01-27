"""
AION Plugin Loader

Handles plugin discovery and loading from various sources.
Supports directory-based, single-file, and package plugins.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Type
import shutil
import tempfile

import structlog

from aion.plugins.types import (
    PluginInfo,
    PluginManifest,
    PluginState,
    SemanticVersion,
)

logger = structlog.get_logger(__name__)


class PluginLoadError(Exception):
    """Raised when plugin loading fails."""

    def __init__(self, plugin_id: str, message: str, cause: Optional[Exception] = None):
        self.plugin_id = plugin_id
        self.message = message
        self.cause = cause
        super().__init__(f"Failed to load plugin '{plugin_id}': {message}")


class PluginLoader:
    """
    Discovers and loads plugins from file system.

    Supports:
    - Directory-based plugins with manifest.json
    - Single-file plugins with embedded manifest
    - Package plugins (zip/wheel)
    - Namespace packages
    - Hot reload with module cache invalidation
    """

    MANIFEST_FILE = "manifest.json"
    PLUGIN_FILE = "plugin.py"
    PACKAGE_EXTENSIONS = (".zip", ".whl", ".tar.gz")

    def __init__(self, plugin_dirs: Optional[List[Path]] = None):
        self.plugin_dirs = plugin_dirs or []
        self._manifest_cache: Dict[str, PluginManifest] = {}
        self._module_cache: Dict[str, Any] = {}
        self._path_cache: Dict[str, Path] = {}
        self._checksum_cache: Dict[str, str] = {}
        self._extracted_packages: Dict[str, Path] = {}

    def add_plugin_dir(self, path: Path) -> None:
        """Add a plugin directory."""
        if path not in self.plugin_dirs:
            self.plugin_dirs.append(path)
            logger.debug(f"Added plugin directory: {path}")

    def remove_plugin_dir(self, path: Path) -> None:
        """Remove a plugin directory."""
        if path in self.plugin_dirs:
            self.plugin_dirs.remove(path)

    # === Discovery ===

    def discover(self) -> List[PluginManifest]:
        """
        Discover all plugins in plugin directories.

        Returns:
            List of discovered plugin manifests
        """
        manifests = []
        seen_ids: Set[str] = set()

        for plugin_dir in self.plugin_dirs:
            if not plugin_dir.exists():
                logger.debug(f"Plugin directory does not exist: {plugin_dir}")
                continue

            logger.debug(f"Scanning plugin directory: {plugin_dir}")

            for item in plugin_dir.iterdir():
                try:
                    manifest = self._discover_plugin(item)
                    if manifest and manifest.id not in seen_ids:
                        manifests.append(manifest)
                        seen_ids.add(manifest.id)
                        self._manifest_cache[manifest.id] = manifest
                        self._path_cache[manifest.id] = item
                except Exception as e:
                    logger.warning(f"Error discovering plugin at {item}: {e}")

        logger.info(f"Discovered {len(manifests)} plugins")
        return manifests

    def discover_single(self, path: Path) -> Optional[PluginManifest]:
        """
        Discover a single plugin from a path.

        Args:
            path: Path to plugin directory or file

        Returns:
            Plugin manifest if found
        """
        manifest = self._discover_plugin(path)
        if manifest:
            self._manifest_cache[manifest.id] = manifest
            self._path_cache[manifest.id] = path
        return manifest

    def _discover_plugin(self, path: Path) -> Optional[PluginManifest]:
        """Discover a single plugin."""
        try:
            if path.is_dir():
                return self._discover_directory_plugin(path)
            elif path.suffix == ".py":
                return self._discover_file_plugin(path)
            elif path.suffix in self.PACKAGE_EXTENSIONS or path.name.endswith(".tar.gz"):
                return self._discover_package_plugin(path)
        except Exception as e:
            logger.warning(f"Error discovering plugin at {path}: {e}")
        return None

    def _discover_directory_plugin(self, path: Path) -> Optional[PluginManifest]:
        """Discover plugin from directory."""
        manifest_path = path / self.MANIFEST_FILE

        if manifest_path.exists():
            manifest = self._load_manifest_file(manifest_path)

            # Calculate checksum of manifest
            checksum = self._calculate_checksum(manifest_path)
            self._checksum_cache[manifest.id] = checksum

            return manifest

        # Try to extract manifest from plugin.py
        plugin_path = path / self.PLUGIN_FILE
        if plugin_path.exists():
            return self._extract_manifest_from_file(plugin_path)

        # Try __init__.py
        init_path = path / "__init__.py"
        if init_path.exists():
            return self._extract_manifest_from_file(init_path)

        return None

    def _discover_file_plugin(self, path: Path) -> Optional[PluginManifest]:
        """Discover single-file plugin."""
        return self._extract_manifest_from_file(path)

    def _discover_package_plugin(self, path: Path) -> Optional[PluginManifest]:
        """Discover plugin from package (zip/wheel)."""
        # Extract to temp directory
        extract_dir = Path(tempfile.mkdtemp(prefix="aion_plugin_"))

        try:
            if path.suffix == ".zip" or path.suffix == ".whl":
                with zipfile.ZipFile(path, "r") as zf:
                    zf.extractall(extract_dir)
            elif path.name.endswith(".tar.gz"):
                shutil.unpack_archive(path, extract_dir)

            # Look for manifest
            for item in extract_dir.iterdir():
                if item.is_dir():
                    manifest = self._discover_directory_plugin(item)
                    if manifest:
                        self._extracted_packages[manifest.id] = extract_dir
                        return manifest

            # Check root
            manifest = self._discover_directory_plugin(extract_dir)
            if manifest:
                self._extracted_packages[manifest.id] = extract_dir
                return manifest

        except Exception as e:
            logger.error(f"Error extracting package {path}: {e}")
            shutil.rmtree(extract_dir, ignore_errors=True)

        return None

    def _load_manifest_file(self, path: Path) -> PluginManifest:
        """Load manifest from JSON file."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return PluginManifest.from_dict(data)

    def _extract_manifest_from_file(self, path: Path) -> Optional[PluginManifest]:
        """Extract manifest from a Python file."""
        try:
            # Load module temporarily
            module_name = f"_aion_discover_{path.stem}_{id(path)}"
            spec = importlib.util.spec_from_file_location(module_name, path)
            if not spec or not spec.loader:
                return None

            module = importlib.util.module_from_spec(spec)

            # Don't add to sys.modules permanently
            try:
                spec.loader.exec_module(module)
            except Exception as e:
                logger.debug(f"Error loading module for manifest extraction: {e}")
                return None

            # Find BasePlugin subclass
            from aion.plugins.interfaces.base import BasePlugin

            for name in dir(module):
                obj = getattr(module, name, None)
                if obj is None:
                    continue
                if (
                    isinstance(obj, type)
                    and issubclass(obj, BasePlugin)
                    and obj is not BasePlugin
                    and hasattr(obj, "get_manifest")
                ):
                    try:
                        return obj.get_manifest()
                    except Exception as e:
                        logger.debug(f"Error getting manifest from {name}: {e}")

        except Exception as e:
            logger.debug(f"Error extracting manifest from {path}: {e}")

        return None

    # === Loading ===

    def load_plugin(self, manifest: PluginManifest) -> Type:
        """
        Load a plugin class from manifest.

        Args:
            manifest: Plugin manifest

        Returns:
            Plugin class

        Raises:
            PluginLoadError: If loading fails
        """
        plugin_id = manifest.id

        # Check cache
        if plugin_id in self._module_cache:
            module = self._module_cache[plugin_id]
            return self._get_class_from_module(module, manifest)

        # Get plugin path
        path = self._path_cache.get(plugin_id)
        if not path:
            # Check extracted packages
            path = self._extracted_packages.get(plugin_id)
            if not path:
                raise PluginLoadError(plugin_id, "Plugin path not found")

        try:
            # Load module based on type
            if path.is_dir():
                module = self._load_directory_plugin(path, manifest)
            else:
                module = self._load_file_plugin(path)

            # Cache module
            self._module_cache[plugin_id] = module

            # Get class
            return self._get_class_from_module(module, manifest)

        except Exception as e:
            raise PluginLoadError(plugin_id, str(e), e) from e

    def _load_directory_plugin(self, path: Path, manifest: PluginManifest) -> Any:
        """Load plugin from directory."""
        # Add plugin directory to path
        plugin_dir = str(path)
        if plugin_dir not in sys.path:
            sys.path.insert(0, plugin_dir)

        # Parse entry point
        entry_point = manifest.entry_point
        if ":" in entry_point:
            module_name, _ = entry_point.split(":", 1)
        else:
            module_name = entry_point or path.name.replace("-", "_")

        # Import module
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            # Try relative import from plugin directory
            plugin_module = path / f"{module_name}.py"
            if plugin_module.exists():
                spec = importlib.util.spec_from_file_location(
                    module_name, plugin_module
                )
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = module
                    spec.loader.exec_module(module)
                else:
                    raise
            else:
                raise

        return module

    def _load_file_plugin(self, path: Path) -> Any:
        """Load single-file plugin."""
        module_name = path.stem

        spec = importlib.util.spec_from_file_location(module_name, path)
        if not spec or not spec.loader:
            raise PluginLoadError(
                module_name, f"Cannot create module spec for {path}"
            )

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        return module

    def _get_class_from_module(self, module: Any, manifest: PluginManifest) -> Type:
        """Get plugin class from module."""
        from aion.plugins.interfaces.base import BasePlugin

        entry_point = manifest.entry_point

        # If explicit class name in entry point
        if ":" in entry_point:
            _, class_name = entry_point.split(":", 1)
            if hasattr(module, class_name):
                return getattr(module, class_name)
            raise PluginLoadError(
                manifest.id, f"Class '{class_name}' not found in module"
            )

        # Find first BasePlugin subclass
        for name in dir(module):
            obj = getattr(module, name, None)
            if obj is None:
                continue
            if (
                isinstance(obj, type)
                and issubclass(obj, BasePlugin)
                and obj is not BasePlugin
            ):
                return obj

        raise PluginLoadError(manifest.id, "No plugin class found in module")

    # === Hot Reload ===

    def reload_plugin(self, plugin_id: str) -> Type:
        """
        Reload a plugin module.

        Args:
            plugin_id: Plugin identifier

        Returns:
            Reloaded plugin class
        """
        manifest = self._manifest_cache.get(plugin_id)
        if not manifest:
            raise PluginLoadError(plugin_id, "Plugin manifest not found")

        # Clear caches
        self.clear_cache(plugin_id)

        # Reload
        return self.load_plugin(manifest)

    def clear_cache(self, plugin_id: str) -> None:
        """Clear cached plugin data for reload."""
        # Clear module cache
        self._module_cache.pop(plugin_id, None)

        # Clear from sys.modules
        manifest = self._manifest_cache.get(plugin_id)
        if manifest and manifest.entry_point:
            module_name = manifest.entry_point.split(":")[0]

            # Remove module and all submodules
            modules_to_remove = [
                name
                for name in sys.modules
                if name == module_name or name.startswith(f"{module_name}.")
            ]
            for name in modules_to_remove:
                sys.modules.pop(name, None)

        logger.debug(f"Cleared cache for plugin: {plugin_id}")

    def clear_all_caches(self) -> None:
        """Clear all plugin caches."""
        plugin_ids = list(self._module_cache.keys())
        for plugin_id in plugin_ids:
            self.clear_cache(plugin_id)

    # === Path Management ===

    def get_plugin_path(self, plugin_id: str) -> Optional[Path]:
        """Get path to plugin directory/file."""
        return self._path_cache.get(plugin_id)

    def get_manifest(self, plugin_id: str) -> Optional[PluginManifest]:
        """Get cached manifest."""
        return self._manifest_cache.get(plugin_id)

    # === Checksum ===

    def _calculate_checksum(self, path: Path) -> str:
        """Calculate SHA256 checksum of a file."""
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def has_changed(self, plugin_id: str) -> bool:
        """Check if plugin files have changed since discovery."""
        path = self._path_cache.get(plugin_id)
        if not path:
            return False

        if path.is_dir():
            manifest_path = path / self.MANIFEST_FILE
            if manifest_path.exists():
                current = self._calculate_checksum(manifest_path)
                cached = self._checksum_cache.get(plugin_id)
                return current != cached

        return False

    # === Installation ===

    def install_from_path(self, source: Path, target_dir: Path) -> Optional[PluginManifest]:
        """
        Install a plugin from a path.

        Args:
            source: Source path (directory, file, or package)
            target_dir: Target plugin directory

        Returns:
            Installed plugin manifest
        """
        # Discover manifest first
        manifest = self._discover_plugin(source)
        if not manifest:
            logger.error(f"No plugin manifest found at {source}")
            return None

        # Create target directory
        plugin_dir = target_dir / manifest.id
        plugin_dir.mkdir(parents=True, exist_ok=True)

        # Copy files
        if source.is_dir():
            shutil.copytree(source, plugin_dir, dirs_exist_ok=True)
        elif source.suffix == ".py":
            shutil.copy2(source, plugin_dir / source.name)
        else:
            # Package - extract
            shutil.unpack_archive(source, plugin_dir)

        # Update caches
        self._path_cache[manifest.id] = plugin_dir
        self._manifest_cache[manifest.id] = manifest

        logger.info(f"Installed plugin: {manifest.id} to {plugin_dir}")
        return manifest

    def uninstall(self, plugin_id: str) -> bool:
        """
        Uninstall a plugin.

        Args:
            plugin_id: Plugin identifier

        Returns:
            True if uninstalled successfully
        """
        path = self._path_cache.get(plugin_id)
        if not path:
            return False

        # Clear caches
        self.clear_cache(plugin_id)
        self._manifest_cache.pop(plugin_id, None)
        self._path_cache.pop(plugin_id, None)
        self._checksum_cache.pop(plugin_id, None)

        # Clean up extracted package
        extracted = self._extracted_packages.pop(plugin_id, None)
        if extracted and extracted.exists():
            shutil.rmtree(extracted, ignore_errors=True)

        # Remove plugin directory (if we installed it)
        if path.exists() and path.is_dir():
            try:
                shutil.rmtree(path)
                logger.info(f"Uninstalled plugin: {plugin_id}")
                return True
            except Exception as e:
                logger.error(f"Error removing plugin directory: {e}")

        return False

    # === Cleanup ===

    def cleanup(self) -> None:
        """Clean up temporary files and resources."""
        # Remove extracted packages
        for plugin_id, extract_dir in list(self._extracted_packages.items()):
            if extract_dir.exists():
                shutil.rmtree(extract_dir, ignore_errors=True)
        self._extracted_packages.clear()

        # Clear caches
        self.clear_all_caches()
