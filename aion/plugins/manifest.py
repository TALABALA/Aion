"""
AION Plugin Manifest Handling

Utilities for working with plugin manifests.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import structlog

from aion.plugins.types import PluginManifest, SemanticVersion

logger = structlog.get_logger(__name__)


class ManifestError(Exception):
    """Error related to manifest operations."""
    pass


def load_manifest(path: Path) -> PluginManifest:
    """
    Load a manifest from a file.

    Args:
        path: Path to manifest.json file

    Returns:
        PluginManifest

    Raises:
        ManifestError: If loading or parsing fails
    """
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return PluginManifest.from_dict(data)
    except json.JSONDecodeError as e:
        raise ManifestError(f"Invalid JSON in manifest: {e}")
    except KeyError as e:
        raise ManifestError(f"Missing required field in manifest: {e}")
    except Exception as e:
        raise ManifestError(f"Failed to load manifest: {e}")


def save_manifest(manifest: PluginManifest, path: Path) -> None:
    """
    Save a manifest to a file.

    Args:
        manifest: PluginManifest to save
        path: Path to save to
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest.to_dict(), f, indent=2)


def create_manifest(
    plugin_id: str,
    name: str,
    version: str = "1.0.0",
    **kwargs,
) -> PluginManifest:
    """
    Create a new manifest with defaults.

    Args:
        plugin_id: Plugin identifier
        name: Plugin display name
        version: Version string
        **kwargs: Additional manifest fields

    Returns:
        PluginManifest
    """
    return PluginManifest(
        id=plugin_id,
        name=name,
        version=SemanticVersion.parse(version),
        **kwargs,
    )


def merge_manifests(
    base: PluginManifest,
    overlay: Dict[str, Any],
) -> PluginManifest:
    """
    Merge overlay values into a base manifest.

    Args:
        base: Base manifest
        overlay: Values to overlay

    Returns:
        New merged manifest
    """
    base_dict = base.to_dict()
    base_dict.update(overlay)
    return PluginManifest.from_dict(base_dict)


def validate_manifest_file(path: Path) -> tuple[bool, list]:
    """
    Validate a manifest file.

    Args:
        path: Path to manifest file

    Returns:
        (is_valid, errors)
    """
    errors = []

    if not path.exists():
        errors.append(f"Manifest file not found: {path}")
        return False, errors

    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        errors.append(f"Invalid JSON: {e}")
        return False, errors

    # Check required fields
    required_fields = ["id", "name", "version"]
    for field in required_fields:
        if field not in data:
            errors.append(f"Missing required field: {field}")

    # Validate ID format
    plugin_id = data.get("id", "")
    if plugin_id:
        import re
        if not re.match(r'^[a-z][a-z0-9_-]*[a-z0-9]$|^[a-z]$', plugin_id):
            errors.append(f"Invalid plugin ID format: {plugin_id}")

    # Validate version
    version = data.get("version", "")
    if version:
        try:
            SemanticVersion.parse(version)
        except Exception:
            errors.append(f"Invalid version format: {version}")

    return len(errors) == 0, errors


def generate_manifest_template(plugin_type: str = "tool") -> Dict[str, Any]:
    """
    Generate a manifest template.

    Args:
        plugin_type: Type of plugin

    Returns:
        Template dictionary
    """
    return {
        "id": "my-plugin",
        "name": "My Plugin",
        "version": "1.0.0",
        "description": "A description of what this plugin does",
        "type": plugin_type,
        "author": {
            "name": "Your Name",
            "email": "you@example.com",
            "url": "https://example.com"
        },
        "homepage": "https://github.com/example/my-plugin",
        "repository": "https://github.com/example/my-plugin",
        "license": "MIT",
        "entry_point": "plugin:MyPlugin",
        "aion_version": ">=1.0.0",
        "python_version": ">=3.10",
        "pip_dependencies": [],
        "dependencies": [],
        "permissions": {
            "level": "standard",
            "network_access": False,
            "file_system_access": False,
            "database_access": False
        },
        "config_schema": {
            "type": "object",
            "properties": {},
            "required": []
        },
        "default_config": {},
        "hooks": [],
        "features": [],
        "tags": [],
        "auto_enable": False,
        "singleton": False,
        "priority": 50
    }


def scaffold_plugin(
    plugin_dir: Path,
    plugin_id: str,
    plugin_name: str,
    plugin_type: str = "tool",
) -> None:
    """
    Scaffold a new plugin directory with template files.

    Args:
        plugin_dir: Directory to create plugin in
        plugin_id: Plugin identifier
        plugin_name: Plugin display name
        plugin_type: Type of plugin
    """
    # Create directory
    plugin_path = plugin_dir / plugin_id
    plugin_path.mkdir(parents=True, exist_ok=True)

    # Generate manifest
    manifest = generate_manifest_template(plugin_type)
    manifest["id"] = plugin_id
    manifest["name"] = plugin_name

    with open(plugin_path / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # Generate plugin.py
    plugin_code = _generate_plugin_code(plugin_id, plugin_name, plugin_type)
    with open(plugin_path / "plugin.py", "w") as f:
        f.write(plugin_code)

    # Generate __init__.py
    with open(plugin_path / "__init__.py", "w") as f:
        f.write(f'"""AION Plugin: {plugin_name}"""\n\nfrom .plugin import *\n')

    # Generate README.md
    readme = f"""# {plugin_name}

{plugin_name} plugin for AION.

## Installation

Copy this directory to your AION plugins folder:
- `~/.aion/plugins/{plugin_id}/`

## Configuration

```json
{{}}
```

## Usage

This plugin provides...

## License

MIT
"""
    with open(plugin_path / "README.md", "w") as f:
        f.write(readme)

    logger.info(f"Scaffolded plugin: {plugin_id} at {plugin_path}")


def _generate_plugin_code(
    plugin_id: str,
    plugin_name: str,
    plugin_type: str,
) -> str:
    """Generate plugin boilerplate code."""
    class_name = "".join(word.capitalize() for word in plugin_id.replace("-", "_").split("_"))

    if plugin_type == "tool":
        return f'''"""
{plugin_name} - AION Tool Plugin
"""

from typing import Any, Dict, List, Optional

from aion.plugins import (
    ToolPlugin,
    PluginManifest,
    PluginType,
    SemanticVersion,
    Tool,
    ToolParameter,
    ToolParameterType,
)


class {class_name}Plugin(ToolPlugin):
    """{plugin_name} plugin implementation."""

    @classmethod
    def get_manifest(cls) -> PluginManifest:
        return PluginManifest(
            id="{plugin_id}",
            name="{plugin_name}",
            version=SemanticVersion(1, 0, 0),
            description="Description of {plugin_name}",
            plugin_type=PluginType.TOOL,
            entry_point="plugin:{class_name}Plugin",
            tags=["example"],
        )

    def get_tools(self) -> List[Tool]:
        return [
            Tool(
                name="example_tool",
                description="An example tool",
                parameters=[
                    ToolParameter(
                        name="input",
                        type=ToolParameterType.STRING,
                        description="Input parameter",
                        required=True,
                    ),
                ],
                category="general",
            ),
        ]

    async def execute(
        self,
        tool_name: str,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Any:
        if tool_name == "example_tool":
            return f"Processed: {{params['input']}}"
        raise ValueError(f"Unknown tool: {{tool_name}}")

    async def initialize(self, kernel, config: Dict[str, Any]) -> None:
        self._kernel = kernel
        self._config = config
        self._initialized = True

    async def shutdown(self) -> None:
        self._initialized = False
'''

    # Default plugin code for other types
    return f'''"""
{plugin_name} - AION Plugin
"""

from typing import Any, Dict

from aion.plugins import (
    BasePlugin,
    PluginManifest,
    PluginType,
    SemanticVersion,
)


class {class_name}Plugin(BasePlugin):
    """{plugin_name} plugin implementation."""

    @classmethod
    def get_manifest(cls) -> PluginManifest:
        return PluginManifest(
            id="{plugin_id}",
            name="{plugin_name}",
            version=SemanticVersion(1, 0, 0),
            description="Description of {plugin_name}",
            plugin_type=PluginType.{plugin_type.upper()},
            entry_point="plugin:{class_name}Plugin",
        )

    async def initialize(self, kernel, config: Dict[str, Any]) -> None:
        self._kernel = kernel
        self._config = config
        self._initialized = True

    async def shutdown(self) -> None:
        self._initialized = False
'''
