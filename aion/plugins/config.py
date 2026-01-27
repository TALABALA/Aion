"""
AION Plugin Configuration

Configuration management for the plugin system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class PluginSystemConfig(BaseModel):
    """Configuration for the plugin system."""

    # Directories
    plugin_dirs: List[str] = Field(
        default_factory=lambda: ["./plugins", "~/.aion/plugins"],
        description="Directories to search for plugins",
    )
    cache_dir: str = Field(
        default="~/.aion/plugin_cache",
        description="Directory for plugin cache",
    )

    # Discovery
    auto_discover: bool = Field(
        default=True,
        description="Automatically discover plugins on startup",
    )
    watch_directories: bool = Field(
        default=False,
        description="Watch plugin directories for changes",
    )
    scan_interval_seconds: int = Field(
        default=60,
        description="Interval for directory scanning",
    )

    # Loading
    auto_load_builtin: bool = Field(
        default=True,
        description="Automatically load builtin plugins",
    )
    parallel_loading: bool = Field(
        default=True,
        description="Load plugins in parallel where possible",
    )
    max_load_workers: int = Field(
        default=4,
        description="Maximum workers for parallel loading",
    )

    # Sandbox
    enable_sandbox: bool = Field(
        default=True,
        description="Enable sandboxed plugin execution",
    )
    sandbox_timeout_seconds: float = Field(
        default=30.0,
        description="Default sandbox execution timeout",
    )
    sandbox_max_memory_mb: int = Field(
        default=256,
        description="Default maximum memory for sandboxed plugins",
    )

    # Hot reload
    enable_hot_reload: bool = Field(
        default=True,
        description="Enable hot reloading of plugins",
    )
    reload_on_change: bool = Field(
        default=False,
        description="Automatically reload when plugin files change",
    )

    # Security
    require_signature: bool = Field(
        default=False,
        description="Require plugins to be signed",
    )
    require_verification: bool = Field(
        default=False,
        description="Require plugins to be verified",
    )
    allowed_permission_levels: List[str] = Field(
        default_factory=lambda: ["minimal", "restricted", "standard", "elevated"],
        description="Allowed permission levels",
    )
    blocked_plugins: List[str] = Field(
        default_factory=list,
        description="List of blocked plugin IDs",
    )

    # Marketplace
    marketplace_enabled: bool = Field(
        default=False,
        description="Enable plugin marketplace",
    )
    marketplace_url: str = Field(
        default="https://plugins.aion.ai",
        description="Marketplace API URL",
    )
    auto_update: bool = Field(
        default=False,
        description="Automatically update plugins",
    )

    # Logging
    log_level: str = Field(
        default="INFO",
        description="Log level for plugin system",
    )
    log_plugin_calls: bool = Field(
        default=False,
        description="Log plugin method calls",
    )

    # Limits
    max_plugins: int = Field(
        default=100,
        description="Maximum number of loaded plugins",
    )
    max_hooks_per_plugin: int = Field(
        default=50,
        description="Maximum hooks per plugin",
    )

    def get_plugin_dirs(self) -> List[Path]:
        """Get plugin directories as Path objects."""
        return [Path(d).expanduser() for d in self.plugin_dirs]


@dataclass
class PluginRuntimeConfig:
    """Runtime configuration for a specific plugin."""

    plugin_id: str
    enabled: bool = True
    config: Dict[str, Any] = field(default_factory=dict)
    permissions_override: Optional[Dict[str, Any]] = None
    priority_override: Optional[int] = None
    sandbox_override: Optional[bool] = None
    timeout_override: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plugin_id": self.plugin_id,
            "enabled": self.enabled,
            "config": self.config,
            "permissions_override": self.permissions_override,
            "priority_override": self.priority_override,
            "sandbox_override": self.sandbox_override,
            "timeout_override": self.timeout_override,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PluginRuntimeConfig":
        return cls(
            plugin_id=data["plugin_id"],
            enabled=data.get("enabled", True),
            config=data.get("config", {}),
            permissions_override=data.get("permissions_override"),
            priority_override=data.get("priority_override"),
            sandbox_override=data.get("sandbox_override"),
            timeout_override=data.get("timeout_override"),
        )


class ConfigManager:
    """
    Manages plugin configurations.

    Features:
    - Load/save plugin configs
    - Runtime config updates
    - Config validation
    - Config migration
    """

    def __init__(self, config_dir: Optional[Path] = None):
        self._config_dir = config_dir or Path.home() / ".aion" / "plugin_configs"
        self._plugin_configs: Dict[str, PluginRuntimeConfig] = {}
        self._system_config = PluginSystemConfig()

    def initialize(self) -> None:
        """Initialize config manager."""
        self._config_dir.mkdir(parents=True, exist_ok=True)
        self._load_configs()

    def _load_configs(self) -> None:
        """Load all plugin configurations."""
        import json

        for config_file in self._config_dir.glob("*.json"):
            try:
                with open(config_file) as f:
                    data = json.load(f)
                    plugin_config = PluginRuntimeConfig.from_dict(data)
                    self._plugin_configs[plugin_config.plugin_id] = plugin_config
            except Exception as e:
                logger.error(f"Error loading config {config_file}: {e}")

    def save_config(self, plugin_id: str) -> None:
        """Save plugin configuration."""
        import json

        config = self._plugin_configs.get(plugin_id)
        if not config:
            return

        config_file = self._config_dir / f"{plugin_id}.json"
        with open(config_file, "w") as f:
            json.dump(config.to_dict(), f, indent=2)

    def get_config(self, plugin_id: str) -> Optional[PluginRuntimeConfig]:
        """Get plugin runtime configuration."""
        return self._plugin_configs.get(plugin_id)

    def set_config(
        self,
        plugin_id: str,
        config: Dict[str, Any],
        save: bool = True,
    ) -> PluginRuntimeConfig:
        """Set plugin configuration."""
        runtime_config = self._plugin_configs.get(plugin_id)

        if runtime_config:
            runtime_config.config = config
        else:
            runtime_config = PluginRuntimeConfig(
                plugin_id=plugin_id,
                config=config,
            )
            self._plugin_configs[plugin_id] = runtime_config

        if save:
            self.save_config(plugin_id)

        return runtime_config

    def enable_plugin(self, plugin_id: str) -> None:
        """Enable a plugin."""
        config = self._plugin_configs.get(plugin_id)
        if config:
            config.enabled = True
            self.save_config(plugin_id)
        else:
            self.set_config(plugin_id, {})

    def disable_plugin(self, plugin_id: str) -> None:
        """Disable a plugin."""
        config = self._plugin_configs.get(plugin_id)
        if config:
            config.enabled = False
            self.save_config(plugin_id)

    def is_enabled(self, plugin_id: str) -> bool:
        """Check if plugin is enabled."""
        config = self._plugin_configs.get(plugin_id)
        return config.enabled if config else True

    def get_system_config(self) -> PluginSystemConfig:
        """Get system configuration."""
        return self._system_config

    def update_system_config(self, **kwargs) -> None:
        """Update system configuration."""
        for key, value in kwargs.items():
            if hasattr(self._system_config, key):
                setattr(self._system_config, key, value)


import structlog
logger = structlog.get_logger(__name__)
