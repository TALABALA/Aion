"""
AION Plugin Hooks

Hook system for plugin extensibility.
"""

from aion.plugins.hooks.system import HookSystem, BUILTIN_HOOKS
from aion.plugins.hooks.events import PluginEventEmitter

__all__ = [
    "HookSystem",
    "BUILTIN_HOOKS",
    "PluginEventEmitter",
]
