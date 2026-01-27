"""
Hot Module Replacement for Plugins

Provides hot reload capability with state preservation,
similar to Webpack/Vite HMR patterns.
"""

from aion.plugins.hotreload.hmr import (
    HotModuleReplacer,
    HMRConfig,
    ModuleState,
    StateTransfer,
    HMREvent,
    HMREventType,
)

__all__ = [
    "HotModuleReplacer",
    "HMRConfig",
    "ModuleState",
    "StateTransfer",
    "HMREvent",
    "HMREventType",
]
