"""
Plugin Isolation Module

Provides true process isolation for plugin execution, preventing
cascading failures and security breaches.
"""

from aion.plugins.isolation.process import (
    ProcessIsolator,
    IsolatedPluginProxy,
    PluginProcess,
    ProcessConfig,
    IPCMessage,
    IPCMessageType,
)
from aion.plugins.isolation.resource_limits import (
    ResourceQuota,
    ResourceEnforcer,
    ResourceViolation,
)

__all__ = [
    "ProcessIsolator",
    "IsolatedPluginProxy",
    "PluginProcess",
    "ProcessConfig",
    "IPCMessage",
    "IPCMessageType",
    "ResourceQuota",
    "ResourceEnforcer",
    "ResourceViolation",
]
