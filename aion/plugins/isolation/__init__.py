"""
Plugin Isolation Module

Provides true process isolation for plugin execution, preventing
cascading failures and security breaches.

IPC Options:
- Queue-based (default): Uses multiprocessing.Queue with pickle/JSON
- gRPC-based (high-performance): Uses gRPC with protobuf/msgpack
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
    ResourceUsage,
    ResourceEnforcer,
    ResourceViolation,
    ResourceMonitor,
)
from aion.plugins.isolation.grpc_ipc import (
    GRPCConfig,
    GRPCPluginServer,
    GRPCPluginClient,
    GRPCIsolatedPlugin,
    SerializationFormat,
    InitializeRequest,
    InitializeResponse,
    ExecuteRequest,
    ExecuteResponse,
    HealthcheckResponse,
    MetricsResponse,
    GRPC_AVAILABLE,
    MSGPACK_AVAILABLE,
)

__all__ = [
    # Queue-based isolation
    "ProcessIsolator",
    "IsolatedPluginProxy",
    "PluginProcess",
    "ProcessConfig",
    "IPCMessage",
    "IPCMessageType",

    # Resource limits
    "ResourceQuota",
    "ResourceUsage",
    "ResourceEnforcer",
    "ResourceViolation",
    "ResourceMonitor",

    # gRPC-based isolation
    "GRPCConfig",
    "GRPCPluginServer",
    "GRPCPluginClient",
    "GRPCIsolatedPlugin",
    "SerializationFormat",
    "InitializeRequest",
    "InitializeResponse",
    "ExecuteRequest",
    "ExecuteResponse",
    "HealthcheckResponse",
    "MetricsResponse",

    # Feature flags
    "GRPC_AVAILABLE",
    "MSGPACK_AVAILABLE",
]
