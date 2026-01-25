"""
Native OpenTelemetry Protocol (OTLP) Implementation.

This module provides fully native OTLP support with:
- OTLP/gRPC and OTLP/HTTP exporters
- Semantic conventions compliance
- Resource detection
- Baggage propagation
"""

from .protocol import (
    OTLPExporter,
    OTLPGRPCExporter,
    OTLPHTTPExporter,
    OTLPCollector,
)

from .semantic import (
    SemanticConventions,
    ResourceAttributes,
    SpanAttributes,
    MetricAttributes,
    LogAttributes,
    HTTPSemantics,
    DatabaseSemantics,
    MessagingSemantics,
    RPCSemantics,
)

from .resource import (
    ResourceDetector,
    Resource,
    ServiceResourceDetector,
    HostResourceDetector,
    ProcessResourceDetector,
    ContainerResourceDetector,
    K8sResourceDetector,
    CloudResourceDetector,
)

from .baggage import (
    Baggage,
    BaggageManager,
    BaggagePropagator,
)

__all__ = [
    # Protocol
    "OTLPExporter",
    "OTLPGRPCExporter",
    "OTLPHTTPExporter",
    "OTLPCollector",
    # Semantic Conventions
    "SemanticConventions",
    "ResourceAttributes",
    "SpanAttributes",
    "MetricAttributes",
    "LogAttributes",
    "HTTPSemantics",
    "DatabaseSemantics",
    "MessagingSemantics",
    "RPCSemantics",
    # Resource
    "ResourceDetector",
    "Resource",
    "ServiceResourceDetector",
    "HostResourceDetector",
    "ProcessResourceDetector",
    "ContainerResourceDetector",
    "K8sResourceDetector",
    "CloudResourceDetector",
    # Baggage
    "Baggage",
    "BaggageManager",
    "BaggagePropagator",
]
