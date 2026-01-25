"""
AION Tracing Module

Distributed tracing for request flows across components.
OpenTelemetry-compatible with W3C Trace Context support.
"""

from aion.observability.tracing.engine import (
    TracingEngine,
    SpanContextManager,
    traced,
)
from aion.observability.tracing.propagation import (
    ContextPropagator,
    W3CTraceContextPropagator,
    B3Propagator,
    CompositePropagator,
)
from aion.observability.tracing.sampling import (
    Sampler,
    AlwaysOnSampler,
    AlwaysOffSampler,
    TraceIdRatioSampler,
    ParentBasedSampler,
    RateLimitingSampler,
    AdaptiveSampler,
)

__all__ = [
    "TracingEngine",
    "SpanContextManager",
    "traced",
    "ContextPropagator",
    "W3CTraceContextPropagator",
    "B3Propagator",
    "CompositePropagator",
    "Sampler",
    "AlwaysOnSampler",
    "AlwaysOffSampler",
    "TraceIdRatioSampler",
    "ParentBasedSampler",
    "RateLimitingSampler",
    "AdaptiveSampler",
]
