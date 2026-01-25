"""
AION Logging Module

Structured logging with trace correlation and multiple outputs.
"""

from aion.observability.logging.engine import (
    LoggingEngine,
    ObservabilityLogger,
)
from aion.observability.logging.correlation import (
    CorrelatedLogger,
    inject_trace_context,
)

__all__ = [
    "LoggingEngine",
    "ObservabilityLogger",
    "CorrelatedLogger",
    "inject_trace_context",
]
