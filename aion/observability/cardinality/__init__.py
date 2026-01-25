"""
Cardinality Management for High-Cardinality Metrics.

This module provides automatic cardinality limiting and adaptive sampling
to handle high-cardinality metric dimensions without overwhelming storage.
"""

from .manager import (
    CardinalityManager,
    CardinalityLimiter,
    AdaptiveSampler,
    LabelDropper,
    LabelHasher,
    CardinalityEstimator,
    HyperLogLog,
    CountMinSketch,
)

from .policies import (
    CardinalityPolicy,
    DropHighCardinalityLabels,
    HashHighCardinalityLabels,
    SampleHighCardinalitySeries,
    AggregateHighCardinalitySeries,
    RelabelingRule,
)

__all__ = [
    # Core
    "CardinalityManager",
    "CardinalityLimiter",
    "AdaptiveSampler",
    "LabelDropper",
    "LabelHasher",
    "CardinalityEstimator",
    # Data Structures
    "HyperLogLog",
    "CountMinSketch",
    # Policies
    "CardinalityPolicy",
    "DropHighCardinalityLabels",
    "HashHighCardinalityLabels",
    "SampleHighCardinalitySeries",
    "AggregateHighCardinalitySeries",
    "RelabelingRule",
]
