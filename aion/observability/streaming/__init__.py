"""
Stream Processing for Real-time Observability.

This module provides Flink/Kafka Streams style stream processing
for real-time anomaly detection and observability data processing.
"""

from .processor import (
    StreamProcessor,
    StreamSource,
    StreamSink,
    StreamOperator,
    WindowedStream,
    KeyedStream,
    DataStream,
)

from .operators import (
    MapOperator,
    FilterOperator,
    FlatMapOperator,
    ReduceOperator,
    AggregateOperator,
    JoinOperator,
    WindowOperator,
)

from .windows import (
    TumblingWindow,
    SlidingWindow,
    SessionWindow,
    GlobalWindow,
    TimeWindow,
    CountWindow,
)

from .anomaly import (
    StreamingAnomalyDetector,
    OnlineStatistics,
    StreamingIsolationForest,
    ExponentialMovingStats,
    AdaptiveThreshold,
)

__all__ = [
    # Core
    "StreamProcessor",
    "StreamSource",
    "StreamSink",
    "StreamOperator",
    "WindowedStream",
    "KeyedStream",
    "DataStream",
    # Operators
    "MapOperator",
    "FilterOperator",
    "FlatMapOperator",
    "ReduceOperator",
    "AggregateOperator",
    "JoinOperator",
    "WindowOperator",
    # Windows
    "TumblingWindow",
    "SlidingWindow",
    "SessionWindow",
    "GlobalWindow",
    "TimeWindow",
    "CountWindow",
    # Anomaly Detection
    "StreamingAnomalyDetector",
    "OnlineStatistics",
    "StreamingIsolationForest",
    "ExponentialMovingStats",
    "AdaptiveThreshold",
]
