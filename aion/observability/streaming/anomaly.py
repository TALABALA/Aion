"""
Real-time Streaming Anomaly Detection.

Implements online anomaly detection algorithms:
- Online statistics (mean, variance, percentiles)
- Streaming Isolation Forest
- Exponential moving statistics
- Adaptive thresholds
- Complex Event Processing (CEP) patterns
"""

import math
import random
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable, Deque
from datetime import datetime, timedelta
from collections import deque
from enum import Enum

logger = logging.getLogger(__name__)


# =============================================================================
# Online Statistics
# =============================================================================

class OnlineStatistics:
    """
    Online (streaming) statistics computation using Welford's algorithm.

    Computes mean, variance, and standard deviation in a single pass
    with O(1) space complexity.
    """

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0  # Sum of squares of differences from mean
        self.min_val = float('inf')
        self.max_val = float('-inf')

    def update(self, value: float):
        """Update statistics with a new value."""
        self.n += 1
        delta = value - self.mean
        self.mean += delta / self.n
        delta2 = value - self.mean
        self.m2 += delta * delta2

        self.min_val = min(self.min_val, value)
        self.max_val = max(self.max_val, value)

    @property
    def variance(self) -> float:
        """Population variance."""
        if self.n < 2:
            return 0.0
        return self.m2 / self.n

    @property
    def sample_variance(self) -> float:
        """Sample variance (Bessel's correction)."""
        if self.n < 2:
            return 0.0
        return self.m2 / (self.n - 1)

    @property
    def std_dev(self) -> float:
        """Population standard deviation."""
        return math.sqrt(self.variance)

    @property
    def sample_std_dev(self) -> float:
        """Sample standard deviation."""
        return math.sqrt(self.sample_variance)

    def z_score(self, value: float) -> float:
        """Compute z-score for a value."""
        if self.std_dev == 0:
            return 0.0
        return (value - self.mean) / self.std_dev

    def is_outlier(self, value: float, threshold: float = 3.0) -> bool:
        """Check if value is an outlier using z-score."""
        return abs(self.z_score(value)) > threshold

    def merge(self, other: 'OnlineStatistics') -> 'OnlineStatistics':
        """Merge two OnlineStatistics instances."""
        if other.n == 0:
            return self
        if self.n == 0:
            return other

        merged = OnlineStatistics()
        merged.n = self.n + other.n

        delta = other.mean - self.mean
        merged.mean = (self.n * self.mean + other.n * other.mean) / merged.n
        merged.m2 = (self.m2 + other.m2 +
                    delta * delta * self.n * other.n / merged.n)

        merged.min_val = min(self.min_val, other.min_val)
        merged.max_val = max(self.max_val, other.max_val)

        return merged


class OnlinePercentile:
    """
    Online percentile estimation using P-squared algorithm.

    Estimates percentiles in streaming fashion with O(1) space.
    """

    def __init__(self, percentile: float = 0.5):
        self.p = percentile
        self.n = 0

        # Marker positions (desired and actual)
        self.marker_count = 5
        self.q = [0.0] * self.marker_count  # Marker heights
        self.n_pos = [0] * self.marker_count  # Marker positions
        self.n_prime = [0.0] * self.marker_count  # Desired positions

        self._initialized = False

    def _initialize(self, values: List[float]):
        """Initialize with first 5 values."""
        sorted_vals = sorted(values[:5])
        self.q = sorted_vals

        self.n_pos = [1, 2, 3, 4, 5]
        self.n_prime = [
            1,
            1 + 2 * self.p,
            1 + 4 * self.p,
            3 + 2 * self.p,
            5
        ]
        self._initialized = True

    def update(self, value: float):
        """Update percentile estimate with new value."""
        self.n += 1

        if not self._initialized:
            if self.n < 5:
                self.q[self.n - 1] = value
                return
            else:
                self.q[4] = value
                self._initialize(self.q)
                return

        # Find cell k
        k = -1
        if value < self.q[0]:
            self.q[0] = value
            k = 0
        elif value >= self.q[4]:
            self.q[4] = value
            k = 3
        else:
            for i in range(1, 5):
                if value < self.q[i]:
                    k = i - 1
                    break

        # Increment positions
        for i in range(k + 1, 5):
            self.n_pos[i] += 1

        # Update desired positions
        self.n_prime[1] = (self.n - 1) * self.p / 2 + 1
        self.n_prime[2] = (self.n - 1) * self.p + 1
        self.n_prime[3] = (self.n - 1) * (1 + self.p) / 2 + 1
        self.n_prime[4] = self.n

        # Adjust marker heights
        for i in range(1, 4):
            d = self.n_prime[i] - self.n_pos[i]

            if (d >= 1 and self.n_pos[i + 1] - self.n_pos[i] > 1) or \
               (d <= -1 and self.n_pos[i - 1] - self.n_pos[i] < -1):
                d_sign = 1 if d >= 0 else -1

                # Try parabolic interpolation
                q_new = self._parabolic(i, d_sign)

                if self.q[i - 1] < q_new < self.q[i + 1]:
                    self.q[i] = q_new
                else:
                    # Fall back to linear
                    self.q[i] = self._linear(i, d_sign)

                self.n_pos[i] += d_sign

    def _parabolic(self, i: int, d: int) -> float:
        """Parabolic interpolation formula."""
        qi = self.q[i]
        n_i = self.n_pos[i]

        term1 = d / (self.n_pos[i + 1] - self.n_pos[i - 1])

        a = (n_i - self.n_pos[i - 1] + d) * (self.q[i + 1] - qi) / (self.n_pos[i + 1] - n_i)
        b = (self.n_pos[i + 1] - n_i - d) * (qi - self.q[i - 1]) / (n_i - self.n_pos[i - 1])

        return qi + term1 * (a + b)

    def _linear(self, i: int, d: int) -> float:
        """Linear interpolation formula."""
        return self.q[i] + d * (self.q[i + d] - self.q[i]) / (self.n_pos[i + d] - self.n_pos[i])

    @property
    def estimate(self) -> float:
        """Get current percentile estimate."""
        if not self._initialized:
            if self.n == 0:
                return 0.0
            idx = int(self.p * self.n)
            return sorted(self.q[:self.n])[min(idx, self.n - 1)]
        return self.q[2]


# =============================================================================
# Exponential Moving Statistics
# =============================================================================

class ExponentialMovingStats:
    """
    Exponential moving statistics for anomaly detection.

    Gives more weight to recent observations using exponential decay.
    """

    def __init__(self, alpha: float = 0.1, span: int = None):
        """
        Initialize with decay factor alpha or span.

        Args:
            alpha: Decay factor (0 < alpha <= 1), higher = faster decay
            span: Alternative: specify span, alpha = 2/(span+1)
        """
        if span is not None:
            self.alpha = 2.0 / (span + 1)
        else:
            self.alpha = alpha

        self.ema = None  # Exponential moving average
        self.emvar = None  # Exponential moving variance
        self.n = 0

    def update(self, value: float):
        """Update statistics with new value."""
        self.n += 1

        if self.ema is None:
            self.ema = value
            self.emvar = 0.0
            return

        # Update EMA
        delta = value - self.ema
        self.ema = self.ema + self.alpha * delta

        # Update exponential moving variance
        self.emvar = (1 - self.alpha) * (self.emvar + self.alpha * delta * delta)

    @property
    def mean(self) -> float:
        return self.ema if self.ema is not None else 0.0

    @property
    def variance(self) -> float:
        return self.emvar if self.emvar is not None else 0.0

    @property
    def std_dev(self) -> float:
        return math.sqrt(self.variance)

    def z_score(self, value: float) -> float:
        """Compute z-score relative to EMA."""
        if self.std_dev == 0:
            return 0.0
        return (value - self.mean) / self.std_dev

    def is_anomaly(self, value: float, threshold: float = 3.0) -> bool:
        """Check if value is anomalous."""
        return abs(self.z_score(value)) > threshold


class ExponentialMovingAverage:
    """Simple EMA with EWMA bands for anomaly detection."""

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.ema = None
        self.ema_upper = None
        self.ema_lower = None
        self._values: Deque[float] = deque(maxlen=100)

    def update(self, value: float) -> Tuple[float, float, float]:
        """Update EMA and return (ema, upper_band, lower_band)."""
        self._values.append(value)

        if self.ema is None:
            self.ema = value
            self.ema_upper = value
            self.ema_lower = value
        else:
            self.ema = self.alpha * value + (1 - self.alpha) * self.ema

            # Update bands based on recent volatility
            if len(self._values) > 1:
                std = math.sqrt(sum((v - self.ema)**2 for v in self._values) / len(self._values))
                self.ema_upper = self.ema + 2 * std
                self.ema_lower = self.ema - 2 * std

        return self.ema, self.ema_upper, self.ema_lower

    def is_anomaly(self, value: float) -> bool:
        """Check if value is outside EMA bands."""
        if self.ema_upper is None:
            return False
        return value > self.ema_upper or value < self.ema_lower


# =============================================================================
# Adaptive Threshold
# =============================================================================

class AdaptiveThreshold:
    """
    Adaptive threshold that adjusts based on data distribution.

    Implements multiple strategies:
    - Percentile-based
    - Standard deviation based
    - Interquartile range (IQR) based
    - CUSUM (Cumulative Sum)
    """

    def __init__(
        self,
        strategy: str = "std",
        initial_threshold: float = 3.0,
        adaptation_rate: float = 0.01,
        min_samples: int = 30
    ):
        self.strategy = strategy
        self.threshold = initial_threshold
        self.adaptation_rate = adaptation_rate
        self.min_samples = min_samples

        self.stats = OnlineStatistics()
        self.p25 = OnlinePercentile(0.25)
        self.p75 = OnlinePercentile(0.75)
        self.p95 = OnlinePercentile(0.95)

        # CUSUM parameters
        self.cusum_pos = 0.0
        self.cusum_neg = 0.0
        self.cusum_threshold = 5.0

        # Recent anomaly rate for adaptation
        self._recent_anomalies: Deque[bool] = deque(maxlen=100)

    def update(self, value: float) -> bool:
        """Update threshold and check if value is anomalous."""
        # Update statistics
        self.stats.update(value)
        self.p25.update(value)
        self.p75.update(value)
        self.p95.update(value)

        # Check anomaly based on strategy
        is_anomaly = self._check_anomaly(value)
        self._recent_anomalies.append(is_anomaly)

        # Adapt threshold based on anomaly rate
        if len(self._recent_anomalies) >= self.min_samples:
            self._adapt_threshold()

        return is_anomaly

    def _check_anomaly(self, value: float) -> bool:
        """Check if value is anomalous based on current strategy."""
        if self.stats.n < self.min_samples:
            return False

        if self.strategy == "std":
            return abs(self.stats.z_score(value)) > self.threshold

        elif self.strategy == "percentile":
            upper = self.p95.estimate
            lower = self.p25.estimate - 1.5 * (self.p75.estimate - self.p25.estimate)
            return value > upper or value < lower

        elif self.strategy == "iqr":
            iqr = self.p75.estimate - self.p25.estimate
            lower = self.p25.estimate - self.threshold * iqr
            upper = self.p75.estimate + self.threshold * iqr
            return value < lower or value > upper

        elif self.strategy == "cusum":
            # Update CUSUM
            target = self.stats.mean
            self.cusum_pos = max(0, self.cusum_pos + value - target - self.threshold)
            self.cusum_neg = max(0, self.cusum_neg - value + target - self.threshold)

            return self.cusum_pos > self.cusum_threshold or self.cusum_neg > self.cusum_threshold

        return False

    def _adapt_threshold(self):
        """Adapt threshold based on recent anomaly rate."""
        anomaly_rate = sum(self._recent_anomalies) / len(self._recent_anomalies)

        # Target anomaly rate: 1-5%
        target_rate = 0.02

        if anomaly_rate > target_rate * 2:
            # Too many anomalies, increase threshold
            self.threshold *= (1 + self.adaptation_rate)
        elif anomaly_rate < target_rate / 2:
            # Too few anomalies, decrease threshold
            self.threshold *= (1 - self.adaptation_rate)

        # Clamp threshold
        self.threshold = max(1.5, min(10.0, self.threshold))

    def get_bounds(self) -> Tuple[float, float]:
        """Get current anomaly bounds."""
        if self.strategy == "std":
            return (
                self.stats.mean - self.threshold * self.stats.std_dev,
                self.stats.mean + self.threshold * self.stats.std_dev
            )
        elif self.strategy == "iqr":
            iqr = self.p75.estimate - self.p25.estimate
            return (
                self.p25.estimate - self.threshold * iqr,
                self.p75.estimate + self.threshold * iqr
            )
        else:
            return (self.stats.min_val, self.stats.max_val)


# =============================================================================
# Streaming Isolation Forest
# =============================================================================

class StreamingIsolationTree:
    """Single isolation tree for streaming data."""

    def __init__(self, max_depth: int = 10, sample_size: int = 256):
        self.max_depth = max_depth
        self.sample_size = sample_size
        self.root = None
        self._samples: List[List[float]] = []

    def update(self, point: List[float]):
        """Update tree with new point."""
        self._samples.append(point)

        # Rebuild tree when we have enough samples
        if len(self._samples) >= self.sample_size:
            self._build_tree()
            # Keep only recent samples
            self._samples = self._samples[-self.sample_size:]

    def _build_tree(self):
        """Build isolation tree from samples."""
        if not self._samples:
            return

        n_features = len(self._samples[0])
        self.root = self._build_node(self._samples, 0, n_features)

    def _build_node(self, data: List[List[float]], depth: int, n_features: int) -> Dict:
        """Recursively build tree node."""
        if depth >= self.max_depth or len(data) <= 1:
            return {"type": "leaf", "size": len(data)}

        # Random feature and split point
        feature = random.randint(0, n_features - 1)
        values = [p[feature] for p in data]
        min_val, max_val = min(values), max(values)

        if min_val == max_val:
            return {"type": "leaf", "size": len(data)}

        split = random.uniform(min_val, max_val)

        # Split data
        left_data = [p for p in data if p[feature] < split]
        right_data = [p for p in data if p[feature] >= split]

        return {
            "type": "internal",
            "feature": feature,
            "split": split,
            "left": self._build_node(left_data, depth + 1, n_features),
            "right": self._build_node(right_data, depth + 1, n_features)
        }

    def path_length(self, point: List[float]) -> float:
        """Compute path length for a point."""
        if self.root is None:
            return 0.0

        return self._path_length(point, self.root, 0)

    def _path_length(self, point: List[float], node: Dict, depth: int) -> float:
        """Recursively compute path length."""
        if node["type"] == "leaf":
            # Add expected path length for remaining nodes
            n = node["size"]
            if n <= 1:
                return depth
            # Average path length in a random tree
            c = 2 * (math.log(n - 1) + 0.5772156649) - 2 * (n - 1) / n
            return depth + c

        if point[node["feature"]] < node["split"]:
            return self._path_length(point, node["left"], depth + 1)
        else:
            return self._path_length(point, node["right"], depth + 1)


class StreamingIsolationForest:
    """
    Streaming Isolation Forest for online anomaly detection.

    Maintains a forest of isolation trees that can be incrementally
    updated as new data arrives.
    """

    def __init__(
        self,
        n_trees: int = 100,
        max_depth: int = 10,
        sample_size: int = 256,
        contamination: float = 0.1
    ):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.sample_size = sample_size
        self.contamination = contamination

        self.trees = [
            StreamingIsolationTree(max_depth, sample_size)
            for _ in range(n_trees)
        ]

        self._threshold: Optional[float] = None
        self._recent_scores: Deque[float] = deque(maxlen=1000)

    def update(self, point: List[float]):
        """Update forest with new point."""
        # Update a random subset of trees
        trees_to_update = random.sample(
            self.trees,
            k=min(10, self.n_trees)
        )
        for tree in trees_to_update:
            tree.update(point)

    def score(self, point: List[float]) -> float:
        """Compute anomaly score for a point."""
        if not any(tree.root for tree in self.trees):
            return 0.5  # Neutral score

        # Average path length across trees
        path_lengths = [
            tree.path_length(point)
            for tree in self.trees
            if tree.root is not None
        ]

        if not path_lengths:
            return 0.5

        avg_path = sum(path_lengths) / len(path_lengths)

        # Normalize using average path length in random tree
        n = self.sample_size
        c = 2 * (math.log(n - 1) + 0.5772156649) - 2 * (n - 1) / n

        # Anomaly score: shorter path = higher score
        score = 2 ** (-avg_path / c)

        self._recent_scores.append(score)
        self._update_threshold()

        return score

    def _update_threshold(self):
        """Update anomaly threshold based on recent scores."""
        if len(self._recent_scores) < 100:
            return

        sorted_scores = sorted(self._recent_scores)
        idx = int((1 - self.contamination) * len(sorted_scores))
        self._threshold = sorted_scores[idx]

    def is_anomaly(self, point: List[float]) -> bool:
        """Check if point is an anomaly."""
        score = self.score(point)

        if self._threshold is None:
            return score > 0.6  # Default threshold

        return score > self._threshold

    def fit_predict(self, points: List[List[float]]) -> List[bool]:
        """Fit on data and predict anomalies."""
        # First pass: update trees
        for point in points:
            self.update(point)

        # Second pass: score and classify
        return [self.is_anomaly(point) for point in points]


# =============================================================================
# Streaming Anomaly Detector
# =============================================================================

@dataclass
class AnomalyEvent:
    """Detected anomaly event."""
    timestamp: datetime
    metric_name: str
    value: float
    expected_value: float
    score: float
    severity: str  # low, medium, high, critical
    context: Dict[str, Any] = field(default_factory=dict)


class StreamingAnomalyDetector:
    """
    Comprehensive streaming anomaly detector.

    Combines multiple detection methods:
    - Statistical (z-score, EMA)
    - Machine learning (Isolation Forest)
    - Pattern-based (seasonality, trends)
    """

    def __init__(
        self,
        methods: List[str] = None,
        window_size: int = 100,
        sensitivity: float = 0.95
    ):
        self.methods = methods or ["statistical", "isolation_forest", "adaptive"]
        self.window_size = window_size
        self.sensitivity = sensitivity

        # Per-metric detectors
        self._stats: Dict[str, OnlineStatistics] = {}
        self._ema: Dict[str, ExponentialMovingStats] = {}
        self._adaptive: Dict[str, AdaptiveThreshold] = {}
        self._forests: Dict[str, StreamingIsolationForest] = {}

        # Recent values for pattern detection
        self._history: Dict[str, Deque[Tuple[datetime, float]]] = {}

        # Anomaly callbacks
        self._callbacks: List[Callable[[AnomalyEvent], None]] = []

    def add_callback(self, callback: Callable[[AnomalyEvent], None]):
        """Add callback for anomaly events."""
        self._callbacks.append(callback)

    def _get_or_create_detectors(self, metric: str):
        """Get or create detectors for a metric."""
        if metric not in self._stats:
            self._stats[metric] = OnlineStatistics()
            self._ema[metric] = ExponentialMovingStats(alpha=0.1)
            self._adaptive[metric] = AdaptiveThreshold(strategy="std")
            self._forests[metric] = StreamingIsolationForest(
                n_trees=50,
                sample_size=128
            )
            self._history[metric] = deque(maxlen=self.window_size)

    def update(
        self,
        metric: str,
        value: float,
        timestamp: datetime = None
    ) -> Optional[AnomalyEvent]:
        """Update detector with new value and check for anomaly."""
        timestamp = timestamp or datetime.now()
        self._get_or_create_detectors(metric)

        # Update all detectors
        self._stats[metric].update(value)
        self._ema[metric].update(value)
        adaptive_anomaly = self._adaptive[metric].update(value)
        self._forests[metric].update([value])
        self._history[metric].append((timestamp, value))

        # Collect anomaly votes
        votes = []
        scores = []

        # Statistical detection
        if "statistical" in self.methods:
            z_score = abs(self._stats[metric].z_score(value))
            threshold = self._z_threshold(self.sensitivity)
            votes.append(z_score > threshold)
            scores.append(min(1.0, z_score / threshold))

        # EMA detection
        if "ema" in self.methods:
            ema_z = abs(self._ema[metric].z_score(value))
            threshold = self._z_threshold(self.sensitivity)
            votes.append(ema_z > threshold)
            scores.append(min(1.0, ema_z / threshold))

        # Adaptive detection
        if "adaptive" in self.methods:
            votes.append(adaptive_anomaly)
            bounds = self._adaptive[metric].get_bounds()
            if bounds[0] != bounds[1]:
                dist = max(0, value - bounds[1], bounds[0] - value)
                scores.append(min(1.0, dist / (bounds[1] - bounds[0] + 1e-8)))
            else:
                scores.append(0.0)

        # Isolation Forest detection
        if "isolation_forest" in self.methods:
            if_score = self._forests[metric].score([value])
            votes.append(self._forests[metric].is_anomaly([value]))
            scores.append(if_score)

        # Combine results
        avg_score = sum(scores) / len(scores) if scores else 0.0
        is_anomaly = sum(votes) >= len(votes) / 2  # Majority vote

        if is_anomaly:
            event = AnomalyEvent(
                timestamp=timestamp,
                metric_name=metric,
                value=value,
                expected_value=self._stats[metric].mean,
                score=avg_score,
                severity=self._severity_from_score(avg_score),
                context={
                    "z_score": self._stats[metric].z_score(value),
                    "mean": self._stats[metric].mean,
                    "std_dev": self._stats[metric].std_dev,
                    "detection_methods": [
                        m for m, v in zip(self.methods, votes) if v
                    ]
                }
            )

            # Notify callbacks
            for callback in self._callbacks:
                try:
                    callback(event)
                except Exception as e:
                    logger.error(f"Anomaly callback error: {e}")

            return event

        return None

    def _z_threshold(self, sensitivity: float) -> float:
        """Convert sensitivity to z-score threshold."""
        # Approximate inverse normal CDF
        p = (1 + sensitivity) / 2
        if p >= 1:
            return 10
        if p <= 0:
            return -10

        t = math.sqrt(-2 * math.log(1 - p))
        c0, c1, c2 = 2.515517, 0.802853, 0.010328
        d1, d2, d3 = 1.432788, 0.189269, 0.001308

        return t - (c0 + c1*t + c2*t*t) / (1 + d1*t + d2*t*t + d3*t*t*t)

    def _severity_from_score(self, score: float) -> str:
        """Map anomaly score to severity level."""
        if score > 0.9:
            return "critical"
        elif score > 0.7:
            return "high"
        elif score > 0.5:
            return "medium"
        return "low"

    def get_stats(self, metric: str) -> Dict[str, Any]:
        """Get current statistics for a metric."""
        if metric not in self._stats:
            return {}

        return {
            "mean": self._stats[metric].mean,
            "std_dev": self._stats[metric].std_dev,
            "min": self._stats[metric].min_val,
            "max": self._stats[metric].max_val,
            "count": self._stats[metric].n,
            "ema": self._ema[metric].mean,
            "adaptive_threshold": self._adaptive[metric].threshold
        }


# =============================================================================
# Complex Event Processing
# =============================================================================

class CEPPattern(ABC):
    """Base class for Complex Event Processing patterns."""

    @abstractmethod
    def match(self, event: Any) -> bool:
        """Check if event matches the pattern."""
        pass

    @abstractmethod
    def reset(self):
        """Reset pattern state."""
        pass


class SequencePattern(CEPPattern):
    """Match a sequence of events in order."""

    def __init__(self, conditions: List[Callable[[Any], bool]], within: timedelta = None):
        self.conditions = conditions
        self.within = within
        self._matched_events: List[Tuple[datetime, Any]] = []
        self._current_idx = 0

    def match(self, event: Any) -> bool:
        now = datetime.now()

        # Check timeout
        if self.within and self._matched_events:
            first_time = self._matched_events[0][0]
            if now - first_time > self.within:
                self.reset()

        # Check if event matches current condition
        if self._current_idx < len(self.conditions):
            if self.conditions[self._current_idx](event):
                self._matched_events.append((now, event))
                self._current_idx += 1

                if self._current_idx >= len(self.conditions):
                    # Pattern complete
                    self.reset()
                    return True

        return False

    def reset(self):
        self._matched_events = []
        self._current_idx = 0


class ThresholdPattern(CEPPattern):
    """Match when threshold is exceeded N times in a window."""

    def __init__(
        self,
        condition: Callable[[Any], bool],
        count: int,
        window: timedelta
    ):
        self.condition = condition
        self.count = count
        self.window = window
        self._events: Deque[datetime] = deque()

    def match(self, event: Any) -> bool:
        now = datetime.now()

        # Remove old events
        while self._events and now - self._events[0] > self.window:
            self._events.popleft()

        # Check condition
        if self.condition(event):
            self._events.append(now)

            if len(self._events) >= self.count:
                return True

        return False

    def reset(self):
        self._events.clear()


class AbsencePattern(CEPPattern):
    """Match when expected event doesn't occur within timeout."""

    def __init__(self, expected: Callable[[Any], bool], timeout: timedelta):
        self.expected = expected
        self.timeout = timeout
        self._last_seen: Optional[datetime] = None
        self._armed = False

    def arm(self):
        """Arm the pattern to start watching for absence."""
        self._armed = True
        self._last_seen = datetime.now()

    def match(self, event: Any) -> bool:
        now = datetime.now()

        if self.expected(event):
            self._last_seen = now
            return False

        if self._armed and self._last_seen:
            if now - self._last_seen > self.timeout:
                self._armed = False
                return True

        return False

    def reset(self):
        self._last_seen = None
        self._armed = False


class CEPEngine:
    """Complex Event Processing engine."""

    def __init__(self):
        self.patterns: Dict[str, CEPPattern] = {}
        self.callbacks: Dict[str, List[Callable]] = defaultdict(list)

    def add_pattern(self, name: str, pattern: CEPPattern):
        """Add a named pattern."""
        self.patterns[name] = pattern

    def on_match(self, pattern_name: str, callback: Callable):
        """Register callback for pattern match."""
        self.callbacks[pattern_name].append(callback)

    def process(self, event: Any):
        """Process event against all patterns."""
        for name, pattern in self.patterns.items():
            if pattern.match(event):
                for callback in self.callbacks[name]:
                    try:
                        callback(event, name)
                    except Exception as e:
                        logger.error(f"CEP callback error: {e}")
